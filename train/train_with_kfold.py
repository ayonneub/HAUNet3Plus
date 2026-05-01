import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
import mlflow
import mlflow.pytorch

from utils.dataset import ImageDataset, base_transform, train_transform
from utils.dataprocess import DatasetSplitter
from models.HAUNET_3Plus import UNet_3Plus_DeepSup_AC, DeepSupervisionLoss, CombinedLoss
from models.UNet_3Plus import UNet_3Plus, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_CGM
from models.Attention_UNet import AttentionUNet
from models.UNet_Ayon import UNet
import pandas as pd


# Reproducibility

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seeds(42)


# Metrics

def compute_iou(pred, target, threshold=0.5, eps=1e-6):
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = (pred + target).sum(dim=(1,2,3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def compute_pixel_accuracy(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return (correct / total).item()


# Dataset paths
image_dir = './Sinkhole_dataset_SAMRefiner/images'
mask_dir  = './Sinkhole_dataset_SAMRefiner/masks'

image_list = sorted(os.listdir(image_dir))
mask_list  = sorted(os.listdir(mask_dir))
assert len(image_list) == len(mask_list), "Images and masks count mismatch."

# Pair images with masks
data = list(zip(image_list, mask_list))


# Device

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


# Model zoo (constructors)
models = [
    # ("UNet", lambda: UNet(n_class=1)),
    # ("Attention UNet", lambda: AttentionUNet(in_channels=3, out_channels=1)),
    # ("UNet_3Plus", lambda: UNet_3Plus(in_channels=3, n_classes=1)),
    # ("UNet_3Plus_DeepSup", lambda: UNet_3Plus_DeepSup(in_channels=3, n_classes=1)),
    # ("UNet_3Plus_DeepSup_CGM", lambda: UNet_3Plus_DeepSup_CGM(in_channels=3, n_classes=1)),
    ("HAUNet3+", lambda: UNet_3Plus_DeepSup_AC(in_channels=3, n_classes=1))
]

# Training hyperparameters

num_epochs  = 500
learning_rate = 5e-4
batch_size  = 16
patience    = 20
min_delta   = 0.0

alpha_values = [0.6, 0.7, 0.8]
ds_weight_sets = {
    "linear":   (1.0, 0.8, 0.6, 0.4, 0.2),
    "exp":      (1.0, 0.5, 0.25, 0.125, 0.0625),
    "uniform":  (1.0, 1.0, 1.0, 1.0, 1.0)
}

os.makedirs("./pretrained_models", exist_ok=True)


# 10-Fold CV:
# Outer fold defines TEST.
# From outer (train+val), take 10% as VAL using an inner split.

outer_kf = KFold(n_splits=10, shuffle=True, random_state=42)

results_val  = []
results_test = []

mlflow.set_experiment("HAUNet3+_10FCV_outerTest_innerVal10pct")

for alpha in alpha_values:
    for ds_name, ds_weights in ds_weight_sets.items():
        if alpha==0.4 and ds_name=="linear":
            continue
        for outer_fold, (trainval_idx, test_idx) in enumerate(outer_kf.split(data), start=1):
            # Outer TEST split
            test_data = [data[i] for i in test_idx]

            # Inner split: 10% of trainval as VAL
            trainval_indices = np.array(trainval_idx)
            tr_idx, val_idx_sub = train_test_split(
                trainval_indices, test_size=0.10, shuffle=True, random_state=42
            )
            train_data = [data[i] for i in tr_idx]
            val_data   = [data[i] for i in val_idx_sub]

            print(f"\n[Outer Fold {outer_fold}/10] alpha={alpha} | ds={ds_name}")
            print(f"Counts -> Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

            #  Datasets / Loaders
            train_dataset = ImageDataset(train_data, image_dir, mask_dir, transform=train_transform, seed=42)
            val_dataset   = ImageDataset(val_data,   image_dir, mask_dir, transform=base_transform,   seed=42)
            test_dataset  = ImageDataset(test_data,  image_dir, mask_dir, transform=base_transform,   seed=42)

            splitter     = DatasetSplitter(train_dataset, val_dataset, None, batch_size=batch_size, seed=42)
            train_loader = splitter.train_loader
            val_loader   = splitter.val_loader
            test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            for model_name, model_fn in models:
                # Fresh model
                model = model_fn().to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

                if ("DeepSup" in model_name) or ("HAUNet3+" in model_name):
                    criterion = DeepSupervisionLoss(alpha=alpha, ds_weights=ds_weights)
                else:
                    criterion = CombinedLoss(alpha=alpha)

                best_val_iou  = 0.0
                best_val_acc  = 0.0
                best_val_loss = float('inf')
                epochs_no_improve = 0
                early_stop = False

                run_name = f"{model_name}_a{alpha}_{ds_name}_outer{outer_fold}"
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params({
                        "alpha": alpha,
                        "ds_name": ds_name,
                        "ds_weights": ds_weights,
                        "outer_fold": outer_fold,
                        "num_epochs": num_epochs,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "model_name": model_name,
                        "device": str(device),
                        "seed": 42,
                        "val_split_from_trainval_pct": 10
                    })

                    # Train loop with early stopping on Val IoU
                    for epoch in range(num_epochs):
                        if early_stop:
                            break
 
                        model.train()
                        running_loss, running_iou = 0.0, 0.0
                        for images, masks in train_loader:
                            images, masks = images.to(device), masks.to(device)
                            if masks.dim() == 3:
                                masks = masks.unsqueeze(1)

                            optimizer.zero_grad()
                            outputs = model(images)
                            loss = criterion(outputs, masks)
                            loss.backward()
                            optimizer.step()

                            preds = outputs[0] if isinstance(outputs, tuple) else outputs
                            running_loss += loss.item()
                            running_iou  += compute_iou(preds, masks)

                        avg_train_loss = running_loss / len(train_loader)
                        avg_train_iou  = running_iou  / len(train_loader)
                        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                        mlflow.log_metric("train_iou",  avg_train_iou,  step=epoch)
                        mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

                        # Validation 
                        model.eval()
                        val_loss_total, val_iou_total, val_acc_total = 0.0, 0.0, 0.0
                        with torch.no_grad():
                            for vimg, vmsk in val_loader:
                                vimg, vmsk = vimg.to(device), vmsk.to(device)
                                if vmsk.dim() == 3:
                                    vmsk = vmsk.unsqueeze(1)
                                vout = model(vimg)
                                vloss = criterion(vout, vmsk)
                                vpred = vout[0] if isinstance(vout, tuple) else vout
                                val_loss_total += vloss.item()
                                val_iou_total  += compute_iou(vpred, vmsk)
                                val_acc_total  += compute_pixel_accuracy(vpred, vmsk)

                        avg_val_loss = val_loss_total / len(val_loader)
                        avg_val_iou  = val_iou_total  / len(val_loader)
                        avg_val_acc  = val_acc_total  / len(val_loader)

                        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                        mlflow.log_metric("val_iou",  avg_val_iou,  step=epoch)
                        mlflow.log_metric("val_acc",  avg_val_acc,  step=epoch)

                        scheduler.step()

                        # Early stopping on IoU
                        if avg_val_iou > best_val_iou + min_delta:
                            best_val_iou  = avg_val_iou
                            best_val_acc  = avg_val_acc
                            best_val_loss = avg_val_loss
                            epochs_no_improve = 0
                            save_path = f"./pretrained_models/{model_name}_a{alpha}_{ds_name}_outer{outer_fold}.pth"
                            torch.save(model.state_dict(), save_path)
                            mlflow.log_artifact(save_path)
                            print(f"[Outer {outer_fold}] Best @ epoch {epoch+1}: val_iou={best_val_iou:.4f}")
                        else:
                            epochs_no_improve += 1
                            if epochs_no_improve >= patience:
                                early_stop = True

                    #  Log best VAL metrics
                    mlflow.log_metric("best_val_iou",  best_val_iou)
                    mlflow.log_metric("best_val_acc",  best_val_acc)
                    mlflow.log_metric("best_val_loss", best_val_loss)

                    # Reload best and evaluate on TEST 
                    best_path = f"./pretrained_models/{model_name}_a{alpha}_{ds_name}_outer{outer_fold}.pth"
                    model.load_state_dict(torch.load(best_path, map_location=device))
                    model.eval()

                    test_loss_total, test_iou_total, test_acc_total = 0.0, 0.0, 0.0
                    with torch.no_grad():
                        for timg, tmsk in test_loader:
                            timg, tmsk = timg.to(device), tmsk.to(device)
                            if tmsk.dim() == 3:
                                tmsk = tmsk.unsqueeze(1)
                            tout = model(timg)
                            tloss = criterion(tout, tmsk)
                            tpred = tout[0] if isinstance(tout, tuple) else tout
                            test_loss_total += tloss.item()
                            test_iou_total  += compute_iou(tpred, tmsk)
                            test_acc_total  += compute_pixel_accuracy(tpred, tmsk)

                    avg_test_loss = test_loss_total / len(test_loader)
                    avg_test_iou  = test_iou_total  / len(test_loader)
                    avg_test_acc  = test_acc_total  / len(test_loader)

                    mlflow.log_metric("test_loss", avg_test_loss)
                    mlflow.log_metric("test_iou",  avg_test_iou)
                    mlflow.log_metric("test_acc",  avg_test_acc)

                    print(f"[Outer {outer_fold}] TEST -> Loss: {avg_test_loss:.4f}, IoU: {avg_test_iou:.4f}, Acc: {avg_test_acc:.4f}")

                    #Collect per-fold results
                    results_val.append({
                        "model": model_name, "alpha": alpha, "ds_name": ds_name,
                        "outer_fold": outer_fold,
                        "val_iou": best_val_iou, "val_acc": best_val_acc, "val_loss": best_val_loss
                    })
                    results_test.append({
                        "model": model_name, "alpha": alpha, "ds_name": ds_name,
                        "outer_fold": outer_fold,
                        "test_iou": avg_test_iou, "test_acc": avg_test_acc, "test_loss": avg_test_loss
                    }) 


# Save CSVs and summaries
df_val  = pd.DataFrame(results_val)
df_test = pd.DataFrame(results_test)
df_val.to_csv("outer10_val_results.csv", index=False)
df_test.to_csv("outer10_test_results.csv", index=False)

summary_val  = df_val.groupby(["model","alpha","ds_name"])[["val_iou","val_acc","val_loss"]].agg(["mean","std"]).reset_index()
summary_test = df_test.groupby(["model","alpha","ds_name"])[["test_iou","test_acc","test_loss"]].agg(["mean","std"]).reset_index()

summary_val.to_csv("outer10_val_summary.csv", index=False)
summary_test.to_csv("outer10_test_summary.csv", index=False)

print("\nVAL Summary:")
print(summary_val)
print("\nTEST Summary:")
print(summary_test)

# Optional: pick best combo by highest mean TEST IoU
best_idx = summary_test[('test_iou','mean')].idxmax()
best_combo = summary_test.iloc[best_idx]
print("\nBest combo by TEST IoU (mean across folds):")
print(best_combo)
