import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
import mlflow
import mlflow.pytorch
import pandas as pd


# Your project imports

from utils.dataset import ImageDataset, base_transform, train_transform
from utils.dataprocess import DatasetSplitter  # (kept in case you need; not mandatory below)
from models.HAUNET_3Plus import UNet_3Plus_DeepSup_AC, DeepSupervisionLoss, CombinedLoss
from models.UNet_3Plus import UNet_3Plus, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_CGM
from models.Attention_UNet import AttentionUNet
from models.UNet_Ayon import UNet



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
    """
    pred: probabilities/logits [B,1,H,W]; we threshold at 'threshold'
    target: binary mask [B,1,H,W] (0/1)
    """
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).sum(dim=(1, 2, 3)) - intersection
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
mask_dir = './Sinkhole_dataset_SAMRefiner/masks'

image_list = sorted(os.listdir(image_dir))
mask_list = sorted(os.listdir(mask_dir))
assert len(image_list) == len(mask_list), "Images and masks count mismatch."

# Pair images with masks
data = list(zip(image_list, mask_list))


# Device

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)



# Model zoo (constructors + deep supervision flag)

models = [
    #("UNet", lambda: UNet(n_class=1), False),
    #("Attention UNet", lambda: AttentionUNet(in_channels=3, out_channels=1), False),
    #("UNet_3Plus", lambda: UNet_3Plus(in_channels=3, n_classes=1), False),
    #("UNet_3Plus_DeepSup", lambda: UNet_3Plus_DeepSup(in_channels=3, n_classes=1), True),
    #("UNet_3Plus_DeepSup_CGM", lambda: UNet_3Plus_DeepSup_CGM(in_channels=3, n_classes=1), True),
     ("HAUNet3+", lambda: UNet_3Plus_DeepSup_AC(in_channels=3, n_classes=1), True),
]



# Training hyperparameters

num_epochs = 1000
learning_rate = 5e-4
batch_size = 16
patience = 20
min_delta = 0.0

criterion_multiple_channels = DeepSupervisionLoss(alpha=0.67, ds_weights=(1.0, 0.5, 0.25, 0.125, 0.0625))
criterion_single_channels = CombinedLoss(alpha=0.5)



# CV setup

outer_kf = KFold(n_splits=10, shuffle=True, random_state=42)
os.makedirs("./pretrained_models", exist_ok=True)

mlflow.set_experiment("Sinkhole_Research_Nobel_Outer10_Inner10Final_PerModel")

all_models_val = []
all_models_test = []

# =========================
# Per-model 10-fold CV
# =========================
for model_name, model_fn, uses_deepsup in models:
    print(f"\n########## 10-FOLD for {model_name} ##########")

    val_rows = []
    test_rows = []

    with mlflow.start_run(run_name=f"{model_name}_parent"):
        mlflow.log_params({
            "cv_scheme": "Outer 10-fold (test) + inner 10% val",
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "device": str(device),
            "seed": 42,
            "patience": patience,
            "min_delta": min_delta,
            "scheduler_type": "CosineAnnealingLR",
            "loss": "BCE+Dice (Deep Supervision)" if uses_deepsup else "BCE+Dice (Single)"
        })

        for outer_fold, (trainval_idx, test_idx) in enumerate(outer_kf.split(data), start=1):
            test_data = [data[i] for i in test_idx]
            tr_idx, val_idx = train_test_split(np.array(trainval_idx), test_size=0.10, shuffle=True, random_state=42)
            train_data = [data[i] for i in tr_idx]
            val_data = [data[i] for i in val_idx]

            print(f"\n[Outer Fold {outer_fold}/10] {model_name} -> Train={len(train_data)} | "
                  f"Val={len(val_data)} | Test={len(test_data)}")

            train_dataset = ImageDataset(train_data, image_dir, mask_dir, transform=train_transform, seed=42)
            val_dataset = ImageDataset(val_data, image_dir, mask_dir, transform=base_transform, seed=42)
            test_dataset = ImageDataset(test_data, image_dir, mask_dir, transform=base_transform, seed=42)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=4, pin_memory=True, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=4, pin_memory=True, drop_last=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=4, pin_memory=True, drop_last=False)

            set_seeds(42)
            model = model_fn().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

            best_val_iou = 0.0
            best_val_loss = float('inf')
            best_val_acc = 0.0
            epochs_no_improve = 0
            early_stop = False

            with mlflow.start_run(run_name=f"{model_name}_outer{outer_fold}", nested=True):
                mlflow.log_params({
                    "outer_fold": outer_fold,
                    "train_count": len(train_dataset),
                    "val_count": len(val_dataset),
                    "test_count": len(test_dataset)
                })

                for epoch in range(num_epochs):
                    if early_stop:
                        print(f"Early stopping at epoch {epoch+1} for {model_name} (outer {outer_fold})")
                        break

                    # ---- TRAIN ----
                    model.train()
                    running_loss, running_iou = 0.0, 0.0
                    for images, masks in train_loader:
                        images, masks = images.to(device), masks.to(device)
                        if masks.dim() == 3:
                            masks = masks.unsqueeze(1)

                        optimizer.zero_grad()
                        outputs = model(images)

                        if uses_deepsup:
                            loss = criterion_multiple_channels(outputs, masks)
                            preds = outputs[0]
                        else:
                            loss = criterion_single_channels(outputs, masks)
                            preds = outputs

                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        running_iou += compute_iou(preds, masks)

                    avg_train_loss = running_loss / max(1, len(train_loader))
                    avg_train_iou = running_iou / max(1, len(train_loader))
                    mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                    mlflow.log_metric("train_iou", avg_train_iou, step=epoch)
                    mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

                    # ---- VALIDATION ----
                    model.eval()
                    v_loss_sum, v_iou_sum, v_acc_sum = 0.0, 0.0, 0.0
                    with torch.no_grad():
                        for v_images, v_masks in val_loader:
                            v_images, v_masks = v_images.to(device), v_masks.to(device)
                            if v_masks.dim() == 3:
                                v_masks = v_masks.unsqueeze(1)

                            v_outputs = model(v_images)
                            if uses_deepsup:
                                v_loss = criterion_multiple_channels(v_outputs, v_masks)
                                v_preds = v_outputs[0]
                            else:
                                v_loss = criterion_single_channels(v_outputs, v_masks)
                                v_preds = v_outputs

                            v_loss_sum += v_loss.item()
                            v_iou_sum += compute_iou(v_preds, v_masks)
                            v_acc_sum += compute_pixel_accuracy(v_preds, v_masks)

                    avg_val_loss = v_loss_sum / max(1, len(val_loader))
                    avg_val_iou = v_iou_sum / max(1, len(val_loader))
                    avg_val_acc = v_acc_sum / max(1, len(val_loader))

                    mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                    mlflow.log_metric("val_iou", avg_val_iou, step=epoch)
                    mlflow.log_metric("val_acc", avg_val_acc, step=epoch)

                    print(f"[Outer {outer_fold}/10 | {model_name} | Epoch {epoch+1}/{num_epochs}] "
                          f"TrainLoss {avg_train_loss:.4f}  ValLoss {avg_val_loss:.4f}  "
                          f"ValIoU {avg_val_iou:.4f}  ValAcc {avg_val_acc:.4f}")

                    scheduler.step()

                    # ---- Early Stopping ----
                    if avg_val_iou > (best_val_iou + min_delta):
                        best_val_iou = avg_val_iou
                        best_val_loss = avg_val_loss
                        best_val_acc = avg_val_acc
                        epochs_no_improve = 0
                        best_path = f"./pretrained_models/{model_name.replace(' ', '_')}_outer{outer_fold}_best.pth"
                        torch.save(model.state_dict(), best_path)
                        mlflow.log_artifact(best_path)
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            early_stop = True

                # ---- TEST ----
                mlflow.log_metric("best_val_iou", best_val_iou)
                mlflow.log_metric("best_val_loss", best_val_loss)
                mlflow.log_metric("best_val_acc", best_val_acc)

                best_path = f"./pretrained_models/{model_name.replace(' ', '_')}_outer{outer_fold}_best.pth"
                model.load_state_dict(torch.load(best_path, map_location=device))
                model.eval()

                t_loss_sum, t_iou_sum, t_acc_sum = 0.0, 0.0, 0.0
                with torch.no_grad():
                    for t_images, t_masks in test_loader:
                        t_images, t_masks = t_images.to(device), t_masks.to(device)
                        if t_masks.dim() == 3:
                            t_masks = t_masks.unsqueeze(1)

                        t_outputs = model(t_images)
                        if uses_deepsup:
                            t_loss = criterion_multiple_channels(t_outputs, t_masks)
                            t_preds = t_outputs[0]
                        else:
                            t_loss = criterion_single_channels(t_outputs, t_masks)
                            t_preds = t_outputs

                        t_loss_sum += t_loss.item()
                        t_iou_sum += compute_iou(t_preds, t_masks)
                        t_acc_sum += compute_pixel_accuracy(t_preds, t_masks)

                avg_test_loss = t_loss_sum / max(1, len(test_loader))
                avg_test_iou = t_iou_sum / max(1, len(test_loader))
                avg_test_acc = t_acc_sum / max(1, len(test_loader))

                mlflow.log_metric("test_loss", avg_test_loss)
                mlflow.log_metric("test_iou", avg_test_iou)
                mlflow.log_metric("test_acc", avg_test_acc)

                print(f"[Outer {outer_fold}] TEST -> Loss: {avg_test_loss:.4f} | "
                      f"IoU: {avg_test_iou:.4f} | Acc: {avg_test_acc:.4f}")

                val_rows.append({
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "val_iou": best_val_iou,
                    "val_acc": best_val_acc,
                    "val_loss": best_val_loss
                })
                test_rows.append({
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "test_iou": avg_test_iou,
                    "test_acc": avg_test_acc,
                    "test_loss": avg_test_loss
                })

            del optimizer, scheduler, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # =========================
    # Save per-model CSVs and summaries
    # =========================
    df_val = pd.DataFrame(val_rows)
    df_test = pd.DataFrame(test_rows)

    df_val.to_csv(f"outer10_val_results_{model_name.replace(' ','_')}.csv", index=False)
    df_test.to_csv(f"outer10_test_results_{model_name.replace(' ','_')}.csv", index=False)

    summary_val = df_val.groupby(["model"])[["val_iou", "val_acc", "val_loss"]].agg(["mean", "std"]).reset_index()
    summary_test = df_test.groupby(["model"])[["test_iou", "test_acc", "test_loss"]].agg(["mean", "std"]).reset_index()

    summary_val.to_csv(f"outer10_val_summary_{model_name.replace(' ','_')}.csv", index=False)
    summary_test.to_csv(f"outer10_test_summary_{model_name.replace(' ','_')}.csv", index=False)

    print("\nVAL Summary (", model_name, "):")
    print(summary_val)
    print("\nTEST Summary (", model_name, "):")
    print(summary_test)

    all_models_val.append(df_val)
    all_models_test.append(df_test)

# =========================
# Global aggregation
# =========================
if all_models_val:
    df_val_all = pd.concat(all_models_val, ignore_index=True)
    df_test_all = pd.concat(all_models_test, ignore_index=True)

    df_val_all.to_csv("outer10_val_results_ALL.csv", index=False)
    df_test_all.to_csv("outer10_test_results_ALL.csv", index=False)

    summary_val_all = df_val_all.groupby(["model"])[["val_iou", "val_acc", "val_loss"]].agg(["mean", "std"]).reset_index()
    summary_test_all = df_test_all.groupby(["model"])[["test_iou", "test_acc", "test_loss"]].agg(["mean", "std"]).reset_index()

    summary_val_all.to_csv("outer10_val_summary_ALL.csv", index=False)
    summary_test_all.to_csv("outer10_test_summary_ALL.csv", index=False)

    print("\nGLOBAL VAL Summary:")
    print(summary_val_all)
    print("\nGLOBAL TEST Summary:")
    print(summary_test_all)

    best_idx = summary_test_all[("test_iou", "mean")].idxmax()
    best_row = summary_test_all.iloc[best_idx]
    print("\nBest by TEST IoU (mean across folds) — GLOBAL:")
    print(best_row)
