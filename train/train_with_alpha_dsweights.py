import os
import csv
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

# -----------------------------
# Imports
# -----------------------------
from utils.dataset import ImageDataset, base_transform, train_transform
from utils.dataprocess import DatasetSplitter
from models.HAUNET_3Plus import UNet_3Plus_DeepSup_AC, DeepSupervisionLoss
from models.UNet_3Plus import UNet_3Plus, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_CGM
from models.Attention_UNet import AttentionUNet
from models.UNet_Ayon import UNet
from models.AUNET_Uncertainty import UNet_3Plus_DeepSup_AU, CombinedLoss, DeepSupervisionLossUncertainty

# -----------------------------
# Reproducibility
# -----------------------------
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)

# -----------------------------
# Metrics
# -----------------------------
def compute_iou(pred, target, threshold=0.5, eps=1e-6):
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

# -----------------------------
# Dataset Split
# -----------------------------
image_dir = "./Sinkhole_dataset_SAMRefiner/images"
mask_dir  = "./Sinkhole_dataset_SAMRefiner/masks"

image_list = sorted(os.listdir(image_dir))
mask_list  = sorted(os.listdir(mask_dir))
data = list(zip(image_list, mask_list))

train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data   = train_test_split(temp_data, test_size=0.5, random_state=42)

train_dataset = ImageDataset(train_data, image_dir, mask_dir, transform=train_transform, seed=42)
val_dataset   = ImageDataset(val_data, image_dir, mask_dir, transform=base_transform, seed=42)
test_dataset  = ImageDataset(test_data, image_dir, mask_dir, transform=base_transform, seed=42)

splitter     = DatasetSplitter(train_dataset, val_dataset, test_dataset, batch_size=16, seed=42)
train_loader = splitter.train_loader
val_loader   = splitter.val_loader
test_loader  = splitter.test_loader

# -----------------------------
# Ensure model save directory exists
# -----------------------------
os.makedirs("./pretrained_models", exist_ok=True)
os.makedirs("./pretrained_models/alpha_fine", exist_ok=True)

# -----------------------------
# Model Constructors
# -----------------------------
models = [
    ("HAUNet3+", lambda: UNet_3Plus_DeepSup_AC(in_channels=3, n_classes=1)),
]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Training Parameters
# -----------------------------
num_epochs = 500
learning_rate = 5e-4
patience = 20
min_delta = 0.0

# Fine alpha sweep: 0.65, 0.66, ..., 0.75 (includes 0.67, etc.)
alpha_values = [round(a, 2) for a in np.arange(0.65, 0.751, 0.01)]

ds_weight_sets = {
    "linear":  (1.0, 0.8, 0.6, 0.4, 0.2),
    "exp":     (1.0, 0.5, 0.25, 0.125, 0.0625),
    "uniform": (1.0, 1.0, 1.0, 1.0, 1.0),
}

# -----------------------------
# Training Loop
# -----------------------------
mlflow.set_experiment("Sinkhole_Research_Nobel_FineTune")

parent_run_name = "HAUNet3+_alpha_fine_0.65_0.75"
with mlflow.start_run(run_name=parent_run_name):
    mlflow.set_tag("sweep_type", "alpha_fine_search")
    mlflow.set_tag("alpha_range", "0.65-0.75 step 0.01")
    mlflow.set_tag("model_family", "HAUNet3+")

    for alpha in alpha_values:
        for ds_name, ds_weights in ds_weight_sets.items():
            for model_name, model_fn in models:
                print(f"\nTraining {model_name} | alpha={alpha} | ds={ds_name}")

                # fresh model for every run
                model = model_fn().to(device)

                # fresh criterion each run
                if "DeepSup" in model_name or "HAUNet3+" in model_name:
                    criterion = DeepSupervisionLoss(alpha=alpha, ds_weights=ds_weights)
                else:
                    criterion = CombinedLoss(alpha=alpha)

                # fresh optimizer + scheduler each run
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

                best_loss = float("inf")
                epochs_no_improve = 0
                early_stop = False

                run_name = f"{model_name}_alpha{alpha}_ds{ds_name}"

                # IMPORTANT: nested=True -> child run under the parent sweep run
                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.log_params({
                        "alpha": alpha,
                        "ds_weights": ds_weights,
                        "num_epochs": num_epochs,
                        "learning_rate": learning_rate,
                        "model_name": model_name,
                        "seed": 42,
                        "device": str(device),
                    })

                    for epoch in range(num_epochs):
                        if early_stop:
                            break

                        # ---- Training ----
                        model.train()
                        train_loss, train_iou = 0.0, 0.0

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
                            train_loss += loss.item()
                            train_iou  += compute_iou(preds, masks)

                        avg_train_loss = train_loss / len(train_loader)
                        avg_train_iou  = train_iou / len(train_loader)

                        # ---- Validation ----
                        model.eval()
                        val_loss, val_iou = 0.0, 0.0

                        with torch.no_grad():
                            for val_images, val_masks in val_loader:
                                val_images, val_masks = val_images.to(device), val_masks.to(device)
                                if val_masks.dim() == 3:
                                    val_masks = val_masks.unsqueeze(1)

                                val_outputs = model(val_images)
                                v_loss = criterion(val_outputs, val_masks)

                                v_preds = val_outputs[0] if isinstance(val_outputs, tuple) else val_outputs
                                val_loss += v_loss.item()
                                val_iou  += compute_iou(v_preds, val_masks)

                        avg_val_loss = val_loss / len(val_loader)
                        avg_val_iou  = val_iou / len(val_loader)

                        mlflow.log_metrics({
                            "train_loss": avg_train_loss,
                            "train_iou": avg_train_iou,
                            "val_loss": avg_val_loss,
                            "val_iou": avg_val_iou,
                        }, step=epoch)

                        scheduler.step()

                        # ---- Early stopping + checkpoint ----
                        if avg_val_loss < best_loss - min_delta:
                            best_loss = avg_val_loss
                            epochs_no_improve = 0

                            save_path = f"./pretrained_models/alpha_fine/{model_name}_a{alpha}_ds{ds_name}.pth"
                            torch.save(model.state_dict(), save_path)
                            mlflow.log_artifact(save_path)

                            print(f"Best {model_name} saved @ epoch {epoch+1}, val_loss={best_loss:.4f}")
                        else:
                            epochs_no_improve += 1
                            if epochs_no_improve >= patience:
                                print(f"Early stopping @ epoch {epoch+1}")
                                early_stop = True

                    # ---- Final Test Evaluation ----
                    model.eval()
                    test_loss, test_iou, test_acc = 0.0, 0.0, 0.0

                    with torch.no_grad():
                        for test_images, test_masks in test_loader:
                            test_images, test_masks = test_images.to(device), test_masks.to(device)
                            if test_masks.dim() == 3:
                                test_masks = test_masks.unsqueeze(1)

                            test_outputs = model(test_images)
                            t_loss = criterion(test_outputs, test_masks)

                            t_preds = test_outputs[0] if isinstance(test_outputs, tuple) else test_outputs
                            test_loss += t_loss.item()
                            test_iou  += compute_iou(t_preds, test_masks)
                            test_acc  += compute_pixel_accuracy(t_preds, test_masks)

                    avg_test_loss = test_loss / len(test_loader)
                    avg_test_iou  = test_iou / len(test_loader)
                    avg_test_acc  = test_acc / len(test_loader)

                    mlflow.log_metrics({
                        "test_loss": avg_test_loss,
                        "test_iou": avg_test_iou,
                        "test_acc": avg_test_acc,
                    })

                    print(f"Test Results: loss={avg_test_loss:.4f}, IoU={avg_test_iou:.4f}, Acc={avg_test_acc:.4f}")

                    # Log full model (optional but you requested logging)
                    mlflow.pytorch.log_model(model, artifact_path=model_name)
