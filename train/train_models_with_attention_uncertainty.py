# -*- coding: utf-8 -*-
"""
Training script for UNet 3+ with Uncertainty-Guided Hybrid Attention on Sinkhole dataset.
Handles UGHA-UNet 3+ model output (segmentation + uncertainty maps) and loss with uncertainty regularization.
"""

from utils.dataset import ImageDataset, base_transform, train_transform
from PIL import Image
import torch
from torch.utils.data import DataLoader
from models.AUNET_Uncertainty import UNet_3Plus_DeepSup_AU, CombinedLoss, DeepSupervisionLossUncertainty  # Import UGHA model and loss
import random
import numpy as np
from utils.dataprocess import DatasetSplitter
import mlflow
import mlflow.pytorch
import datetime
import os
from sklearn.model_selection import train_test_split

def compute_iou(pred, target, threshold=0.5, eps=1e-6):
    """
    pred: model output (probabilities) [B,1,H,W]
    target: ground truth mask [B,1,H,W]
    """
    pred = (pred > threshold).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = (pred + target).sum(dim=(1,2,3)) - intersection
    iou = (intersection + eps) / (union + eps)

    return iou.mean().item()

# Seeding
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)

# Dataset & Dataloaders
image_dir = './Sinkhole_dataset_SAMRefiner/images'
mask_dir = './Sinkhole_dataset_SAMRefiner/masks'

image_list = sorted(os.listdir(image_dir))
mask_list = sorted(os.listdir(mask_dir))
assert len(image_list) == len(mask_list)
data = list(zip(image_list, mask_list))

# Split
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_dataset = ImageDataset(train_data, image_dir, mask_dir, transform=train_transform, seed=42)
val_dataset = ImageDataset(val_data, image_dir, mask_dir, transform=base_transform, seed=42)
test_dataset = ImageDataset(test_data, image_dir, mask_dir, transform=base_transform, seed=42)

splitter = DatasetSplitter(train_dataset, val_dataset, test_dataset, batch_size=16, seed=42)
train_loader = splitter.train_loader
val_loader = splitter.val_loader
test_loader = splitter.test_loader

# Device
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Models
models = [
    ("AUNET_Uncertainty", UNet_3Plus_DeepSup_AU(in_channels=3, n_classes=1))  # UGHA model
]

# Training Parameters
num_epochs = 1000
learning_rate = 5e-4
batch_size = 16
patience = 20
min_delta = 0.0

# Loss function
criterion = DeepSupervisionLossUncertainty(alpha=0.5, ds_weights=(1.0, 0.8, 0.6, 0.4, 0.2), unc_reg_weight=0.01)

# MLflow Experiment
mlflow.set_experiment("Sinkhole_Research_Nobel")

for model_name, model in models:
    print(f"\nTraining model: {model_name}")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    run_name = f"{model_name}_Experiment_1"

    with mlflow.start_run(run_name=run_name):
        # Log params
        mlflow.log_param("scheduler_type", "CosineAnnealingLR")
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("device", str(device))
        mlflow.log_param("seed", 42)
        mlflow.log_param("patience", patience)
        mlflow.log_param("min_delta", min_delta)
        mlflow.log_param("loss", "BCE+Dice (Deep Supervision with Uncertainty Regularization)")

        for epoch in range(num_epochs):
            if early_stop:
                print(f"Early stopping triggered for {model_name} at epoch {epoch+1}")
                break

            # -------- Training --------
            model.train()
            running_loss, running_iou = 0.0, 0.0

            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)

                optimizer.zero_grad()
                outputs, uncertainties = model(images)  # Unpack segmentation outputs and uncertainties
                loss = criterion(outputs, masks, uncertainties)
                loss.backward()
                optimizer.step()

                # Get main output (d1) for IoU
                preds = outputs[0]

                running_loss += loss.item()
                running_iou += compute_iou(preds, masks)

            avg_train_loss = running_loss / len(train_loader)
            avg_train_iou = running_iou / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_iou", avg_train_iou, step=epoch)

            print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}")

            # -------- Validation --------
            model.eval()
            val_running_loss, val_running_iou = 0.0, 0.0
            with torch.no_grad():
                for val_images, val_masks in val_loader:
                    val_images = val_images.to(device)
                    val_masks = val_masks.to(device)
                    if val_masks.dim() == 3:
                        val_masks = val_masks.unsqueeze(1)

                    val_outputs, val_uncertainties = model(val_images)
                    val_loss = criterion(val_outputs, val_masks, val_uncertainties)

                    # Get main output for IoU
                    val_preds = val_outputs[0]

                    val_running_loss += val_loss.item()
                    val_running_iou += compute_iou(val_preds, val_masks)

            avg_val_loss = val_running_loss / len(val_loader)
            avg_val_iou = val_running_iou / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_iou", avg_val_iou, step=epoch)

            print(f"[Epoch {epoch+1}/{num_epochs}] Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")

            # Scheduler + Early Stopping
            scheduler.step()
            if avg_val_loss < best_loss - min_delta:
                best_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f"./pretrained_models/final_model_{model_name}_nobel.pth")
                mlflow.log_artifact(f"./pretrained_models/final_model_{model_name}_nobel.pth")
                print(f"Best model saved at epoch {epoch+1}, val_loss={best_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    early_stop = True
                    print(f"Early stopping at epoch {epoch+1}")

        # Log final model
        mlflow.pytorch.log_model(model, artifact_path=model_name)
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
        mlflow.register_model(model_uri, f"{model_name}_Sinkhole_Model")