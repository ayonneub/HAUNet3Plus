
from utils.dataset import ImageDataset, base_transform, train_transform
from PIL import Image
import torchvision.transforms as T
import torch
from torch.utils.data import random_split, DataLoader
from models.UNet_3Plus import UNet_3Plus, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_CGM
from models.Attention_UNet import AttentionUNet
#from models.DepthWise_UNet import DepthWiseUNet
from models.UNet_3Plus_Attention import UNet_3Plus_DeepSup
from models.UNet_2Plus import UNet_2Plus
from models.UNet_Ayon import UNet
from models.UNet_Deep_Attention_2 import UNet_3Plus_DeepSup
from loss.bceLoss import BCE_loss
from loss.iouLoss import IOU_loss
import torch.optim as optim
#from loss.msssimLoss import MSSSIM
import random
import numpy as np
from utils.dataprocess import DatasetSplitter
import matplotlib.pyplot as plt
import csv
import mlflow
import mlflow.pytorch
import datetime
import os
from sklearn.model_selection import train_test_split
from utils.dataset import ImageDataset, base_transform, train_transform
from pytorch_msssim import ms_ssim


# Seeding everything
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)

#Handling dataset and augmentation
image_dir = './Sinkhole_dataset_SAMRefiner/images'
mask_dir = './Sinkhole_dataset_SAMRefiner/masks'

image_list = sorted(os.listdir(image_dir))
mask_list = sorted(os.listdir(mask_dir))

# Ensure image and mask pairs match
assert len(image_list) == len(mask_list)
data = list(zip(image_list, mask_list)) 

# First split into train and temp (val + test)
train_data, temp_data = train_test_split(
    data, test_size=0.3, random_state=42)

# Then split temp into val and test
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, random_state=42)

train_dataset = ImageDataset(train_data, image_dir, mask_dir, transform=train_transform, seed=42)
val_dataset   = ImageDataset(val_data, image_dir, mask_dir, transform=base_transform, seed=42)
test_dataset  = ImageDataset(test_data, image_dir, mask_dir, transform=base_transform, seed=42)


splitter= DatasetSplitter(
    train_dataset= train_dataset,
    val_dataset= val_dataset,
    test_dataset= test_dataset,
    batch_size=16,
    seed=42
)
train_loader= splitter.train_loader
val_loader= splitter.val_loader
test_loader= splitter.test_loader



# Set device
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Define models to train
models = [

    ("UNet_3Plus_DeepSup_Attention_version_2", UNet_3Plus_DeepSup(in_channels=3,n_classes=1))
   # ("Attention_UNet", AttentionUNet(in_channels=3, out_channels=1)),
    #("UNet_2Plus", UNet_2Plus(in_channels=3, n_classes=1)),
    #("UNet_3Plus", UNet_3Plus(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True,is_batchnorm=True)),
   # ("UNet_3Plus_DeepSup", UNet_3Plus_DeepSup(in_channels=3, n_classes=1,feature_scale=4, is_deconv=True, is_batchnorm=True)),
    #("UNet_3Plus_DeepSup_CGM", UNet_3Plus_DeepSup_CGM(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True))
]

# Instantiate MS-SSIM loss
#msssim_loss = MSSSIM(window_size=11, size_average=True, channel=1)

# Define hyperparameters
num_epochs = 1000
learning_rate = 0.01
batch_size = 16
patience = 20
min_delta = 0 #Any improvement is considered.

# Set MLflow experiment
mlflow.set_experiment("Sinkhole_Research_2")

# Train each model
for model_name, model in models:
    print(f"\nTraining model: {model_name}")
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #Schedular
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode= 'min',
        factor=0.1,
        patience=5,
        min_lr= 1e-6

    )
    # Early stopping parameters
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    
    # Generate unique run name
    run_name = f"{model_name}_Experiment_1"
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_param("scheduler_type", "ReduceLROnPlateau")
        mlflow.log_param("scheduler_factor", scheduler.factor)
        mlflow.log_param("scheduler_patience", scheduler.patience)
        mlflow.log_param("scheduler_min_lr", scheduler.min_lrs[0])
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("device", str(device))
        mlflow.log_param("seed", 42)
        mlflow.log_param("patience", patience)
        mlflow.log_param("min_delta", min_delta)
        mlflow.log_param("Data Size","320*320")

        # Training loop
        for epoch in range(num_epochs):
            if early_stop:
                print(f"Early stopping triggered for {model_name} at epoch {epoch+1}")
                break

            model.train()
            running_loss = 0.0
            running_iou = 0.0
            running_bce= 0.0
            running_msssim= 0.0
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                # Handle tuple or single output
                if isinstance(outputs, tuple):
                    bce = sum(BCE_loss(out, masks) for out in outputs) / len(outputs)
                    iou = sum(IOU_loss(out, masks) for out in outputs) / len(outputs)
                    #msssim = sum(msssim_loss(out, masks) for out in outputs) / len(outputs)
                    #pytorch msssim loss
                    msssim= sum(ms_ssim(out,masks, data_range=1.0, size_average=True) for out in outputs)/len(outputs)
                else:
                    bce = BCE_loss(outputs, masks)
                    iou = IOU_loss(outputs, masks)
                    #msssim = msssim_loss(outputs, masks)
                    msssim= ms_ssim(outputs,masks, data_range=1.0, size_average= True)
                
                total_loss = bce + iou + (1 - msssim)
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
                running_iou += iou.item()
                running_bce+= bce.item()
                running_msssim+=(1-msssim).item()
            
            avg_train_loss = running_loss / len(train_loader)
            mean_iou = running_iou / len(train_loader)
            msssim_loss= running_msssim/len(train_loader)
            bce_loss= running_bce/len(train_loader)
            # Log training metrics to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("mean_iou", mean_iou, step=epoch)
            mlflow.log_metric("bce_loss", bce_loss, step=epoch)
            mlflow.log_metric("msssim_loss", msssim_loss, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
        
            print(f"Model: {model_name}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Mean IOU: {mean_iou:.4f}")

            # Validation phase
            model.eval()
            val_running_loss = 0.0
            val_running_iou = 0.0
            val_running_bce= 0.0
            val_running_msssim= 0.0
            with torch.no_grad():
                for val_images, val_masks in val_loader:
                    val_images = val_images.to(device)
                    val_masks = val_masks.to(device)

                    val_outputs = model(val_images)
                    if val_masks.dim()==3:
                        val_masks= val_masks.unsqueeze(1)
                    # Handle tuple or single output
                    if isinstance(val_outputs, tuple):
                        val_bce = sum(BCE_loss(out, val_masks) for out in val_outputs) / len(val_outputs)
                        val_iou = sum(IOU_loss(out, val_masks) for out in val_outputs) / len(val_outputs)
                        val_msssim = sum(ms_ssim(out, val_masks,data_range=1.0, size_average=True) for out in val_outputs) / len(val_outputs)
                    else:
                        val_bce = BCE_loss(val_outputs, val_masks)
                        val_iou = IOU_loss(val_outputs, val_masks)
                        val_msssim = ms_ssim(val_outputs, val_masks, data_range=1.0, size_average=True)
                    
                    val_total_loss = val_bce + val_iou + (1 - val_msssim)

                    val_running_loss += val_total_loss.item()
                    val_running_iou += val_iou.item()
                    val_running_bce+= val_bce.item()
                    val_running_msssim+= (1-val_msssim).item()

            avg_val_loss = val_running_loss / len(val_loader)
            val_mean_iou = val_running_iou / len(val_loader)
            val_msssim_loss= val_running_msssim/len(val_loader)
            val_bce_loss= val_running_bce/len(val_loader)
            # Log validation metrics to MLflow
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_mean_iou", val_mean_iou, step=epoch)
            mlflow.log_metric("val_bce_loss", val_bce_loss, step=epoch)
            mlflow.log_metric("val_msssim_loss", val_msssim_loss, step=epoch)

            print(f"Model: {model_name}, Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Mean IOU: {val_mean_iou:.4f}")

            #Step the schedular with validation loss
            scheduler.step(avg_val_loss)
            # Early stopping logic
            if avg_val_loss < best_loss - min_delta:
                best_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f"./pretrained_models/final_model_{model_name}_experiment_2.pth")
                mlflow.log_artifact(f"./pretrained_models/final_model_{model_name}_experiment_2.pth")
                print(f"Best model saved for {model_name} at epoch {epoch+1} with validation loss {best_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    early_stop = True
                    print(f"No improvement in validation loss for {patience} epochs for {model_name}. Stopping training.")

        # Log the final model to MLflow
        mlflow.pytorch.log_model(model, name=model_name)
        # Register the model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
        mlflow.register_model(model_uri, f"{model_name}_Sinkhole_Model")
