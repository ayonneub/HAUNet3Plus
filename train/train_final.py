from utils.dataset import ImageDataset
from PIL import Image
import torchvision.transforms as T
#from utils.dataloader import train_loader, val_loader
import torch
from torch.utils.data import random_split, DataLoader
from models.UNet_3Plus import UNet_3Plus, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_CGM
from models.UNet import UNet
from loss.bceLoss import BCE_loss
from loss.iouLoss import IOU_loss
from loss.msssimLoss import MSSSIM
import random
import numpy as np
from utils.dataprocess import DatasetSplitter
import matplotlib.pyplot as plt
import csv
import mlflow
import mlflow.pytorch

#seeding everything
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark= False

set_seeds(42)

#Splitting the dataset

train_transform =T.Compose([
    T.Resize((320,320)),
    T.ToTensor()
])
image_transform= T.Compose([
    T.Resize((320,320)),
    T.ToTensor()
])

mask_transform= T.Compose([
    T.Resize((320,320),interpolation=Image.NEAREST),
    T.ToTensor()

])

dataset= ImageDataset(
    image_dir='./Sinkhole_dataset_SAMRefiner/images',
    mask_dir='./Sinkhole_dataset_SAMRefiner/masks',
    image_transform=image_transform,
    mask_transform=mask_transform
)

splitter= DatasetSplitter(dataset, batch_size=8, train_ratio= 0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
train_loader= splitter.train_loader
val_loader= splitter.val_loader
test_loader= splitter.test_loader

#set device
device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
#Instatiate the model
#model= UNet_3Plus(in_channels=3, n_classes=1)
model=UNet_3Plus_DeepSup(in_channels=3, n_classes=1)
model=model.to(device)

#Instantiate MS-SSIM loss
msssim_loss= MSSSIM(window_size=11, size_average=True, channel=1)

#optimizer
optimizer= torch.optim.Adam(model.parameters(), lr=1e-4)

best_loss= float('inf')

#Training the model
#set ML flow experiment
mlflow.set_experiment("UNet3Plus_DeepSup_Sinkhole_Research")

#define hyperparameters
num_epochs= 100
learning_rate= 1e-4
batch_size= 8

#Start MLflow run
with mlflow.start_run():
    #Log hyperparameters
    # Logs the number of epochs, Learning rate and Batch size as parameters in MLflow
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("model_name", "UNet3Plus_DeepSup")
    mlflow.log_param("model_type", "UNet3Plus_DeepSup")
    mlflow.log_param("device", str(device))
    mlflow.log_param("seed", 42)



    #Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss= 0.0
        running_iou= 0.0
        for images, masks in train_loader:
            images= images.to(device)
            masks= masks.to(device)

            optimizer.zero_grad()
            outputs= model(images)

            #Handle tuple or single output
            if isinstance(outputs,tuple):
                bce=sum(BCE_loss(out, masks) for out in outputs)/ len(outputs)
                iou=sum(IOU_loss(out, masks) for out in outputs)/ len(outputs)
                msssim=sum(msssim_loss(out,masks) for out in outputs)/ len(outputs)
            else:
                bce= BCE_loss(outputs, masks)
                iou= IOU_loss(outputs, masks)
                msssim= msssim_loss(outputs, masks)
            
            total_loss= bce + iou + (1-msssim)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_iou += iou.item()
        
        avg_train_loss= running_loss / len(train_loader)
        mean_iou= running_iou / len(train_loader)

        #log metrics to MLflow
        # The average training loss, mean IOU, BCE loss and MSSSIM loss are logged as metrics in mlflow.
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("mean_iou", mean_iou, step= epoch)
        mlflow.log_metric("bce_loss", bce.item(), step=epoch)
        mlflow.log_metric("msssim_loss", msssim.item(), step= epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Mean IOU: {mean_iou:.4f}")

        #Save the best model
        if avg_train_loss < best_loss:
            best_loss=avg_train_loss
            #Save the model locally
            torch.save(model.state_dict(), "./pretrained_models/best_model_unet3plus_DeepSup.pth")
            #Log the model as an artifact in MLflow
            # This logs the best model as an artifact in mlflow, so it shows up under the artifacts section of the run.
            mlflow.log_artifact("./pretrained_models/best_model_unet3plus_DeepSup.pth")
            #Log the PYTorch model to MLflow
           
            print(f"Best model saved at epoch {epoch+1} with loss {best_loss:.4f}")
    mlflow.pytorch.log_model(model, name= "UNet3plus_DeepSup")
    #Register the model in MLflow Model Registry
    model_uri= f"run:/{mlflow.active_run().info.run_id}/UNet3+_DeepSup"
    mlflow.register_model(model_uri, "UNet3+_DeepSup_Sinkhole_Model")

