from utils.dataset import ImageDataset
from PIL import Image
import torchvision.transforms as T
from utils.dataloader import train_loader, val_loader
import torch
from models.UNet_3Plus import UNet_3Plus, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_CGM
from loss.bceLoss import BCE_loss
from loss.iouLoss import IOU_loss
from loss.msssimLoss import MSSSIM
import numpy as np

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

#set device
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')

#Instatiate the model
#model= UNet_3Plus(in_channels=3, n_classes=1)
model=UNet_3Plus_DeepSup(in_channels=3, n_classes=1)
model=model.to(device)

#Instantiate MS-SSIM loss
msssim_loss= MSSSIM(window_size=11, size_average=True, channel=1)

#optimizer
optimizer= torch.optim.Adam(model.parameters(), lr=1e-4)

best_loss= float('inf')

model= UNet_3Plus_DeepSup(in_channels=3, n_classes=1)
model.load_state_dict(torch.load('best_model_UNet_3Plus_DeepSup_with_sam_refiner.pth', map_location=device))
model=model.to(device)
model.eval()

thresholds=[round(f,2) for f in np.arange(0.5,0.8, 0.01)]
best_iou= 0
best_threshold=0.5
for th in thresholds:
    total_iou=0.0
    num_batches=0
    total_acc= 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images= images.to(device)
            masks= masks.to(device)

            outputs= model(images)

            if isinstance(outputs, tuple):
                outputs= outputs[0]

            preds= torch.sigmoid(outputs)>th
            iou_loss= IOU_loss(outputs, masks)
            iou_score= 1-iou_loss.item()
            total_iou+= iou_score

            correct= (preds.cpu()==masks.cpu()).float().mean().item()
            total_acc+=correct
            num_batches+=1

    mean_iou= total_iou/num_batches
    mean_acc=total_acc/num_batches
    print(f"Threshold: {th}, Mean IoU: {mean_iou:.4f}, Mean Accuracy: {mean_acc:.4f}")
    if mean_iou>best_iou:
        best_iou=mean_iou
        best_threshold= th
print(f"Best Threshold: {best_threshold}, Best Mean IoU: {best_iou:.4f}, Mean Accuracy: {mean_acc:.4f}")