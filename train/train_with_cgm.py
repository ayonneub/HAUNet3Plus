from utils.dataset import ImageDataset
from PIL import Image
import torchvision.transforms as T
from utils.dataloader import train_loader, val_loader
import torch
from models.UNet_3Plus import UNet_3Plus, UNet_3Plus_DeepSup_CGM, UNet_3Plus_DeepSup
from loss.bceLoss import BCE_loss
from loss.iouLoss import IOU_loss
from loss.msssimLoss import MSSSIM

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
#image, mask= dataset[0]
# print("Image shape:", image.shape)
# print("Mask shape:", mask.shape)
# print(len(dataset))
# for images, masks in train_loader:
#     print(images.shape, masks.shape)
#     break
#set device
#device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
#print("Is CUDA available?", torch.cuda.is_available())
#Instatiate the model
model=UNet_3Plus_DeepSup_CGM(in_channels=3, n_classes=1)
model=model.to(device)

#Instantiate MS-SSIM loss
msssim_loss= MSSSIM(window_size=11, size_average=True, channel=1)

#optimizer
optimizer= torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs= 100
#image, mask=dataset[0]
#mask.min()
best_loss= float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_iou= 0.0
    for images,masks in train_loader:
        images= images.to(device)
        masks= masks.to(device)

        optimizer.zero_grad()
        outputs= model(images)

        #updated  by ayon
        if isinstance(outputs,tuple):
            bce= sum(BCE_loss(out,masks) for out in outputs) / len(outputs)
            iou= sum(IOU_loss(out, masks) for out in outputs) / len(outputs)
            msssim= sum(msssim_loss(out, masks) for out in outputs) / len(outputs)
        else:
            bce= BCE_loss(outputs, masks)
            iou= IOU_loss(outputs, masks)
            msssim= msssim_loss(outputs, masks)
        
        
        # bce= BCE_loss(outputs, masks)
        # iou= IOU_loss(outputs, masks)
        # msssim= msssim_loss(outputs, masks)

        total_loss=bce+iou+(1-msssim)
        total_loss.backward()
        optimizer.step()

        running_loss+=total_loss.item()
        running_iou+= iou.item()

    avg_train_loss=running_loss/len(train_loader)
    mean_iou= running_iou/len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Mean IOU: {mean_iou:.4f}")

    #save the best model
    if avg_train_loss<best_loss:
        best_loss= avg_train_loss
        torch.save(model.state_dict(), 'best_model_UNet_3Plus_DeepSup_CGM_with_sam_refiner_softgating.pth')
        print(f"Best model saved at epoch {epoch+1} with loss {best_loss:.4f}")

