import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2


class ImageDataset(Dataset):
    def __init__(self, data, image_dir, mask_dir, transform=None, seed=42):
        self.data= data
        self.image_dir= image_dir
        self.mask_dir= mask_dir
        self.transform= transform
        self.seed= seed
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        image_filename, mask_filename = self.data[idx]

        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_np = np.array(image)
        mask_np = np.array(mask)

        if self.transform:
            np.random.seed(self.seed + idx)
            augmented = self.transform(image=image_np, mask=mask_np)
            image, mask = augmented['image'], augmented['mask']

            if mask.dtype == torch.uint8:
                mask = mask.float() / 255.0
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            

        return image, mask

base_transform= A.Compose([
    A.Resize(
        height= 320,
        width= 320,
        interpolation= cv2.INTER_LINEAR,
        mask_interpolation= cv2.INTER_NEAREST
    ),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

train_transform= A.Compose([
    A.Resize(
        height= 320,
        width= 320,
        interpolation= cv2.INTER_LINEAR,
        mask_interpolation= cv2.INTER_NEAREST
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GridDistortion(p=0.2),
    A.ElasticTransform(p=0.2),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

