import os
import torch
from torch import tensor
from torch.utils.data import Dataset,DataLoader
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import cv2


class ShopeeDataset(Dataset):
    
    def __init__(self, df, image_folder, transform = None, device="cpu"):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.image_folder = image_folder
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        img_path = os.path.join(self.image_folder,row.image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        label = row.label_group
        
        return {
            "image": image.float().to(self.device),
            "label": tensor(label, dtype=torch.long, device=self.device)
        }

    def __len__(self):
        return self.df.shape[0]


class ImageDataLoader:
    def __init__(self, IMG_SIZE=512, MEAN=None, STD=None):
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        self.SHOPEE_TRANSFORM = albumentations.Compose([
                        albumentations.Resize(IMG_SIZE,IMG_SIZE,always_apply=True),
                        albumentations.HorizontalFlip(p=0.5),
                        albumentations.VerticalFlip(p=0.5),
                        albumentations.Rotate(limit=120, p=0.8),
                        albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
                        albumentations.Normalize(mean = MEAN, std = STD),
                        ToTensorV2(p=1.0),  ## outshape [w,h,3] -> [3,w,h]
                    ])
        
    def BuildImageDataloader(self, df, image_folder, transform=None, batch_size=32, num_workers=4, device='cpu'):
        if not transform:
            transform=self.SHOPEE_TRANSFORM
        dataset = ShopeeDataset(df, image_folder, transform=transform, device=device)

        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            pin_memory = device == 'cpu',
            num_workers = num_workers,
            shuffle = True,
            drop_last = True
        )

        return trainloader