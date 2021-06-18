import os
import torch
from torch import tensor
from torch.utils.data import Dataset,DataLoader
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import pandas as pd


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


IMG_SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SHOPEE_TRANSFORM = albumentations.Compose([
                albumentations.Resize(IMG_SIZE,IMG_SIZE,always_apply=True),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.Rotate(limit=120, p=0.8),
                albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
                albumentations.Normalize(mean = MEAN, std = STD),
                ToTensorV2(p=1.0),  ## outshape [w,h,3] -> [3,w,h]
            ])
    
def BuildImageDataloader(df, image_folder, transform=SHOPEE_TRANSFORM, batch_size=32, num_workers=4, device='cpu'):
    dataset = ShopeeDataset(df, image_folder, transform=transform, device=device)

    trainloader = DataLoader(
        dataset,
        batch_size = batch_size,
        pin_memory = device == 'cpu',
        num_workers = num_workers,
        shuffle = True,
        drop_last = True
    )

    return trainloader


SHOPEE_TRANSFORM_INFER = albumentations.Compose([
            albumentations.Resize(IMG_SIZE,IMG_SIZE,always_apply=True),
            albumentations.Normalize(mean = MEAN, std = STD),
            ToTensorV2(p=1.0)
        ])

def infer_transform_with_size(img_size):
    return albumentations.Compose([
            albumentations.Resize(img_size,img_size,always_apply=True),
            albumentations.Normalize(mean = MEAN, std = STD),
            ToTensorV2(p=1.0)
        ])

class ShopeeInferenceDataset(Dataset):
    def __init__(self, image_paths, transforms=SHOPEE_TRANSFORM_INFER):

        self.image_paths = image_paths
        self.augmentations = transforms

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']       

        return image,torch.tensor(1)


def read_matching_dataset(csv, img_folder):
    df = pd.read_csv(csv)
    if 'label_group' in df:
        tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
        df['matches'] = df['label_group'].map(tmp)
        df['matches'] = df['matches'].apply(lambda x: ' '.join(x))
    image_paths = img_folder + df['image']
    return df, image_paths

def BuildInferDataloader(csv, img_folder, img_size=512, batch_size=32, num_workers=4, device='cpu'):
    df, img_paths = read_matching_dataset(csv, img_folder)

    transfroms = infer_transform_with_size(img_size)

    infer_dataset = ShopeeInferenceDataset(img_paths, transforms=transfroms)

    infer_dataloader = DataLoader(
        infer_dataset,
        batch_size = batch_size,
        pin_memory = device == 'cpu',
        num_workers = num_workers,
        shuffle = False,
        drop_last = False,
    )

    return df, infer_dataloader
