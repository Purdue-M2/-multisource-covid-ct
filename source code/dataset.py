"""CT scan dataset and augmentation pipeline."""

import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class CovidCTDataset(Dataset):
    """Dataset for preprocessed multi-source COVID-19 CT scans.

    Each sample consists of 8 KDS-selected slices, a binary COVID label,
    and a source (hospital) identifier.
    """

    def __init__(self, scans, transform=None):
        self.scans = scans
        self.transform = transform

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        info = self.scans[idx]
        files = sorted([
            f for f in os.listdir(info['path'])
            if f.endswith('.png') and not f.startswith('.')
        ])

        images = []
        for f in files[:8]:
            img = cv2.imread(os.path.join(info['path'], f), cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if self.transform:
                img = self.transform(image=img)['image']
            images.append(img)

        while len(images) < 8:
            images.append(images[-1])

        slices = torch.stack(images[:8], dim=0)
        label = torch.tensor(info['label'], dtype=torch.float32)
        source = torch.tensor(info['source'], dtype=torch.long)
        return slices, label, source
