"""
Dataset Loader for Thai Land Title Deeds
========================================

Supports CHULA training and validation pipeline with standard
augmentations and preprocessing.

Author: Teerapong Panboonyuen
"""

import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class LandTitleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".png") or f.endswith(".jpg")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx].replace(".jpg", ".png").replace(".jpeg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask).squeeze().long()

        return image, mask

def get_land_title_dataset(data_config):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    train_dataset = LandTitleDataset(data_config['train_path'], transform=transform)
    val_dataset = LandTitleDataset(data_config['val_path'], transform=transform)
    return train_dataset, val_dataset