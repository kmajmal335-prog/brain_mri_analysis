import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2


class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None, target_size=(224, 224)):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.target_size = target_size
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []
        self.masks = []  # For segmentation

        self._load_data()

    def _load_data(self):
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])

                    # Load corresponding mask if exists
                    mask_path = img_path.with_suffix('.png')
                    if mask_path.exists():
                        self.masks.append(str(mask_path))
                    else:
                        self.masks.append(None)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        # Resize
        image = image.resize(self.target_size, Image.BILINEAR)

        # Load mask if available
        mask = None
        if self.masks[idx]:
            mask = Image.open(self.masks[idx]).convert('L')
            mask = mask.resize(self.target_size, Image.NEAREST)
            mask = np.array(mask) / 255.0
            mask = torch.FloatTensor(mask).unsqueeze(0)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if mask is not None:
            return image, label, mask
        return image, label


def get_data_loaders(config):
    """Create train, validation, and test data loaders"""

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(config.ROTATION_RANGE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=config.BRIGHTNESS_RANGE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = BrainTumorDataset(
        config.TRAIN_DIR,
        mode='train',
        transform=train_transform,
        target_size=config.IMG_SIZE_CLF
    )

    # Split train into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    test_dataset = BrainTumorDataset(
        config.TEST_DIR,
        mode='test',
        transform=val_transform,
        target_size=config.IMG_SIZE_CLF
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader