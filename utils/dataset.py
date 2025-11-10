"""
Dataset loader and preprocessing for person segmentation

Handles:
- Image and mask loading
- Data augmentation using Albumentations
- Train/Val/Test splits
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PersonSegmentationDataset(Dataset):
    """
    Dataset class for person segmentation
    
    Expected directory structure:
    data/
        images/
            img1.jpg
            img2.jpg
            ...
        masks/
            img1.png
            img2.png
            ...
    """
    
    def __init__(self, image_dir, mask_dir, transform=None, image_size=256):
        """
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
            transform: Albumentations transform pipeline
            image_size: Size to resize images to
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        
        # Get list of images
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"Found {len(self.images)} images in {image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Load and preprocess image and mask
        
        Returns:
            image: Tensor (3, H, W)
            mask: Tensor (1, H, W)
        """
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (assuming same name but in masks folder)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # If mask doesn't exist, create empty mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Ensure mask is binary
        mask = (mask > 127).astype(np.uint8)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Add channel dimension to mask
        mask = mask.unsqueeze(0).float()
        
        return image, mask


def get_train_transform(image_size=256):
    """
    Training data augmentation pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transform(image_size=256):
    """
    Validation/test data transformation (no augmentation)
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def create_data_loaders(data_dir, batch_size=8, num_workers=2, image_size=256):
    """
    Create train and validation data loaders
    
    Args:
        data_dir: Root directory containing 'images' and 'masks' folders
        batch_size: Batch size for training
        num_workers: Number of worker processes
        image_size: Size to resize images to
        
    Returns:
        train_loader, val_loader
    """
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    
    # Create datasets
    train_dataset = PersonSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=get_train_transform(image_size),
        image_size=image_size
    )
    
    val_dataset = PersonSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=get_val_transform(image_size),
        image_size=image_size
    )
    
    # Split dataset (80-20 train-val split)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    _, val_dataset = torch.utils.data.random_split(
        val_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    data_dir = "data"
    if os.path.exists(data_dir):
        train_loader, val_loader = create_data_loaders(data_dir, batch_size=4)
        
        # Test loading a batch
        images, masks = next(iter(train_loader))
        print(f"Batch images shape: {images.shape}")
        print(f"Batch masks shape: {masks.shape}")
    else:
        print(f"Data directory '{data_dir}' not found. Please run download_dataset.py first.")
