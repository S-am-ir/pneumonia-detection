"""
Data loading, preprocessing, and dataset creation.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import DATA_DIR, BATCH_SIZE


def create_dataframe(base_dir):
    """Create DataFrame with paths, labels, and splits."""
    data = []
    for split in ['train', 'val', 'test']:
        for label in ['NORMAL', 'PNEUMONIA']:
            folder = base_dir / split / label
            if not folder.exists():
                print(f"⚠️ Folder not found: {folder}")
                continue
            for img_path in folder.glob('*.jpeg'):
                data.append({
                    'image_path': str(img_path),
                    'label': 0 if label == 'NORMAL' else 1,
                    'label_name': label,
                    'split': split
                })
    return pd.DataFrame(data)


class ChestXRayDataset(Dataset):
    """PyTorch Dataset for chest X-ray images."""
    
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.array(Image.open(row['image_path']).convert('RGB'))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = torch.tensor(row['label'], dtype=torch.long)
        return image, label


def get_transforms():
    """Return training and validation transforms."""
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Rotate(limit=10, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


def get_dataloaders():
    """Create and return train, validation, and test dataloaders."""
    df = create_dataframe(DATA_DIR)
    print("✅ Total images found:", len(df))
    
    train_transform, val_transform = get_transforms()
    
    # Improved split: combine original train + val, then create proper validation set
    combined_df = df[df['split'].isin(['train', 'val'])].copy()
    
    train_df, val_df = train_test_split(
        combined_df,
        test_size=0.20,
        stratify=combined_df['label'],
        random_state=42
    )
    
    test_df = df[df['split'] == 'test'].copy()
    
    print(f"✅ Split created:")
    print(f"Train samples : {len(train_df)}")
    print(f"Val samples   : {len(val_df)}")
    print(f"Test samples  : {len(test_df)}")
    
    # Create datasets and loaders
    train_dataset = ChestXRayDataset(train_df, train_transform)
    val_dataset = ChestXRayDataset(val_df, val_transform)
    test_dataset = ChestXRayDataset(test_df, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=2, pin_memory=True)
    
    print(f"✅ Dataloaders ready!")
    
    return train_loader, val_loader, test_loader, train_df
