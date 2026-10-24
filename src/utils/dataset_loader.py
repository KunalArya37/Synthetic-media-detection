"""
Dataset Pipeline & DataLoaders with Advanced Forensic Augmentation
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from src.config import (
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
)


class SyntheticMediaDataset(Dataset):
    """
    Standard PyTorch dataset for binary real vs synthetic media classification.
    Expects subdirectories 'real' (label 0) and 'synthetic' (label 1).
    """
    def __init__(self, root_dir: Path, transform: Optional[transforms.Compose] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        real_dir = self.root_dir / "real"
        synth_dir = self.root_dir / "synthetic"

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        if real_dir.exists():
            for p in real_dir.glob("*"):
                if p.suffix.lower() in valid_exts and p.is_file():
                    self.samples.append((p, 0))

        if synth_dir.exists():
            for p in synth_dir.glob("*"):
                if p.suffix.lower() in valid_exts and p.is_file():
                    self.samples.append((p, 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to black image if corrupted
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(img_size: int = IMG_SIZE) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Generates training augmentations and validation/test transforms.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, eval_transform


def create_dataloaders(
    train_dir: Path, val_dir: Path, test_dir: Optional[Path] = None,
    batch_size: int = BATCH_SIZE, img_size: int = IMG_SIZE
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Creates optimized PyTorch DataLoaders for training, validation, and testing.
    """
    train_tf, eval_tf = get_transforms(img_size)

    train_dataset = SyntheticMediaDataset(train_dir, transform=train_tf)
    val_dataset = SyntheticMediaDataset(val_dir, transform=eval_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True if len(train_dataset) > batch_size else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = None
    if test_dir and test_dir.exists():
        test_dataset = SyntheticMediaDataset(test_dir, transform=eval_tf)
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY
            )

    return train_loader, val_loader, test_loader
