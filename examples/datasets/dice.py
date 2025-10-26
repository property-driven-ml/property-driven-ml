"""
Dice dataset creation utilities.

This module provides functions to create and load a custom datasets
for multi-label classification of playing dice faces.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple
from PIL import Image

from property_driven_ml.training.mode import Mode

from examples.models import DiceNet

import pandas as pd
import numpy as np
import os


class DiceDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_path: str, image_dir: str, transform=None, indices: np.ndarray = None
    ):
        """
        Initialize DiceDataset.

        Args:
            csv_path: Path to the file defining the labels for each image.
            image_dir: Path to the image files.
            transform: A function/transform that takes in an PIL image
            and returns a transformed version.
        """
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        self.data = self.data.iloc[indices].reset_index(drop=True)

    def get_mean_std(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        imgs = []

        for row in self.data.itertuples():
            img_path = os.path.join(self.image_dir, row.filename)
            img = Image.open(img_path).convert("RGB")
            img = np.asarray(img, dtype=np.float32) / 255.0
            imgs.append(img)

        imgs = np.stack(imgs)  # N,H,W,C
        return tuple(imgs.mean(axis=(0, 1, 2)).tolist()), tuple(
            imgs.std(axis=(0, 1, 2)).tolist()
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row.iloc[1:].to_numpy(dtype=np.float32))
        return image, labels


def create_dice_datasets(
    batch_size: int, train_split: float = 0.8, normalise: bool = True, seed: int = 42
):
    """
    Create dice train and test data loaders.

    Args:
        batch_size: Size of training batches

    Returns:
        Tuple of (train_loader, test_loader, model, (mean, std))
    """
    csv_path = "../data/dice/labels.csv"
    image_dir = "../data/dice/"

    # create train and test splits
    data_all = pd.read_csv(csv_path)
    perm = np.random.RandomState(seed).permutation(len(data_all))
    split_idx = int(train_split * len(data_all))
    train_idx, test_idx = perm[:split_idx], perm[split_idx:]

    # get mean and std
    mean, std = DiceDataset(csv_path, image_dir, indices=train_idx).get_mean_std()
    print(f"mean={mean}, std={std}")

    train_transforms = [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        transforms.ToTensor(),
    ]

    test_transforms = [
        transforms.ToTensor(),
    ]

    # only add mean / std normalisation for training, not for verification
    if normalise:
        train_transforms.append(transforms.Normalize(mean, std))
        test_transforms.append(transforms.Normalize(mean, std))

    dataset_train = DiceDataset(
        csv_path,
        image_dir,
        transform=transforms.Compose(train_transforms),
        indices=train_idx,
    )
    dataset_test = DiceDataset(
        csv_path,
        image_dir,
        transform=transforms.Compose(test_transforms),
        indices=test_idx,
    )

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = DiceNet()
    return train_loader, test_loader, model, (mean, std), Mode.MultiLabelClassification
