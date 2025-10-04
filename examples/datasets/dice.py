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
        self,
        csv_path: str,
        image_dir: str,
        transform=None,
        train: bool = True,
        split: float = 0.8,
    ):
        """
        Initialize DiceDataset.

        Args:
            csv_path: Path to the file defining the labels for each image.
            image_dir: Path to the image files.
            train: If True, creates dataset for training, otherwise for testing.
            transform: A function/transform that takes in an PIL image
            and returns a transformed version.
            split: Train / test set ratio.
        """
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.train = train

        indices = np.random.permutation(len(self.data))
        split_idx = int(split * len(self.data))

        if self.train:
            self.data = self.data.iloc[indices[:split_idx]]
        else:
            self.data = self.data.iloc[indices[split_idx:]]

    def get_mean_std(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        assert self.train, (  # nosec
            "mean/std should be calculated on the training data so as not to leak information about unseen data"
        )

        imgs = []

        for row in self.data.itertuples():
            img_path = os.path.join(self.image_dir, row.filename)
            img = Image.open(img_path).convert("RGB")
            img = np.array(img) / 255.0
            imgs.append(img)

        imgs = np.stack(imgs)  # N, H, W, C

        return tuple(imgs.mean(axis=(0, 1, 2)).tolist()), tuple(
            imgs.std(axis=(0, 1, 2)).tolist()
        )

    def print_label_balance_stats(self):
        label_counts = [0] * 6
        total = 0

        for row in self.data.itertuples(index=False):
            labels = row[1:]

            for i, val in enumerate(labels):
                label_counts[i] += int(val)

            total += 1

        mean = sum(label_counts) / len(label_counts)

        print(f"mean: {mean:.2f}\n")

        for i, count in enumerate(label_counts):
            print(f"label {i + 1}: on {count} images ({count / total:.2%} of images)")

        print(f"\ntotal images: {total}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[1:].values.astype("float32"))
        return image, labels


def create_dice_datasets(
    batch_size: int,
) -> Tuple[
    DataLoader,
    DataLoader,
    torch.nn.Module,
    Tuple[Tuple[float, ...], Tuple[float, ...]],
    Mode,
]:
    """
    Create dice train and test data loaders.

    Args:
        batch_size: Size of training batches

    Returns:
        Tuple of (train_loader, test_loader, model, (mean, std))
    """
    csv_path = "../data/dice/labels.csv"
    image_dir = "../data/dice/"

    mean, std = DiceDataset(csv_path, image_dir, train=True).get_mean_std()
    print(f"mean={mean}, std={std}")

    transform_train = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=0.1, contrast=0.2, saturation=0.3, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset_train = DiceDataset(
        csv_path, image_dir, train=True, transform=transform_train
    )
    dataset_test = DiceDataset(
        csv_path, image_dir, train=False, transform=transform_test
    )

    print("Train stats:\n")
    dataset_train.print_label_balance_stats()

    print("Test stats:\n")
    dataset_test.print_label_balance_stats()

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = DiceNet()

    return train_loader, test_loader, model, (mean, std), Mode.MultiLabelClassification
