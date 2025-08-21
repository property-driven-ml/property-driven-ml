"""
GTSRB dataset creation utilities.

This module provides functions to create and load GTSRB (German Traffic Sign Recognition Benchmark)
datasets for traffic sign classification tasks.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

from examples.models import GTSRBNet


def create_gtsrb_datasets(
    batch_size: int,
) -> Tuple[
    DataLoader, DataLoader, torch.nn.Module, Tuple[Tuple[float, ...], Tuple[float, ...]]
]:
    """
    Create GTSRB train and test data loaders.

    Args:
        batch_size: Size of training batches

    Returns:
        Tuple of (train_loader, test_loader, model, (mean, std))

    Raises:
        ValueError: If GTSRB dataset is not available in the current torchvision version
    """
    mean, std = (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)

    transform_train = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Note: GTSRB might not be available in older torchvision versions
    # This will need to be handled gracefully
    try:
        dataset_train = datasets.GTSRB(
            "data", split="train", download=True, transform=transform_train
        )
        dataset_test = datasets.GTSRB(
            "data", split="test", download=True, transform=transform_test
        )
    except AttributeError:
        raise ValueError("GTSRB dataset not available in this torchvision version")

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = GTSRBNet()

    return train_loader, test_loader, model, (mean, std)
