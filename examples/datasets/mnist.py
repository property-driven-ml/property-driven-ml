"""
MNIST dataset creation utilities.

This module provides functions to create and load MNIST datasets
for handwritten digit classification tasks.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

from property_driven_ml.training.mode import Mode

from examples.models import MnistNet


def create_mnist_datasets(
    batch_size: int,
) -> Tuple[
    DataLoader,
    DataLoader,
    torch.nn.Module,
    Tuple[Tuple[float, ...], Tuple[float, ...]],
    Mode,
]:
    """
    Create MNIST train and test data loaders.

    Args:
        batch_size: Size of training batches

    Returns:
        Tuple of (train_loader, test_loader, model, (mean, std))
    """
    mean, std = (0.1307,), (0.3081,)

    transform_train = transforms.Compose(
        [
            transforms.RandomRotation(10),
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

    dataset_train = datasets.MNIST(
        "data", train=True, download=True, transform=transform_train
    )
    dataset_test = datasets.MNIST(
        "data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = MnistNet()

    return train_loader, test_loader, model, (mean, std), Mode.MultiClassClassification
