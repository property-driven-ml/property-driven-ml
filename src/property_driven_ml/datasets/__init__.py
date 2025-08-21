"""
Dataset creation and loading utilities.

This module provides functions to create and load datasets for different
machine learning tasks supported by the framework.
"""

from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader

from .base import DatasetCreator, SizedDataset, validate_dataset_result
from .mnist import create_mnist_datasets
from .alsomitra import create_alsomitra_datasets
from .gtsrb import create_gtsrb_datasets


# Registry of available dataset creators
DATASET_CREATORS: Dict[str, DatasetCreator] = {
    "mnist": create_mnist_datasets,
    "alsomitra": create_alsomitra_datasets,
    "gtsrb": create_gtsrb_datasets,
}


def create_dataset(
    dataset_name: str, batch_size: int
) -> Tuple[
    DataLoader, DataLoader, torch.nn.Module, Tuple[Tuple[float, ...], Tuple[float, ...]]
]:
    """
    Create dataset loaders based on dataset name.

    Args:
        dataset_name: Name of the dataset ('mnist', 'alsomitra', 'gtsrb')
        batch_size: Size of training batches

    Returns:
        Tuple of (train_loader, test_loader, model, (mean, std))

    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name not in DATASET_CREATORS:
        available = ", ".join(DATASET_CREATORS.keys())
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Available datasets: {available}"
        )

    creator = DATASET_CREATORS[dataset_name]
    result = creator(batch_size)

    # Validate the result format for consistency
    validate_dataset_result(result)

    return result


def list_available_datasets() -> list[str]:
    """
    Get a list of available dataset names.

    Returns:
        List of available dataset names
    """
    return list(DATASET_CREATORS.keys())


def register_dataset(name: str, creator: DatasetCreator) -> None:
    """
    Register a new dataset creator.

    Args:
        name: Name of the dataset
        creator: Function that creates the dataset

    Raises:
        ValueError: If name is already registered
    """
    if name in DATASET_CREATORS:
        raise ValueError(f"Dataset '{name}' is already registered")

    DATASET_CREATORS[name] = creator


# Export the main interface functions
__all__ = [
    "create_dataset",
    "list_available_datasets",
    "register_dataset",
    "DATASET_CREATORS",
    "DatasetCreator",
    "SizedDataset",
    "create_mnist_datasets",
    "create_alsomitra_datasets",
    "create_gtsrb_datasets",
]
