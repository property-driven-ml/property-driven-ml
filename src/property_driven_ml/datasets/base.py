"""
Base dataset interface and utilities.

This module provides the base interface and common utilities for
dataset creation and loading.
"""

import torch
from torch.utils.data import DataLoader
from typing import Tuple, Protocol


class DatasetCreator(Protocol):
    """Protocol for dataset creation functions."""
    
    def __call__(self, batch_size: int) -> Tuple[DataLoader, DataLoader, torch.nn.Module, Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """
        Create dataset loaders and associated model.
        
        Args:
            batch_size: Size of training batches
            
        Returns:
            Tuple of (train_loader, test_loader, model, (mean, std))
        """
        ...


def validate_dataset_result(result: Tuple[DataLoader, DataLoader, torch.nn.Module, Tuple[Tuple[float, ...], Tuple[float, ...]]]) -> None:
    """
    Validate that a dataset creation function returns the expected format.
    
    Args:
        result: The result tuple from a dataset creation function
        
    Raises:
        ValueError: If the result format is invalid
    """
    if not isinstance(result, tuple) or len(result) != 4:
        raise ValueError("Dataset creator must return tuple of (train_loader, test_loader, model, (mean, std))")
    
    train_loader, test_loader, model, (mean, std) = result
    
    if not isinstance(train_loader, DataLoader):
        raise ValueError("First element must be a DataLoader for training")
    
    if not isinstance(test_loader, DataLoader):
        raise ValueError("Second element must be a DataLoader for testing")
    
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Third element must be a PyTorch model")
    
    if not isinstance(mean, tuple) or not isinstance(std, tuple):
        raise ValueError("Fourth element must be tuple of (mean, std) tuples")
