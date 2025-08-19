"""
Alsomitra dataset creation utilities.

This module provides functions to create and load Alsomitra datasets
for dynamical system modeling tasks.
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple

from examples.alsomitra_dataset import AlsomitraDataset
from examples.models import AlsomitraNet


def create_alsomitra_datasets(batch_size: int) -> Tuple[DataLoader, DataLoader, torch.nn.Module, Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    """
    Create Alsomitra train and test data loaders.
    
    Args:
        batch_size: Size of training batches
        
    Returns:
        Tuple of (train_loader, test_loader, model, (mean, std))
    """
    dataset = AlsomitraDataset('examples/alsomitra_data_680.csv')
    train_size = int(.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    dataset_train, dataset_test = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    model = AlsomitraNet()
    mean, std = (0.,), (1.,)  # No normalization needed for Alsomitra
    
    return train_loader, test_loader, model, (mean, std)
