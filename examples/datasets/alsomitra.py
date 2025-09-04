"""
Alsomitra dataset creation utilities.

This module provides functions to create and load Alsomitra datasets
for dynamical system modeling tasks.

Alsomitra reference: https://arxiv.org/abs/2505.00622
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple

from examples.models import AlsomitraNet

import pandas as pd


class AlsomitraDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file, header=None)

        self.inputs, AlsomitraDataset.C_in, AlsomitraDataset.S_in = (
            self.normalise_dataset(
                torch.tensor(data.iloc[:, :-2].values, dtype=torch.float32)
            )
        )
        self.outputs, AlsomitraDataset.C_out, AlsomitraDataset.S_out = (
            self.normalise_dataset(
                torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)
            )
        )

    # min-max normalise to [0, 1]
    def normalise_dataset(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centre = x.min(dim=0).values
        scale = x.max(dim=0).values - centre

        return (x - centre) / scale, centre, scale

    def normalise_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - AlsomitraDataset.C_in) / AlsomitraDataset.S_in

    def denormalise_input(self, x: torch.Tensor) -> torch.Tensor:
        return x * AlsomitraDataset.S_in + AlsomitraDataset.C_in

    def normalise_output(self, x: torch.Tensor) -> torch.Tensor:
        return (x - AlsomitraDataset.C_out) / AlsomitraDataset.S_out

    def denormalise_output(self, x: torch.Tensor) -> torch.Tensor:
        return x * AlsomitraDataset.S_out + AlsomitraDataset.C_out

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def create_alsomitra_datasets(
    batch_size: int,
) -> Tuple[
    DataLoader, DataLoader, torch.nn.Module, Tuple[Tuple[float, ...], Tuple[float, ...]]
]:
    """
    Create Alsomitra train and test data loaders.

    Args:
        batch_size: Size of training batches

    Returns:
        Tuple of (train_loader, test_loader, model, (mean, std))
    """
    dataset = AlsomitraDataset("examples/alsomitra_data_680.csv")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    dataset_train, dataset_test = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = AlsomitraNet()
    mean, std = (0.0,), (1.0,)  # No normalization needed for Alsomitra

    return train_loader, test_loader, model, (mean, std)
