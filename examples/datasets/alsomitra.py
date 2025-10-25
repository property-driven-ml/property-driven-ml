"""
Alsomitra dataset creation utilities.

This module provides functions to create and load Alsomitra datasets
for dynamical system modeling tasks.

Alsomitra reference: https://arxiv.org/abs/2505.00622
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Tuple

from property_driven_ml.training.mode import Mode
from examples.models import AlsomitraNet


class AlsomitraDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_all: pd.DataFrame,
        indices: np.ndarray,
        in_stats: Tuple[
            pd.Series, pd.Series
        ],  # (centre, scale) to min-max normalise inputs to [0, 1]
        out_stats: Tuple[
            float, float
        ],  # (y_min, y_max) to min-max normalise outputs to [0, 1]
    ):
        data = data_all.iloc[indices].reset_index(drop=True).copy()

        centre, scale = in_stats
        data.iloc[:, :-2] = (data.iloc[:, :-2] - centre) / scale

        y_min, y_max = out_stats
        data.iloc[:, -1] = (data.iloc[:, -1] - y_min) / (y_max - y_min)

        self.X = data.iloc[:, :-2].to_numpy(dtype=np.float32, copy=False)
        self.y = data.iloc[:, -1].to_numpy(dtype=np.float32, copy=False)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # y as [1] to match (N,1) predictors
        return torch.from_numpy(self.X[idx]), torch.tensor(
            [self.y[idx]], dtype=torch.float32
        )


def create_alsomitra_datasets(
    batch_size: int,
    train_split: float = 0.8,
    seed: int = 42,
) -> tuple[
    DataLoader,
    DataLoader,
    torch.nn.Module,
    tuple[tuple[float, ...], tuple[float, ...]],
    Mode,
]:
    """
    Create Alsomitra train and test data loaders.

    Args:
        batch_size: Size of training batches

    Returns:
        Tuple of (train_loader, test_loader, model, (mean, std))
    """
    csv_path = "../data/alsomitra/alsomitra_data_680.csv"
    data_all = pd.read_csv(csv_path, header=None)

    perm = np.random.RandomState(seed).permutation(len(data_all))
    split_idx = int(train_split * len(data_all))
    train_idx, test_idx = perm[:split_idx], perm[split_idx:]

    # compute min/max on train inputs only
    train_inputs = data_all.iloc[train_idx, :-2]
    min = train_inputs.min(axis=0)
    max = train_inputs.max(axis=0)
    scale = max - min

    # compute min/max on train outputs only
    train_outputs = data_all.iloc[train_idx, -1]
    y_min = train_outputs.min()
    y_max = train_outputs.max()

    dataset_train = AlsomitraDataset(
        data_all, train_idx, in_stats=(min, scale), out_stats=(y_min, y_max)
    )
    dataset_test = AlsomitraDataset(
        data_all, test_idx, in_stats=(min, scale), out_stats=(y_min, y_max)
    )

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = AlsomitraNet()
    mean, std = (0.0,), (1.0,)  # no normalisation needed for Alsomitra

    return train_loader, test_loader, model, (mean, std), Mode.Regression
