import torch

import pandas as pd

from typing import Protocol


class AlsomitraDatasetLike(Protocol):
    """Protocol for datasets that support Alsomitra-style normalization."""

    def __getitem__(self, index: int) -> tuple: ...
    def __len__(self) -> int: ...
    def normalise_input(self, x: torch.Tensor) -> torch.Tensor: ...
    def denormalise_input(self, x: torch.Tensor) -> torch.Tensor: ...


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
