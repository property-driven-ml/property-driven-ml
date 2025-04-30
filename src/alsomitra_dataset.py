import torch

import pandas as pd

from typing import Callable

from bounded_datasets import BoundedDataset

class AlsomitraDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file, header=None)

        self.inputs, AlsomitraDataset.C_in, AlsomitraDataset.S_in = self.normalise_dataset(torch.tensor(data.iloc[:, :-2].values, dtype=torch.float32))
        self.outputs, AlsomitraDataset.C_out, AlsomitraDataset.S_out = self.normalise_dataset(torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1))

    # min-max normalise to [0, 1]
    def normalise_dataset(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centre = x.min(dim=0).values
        scale = x.max(dim=0).values - centre

        return (x - centre) / scale, centre, scale

    def normalise_input(x: torch.Tensor) -> torch.Tensor:
        return (x - AlsomitraDataset.C_in) / AlsomitraDataset.S_in

    def denormalise_input(x: torch.Tensor) -> torch.Tensor:
        return x * AlsomitraDataset.S_in + AlsomitraDataset.C_in
    
    def normalise_output(x: torch.Tensor) -> torch.Tensor:
        return (x - AlsomitraDataset.C_out) / AlsomitraDataset.S_out

    def denormalise_output(x: torch.Tensor) -> torch.Tensor:
        return x * AlsomitraDataset.S_out + AlsomitraDataset.C_out

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class AlsomitraInputRegion(BoundedDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, bounds_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]], mean: torch.Tensor | tuple[float, ...] = (0.,), std: torch.Tensor | tuple[float, ...] = (1.,)):

        # bounds_fn gets a denormalised input
        super().__init__(dataset, lambda x: self.combine_bounds(bounds_fn(self.denormalise(x))), mean, std)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return AlsomitraDataset.normalise_input(x)

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        return AlsomitraDataset.denormalise_input(x)

    def combine_bounds(self, bounds: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]: 
        return self.normalise(torch.maximum(bounds[0], self.denormalise(self.min))), self.normalise(torch.minimum(bounds[1], self.denormalise(self.max)))