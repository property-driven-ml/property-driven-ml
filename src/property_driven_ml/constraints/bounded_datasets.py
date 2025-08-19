import torch

from typing import Callable

class BoundedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, bounds_or_bounds_fn: tuple[torch.Tensor, torch.Tensor] | Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]], mean: torch.Tensor | tuple[float, ...] = (0.,), std: torch.Tensor | tuple[float, ...] = (1.,)):
        self.dataset = dataset
        
        if isinstance(bounds_or_bounds_fn, tuple):
            self.lo, self.hi = bounds_or_bounds_fn
            self.has_static_bounds = True
        elif isinstance(bounds_or_bounds_fn, Callable):
            self.bounds_fn = bounds_or_bounds_fn
            self.has_static_bounds = False
        else:
            raise ValueError('bounds_or_bounds_fn must be a tuple or a Callable')
        
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

        self.min = (0. - self.mean) / self.std
        self.max = (1. - self.mean) / self.std

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        if self.has_static_bounds:
            lo, hi = self.lo, self.hi
        else:
            lo, hi = self.bounds_fn(x)

        assert x.shape == lo.shape == hi.shape, f'x.shape={x.shape}, lo.shape={lo.shape}, hi.shape={hi.shape}'
        return x, y, lo, hi
    
    def __len__(self):
        return len(self.dataset)

# standard epsilon ball (cube, \ell_{infty}) relative to each data point
class EpsilonBall(BoundedDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, eps: float | torch.Tensor, mean: torch.Tensor | tuple[float, ...] = (0.,), std: torch.Tensor | tuple[float, ...] = (1.,)):
        x, _ = dataset[0]
        eps = torch.as_tensor(eps) / torch.as_tensor(std) # TODO: check for MNIST + Alsomitra + GTSRB
        eps = eps.view(*eps.shape, *([1] * (x.ndim - eps.ndim)))

        # NOTE: this will lead to issues on Windows as lambda cannot be pickled (so use num_workers=0 on Windows for the dataloader)
        super().__init__(dataset, lambda x: (x - eps, x + eps), mean, std)

# the same absolute bounds for all data
class GlobalBounds(BoundedDataset):  
    def __init__(self, dataset: torch.utils.data.Dataset, lo: torch.Tensor, hi: torch.Tensor, mean: tuple[float, ...] = (0.,), std: tuple[float, ...] = (1.,), normalise=True):
        x, _ = dataset[0]
        assert x.shape == lo.shape == hi.shape, f'unsupported bounds shape: lo.shape={lo.shape}, hi.shape={hi.shape} must match data shape {x.shape}'

        def normalise(x: torch.Tensor) -> torch.Tensor:
            return (x - torch.tensor(mean)) / torch.tensor(std)

        if normalise:
            lo, hi = normalise(lo), normalise(hi)

        print(f'lo.shape={lo.shape}, hi.shape={hi.shape}')

        super().__init__(dataset, (lo, hi), mean, std)