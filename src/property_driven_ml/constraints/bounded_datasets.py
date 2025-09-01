import torch

from typing import Callable

from .base import SizedDataset, NormalizedDataset


# Import the protocol after the module structure is established


class BoundedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that enforces input bounds for constraint checking.

    Wraps an existing dataset and provides bounded versions of inputs
    for training with input constraints and adversarial perturbations.
    """

    def __init__(
        self,
        dataset: SizedDataset,
        bounds_or_bounds_fn: tuple[torch.Tensor, torch.Tensor]
        | Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        mean: torch.Tensor | tuple[float, ...] = (0.0,),
        std: torch.Tensor | tuple[float, ...] = (1.0,),
    ):
        """
        Initialize the BoundedDataset.

        Args:
            dataset: Base dataset to wrap.
            bounds_or_bounds_fn: Either static bounds tuple or function computing bounds.
            mean: Normalization mean values.
            std: Normalization standard deviation values.

        Raises:
            ValueError: If bounds_or_bounds_fn is not a tuple or Callable.
        """
        self.dataset = dataset

        if isinstance(bounds_or_bounds_fn, tuple):
            self.lo, self.hi = bounds_or_bounds_fn
            self.has_static_bounds = True
        elif isinstance(bounds_or_bounds_fn, Callable):
            self.bounds_fn = bounds_or_bounds_fn
            self.has_static_bounds = False
        else:
            raise ValueError("bounds_or_bounds_fn must be a tuple or a Callable")

        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

        self.min = (0.0 - self.mean) / self.std
        self.max = (1.0 - self.mean) / self.std

    def __getitem__(self, idx):
        """
        Get a bounded item from the dataset. Intended to be used during
        training by iterating over the dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            A tuple (x, y, lo, hi) where x is the input, y is the target,
            and lo, hi are the bounds.

        Raises:
            ValueError: If the bounds are invalid or have an incompatible shape.
        """
        x, y = self.dataset[idx]

        if self.has_static_bounds:
            lo, hi = self.lo, self.hi
        else:
            lo, hi = self.bounds_fn(x)

        # Need explicit error handling due to user provided bounds or bound functions
        if not torch.all(lo <= hi):
            raise ValueError(f"Invalid bounds: lo={lo}, hi={hi}")
        elif not x.shape == lo.shape == hi.shape:
            raise ValueError(
                f"Shape mismatch: x.shape={x.shape}, lo.shape={lo.shape}, hi.shape={hi.shape}"
            )
        return x, y, lo, hi

    def __len__(self):
        return len(self.dataset)


# standard epsilon ball (cube, \ell_{infty}) relative to each data point
class EpsilonBall(BoundedDataset):
    """
    Dataset with epsilon-ball bounds around each input.

    Creates bounded regions by adding/subtracting epsilon from each
    input, useful for adversarial robustness training.
    """

    def __init__(
        self,
        dataset: SizedDataset,
        eps: float | torch.Tensor,
        mean: torch.Tensor | tuple[float, ...] = (0.0,),
        std: torch.Tensor | tuple[float, ...] = (1.0,),
    ):
        """
        Initialize the EpsilonBall dataset.

        Args:
            dataset: Base dataset to wrap.
            eps: Epsilon value for ball radius.
            mean: Normalization mean values.
            std: Normalization standard deviation values.
        """
        x, _ = dataset[0]
        eps = torch.as_tensor(eps) / torch.as_tensor(
            std
        )  # TODO: check for MNIST + Alsomitra + GTSRB
        eps = eps.view(*eps.shape, *([1] * (x.ndim - eps.ndim)))

        # NOTE: this will lead to issues on Windows as lambda cannot be pickled (so use num_workers=0 on Windows for the dataloader)
        super().__init__(dataset, lambda x: (x - eps, x + eps), mean, std)


# the same absolute bounds for all data
class GlobalBounds(BoundedDataset):
    """Dataset with global bounds applied to all inputs.

    Uses the same absolute lower and upper bounds for all data points,
    useful when all inputs should be constrained to the same region.
    """

    def __init__(
        self,
        dataset: SizedDataset,
        lo: torch.Tensor,
        hi: torch.Tensor,
        mean: tuple[float, ...] = (0.0,),
        std: tuple[float, ...] = (1.0,),
        normalize: bool = True,
    ):
        """
        Initialize the GlobalBounds dataset.

        Args:
            dataset: Base dataset to wrap.
            lo: Lower bound tensor (same shape as data).
            hi: Upper bound tensor (same shape as data).
            mean: Normalization mean values.
            std: Normalization standard deviation values.
        """
        x, _ = dataset[0]
        assert x.shape == lo.shape == hi.shape, (  # nosec
            f"unsupported bounds shape: lo.shape={lo.shape}, hi.shape={hi.shape} must match data shape {x.shape}"
        )

        def normalize_bounds(x: torch.Tensor) -> torch.Tensor:
            return (x - torch.tensor(mean)) / torch.tensor(std)

        if normalize:
            lo, hi = normalize_bounds(lo), normalize_bounds(hi)

        print(f"lo.shape={lo.shape}, hi.shape={hi.shape}")

        super().__init__(dataset, (lo, hi), mean, std)


class AlsomitraInputRegion(BoundedDataset):
    """Specialized bounded dataset for Alsomitra aerodynamics data.

    Provides input bounds specific to the Alsomitra dataset using
    a bounds function that considers the domain constraints.

    Look at examples/ in GitHub for details.

    Args:
        dataset: Alsomitra dataset instance.
        bounds_fn: Function that computes bounds given input tensor.
        mean: Normalization mean values.
        std: Normalization standard deviation values.
    """

    def __init__(
        self,
        dataset: NormalizedDataset,
        bounds_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        mean: torch.Tensor | tuple[float, ...] = (0.0,),
        std: torch.Tensor | tuple[float, ...] = (1.0,),
    ):
        self.dataset = dataset
        # bounds_fn gets a denormalised input
        super().__init__(
            dataset,
            lambda x: self.combine_bounds(bounds_fn(self.denormalise(x))),
            mean,
            std,
        )

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return self.dataset.normalise_input(x)

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        return self.dataset.denormalise_input(x)

    def combine_bounds(
        self, bounds: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.normalise(
            torch.maximum(bounds[0], self.denormalise(self.min))
        ), self.normalise(torch.minimum(bounds[1], self.denormalise(self.max)))
