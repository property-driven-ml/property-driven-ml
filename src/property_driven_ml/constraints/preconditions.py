import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Tuple


class Precondition(ABC):
    """
    Abstract base class for preconditions/ input postconditions.
    """

    @abstractmethod
    def get_bounds(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the input constraint function for this property.

        Returns:
            lo, hi: Tensors defining lower and upper bounds for adversarial example search space.
        """
        pass


class EpsilonBall(Precondition):
    """
    Precondition defining an epsilon ball around the input.
    """

    def __init__(
        self,
        device: torch.device,
        epsilon: float,
        std: Tuple[float, ...] | float | None = None,
    ):
        """
        Initialize the epsilon ball precondition.
        Assumes input is normalized with mean 0 and standard deviation 1.
        If not, provide the std of the data, which will be used for projecting epsilon to problem space.

        Args:
            device: PyTorch device for tensor computations.
            epsilon: Radius of the epsilon ball (assuming normalized data as described above).
            std (optional): Standard deviation for input normalization. Can be a float or a tuple of floats for each dimension.
        """
        self.device = device
        self.epsilon = epsilon
        self.std = torch.as_tensor(std, device=self.device) if std is not None else None

    def get_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the precondition function for the epsilon ball.

        Args:
            x: Original input tensor.

        Returns:
            lo, hi: Tensors defining lower and upper bounds for epsilon ball (in this case epsilon cube).
        """
        x = x.to(self.device)
        epsilon = self.epsilon * torch.ones_like(x, device=self.device)

        if self.std is not None:
            std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)

            if std.numel() == 1:
                scale = std  # scalar
            elif std.ndim == 1 and std.shape[0] == x.shape[1]:
                scale = std.view(1, -1, 1, 1)  # [C] -> [1, C, 1, 1]
            elif std.shape == x.shape:
                scale = std
            else:
                raise ValueError(
                    "std must be either a scalar or have the same shape as data."
                )

            epsilon = epsilon / scale

        lo = x - epsilon
        hi = x + epsilon

        return lo, hi


class GlobalBounds(Precondition):
    """
    Precondition defining global bounds for the input.
    """

    def __init__(
        self,
        device: torch.device,
        lower_bound: float | torch.Tensor | Tuple[float, ...] = 0.0,
        upper_bound: float | torch.Tensor | Tuple[float, ...] = 1.0,
        mean: Tuple[float, ...] | float | torch.Tensor | None = None,
        std: Tuple[float, ...] | float | torch.Tensor | None = None,
    ):
        """
        Initialize the global bounds precondition.

        Args:
            device: PyTorch device for tensor computations.
            lower_bound: Minimum value(s) for input dimensions. Can be:
                - A scalar float (applied to all dimensions)
                - A tensor with shape matching the input dimensions
                - A tuple of floats (one per dimension)
            upper_bound: Maximum value(s) for input dimensions. Can be:
                - A scalar float (applied to all dimensions)
                - A tensor with shape matching the input dimensions
                - A tuple of floats (one per dimension)
            mean (optional): Mean for input normalization. Can be a float, tuple, or tensor.
            std (optional): Standard deviation for input normalization. Can be a float, tuple, or tensor.
        """
        self.device = device

        # Convert bounds to tensors if they're not already
        self.lower_bound = torch.as_tensor(lower_bound, device=self.device)
        self.upper_bound = torch.as_tensor(upper_bound, device=self.device)

        # Ensure bounds have compatible shapes
        if self.lower_bound.shape != self.upper_bound.shape:
            # Try to broadcast them to the same shape
            try:
                self.lower_bound, self.upper_bound = torch.broadcast_tensors(
                    self.lower_bound, self.upper_bound
                )
            except RuntimeError as e:
                raise ValueError(
                    f"lower_bound and upper_bound shapes are not compatible: {e}"
                )

        # Validate bounds
        if torch.any(self.lower_bound >= self.upper_bound):
            raise ValueError(
                "lower_bound must be strictly less than upper_bound for all dimensions"
            )

        self.mean = (
            torch.as_tensor(mean, device=self.device) if mean is not None else None
        )
        self.std = torch.as_tensor(std, device=self.device) if std is not None else None
        if (self.mean is None) != (self.std is None):
            raise ValueError("mean and std must be provided together or not at all")
        if self.std is not None and torch.any(self.std == 0):
            raise ValueError("std must not contain zero values")

    def get_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the precondition function for the global bounds.

        Args:
            x: Original input tensor.

        Returns:
            lo, hi: Tensors defining lower and upper bounds for the input.
        """
        x = x.to(self.device)

        if self.mean is not None and self.std is not None:
            # Apply normalization: (bound - mean) / std
            lo = (self.lower_bound - self.mean) / self.std
            hi = (self.upper_bound - self.mean) / self.std
        else:
            # No normalization - use bounds directly
            lo = self.lower_bound
            hi = self.upper_bound

        # Expand bounds to match batch size if needed
        if lo.ndim == 0:  # scalar
            lo = lo.expand_as(x)
            hi = hi.expand_as(x)
        elif lo.ndim < x.ndim:  # Need to add batch dimension
            # Expand to match input tensor shape (add batch dimension)
            batch_size = x.shape[0]
            lo = lo.unsqueeze(0).expand(batch_size, *lo.shape)
            hi = hi.unsqueeze(0).expand(batch_size, *hi.shape)
        elif lo.shape != x.shape:
            # Try to broadcast to input shape
            try:
                lo = torch.broadcast_to(lo, x.shape)
                hi = torch.broadcast_to(hi, x.shape)
            except RuntimeError as e:
                raise ValueError(
                    f"Cannot broadcast bounds shape {lo.shape} to input shape {x.shape}: {e}"
                )

        return lo, hi


class AlsomitraBase(Precondition):
    """
    Base class for Alsomitra aerodynamics data preconditions.
    """

    def __init__(self, min: torch.Tensor, max: torch.Tensor) -> None:
        """
        Initialize the Alsomitra input region precondition.
        Args:
            min: Tensor defining minimum bounds for input dimensions.
            max: Tensor defining maximum bounds for input dimensions.

        Easiest to get by tensor.min(dim=0).values and tensor.max(dim=0).values on the training data.
        """
        self.min = min
        self.max = max
        # Define indices for each feature for clarity
        self.v_x = 0
        self.v_y = 1
        self.omega = 2
        self.theta = 3
        self.x = 4
        self.y = 5

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor x based on the min and max bounds.

        Args:
            x: Input tensor to normalize.

        Returns:
            Normalized tensor.
        """
        return (x - self.min) / (self.max - self.min)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the input tensor x based on the min and max bounds.

        Args:
            x: Input tensor to denormalize.

        Returns:
            Denormalized tensor.
        """
        return x * (self.max - self.min) + self.min


class AlsomitraProperty1(AlsomitraBase):
    """
    Specialized precondition for Alsomitra aerodynamics data.

    Intended for use in fir = x_problem[-v]st Alsomitra constraint.
    """

    def __init__(self, threshold: float, min: torch.Tensor, max: torch.Tensor) -> None:
        super().__init__(min, max)
        self.threshold = threshold

    def get_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the precondition function for the Alsomitra input region.

        Returns:
            lo, hi: Tensors defining lower and upper bounds for the Alsomitra input region.
        """
        x_problem = self.denormalize(x)

        lo = x_problem.clone()
        lo[self.y] = self.threshold - x_problem[self.x]

        hi = x_problem.clone()

        lo = self.normalize(lo)
        hi = self.normalize(hi)
        hi[self.y] = np.nan  # No upper bound for y-coordinate

        return lo, hi


class AlsomitraProperty2(AlsomitraBase):
    """
    Specialized precondition for Alsomitra aerodynamics data.

    Intended for use in second Alsomitra constraint.
    """

    def __init__(
        self,
        y_threshold: float,
        theta_thresholds: Tuple[float, float],
        min: torch.Tensor,
        max: torch.Tensor,
    ) -> None:
        """
        Initialize the Alsomitra input region precondition.

        Args:
            y_threshold: Threshold for the y-coordinate relative to x.
            theta_thresholds: Tuple defining (min, max) bounds for the theta angle.
            min: Tensor defining minimum bounds for input dimensions.
            max: Tensor defining maximum bounds for input dimensions.
        """
        super().__init__(min, max)
        self.y_threshold = y_threshold
        self.theta_thresholds = theta_thresholds

    def get_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the precondition function for the Alsomitra input region.

        Returns:
            lo, hi: Tensors defining lower and upper bounds for the Alsomitra input region.
        """

        x_problem = self.denormalize(x)

        lo = x_problem.clone()
        hi = x_problem.clone()

        lo[self.y] = -self.y_threshold - x_problem[self.x]
        hi[self.y] = self.y_threshold - x_problem[self.x]
        lo[self.theta] = self.theta_thresholds[0]
        hi[self.theta] = self.theta_thresholds[1]

        return lo, hi


class AlsomitraProperty3(AlsomitraBase):
    """
    Specialized precondition for Alsomitra aerodynamics data.

    Intended for use in third Alsomitra constraint.
    """

    def __init__(
        self,
        v_y_threshold: float,
        y_threshold: float,
        omega_threshold: float,
        min: torch.Tensor,
        max: torch.Tensor,
    ) -> None:
        """
        Initialize the Alsomitra input region precondition.

        Args:
            v_y_threshold: Threshold for the y-coordinate relative to x.
            theta_threshold: Threshold for the theta angle.
            min: Tensor defining minimum bounds for input dimensions.
            max: Tensor defining maximum bounds for input dimensions.
        """
        super().__init__(min, max)
        self.v_y_threshold = v_y_threshold
        self.y_threshold = y_threshold
        self.omega_threshold = omega_threshold

    def get_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the precondition function for the Alsomitra input region.

        Returns:
            lo, hi: Tensors defining lower and upper bounds for the Alsomitra input region.
        """

        x_problem = self.denormalize(x)

        lo = x_problem.clone()
        hi = x_problem.clone()

        lo[self.y] = -x_problem[self.x]
        hi[self.y] = self.y_threshold - x_problem[self.x]
        lo[self.v_y] = np.nan  # No lower bound for v_y
        hi[self.v_y] = self.v_y_threshold
        lo[self.omega] = np.nan  # No lower bound for omega
        hi[self.omega] = self.omega_threshold

        lo = self.normalize(lo)
        hi = self.normalize(hi)

        return lo, hi


class AlsomitraProperty4(AlsomitraBase):
    """
    Specialized precondition for Alsomitra aerodynamics data.

    Intended for use in fourth Alsomitra constraint.
    """

    def __init__(
        self, y_threshold: float, min: torch.Tensor, max: torch.Tensor
    ) -> None:
        """
        Initialize the Alsomitra input region precondition.

        Args:
            y_threshold: Threshold for the y-coordinate relative to x.
            min: Tensor defining minimum bounds for input dimensions.
            max: Tensor defining maximum bounds for input dimensions.
        """
        super().__init__(min, max)
        self.y_threshold = y_threshold

    def get_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the precondition function for the Alsomitra input region.

        Returns:
            lo, hi: Tensors defining lower and upper bounds for the Alsomitra input region.
        """

        x_problem = self.denormalize(x)

        lo = x_problem.clone()
        hi = x_problem.clone()

        lo[self.y] = -self.y_threshold - x_problem[self.x]
        hi[self.y] = self.y_threshold - x_problem[self.x]

        return lo, hi
