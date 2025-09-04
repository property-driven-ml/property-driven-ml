import torch

from abc import ABC, abstractmethod
from typing import Tuple


class Precondition(ABC):
    """
    Abstract base class for preconditions/ input postconditions.
    """

    @abstractmethod
    def get_precondition(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def get_precondition(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the precondition function for the epsilon ball.

        Args:
            x: Original input tensor.

        Returns:
            lo, hi: Tensors defining lower and upper bounds for epsilon ball (in this case epsilon cube).
        """
        x = x.to(self.device)
        epsilon = self.epsilon * torch.ones_like(x, device=self.device)

        if self.std is not None:
            if not (self.std.shape == x.shape or self.std.numel() == 1):
                raise ValueError(
                    "std must be either a scalar or have the same shape as data"
                )
            epsilon = epsilon * self.std

        lo = x - epsilon
        hi = x + epsilon

        return lo, hi
