import torch
import torch.nn.functional as F
import torch.linalg as LA

from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..logics.logic import Logic


class Postcondition(ABC):
    """
    Abstract base class for postconditions/ output properties.
    """

    @abstractmethod
    def get_postcondition(self, *args, **kwargs) -> Callable[[Logic], torch.Tensor]:
        """
        Get the postcondition function for this property.

        This method should be implemented by subclasses with their specific signature.
        Common parameters include:
            N: Neural network model.
            x: Original input tensor.
            x_adv: Adversarial input tensor.
            y_target: Target output tensor.
            device: Optional PyTorch device for tensor computations.

        Additional parameters may be specific to the postcondition implementation.

        Args:
            *args: Positional arguments specific to the postcondition.
            **kwargs: Keyword arguments specific to the postcondition.

        Returns:
            Function that takes a Logic instance and returns postcondition tensor.
        """
        pass


class StandardRobustnessPostcondition(Postcondition):
    """
    Postcondition ensuring model robustness to adversarial perturbations.

    Enforces that the change in output probabilities between original and
    adversarial inputs remains within a specified threshold delta.

    Args:
        device: PyTorch device for tensor computations.
        delta: Maximum allowed change in output probabilities.
    """

    def __init__(self, device: torch.device, delta: float | torch.Tensor):
        self.device = device
        assert 0.0 <= delta <= 1.0, (  # nosec
            "delta is a probability and should be within the range [0, 1]"
        )
        self.delta = torch.as_tensor(delta, device=self.device)

    def get_postcondition(
        self,
        N: torch.nn.Module,
        x: torch.Tensor,
        x_adv: torch.Tensor,
    ) -> Callable[[Logic], torch.Tensor]:
        """Get robustness postcondition for probability difference bounds.

        Args:
            N: Neural network model.
            x: Original input tensor.
            x_adv: Adversarial input tensor.

        Returns:
            Function that constrains infinity norm of probability differences.
        """
        y = N(x)
        y_adv = N(x_adv)

        diff = F.softmax(y_adv, dim=1) - F.softmax(y, dim=1)

        return lambda logic: logic.LEQ(
            LA.vector_norm(diff, ord=float("inf"), dim=1), self.delta
        )


class LipschitzRobustnessPostcondition(Postcondition):
    """
    Postcondition enforcing Lipschitz continuity for model robustness.

    Ensures that the rate of change in model outputs is bounded by the
    Lipschitz constant L relative to input perturbations.

    Args:
        device: PyTorch device for tensor computations.
        L: Lipschitz constant bounding the rate of output change.
    """

    def __init__(self, device: torch.device, L: float):
        self.device = device
        self.L = torch.as_tensor(L, device=device)

    def get_postcondition(
        self, N: torch.nn.Module, x: torch.Tensor, x_adv: torch.Tensor
    ) -> Callable[[Logic], torch.Tensor]:
        """Get Lipschitz postcondition relating input and output changes.

        Args:
            N: Neural network model.
            x: Original input tensor.
            x_adv: Perturbed input tensor.

        Returns:
            Function that constrains output change by L times input change.
        """
        y = N(x)
        y_adv = N(x_adv)

        diff_x = LA.vector_norm(x_adv - x, ord=2, dim=1)
        diff_y = LA.vector_norm(y_adv - y, ord=2, dim=1)

        return lambda logic: logic.LEQ(diff_y, self.L * diff_x)


class OppositeFacesPostcondition(Postcondition):
    """
    Postcondition ensuring a physical-world inspired constraint on dice images.

    Enforces that the network may not predict faces at the same time that are
    on opposite sides of the die (e.g. faces 1 and 6).

    Args:
        device: PyTorch device for tensor computations.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.zero = torch.tensor(0.0, device=self.device)
        self.opposingFacePairs = [(0, 5), (1, 4), (2, 3)]

    def get_postcondition(
        self,
        N: torch.nn.Module,
        x_adv: torch.Tensor,
    ) -> Callable[[Logic], torch.Tensor]:
        """Get postcondition for opposite faces.

        Args:
            N: Neural network model.
            x_adv: Adversarial input tensor.

        Returns:
            Function that ensures network predictions align with real-world knowledge.
        """
        y_adv = N(x_adv)

        # Note: label i is predicted if y_adv[i] > 0; equivalently, label i is not predicted if y_adv[i] <= 0.
        return lambda logic: logic.AND(
            *[
                logic.OR(
                    logic.AND(
                        logic.GT(y_adv[:, i], self.zero),  # predicts label i ...
                        logic.LEQ(y_adv[:, j], self.zero),  # ... but not label j
                    ),
                    logic.AND(
                        logic.GT(y_adv[:, j], self.zero),  # predicts label j ...
                        logic.LEQ(y_adv[:, i], self.zero),  # ... but not label i
                    ),
                )
                for i, j in self.opposingFacePairs
            ]
        )


class AlsomitraOutputPostcondition(Postcondition):
    """
    Postcondition ensuring model outputs fall within specified bounds.

    Enforces that neural network outputs remain within lower and upper bounds,
    with optional normalization to handle different output scales.

    Args:
        device: PyTorch device for tensor computations.
        lo: Lower bound for outputs (None means no lower bound).
        hi: Upper bound for outputs (None means no upper bound).
        normalize: Whether to normalize bounds to output statistics.
    """

    def __init__(
        self,
        device: torch.device,
        lo: Optional[float | torch.Tensor] = None,
        hi: Optional[float | torch.Tensor] = None,
        normalize: bool = True,
    ):
        self.device = device
        # Store raw bounds and normalization flag
        self.lo_raw = lo
        self.hi_raw = hi
        self.normalize = normalize

        # If normalization is disabled (for backwards compatibility), store as tensors directly
        if not normalize:
            self.lo = torch.as_tensor(lo, device=device) if lo is not None else None
            self.hi = torch.as_tensor(hi, device=device) if hi is not None else None

    def get_postcondition(
        self,
        N: torch.nn.Module,
        x_adv: torch.Tensor,
        scale: torch.Tensor | None = None,
        centre: torch.Tensor | None = None,
    ) -> Callable[[Logic], torch.Tensor]:
        """
        Get output bounds postcondition for adversarial inputs.

        This implementation uses a specialized signature that only includes
        the parameters actually needed by this postcondition type.

        Args:
            N: Neural network model.
            x_adv: Adversarial input tensor.
            scale: Optional scaling factor for normalization.
            centre: Optional centre point for normalization.

        Returns:
            Function that constrains outputs to specified bounds.
        """
        y_adv = N(x_adv).squeeze()

        if self.normalize:
            # Normalize bounds at postcondition time using class constants
            lo_normalized = (
                None
                if self.lo_raw is None
                else (torch.tensor(self.lo_raw, device=self.device) - centre) / scale
            )
            hi_normalized = (
                None
                if self.hi_raw is None
                else (torch.tensor(self.hi_raw, device=self.device) - centre) / scale
            )

            lo_normalized = (
                lo_normalized.squeeze() if lo_normalized is not None else None
            )
            hi_normalized = (
                hi_normalized.squeeze() if hi_normalized is not None else None
            )
        else:
            # Use pre-normalized bounds
            lo_normalized = self.lo
            hi_normalized = self.hi

        if lo_normalized is None and hi_normalized is not None:
            return lambda logic: logic.LEQ(y_adv, hi_normalized)
        elif lo_normalized is not None and hi_normalized is not None:
            return lambda logic: logic.AND(
                logic.LEQ(lo_normalized, y_adv), logic.LEQ(y_adv, hi_normalized)
            )
        elif lo_normalized is not None and hi_normalized is None:
            return lambda logic: logic.LEQ(lo_normalized, y_adv)
        else:
            raise ValueError(
                "need to specify either lower or upper (or both) bounds for e_x"
            )


class GroupPostcondition(Postcondition):
    """
    Postcondition ensuring similar outputs for grouped inputs.

    Enforces that neural network outputs remain within delta for inputs
    that belong to the same group, promoting consistency within groups.

    Args:
        device: PyTorch device for tensor computations.
        indices: List of lists, each inner list contains indices of inputs in a group.
        delta: Maximum allowed difference between outputs within each group.
    """

    def __init__(self, device: torch.device, indices: list[list[int]], delta: float):
        self.device = device

        self.indices = indices

        assert 0.0 <= delta <= 1.0, (  # nosec
            "delta is a probability and should be within the range [0, 1]"
        )
        self.delta = torch.as_tensor(delta, device=self.device)

    def get_postcondition(
        self,
        N: torch.nn.Module,
        x_adv: torch.Tensor | None,
    ) -> Callable[[Logic], torch.Tensor]:
        """Get group consistency postcondition for adversarial inputs.

        This implementation demonstrates another specialized signature,
        using only the model and adversarial inputs (no original input needed).

        Args:
            N: Neural network model.
            x_adv: Adversarial input tensor.

        Returns:
            Function that constrains grouped outputs to be within delta bounds.
        """
        y_adv = F.softmax(N(x_adv), dim=1)
        sums = [torch.sum(y_adv[:, i], dim=1) for i in self.indices]

        return lambda logic: logic.AND(
            *[
                logic.OR(logic.LEQ(s, self.delta), logic.GEQ(s, 1.0 - self.delta))
                for s in sums
            ]
        )
