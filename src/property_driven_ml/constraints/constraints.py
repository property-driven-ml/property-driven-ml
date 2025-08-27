import torch
import torch.nn.functional as F
import torch.linalg as LA

from abc import ABC, abstractmethod
from typing import Callable

from ..logics.logic import Logic
from ..logics.boolean_logic import BooleanLogic
from ..logics.fuzzy_logics import FuzzyLogic
from ..logics.stl import STL


class Constraint(ABC):
    """Abstract base class for neural network property constraints.

    Provides a common interface for evaluating logical constraints on neural
    network outputs, supporting different logical frameworks.

    Args:
        device: PyTorch device for tensor computations.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.boolean_logic = BooleanLogic()

    @abstractmethod
    def get_constraint(
        self,
        N: torch.nn.Module,
        x: torch.Tensor | None,
        x_adv: torch.Tensor | None,
        y_target: torch.Tensor | None,
    ) -> Callable[[Logic], torch.Tensor]:
        """Get the constraint function for this property.

        Args:
            N: Neural network model.
            x: Original input tensor.
            x_adv: Adversarial input tensor.
            y_target: Target output tensor.

        Returns:
            Function that takes a Logic instance and returns constraint tensor.
        """
        pass

    def eval(
        self,
        N: torch.nn.Module,
        x: torch.Tensor,
        x_adv: torch.Tensor,
        y_target: torch.Tensor | None,
        logic: Logic,
        reduction: str | None = None,
        skip_sat: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the constraint and compute loss and satisfaction.

        Args:
            N: Neural network model.
            x: Original input tensor.
            x_adv: Adversarial input tensor.
            y_target: Target output tensor.
            logic: Logic framework for constraint evaluation.
            reduction: Optional reduction method for loss aggregation.
            skip_sat: Whether to skip satisfaction computation.

        Returns:
            Tuple of (loss, satisfaction) tensors.
        """
        constraint = self.get_constraint(N, x, x_adv, y_target)

        loss = constraint(logic)
        assert not torch.isnan(loss).any()  # nosec

        if isinstance(logic, FuzzyLogic):
            loss = torch.ones_like(loss) - loss
        elif isinstance(logic, STL):
            loss = torch.clamp(logic.NOT(loss), min=0.0)

        if skip_sat:
            # When skipping sat calculation, return a dummy tensor with same shape as loss
            sat = torch.zeros_like(loss)
        else:
            sat = constraint(self.boolean_logic).float()

        def agg(value: torch.Tensor) -> torch.Tensor:
            if reduction is None:
                return value
            elif reduction == "mean":
                # Convert boolean tensors to float for mean calculation
                if value.dtype == torch.bool:
                    value = value.float()
                return torch.mean(value)
            elif reduction == "sum":
                return torch.sum(value)
            else:
                raise ValueError(f"Unsupported reduction: {reduction}")

        return agg(loss), agg(sat)


class StandardRobustnessConstraint(Constraint):
    """Constraint ensuring model robustness to adversarial perturbations.

    Enforces that the change in output probabilities between original and
    adversarial inputs remains within a specified threshold delta.

    Args:
        device: PyTorch device for tensor computations.
        delta: Maximum allowed change in output probabilities.
    """

    def __init__(self, device: torch.device, delta: float | torch.Tensor):
        super().__init__(device)

        assert 0.0 <= delta <= 1.0, (  # nosec
            "delta is a probability and should be within the range [0, 1]"
        )
        self.delta = torch.as_tensor(delta, device=self.device)

    def get_constraint(  # type: ignore
        self, N: torch.nn.Module, x: torch.Tensor, x_adv: torch.Tensor, _y_target: None
    ) -> Callable[[Logic], torch.Tensor]:
        """Get robustness constraint for probability difference bounds.

        Args:
            N: Neural network model.
            x: Original input tensor.
            x_adv: Adversarial input tensor.
            _y_target: Unused target tensor.

        Returns:
            Function that constrains infinity norm of probability differences.
        """
        y = N(x)
        y_adv = N(x_adv)

        diff = F.softmax(y_adv, dim=1) - F.softmax(y, dim=1)

        return lambda logic: logic.LEQ(
            LA.vector_norm(diff, ord=float("inf"), dim=1), self.delta
        )


class LipschitzRobustnessConstraint(Constraint):
    """Constraint enforcing Lipschitz continuity for model robustness.

    Ensures that the rate of change in model outputs is bounded by the
    Lipschitz constant L relative to input perturbations.

    Args:
        device: PyTorch device for tensor computations.
        L: Lipschitz constant bounding the rate of output change.
    """

    def __init__(self, device: torch.device, L: float):
        super().__init__(device)

        self.L = torch.as_tensor(L, device=device)

    def get_constraint(  # type: ignore
        self, N: torch.nn.Module, x: torch.Tensor, x_adv: torch.Tensor, _y_target: None
    ) -> Callable[[Logic], torch.Tensor]:
        """Get Lipschitz constraint relating input and output changes.

        Args:
            N: Neural network model.
            x: Original input tensor.
            x_adv: Perturbed input tensor.
            _y_target: Unused target tensor.

        Returns:
            Function that constrains output change by L times input change.
        """
        y = N(x)
        y_adv = N(x_adv)

        diff_x = LA.vector_norm(x_adv - x, ord=2, dim=1)
        diff_y = LA.vector_norm(y_adv - y, ord=2, dim=1)

        return lambda logic: logic.LEQ(diff_y, self.L * diff_x)


class AlsomitraOutputConstraint(Constraint):
    """Constraint ensuring model outputs fall within specified bounds.

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
        lo: float | torch.Tensor | None,
        hi: float | torch.Tensor | None,
        normalize: bool = True,
    ):
        super().__init__(device)

        # Store raw bounds and normalization flag
        self.lo_raw = lo
        self.hi_raw = hi
        self.normalize = normalize

        # If normalization is disabled (for backwards compatibility), store as tensors directly
        if not normalize:
            self.lo = torch.as_tensor(lo, device=device) if lo is not None else None
            self.hi = torch.as_tensor(hi, device=device) if hi is not None else None

    def get_constraint(  # type: ignore
        self,
        N: torch.nn.Module,
        _x: None,
        x_adv: torch.Tensor,
        _y_target: None,
        scale: torch.Tensor | None = None,
        centre: torch.Tensor | None = None,
    ) -> Callable[[Logic], torch.Tensor]:
        """Get output bounds constraint for adversarial inputs.

        Args:
            N: Neural network model.
            _x: Unused original input tensor.
            x_adv: Adversarial input tensor.
            _y_target: Unused target tensor.
            scale: Optional scaling factor for normalization.
            centre: Optional centre point for normalization.

        Returns:
            Function that constrains outputs to specified bounds.
        """
        y_adv = N(x_adv).squeeze()

        if self.normalize:
            # Normalize bounds at constraint time using class constants
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


class GroupConstraint(Constraint):
    """Constraint ensuring similar outputs for grouped inputs.

    Enforces that neural network outputs remain within delta for inputs
    that belong to the same group, promoting consistency within groups.

    Args:
        device: PyTorch device for tensor computations.
        indices: List of lists, each inner list contains indices of inputs in a group.
        delta: Maximum allowed difference between outputs within each group.
    """

    def __init__(self, device: torch.device, indices: list[list[int]], delta: float):
        super().__init__(device)

        self.indices = indices

        assert 0.0 <= delta <= 1.0, (  # nosec
            "delta is a probability and should be within the range [0, 1]"
        )
        self.delta = torch.as_tensor(delta, device=self.device)

    def get_constraint(  # type: ignore
        self, N: torch.nn.Module, _x: None, x_adv: torch.Tensor, _y_target: None
    ) -> Callable[[Logic], torch.Tensor]:
        """Get group consistency constraint for adversarial inputs.

        Args:
            N: Neural network model.
            _x: Unused original input tensor.
            x_adv: Adversarial input tensor.
            _y_target: Unused target tensor.

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
