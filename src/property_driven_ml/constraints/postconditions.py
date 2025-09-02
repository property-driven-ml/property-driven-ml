import torch
import torch.nn.functional as F
import torch.linalg as LA
import inspect

from abc import ABC, abstractmethod
from typing import Callable

from ..logics.logic import Logic
from ..logics.boolean_logic import BooleanLogic
from ..logics.fuzzy_logics import FuzzyLogic
from ..logics.stl import STL

BOOLEAN_LOGIC = BooleanLogic()


class Precondition(ABC):
    """
    Abstract base class for preconditions/ input postconditions.
    """

    @abstractmethod
    def get_precondition(self, *args, **kwargs) -> Callable[[Logic], torch.Tensor]:
        """Get the input constraint function for this property.

        Returns:
            Function that takes a Logic instance and returns constraint tensor.
        """
        pass


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


class Constraint(ABC):
    """
    Abstract base class for neural network property constraints, which are a combination of a precondition and postcondition.

    Provides a common interface for evaluating logical constraints on neural
    network outputs, supporting different logical frameworks.

    Args:
        device: PyTorch device for tensor computations.
        precondition: Precondition instance defining input constraints.
        postcondition: Postcondition instance defining output property.
    """

    @abstractmethod
    def __init__(
        self,
        device: torch.device,
        precondition: Precondition,
        postcondition: Postcondition,
    ):
        """
        Initialize the constraint with the given device, precondition, and postcondition.
        The exact details of how pre and postconditions are initialized may vary
        depending on the specific constraint implementation.

        Args:
            device: PyTorch device for tensor computations.
            precondition: Precondition instance defining input constraints.
            postcondition: Postcondition instance defining output property.
        """
        self.postcondition = postcondition
        self.precondition = precondition
        self.device = device

    def eval(
        self,
        N: torch.nn.Module,
        x: torch.Tensor,
        x_adv: torch.Tensor,
        y_target: torch.Tensor | None,
        logic: Logic,
        reduction: str | None = None,
        skip_sat: bool = False,
        postcondition_kwargs: dict = {},
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the constraint and compute loss and satisfaction.

        This method automatically adapts to any postcondition signature by using
        introspection to determine which parameters the postcondition needs and
        only passing those parameters.

        Examples of supported postcondition signatures:
            get_postcondition(self, N, x, x_adv)              # StandardRobustness
            get_postcondition(self, N, x_adv)                 # GroupConstraint
            get_postcondition(self, N, x_adv, scale, centre)  # AlsomitraOutput
            get_postcondition(self, N, x, x_adv, y_target)    # Future constraints

        Args:
            N: Neural network model.
            x: Original input tensor.
            x_adv: Adversarial input tensor.
            y_target: Target output tensor.
            logic: Logic framework for constraint evaluation.
            reduction: Optional reduction method for loss aggregation.
            skip_sat: Whether to skip satisfaction computation.
            postcondition_args: Additional arguments to pass to get_postcondition
                                  (e.g., scale, centre for AlsomitraOutputConstraint).

        Returns:
            Tuple of (loss, satisfaction) tensors.
        """
        # Get the signature of the postcondition's get_postcondition method
        sig = inspect.signature(self.postcondition.get_postcondition)

        # Build a dictionary of all available parameters
        available_params = {
            "N": N,
            "x": x,
            "x_adv": x_adv,
            "y_target": y_target,
            **postcondition_kwargs,
        }

        # Filter to only include parameters that the method accepts
        method_params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue  # Skip 'self' parameter
            if param_name in available_params:
                method_params[param_name] = available_params[param_name]
            elif param.default is not param.empty:
                # Parameter has a default value, don't need to provide it
                continue
            else:
                # Required parameter not available - this could be an error
                # but we'll let the method call fail naturally with a clear error
                pass

        # Call the method with only the parameters it accepts
        postcondition = self.postcondition.get_postcondition(**method_params)

        loss = postcondition(logic)
        assert not torch.isnan(loss).any()  # nosec

        if isinstance(logic, FuzzyLogic):
            loss = torch.ones_like(loss) - loss
        elif isinstance(logic, STL):
            loss = torch.clamp(logic.NOT(loss), min=0.0)

        if skip_sat:
            # When skipping sat calculation, return a dummy tensor with same shape as loss
            sat = torch.zeros_like(loss)
        else:
            sat = postcondition(BOOLEAN_LOGIC).float()

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


class StandardRobustnessPostcondition(Postcondition):
    """postcondition ensuring model robustness to adversarial perturbations.

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
        lo: float | torch.Tensor | None,
        hi: float | torch.Tensor | None,
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
