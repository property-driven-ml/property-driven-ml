"""
Enhanced constraint system that unifies input regions and output constraints.

This module provides constraint classes that can handle both input region
specification and output constraint evaluation, eliminating the need for
separate BoundedDataset classes.
"""

import torch
from abc import abstractmethod
from typing import Callable, Optional, Tuple

from ..logics.logic import Logic
from .constraints import Constraint


class InputRegionConstraint(Constraint):
    """Enhanced constraint that includes input region specification.

    This class extends the base Constraint to also handle input region
    bounds computation, unifying the previously separate BoundedDataset
    and Constraint class hierarchies.
    """

    def __init__(
        self,
        device: torch.device,
        transform: Optional[torch.nn.Module] = None,
    ):
        """Initialize InputRegionConstraint.

        Args:
            device: PyTorch device for tensor computations.
            transform: Optional transformation pipeline from problem space to input space.
        """
        super().__init__(device)
        self.transform = transform

    @abstractmethod
    def get_input_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute input bounds for the given input tensor.

        Args:
            x: Input tensor in the model's input space.

        Returns:
            Tuple of (lower_bounds, upper_bounds) tensors.
        """
        pass

    def apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformation pipeline to input.

        Args:
            x: Input tensor in problem space.

        Returns:
            Transformed tensor in model input space.
        """
        if self.transform is not None:
            return self.transform(x)
        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse transformation to get back to problem space.

        Args:
            x: Input tensor in model input space.

        Returns:
            Tensor in problem space.
        """
        # For now, assume identity if no transform specified
        # TODO: Add support for invertible transforms
        if self.transform is not None:
            # This would need to be implemented based on the specific transform
            raise NotImplementedError("Inverse transform not yet implemented")
        return x


class EpsilonBallConstraint(InputRegionConstraint):
    """Constraint with epsilon-ball input regions around each input."""

    def __init__(
        self,
        device: torch.device,
        output_constraint: Constraint,
        eps: float,
        transform: Optional[torch.nn.Module] = None,
    ):
        """Initialize EpsilonBallConstraint.

        Args:
            device: PyTorch device for tensor computations.
            output_constraint: The underlying output constraint to enforce.
            eps: Epsilon value for ball radius.
            transform: Optional transformation pipeline.
        """
        super().__init__(device, transform)
        self.output_constraint = output_constraint
        self.eps = torch.as_tensor(eps, device=device)

    def get_input_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute epsilon-ball bounds around input x."""
        eps_expanded = self.eps.view(*self.eps.shape, *([1] * (x.ndim - self.eps.ndim)))
        return x - eps_expanded, x + eps_expanded

    def get_constraint(
        self,
        N: torch.nn.Module,
        x: torch.Tensor | None,
        x_adv: torch.Tensor | None,
        y_target: torch.Tensor | None,
    ) -> Callable[[Logic], torch.Tensor]:
        """Delegate to the underlying output constraint."""
        # Handle type conversion - robustness constraints expect non-None x and x_adv
        assert x is not None and x_adv is not None  # nosec
        # Most robustness constraints expect None for y_target
        return self.output_constraint.get_constraint(N, x, x_adv, None)


class GlobalBoundsConstraint(InputRegionConstraint):
    """Constraint with global bounds applied to all inputs."""

    def __init__(
        self,
        device: torch.device,
        output_constraint: Constraint,
        lo: torch.Tensor,
        hi: torch.Tensor,
        transform: Optional[torch.nn.Module] = None,
    ):
        """Initialize GlobalBoundsConstraint.

        Args:
            device: PyTorch device for tensor computations.
            output_constraint: The underlying output constraint to enforce.
            lo: Lower bound tensor.
            hi: Upper bound tensor.
            transform: Optional transformation pipeline.
        """
        super().__init__(device, transform)
        self.output_constraint = output_constraint
        self.lo = lo.to(device)
        self.hi = hi.to(device)

    def get_input_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the global bounds for any input."""
        batch_size = x.shape[0]
        lo_batch = self.lo.expand(batch_size, *self.lo.shape)
        hi_batch = self.hi.expand(batch_size, *self.hi.shape)
        return lo_batch, hi_batch

    def get_constraint(
        self,
        N: torch.nn.Module,
        x: torch.Tensor | None,
        x_adv: torch.Tensor | None,
        y_target: torch.Tensor | None,
    ) -> Callable[[Logic], torch.Tensor]:
        """Delegate to the underlying output constraint."""
        # Handle type conversion - robustness constraints expect non-None x and x_adv
        assert x is not None and x_adv is not None  # nosec
        # Most robustness constraints expect None for y_target
        return self.output_constraint.get_constraint(N, x, x_adv, None)


class AlsomitraConstraint(InputRegionConstraint):
    """Combined input region and output constraint for Alsomitra aerodynamics."""

    def __init__(
        self,
        device: torch.device,
        bounds_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        output_constraint: Constraint,
        transform: Optional[torch.nn.Module] = None,
    ):
        """Initialize AlsomitraConstraint.

        Args:
            device: PyTorch device for tensor computations.
            bounds_fn: Function that computes bounds given input tensor in problem space.
            output_constraint: The output constraint to enforce.
            transform: Optional transformation pipeline.
        """
        super().__init__(device, transform)
        self.bounds_fn = bounds_fn
        self.output_constraint = output_constraint

    def get_input_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute bounds using the provided bounds function."""
        # Convert to problem space if needed, compute bounds, convert back
        x_problem = self.inverse_transform(x)
        lo_problem, hi_problem = self.bounds_fn(x_problem)
        lo_input = self.apply_transform(lo_problem)
        hi_input = self.apply_transform(hi_problem)
        return lo_input, hi_input

    def get_constraint(
        self,
        N: torch.nn.Module,
        x: torch.Tensor | None,
        x_adv: torch.Tensor | None,
        y_target: torch.Tensor | None,
    ) -> Callable[[Logic], torch.Tensor]:
        """Delegate to the underlying output constraint."""
        # AlsomitraOutputConstraint may have different signature requirements
        # Handle based on the specific constraint type
        return self.output_constraint.get_constraint(N, x, x_adv, y_target)


class StandardRobustnessWithInputRegion(InputRegionConstraint):
    """Standard robustness constraint with integrated input region."""

    def __init__(
        self,
        device: torch.device,
        delta: float,
        eps: float,
        transform: Optional[torch.nn.Module] = None,
    ):
        """Initialize StandardRobustnessWithInputRegion.

        Args:
            device: PyTorch device for tensor computations.
            delta: Maximum allowed change in output probabilities.
            eps: Epsilon value for input perturbation bounds.
            transform: Optional transformation pipeline.
        """
        super().__init__(device, transform)
        from .constraints import StandardRobustnessConstraint

        self.output_constraint = StandardRobustnessConstraint(device, delta)
        self.eps = torch.as_tensor(eps, device=device)

    def get_input_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute epsilon-ball bounds around input x."""
        eps_expanded = self.eps.view(*self.eps.shape, *([1] * (x.ndim - self.eps.ndim)))
        return x - eps_expanded, x + eps_expanded

    def get_constraint(
        self,
        N: torch.nn.Module,
        x: torch.Tensor | None,
        x_adv: torch.Tensor | None,
        y_target: torch.Tensor | None,
    ) -> Callable[[Logic], torch.Tensor]:
        """Delegate to the underlying output constraint."""
        # StandardRobustnessConstraint expects non-None x and x_adv, None for y_target
        assert x is not None and x_adv is not None  # nosec
        return self.output_constraint.get_constraint(N, x, x_adv, None)


class LipschitzRobustnessWithInputRegion(InputRegionConstraint):
    """Lipschitz robustness constraint with integrated input region."""

    def __init__(
        self,
        device: torch.device,
        L: float,
        eps: float,
        transform: Optional[torch.nn.Module] = None,
    ):
        """Initialize LipschitzRobustnessWithInputRegion.

        Args:
            device: PyTorch device for tensor computations.
            L: Lipschitz constant.
            eps: Epsilon value for input perturbation bounds.
            transform: Optional transformation pipeline.
        """
        super().__init__(device, transform)
        from .constraints import LipschitzRobustnessConstraint

        self.output_constraint = LipschitzRobustnessConstraint(device, L)
        self.eps = torch.as_tensor(eps, device=device)

    def get_input_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute epsilon-ball bounds around input x."""
        eps_expanded = self.eps.view(*self.eps.shape, *([1] * (x.ndim - self.eps.ndim)))
        return x - eps_expanded, x + eps_expanded

    def get_constraint(
        self,
        N: torch.nn.Module,
        x: torch.Tensor | None,
        x_adv: torch.Tensor | None,
        y_target: torch.Tensor | None,
    ) -> Callable[[Logic], torch.Tensor]:
        """Delegate to the underlying output constraint."""
        # LipschitzRobustnessConstraint expects non-None x and x_adv, None for y_target
        assert x is not None and x_adv is not None  # nosec
        return self.output_constraint.get_constraint(N, x, x_adv, None)
