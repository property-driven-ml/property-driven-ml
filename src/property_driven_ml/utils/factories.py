"""
Factory functions for creating constraints and input regions.

This module provides factory functions that can be safely called from
user-provided string expressions to create constraint objects.
"""

import torch
from typing import Tuple, Optional, Callable

import property_driven_ml.constraints as constraints
from ..datasets.base import SizedDataset
from .safe_eval import safe_bounds


def CreateEpsilonBall(
    eps: float,
) -> Tuple[
    Callable[..., constraints.EpsilonBall], Callable[..., constraints.EpsilonBall]
]:
    """Create epsilon ball constraint factory functions.

    Args:
        eps: Radius of epsilon ball for perturbations.

    Returns:
        Tuple of (train_factory, test_factory) functions that create EpsilonBall constraints.
    """

    # Note: These will need actual datasets passed to them when created
    # This is a factory that returns a function to create the actual constraints
    def create_eps_ball(dataset: SizedDataset, mean, std):
        return constraints.EpsilonBall(dataset, eps, mean, std)

    return create_eps_ball, create_eps_ball


def CreateAlsomitraInputRegion(
    v_x: Optional[str] = None,
    v_y: Optional[str] = None,
    omega: Optional[str] = None,
    theta: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
) -> Tuple[Callable, Callable]:  # Using generic Callable to avoid circular imports
    """Create Alsomitra input region constraints for train and test."""

    def bounds_fn(input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context = {
            "v_x": input_tensor[0].item(),
            "v_y": input_tensor[1].item(),
            "omega": input_tensor[2].item(),
            "theta": input_tensor[3].item(),
            "x": input_tensor[4].item(),
            "y": input_tensor[5].item(),
        }

        bounds = []
        for param_name, param_expr in [
            ("v_x", v_x),
            ("v_y", v_y),
            ("omega", omega),
            ("theta", theta),
            ("x", x),
            ("y", y),
        ]:
            if param_expr is None:
                # Use original value if no bounds specified
                bounds.extend([context[param_name], context[param_name]])
            else:
                lo, hi = safe_bounds(param_expr, context)
                bounds.extend([lo, hi])

        # Return as lo and hi tensors
        lo_tensor = torch.tensor(
            bounds[::2], device=input_tensor.device, dtype=input_tensor.dtype
        )
        hi_tensor = torch.tensor(
            bounds[1::2], device=input_tensor.device, dtype=input_tensor.dtype
        )
        return lo_tensor, hi_tensor

    def create_region(dataset, mean, std):
        # Import here to avoid circular imports
        from ..constraints.bounded_datasets import AlsomitraInputRegion

        return AlsomitraInputRegion(dataset, bounds_fn, mean, std)

    return create_region, create_region


def CreateStandardRobustnessConstraint(
    delta: float,
) -> constraints.StandardRobustnessConstraint:
    """Create standard robustness constraint."""
    device = torch.device("cpu")  # Will be moved to correct device later
    return constraints.StandardRobustnessConstraint(device, delta)


def CreateLipschitzRobustnessConstraint(
    L: float,
) -> constraints.LipschitzRobustnessConstraint:
    """Create Lipschitz robustness constraint."""
    device = torch.device("cpu")  # Will be moved to correct device later
    return constraints.LipschitzRobustnessConstraint(device, L)


def CreateAlsomitraOutputConstraint(
    e_x: Tuple[float, float],
) -> constraints.AlsomitraOutputConstraint:
    """Create Alsomitra output constraint.

    This factory stores raw bounds and lets the constraint handle normalization
    at runtime, making it independent of AlsomitraDataset initialization order.
    """
    device = torch.device("cpu")  # Will be moved to correct device later
    lo, hi = e_x

    # Pass raw values and enable normalization - constraint will handle it at runtime
    return constraints.AlsomitraOutputConstraint(device, lo, hi, normalize=True)


def CreateGroupConstraint(delta: float) -> constraints.GroupConstraint:
    """Create group fairness constraint."""
    device = torch.device("cpu")  # Will be moved to correct device later
    # Note: This is a placeholder - actual groups would need to be provided
    groups: list[list[int]] = []  # Empty list of groups for now
    return constraints.GroupConstraint(device, groups, delta)


# Registry of allowed constraint factory functions
CONSTRAINT_FACTORIES = {
    "EpsilonBall": CreateEpsilonBall,
    "AlsomitraInputRegion": CreateAlsomitraInputRegion,
    "StandardRobustness": CreateStandardRobustnessConstraint,
    "LipschitzRobustness": CreateLipschitzRobustnessConstraint,
    "AlsomitraOutput": CreateAlsomitraOutputConstraint,
    "GroupConstraint": CreateGroupConstraint,
}
