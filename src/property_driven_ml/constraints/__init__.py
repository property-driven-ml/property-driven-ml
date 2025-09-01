"""
Constraint definitions for property-driven machine learning.

This module provides constraint classes that define properties that
machine learning models should satisfy.

The new unified constraint architecture combines input regions and output
constraints in a single class hierarchy, eliminating the need for separate
BoundedDataset classes.
"""

from .constraints import (
    Constraint,
    StandardRobustnessConstraint,
    LipschitzRobustnessConstraint,
    AlsomitraOutputConstraint,
    GroupConstraint,
)
from .bounded_datasets import EpsilonBall, BoundedDataset, AlsomitraInputRegion
from .base import SizedDataset

# New unified constraint architecture
from .unified_constraints import (
    InputRegionConstraint,
    EpsilonBallConstraint,
    GlobalBoundsConstraint,
    AlsomitraConstraint,
    StandardRobustnessWithInputRegion,
    LipschitzRobustnessWithInputRegion,
)

from ..constraints.factories import (
    CreateEpsilonBall,
    CreateAlsomitraInputRegion,
    CreateStandardRobustnessConstraint,
    CreateLipschitzRobustnessConstraint,
    CreateAlsomitraOutputConstraint,
)

__all__ = [
    # Base constraint classes
    "Constraint",
    "StandardRobustnessConstraint",
    "LipschitzRobustnessConstraint",
    "AlsomitraOutputConstraint",
    "GroupConstraint",
    # Legacy bounded dataset classes (deprecated)
    "EpsilonBall",
    "BoundedDataset",
    "AlsomitraInputRegion",
    "SizedDataset",
    # New unified constraint architecture
    "InputRegionConstraint",
    "EpsilonBallConstraint",
    "GlobalBoundsConstraint",
    "AlsomitraConstraint",
    "StandardRobustnessWithInputRegion",
    "LipschitzRobustnessWithInputRegion",
    # Factory functions
    "CreateEpsilonBall",
    "CreateStandardRobustnessConstraint",
    "CreateLipschitzRobustnessConstraint",
    "CreateAlsomitraOutputConstraint",
    "CreateAlsomitraInputRegion",
]
