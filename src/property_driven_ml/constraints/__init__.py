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
)
from .preconditions import EpsilonBall
from .postconditions import (
    StandardRobustnessPostcondition,
    LipschitzRobustnessPostcondition,
    GroupPostcondition,
    AlsomitraOutputPostcondition,
)

__all__ = [
    # Constraints
    "Constraint",
    "StandardRobustnessConstraint",
    # Preconditions
    "EpsilonBall",
    # Postconditions
    "StandardRobustnessPostcondition",
    "LipschitzRobustnessPostcondition",
    "GroupPostcondition",
    "AlsomitraOutputPostcondition",
]
