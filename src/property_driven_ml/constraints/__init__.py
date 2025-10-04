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
    OppositeFacesConstraint,
)
from .preconditions import EpsilonBall
from .postconditions import (
    StandardRobustnessPostcondition,
    OppositeFacesPostcondition,
    LipschitzRobustnessPostcondition,
    GroupPostcondition,
    AlsomitraOutputPostcondition,
)

__all__ = [
    # Constraints
    "Constraint",
    "StandardRobustnessConstraint",
    "OppositeFacesConstraint",
    # Preconditions
    "EpsilonBall",
    # Postconditions
    "StandardRobustnessPostcondition",
    "OppositeFacesPostcondition",
    "LipschitzRobustnessPostcondition",
    "GroupPostcondition",
    "AlsomitraOutputPostcondition",
]
