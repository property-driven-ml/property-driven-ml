"""
Training utilities for property-driven machine learning.

This module provides attack algorithms, gradient normalization utilities,
and training/testing engines for training models with property constraints.

The enhanced modules support the new unified constraint architecture.
"""

from .attacks import Attack, PGD, APGD
from .grad_norm import GradNorm
from .epoch_info import EpochInfoTrain, EpochInfoTest
from .engine import train, test

__all__ = [
    # Attacks
    "Attack",
    "PGD",
    "APGD",
    # Gradient normalization
    "GradNorm",
    # Epoch info
    "EpochInfoTrain",
    "EpochInfoTest",
    # Training/testing engines
    "train",
    "test",
]
