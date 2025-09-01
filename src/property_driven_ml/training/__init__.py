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

# Enhanced components for unified constraint architecture
from .enhanced_attacks import EnhancedAttack, EnhancedPGD, EnhancedAPGD
from .enhanced_engine import (
    train_enhanced,
    test_enhanced,
    create_enhanced_constraint_from_specs,
)

__all__ = [
    # Legacy training components
    "Attack",
    "PGD",
    "APGD",
    "GradNorm",
    "EpochInfoTrain",
    "EpochInfoTest",
    "train",
    "test",
    # Enhanced components for unified constraint architecture
    "EnhancedAttack",
    "EnhancedPGD",
    "EnhancedAPGD",
    "train_enhanced",
    "test_enhanced",
    "create_enhanced_constraint_from_specs",
]
