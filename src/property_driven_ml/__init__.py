"""
Property-Driven Machine Learning Framework

A general framework for incorporating logical properties and constraints
into machine learning training and evaluation.

Main exports:
- logics: Logic systems submodule (fuzzy, Boolean, STL, DL2)
- constraints: Constraint classes for defining properties
- training: Training utilities for property-driven learning
"""

# Import submodules
from . import logics
from . import constraints
from . import training

# Import utilities directly for convenience
from .util import safe_div, safe_pow, maybe

__version__ = "0.1.0"
__author__ = "Thomas Flinkow"

__all__ = [
    # Submodules
    "logics",
    "constraints", 
    "training",
    
    # Utilities
    "safe_div", "safe_pow", "maybe"
]
