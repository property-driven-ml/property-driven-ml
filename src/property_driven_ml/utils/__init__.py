"""
Utility functions and modules for property-driven machine learning.

This module provides various utility functions and safe evaluation utilities
used throughout the framework.
"""

from .util import safe_div, safe_pow, safe_zero, maybe
from .safe_eval import safe_call

__all__ = [
    "safe_div",
    "safe_pow",
    "safe_zero",
    "maybe",
    "safe_call",
]
