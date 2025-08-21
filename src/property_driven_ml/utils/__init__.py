"""
Utility functions and modules for property-driven machine learning.

This module provides various utility functions and safe evaluation utilities
used throughout the framework.
"""

from .util import safe_div, safe_pow, safe_zero, maybe
from .safe_eval import safe_call, safe_bounds

# Note: CONSTRAINT_FACTORIES is not imported here to avoid circular imports
# Import it directly from .factories when needed

__all__ = ["safe_div", "safe_pow", "safe_zero", "maybe", "safe_call", "safe_bounds"]
