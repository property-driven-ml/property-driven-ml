"""
Safe constraint instantiation utility.

This module provides a safe way to instantiate constraint classes from
string names, avoiding the use of eval() or exec().
"""

from typing import Any, Dict, Callable


def safe_call(constraint_name: str, allowed: Dict[str, Any]) -> Callable:
    """Safely get a constraint class by name.

    Args:
        constraint_name: String name of the constraint class to get.
        allowed: Dictionary mapping constraint names to constraint classes.

    Returns:
        The constraint class (not instantiated).

    Raises:
        ValueError: If the constraint name is not in the allowed dictionary.
    """
    if constraint_name not in allowed:
        available = ", ".join(allowed.keys())
        raise ValueError(
            f"Constraint '{constraint_name}' is not allowed. Available: {available}"
        )

    return allowed[constraint_name]
