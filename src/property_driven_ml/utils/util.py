import torch

from contextlib import contextmanager


@contextmanager
def maybe(context_manager, flag: bool):
    """Conditionally apply a context manager based on a flag.

    Args:
        context_manager: Context manager to apply if flag is True.
        flag: Whether to apply the context manager.

    Yields:
        Context manager result if flag is True, otherwise None.
    """
    if flag:
        with context_manager as cm:
            yield cm
    else:
        yield None


def safe_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Safely divide tensors, avoiding division by zero.

    Args:
        x: Numerator tensor.
        y: Denominator tensor.

    Returns:
        Division result with zeros replaced by epsilon.
    """
    return x / torch.where(y == 0.0, torch.finfo(y.dtype).eps, y)


def safe_zero(x: torch.Tensor) -> torch.Tensor:
    """Replace zeros in tensor with epsilon to avoid numerical issues.

    Args:
        x: Input tensor.

    Returns:
        Tensor with zeros replaced by epsilon.
    """
    return torch.where(x == 0.0, torch.full_like(x, torch.finfo(x.dtype).eps), x)


def safe_pow(x: torch.Tensor, y: torch.Tensor | float | int) -> torch.Tensor:
    """Safely compute power, avoiding issues with zero base.

    Args:
        x: Base tensor.
        y: Exponent (tensor, float, or int).

    Returns:
        Power result with zero base replaced by epsilon.
    """
    return torch.pow(safe_zero(x), y)
