import torch

from contextlib import contextmanager

@contextmanager
def maybe(context_manager, flag: bool):
    if flag:
        with context_manager as cm:
            yield cm
    else:
        yield None

def safe_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x / torch.where(y == 0., torch.finfo(y.dtype).eps, y)

def safe_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x == 0., torch.full_like(x, torch.finfo(x.dtype).eps), x)

def safe_pow(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.pow(safe_zero(x), y)