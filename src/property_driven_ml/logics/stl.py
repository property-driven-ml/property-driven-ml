import torch

from .logic import Logic

from ..utils import safe_div


class STL(Logic):
    """Signal Temporal Logic implementation for real-valued constraints.

    Provides smooth approximations of logical operations using
    exponential functions, enabling gradient-based optimization
    while preserving logical semantics.

    Args:
        k: Smoothness parameter (higher values give sharper approximations).
    """

    def __init__(self, k=1.0):
        super().__init__("STL")
        self.k = k

    def NOT(self, x: torch.Tensor) -> torch.Tensor:
        """STL logical negation.

        Args:
            x: Tensor to negate.

        Returns:
            Negated tensor -x.
        """
        return -x

    def NEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass # TODO!

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """STL less than or equal operation.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Real-valued difference y - x (positive when x <= y).
        """
        return y - x

    def AND(self, *xs) -> torch.Tensor:
        """STL smooth minimum approximation for conjunction.

        Uses exponential smoothing to approximate the minimum function
        in a differentiable way, enabling gradient-based optimization.

        Args:
            *xs: Variable number of tensors to combine with AND.

        Returns:
            Smooth minimum approximation of input tensors.
        """
        xs = torch.stack(xs)
        x_min, _ = torch.min(xs, dim=0)
        rel = safe_div(xs - x_min, x_min)

        # NOTE: for numerical stability do not directly calculate exp

        # case 1: x_min < 0
        rel_max = rel.max(dim=0, keepdim=True).values
        exp1 = torch.exp(rel - rel_max)
        exp2 = torch.exp(self.k * rel - self.k * rel_max)

        num = (x_min * exp1 * exp2).sum(dim=0)
        denom = exp1.sum(dim=0)
        neg = safe_div(num, denom)

        # case 2: x_min > 0
        krel = -self.k * rel
        krel_max = krel.max(dim=0, keepdim=True).values
        exp_krel = torch.exp(krel - krel_max)

        num = (xs * exp_krel).sum(dim=0)
        denom = exp_krel.sum(dim=0)
        pos = safe_div(num, denom)

        return torch.where(x_min < 0, neg, torch.where(x_min > 0, pos, x_min))

    def OR(self, *xs) -> torch.Tensor:
        return self.NOT(self.AND(*[self.NOT(x) for x in xs]))
