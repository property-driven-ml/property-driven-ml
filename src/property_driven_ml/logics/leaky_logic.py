import torch

from .logic import Logic

from typing import NoReturn

from ..utils import safe_zero


class LeakyLogic(Logic):
    """Implementation of LeakyLogic, based on DL2 but with gradients even when constraints are satisfied.

    Provides differentiable, positive real-valued operators for translating
    logical formulas into loss.
    """

    def __init__(self, p: float = 2):
        super().__init__("LL")
        self.p = p

    def NOT(self, x: torch.Tensor) -> NoReturn:
        """LeakyLogic logical negation.

        This function is unsupported and must not be called. LeakyLogic does **not**
        provide general negation. Rewrite constraints to push negation
        inwards (e.g., ``NOT(x <= y)`` should be ``y < x``).

        Args:
            x: Tensor to negate.

        Raises:
            NotImplementedError: Always. General negation is not supported.
        """
        raise NotImplementedError(
            "LeakyLogic does not have general negation - rewrite the constraint to push negation inwards, e.g. NOT(LEQ(x, y)) should be GT(x, y)"
        )

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus((x - y))

    def LT(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        delta = 1e-3
        return self.LEQ(x + delta, y)

    def p_sum(self, *xs: torch.Tensor, p: float) -> torch.Tensor:
        # Log-domain p-norm: exp(LSE(p log x) / p) == (sum x^p)^(1/p).
        # Avoids the safe_pow eps-clamp underflow that saturates the naive
        # form at high |p|. See issue #9.
        x = torch.stack(xs, dim=0)
        return torch.exp(torch.logsumexp(p * torch.log(safe_zero(x)), dim=0) / p)

    def AND(self, *xs: torch.Tensor) -> torch.Tensor:
        # p -> infty means sharper max (i.e. closer to standard max)
        return self.p_sum(*xs, p=self.p)

    def OR(self, *xs: torch.Tensor) -> torch.Tensor:
        # p -> -infty means sharper min (i.e. closer to standard min)
        return self.p_sum(*xs, p=-self.p)
