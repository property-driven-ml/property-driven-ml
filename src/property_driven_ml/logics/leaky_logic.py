import torch

from .logic import Logic

from typing import NoReturn


class LeakyLogic(Logic):
    """Implementation of LeakyLogic, based on DL2 but with gradients even when constraints are satisfied.

    Provides differentiable, positive real-valued operators for translating
    logical formulas into loss.
    """

    def __init__(self):
        super().__init__("LL")

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
        tau = 2e-1
        delta = 0.0  # 3e-1
        return torch.nn.functional.softplus((x - y - delta) / tau) * tau

    def GT(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        delta = 1e-3
        return self.LEQ(y + delta, x)

    # 1. LSE
    # def AND(self, *xs: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    #     # smaller tau -> sharper, closer to exact max
    #     return tau * torch.logsumexp(torch.stack(xs, dim=0) / tau, dim=0)

    # def OR(self, *xs: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    #     # smaller tau -> sharper, closer to exact min
    #     return -tau * torch.logsumexp(-torch.stack(xs, dim=0) / tau, dim=0)

    # generalised mean
    def p_mean(self, *xs: torch.Tensor, p: float, eps: float = 1e-8) -> torch.Tensor:
        values = torch.stack([torch.clamp(x, min=0.0) + eps for x in xs], dim=0)
        return torch.pow(torch.mean(torch.pow(values, p), dim=0), 1.0 / p)

    # 2. p mean
    def AND(self, *xs: torch.Tensor) -> torch.Tensor:
        # p > 0 !! important
        # greater p = sharper max (i.e. closer to normal max)
        return self.p_mean(*xs, p=2.0)

    def OR(self, *xs: torch.Tensor) -> torch.Tensor:
        # p < 0 !! important
        # smaller p = sharper min (i.e. closer to normal min)
        return self.p_mean(*xs, p=-2.0)
