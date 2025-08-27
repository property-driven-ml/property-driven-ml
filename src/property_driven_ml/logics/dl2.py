import torch

from .logic import Logic

from typing import NoReturn


class DL2(Logic):
    """Implementation of DL2.

    Reference: https://proceedings.mlr.press/v97/fischer19a.html

    Provides differentiable, positive real-valued operators for translating
    logical formulas into loss.
    """

    def __init__(self):
        super().__init__("DL2")

    def NOT(self, _x: None) -> NoReturn:
        """DL2 logical negation.

        This function is unsupported and must not be called. DL2 does **not**
        provide general negation. Rewrite constraints to push negation
        inwards (e.g., ``NOT(x <= y)`` should be ``y < x``).

        Args:
            _x: Tensor to negate.

        Raises:
            NotImplementedError: Always. General negation is not supported.
        """
        raise NotImplementedError(
            "DL2 does not have general negation - rewrite the constraint to push negation inwards, e.g. NOT(x <= y) should be (y < x)"
        )

    def NEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xi = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        return xi * (x == y).float()

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x - y, min=0.0)

    def AND2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

    def OR2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y
