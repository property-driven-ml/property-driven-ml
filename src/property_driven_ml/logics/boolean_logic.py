import torch

from .logic import Logic


class BooleanLogic(Logic):
    """Boolean logic implementation for constraint evaluation.

    Provides standard Boolean operations (AND, OR, NOT, LEQ) using
    PyTorch's logical operations for crisp true/false evaluations.
    """

    def __init__(self):
        super().__init__("bool")

    def NOT(self, x: torch.Tensor) -> torch.Tensor:
        """Boolean logical negation.

        Args:
            x: Tensor to negate.

        Returns:
            Boolean tensor with negated values.
        """
        return torch.logical_not(x)
    
    def NEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Boolean inequality comparison.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Boolean tensor with True where x != y.
        """
        return x != y

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Boolean less than or equal comparison.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Boolean tensor with True where x <= y.
        """
        return x <= y

    def AND2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Boolean logical conjunction.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Boolean tensor with True where both x and y are True.
        """
        return torch.logical_and(x, y)

    def OR2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Boolean logical disjunction.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Boolean tensor with True where either x or y is True.
        """
        return torch.logical_or(x, y)
