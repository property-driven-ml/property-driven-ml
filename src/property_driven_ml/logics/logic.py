import torch

from abc import ABC, abstractmethod
from functools import reduce


class Logic(ABC):
    """Abstract base class for logical frameworks used in property-driven learning.

    Provides a common interface for different logical systems (Boolean, fuzzy, STL)
    used to evaluate constraints and properties in neural network training.

    Args:
        name: Human-readable name for this logic framework.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def NOT(self, x: torch.Tensor) -> torch.Tensor:
        """Logical negation in this logic framework.

        Args:
            x: Tensor to negate.

        Returns:
            Tensor representing NOT x in this logic.
        """
        pass

    @abstractmethod
    def NEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Inequality in this logic framework.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x != y in this logic.
        """
        pass

    def EQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Equality in this logic framework.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x == y in this logic.
        """
        return self.NOT(self.NEQ(x, y))

    @abstractmethod
    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Less than or equal comparison in this logic framework.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x <= y in this logic.
        """
        pass

    def GEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Greater than or equal comparison in this logic framework.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x >= y in this logic.
        """
        return self.LEQ(y, x)

    def LT(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Less than comparison in this logic framework.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x < y in this logic.
        """
        return self.AND(self.LEQ(x, y), self.NEQ(x, y))

    def GT(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Greater than comparison in this logic framework.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x > y in this logic.
        """
        return self.LT(y, x)

    def AND(self, *xs: torch.Tensor) -> torch.Tensor:
        """Logical conjunction of multiple tensors.

        Args:
            *xs: Variable number of tensors to combine with AND.

        Returns:
            Tensor representing the conjunction of all inputs.

        Raises:
            ValueError: If fewer than 2 arguments provided.
        """
        if len(xs) < 2:
            raise ValueError(
                "AND requires at least 2 arguments. If you have a list xs, make sure to unpack it with *xs."
            )

        return reduce(self.AND2, xs)

    def AND2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Binary logical conjunction.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Tensor representing x AND y.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("AND2 must be implemented if AND is not overridden.")

    def OR(self, *xs: torch.Tensor) -> torch.Tensor:
        """Logical disjunction of multiple tensors.

        Args:
            *xs: Variable number of tensors to combine with OR.

        Returns:
            Tensor representing the disjunction of all inputs.

        Raises:
            ValueError: If fewer than 2 arguments provided.
        """
        if len(xs) < 2:
            raise ValueError(
                "OR requires at least 2 arguments. If you have a list xs, make sure to unpack it with *xs."
            )

        return reduce(self.OR2, xs)

    def OR2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Binary logical disjunction.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Tensor representing x OR y.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("OR2 must be implemented if OR is not overridden.")

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logical implication in this logic.

        Args:
            x: Antecedent tensor.
            y: Consequent tensor.

        Returns:
            Tensor representing x ==> y.
        """
        return self.OR(self.NOT(x), y)

    def EQUIV(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logical equivalence in this logic.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x <==> y.
        """
        return self.AND(self.IMPL(x, y), self.IMPL(y, x))
