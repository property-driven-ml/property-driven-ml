import torch

from .logic import Logic

from ..utils import safe_div, safe_pow


class FuzzyLogic(Logic):
    """Base class for fuzzy logic implementations.

    Provides fuzzy variants of logical operations that return values
    in [0,1] rather than crisp Boolean values, enabling gradual
    constraint satisfaction in neural network training.

    Args:
        name: Human-readable name for this fuzzy logic variant.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def NOT(self, x: torch.Tensor) -> torch.Tensor:
        """Fuzzy logical negation.

        Args:
            x: Tensor to negate.

        Returns:
            Fuzzy strong negation (1 - x).
        """
        return 1.0 - x
    
    def NEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fuzzy inequality.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Grounds x != y into [0, 1] for real-valued x, y.
        """
        return safe_div(torch.abs(x - y), (torch.abs(x) + torch.abs(y)))

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fuzzy less than or equal comparison.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Grounds x <= y into [0, 1] for real-valued x, y.
        """
        return 1.0 - safe_div(
            torch.clamp(x - y, min=0.0), (torch.abs(x) + torch.abs(y))
        )


class GoedelFuzzyLogic(FuzzyLogic):
    """Gödel fuzzy logic implementation.

    Uses minimum for conjunction, maximum for disjunction, and
    a conditional-based implication operator.

    Args:
        name: Logic name (defaults to "GD").
    """

    def __init__(self, name="GD"):
        super().__init__(name)

    def AND2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gödel fuzzy conjunction using minimum.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Element-wise minimum of x and y.
        """
        return torch.minimum(x, y)

    def OR2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gödel fuzzy disjunction using maximum.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Element-wise maximum of x and y.
        """
        return torch.maximum(x, y)

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gödel fuzzy implication.

        Args:
            x: Antecedent tensor.
            y: Consequent tensor.

        Returns:
            1.0 where x < y, otherwise y.
        """
        return torch.where(x < y, 1.0, y)


class KleeneDienesFuzzyLogic(GoedelFuzzyLogic):
    """Kleene-Dienes fuzzy logic implementation.

    Extends Gödel logic with a different implication operator
    based on standard logical implication.
    """

    def __init__(self):
        super().__init__(name="KD")

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Kleene-Dienes fuzzy implication.

        Args:
            x: Antecedent tensor.
            y: Consequent tensor.

        Returns:
            Standard logical implication NOT(x) OR y.
        """
        return Logic.IMPL(self, x, y)


class LukasiewiczFuzzyLogic(GoedelFuzzyLogic):
    """Łukasiewicz fuzzy logic implementation.

    Uses bounded sum and difference for conjunction and disjunction,
    providing stronger interaction between fuzzy values.
    """

    def __init__(self):
        super().__init__(name="LK")

    def AND2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Łukasiewicz fuzzy conjunction using bounded difference.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            max(0, x + y - 1) for bounded conjunction.
        """
        return torch.maximum(torch.zeros_like(x), x + y - 1.0)

    def OR2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Łukasiewicz fuzzy disjunction using bounded sum.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            min(1, x + y) for bounded disjunction.
        """
        return torch.minimum(torch.ones_like(x), x + y)

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Łukasiewicz fuzzy implication.

        Args:
            x: Antecedent tensor.
            y: Consequent tensor.

        Returns:
            Standard logical implication NOT(x) OR y.
        """
        return Logic.IMPL(self, x, y)


class ReichenbachFuzzyLogic(FuzzyLogic):
    """Reichenbach fuzzy logic implementation.

    Uses probabilistic operators where conjunction is multiplication
    and disjunction follows probabilistic sum rules.

    Args:
        name: Logic name (defaults to "RC").
    """

    def __init__(self, name="RC"):
        super().__init__(name)

    def AND2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Reichenbach fuzzy conjunction using multiplication.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Product x * y for probabilistic conjunction.
        """
        return x * y

    def OR2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Reichenbach fuzzy disjunction using probabilistic sum.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            x + y - x*y for probabilistic disjunction.
        """
        return x + y - x * y


class GoguenFuzzyLogic(ReichenbachFuzzyLogic):
    """Goguen fuzzy logic implementation.

    Extends Reichenbach logic with a ratio-based implication operator
    that handles division by zero gracefully.
    """

    def __init__(self):
        super().__init__(name="GG")

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Goguen fuzzy implication using ratio.

        Args:
            x: Antecedent tensor.
            y: Consequent tensor.

        Returns:
            1.0 if x <= y or x == 0, otherwise y/x.
        """
        return torch.where(
            torch.logical_or(x <= y, x == 0.0),
            torch.tensor(1.0, device=x.device),
            safe_div(y, x),
        )


class ReichenbachSigmoidalFuzzyLogic(ReichenbachFuzzyLogic):
    """Reichenbach fuzzy logic with sigmoidal approximation.

    Uses sigmoid functions to provide smooth approximations of fuzzy
    operations, making them more suitable for gradient-based optimization.

    Args:
        s: Sigmoid steepness parameter (higher values give sharper transitions).
    """

    def __init__(self, s=9.0):
        super().__init__(name="RCS")
        self.s = s

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sigmoidal approximation of Reichenbach implication.

        Args:
            x: Antecedent tensor.
            y: Consequent tensor.

        Returns:
            Smooth sigmoid-based approximation of implication.
        """
        exp = torch.exp(torch.tensor(self.s / 2, device=x.device))

        numerator = (1.0 + exp) * torch.sigmoid(
            self.s * super().IMPL(x, y) - self.s / 2
        ) - 1.0
        denominator = exp - 1.0

        I_s = torch.clamp(safe_div(numerator, denominator), max=1.0)

        return I_s


class YagerFuzzyLogic(FuzzyLogic):
    """Yager fuzzy logic implementation.

    Uses parameterized operations based on the Yager class of t-norms
    and t-conorms with adjustable parameter p.

    Args:
        p: Yager parameter controlling operator behavior (p >= 1).
    """

    def __init__(self, p=2):
        super().__init__(name="YG")
        self.p = p

    def AND2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Yager fuzzy conjunction.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Yager t-norm with parameter p.
        """
        return torch.clamp(
            1.0
            - safe_pow(
                safe_pow(1.0 - x, self.p) + safe_pow(1.0 - y, self.p), 1.0 / self.p
            ),
            min=0.0,
        )

    def OR2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Yager fuzzy disjunction.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Yager t-conorm with parameter p.
        """
        return torch.clamp(
            safe_pow(safe_pow(x, self.p) + safe_pow(y, self.p), 1.0 / self.p), max=1.0
        )

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.where(
            torch.logical_and(x == 0.0, y == 0.0), torch.ones_like(x), safe_pow(y, x)
        )
