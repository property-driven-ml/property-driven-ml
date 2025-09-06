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
            Fuzzy standard negation (1 - x).
        """
        return 1.0 - x

    def NEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fuzzy inequality.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Maps x != y into [0, 1] for real-valued x, y.
        """
        return safe_div(torch.abs(x - y), (torch.abs(x) + torch.abs(y)))

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fuzzy less than or equal comparison.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Maps x <= y into [0, 1] for real-valued x, y.
        """
        return 1.0 - safe_div(
            torch.clamp(x - y, min=0.0), (torch.abs(x) + torch.abs(y))
        )


class FuzzyLogicWithSNImplication(FuzzyLogic):
    """Provides (S,N)-implication: NOT(x) OR y.

    Requires the base class to implement OR(x, y) and NOT(x).
    """

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.OR(self.NOT(x), y)


class GoedelFuzzyLogic(FuzzyLogic):
    """Gödel fuzzy logic implementation.

    Uses the minimum t-norm for conjunction, its t-conorm for disjunction,
    and the R-implication based on the t-norm residuum.

    Args:
        name: Logic name (defaults to "GD").
    """

    def __init__(self, name="GD"):
        super().__init__(name)

    def AND2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gödel conjunction using the minimum t-norm.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Element-wise minimum of x and y.
        """
        return torch.minimum(x, y)

    def OR2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gödel disjunction using the minimum t-conorm.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Element-wise maximum of x and y.
        """
        return torch.maximum(x, y)

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gödel R-implication using the minimum t-norm residuum.

        Args:
            x: Antecedent tensor.
            y: Consequent tensor.

        Returns:
            1.0 where x < y, otherwise y.
        """
        return torch.where(x < y, 1.0, y)


class KleeneDienesFuzzyLogic(FuzzyLogicWithSNImplication, GoedelFuzzyLogic):
    """Kleene-Dienes fuzzy logic implementation.

    Uses the minimum t-norm for conjunction, its t-conorm for disjunction,
    and the (S,N)-implication based on t-conorm S and standard negation N.
    """

    def __init__(self):
        super().__init__(name="KD")


class LukasiewiczFuzzyLogic(FuzzyLogicWithSNImplication, FuzzyLogic):
    """Łukasiewicz fuzzy logic implementation.

    Uses the Łukasiewicz t-norm for conjunction, its t-conorm for disjunction.
    Its implication is both an R-implication and (S,N)-implication
    based on t-conorm S and standard negation N.
    """

    def __init__(self):
        super().__init__(name="LK")

    def AND2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Łukasiewicz conjunction using the Łukasiewicz t-norm.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            max(0, x + y - 1) for bounded conjunction.
        """
        return torch.maximum(torch.zeros_like(x), x + y - 1.0)

    def OR2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Łukasiewicz disjunction using the Łukasiewicz t-conorm.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            min(1, x + y) for bounded disjunction.
        """
        return torch.minimum(torch.ones_like(x), x + y)


class ReichenbachFuzzyLogic(FuzzyLogicWithSNImplication, FuzzyLogic):
    """Reichenbach fuzzy logic implementation.

    Uses the product t-norm for conjunction, its t-conorm (probabilistic sum) for disjunction,
    and the (S,N)-implication based on t-conorm S and standard negation N.

    Args:
        name: Logic name (defaults to "RC").
    """

    def __init__(self, name="RC"):
        super().__init__(name)

    def AND2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Reichenbach conjunction using the product t-norm.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            The product t-norm x * y.
        """
        return x * y

    def OR2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Reichenbach disjunction using probabilistic sum.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            The probabilistic sum x + y - x*y.
        """
        return x + y - x * y


class GoguenFuzzyLogic(ReichenbachFuzzyLogic):
    """Goguen fuzzy logic implementation.

    Uses the product t-norm for conjunction, its t-conorm (probabilistic sum) for disjunction,
    and the R-implication based on the t-norm residuum.
    """

    def __init__(self):
        super().__init__(name="GG")

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Goguen R-implication.

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

    Reference: https://doi.org/10.1016/j.artint.2021.103602

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
