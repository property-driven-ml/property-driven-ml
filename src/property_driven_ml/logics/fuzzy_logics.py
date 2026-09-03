import torch

from .logic import Logic

from ..utils import safe_div, safe_pow, safe_zero


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

    def EQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fuzzy equality.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Maps x == y into [0, 1] for real-valued x, y.
        """
        return torch.clamp(1.0 - torch.abs(safe_div(x - y, x + y)), min=0.0)

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fuzzy less than or equal comparison.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Maps x <= y into [0, 1] for real-valued x, y. TODO! this is now incorrect!
        """
        return torch.clamp(1.0 - torch.clamp(safe_div(x - y, x + y), min=0.0), min=0.0)

    def GT(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.NOT(self.LEQ(x, y))


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

    def AND(self, *xs: torch.Tensor) -> torch.Tensor:
        """n-ary Gödel conjunction using the minimum t-norm min(x, y)."""
        return torch.min(torch.stack(xs, dim=0), dim=0).values

    def OR(self, *xs: torch.Tensor) -> torch.Tensor:
        """n-ary Gödel disjunction using the minimum t-conorm max(x, y)."""
        return torch.max(torch.stack(xs, dim=0), dim=0).values

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gödel R-implication using the minimum t-norm residuum.

        Args:
            x: Antecedent tensor.
            y: Consequent tensor.

        Returns:
            1.0 where x < y, otherwise y.
        """
        return torch.where(x < y, 1.0, y)

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x <= y,
            1.0 + 0.0 * x,  # torch.ones_like(x),
            y,
        )


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

    def AND(self, *xs: torch.Tensor) -> torch.Tensor:
        """n-ary Łukasiewicz conjunction using the Łukasiewicz t-norm max(0, x + y - 1)."""
        return torch.clamp(
            torch.sum(torch.stack(xs, dim=0), dim=0) - len(xs) + 1, min=0.0
        )

    def OR(self, *xs: torch.Tensor) -> torch.Tensor:
        """Łukasiewicz disjunction using the Łukasiewicz t-conorm min(1, x + y)."""
        return torch.clamp(torch.sum(torch.stack(xs), dim=0), max=1.0)

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.clamp(1.0 - x + y, max=1.0)


class ReichenbachFuzzyLogic(FuzzyLogicWithSNImplication, FuzzyLogic):
    """Reichenbach fuzzy logic implementation.

    Uses the product t-norm for conjunction, its t-conorm (probabilistic sum) for disjunction,
    and the (S,N)-implication based on t-conorm S and standard negation N.

    Args:
        name: Logic name (defaults to "RC").
    """

    def __init__(self, name="RC"):
        super().__init__(name)

    def AND(self, *xs: torch.Tensor) -> torch.Tensor:
        """n-ary Reichenbach conjunction using the product t-norm x * y."""
        return torch.prod(torch.stack(xs, dim=0), dim=0)

    def OR(self, *xs: torch.Tensor) -> torch.Tensor:
        """n-ary Reichenbach disjunction using probabilistic sum x + y - x * y (i.e. 1 - (1 - x) (1 - y))."""
        return 1.0 - torch.prod(1.0 - torch.stack(xs, dim=0), dim=0)

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x <= y,
            1.0 + 0.0 * x,  # torch.ones_like(x),
            safe_div(y, x),
        )


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

    def __init__(self, p=5):
        super().__init__(name="YG")
        self.p = p

    def AND(self, *xs: torch.Tensor) -> torch.Tensor:
        """n-ary Yager t-norm."""
        # Log-domain p-norm of the complements:
        #   (sum (1-x_i)^p)^(1/p) == exp(LSE(p log(1-x_i)) / p)
        # The previous form clamped the inner sum to eps=1e-6, and the clamp
        # survived the outer p-th root: for all-true inputs the result was
        # 1 - eps^(1/p) instead of 1, violating the t-norm boundary axiom
        # T(1, 1) = 1 (issue #11; same bug class as #9). In log domain no
        # clamp on the sum is needed; safe_zero only guards log(0), and its
        # effect on the result is bounded by machine eps for every p instead
        # of eps^(1/p).
        one_minus = 1.0 - torch.stack(xs, dim=0)
        root = torch.exp(
            torch.logsumexp(self.p * torch.log(safe_zero(one_minus)), dim=0) / self.p
        )
        return torch.clamp(1.0 - root, min=0.0, max=1.0)

    def OR(self, *xs: torch.Tensor) -> torch.Tensor:
        """n-ary Yager t-conorm."""
        return torch.clamp(
            torch.pow(
                torch.sum(torch.pow(torch.stack(xs, dim=0), self.p), dim=0),
                1.0 / self.p,
            ),
            max=1.0,
        )

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.where(
            torch.logical_and(x == 0.0, y == 0.0), torch.ones_like(x), safe_pow(y, x)
        )

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        eps = 1e-6

        diff = torch.pow(torch.clamp(1.0 - y, min=0.0), self.p) - torch.pow(
            torch.clamp(1.0 - x, min=0.0), self.p
        )

        violation = 1.0 - torch.pow(
            torch.clamp(diff, min=eps),
            1.0 / self.p,
        )

        return torch.where(
            x <= y,
            torch.ones_like(x),
            violation,
        )
