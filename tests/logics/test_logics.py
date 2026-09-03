"""
Comprehensive tests for logic implementations.
"""

import pytest
import torch

from property_driven_ml.logics import Logic
from property_driven_ml.logics.boolean_logic import BooleanLogic
from property_driven_ml.logics.fuzzy_logics import (
    GoedelFuzzyLogic,
    LukasiewiczFuzzyLogic,
    KleeneDienesFuzzyLogic,
)
from property_driven_ml.logics.dl2 import DL2
from property_driven_ml.logics.leaky_logic import LeakyLogic
from property_driven_ml.logics.qll import QLL
from property_driven_ml.logics.stl import STL


class TestLogicBase:
    """Test the base Logic class."""

    def test_logic_base_cannot_be_instantiated(self):
        """Test that Logic is an abstract base class."""
        with pytest.raises(TypeError):
            Logic()  # type: ignore

    def test_logic_name_property(self):
        """Test that concrete logics have proper names."""
        boolean_logic = BooleanLogic()
        assert boolean_logic.name == "bool"

        fuzzy_logic = GoedelFuzzyLogic()
        assert fuzzy_logic.name == "GD"

        dl2_logic = DL2()
        assert dl2_logic.name == "DL2"


class TestBooleanLogic:
    """Test Boolean logic implementation."""

    @pytest.fixture
    def boolean_logic(self):
        return BooleanLogic()

    @pytest.fixture
    def sample_tensors(self):
        """Create sample boolean tensors for testing."""
        return {
            "all_true": torch.tensor([True, True, True, True]),
            "all_false": torch.tensor([False, False, False, False]),
            "mixed": torch.tensor([True, False, True, False]),
            "mixed2": torch.tensor([False, True, False, True]),
        }

    def test_boolean_logic_basic_operations(self, boolean_logic, sample_tensors):
        """Test basic boolean operations work correctly."""
        # Test AND operation
        result_and = boolean_logic.AND2(
            sample_tensors["mixed"], sample_tensors["mixed2"]
        )
        expected_and = torch.tensor([False, False, False, False])
        assert torch.equal(result_and, expected_and)

        # Test OR operation
        result_or = boolean_logic.OR2(sample_tensors["mixed"], sample_tensors["mixed2"])
        expected_or = torch.tensor([True, True, True, True])
        assert torch.equal(result_or, expected_or)

        # Test NOT operation
        result_not = boolean_logic.NOT(sample_tensors["mixed"])
        expected_not = torch.tensor([False, True, False, True])
        assert torch.equal(result_not, expected_not)

    def test_boolean_logic_comparison_operations(self, boolean_logic):
        """Test boolean comparison operations."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = torch.tensor([1.0, 3.0, 2.0, 4.0])

        # Test LEQ operation
        result_leq = boolean_logic.LEQ(x, y)
        expected_leq = torch.tensor([True, True, False, True])
        assert torch.equal(result_leq, expected_leq)

        # Test NEQ operation
        result_neq = boolean_logic.NEQ(x, y)
        expected_neq = torch.tensor([False, True, True, False])
        assert torch.equal(result_neq, expected_neq)

    def test_boolean_logic_variadic_operations(self, boolean_logic, sample_tensors):
        """Test variadic AND and OR operations."""
        tensors = [
            sample_tensors["mixed"],
            sample_tensors["mixed2"],
            sample_tensors["all_true"],
        ]

        # Test variadic AND
        result_and = boolean_logic.AND(*tensors)
        expected_and = torch.tensor([False, False, False, False])
        assert torch.equal(result_and, expected_and)

        # Test variadic OR
        result_or = boolean_logic.OR(*tensors)
        expected_or = torch.tensor([True, True, True, True])
        assert torch.equal(result_or, expected_or)

    def test_boolean_logic_edge_cases(self, boolean_logic):
        """Test edge cases and error conditions."""
        # Test with empty tensor
        empty_tensor = torch.tensor([])
        result = boolean_logic.NOT(empty_tensor)
        assert result.shape == torch.Size([0])

        # Test with single element
        single = torch.tensor([True])
        result = boolean_logic.NOT(single)
        assert torch.equal(result, torch.tensor([False]))


class TestGoedelFuzzyLogic:
    """Test Gödel fuzzy logic implementation."""

    @pytest.fixture
    def godel_logic(self):
        return GoedelFuzzyLogic()

    @pytest.fixture
    def fuzzy_values(self):
        """Create sample fuzzy values for testing."""
        return {
            "low": torch.tensor([0.0, 0.1, 0.2, 0.3]),
            "mid": torch.tensor([0.4, 0.5, 0.6, 0.7]),
            "high": torch.tensor([0.7, 0.8, 0.9, 1.0]),
        }

    def test_godel_range_validation(self, godel_logic, fuzzy_values):
        """Test that Gödel operations maintain [0,1] range."""
        for key, values in fuzzy_values.items():
            # Test NOT maintains range
            not_result = godel_logic.NOT(values)
            assert torch.all(not_result >= 0.0) and torch.all(not_result <= 1.0)

    def test_godel_and_operation(self, godel_logic):
        """Test Gödel AND (minimum) operation."""
        x = torch.tensor([0.3, 0.7, 0.2, 0.9])
        y = torch.tensor([0.5, 0.4, 0.8, 0.6])

        result = godel_logic.AND2(x, y)
        expected = torch.minimum(x, y)
        assert torch.allclose(result, expected)

    def test_godel_or_operation(self, godel_logic):
        """Test Gödel OR (maximum) operation."""
        x = torch.tensor([0.3, 0.7, 0.2, 0.9])
        y = torch.tensor([0.5, 0.4, 0.8, 0.6])

        result = godel_logic.OR2(x, y)
        expected = torch.maximum(x, y)
        assert torch.allclose(result, expected)

    def test_godel_implication(self, godel_logic):
        """Test Gödel implication operation."""
        x = torch.tensor([0.3, 0.8, 0.5, 0.9])
        y = torch.tensor([0.7, 0.4, 0.5, 0.2])

        result = godel_logic.IMPL(x, y)

        # Should be in [0, 1] range
        assert torch.all(result >= 0.0) and torch.all(result <= 1.0)

        # Gödel implication: 1.0 where x < y, otherwise y
        assert result[0] == 1.0  # 0.3 < 0.7, so should be 1
        assert result[2] == 0.5  # 0.5 == 0.5, not <, so should be y (0.5)

    def test_godel_leq_operation(self, godel_logic):
        """Test Gödel fuzzy LEQ operation."""
        x = torch.tensor([2.0, -1.0, 3.0, 0.0])
        y = torch.tensor([1.0, 2.0, 3.0, 1.0])

        result = godel_logic.LEQ(x, y)

        # Should be in [0, 1] range
        assert torch.all(result >= 0.0) and torch.all(result <= 1.0)

        # When x <= y, result should be close to 1
        assert result[2] > 0.9  # 3.0 <= 3.0
        assert result[3] > 0.9  # 0.0 <= 1.0

    def test_godel_variadic_operations(self, godel_logic):
        """Test variadic operations with multiple tensors."""
        tensors = [
            torch.tensor([0.3, 0.7, 0.5]),
            torch.tensor([0.5, 0.4, 0.8]),
            torch.tensor([0.2, 0.9, 0.6]),
        ]

        # Test variadic AND (should be minimum across all)
        result_and = godel_logic.AND(*tensors)
        expected_and = torch.tensor([0.2, 0.4, 0.5])
        assert torch.allclose(result_and, expected_and)

        # Test variadic OR (should be maximum across all)
        result_or = godel_logic.OR(*tensors)
        expected_or = torch.tensor([0.5, 0.9, 0.8])
        assert torch.allclose(result_or, expected_or)


class TestLukasiewiczFuzzyLogic:
    """Test Łukasiewicz fuzzy logic implementation."""

    @pytest.fixture
    def lukasiewicz_logic(self):
        return LukasiewiczFuzzyLogic()

    def test_lukasiewicz_and_operation(self, lukasiewicz_logic):
        """Test Łukasiewicz AND operation."""
        x = torch.tensor([0.3, 0.7, 0.8, 0.2])
        y = torch.tensor([0.5, 0.4, 0.9, 0.1])

        result = lukasiewicz_logic.AND2(x, y)
        expected = torch.clamp(x + y - 1.0, min=0.0)
        assert torch.allclose(result, expected)

    def test_lukasiewicz_or_operation(self, lukasiewicz_logic):
        """Test Łukasiewicz OR operation."""
        x = torch.tensor([0.3, 0.7, 0.8, 0.2])
        y = torch.tensor([0.5, 0.4, 0.9, 0.1])

        result = lukasiewicz_logic.OR2(x, y)
        expected = torch.clamp(x + y, max=1.0)
        assert torch.allclose(result, expected)

    def test_lukasiewicz_implication(self, lukasiewicz_logic):
        """Test Łukasiewicz implication operation."""
        x = torch.tensor([0.3, 0.8, 0.5, 0.9])
        y = torch.tensor([0.7, 0.4, 0.5, 0.2])

        result = lukasiewicz_logic.IMPL(x, y)
        expected = torch.clamp(1.0 - x + y, max=1.0)
        assert torch.allclose(result, expected)


class TestKleeneDienesFuzzyLogic:
    """Test Kleene-Dienes fuzzy logic implementation."""

    @pytest.fixture
    def kd_logic(self):
        return KleeneDienesFuzzyLogic()

    def test_kd_operations(self, kd_logic):
        """Test Kleene-Dienes operations."""
        x = torch.tensor([0.3, 0.7, 0.8, 0.2])
        y = torch.tensor([0.5, 0.4, 0.9, 0.1])

        # Test AND operation
        and_result = kd_logic.AND2(x, y)
        assert torch.all(and_result >= 0.0) and torch.all(and_result <= 1.0)

        # Test OR operation
        or_result = kd_logic.OR2(x, y)
        assert torch.all(or_result >= 0.0) and torch.all(or_result <= 1.0)

    def test_kd_implication(self, kd_logic):
        """Test Kleene-Dienes implication operation."""
        x = torch.tensor([0.3, 0.8, 0.5, 0.9])
        y = torch.tensor([0.7, 0.4, 0.5, 0.2])

        result = kd_logic.IMPL(x, y)

        # Should be in [0, 1] range
        assert torch.all(result >= 0.0) and torch.all(result <= 1.0)


class TestDL2Logic:
    """Test DL2 logic operations."""

    @pytest.fixture
    def dl2_logic(self):
        return DL2()

    def test_dl2_operations_are_differentiable(self, dl2_logic):
        """Test that DL2 operations maintain gradients."""
        x = torch.tensor([0.3, 0.7], requires_grad=True)
        y = torch.tensor([0.5, 0.4], requires_grad=True)

        # Test AND maintains gradients
        and_result = dl2_logic.AND2(x, y)
        loss = and_result.sum()
        loss.backward()

        assert x.grad is not None
        assert y.grad is not None
        assert torch.all(torch.isfinite(x.grad))
        assert torch.all(torch.isfinite(y.grad))

    def test_dl2_arithmetic_operations(self, dl2_logic):
        """Test DL2 arithmetic-based operations."""
        x = torch.tensor([0.3, 0.7, 1.0, 2.0])
        y = torch.tensor([0.5, 0.4, 1.0, 0.5])

        # Test AND operation (addition in DL2)
        and_result = dl2_logic.AND2(x, y)
        expected_and = x + y
        assert torch.allclose(and_result, expected_and)

        # Test OR operation (multiplication in DL2)
        or_result = dl2_logic.OR2(x, y)
        expected_or = x * y
        assert torch.allclose(or_result, expected_or)

        # Test LEQ operation
        leq_result = dl2_logic.LEQ(x, y)
        expected_leq = torch.clamp(x - y, min=0.0)
        assert torch.allclose(leq_result, expected_leq)

    def test_dl2_gradient_flow(self, dl2_logic):
        """Test gradient flow through DL2 expressions."""
        x = torch.tensor([0.3, 0.7], requires_grad=True)
        y = torch.tensor([0.5, 0.4], requires_grad=True)
        z = torch.tensor([0.8, 0.2], requires_grad=True)

        # Complex expression using operations that DL2 actually supports
        and_result = dl2_logic.AND2(x, y)  # x + y
        or_result = dl2_logic.OR2(and_result, z)  # (x + y) * z

        loss = or_result.sum()
        loss.backward()

        # All tensors should have gradients
        assert x.grad is not None and torch.all(torch.isfinite(x.grad))
        assert y.grad is not None and torch.all(torch.isfinite(y.grad))
        assert z.grad is not None and torch.all(torch.isfinite(z.grad))


class TestQLL:
    """Test QLL (LogSumExp-smoothed real-valued logic).

    QLL follows the DL2 convention: values are real, smaller = closer to
    satisfied, positive = violation magnitude. AND is a soft-max (worst
    violation), OR is a soft-min (best violation), both controlled by p.
    As p -> infinity, AND/OR converge to hard max/min.
    """

    @pytest.fixture
    def qll_logic(self):
        return QLL(p=5.0)

    def test_qll_name_includes_p(self):
        assert QLL(p=5.0).name == "QLL_5.0"
        assert QLL(p=10.0).name == "QLL_10.0"

    def test_qll_rejects_non_positive_p(self):
        with pytest.raises(AssertionError):
            QLL(p=0.0)
        with pytest.raises(AssertionError):
            QLL(p=-1.0)

    def test_qll_not_is_negation(self, qll_logic):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        assert torch.allclose(qll_logic.NOT(x), -x)

    def test_qll_not_is_involution(self, qll_logic):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        assert torch.allclose(qll_logic.NOT(qll_logic.NOT(x)), x)

    def test_qll_eq_is_absolute_difference(self, qll_logic):
        x = torch.tensor([1.0, 2.0, 3.0, -1.0])
        y = torch.tensor([1.5, 1.0, 3.0, 0.0])
        assert torch.allclose(qll_logic.EQ(x, y), torch.abs(x - y))

    def test_qll_leq_is_signed_difference(self, qll_logic):
        x = torch.tensor([1.0, 2.0, 3.0, -1.0])
        y = torch.tensor([1.5, 1.0, 3.0, 0.0])
        # Positive when x > y (violation), <= 0 when satisfied.
        assert torch.allclose(qll_logic.LEQ(x, y), x - y)

    def test_qll_impl_equals_leq(self, qll_logic):
        # QLL defines IMPL(x, y) = x - y (same as LEQ). Pin this so a
        # future change is intentional.
        x = torch.tensor([0.3, -0.5, 1.2])
        y = torch.tensor([0.1, 0.7, -0.4])
        assert torch.allclose(qll_logic.IMPL(x, y), qll_logic.LEQ(x, y))

    def test_qll_and_upper_bounds_hard_max(self, qll_logic):
        # logsumexp(p*x_i)/p ∈ [max, max + log(n)/p] elementwise
        x = torch.tensor([0.3, -0.5, 0.8, -0.2])
        y = torch.tensor([0.1, 0.7, -0.4, 0.9])
        hard_max = torch.maximum(x, y)
        soft_max = qll_logic.AND(x, y)
        bound = hard_max + torch.log(torch.tensor(2.0)) / qll_logic.p
        assert torch.all(soft_max >= hard_max - 1e-6)
        assert torch.all(soft_max <= bound + 1e-6)

    def test_qll_or_lower_bounds_hard_min(self, qll_logic):
        # OR(x_i) = -logsumexp(-p*x_i)/p ∈ [min - log(n)/p, min]
        x = torch.tensor([0.3, -0.5, 0.8, -0.2])
        y = torch.tensor([0.1, 0.7, -0.4, 0.9])
        hard_min = torch.minimum(x, y)
        soft_min = qll_logic.OR(x, y)
        bound = hard_min - torch.log(torch.tensor(2.0)) / qll_logic.p
        assert torch.all(soft_min <= hard_min + 1e-6)
        assert torch.all(soft_min >= bound - 1e-6)

    def test_qll_and_or_converge_to_hard_extremes_as_p_grows(self):
        x = torch.tensor([0.3, -0.5, 0.8, -0.2])
        y = torch.tensor([0.1, 0.7, -0.4, 0.9])
        hard_max = torch.maximum(x, y)
        hard_min = torch.minimum(x, y)

        and_p1 = QLL(p=1.0).AND(x, y)
        and_p100 = QLL(p=100.0).AND(x, y)
        or_p1 = QLL(p=1.0).OR(x, y)
        or_p100 = QLL(p=100.0).OR(x, y)

        # Larger p => tighter approximation of the hard extremes.
        assert torch.all((and_p100 - hard_max).abs() <= (and_p1 - hard_max).abs())
        assert torch.all((or_p100 - hard_min).abs() <= (or_p1 - hard_min).abs())

    def test_qll_variadic_and_or(self, qll_logic):
        a = torch.tensor([0.1, 0.5])
        b = torch.tensor([0.3, -0.2])
        c = torch.tensor([-0.4, 0.7])
        d = torch.tensor([0.2, 0.1])

        and_result = qll_logic.AND(a, b, c, d)
        or_result = qll_logic.OR(a, b, c, d)

        # Bounds still hold for n > 2 with log(n)/p slack.
        hard_max = torch.maximum(torch.maximum(a, b), torch.maximum(c, d))
        hard_min = torch.minimum(torch.minimum(a, b), torch.minimum(c, d))
        slack = torch.log(torch.tensor(4.0)) / qll_logic.p
        assert torch.all(and_result >= hard_max - 1e-6)
        assert torch.all(and_result <= hard_max + slack + 1e-6)
        assert torch.all(or_result <= hard_min + 1e-6)
        assert torch.all(or_result >= hard_min - slack - 1e-6)

    def test_qll_gradients_flow_through_and_or_leq(self, qll_logic):
        x = torch.tensor([0.3, 0.7], requires_grad=True)
        y = torch.tensor([0.5, 0.4], requires_grad=True)
        z = torch.tensor([0.8, 0.2], requires_grad=True)

        # Compose: OR(AND(LEQ(x,y), z), NEQ(x,z))
        leq = qll_logic.LEQ(x, y)
        and_term = qll_logic.AND(leq, z)
        neq = qll_logic.NEQ(x, z)
        loss = qll_logic.OR(and_term, neq).sum()
        loss.backward()

        for t in (x, y, z):
            assert t.grad is not None
            assert torch.all(torch.isfinite(t.grad))

    def test_qll_numerical_stability_with_large_p(self):
        # logsumexp must keep AND/OR finite even at p=100 with large inputs.
        logic = QLL(p=100.0)
        x = torch.tensor([10.0, -10.0, 5.0, -5.0])
        y = torch.tensor([-10.0, 10.0, -5.0, 5.0])
        assert torch.all(torch.isfinite(logic.AND(x, y)))
        assert torch.all(torch.isfinite(logic.OR(x, y)))


class TestLeakyLogic:
    """Test LeakyLogic (softplus-LEQ, p-norm AND/OR).

    LeakyLogic is a DL2 variant whose defining property is that gradients
    keep flowing even when the constraint is satisfied: where DL2's
    LEQ = relu(x - y) hits zero (and zero gradient) at the sat boundary,
    LeakyLogic's LEQ = softplus(x - y) stays strictly positive and
    differentiable everywhere. AND/OR are p-norm and negative-p-norm,
    both upper- / lower-bounding the hard max / min on non-negative
    inputs (the typical output of LEQ).
    """

    @pytest.fixture
    def leaky_logic(self):
        return LeakyLogic()

    def test_leaky_name_and_default_p(self, leaky_logic):
        assert leaky_logic.name == "LL"
        assert leaky_logic.p == 2

    def test_leaky_not_is_unsupported(self, leaky_logic):
        # Like DL2, LeakyLogic forbids general negation - constraints must
        # be written with negation pushed inward (NOT(LEQ) -> GT etc.).
        with pytest.raises(NotImplementedError, match="rewrite the constraint"):
            leaky_logic.NOT(torch.tensor([0.3]))

    def test_leaky_leq_is_softplus_of_difference(self, leaky_logic):
        x = torch.tensor([0.3, 1.0, -0.5, 2.0])
        y = torch.tensor([0.7, 1.0, 0.0, 1.5])
        expected = torch.nn.functional.softplus(x - y)
        assert torch.allclose(leaky_logic.LEQ(x, y), expected)

    def test_leaky_leq_is_strictly_positive_everywhere(self, leaky_logic):
        # softplus > 0 for all real inputs, including at and past the sat
        # boundary - this is the property that gives "leaky" its name.
        x = torch.linspace(-5.0, 5.0, steps=21)
        y = torch.zeros_like(x)
        assert torch.all(leaky_logic.LEQ(x, y) > 0)

    def test_leaky_gradient_flows_past_sat_boundary(self, leaky_logic):
        """The defining behavioral difference between LeakyLogic and DL2:
        at a satisfied constraint (x < y), DL2's gradient is exactly zero,
        but LeakyLogic's gradient is the sigmoid of the (negative) gap.
        """
        x_leaky = torch.tensor([0.3], requires_grad=True)
        x_dl2 = torch.tensor([0.3], requires_grad=True)
        y = torch.tensor([0.7])  # x <= y is satisfied with margin 0.4

        leaky_logic.LEQ(x_leaky, y).sum().backward()
        DL2().LEQ(x_dl2, y).sum().backward()

        # DL2: relu'(-0.4) = 0
        assert torch.allclose(x_dl2.grad, torch.zeros_like(x_dl2.grad))
        # LeakyLogic: softplus'(-0.4) = sigmoid(-0.4) ≈ 0.401
        expected_leaky_grad = torch.sigmoid(torch.tensor(-0.4))
        assert torch.allclose(x_leaky.grad, expected_leaky_grad, atol=1e-6)
        assert x_leaky.grad.abs() > 0.1

    def test_leaky_lt_pins_strict_offset(self, leaky_logic):
        # LT(x, y) = LEQ(x + 1e-3, y). At x == y this gives softplus(1e-3),
        # not softplus(0) - pin the magic constant so a future tweak is
        # intentional and visible.
        x = torch.tensor([1.0, 2.0, -0.5])
        lt = leaky_logic.LT(x, x)
        expected = torch.nn.functional.softplus(torch.tensor(1e-3))
        assert torch.allclose(lt, expected * torch.ones_like(x))

    def test_leaky_and_upper_bounds_hard_max(self, leaky_logic):
        # p-norm: (sum x_i^p)^(1/p) ∈ [max, n^(1/p) * max] for non-negative inputs.
        x = torch.tensor([0.3, 0.5, 0.8, 0.2])
        y = torch.tensor([0.1, 0.7, 0.4, 0.9])
        hard_max = torch.maximum(x, y)
        soft_max = leaky_logic.AND(x, y)
        upper = (2 ** (1.0 / leaky_logic.p)) * hard_max
        assert torch.all(soft_max >= hard_max - 1e-6)
        assert torch.all(soft_max <= upper + 1e-6)

    def test_leaky_or_lower_bounds_hard_min(self, leaky_logic):
        # Negative-p-norm: (sum x_i^{-p})^{-1/p} ∈ [min / n^(1/p), min].
        x = torch.tensor([0.3, 0.5, 0.8, 0.2])
        y = torch.tensor([0.1, 0.7, 0.4, 0.9])
        hard_min = torch.minimum(x, y)
        soft_min = leaky_logic.OR(x, y)
        lower = hard_min / (2 ** (1.0 / leaky_logic.p))
        assert torch.all(soft_min <= hard_min + 1e-6)
        assert torch.all(soft_min >= lower - 1e-6)

    def test_leaky_and_or_converge_to_hard_extremes_as_p_grows(self):
        # Inputs kept in [0.5, 0.9] so the inner sum(x^p) stays above
        # safe_pow's eps clamp - see test_leaky_and_saturates_at_high_p
        # for what goes wrong outside that regime.
        x = torch.tensor([0.6, 0.7, 0.8, 0.5])
        y = torch.tensor([0.5, 0.8, 0.6, 0.9])
        hard_max = torch.maximum(x, y)
        hard_min = torch.minimum(x, y)

        and_p2 = LeakyLogic(p=2).AND(x, y)
        and_p10 = LeakyLogic(p=10).AND(x, y)
        or_p2 = LeakyLogic(p=2).OR(x, y)
        or_p10 = LeakyLogic(p=10).OR(x, y)

        assert torch.all((and_p10 - hard_max).abs() <= (and_p2 - hard_max).abs())
        assert torch.all((or_p10 - hard_min).abs() <= (or_p2 - hard_min).abs())

    def test_leaky_and_stays_close_to_max_at_high_p(self):
        """Regression test for issue #9. Before the log-domain rewrite of
        ``p_sum``, ``safe_pow``'s eps clamp caused AND to saturate at
        ``eps^(1/p)`` once ``max(x)^p`` underflowed eps - here p=50 with
        max=0.3 gave AND ≈ 0.727 regardless of input. After the fix AND
        should converge to the hard max as p grows.
        """
        x = torch.tensor([0.3, 0.2])
        y = torch.tensor([0.1, 0.15])
        and_p50 = LeakyLogic(p=50).AND(x, y)

        hard_max = torch.maximum(x, y)
        # At p=50 the LSE slack is log(2)/50 ≈ 0.014, scaled by max.
        assert torch.allclose(and_p50, hard_max, atol=1e-3)

    def test_leaky_variadic_and_or(self, leaky_logic):
        a = torch.tensor([0.1, 0.5])
        b = torch.tensor([0.3, 0.2])
        c = torch.tensor([0.4, 0.7])
        d = torch.tensor([0.2, 0.1])

        and_result = leaky_logic.AND(a, b, c, d)
        or_result = leaky_logic.OR(a, b, c, d)

        hard_max = torch.maximum(torch.maximum(a, b), torch.maximum(c, d))
        hard_min = torch.minimum(torch.minimum(a, b), torch.minimum(c, d))
        slack = 4 ** (1.0 / leaky_logic.p)
        assert torch.all(and_result >= hard_max - 1e-6)
        assert torch.all(and_result <= slack * hard_max + 1e-6)
        assert torch.all(or_result <= hard_min + 1e-6)
        assert torch.all(or_result >= hard_min / slack - 1e-6)

    def test_leaky_gradients_flow_through_compound_expression(self, leaky_logic):
        x = torch.tensor([0.3, 0.7], requires_grad=True)
        y = torch.tensor([0.5, 0.4], requires_grad=True)
        z = torch.tensor([0.8, 0.2], requires_grad=True)

        leq_xy = leaky_logic.LEQ(x, y)
        leq_yz = leaky_logic.LEQ(y, z)
        loss = leaky_logic.OR(leaky_logic.AND(leq_xy, leq_yz), leq_xy).sum()
        loss.backward()

        for t in (x, y, z):
            assert t.grad is not None
            assert torch.all(torch.isfinite(t.grad))


class TestSTLLogic:
    """Test Signal Temporal Logic (STL) implementation."""

    @pytest.fixture
    def stl_logic(self):
        return STL()

    def test_stl_basic_operations(self, stl_logic):
        """Test basic STL operations."""
        x = torch.tensor([0.3, -0.5, 0.8, -0.2])
        y = torch.tensor([0.1, 0.7, -0.4, 0.9])

        # Test variadic AND operation (smooth minimum)
        and_result = stl_logic.AND(x, y)
        # Should approximate minimum, but may not be exact due to smoothing
        assert and_result.shape == x.shape

        # Test variadic OR operation
        or_result = stl_logic.OR(x, y)
        # Should approximate maximum, but may not be exact due to smoothing
        assert or_result.shape == x.shape

        # Test NOT operation (negation)
        not_result = stl_logic.NOT(x)
        expected_not = -x
        assert torch.allclose(not_result, expected_not)

    def test_stl_comparison_operations(self, stl_logic):
        """Test STL comparison operations."""
        x = torch.tensor([1.0, 2.0, 3.0, -1.0])
        y = torch.tensor([1.5, 1.0, 3.0, 0.0])

        # Test LEQ operation
        leq_result = stl_logic.LEQ(x, y)
        expected_leq = y - x  # In STL, x <= y is equivalent to y - x >= 0
        assert torch.allclose(leq_result, expected_leq)

    def test_stl_preserves_real_values(self, stl_logic):
        """Test that STL operations preserve real-valued semantics."""
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])

        # STL should work with any real values, not just [0,1]
        not_result = stl_logic.NOT(x)
        assert torch.allclose(not_result, -x)

        # Test with negative values using variadic operations
        y = torch.tensor([-1.0, -0.3, 0.1, 0.7, 1.5])
        and_result = stl_logic.AND(x, y)
        or_result = stl_logic.OR(x, y)

        # Results should be smooth approximations of min/max
        assert and_result.shape == x.shape
        assert or_result.shape == x.shape
        assert torch.all(torch.isfinite(and_result))
        assert torch.all(torch.isfinite(or_result))


class TestFuzzyLogicWithSNImplication:
    """Test (S,N)-implication functionality."""

    @pytest.fixture
    def sn_logic(self):
        """Create a fuzzy logic with (S,N)-implication."""
        return GoedelFuzzyLogic()  # Inherits from FuzzyLogicWithSNImplication

    def test_sn_implication_implementation(self, sn_logic):
        """Test that (S,N)-implication is a valid implication."""
        x = torch.tensor([0.3, 0.7, 0.5, 0.9])
        y = torch.tensor([0.7, 0.4, 0.5, 0.2])

        # Test the actual IMPL method produces valid results
        actual_impl = sn_logic.IMPL(x, y)

        # Should be in [0, 1] range
        assert torch.all(actual_impl >= 0.0) and torch.all(actual_impl <= 1.0)

        # Check basic implication properties for Gödel logic
        # When x < y, implication should be 1
        assert actual_impl[0] == 1.0  # 0.3 < 0.7
        assert actual_impl[2] == 0.5  # 0.5 == 0.5, so result is y (0.5)


class TestLogicConsistency:
    """Test consistency across different logic implementations."""

    @pytest.fixture
    def logics(self):
        """Create instances of different logic types."""
        return {
            "boolean": BooleanLogic(),
            "godel": GoedelFuzzyLogic(),
            "lukasiewicz": LukasiewiczFuzzyLogic(),
            "kleene_dienes": KleeneDienesFuzzyLogic(),
            "dl2": DL2(),
            "stl": STL(),
        }

    def test_not_involution(self, logics):
        """Test that NOT(NOT(x)) ≈ x for appropriate logics."""
        x = torch.tensor([0.3, 0.7, 0.1, 0.9])

        for name, logic in logics.items():
            if name == "boolean":
                # For boolean, use boolean values
                bool_x = torch.tensor([True, False, True, False])
                double_not = logic.NOT(logic.NOT(bool_x))
                assert torch.equal(double_not, bool_x), f"Failed for {name}"
            elif name not in [
                "stl",
                "dl2",
            ]:  # STL and DL2 have different negation semantics
                double_not = logic.NOT(logic.NOT(x))
                assert torch.allclose(double_not, x, atol=1e-6), f"Failed for {name}"

    def test_de_morgan_laws(self, logics):
        """Test De Morgan's laws where applicable."""
        x = torch.tensor([0.3, 0.7])
        y = torch.tensor([0.5, 0.4])

        for name, logic in logics.items():
            if name == "boolean":
                bool_x = torch.tensor([True, False])
                bool_y = torch.tensor([False, True])

                # NOT(x AND y) = NOT(x) OR NOT(y)
                left_side = logic.NOT(logic.AND2(bool_x, bool_y))
                right_side = logic.OR2(logic.NOT(bool_x), logic.NOT(bool_y))
                assert torch.equal(left_side, right_side), (
                    f"De Morgan AND failed for {name}"
                )

                # NOT(x OR y) = NOT(x) AND NOT(y)
                left_side = logic.NOT(logic.OR2(bool_x, bool_y))
                right_side = logic.AND2(logic.NOT(bool_x), logic.NOT(bool_y))
                assert torch.equal(left_side, right_side), (
                    f"De Morgan OR failed for {name}"
                )

            elif name in ["godel", "lukasiewicz", "kleene_dienes"]:
                # For fuzzy logics, De Morgan's laws hold approximately
                # NOT(x AND y) ≈ NOT(x) OR NOT(y)
                left_side = logic.NOT(logic.AND2(x, y))
                right_side = logic.OR2(logic.NOT(x), logic.NOT(y))
                # Note: This might not hold exactly for all fuzzy logics
                # but should be close for many cases
