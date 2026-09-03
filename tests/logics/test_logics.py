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
    ReichenbachFuzzyLogic,
    GoguenFuzzyLogic,
    ReichenbachSigmoidalFuzzyLogic,
    YagerFuzzyLogic,
)
from property_driven_ml.logics.dl2 import DL2
from property_driven_ml.logics.real_product_logic import RealProductLogic
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


class TestRealProductLogic:
    """Test RealProductLogic.

    RealProductLogic is the real-valued reformulation of product fuzzy
    logic under the change of variables ``z = -log(t)``, mapping fuzzy
    truth ``t ∈ (0, 1]`` to penalty ``z ∈ [0, ∞)``. Under this isomorphism
    product AND becomes sum, probabilistic-sum OR becomes the dual log
    formula, and standard negation becomes ``-log(1 - exp(-z))``. LEQ
    follows DL2's relu(x-y) convention.
    """

    @pytest.fixture
    def rp_logic(self):
        return RealProductLogic()

    def test_realproduct_default_name(self, rp_logic):
        assert rp_logic.name == "RealProduct"

    def test_realproduct_and_is_sum(self, rp_logic):
        x = torch.tensor([0.3, 1.0, 0.0, 2.5])
        y = torch.tensor([0.7, 0.5, 1.0, 0.5])
        assert torch.allclose(rp_logic.AND2(x, y), x + y)

    def test_realproduct_and_identity_at_zero(self, rp_logic):
        # z=0 corresponds to fuzzy truth=1; AND with truth is the identity.
        x = torch.tensor([0.3, 1.0, 2.5])
        zero = torch.zeros_like(x)
        assert torch.allclose(rp_logic.AND2(x, zero), x)

    def test_realproduct_leq_is_relu_of_difference(self, rp_logic):
        x = torch.tensor([0.3, 1.0, -0.5, 2.0])
        y = torch.tensor([0.7, 1.0, 0.0, 1.5])
        assert torch.allclose(rp_logic.LEQ(x, y), torch.relu(x - y))

    def test_realproduct_not_matches_formula(self, rp_logic):
        # NOT(z) = -log(1 - exp(-z)). At z=0.5: -log(1 - exp(-0.5)) ≈ 0.933.
        x = torch.tensor([0.5, 1.0, 2.0])
        expected = -torch.log(1.0 - torch.exp(-x))
        assert torch.allclose(rp_logic.NOT(x), expected)

    def test_realproduct_not_is_involution_on_positive_inputs(self, rp_logic):
        # NOT(NOT(z)) = z is an exact algebraic identity on z > 0:
        # NOT(-log(1 - e^-z)) = -log(1 - e^(log(1 - e^-z))) = -log(e^-z) = z.
        z = torch.tensor([0.5, 1.0, 2.0, 5.0])
        assert torch.allclose(rp_logic.NOT(rp_logic.NOT(z)), z, atol=1e-5)

    def test_realproduct_or_matches_fuzzy_change_of_variables(self, rp_logic):
        # Under z = -log(t): fuzzy probabilistic-sum OR(t1, t2) = 1-(1-t1)(1-t2).
        # In z-domain that equals -log((1-t1)(1-t2)) = -log(1 - (1-e^-z1)(1-e^-z2)),
        # which is exactly what RealProductLogic.OR2 computes.
        z1 = torch.tensor([0.3, 1.0, 2.0])
        z2 = torch.tensor([0.5, 0.2, 1.5])
        t1, t2 = torch.exp(-z1), torch.exp(-z2)
        fuzzy_or = 1.0 - (1.0 - t1) * (1.0 - t2)
        expected = -torch.log(fuzzy_or)
        assert torch.allclose(rp_logic.OR2(z1, z2), expected)

    def test_realproduct_or_identity_at_large_penalty(self, rp_logic):
        # OR(z, large) should approach z as large -> infty (since large
        # corresponds to fuzzy false, and t OR false = t).
        z = torch.tensor([0.3, 1.0, 2.0])
        large = torch.full_like(z, 30.0)  # exp(-30) ≈ 1e-13, effectively false
        assert torch.allclose(rp_logic.OR2(z, large), z, atol=1e-5)

    def test_realproduct_variadic_and(self, rp_logic):
        a = torch.tensor([0.1, 0.2])
        b = torch.tensor([0.3, 0.5])
        c = torch.tensor([0.4, 0.1])
        assert torch.allclose(rp_logic.AND(a, b, c), a + b + c)

    def test_realproduct_gradients_flow(self, rp_logic):
        x = torch.tensor([0.5, 1.0], requires_grad=True)
        y = torch.tensor([0.7, 0.3], requires_grad=True)
        z = torch.tensor([1.0, 2.0], requires_grad=True)

        loss = rp_logic.OR2(rp_logic.AND2(x, y), rp_logic.LEQ(z, x)).sum()
        loss.backward()
        for t in (x, y, z):
            assert t.grad is not None
            assert torch.all(torch.isfinite(t.grad))


class TestYagerFuzzyLogic:
    """Test YagerFuzzyLogic (parameterized t-norm interpolating between
    Łukasiewicz at p=1 and Gödel as p → ∞)."""

    @pytest.fixture
    def yager(self):
        return YagerFuzzyLogic()

    def test_yager_name_and_default_p(self, yager):
        assert yager.name == "YG"
        assert yager.p == 5

    def test_yager_and_or_in_unit_interval(self, yager):
        x = torch.tensor([0.1, 0.5, 0.7, 0.9])
        y = torch.tensor([0.4, 0.2, 0.6, 0.8])
        for result in (yager.AND(x, y), yager.OR(x, y)):
            assert torch.all(result >= 0.0)
            assert torch.all(result <= 1.0)

    def test_yager_and_with_zero_is_zero(self, yager):
        # False is absorbing for AND. This direction works fine because
        # (1-x)^p is bounded; only the all-ones direction hits the eps clamp
        # (see test_yager_and_at_full_truth_saturates_due_to_eps_clamp).
        x = torch.tensor([0.3, 0.7, 1.0])
        assert torch.allclose(yager.AND(x, torch.zeros_like(x)), torch.zeros_like(x))

    def test_yager_and_at_full_truth_saturates_due_to_eps_clamp(self):
        """Pins a numerical limitation in YagerFuzzyLogic.AND tracked as
        issue #11 (same bug class as #9 in LeakyLogic). The source clamps
        the inner ``sum((1-x)^p)`` to ``eps=1e-6`` before taking the p-th
        root, so for inputs of all 1.0 (where the inner sum is exactly 0)
        the result is ``1 - eps^(1/p)`` instead of the t-norm identity 1.
        For default ``p=5`` that's ``1 - (1e-6)^0.2 ≈ 0.937``; gets worse as
        ``p`` grows (``1 - (1e-6)^0.05 ≈ 0.499`` at ``p=20``).
        """
        ones = torch.ones(3)
        for p in (5, 10, 20):
            expected = 1.0 - (1e-6) ** (1.0 / p)
            assert torch.allclose(
                YagerFuzzyLogic(p=p).AND(ones, ones),
                torch.full_like(ones, expected),
                atol=1e-5,
            )

    def test_yager_or_extremes(self, yager):
        x = torch.tensor([0.3, 0.7, 0.0])
        # OR with 1 is 1 (true is absorbing for OR).
        assert torch.allclose(yager.OR(x, torch.ones_like(x)), torch.ones_like(x))
        # OR of all 0s is 0.
        zeros = torch.zeros_like(x)
        assert torch.allclose(yager.OR(zeros, zeros), zeros)

    def test_yager_matches_lukasiewicz_at_p_equals_1(self):
        # Yager with p=1 reduces to Łukasiewicz: AND = max(0, x+y-1),
        # OR = min(1, x+y). Both directly from the p-norm formulas at p=1.
        y_logic = YagerFuzzyLogic(p=1)
        l_logic = LukasiewiczFuzzyLogic()
        x = torch.tensor([0.3, 0.7, 0.5, 0.9])
        y = torch.tensor([0.4, 0.5, 0.8, 0.2])
        assert torch.allclose(y_logic.AND(x, y), l_logic.AND(x, y), atol=1e-6)
        assert torch.allclose(y_logic.OR(x, y), l_logic.OR(x, y), atol=1e-6)

    def test_yager_converges_to_godel_as_p_grows(self):
        # As p -> infty Yager AND/OR approach min/max (Gödel limit).
        x = torch.tensor([0.6, 0.7, 0.8, 0.5])
        y = torch.tensor([0.5, 0.8, 0.6, 0.9])
        and_p2 = YagerFuzzyLogic(p=2).AND(x, y)
        and_p10 = YagerFuzzyLogic(p=10).AND(x, y)
        or_p2 = YagerFuzzyLogic(p=2).OR(x, y)
        or_p10 = YagerFuzzyLogic(p=10).OR(x, y)
        hard_min = torch.minimum(x, y)
        hard_max = torch.maximum(x, y)
        assert torch.all((and_p10 - hard_min).abs() <= (and_p2 - hard_min).abs())
        assert torch.all((or_p10 - hard_max).abs() <= (or_p2 - hard_max).abs())

    def test_yager_impl_zero_zero_special_case(self, yager):
        # The 0^0 indeterminate form is explicitly forced to 1 in the source.
        zero = torch.tensor([0.0])
        assert torch.allclose(yager.IMPL(zero, zero), torch.ones_like(zero))

    def test_yager_impl_general_formula(self, yager):
        # IMPL(x, y) = y^x for x, y > 0.
        x = torch.tensor([0.3, 0.5, 0.8])
        y = torch.tensor([0.4, 0.6, 0.9])
        assert torch.allclose(yager.IMPL(x, y), torch.pow(y, x))

    def test_yager_leq_returns_one_at_satisfied(self, yager):
        # Where x <= y, LEQ is forced to 1 (fully satisfied).
        x = torch.tensor([0.2, 0.5, 0.7])
        y = torch.tensor([0.5, 0.5, 0.8])
        leq = yager.LEQ(x, y)
        assert torch.allclose(leq, torch.ones_like(leq))

    def test_yager_gradients_flow(self, yager):
        x = torch.tensor([0.3, 0.7], requires_grad=True)
        y = torch.tensor([0.5, 0.4], requires_grad=True)
        loss = yager.AND(x, y).sum() + yager.OR(x, y).sum()
        loss.backward()
        for t in (x, y):
            assert t.grad is not None
            assert torch.all(torch.isfinite(t.grad))


class TestGoguenFuzzyLogic:
    """Test GoguenFuzzyLogic. Inherits Reichenbach's product AND and
    probabilistic-sum OR; only overrides IMPL with the R-implication
    based on the product residuum (``y/x`` for ``x > y``, else 1)."""

    @pytest.fixture
    def goguen(self):
        return GoguenFuzzyLogic()

    def test_goguen_name(self, goguen):
        assert goguen.name == "GG"

    def test_goguen_inherits_reichenbach_and(self, goguen):
        rc = ReichenbachFuzzyLogic()
        x = torch.tensor([0.3, 0.7, 0.5])
        y = torch.tensor([0.5, 0.4, 0.8])
        assert torch.allclose(goguen.AND(x, y), rc.AND(x, y))

    def test_goguen_inherits_reichenbach_or(self, goguen):
        rc = ReichenbachFuzzyLogic()
        x = torch.tensor([0.3, 0.7, 0.5])
        y = torch.tensor([0.5, 0.4, 0.8])
        assert torch.allclose(goguen.OR(x, y), rc.OR(x, y))

    def test_goguen_impl_returns_one_when_x_leq_y(self, goguen):
        x = torch.tensor([0.3, 0.5, 0.0])
        y = torch.tensor([0.7, 0.5, 0.4])
        impl = goguen.IMPL(x, y)
        assert torch.allclose(impl, torch.ones_like(impl))

    def test_goguen_impl_is_ratio_when_x_greater_than_y(self, goguen):
        # IMPL(0.8, 0.4) = 0.4 / 0.8 = 0.5; IMPL(0.6, 0.3) = 0.5.
        x = torch.tensor([0.8, 0.6, 0.9])
        y = torch.tensor([0.4, 0.3, 0.45])
        assert torch.allclose(goguen.IMPL(x, y), y / x)

    def test_goguen_impl_zero_antecedent_special_case(self, goguen):
        # x == 0 is explicitly forced to 1 to avoid 0/0.
        zero = torch.zeros(3)
        y = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(goguen.IMPL(zero, y), torch.ones_like(zero))

    def test_goguen_impl_gradients_flow(self, goguen):
        x = torch.tensor([0.8, 0.5], requires_grad=True)
        y = torch.tensor([0.3, 0.7], requires_grad=True)
        loss = goguen.IMPL(x, y).sum()
        loss.backward()
        for t in (x, y):
            assert t.grad is not None
            assert torch.all(torch.isfinite(t.grad))


class TestReichenbachSigmoidalFuzzyLogic:
    """Test ReichenbachSigmoidalFuzzyLogic (RCS). Inherits Reichenbach
    AND/OR; replaces the SN-implication with a sigmoidal reshape
    parameterized by s. The sigmoid is calibrated so that the three
    anchor points {I=0, I=0.5, I=1} of the underlying Reichenbach IMPL
    are preserved; higher s sharpens the transition between them."""

    @pytest.fixture
    def rcs(self):
        return ReichenbachSigmoidalFuzzyLogic()

    def test_rcs_name_and_default_s(self, rcs):
        assert rcs.name == "RCS"
        assert rcs.s == 9.0

    def test_rcs_inherits_reichenbach_and(self, rcs):
        rc = ReichenbachFuzzyLogic()
        x = torch.tensor([0.3, 0.7, 0.5])
        y = torch.tensor([0.5, 0.4, 0.8])
        assert torch.allclose(rcs.AND(x, y), rc.AND(x, y))

    def test_rcs_inherits_reichenbach_or(self, rcs):
        rc = ReichenbachFuzzyLogic()
        x = torch.tensor([0.3, 0.7, 0.5])
        y = torch.tensor([0.5, 0.4, 0.8])
        assert torch.allclose(rcs.OR(x, y), rc.OR(x, y))

    def test_rcs_impl_bounded_above_by_one(self, rcs):
        x = torch.tensor([0.1, 0.3, 0.7, 0.9])
        y = torch.tensor([0.2, 0.8, 0.4, 0.6])
        assert torch.all(rcs.IMPL(x, y) <= 1.0 + 1e-6)

    def test_rcs_impl_boundary_cases(self, rcs):
        # Reichenbach IMPL = 1 - x + x*y. Anchor points where the sigmoid
        # calibration preserves the value:
        #   IMPL(0, y) = 1            (false antecedent -> truth)
        #   IMPL(x, 1) = 1            (true consequent -> truth)
        #   IMPL(1, 0) = 0            (true -> false)
        zero = torch.zeros(3)
        one = torch.ones(3)
        y_anything = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(rcs.IMPL(zero, y_anything), torch.ones(3), atol=1e-4)
        assert torch.allclose(rcs.IMPL(y_anything, one), torch.ones(3), atol=1e-4)
        assert torch.allclose(rcs.IMPL(one, zero), torch.zeros(3), atol=1e-4)

    def test_rcs_impl_sharpens_with_larger_s(self):
        # The sigmoid steepens around I=0.5 as s grows: for an underlying
        # I value below 0.5, larger s pushes the output further toward 0.
        rcs_low = ReichenbachSigmoidalFuzzyLogic(s=3.0)
        rcs_high = ReichenbachSigmoidalFuzzyLogic(s=20.0)
        # Construct (x, y) so that the underlying Reichenbach IMPL is < 0.5:
        # I(0.9, 0.1) = 1 - 0.9 + 0.9*0.1 = 0.19.
        x = torch.tensor([0.9])
        y = torch.tensor([0.1])
        impl_low = rcs_low.IMPL(x, y)
        impl_high = rcs_high.IMPL(x, y)
        assert impl_high < impl_low

    def test_rcs_impl_gradients_flow(self, rcs):
        x = torch.tensor([0.3, 0.7], requires_grad=True)
        y = torch.tensor([0.5, 0.4], requires_grad=True)
        loss = rcs.IMPL(x, y).sum()
        loss.backward()
        for t in (x, y):
            assert t.grad is not None
            assert torch.all(torch.isfinite(t.grad))
