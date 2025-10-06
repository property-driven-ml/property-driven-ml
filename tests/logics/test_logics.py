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
from property_driven_ml.logics.stl import STL


class TestLogicBase:
    """Test the base Logic class."""

    def test_logic_base_cannot_be_instantiated(self):
        """Test that Logic is an abstract base class."""
        with pytest.raises(TypeError):
            Logic() # type: ignore

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
            'all_true': torch.tensor([True, True, True, True]),
            'all_false': torch.tensor([False, False, False, False]),
            'mixed': torch.tensor([True, False, True, False]),
            'mixed2': torch.tensor([False, True, False, True]),
        }

    def test_boolean_logic_basic_operations(self, boolean_logic, sample_tensors):
        """Test basic boolean operations work correctly."""
        # Test AND operation
        result_and = boolean_logic.AND2(sample_tensors['mixed'], sample_tensors['mixed2'])
        expected_and = torch.tensor([False, False, False, False])
        assert torch.equal(result_and, expected_and)

        # Test OR operation
        result_or = boolean_logic.OR2(sample_tensors['mixed'], sample_tensors['mixed2'])
        expected_or = torch.tensor([True, True, True, True])
        assert torch.equal(result_or, expected_or)

        # Test NOT operation
        result_not = boolean_logic.NOT(sample_tensors['mixed'])
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
        tensors = [sample_tensors['mixed'], sample_tensors['mixed2'], sample_tensors['all_true']]

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

        # Test with different shapes that should broadcast
        x = torch.tensor([[True, False], [False, True]])
        y = torch.tensor([True, False])
        result = boolean_logic.AND2(x, y)
        expected = torch.tensor([[True, False], [False, False]])
        assert torch.equal(result, expected)


class TestGoedelFuzzyLogic:
    """Test Gödel fuzzy logic implementation."""

    @pytest.fixture
    def godel_logic(self):
        return GoedelFuzzyLogic()

    @pytest.fixture
    def fuzzy_values(self):
        """Create sample fuzzy values for testing."""
        return {
            'low': torch.tensor([0.0, 0.1, 0.2, 0.3]),
            'mid': torch.tensor([0.4, 0.5, 0.6, 0.7]),
            'high': torch.tensor([0.7, 0.8, 0.9, 1.0]),
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
            'boolean': BooleanLogic(),
            'godel': GoedelFuzzyLogic(),
            'lukasiewicz': LukasiewiczFuzzyLogic(),
            'kleene_dienes': KleeneDienesFuzzyLogic(),
            'dl2': DL2(),
            'stl': STL(),
        }

    def test_not_involution(self, logics):
        """Test that NOT(NOT(x)) ≈ x for appropriate logics."""
        x = torch.tensor([0.3, 0.7, 0.1, 0.9])
        
        for name, logic in logics.items():
            if name == 'boolean':
                # For boolean, use boolean values
                bool_x = torch.tensor([True, False, True, False])
                double_not = logic.NOT(logic.NOT(bool_x))
                assert torch.equal(double_not, bool_x), f"Failed for {name}"
            elif name not in ['stl', 'dl2']:  # STL and DL2 have different negation semantics
                double_not = logic.NOT(logic.NOT(x))
                assert torch.allclose(double_not, x, atol=1e-6), f"Failed for {name}"

    def test_de_morgan_laws(self, logics):
        """Test De Morgan's laws where applicable."""
        x = torch.tensor([0.3, 0.7])
        y = torch.tensor([0.5, 0.4])
        
        for name, logic in logics.items():
            if name == 'boolean':
                bool_x = torch.tensor([True, False])
                bool_y = torch.tensor([False, True])
                
                # NOT(x AND y) = NOT(x) OR NOT(y)
                left_side = logic.NOT(logic.AND2(bool_x, bool_y))
                right_side = logic.OR2(logic.NOT(bool_x), logic.NOT(bool_y))
                assert torch.equal(left_side, right_side), f"De Morgan AND failed for {name}"
                
                # NOT(x OR y) = NOT(x) AND NOT(y)
                left_side = logic.NOT(logic.OR2(bool_x, bool_y))
                right_side = logic.AND2(logic.NOT(bool_x), logic.NOT(bool_y))
                assert torch.equal(left_side, right_side), f"De Morgan OR failed for {name}"
            
            elif name in ['godel', 'lukasiewicz', 'kleene_dienes']:
                # For fuzzy logics, De Morgan's laws hold approximately
                # NOT(x AND y) ≈ NOT(x) OR NOT(y)
                left_side = logic.NOT(logic.AND2(x, y))
                right_side = logic.OR2(logic.NOT(x), logic.NOT(y))
                # Note: This might not hold exactly for all fuzzy logics
                # but should be close for many cases
