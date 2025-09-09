"""
Simplified tests for logic implementations - focusing on essential functionality only.
"""

import pytest
import torch

from property_driven_ml.logics import Logic
from property_driven_ml.logics.boolean_logic import BooleanLogic
from property_driven_ml.logics.fuzzy_logics import FuzzyLogics
from property_driven_ml.logics.dl2 import DL2


class TestLogicBase:
    """Test the base Logic class."""

    def test_logic_base_cannot_be_instantiated(self):
        """Test that Logic is an abstract base class."""
        with pytest.raises(TypeError):
            Logic()


class TestBooleanLogic:
    """Test Boolean logic implementation."""

    @pytest.fixture
    def boolean_logic(self):
        return BooleanLogic()

    def test_boolean_logic_basic_operations(self, boolean_logic):
        """Test basic boolean operations work correctly."""
        # Test AND operation
        result_and = boolean_logic.AND(
            torch.tensor([True, True, False, False]),
            torch.tensor([True, False, True, False]),
        )
        expected_and = torch.tensor([True, False, False, False])
        assert torch.equal(result_and, expected_and)

        # Test OR operation
        result_or = boolean_logic.OR(
            torch.tensor([True, True, False, False]),
            torch.tensor([True, False, True, False]),
        )
        expected_or = torch.tensor([True, True, True, False])
        assert torch.equal(result_or, expected_or)

        # Test NOT operation
        result_not = boolean_logic.NOT(torch.tensor([True, False, True, False]))
        expected_not = torch.tensor([False, True, False, True])
        assert torch.equal(result_not, expected_not)


class TestFuzzyLogic:
    """Test Fuzzy logic implementation."""

    @pytest.fixture
    def fuzzy_logic(self):
        return FuzzyLogics()

    def test_fuzzy_logic_range_validation(self, fuzzy_logic):
        """Test that fuzzy logic operations maintain [0,1] range."""
        # Test with values in [0,1] range
        x = torch.tensor([0.0, 0.3, 0.7, 1.0])

        # Test NOT maintains range
        not_result = fuzzy_logic.NOT(x)
        assert torch.all(not_result >= 0.0) and torch.all(not_result <= 1.0)

        # Test AND maintains range
        y = torch.tensor([0.2, 0.5, 0.8, 0.9])
        and_result = fuzzy_logic.AND([x, y])
        assert torch.all(and_result >= 0.0) and torch.all(and_result <= 1.0)

        # Test OR maintains range
        or_result = fuzzy_logic.OR([x, y])
        assert torch.all(or_result >= 0.0) and torch.all(or_result <= 1.0)


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
        and_result = dl2_logic.AND([x, y])
        loss = and_result.sum()
        loss.backward()

        assert x.grad is not None
        assert y.grad is not None
        assert torch.all(torch.isfinite(x.grad))
        assert torch.all(torch.isfinite(y.grad))
