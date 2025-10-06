"""
Tests for utility functions and safe evaluation.
"""

import pytest
import torch

from property_driven_ml.utils.safe_eval import safe_call
from property_driven_ml.constraints import StandardRobustnessConstraint


class TestSafeEval:
    """Test safe constraint evaluation utility."""

    def test_safe_call_with_valid_constraint(self):
        """Test safe_call with a valid constraint name."""
        allowed_constraints = {
            "StandardRobustness": StandardRobustnessConstraint,
        }

        result = safe_call("StandardRobustness", allowed_constraints)
        assert result is StandardRobustnessConstraint

    def test_safe_call_with_invalid_constraint(self):
        """Test safe_call raises error for invalid constraint name."""
        allowed_constraints = {
            "StandardRobustness": StandardRobustnessConstraint,
        }

        with pytest.raises(ValueError) as excinfo:
            safe_call("InvalidConstraint", allowed_constraints)

        assert "InvalidConstraint" in str(excinfo.value)
        assert "not allowed" in str(excinfo.value)
        assert "StandardRobustness" in str(excinfo.value)

    def test_safe_call_returns_callable(self):
        """Test that safe_call returns the actual class, not an instance."""
        allowed_constraints = {
            "StandardRobustness": StandardRobustnessConstraint,
        }

        result = safe_call("StandardRobustness", allowed_constraints)

        # Should return the class itself
        assert result is StandardRobustnessConstraint

        # Should be callable (can instantiate)
        assert callable(result)

        # Should be able to create instance
        device = torch.device("cpu")
        instance = result(device, epsilon=0.1)
        assert isinstance(instance, StandardRobustnessConstraint)
