"""
Essential tests for training components - focusing on core initialization only.
"""

import pytest
import torch
import torch.nn as nn

from property_driven_ml.logics.boolean_logic import BooleanLogic
from property_driven_ml.training.attacks import PGD, APGD
from property_driven_ml.training.grad_norm import GradNorm


class TestPGDAttack:
    """Test the PGD attack."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def logic(self):
        return BooleanLogic()

    def test_pgd_initialization(self, logic, device):
        """Test PGD attack can be initialized with required parameters."""
        attack = PGD(logic=logic, device=device, steps=20, restarts=1, step_size=0.01)
        assert attack.steps == 20
        assert attack.restarts == 1
        assert attack.step_size == 0.01
        assert attack.device == device
        assert attack.logic == logic


class TestAPGDAttack:
    """Test the APGD attack."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def logic(self):
        return BooleanLogic()

    def test_apgd_initialization(self, logic, device):
        """Test APGD attack can be initialized with required parameters."""
        attack = APGD(logic=logic, device=device, steps=100, restarts=1)
        assert attack.steps == 100
        assert attack.restarts == 1
        assert attack.device == device
        assert attack.logic == logic


class TestGradNorm:
    """Test gradient normalization utility."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def simple_model(self):
        return nn.Linear(5, 2)

    @pytest.fixture
    def optimizer(self, simple_model):
        return torch.optim.Adam(simple_model.parameters(), lr=0.01)

    def test_grad_norm_initialization(self, simple_model, device, optimizer):
        """Test GradNorm can be initialized with required parameters."""
        grad_norm = GradNorm(
            N=simple_model, device=device, optimizer=optimizer, lr=0.01, alpha=0.12
        )
        assert grad_norm.N is simple_model
        assert grad_norm.device == device
        assert grad_norm.alpha == 0.12
        assert grad_norm.initial_loss is None
