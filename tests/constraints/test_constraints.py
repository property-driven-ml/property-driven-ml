"""
Tests for constraint implementations and related components.

This module tests the constraint system including preconditions, postconditions,
and the unified constraint classes.
"""

import pytest
import torch
import torch.nn as nn

from property_driven_ml.constraints import (
    Constraint,
    StandardRobustnessConstraint,
    EpsilonBall,
    StandardRobustnessPostcondition,
)
from property_driven_ml.logics import BooleanLogic, FuzzyLogic


class SimpleMLP(nn.Module):
    """Simple MLP for testing purposes."""

    def __init__(self, input_dim=4, hidden_dim=10, output_dim=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TestConstraintBase:
    """Test the base Constraint class."""

    def test_constraint_base_cannot_be_instantiated(self):
        """Test that Constraint is an abstract base class."""
        with pytest.raises(TypeError):
            Constraint()  # type: ignore


class TestEpsilonBall:
    """Test EpsilonBall precondition."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def epsilon_ball(self, device):
        return EpsilonBall(device, epsilon=0.1)

    @pytest.fixture
    def sample_input(self, device):
        """Create a sample input tensor."""
        return torch.randn(2, 4, device=device)

    def test_epsilon_ball_initialization(self, device):
        """Test EpsilonBall initialization with different parameters."""
        # Test basic initialization
        eps_ball = EpsilonBall(device, epsilon=0.1)
        assert eps_ball.epsilon == 0.1
        assert eps_ball.device == device
        assert eps_ball.std is None

        # Test initialization with std
        std_val = 0.5
        eps_ball_with_std = EpsilonBall(device, epsilon=0.1, std=std_val)
        assert eps_ball_with_std.std is not None
        assert torch.equal(eps_ball_with_std.std, torch.tensor(std_val, device=device))

    def test_epsilon_ball_get_bounds_basic(self, epsilon_ball, sample_input):
        """Test basic bounds calculation."""
        lo, hi = epsilon_ball.get_bounds(sample_input)

        # Check shapes match
        assert lo.shape == sample_input.shape
        assert hi.shape == sample_input.shape

        # Check bounds are correct
        expected_lo = sample_input - 0.1
        expected_hi = sample_input + 0.1
        assert torch.allclose(lo, expected_lo)
        assert torch.allclose(hi, expected_hi)

    def test_epsilon_ball_get_bounds_with_std(self, device, sample_input):
        """Test bounds calculation with standard deviation scaling."""
        std_val = 0.5
        epsilon_ball = EpsilonBall(device, epsilon=0.1, std=std_val)

        lo, hi = epsilon_ball.get_bounds(sample_input)

        # With std, epsilon should be scaled
        scaled_epsilon = 0.1 / std_val
        expected_lo = sample_input - scaled_epsilon
        expected_hi = sample_input + scaled_epsilon

        assert torch.allclose(lo, expected_lo)
        assert torch.allclose(hi, expected_hi)

    def test_epsilon_ball_invalid_std_shape(self, device, sample_input):
        """Test that invalid std shape raises an error."""
        # Create std with wrong shape - use a tuple instead of tensor
        wrong_std = (
            0.1,
            0.2,
            0.3,
        )  # sample_input has shape (2, 4), so this is wrong size
        epsilon_ball = EpsilonBall(device, epsilon=0.1, std=wrong_std)

        with pytest.raises(ValueError, match="std must be either a scalar"):
            epsilon_ball.get_bounds(sample_input)


class TestStandardRobustnessPostcondition:
    """Test StandardRobustnessPostcondition."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def postcondition(self, device):
        return StandardRobustnessPostcondition(device, delta=0.1)

    @pytest.fixture
    def simple_model(self):
        return SimpleMLP(input_dim=4, output_dim=3)

    @pytest.fixture
    def sample_inputs(self, device):
        """Create sample input tensors."""
        x = torch.randn(2, 4, device=device)
        x_adv = x + torch.randn_like(x) * 0.01  # Small perturbation
        return x, x_adv

    def test_postcondition_initialization(self, device):
        """Test postcondition initialization with valid delta values."""
        # Test valid delta values
        for delta in [0.0, 0.05, 0.5, 1.0]:
            postcond = StandardRobustnessPostcondition(device, delta=delta)
            assert torch.equal(postcond.delta, torch.tensor(delta, device=device))

    def test_postcondition_invalid_delta(self, device):
        """Test that invalid delta values raise assertions."""
        # Test invalid delta values
        with pytest.raises(AssertionError):
            StandardRobustnessPostcondition(device, delta=-0.1)

        with pytest.raises(AssertionError):
            StandardRobustnessPostcondition(device, delta=1.1)

    def test_postcondition_function_returns_callable(
        self, postcondition, simple_model, sample_inputs
    ):
        """Test that get_postcondition returns a callable."""
        x, x_adv = sample_inputs

        postcond_fn = postcondition.get_postcondition(simple_model, x, x_adv)
        assert callable(postcond_fn)

        # Test that the callable works with a logic
        logic = BooleanLogic()
        result = postcond_fn(logic)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == x.shape[0]  # Should have same batch size

    def test_postcondition_with_different_logics(
        self, postcondition, simple_model, sample_inputs
    ):
        """Test postcondition evaluation with different logic types."""
        x, x_adv = sample_inputs
        postcond_fn = postcondition.get_postcondition(simple_model, x, x_adv)

        # Test with BooleanLogic
        boolean_logic = BooleanLogic()
        result_bool = postcond_fn(boolean_logic)
        assert result_bool.dtype == torch.bool

        # Test with FuzzyLogic
        fuzzy_logic = FuzzyLogic(name="test_fuzzy")
        result_fuzzy = postcond_fn(fuzzy_logic)
        assert result_fuzzy.dtype == torch.float32
        assert torch.all(result_fuzzy >= 0.0) and torch.all(result_fuzzy <= 1.0)


class TestStandardRobustnessConstraint:
    """Test StandardRobustnessConstraint implementation."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def constraint(self, device):
        return StandardRobustnessConstraint(
            device=device,
            epsilon=0.1,
            delta=0.05,
        )

    @pytest.fixture
    def simple_model(self):
        return SimpleMLP(input_dim=4, output_dim=3)

    @pytest.fixture
    def sample_batch(self, device):
        """Create a sample batch for testing."""
        x = torch.randn(4, 4, device=device)
        y = torch.randint(0, 3, (4,), device=device)
        return x, y

    def test_constraint_initialization(self, device):
        """Test constraint initialization with various parameters."""
        constraint = StandardRobustnessConstraint(
            device=device,
            epsilon=0.2,
            delta=0.1,
        )

        # Check that precondition and postcondition are set
        assert hasattr(constraint, "precondition")
        assert hasattr(constraint, "postcondition")
        assert isinstance(constraint.precondition, EpsilonBall)
        assert isinstance(constraint.postcondition, StandardRobustnessPostcondition)

        # Check parameters
        assert constraint.precondition.epsilon == 0.2
        assert torch.equal(
            constraint.postcondition.delta, torch.tensor(0.1, device=device)
        )

    def test_uniform_sample_generation(self, constraint, sample_batch):
        """Test uniform sample generation within precondition bounds."""
        x, _ = sample_batch
        num_samples = 3

        samples = constraint.uniform_sample(x, num_samples)

        # Check shape
        expected_shape = [num_samples] + list(x.shape)
        assert list(samples.shape) == expected_shape

        # Check that samples are within bounds (with some tolerance for floating point)
        lo, hi = constraint.precondition.get_bounds(x)
        lo_expanded = lo.unsqueeze(0).expand_as(samples)
        hi_expanded = hi.unsqueeze(0).expand_as(samples)

        assert torch.all(samples >= lo_expanded - 1e-6)
        assert torch.all(samples <= hi_expanded + 1e-6)

    def test_uniform_sample_with_bounds(self, device, sample_batch):
        """Test uniform sampling when global bounds are set."""
        x, _ = sample_batch

        # Test with a reasonable epsilon that won't exceed bounds for most inputs
        constraint = StandardRobustnessConstraint(
            device=device, epsilon=0.1, delta=0.05
        )

        samples = constraint.uniform_sample(x, num_samples=2)

        # Check that samples are within epsilon ball around x
        lo, hi = constraint.precondition.get_bounds(x)
        lo_expanded = lo.unsqueeze(0).expand_as(samples)
        hi_expanded = hi.unsqueeze(0).expand_as(samples)

        # Samples should be within the epsilon ball bounds (with small tolerance for floating point)
        assert torch.all(samples >= lo_expanded - 1e-6)
        assert torch.all(samples <= hi_expanded + 1e-6)

    def test_constraint_eval_basic(self, constraint, simple_model, sample_batch):
        """Test basic constraint evaluation."""
        x, _ = sample_batch
        logic = BooleanLogic()

        loss, sat = constraint.eval(
            N=simple_model,
            x=x,
            x_adv=None,  # Will generate random samples
            y_target=None,
            logic=logic,
        )

        # Check output types and shapes
        assert isinstance(loss, torch.Tensor)
        assert isinstance(sat, torch.Tensor)
        assert loss.shape == (x.shape[0],)
        assert sat.shape == (x.shape[0],)

        # Check that loss and sat are finite
        assert torch.all(torch.isfinite(loss))
        assert torch.all(torch.isfinite(sat))

    def test_constraint_eval_with_adversarial(
        self, constraint, simple_model, sample_batch
    ):
        """Test constraint evaluation with provided adversarial examples."""
        x, _ = sample_batch
        x_adv = x + torch.randn_like(x) * 0.01  # Small perturbation
        logic = BooleanLogic()

        loss, sat = constraint.eval(
            N=simple_model,
            x=x,
            x_adv=x_adv,
            y_target=None,
            logic=logic,
        )

        assert isinstance(loss, torch.Tensor)
        assert isinstance(sat, torch.Tensor)
        assert torch.all(torch.isfinite(loss))
        assert torch.all(torch.isfinite(sat))

    def test_constraint_eval_with_reduction(
        self, constraint, simple_model, sample_batch
    ):
        """Test constraint evaluation with different reduction methods."""
        x, _ = sample_batch
        logic = BooleanLogic()

        # Test mean reduction
        loss_mean, sat_mean = constraint.eval(
            N=simple_model,
            x=x,
            x_adv=None,
            y_target=None,
            logic=logic,
            reduction="mean",
        )

        assert loss_mean.shape == torch.Size([])  # Scalar
        assert sat_mean.shape == torch.Size([])  # Scalar

        # Test sum reduction
        loss_sum, sat_sum = constraint.eval(
            N=simple_model,
            x=x,
            x_adv=None,
            y_target=None,
            logic=logic,
            reduction="sum",
        )

        assert loss_sum.shape == torch.Size([])  # Scalar
        assert sat_sum.shape == torch.Size([])  # Scalar

    def test_constraint_eval_skip_sat(self, constraint, simple_model, sample_batch):
        """Test constraint evaluation with skip_sat option."""
        x, _ = sample_batch
        logic = BooleanLogic()

        loss, sat = constraint.eval(
            N=simple_model,
            x=x,
            x_adv=None,
            y_target=None,
            logic=logic,
            skip_sat=True,
        )

        # When skip_sat is True, sat should be zeros with same shape as loss
        assert torch.all(sat == 0.0)
        assert sat.shape == loss.shape

    def test_constraint_eval_fuzzy_logic(self, constraint, simple_model, sample_batch):
        """Test constraint evaluation with fuzzy logic."""
        x, _ = sample_batch
        logic = FuzzyLogic(name="test_fuzzy")

        loss, sat = constraint.eval(
            N=simple_model,
            x=x,
            x_adv=None,
            y_target=None,
            logic=logic,
        )

        # With fuzzy logic, loss should be (1 - postcondition_result)
        assert torch.all(loss >= 0.0)
        assert torch.all(loss <= 1.0)
        assert torch.all(torch.isfinite(loss))
        assert torch.all(torch.isfinite(sat))

    def test_constraint_eval_invalid_reduction(
        self, constraint, simple_model, sample_batch
    ):
        """Test that invalid reduction methods raise errors."""
        x, _ = sample_batch
        logic = BooleanLogic()

        with pytest.raises(ValueError, match="Unsupported reduction"):
            constraint.eval(
                N=simple_model,
                x=x,
                x_adv=None,
                y_target=None,
                logic=logic,
                reduction="invalid",
            )
