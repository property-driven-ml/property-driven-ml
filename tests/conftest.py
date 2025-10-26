"""
Configuration file for pytest.

This module contains fixtures and configuration that are shared across
all test modules.
"""

import pytest
import torch
import numpy as np


@pytest.fixture(scope="session")
def device():
    """Provide a consistent device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def sample_mnist_batch():
    """Generate a sample MNIST-like batch for testing."""
    batch_size = 4
    channels = 1
    height = 28
    width = 28

    x = torch.randn(batch_size, channels, height, width)
    y = torch.randint(0, 10, (batch_size,))
    return x, y


@pytest.fixture
def sample_regression_batch():
    """Generate a sample regression batch for testing."""
    batch_size = 8
    input_dim = 4

    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, 1)
    return x, y


@pytest.fixture
def simple_classification_model():
    """Create a simple classification model for testing."""

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((7, 7))
            self.fc = torch.nn.Linear(32 * 7 * 7, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    return SimpleModel()


@pytest.fixture
def simple_regression_model():
    """Create a simple regression model for testing."""

    class SimpleRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(4, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 1),
            )

        def forward(self, x):
            return self.layers(x)

    return SimpleRegressor()
