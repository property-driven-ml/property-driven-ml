#!/usr/bin/env python3
"""
Example demonstrating the new unified constraint architecture.

This example shows how to use the enhanced InputRegionConstraint classes
that combine input regions and output constraints, eliminating the need
for BoundedDataset classes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import the new unified constraint architecture
from property_driven_ml.constraints.unified_constraints import (
    StandardRobustnessWithInputRegion,
)
from property_driven_ml.training.enhanced_attacks import EnhancedPGD
from property_driven_ml.training.enhanced_engine import train_enhanced, test_enhanced
from property_driven_ml.training.grad_norm import GradNorm
import property_driven_ml.logics as logics


def create_simple_model(input_dim: int, output_dim: int) -> nn.Module:
    """Create a simple neural network model."""
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, output_dim),
    )


def create_toy_dataset(n_samples: int = 1000, input_dim: int = 10, n_classes: int = 3):
    """Create a simple toy dataset for demonstration."""
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    return TensorDataset(X, y)


def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model and dataset
    input_dim, n_classes = 10, 3
    model = create_simple_model(input_dim, n_classes).to(device)

    # Create regular datasets (no BoundedDataset needed!)
    train_dataset = create_toy_dataset(1000, input_dim, n_classes)
    test_dataset = create_toy_dataset(200, input_dim, n_classes)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create unified constraint that handles both input region and output constraint
    constraint = StandardRobustnessWithInputRegion(
        device=device,
        delta=0.1,  # Maximum allowed change in output probabilities
        eps=0.3,  # Epsilon ball radius for input perturbations
        transform=None,  # No transformation pipeline needed for this example
    )

    # Set up training components
    logic = logics.GoedelFuzzyLogic()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create gradient normalization
    grad_norm = GradNorm(
        N=model,
        device=device,
        optimizer=optimizer,
        lr=0.01,
        alpha=1.5,
        initial_dl_weight=1.0,
    )

    # Create enhanced attack that works with unified constraints
    # Get sample input to initialize attack
    sample_x, _ = next(iter(train_loader))
    sample_x = sample_x[:1].to(device)

    attack = EnhancedPGD(
        x0=sample_x, logic=logic, device=device, steps=20, restarts=5, step_size=0.01
    )

    print("Starting training with unified constraint architecture...")

    # Train for a few epochs
    for epoch in range(3):
        print(f"Epoch {epoch + 1}")

        # Train
        train_info = train_enhanced(
            N=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            oracle=attack,
            grad_norm=grad_norm,
            logic=logic,
            constraint=constraint,
            with_dl=True,  # Use property-driven learning
            is_classification=True,
        )

        # Test
        test_info = test_enhanced(
            N=model,
            device=device,
            test_loader=test_loader,
            oracle=attack,
            logic=logic,
            constraint=constraint,
            is_classification=True,
        )

        print(
            f"  Train - Accuracy: {train_info.pred_metric:.3f}, "
            f"Constraint Security: {train_info.constr_sec:.3f}"
        )
        print(
            f"  Test  - Accuracy: {test_info.pred_metric:.3f}, "
            f"Constraint Security: {test_info.constr_sec:.3f}"
        )

    print("\\nTraining completed successfully!")
    print("\\nKey benefits of the new architecture:")
    print("1. No need for BoundedDataset classes")
    print("2. Unified constraint handling (input regions + output constraints)")
    print("3. Regular DataLoader with any dataset")
    print("4. Dynamic bound computation during training")
    print("5. Cleaner, more flexible API")


if __name__ == "__main__":
    main()
