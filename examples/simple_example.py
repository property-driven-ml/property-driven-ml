#!/usr/bin/env python3
"""
Simple example demonstrating the property-driven-ml library usage.

This example shows how to define properties and use them in a training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import the property-driven ML framework
import property_driven_ml.logics as logics
import property_driven_ml.constraints as constraints
import property_driven_ml.training as training


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
    dataset = create_toy_dataset(1000, input_dim, n_classes)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Set up property-driven training components
    logic = logics.GoedelFuzzyLogic()  # Use Gödel fuzzy logic
    constraint = constraints.StandardRobustnessConstraint(
        device, epsilon=0.001, delta=0.1
    )  # 10% robustness margin

    # Set up adversarial oracle for constraint evaluation
    oracle = training.PGD(logic, device, steps=10, restarts=5, step_size=0.01)

    # Set up optimizers
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    grad_norm = training.GradNorm(model, device, optimizer, lr=0.001, alpha=1.5)

    print("Starting property-driven training...")

    # Training loop
    for epoch in range(5):  # Just a few epochs for demo
        model.train()
        total_pred_loss = 0.0
        total_constraint_loss = 0.0
        total_constraint_sat = 0.0
        n_batches = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Standard forward pass
            pred = model(x)
            pred_loss = nn.CrossEntropyLoss()(pred, y)

            # Generate adversarial examples for constraint evaluation
            x_adv = oracle.attack(model, x, y, constraint)

            # Evaluate constraint
            constraint_loss, constraint_sat = constraint.eval(
                model, x, x_adv, None, logic, reduction="mean"
            )

            if constraint_loss is not None:
                # Use GradNorm to balance losses
                grad_norm.balance(pred_loss, constraint_loss)
                total_constraint_loss += constraint_loss.item()
            else:
                pred_loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            # Track metrics
            total_pred_loss += pred_loss.item()
            if constraint_sat is not None:
                total_constraint_sat += constraint_sat.item()
            n_batches += 1

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Pred Loss: {pred_loss.item():.4f}, "
                    f"Constraint Loss: {constraint_loss.item() if constraint_loss is not None else 0:.4f}"
                )

        # Epoch summary
        avg_pred_loss = total_pred_loss / n_batches
        avg_constraint_loss = total_constraint_loss / n_batches
        avg_constraint_sat = total_constraint_sat / n_batches

        print(f"Epoch {epoch} Summary:")
        print(f"  Average Prediction Loss: {avg_pred_loss:.4f}")
        print(f"  Average Constraint Loss: {avg_constraint_loss:.4f}")
        print(f"  Average Constraint Satisfaction: {avg_constraint_sat:.4f}")
        print()

    print("Training completed!")

    # Evaluate final model
    model.eval()
    with torch.no_grad():
        test_x, test_y = next(iter(dataloader))
        test_x, test_y = test_x.to(device), test_y.to(device)

        # Standard accuracy
        pred = model(test_x)
        accuracy = (pred.argmax(dim=1) == test_y).float().mean()

        # Constraint satisfaction on adversarial examples
        test_x_adv = oracle.attack(model, test_x, test_y, constraint)
        _, constraint_sat = constraint.eval(
            model, test_x, test_x_adv, None, logics.BooleanLogic(), reduction="mean"
        )

        print("Final Evaluation:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(
            f"  Constraint Satisfaction Rate: {constraint_sat.item() if constraint_sat is not None else 'N/A':.4f}"
        )


if __name__ == "__main__":
    main()
