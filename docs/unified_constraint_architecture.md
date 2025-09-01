# Unified Constraint Architecture

This document describes the new unified constraint architecture that eliminates the need for `BoundedDataset` classes and provides a cleaner, more flexible API for property-driven machine learning.

## Overview

The previous architecture required users to:
1. Understand two separate class hierarchies: `BoundedDataset` for input regions and `Constraint` for output constraints
2. Create special dataset wrappers that computed bounds during iteration
3. Use custom dataloaders with these bounded datasets

The new architecture:
1. Unifies input region specification and output constraint evaluation in `InputRegionConstraint` classes
2. Works with standard PyTorch datasets and dataloaders
3. Computes bounds dynamically when needed
4. Provides a transformation pipeline system for problem-space to input-space mappings

## Key Components

### InputRegionConstraint

The base class that extends `Constraint` to also handle input region bounds:

```python
class InputRegionConstraint(Constraint):
    def get_input_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute input bounds for the given input tensor."""
        pass

    def apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformation pipeline to input."""
        pass
```

### Unified Constraint Classes

#### StandardRobustnessWithInputRegion
Combines epsilon-ball input regions with standard robustness output constraints:

```python
constraint = StandardRobustnessWithInputRegion(
    device=device,
    delta=0.1,  # Maximum allowed change in output probabilities
    eps=0.3,    # Epsilon ball radius for input perturbations
    transform=None  # Optional transformation pipeline
)
```

#### LipschitzRobustnessWithInputRegion
Combines epsilon-ball input regions with Lipschitz robustness output constraints:

```python
constraint = LipschitzRobustnessWithInputRegion(
    device=device,
    L=1.0,      # Lipschitz constant
    eps=0.3,    # Epsilon ball radius for input perturbations
    transform=None  # Optional transformation pipeline
)
```

#### EpsilonBallConstraint
Generic wrapper that adds epsilon-ball input regions to any output constraint:

```python
# Create any output constraint
output_constraint = StandardRobustnessConstraint(device, delta=0.1)

# Wrap it with epsilon-ball input region
constraint = EpsilonBallConstraint(
    device=device,
    output_constraint=output_constraint,
    eps=0.3,
    transform=None
)
```

### Enhanced Attack Classes

The new attack classes can work directly with `InputRegionConstraint` objects:

```python
class EnhancedPGD(EnhancedAttack):
    def attack_enhanced(self, N, x, y, constraint: InputRegionConstraint):
        # Get bounds dynamically from the constraint
        lo, hi = constraint.get_input_bounds(x)
        # Use the bounds for the attack
        return self.attack(N, x, y, (lo, hi), constraint)
```

### Enhanced Training Engine

The new training functions work with regular datasets:

```python
def train_enhanced(
    N: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,  # Regular DataLoader!
    optimizer,
    oracle: EnhancedAttack,
    grad_norm: training.GradNorm,
    logic: logics.Logic,
    constraint: InputRegionConstraint,  # Unified constraint
    with_dl: bool,
    is_classification: bool,
    denorm_scale: None | torch.Tensor = None,
) -> EpochInfoTrain:
    # ... implementation
```

## Migration Guide

### Before (Old Architecture)

```python
# Create base dataset
base_dataset = create_dataset()

# Wrap with BoundedDataset
bounded_dataset = EpsilonBall(base_dataset, eps=0.3, mean=mean, std=std)

# Create special dataloader
train_loader = DataLoader(bounded_dataset, batch_size=32)

# Create separate output constraint
constraint = StandardRobustnessConstraint(device, delta=0.1)

# Training loop expects (x, y, lo, hi) from bounded dataset
for x, y, lo, hi in train_loader:
    # bounds come from dataset iteration
    pass
```

### After (New Architecture)

```python
# Create regular dataset (no wrapping needed)
dataset = create_dataset()

# Create regular dataloader
train_loader = DataLoader(dataset, batch_size=32)

# Create unified constraint
constraint = StandardRobustnessWithInputRegion(
    device=device,
    delta=0.1,
    eps=0.3
)

# Training loop expects (x, y) from regular dataset
for x, y in train_loader:
    # bounds computed dynamically: lo, hi = constraint.get_input_bounds(x)
    pass
```

## Benefits

1. **Simplified API**: Users only need to understand one constraint class hierarchy
2. **No Dataset Wrapping**: Works with any PyTorch dataset directly
3. **Dynamic Bounds**: Bounds computed when needed, not during every iteration
4. **Flexible Transformations**: Built-in support for problem-space to input-space transformations
5. **Better Composability**: Easy to combine different input regions with different output constraints
6. **Reduced Memory**: No need to store bounds in datasets
7. **Better Performance**: Bounds computed only when needed (e.g., during attacks)

## Transformation Pipelines

The new architecture supports transformation pipelines for cases where you need to map from problem space to input space:

```python
# Define transformation pipeline
transform = torch.nn.Sequential(
    torch.nn.Lambda(lambda x: (x - mean) / std),  # Normalization
    # Add other transformations as needed
)

# Use in constraint
constraint = StandardRobustnessWithInputRegion(
    device=device,
    delta=0.1,
    eps=0.3,
    transform=transform
)
```

For datasets like Alsomitra where bounds are computed in problem space but constraints are evaluated in input space, the transformation system automatically handles the conversion.

## Implementation Status

✅ **Completed**:
- `InputRegionConstraint` base class
- `StandardRobustnessWithInputRegion`
- `LipschitzRobustnessWithInputRegion`
- `EpsilonBallConstraint`, `GlobalBoundsConstraint`, `AlsomitraConstraint`
- `EnhancedPGD` and `EnhancedAPGD` attack classes
- Enhanced training engine functions
- Example demonstrating the new architecture

🚧 **Future Work**:
- Invertible transformation support for complex pipelines
- Additional unified constraint combinations
- Migration utilities for existing codebases
- Performance optimizations for bound computation

## Example Usage

See `examples/unified_constraint_example.py` for a complete working example of the new architecture.
