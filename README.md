# Property-Driven Machine Learning

A general framework for property-driven machine learning that enables incorporating logical properties and constraints into neural network training and evaluation.

## Features

- **Multiple Logic Systems**: Support for Boolean logic, fuzzy logics (Gödel, Łukasiewicz, Reichenbach, Yager, etc.), Signal Temporal Logic (STL), and DL2
- **Property Constraints**: Built-in constraint classes for robustness properties, Lipschitz constraints, output bounds, and group fairness
- **Adversarial Training**: PGD and Auto-PGD attack implementations for constraint evaluation
- **Gradient Normalization**: GradNorm for balancing multiple training objectives

## Installation

### From PyPI (Recommended)

```bash
# Install the latest version from PyPI
pip install property-driven-ml

# Or install with uv (faster)
uv add property-driven-ml
```

### From Source (Development)

For development or to run the latest features:

#### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package and project management. Install uv first:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

#### Installation Steps

```bash
# Clone the repository
git clone https://github.com/tflinkow/property-driven-ml.git
cd property-driven-ml

# Install dependencies and the package (recommended)
uv sync

# Or install in development mode with pip
pip install -e .

# Or install directly
pip install .
```

### Requirements

- Python 3.11+
- PyTorch 2.5.1+
- CUDA support (optional, for GPU acceleration)

**Note**: 
- For PyPI installation: Use standard `python` commands
- For local development with uv: Use `uv run python` to run scripts with the managed environment

## Quick Start

### Using the Command Line Interface

After installation from PyPI:
```bash
# Get help on available options
property-driven-ml --help

# Run a training experiment
property-driven-ml \
  --data-set=mnist \
  --batch-size=64 \
  --lr=0.001 \
  --epochs=10 \
  --input-region="EpsilonBall(eps=0.1)" \
  --output-constraint="StandardRobustness(delta=0.05)" \
  --experiment-name="mnist_robustness" \
  --logic=GD
```

For local development:
```bash
# Run from the repository root
uv run python main.py --help
```

### Basic API Usage

```python
import torch
import property_driven_ml.logics as logics
import property_driven_ml.constraints as constraints
import property_driven_ml.training as training

# Create a logic system
logic = logics.GoedelFuzzyLogic()

# Define a robustness constraint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
constraint = constraints.StandardRobustnessConstraint(device, delta=0.1)

# Set up adversarial attack for constraint evaluation
x_sample = torch.randn(1, 3, 32, 32).to(device)  # Example input
oracle = training.PGD(x_sample, logic, device, steps=20, restarts=10, step_size=0.01)

# Use in training loop
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(3072, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
).to(device)

# Evaluate constraint satisfaction
x = torch.randn(8, 3, 32, 32).to(device)
x_adv = oracle.attack(model, x, None, x.min(), x.max(), constraint)
loss, satisfaction = constraint.eval(model, x, x_adv, None, logic)
```

### Available Logic Systems

```python
import property_driven_ml.logics as logics

# Access logic systems through the logics module
logic_boolean = logics.BooleanLogic()                    # Classical Boolean logic
logic_godel = logics.GoedelFuzzyLogic()                  # Gödel fuzzy logic
logic_lukasiewicz = logics.LukasiewiczFuzzyLogic()       # Łukasiewicz fuzzy logic
logic_reichenbach = logics.ReichenbachFuzzyLogic()       # Reichenbach fuzzy logic
logic_yager = logics.YagerFuzzyLogic()                   # Yager fuzzy logic
logic_stl = logics.STL()                                 # Signal Temporal Logic
logic_dl2 = logics.DL2()                                 # DL2 logic
```

### Available Constraints

```python
import property_driven_ml.constraints as constraints

# Access constraint classes through the constraints module
robustness = constraints.StandardRobustnessConstraint(device, delta=0.1)    # Adversarial robustness
lipschitz = constraints.LipschitzRobustnessConstraint(device, L=1.0)        # Lipschitz continuity
output_bounds = constraints.AlsomitraOutputConstraint(device, lo, hi)       # Output bounds
group_fair = constraints.GroupConstraint(device, groups, delta=0.1)         # Group fairness
input_region = constraints.EpsilonBall(dataset, eps=0.1, mean, std)         # Input perturbation regions
```

## Architecture

The framework is organized into several key modules:

- `property_driven_ml.logics`: Logic systems for constraint evaluation
- `property_driven_ml.constraints`: Property constraint definitions
- `property_driven_ml.training`: Training utilities (attacks, gradient normalization)
- `property_driven_ml.util`: Utility functions

## Examples

### Command-Line Usage

If you installed from PyPI, use the CLI directly:
```bash
# Run a complete training experiment
property-driven-ml \
  --data-set=mnist \
  --batch-size=64 \
  --lr=0.001 \
  --epochs=10 \
  --input-region="EpsilonBall(eps=0.1)" \
  --output-constraint="StandardRobustness(delta=0.05)" \
  --experiment-name="mnist_robustness" \
  --logic=GD
```

For local development:
```bash
# Run from the root directory with uv
uv run python main.py \
  --data-set=mnist \
  --batch-size=64 \
  --lr=0.001 \
  --epochs=10 \
  --input-region="EpsilonBall(eps=0.1)" \
  --output-constraint="StandardRobustness(delta=0.05)" \
  --experiment-name="mnist_robustness" \
  --logic=GD

# Or run the shell script with multiple experiments
cd examples && bash run.sh
```

### Programmatic Usage

Write your own scripts using the framework:
```python
import property_driven_ml.logics as logics
import property_driven_ml.constraints as constraints

# Your custom training code here...
```

For local development examples:
```bash
# Simple API demonstration
uv run python examples/simple_example.py
```

### Available Examples (Local Development)
- `main.py`: Full training pipeline with command-line interface (root level)
- `examples/simple_example.py`: Clean API demonstration
- `examples/models.py`: Neural network architectures for different tasks
- `examples/alsomitra_dataset.py`: Custom dataset implementation
- `examples/run.sh`: Batch training experiments

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Development Setup

This project uses uv for dependency management and development workflows:

```bash
# Clone and set up the development environment
git clone https://github.com/tflinkow/property-driven-ml.git
cd property-driven-ml

# Install all dependencies (including dev dependencies)
uv sync

# Run tests
uv run pytest

# Run examples and scripts
uv run python main.py --help
uv run python examples/simple_example.py

# Add new dependencies
uv add torch torchvision  # Runtime dependency
uv add --dev pytest mypy  # Development dependency
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{flinkow2025general,
  title={A General Framework for Property-Driven Machine Learning},
  author={Flinkow, Thomas and Casadio, Marco and Kessler, Colin and Monahan, Rosemary and Komendantskaya, Ekaterina},
  journal={arXiv preprint arXiv:2505.00466},
  year={2025},
  url={https://arxiv.org/abs/2505.00466}
}
```

You can also cite the software implementation:

```bibtex
@software{property_driven_ml,
  title={Property-Driven Machine Learning Framework},
  author={Thomas Flinkow},
  year={2025},
  url={https://github.com/tflinkow/property-driven-ml}
}
```