#!/usr/bin/env python3
"""
Property-Driven ML Command Line Interface

This CLI is only available when developing from the repository.
When the package is installed via pip, this CLI is not available
since it depends on the main.py training script which is not included
in the distributed package.

To use the property-driven-ml library in your own projects, import it
as a Python package instead of using this CLI.
"""

import sys
import os


def main():
    """Main entry point that delegates to the root training script."""
    try:
        # Add the root directory to the path so we can import the main script
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        main_py_path = os.path.join(root_dir, "main.py")

        # Check if we're in a development environment (main.py exists)
        if not os.path.exists(main_py_path):
            print(
                "Error: This CLI is only available when developing from the repository."
            )
            print(
                "The main.py training script is not available in pip-installed versions."
            )
            print("")
            print(
                "To use property-driven-ml in your projects, import it as a Python package:"
            )
            print("  from property_driven_ml import constraints, logics, training")
            print("")
            print(
                "For examples, see: https://github.com/ggustavs/property-driven-ml/tree/main/examples"
            )
            sys.exit(1)

        sys.path.insert(0, root_dir)

        # Import and run the main training script
        from main import main as train_main

        train_main()
    except ImportError as e:
        print(f"Error importing training script: {e}")
        print(
            "Make sure you're running this from the property-driven-ml package directory."
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
