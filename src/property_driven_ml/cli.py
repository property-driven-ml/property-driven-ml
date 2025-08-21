#!/usr/bin/env python3
"""
Property-Driven ML Command Line Interface

This script provides a command-line interface to the property-driven-ml library,
allowing users to train models with logical constraints.
"""

import sys
import os


def main():
    """Main entry point that delegates to the root training script."""
    try:
        # Add the root directory to the path so we can import the main script
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
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
