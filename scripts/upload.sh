#!/bin/bash
# Upload script for TestPyPI and PyPI distribution

set -e

echo "🚀 Property-Driven ML Distribution Helper"
echo "========================================="

# Check if we're in the right directory
if [[ ! -f pyproject.toml ]]; then
    echo "❌ Error: pyproject.toml not found. Please run from the repository root."
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo "🔍 Checking requirements..."
if ! command_exists uv; then
    echo "❌ uv is required but not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command_exists twine; then
    echo "📦 Installing twine for package uploads..."
    uv add --dev twine
fi

echo "✅ All requirements satisfied"

# Clean and build
echo ""
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ src/*.egg-info/

echo "🔨 Building package..."
uv build

echo "✅ Build completed successfully"

# Display options
echo ""
echo "📋 Available commands:"
echo "1. upload-test    - Upload to TestPyPI"
echo "2. upload-prod    - Upload to PyPI (production)"
echo "3. check          - Check distribution files"
echo "4. status         - Check token configuration status"

if [[ $# -eq 0 ]]; then
    echo ""
    echo "Usage: $0 [command]"
    echo "Example: $0 upload-test"
    exit 0
fi

case "$1" in
    "upload-test")
        echo ""
        echo "🧪 Uploading to TestPyPI..."

        # Check if ~/.pypirc exists and has testpypi token
        if [[ ! -f ~/.pypirc ]]; then
            echo "❌ ~/.pypirc file not found. Copying local .pypirc to home directory..."
            if [[ -f .pypirc ]]; then
                cp .pypirc ~/.pypirc
                echo "✅ Copied .pypirc to home directory"
            else
                echo "❌ No .pypirc file found. Please run ./scripts/setup-tokens.sh first."
                exit 1
            fi
        fi

        if grep -q "YOUR_TESTPYPI_TOKEN_HERE" ~/.pypirc; then
            echo "❌ TestPyPI token not configured. Please run ./scripts/setup-tokens.sh first."
            exit 1
        fi

        echo "✅ Using configured TestPyPI token from ~/.pypirc"
        uv run twine upload --repository testpypi dist/*
        echo ""
        echo "✅ Upload to TestPyPI completed!"
        echo "📝 To test the installation:"
        echo "   # Using pip:"
        echo "   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ property-driven-ml"
        echo "   # Using uv:"
        echo "   uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ property-driven-ml"
        ;;
    "upload-prod")
        echo ""
        echo "🚀 Uploading to PyPI (PRODUCTION)..."

        # Check if ~/.pypirc exists and has pypi token
        if [[ ! -f ~/.pypirc ]]; then
            echo "❌ ~/.pypirc file not found. Copying local .pypirc to home directory..."
            if [[ -f .pypirc ]]; then
                cp .pypirc ~/.pypirc
                echo "✅ Copied .pypirc to home directory"
            else
                echo "❌ No .pypirc file found. Please run ./scripts/setup-tokens.sh first."
                exit 1
            fi
        fi

        if grep -q "YOUR_PYPI_TOKEN_HERE" ~/.pypirc; then
            echo "❌ PyPI token not configured. Please run ./scripts/setup-tokens.sh first."
            exit 1
        fi

        echo "✅ Using configured PyPI token from ~/.pypirc"
        echo ""
        echo "⚠️  This will upload to the real PyPI! Are you sure? (y/N)"
        read -r confirm
        if [[ $confirm == "y" || $confirm == "Y" ]]; then
            uv run twine upload dist/*
            echo ""
            echo "✅ Upload to PyPI completed!"
            echo "📝 Your package is now available:"
            echo "   # Using pip:"
            echo "   pip install property-driven-ml"
            echo "   # Using uv:"
            echo "   uv pip install property-driven-ml"
        else
            echo "❌ Upload cancelled"
        fi
        ;;
    "check")
        echo ""
        echo "🔍 Checking distribution files..."
        uv run twine check dist/*
        echo ""
        echo "📊 Distribution files:"
        ls -lh dist/
        ;;
    "status")
        echo ""
        echo "🔍 Token Configuration Status:"
        if [[ ! -f ~/.pypirc ]]; then
            echo "❌ ~/.pypirc file not found"
            echo "📝 Run ./scripts/setup-tokens.sh to configure tokens"
        else
            echo "✅ ~/.pypirc file exists"

            if grep -q "YOUR_TESTPYPI_TOKEN_HERE" ~/.pypirc; then
                echo "❌ TestPyPI token: Not configured"
            else
                echo "✅ TestPyPI token: Configured"
            fi

            if grep -q "YOUR_PYPI_TOKEN_HERE" ~/.pypirc; then
                echo "❌ PyPI token: Not configured"
            else
                echo "✅ PyPI token: Configured"
            fi

            echo ""
            echo "📝 To update tokens, run: ./scripts/setup-tokens.sh"
        fi
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Available commands: upload-test, upload-prod, check, install-test, status"
        exit 1
        ;;
esac
