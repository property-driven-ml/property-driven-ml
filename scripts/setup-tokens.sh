#!/bin/bash
# Token setup helper for TestPyPI and PyPI

echo "🔑 Property-Driven ML Token Setup Helper"
echo "========================================"
echo ""

# Check if project .pypirc template exists
if [[ ! -f .pypirc ]]; then
    echo "❌ .pypirc template file not found. Please run from the repository root."
    exit 1
fi

echo "📝 This script will configure your API tokens in ~/.pypirc (where twine expects them)."
echo ""

# Copy template to home directory if it doesn't exist or ask to overwrite
if [[ -f ~/.pypirc ]]; then
    echo "📁 ~/.pypirc already exists."
    echo "Do you want to overwrite it with the template? (y/N)"
    read -r overwrite
    if [[ $overwrite == "y" || $overwrite == "Y" ]]; then
        cp .pypirc ~/.pypirc
        echo "✅ Copied template to ~/.pypirc"
    else
        echo "📝 Using existing ~/.pypirc file"
    fi
else
    cp .pypirc ~/.pypirc
    echo "✅ Created ~/.pypirc from template"
fi

echo ""

# TestPyPI token setup
echo "🧪 TestPyPI Token Setup:"
echo "1. Go to https://test.pypi.org/manage/account/token/"
echo "2. Create a new API token with scope 'Entire account'"
echo "3. Copy the token (starts with 'pypi-')"
echo ""
echo "Enter your TestPyPI token (or press Enter to skip):"
read -r testpypi_token

if [[ -n "$testpypi_token" ]]; then
    # Replace the placeholder in ~/.pypirc
    sed -i "s/YOUR_TESTPYPI_TOKEN_HERE/$testpypi_token/" ~/.pypirc
    echo "✅ TestPyPI token configured in ~/.pypirc!"
else
    echo "⏭️  Skipped TestPyPI token setup"
fi

echo ""

# PyPI token setup
echo "🚀 PyPI Token Setup:"
echo "1. Go to https://pypi.org/manage/account/token/"
echo "2. Create a new API token with scope 'Entire account'"
echo "3. Copy the token (starts with 'pypi-')"
echo ""
echo "Enter your PyPI token (or press Enter to skip):"
read -r pypi_token

if [[ -n "$pypi_token" ]]; then
    # Replace the placeholder in ~/.pypirc
    sed -i "s/YOUR_PYPI_TOKEN_HERE/$pypi_token/" ~/.pypirc
    echo "✅ PyPI token configured in ~/.pypirc!"
else
    echo "⏭️  Skipped PyPI token setup"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📁 Configuration saved to: ~/.pypirc"
echo "🔍 You can check the configuration with: ./scripts/upload.sh status"
echo ""
echo "Next steps:"
echo "1. Build your package: uv build"
echo "2. Upload to TestPyPI: ./scripts/upload.sh upload-test"
echo "3. Test installation from TestPyPI"
echo "4. Upload to PyPI: ./scripts/upload.sh upload-prod"
