#!/usr/bin/env bash
set -eo pipefail

# Lightweight, non-interactive token installer.
# Usage:
#   TESTPYPI_TOKEN=... PYPI_TOKEN=... ./scripts/setup-tokens.sh
#   ./scripts/setup-tokens.sh --interactive   # prompts for tokens

ROOT_PYPIRC=".pypirc"
TARGET_PYPIRC="${HOME}/.pypirc"

interactive=0
if [[ "${1:-}" == "--interactive" ]]; then
    interactive=1
fi

# Backup existing target if present
if [[ -f "$TARGET_PYPIRC" ]]; then
    cp "$TARGET_PYPIRC" "${TARGET_PYPIRC}.bak.$(date +%s)"
fi

# If a repo template exists, use it; otherwise create a minimal template
if [[ -f "$ROOT_PYPIRC" ]]; then
    cp "$ROOT_PYPIRC" "$TARGET_PYPIRC"
else
    cat > "$TARGET_PYPIRC" <<'EOF'
[distutils]
index-servers =
        pypi
        testpypi

[testpypi]
repository: https://test.pypi.org/legacy/
username: __token__
password: YOUR_TESTPYPI_TOKEN_HERE

[pypi]
username: __token__
password: YOUR_PYPI_TOKEN_HERE
EOF
    chmod 600 "$TARGET_PYPIRC"
fi

if [[ $interactive -eq 1 ]]; then
    read -r -p "TestPyPI token (leave empty to skip): " ttoken
    read -r -p "PyPI token (leave empty to skip): " ptoken
    TESTPYPI_TOKEN="${ttoken:-}" PYPI_TOKEN="${ptoken:-}"
fi

# Replace placeholders from env if provided
if [[ -n "${TESTPYPI_TOKEN:-}" ]]; then
    sed -i "s|YOUR_TESTPYPI_TOKEN_HERE|${TESTPYPI_TOKEN}|g" "$TARGET_PYPIRC"
fi
if [[ -n "${PYPI_TOKEN:-}" ]]; then
    sed -i "s|YOUR_PYPI_TOKEN_HERE|${PYPI_TOKEN}|g" "$TARGET_PYPIRC"
fi

echo "wrote: $TARGET_PYPIRC"
if [[ -f "$TARGET_PYPIRC" ]]; then
    printf "TestPyPI: %s\n" "$(grep -q 'YOUR_TESTPYPI_TOKEN_HERE' "$TARGET_PYPIRC" && echo 'missing' || echo 'configured')"
    printf "PyPI: %s\n" "$(grep -q 'YOUR_PYPI_TOKEN_HERE' "$TARGET_PYPIRC" && echo 'missing' || echo 'configured')"
fi
