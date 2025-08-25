#!/usr/bin/env bash
set -eo pipefail

# Minimal upload helper for TestPyPI and PyPI.
# Usage:
#   ./scripts/upload.sh upload-test
#   ./scripts/upload.sh upload-prod     # requires UPLOAD_CONFIRM=1 or --yes
#   ./scripts/upload.sh check
#   ./scripts/upload.sh status

ROOT="$(pwd)"
if [[ ! -f "$ROOT/pyproject.toml" ]]; then
    echo "error: run from project root (pyproject.toml missing)" >&2
    exit 1
fi

ensure_pypirc() {
    # If ~/.pypirc exists, nothing to do
    if [[ -f "$HOME/.pypirc" ]]; then
        return 0
    fi

    # Prefer repo template if present
    if [[ -f "$ROOT/.pypirc" ]]; then
        cp "$ROOT/.pypirc" "$HOME/.pypirc"
        return 0
    fi

    # Otherwise try to create a minimal ~/.pypirc from env tokens
    if [[ -n "${TESTPYPI_TOKEN:-}" || -n "${PYPI_TOKEN:-}" ]]; then
        cat > "$HOME/.pypirc" <<EOF
[distutils]
index-servers =
    testpypi
    pypi

[testpypi]
repository: https://test.pypi.org/legacy/
username: __token__
password: ${TESTPYPI_TOKEN:-YOUR_TESTPYPI_TOKEN_HERE}

[pypi]
username: __token__
password: ${PYPI_TOKEN:-YOUR_PYPI_TOKEN_HERE}
EOF
        chmod 600 "$HOME/.pypirc"
        return 0
    fi

    # No template and no env tokens — leave missing and let caller handle error
    return 1
}

has_token_placeholder() {
    local placeholder="$1"
    grep -q "$placeholder" "$HOME/.pypirc" 2>/dev/null && return 0 || return 1
}

build() {
    rm -rf dist/ build/ src/*.egg-info/ || true
    uv build
}

case "${1:-}" in
    upload-test)
            if ! ensure_pypirc; then
                echo "error: ~/.pypirc missing and could not be created; run ./scripts/setup-tokens.sh or set TESTPYPI_TOKEN" >&2
                exit 1
            fi
            if has_token_placeholder 'YOUR_TESTPYPI_TOKEN_HERE'; then
            echo "error: TestPyPI token not configured in ~/.pypirc" >&2
            echo "Run: TESTPYPI_TOKEN=... ./scripts/setup-tokens.sh" >&2
            exit 1
        fi
        build
        uv run twine upload --repository testpypi dist/*
        ;;

    upload-prod)
            if ! ensure_pypirc; then
                echo "error: ~/.pypirc missing and could not be created; run ./scripts/setup-tokens.sh or set PYPI_TOKEN" >&2
                exit 1
            fi
            if has_token_placeholder 'YOUR_PYPI_TOKEN_HERE'; then
                echo "error: PyPI token not configured in ~/.pypirc" >&2
                echo "Run: PYPI_TOKEN=... ./scripts/setup-tokens.sh" >&2
                exit 1
            fi
        if [[ "$UPLOAD_CONFIRM" != "1" && "$2" != "--yes" && "$3" != "--yes" ]]; then
            echo "This will upload to PyPI. Set UPLOAD_CONFIRM=1 or pass --yes to proceed." >&2
            exit 2
        fi
        build
        uv run twine upload dist/*
        ;;

    check)
        uv run twine check dist/*
        ls -lh dist/ || true
        ;;

    status)
        if [[ ! -f "$HOME/.pypirc" ]]; then
            echo "~/.pypirc: missing"
            exit 0
        fi
        printf "TestPyPI: %s\n" "$(grep -q 'YOUR_TESTPYPI_TOKEN_HERE' "$HOME/.pypirc" && echo 'missing' || echo 'configured')"
        printf "PyPI: %s\n" "$(grep -q 'YOUR_PYPI_TOKEN_HERE' "$HOME/.pypirc" && echo 'missing' || echo 'configured')"
        ;;

    *)
        echo "usage: $0 {upload-test|upload-prod|check|status}" >&2
        exit 1
        ;;
esac
