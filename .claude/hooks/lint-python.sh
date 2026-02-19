#!/bin/bash
# Auto-lint Python files after Write/Edit hooks
FILE_PATH="$1"

# Only lint Python files
if [[ "$FILE_PATH" != *.py ]]; then
    exit 0
fi

# Check if ruff is available
if ! command -v ruff &> /dev/null; then
    if command -v uv &> /dev/null; then
        uv run ruff check --fix "$FILE_PATH" 2>/dev/null
        uv run ruff format "$FILE_PATH" 2>/dev/null
    fi
else
    ruff check --fix "$FILE_PATH" 2>/dev/null
    ruff format "$FILE_PATH" 2>/dev/null
fi
