# KS Vote Scraper — Command Runner

# Default: show available commands
default:
    @just --list

# Scrape current session (cached)
scrape *args:
    uv run ks-vote-scraper {{args}}

# Scrape with fresh cache
scrape-fresh *args:
    uv run ks-vote-scraper --clear-cache {{args}}

# Lint and format
lint:
    uv run ruff check --fix src/
    uv run ruff format src/

# Lint check only (no fix)
lint-check:
    uv run ruff check src/
    uv run ruff format --check src/

# Install dependencies
install:
    uv sync

# List available sessions
sessions:
    uv run ks-vote-scraper --list-sessions

# Full check (lint)
check:
    just lint-check
