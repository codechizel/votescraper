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

# Run EDA analysis
eda *args:
    uv run python analysis/eda.py {{args}}

# Run PCA analysis
pca *args:
    uv run python analysis/pca.py {{args}}

# Run Bayesian IRT analysis
irt *args:
    uv run python analysis/irt.py {{args}}

# Run clustering analysis
clustering *args:
    uv run python analysis/clustering.py {{args}}

# Run network analysis
network *args:
    uv run python analysis/network.py {{args}}

# Run prediction analysis
prediction *args:
    uv run python analysis/prediction.py {{args}}

# Run synthesis report
synthesis *args:
    uv run python analysis/synthesis.py {{args}}

# Full check (lint)
check:
    just lint-check
