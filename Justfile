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

# Run UMAP analysis
umap *args:
    uv run python analysis/umap_viz.py {{args}}

# Run synthesis report
synthesis *args:
    uv run python analysis/synthesis.py {{args}}

# Run all tests
test *args:
    uv run pytest tests/ {{args}} -v

# Run scraper tests only
test-scraper *args:
    uv run pytest tests/test_session.py tests/test_scraper_pure.py tests/test_scraper_html.py tests/test_models.py tests/test_output.py tests/test_cli.py {{args}} -v

# Full check (lint + tests)
check:
    just lint-check
    just test
