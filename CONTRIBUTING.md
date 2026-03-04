# Contributing to Tallgrass

## Setup

```bash
# Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.14 and project dependencies
uv python install 3.14
uv sync

# Install Just (optional but recommended)
# macOS: brew install just
# Linux: https://github.com/casey/just#installation
```

## Development Workflow

```bash
# Run the quality gate before submitting changes
just check          # lint + typecheck + tests

# Or run each step individually
just lint           # ruff check --fix + ruff format
just lint-check     # check without fixing
just typecheck      # ty check src/ + ty check analysis/
just test           # ~1680 tests
just test-fast      # skip slow/integration tests
```

## Code Style

- **Python 3.14+** — use modern type hints (`list[str]` not `List[str]`, `X | None` not `Optional[X]`)
- **Ruff** — line length 100, rules E/F/I/W
- **ty** — type checker (beta). `src/` must pass clean; `analysis/` is warnings-only
- Frozen dataclasses for data models
- Type hints on all function signatures

## Commit Messages

Use [conventional commits](https://www.conventionalcommits.org/) with a CalVer version tag:

```
type(scope): description [vYYYY.MM.DD.N]
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`

**Scopes:** `scraper`, `models`, `output`, `session`, `config`, `cli`, `docs`, `infra`

**Examples:**
```
feat(scraper): add retry wave backoff for 5xx errors [v2026.03.01.1]
fix(cli): validate year range before creating session [v2026.03.01.2]
docs: update roadmap with open-source readiness items [v2026.03.01.3]
```

The `N` suffix is sequential within each day, starting at 1.

## Project Layout

```
src/tallgrass/     Scraper package (pip-installable)
analysis/          17 numbered phase subdirectories + shared infrastructure
tests/             Pytest tests (~1680 total)
docs/              Deep dives, ADRs, field surveys, primers
docs/adr/          Architectural Decision Records (65 decisions)
analysis/design/   Per-phase methodology and implementation design docs
```

## Testing Conventions

- Tests live in `tests/` with one file per module (e.g., `test_irt.py` for `analysis/05_irt/irt.py`)
- Class-based organization: `class TestFeatureName:` groups related tests
- Pytest markers: `@pytest.mark.scraper`, `@pytest.mark.integration`, `@pytest.mark.slow`
- All HTTP calls are mocked — no real network access in tests
- Use `tmp_path` fixture for file I/O tests

## Adding an Analysis Phase

1. Create a numbered subdirectory: `analysis/NN_name/`
2. Write a design doc in `analysis/design/name.md`
3. Follow the standard pattern: `parse_args()`, `main()`, `RunContext` for output
4. Add a recipe to `Justfile`
5. Add the import alias to `analysis/__init__.py`'s redirect map
6. Write tests in `tests/test_name.py`
7. Update `docs/roadmap.md`

## External Data Requirements

- **Shor-McCarty scores** — auto-downloaded on first use (Phase 14)
- **DIME/CFscores** — manual download from [Stanford DIME project](https://data.stanford.edu/dime) (Phase 18)
- **R + CRAN packages** — required for Phase 16 (Dynamic IRT) and Phase 17 (W-NOMINATE/OC)

## Reporting Issues

Open an issue at https://github.com/codechizel/tallgrass/issues with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version (`tallgrass --version`)
