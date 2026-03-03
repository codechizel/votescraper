# ADR-0067: Open-source readiness polish

**Date:** 2026-03-01
**Status:** Accepted

## Context

The project was approaching its first public release on GitHub. An audit identified 33 items across blockers, bugs, documentation gaps, CI/testing gaps, code quality, and security hygiene that needed addressing before open-sourcing.

## Decision

Systematic open-source readiness pass covering:

**Project files:** MIT LICENSE, README.md, CONTRIBUTING.md, pyproject.toml metadata (authors, keywords, classifiers, URLs), CalVer `2026.3.1`.

**Bug fixes found during audit:**
- `except A, B:` syntax (Python 2 holdover) in 5 locations across `run_context.py`, `experiment_monitor.py`, `network.py`, `tsa.py` — silently caught only the first exception type
- Jinja2 `autoescape=False` in report system — switched to `autoescape=True` with `Markup()` for pre-rendered HTML
- `sys.exit(1)` in `experiment_runner.py` — replaced with `raise RuntimeError` for testability
- Hardcoded upstream paths in `experiment_runner.py` — replaced with `resolve_upstream_dir()`
- PID file race condition in `experiment_monitor.py` — lock before truncate
- Circular import between `scraper.py` and `odt_parser.py` — moved `VOTE_CATEGORIES` and `_normalize_bill_code` to `models.py`
- F821 undefined `plots_dir` in `bipartite_report.py` — added parameter to function signature
- `apply_dim1_sign_check()` Polars DuplicateError in Phase 06 — `with_columns` + `rename` created duplicate columns; fixed with `.drop()` before `.rename()`
- `test_tsa.py` referenced old `vote_category` column name instead of `vote`

**Documentation:**
- Phase docstrings: updated 14 stale `Usage:` paths, fixed 3 phase number mismatches, added `main()` docstrings to 21 phases
- Model/scraper docstrings: `IndividualVote`, `RollCall` attribute docs
- `KLISS_API_VERSION` constant extracted in `session.py`
- `--version` CLI flag, year validation with error message

**Testing:**
- Phase 06 tests: 16 new tests covering sign-flip logic, PCA correlation, convergence constants
- `test_data_integrity.py`: skipif guard when data CSVs aren't present
- `sys.path.insert` consistency across test files
- E402 suppression for standard `pytestmark`-between-imports pattern

**Cleanup:**
- Removed stale `plan.md` (superseded by ADR-0052)
- Removed empty `tallgrass/` directory (IDE artifact)
- Renamed ADR-0030 to consistent naming (`0030-*` not `adr-0030-*`)
- Removed redundant `from __future__ import annotations` from 9 files
- Removed duplicated `_fmt_elapsed()` — imports `_format_elapsed` from `run_context`
- Applied `ruff format` across all files
- Added `.env` to `.gitignore`

**CI:** Expanded from lint-only to 3-job pipeline (lint, typecheck, test).

## Consequences

- Test suite: 1680 → 1696 tests, all passing
- Lint + format: zero errors
- Project ready for public GitHub release under MIT license
- Phase 06 sign-flip bug would have caused `DuplicateError` on any run where Republicans had negative Dim 1 mean
- `except A, B:` bugs were silently swallowing `TypeError` exceptions in 5 code paths
