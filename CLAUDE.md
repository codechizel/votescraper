# Tallgrass

Kansas Legislature roll call vote scraper + analysis platform. Scrapes kslegislature.gov into CSV files for statistical/Bayesian analysis. Coverage: 2011-2026 (84th-91st legislatures).

## Commits

- **No Co-Authored-By lines.** Never append co-author trailers.
- Use conventional commits with version tags: `type(scope): description [vYYYY.MM.DD.N]`
- **After every feature/fix:** update relevant docs (CLAUDE.md, ADRs, design docs) before committing. Code and docs ship in the same commit.
- Never push without explicit permission.
- See `.claude/rules/commit-workflow.md` for types, scopes, and full details.

## Commands

[Just](https://github.com/casey/just) is used as a command runner — thin aliases over `uv run` commands. The `Justfile` also sets `OMP_NUM_THREADS=6` and `OPENBLAS_NUM_THREADS=6` globally to cap thread pools on Apple Silicon (ADR-0022). Run `just --list` to see all recipes.

```bash
just scrape 2025                             # → uv run tallgrass 2025
just scrape-fresh 2025                       # → uv run tallgrass --clear-cache 2025
just text 2025                               # → uv run tallgrass-text 2025 (bill text retrieval)
just kanfocus 1999                           # → uv run tallgrass-kanfocus 1999 (KanFocus vote scrape)
just alec                                    # → uv run tallgrass-alec (ALEC model legislation scrape)
just lint                                    # → ruff check --fix + ruff format
just lint-check                              # → ruff check + ruff format --check
just typecheck                               # → ty check src/ + ty check analysis/
just check                                   # → lint-check + typecheck + test (quality gate)
just test                                    # → uv run pytest tests/ -v (~2726 tests)
just test-scraper                            # → pytest -m scraper (~628 tests)
just test-fast                               # → pytest -m "not slow" (skip integration)
just test-web                                # → Django tests only (-m web, requires PostgreSQL)
just pipeline 2025-26                        # → single-biennium pipeline (phases 01-25)
just cross-pipeline                          # → cross-biennium pipeline (phases 26-27)
just scrape 2025 --auto-load                 # → scrape + load CSVs into PostgreSQL
```

Analysis recipes (all pass `*args` through): `just eda`, `just pca`, `just mca`, `just umap`, `just irt`, `just irt-2d`, `just ppc`, `just clustering`, `just lca`, `just network`, `just bipartite`, `just indices`, `just betabinom`, `just hierarchical`, `just synthesis`, `just profiles`, `just tsa`, `just cross-session`, `just external-validation`, `just dime`, `just dynamic-irt`, `just wnominate`, `just text-analysis`, `just tbip`, `just issue-irt`, `just model-legislation`. Each maps to `uv run python analysis/NN_phase/phase.py`.

Database recipes: `just db-up`, `just db-down`, `just db-migrate`, `just db-load`, `just db-load-all`, `just db-admin`, `just db-shell`, `just django <cmd>`. See `.claude/rules/database.md`.

## Worktrees

Git worktrees for branch isolation. See `.claude/rules/worktree-workflow.md` for full rules.

```bash
just wt-new feature-name                     # create .claude/worktrees/feature-name/
just wt-done feature-name                    # merge to main + cleanup (from main repo)
```

**Hard rules:** Never `git checkout main` from a worktree. Never remove a worktree while CWD is inside it. Claude Code sessions **must** call `just wt-done <name>` from the main repo.

## Build Philosophy

- **Check for existing open source solutions first.** Don't reinvent the wheel, but don't force a shoehorned dependency either.

## Code Style

- Python 3.14.3+, modern type hints (`list[str]` not `List[str]`, `X | None` not `Optional[X]`)
- Ruff: line-length 100, rules E/F/I/W. Multi-exception `except` uses PEP 758 bracketless syntax: `except ValueError, TypeError:` (no parens needed in 3.14+). Ruff formats this automatically.
- ty: type checking (beta) — `src/` must pass clean; `analysis/` warnings-only for third-party stub noise
- Frozen dataclasses for data models; type hints on all function signatures
- Libraries with incomplete stubs configured as `replace-imports-with-any` in `pyproject.toml`

## Architecture

```
src/tallgrass/
  config.py     - Constants (BASE_URL, delays, workers, user agent)
  session.py    - KSSession: biennium URL resolution, STATE_DIR, data_dir/results_dir
  bills.py      - Shared bill discovery (HTML + JS fallback)
  models.py     - IndividualVote + RollCall + BillAction dataclasses
  scraper.py    - KSVoteScraper: 4-step pipeline (bill URLs -> API filter -> vote parse -> enrich)
  odt_parser.py - ODT vote file parser (2011-2014)
  roster.py     - OpenStates slug→ocd_id mapping (ADR-0085)
  output.py     - CSV export (5 files: votes, rollcalls, legislators, bill_actions, bill_texts)
  merge_special.py - Special session merge into parent bienniums
  db_hook.py    - Post-scrape PostgreSQL loader (subprocess, fail-soft)
  cli.py        - argparse CLI entry point
  text/         - Bill text retrieval subpackage (tallgrass-text)
  alec/         - ALEC model legislation scraper (tallgrass-alec)
  kanfocus/     - KanFocus vote data adapter (tallgrass-kanfocus, 1999-2026)
```

27-phase analysis pipeline in `analysis/01_eda/` through `analysis/27_dynamic_irt/`. PEP 302 meta-path finder redirects `from analysis.eda import X` to numbered subdirectories (ADR-0030). `analysis/db.py` provides PostgreSQL loading (psycopg3 + Polars `read_database()`) with CSV fallback; `--csv` flag forces CSV-only mode (ADR-0099). See `.claude/rules/analysis-framework.md`.

Django project at `src/web/` for PostgreSQL-backed REST API at `/api/v1/`. See `.claude/rules/database.md`.

## Key Conventions

- Scraper: concurrent fetch (ThreadPoolExecutor), sequential parse. Never mutate shared state during fetch.
- MCMC: nutpie Rust NUTS sampler for all models (ADR-0051, ADR-0053). PCA-informed init.
- Apple Silicon (M3 Pro): run bienniums sequentially; cap thread pools (`OMP_NUM_THREADS=6`).
- PyTensor C compiler: requires `clang++`. Xcode updates can break it silently (~18x slower fallback).

## Contextual Rules (loaded when editing matching files)

| Rule file | Loads when editing | Content |
|-----------|-------------------|---------|
| `scraper-architecture.md` | `src/**/*.py` | Session coverage, retry strategy, ODT parsing, concurrency |
| `analysis-framework.md` | `analysis/**/*.py` | 27-phase pipeline, report system, experiment framework, MCMC concurrency |
| `analytic-workflow.md` | `analysis/**/*.py` | Methodology rules, validation, audience guidance |
| `testing.md` | `tests/**/*.py` | Test inventory, markers, conventions |
| `html-pitfalls.md` | `src/tallgrass/scraper.py`, `odt_parser.py`, `bills.py` | 10 hard-won HTML parsing lessons, session URL logic |
| `data-model.md` | `src/tallgrass/models.py`, `output.py`, `src/web/legislature/models.py` | vote_id, slugs, CSV output, results layout, external data |
| `database.md` | `src/web/**/*.py`, `db_hook.py` | Django models, DB loader, REST API, Docker setup |
| `commit-workflow.md` | (always) | Commit types, scopes, doc standards |
| `worktree-workflow.md` | (always) | Worktree lifecycle, merge primitives |

## Documentation

- ADRs: `docs/adr/README.md` (101 decisions)
- Design docs: `analysis/design/README.md`
- Deep dives: `docs/*.md` (search by topic name)
