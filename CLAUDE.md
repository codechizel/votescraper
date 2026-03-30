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
just test                                    # → uv run pytest tests/ -v (~2980 tests)
just test-scraper                            # → pytest -m scraper (~643 tests)
just test-fast                               # → pytest -m "not slow" (skip integration)
just test-web                                # → Django tests only (-m web, requires PostgreSQL)
just pipeline 2025-26                        # → single-biennium pipeline (phases 01-25 + 02b + 06b + 07b)
just cross-pipeline                          # → cross-biennium pipeline (phases 26-30)
just common-space                            # → common space ideal points (cross-temporal scale)
just wnominate-common-space                  # → W-NOMINATE common space (Phase 30)
just scrape 2025 --auto-load                 # → scrape + load CSVs into PostgreSQL
just extract report.html --section 15        # → tallgrass-extract (bespoke report extraction)
```

Analysis recipes (all pass `*args` through): `just eda`, `just pca`, `just ega`, `just mca`, `just umap`, `just irt`, `just irt-2d`, `just bifactor`, `just hierarchical`, `just hierarchical-2d`, `just ppc`, `just clustering`, `just lca`, `just network`, `just bipartite`, `just indices`, `just betabinom`, `just synthesis`, `just profiles`, `just tsa`, `just cross-session`, `just external-validation`, `just dime`, `just dynamic-irt`, `just common-space`, `just wnominate`, `just text-analysis`, `just tbip`, `just issue-irt`, `just model-legislation`. Each maps to `uv run python analysis/NN_phase/phase.py`.

Database recipes: `just db-up`, `just db-down`, `just db-migrate`, `just db-load`, `just db-load-all`, `just db-admin`, `just db-shell`, `just django <cmd>`. See `.claude/rules/database.md`.

## Worktrees

Git worktrees for branch isolation. See `.claude/rules/worktree-workflow.md` for full rules.

```bash
just wt-new feature-name                     # create .claude/worktrees/feature-name/
just wt-done feature-name                    # merge to main + cleanup (from main repo)
```

**Hard rules:** Never `git checkout main` from a worktree. Never remove a worktree while CWD is inside it. Claude Code sessions **must** call `just wt-done <name>` from the main repo.

## Build Philosophy

- **Correctness over speed.** Do things right, not fast. Never cut corners to save time.
- **No backward-compatibility shims.** Don't write code that papers over old bugs or stale data. Let it fail, then fix the root cause. Technical debt compounds — don't take it on.
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
  session.py    - KSSession: biennium URL resolution, STATE_DIR, data_dir/results_dir, legislature number support
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
  extract/      - Bespoke report extraction (tallgrass-extract)
```

30-phase analysis pipeline in `analysis/01_eda/` through `analysis/30_wnominate_common_space/` plus `analysis/07b_hierarchical_2d/` (Hierarchical 2D IRT — party-pooled M2PL with PLT identification, ADR-0117), `analysis/06b_bifactor/` (Bifactor IRT — general ideology factor + specific partisan/bipartisan factors, ADR-0131), and `analysis/02b_ega/` (Exploratory Graph Analysis — network psychometrics dimensionality, ADR-0124). Shared EGA library at `analysis/ega/` (tetrachoric correlations, GLASSO via direct covariance, community detection with fragmentation guard, bootEGA, TEFI, UVA — ADR-0126). PEP 302 meta-path finder redirects `from analysis.eda import X` to numbered subdirectories (ADR-0030). `analysis/db.py` provides PostgreSQL loading (psycopg3 + Polars `read_database()`) with CSV fallback; `--csv` flag forces CSV-only mode (ADR-0099). `analysis/tuning.py` centralizes pipeline tuning parameters: `CONTESTED_THRESHOLD`, `MIN_VOTES`, `SENSITIVITY_THRESHOLD`, `PARTY_COLORS`, `SUPERMAJORITY_THRESHOLD`, `HIGH_DISC_THRESHOLD`, `LOW_DISC_THRESHOLD`, and external validation correlation thresholds — change values here to retune the entire pipeline (future Django dashboard integration). `analysis/init_strategy.py` provides shared MCMC initialization strategies (`--init-strategy {auto,irt-informed,pca-informed,2d-dim1}`, ADR-0107). `analysis/phase_utils.py` provides cross-phase utilities: horseshoe detection (`load_horseshoe_status()`), horseshoe warning banners (`horseshoe_warning_html()`), and column pruning (`drop_empty_optional_columns()`) — see ADR-0114. Experimental lab at `analysis/experimental/` — standalone experiments that bypass the pipeline (e.g., `joint_irt_experiment.py` for flat pooled cross-chamber IRT). See `.claude/rules/analysis-framework.md`.

Django project at `src/web/` for PostgreSQL-backed REST API at `/api/v1/`. See `.claude/rules/database.md`.

## Key Conventions

- Scraper: concurrent fetch (ThreadPoolExecutor), sequential parse. Never mutate shared state during fetch.
- MCMC: nutpie Rust NUTS sampler for all models (ADR-0051, ADR-0053). Init strategy: `--init-strategy {auto,irt-informed,pca-informed,2d-dim1,canonical}` (ADR-0107, ADR-0111; Phase 06 defaults to `pca-informed`; Phase 07 `auto` prefers canonical routing output). IRT identification: `--identification {auto,anchor-pca,anchor-agreement,...}` (ADR-0103). Robustness flags: `--horseshoe-diagnostic`, `--horseshoe-remediate`, `--contested-only`, `--promote-2d`, `--dim1-prior` (ADR-0104, ADR-0108; research-only, superseded by canonical routing).
- Canonical ideal points: 2-layer routing (ADR-0109, ADR-0110, ADR-0129). Layer 1 (manual PCA overrides) removed — Fisher's LDA in Phase 02 now handles axis instability automatically. Layer 2: horseshoe detection. Layer 3: tiered convergence — Tier 1 (converged, R-hat < 1.10), Tier 2 (point estimates credible, R-hat < 2.50 + party separation d > 1.5), Tier 3 (fall back to 1D). Routing preference: H2D Dim 1 → Flat 2D Dim 1 → 1D IRT. W-NOMINATE gate removed (ADR-0127: shares same variance-ordering vulnerability as PCA). Phase 16 (W-NOMINATE) is standalone publication validation, not a routing input. 2D IRT adaptive tuning for supermajority chambers (ADR-0112): N_TUNE doubles to 4000 when majority > 70%, beta init from PCA loadings, `--contested-only` flag. **All downstream phases that use IRT scores must load canonical ideal points** (Phase 06 routing output), not raw Phase 05 — Phase 15 (prediction) uses `_load_canonical_irt()` for this (ADR-0121).
- **PCA axis instability (RESOLVED, ADR-0118, ADR-0129):** Fisher's LDA ideology projection in Phase 02 finds the optimal party-separating direction across all PCs automatically, producing `ideology_score` and `establishment_score` columns. This replaces the manual PCA overrides (`pca_overrides.yaml`, now deprecated) and `detect_ideology_pc()` for IRT initialization. The LDA uses shrinkage (Ledoit-Wolf) for stability with small minority groups. See `docs/lda-ideology-projection.md`.
- Apple Silicon (M3 Pro): run bienniums sequentially; cap thread pools (`OMP_NUM_THREADS=6`).
- PyTensor C compiler: requires `clang++`. Xcode updates can break it silently (~18x slower fallback).

## Contextual Rules (loaded when editing matching files)

| Rule file | Loads when editing | Content |
|-----------|-------------------|---------|
| `scraper-architecture.md` | `src/**/*.py` | Session coverage, retry strategy, ODT parsing, concurrency |
| `analysis-framework.md` | `analysis/**/*.py` | 30-phase pipeline, report system, experiment framework, MCMC concurrency |
| `analytic-workflow.md` | `analysis/**/*.py` | Methodology rules, validation, audience guidance |
| `testing.md` | `tests/**/*.py` | Test inventory, markers, conventions |
| `html-pitfalls.md` | `src/tallgrass/scraper.py`, `odt_parser.py`, `bills.py` | 10 hard-won HTML parsing lessons, session URL logic |
| `data-model.md` | `src/tallgrass/models.py`, `output.py`, `src/web/legislature/models.py` | vote_id, slugs, CSV output, results layout, external data |
| `database.md` | `src/web/**/*.py`, `db_hook.py` | Django models, DB loader, REST API, Docker setup |
| `commit-workflow.md` | (always) | Commit types, scopes, doc standards |
| `worktree-workflow.md` | (always) | Worktree lifecycle, merge primitives |

## Documentation

- ADRs: `docs/adr/README.md` (131 decisions)
- Design docs: `analysis/design/README.md`
- Deep dives: `docs/*.md` (search by topic name)
- Resolved: `docs/pca-ideology-axis-instability.md` — PCA axis instability resolved via Fisher's LDA ideology projection (ADR-0129, `docs/lda-ideology-projection.md`)
