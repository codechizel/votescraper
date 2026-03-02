# Tallgrass

Kansas Legislature roll call vote scraper + analysis platform. Scrapes kslegislature.gov into CSV files for statistical/Bayesian analysis. Coverage: 2011-2026 (84th-91st legislatures).

## Commits

- **No Co-Authored-By lines.** Never append co-author trailers.
- Use conventional commits with version tags: `type(scope): description [vYYYY.MM.DD.N]`
- **After every feature/fix:** update relevant docs (CLAUDE.md, ADRs, design docs) before committing. Code and docs ship in the same commit.
- Never push without explicit permission.
- See `.claude/rules/commit-workflow.md` for types, scopes, and full details.

## Commands

[Just](https://github.com/casey/just) is used as a command runner — thin aliases over `uv run` commands. The `Justfile` also sets `OMP_NUM_THREADS=6` and `OPENBLAS_NUM_THREADS=6` globally to cap thread pools on Apple Silicon (ADR-0022). Run `just --list` to see all recipes, or use the underlying `uv run` commands directly.

```bash
just scrape 2025                             # → uv run tallgrass 2025
just scrape-fresh 2025                       # → uv run tallgrass --clear-cache 2025
just lint                                    # → ruff check --fix + ruff format
just lint-check                              # → ruff check + ruff format --check
just typecheck                               # → ty check src/ + ty check analysis/
just sessions                                # → uv run tallgrass --list-sessions
just check                                   # → lint-check + typecheck + test (quality gate)
just test                                    # → uv run pytest tests/ -v (~1803 tests)
just test-scraper                            # → pytest -m scraper (~264 tests)
just test-fast                               # → pytest -m "not slow" (skip integration)
just monitor                                 # → check running experiment status
just pipeline 2025-26                        # → full analysis pipeline (all phases grouped)
uv run tallgrass 2023                  # historical session (direct)
uv run tallgrass 2024 --special        # special session (direct)
```

Analysis recipes (all pass `*args` through to the underlying script):

`just eda`, `just pca`, `just mca`, `just umap`, `just irt`, `just irt-2d`, `just ppc`, `just clustering`, `just lca`, `just network`, `just bipartite`, `just indices`, `just betabinom`, `just hierarchical`, `just synthesis`, `just profiles`, `just tsa`, `just cross-session`, `just external-validation`, `just dime`, `just dynamic-irt`, `just wnominate`.

Each maps to `uv run python analysis/NN_phase/phase.py`. Example: `just profiles --names "Masterson"` runs `uv run python analysis/12_profiles/profiles.py --names "Masterson"`.

`just pipeline 2025-26` runs all 17 phases in order under a single run ID (ADR-0052). Each phase gets `--run-id` automatically. Phase 04b (2D IRT) is experimental with relaxed convergence thresholds.

## Build Philosophy

- **Check for existing open source solutions first.** Don't reinvent the wheel, but don't force a shoehorned dependency either.

## Code Style

- Python 3.14+, modern type hints (`list[str]` not `List[str]`, `X | None` not `Optional[X]`)
- Ruff: line-length 100, rules E/F/I/W. **Known ruff bug:** `ruff format` strips parentheses from `except (A, B):`, converting it to the Python 2 form `except A, B:` (= `except A as B:`). Always add `# fmt: skip` to multi-exception `except` lines (ADR-0072).
- ty: type checking (beta) — `src/` must pass clean; `analysis/` warnings-only for third-party stub noise
- Frozen dataclasses for data models; type hints on all function signatures
- Libraries with incomplete stubs configured as `replace-imports-with-any` in `pyproject.toml`

## Architecture

```
src/tallgrass/
  config.py     - Constants (BASE_URL, delays, workers, user agent)
  session.py    - KSSession: biennium URL resolution, STATE_DIR, data_dir/results_dir
  models.py     - IndividualVote + RollCall dataclasses
  scraper.py    - KSVoteScraper: 4-step pipeline (bill URLs -> API filter -> vote parse -> enrich)
  odt_parser.py - ODT vote file parser (2011-2014): pure functions, no I/O
  output.py     - CSV export (3 files: votes, rollcalls, legislators)
  cli.py        - argparse CLI entry point
```

Pipeline: `get_bill_urls()` -> `_filter_bills_with_votes()` -> `get_vote_links()` -> `parse_vote_pages()` -> `enrich_legislators()` -> `save_csvs()`

See `.claude/rules/scraper-architecture.md` for session coverage table, retry strategy, concurrency details, and ODT parsing.

## HTML Parsing Pitfalls (Hard-Won Lessons)

These are real bugs that were found and fixed. Do NOT regress on them:

1. **Tag hierarchy on vote pages is NOT what you'd expect.** `<h2>` = bill number, `<h4>` = bill title, `<h3>` = chamber/date/motion AND vote category headings.

2. **Party detection via full page text will always match "Republican".** Every legislator page has a party filter dropdown. Must parse the specific `<h2>` containing "District \d+".

2b. **Legislator `<h1>` is NOT the member name.** First `<h1>` is a nav heading. Must use `soup.find("h1", string=re.compile(r"^(Senator|Representative)\s+"))`. Also strip leadership suffixes.

3. **Vote category parsing requires scanning BOTH h2 and h3.** Uses `soup.find_all(["h2", "h3", "a"])` — do not simplify.

4. **KLISS API response structure varies.** Raw list or `{"content": [...]}`. Always handle both.

5. **Pre-2015 party detection uses `<h3>Party: Republican</h3>`** instead of `<h2>District N - Republican</h2>`.

6. **Pre-2021 bill lists are JavaScript-rendered.** JS fallback fetches `bills_li_{end_year}.js` data files.

6b. **Pre-2021 JS data uses two key formats.** 88th uses quoted JSON keys; 87th and earlier use unquoted JS object literal syntax.

6c. **JS data files live at `/m/` not `/s/` for all sessions except the 88th.**

7. **Pre-2015 vote pages are ODT files, not HTML.** ZIP archives with `content.xml`. House/Senate use different vote category names.

8. **Pre-2021 member directories are JavaScript-rendered.** Same unquoted-key issue as bill data.

9. **KS Legislature server returns HTML error pages with HTTP 200 for binary URLs.** `_get()` checks `content[:5]` for `<html` prefix.

10. **84th session ODTs often lack individual vote data.** ~30% are committee-of-the-whole (tally-only). Not a parser bug.

## Session URL Logic

- Current (2025-26): `/li/b2025_26/...`
- Historical (2023-24): `/li_2024/b2023_24/...`
- Special (2024): `/li_2024s/...`
- API: current uses `/li/api/v13/rev-1`, historical uses `/li_{end_year}/api/v13/rev-1`
- `CURRENT_BIENNIUM_START` in session.py must be updated when a new biennium becomes current (next: 2027).

## Data Model

- `vote_id` encodes a timestamp: `je_20250320203513` -> `2025-03-20T20:35:13`
- `passed`: passed/adopted/prevailed/concurred -> True; failed/rejected/sustained -> False; else null
- Vote categories: Yea, Nay, Present and Passing, Absent and Not Voting, Not Voting (exactly 5)
- Legislator slugs: `sen_` = Senate, `rep_` = House
- Column naming: scraper CSVs use `slug` and `vote`; analysis phases rename to `legislator_slug` and expect `vote` (not `vote_category`). Each phase handles the rename at load time (ADR-0066).
- Independent party handling: scraper outputs empty string; all analysis fills to "Independent" at load time (ADR-0021)

## Output

Three CSVs in `data/kansas/{legislature}_{start}-{end}/`:
- `{name}_votes.csv` — one per legislator per roll call (deduped by `legislator_slug` + `vote_id`)
- `{name}_rollcalls.csv` — one per roll call
- `{name}_legislators.csv` — one per legislator

Directory naming: `(start_year - 1879) // 2 + 18` -> legislature number. Special sessions: `{year}s`.
Cache: `data/kansas/{name}/.cache/`. Failed fetches -> `failure_manifest.json` + `missing_votes.md`.
External data: `data/external/shor_mccarty.tab` (Shor-McCarty scores, auto-downloaded from Harvard Dataverse).
External data: `data/external/dime_recipients_1979_2024.csv` (DIME CFscores, manually placed, ODC-BY license).

## Results Directory

Output layout (ADR-0052):

**Run-directory mode** (all biennium sessions): all phases grouped under a run ID.
`results/kansas/{session}/{run_id}/{NN_phase}/` with a session-level `latest` symlink (e.g. `results/kansas/91st_2025-2026/91-260228.1/01_eda/`). Run ID format: `{bb}-{YYMMDD}.{n}` where n starts at 1 (e.g. `91-260228.1`, second run same day: `91-260228.2`). When running a single phase without `--run-id`, a run_id is auto-generated so the directory structure is always consistent.

**Flat mode** (cross-session and special sessions): each analysis writes to its own date directory.
`results/kansas/cross-session/{aa}-vs-{bb}/{YYMMDD}.{n}/` where `aa`/`bb` are legislature numbers. Flat structure (no phase subdirectory). Example: `results/kansas/cross-session/90-vs-91/260226.1/`.

Both modes use `RunContext` for structured output, elapsed timing, HTML reports, and auto-primers. `resolve_upstream_dir()` handles 4-level path resolution (CLI override → run_id → flat/latest → latest/phase) so phases find their upstream data in either layout.

Experiments in `results/experimental_lab/YYYY-MM-DD_short-description/`. Each contains `experiment.md` (structured record from TEMPLATE.md), `run_experiment.py`, and `run_NN_description/` output directories.

## Analysis Pipeline

Phases live in numbered subdirectories (`analysis/01_eda/`, `analysis/07_indices/`, etc.). A PEP 302 meta-path finder in `analysis/__init__.py` redirects `from analysis.eda import X` to `analysis/01_eda/eda.py` — zero import changes needed (ADR-0030). Shared infrastructure (`run_context.py`, `report.py`) stays at the root. Phase `04b_irt_2d` is experimental (2D Bayesian IRT with PLT identification, relaxed thresholds — ADR-0054). Phase `16_dynamic_irt` is a cross-session phase (Martin-Quinn state-space IRT, runs standalone like Phase 13 — ADR-0058; post-hoc sign correction via static IRT correlation — ADR-0068). Phase `14b_external_validation_dime` validates IRT against DIME/CFscores (campaign-finance ideology, 84th-89th bienniums — ADR-0062). Phase `04c_ppc` is a standalone PPC + LOO-CV model comparison phase — loads InferenceData from all three IRT phases, runs posterior predictive checks, Yen's Q3, and PSIS-LOO (ADR-0063). Phase `17_wnominate` is a standalone validation phase comparing IRT to W-NOMINATE + Optimal Classification via R subprocess (ADR-0059). Phase `05b_lca` is Latent Class Analysis (Bernoulli mixture on binary vote matrix via StepMix, BIC model selection, Salsa effect detection, class membership tables with IRT ideal points). Phase `06b_network_bipartite` is bipartite bill-legislator network analysis (bill polarization, bridge bills, BiCM backbone extraction, bill communities — ADR-0065).

See `.claude/rules/analysis-framework.md` for the full pipeline, report system architecture, and design doc index. See `.claude/rules/analytic-workflow.md` for methodology rules, validation requirements, and audience guidance.

Key references:
- Design docs: `analysis/design/README.md`
- ADRs: `docs/adr/README.md` (73 decisions)
- Analysis primer: `docs/analysis-primer.md` (plain-English guide)
- How IRT works: `docs/how-irt-works.md` (general-audience explanation of anchors, identification, and MCMC divergences)
- External validation: `docs/external-validation-results.md` (5-biennium results, all 20 correlations "strong")
- DIME/CFscores: `docs/dime-cfscore-deep-dive.md` (campaign-finance ideology, V4.0 codebook/validation, 6-biennium coverage)
- Hierarchical deep dive: `docs/hierarchical-shrinkage-deep-dive.md` (over-shrinkage analysis with literature)
- PCA deep dive: `docs/pca-deep-dive.md` (literature review, code audit, implementation recommendations)
- MCA deep dive: `docs/mca-deep-dive.md` (theory survey, ecosystem evaluation, integration design)
- UMAP deep dive: `docs/umap-deep-dive.md` (literature survey, code review, implementation recommendations)
- IRT deep dive: `docs/irt-deep-dive.md` (field survey, code audit, test gaps, recommendations)
- IRT field survey: `docs/irt-field-survey.md` (identification problem, unconstrained β contribution, Python ecosystem gap)
- Clustering deep dive: `docs/clustering-deep-dive.md` (literature survey, code audit, test gaps, recommendations)
- Hierarchical IRT deep dive: `docs/hierarchical-irt-deep-dive.md` (ecosystem survey, code audit, 9 issues fixed, 35 tests, ADR-0033)
- Joint hierarchical IRT diagnosis: `docs/joint-hierarchical-irt-diagnosis.md` (bill-matching bug: vote_id vs bill_number, sign flip cascade, fix plan)
- Hierarchical PCA init experiment: `docs/hierarchical-pca-init-experiment.md` (R-hat fix, ESS threshold analysis, ADR-0044)
- 4-chain hierarchical IRT experiment: `docs/hierarchical-4-chain-experiment.md` (ESS fix, jitter mode-splitting discovery, adapt_diag fix)
- Hierarchical convergence improvement: `docs/hierarchical-convergence-improvement.md` (House vs Senate theory, β>0 constraint, 9-priority improvement plan, experiment results)
- Joint model deep dive: `docs/joint-model-deep-dive.md` (concurrent calibration failure, reparameterized LogNormal experiments, Stocking-Lord IRT linking, production recommendation)
- Positive beta experiment: `results/experimental_lab/2026-02-27_positive-beta/experiment.md` (LogNormal fixes R-hat but not ESS; joint improves but still fails; positive β necessary but not sufficient)
- Prediction deep dive: `docs/prediction-deep-dive.md` (literature survey, code audit, IRT circularity analysis, test gaps)
- Beta-Binomial deep dive: `docs/beta-binomial-deep-dive.md` (ecosystem survey, code audit, ddof fix, Tarone's test)
- Synthesis deep dive: `docs/synthesis-deep-dive.md` (field survey, code audit, detection algorithms, test gaps, refactoring)
- Profiles deep dive: `docs/profiles-deep-dive.md` (code audit, ecosystem survey, name-based lookup)
- Cross-session deep dive: `docs/cross-session-deep-dive.md` (ecosystem survey, code audit, 3 bugs fixed, 18 new tests)
- Scraper deep dive: `docs/scraper-deep-dive.md` (ecosystem comparison, code audit, data quality review, test gap analysis)
- 2D IRT deep dive: `docs/2d-irt-deep-dive.md` (ecosystem survey, PLT identification, Tyson paradox resolution, pipeline phase 04b)
- TSA deep dive: `docs/tsa-deep-dive.md` (literature survey, ecosystem comparison, code audit, 7 recommendations — all resolved)
- TSA R enrichment: ADR-0061 (CROPS penalty selection + Bai-Perron CIs via R subprocess)
- Dynamic ideal points: `docs/dynamic-ideal-points-deep-dive.md` (ecosystem survey, Martin-Quinn model, state-space IRT, decomposition)
- Dynamic IRT sign correction: ADR-0068 (post-hoc per-period xi negation using static IRT correlation, 87th House sign flip case)
- Dynamic IRT convergence: ADR-0070 (informative xi_init prior from static IRT, adaptive tau for small chambers, symlink race guard, MCMC budget increase)
- W-NOMINATE + OC: `docs/w-nominate-deep-dive.md` (field-standard comparison, R subprocess, validation-only design); design: `analysis/design/wnominate.md`
- PPC + LOO-CV: design: `analysis/design/ppc.md` (manual log-likelihood, Q3 local dependence, PSIS-LOO model comparison, ADR-0063)
- LCA deep dive: `docs/latent-class-deep-dive.md` (literature survey, StepMix evaluation, Salsa effect, Lubke & Neale impossibility)
- LCA design: `analysis/design/lca.md` (Bernoulli mixture, BIC selection, FIML missing data, Salsa threshold)
- Bipartite network deep dive: `docs/bipartite-network-deep-dive.md` (literature survey, BiCM, bill-centric metrics, Kansas-specific considerations)
- Bipartite design: `analysis/design/bipartite.md` (BiCM backbone, Newman projection, bill communities, Phase 6 comparison)
- Future bill text analysis: `docs/future-bill-text-analysis.md` (bb25, topic modeling, retrieval, open questions)
- Apple Silicon MCMC tuning: `docs/apple-silicon-mcmc-tuning.md` (P/E core scheduling, thread pool caps, parallel chains, batch job rules)
- Ward linkage article: `docs/ward-linkage-non-euclidean.md` (why Ward on Kappa distances is impure, the fix)
- Experiment framework: `docs/experiment-framework-deep-dive.md` (ecosystem survey, design patterns, BetaPriorSpec, PlatformCheck, monitoring)
- Nutpie deep dive: `docs/nutpie-deep-dive.md` (Rust NUTS sampler, normalizing flow adaptation, integration plan)
- Nutpie flat IRT experiment: `docs/nutpie-flat-irt-experiment.md` (Experiment 1 results: compilation, sampling, sign flip, |r|=0.994 vs PyMC)
- Nutpie hierarchical experiment: `results/experimental_lab/2026-02-27_nutpie-hierarchical/experiment.md` (Experiment 2: hierarchical per-chamber with Numba, House convergence test)
- Nutpie production migration: ADR-0051 (per-chamber hierarchical), ADR-0053 (flat IRT + joint hierarchical — all models now use nutpie)
- 84th biennium analysis: `docs/84th-biennium-analysis.md` (full pipeline review, moderate Republican faction, 2012 purge, data quality flags)
- Report enhancement survey: `docs/report-enhancement-survey.md` (current report inventory, gap analysis, open-source tools, 26 prioritized recommendations — R1-R13 implemented ADR-0069, R14-R20 implemented ADR-0071)
- Pipeline audit: ADR-0072 (8-biennium review, 18 findings, 6 fixes — except syntax, prediction leakage, sample threshold, logging)
- W-NOMINATE all-biennium run: ADR-0073 (6 R compatibility bugs fixed, all 8 bienniums validated, PPC expanded to 6/8)
- Analytic flags: `docs/analytic-flags.md` (living document of observations)
- Field survey: `docs/landscape-legislative-vote-analysis.md`
- Method evaluation: `docs/method-evaluation.md`

## Experiment Framework

Three components eliminate code duplication in MCMC experiments (ADR-0048):

- **`analysis/10_hierarchical/model_spec.py`** — `BetaPriorSpec` frozen dataclass + `PRODUCTION_BETA`, `JOINT_BETA`. Both `build_per_chamber_model()` and `build_joint_model()` accept `beta_prior=` parameter (defaults to production). Joint model uses `JOINT_BETA` (`lognormal_reparam`: exp(Normal(0,1)) for positive discrimination without boundary geometry). Experiments pass alternative specs — no code duplication. `build_per_chamber_graph()` returns the PyMC model without sampling (importable by experiments).
- **`analysis/10_hierarchical/irt_linking.py`** — IRT scale linking for cross-chamber alignment. Implements Stocking-Lord, Haebara, Mean-Sigma, and Mean-Mean linking using shared (anchor) bills. Sign-aware anchor extraction handles unconstrained per-chamber betas. Production alternative to concurrent calibration (joint model).
- **`analysis/experiment_monitor.py`** — `PlatformCheck` (validates Apple Silicon constraints before sampling), monitoring callback (`setproctitle` + atomic JSON status file), `ExperimentLifecycle` (PID lock, process group, cleanup). `just monitor` shows experiment progress.
- **`analysis/experiment_runner.py`** — `ExperimentConfig` frozen dataclass + `run_experiment()`. Orchestrates: platform check → data load → per-chamber models → optional joint → HTML report → metrics.json. 799-line experiment scripts become ~25-line configs.

All hierarchical experiments (whether using `ExperimentRunner` or standalone scripts) produce the full production HTML report via `build_hierarchical_report()` — the same 18-22 section report as `just hierarchical` (party posteriors, ICC, variance decomposition, dispersion, shrinkage scatter/table, forest plots, convergence, cross-chamber comparison, flat vs hier comparison). The nutpie flat IRT experiment uses a bespoke report (different model type).

## Concurrency

- **Scraper**: concurrent fetch via ThreadPoolExecutor (MAX_WORKERS=5), sequential parse. Never mutate shared state during fetch.
- **MCMC (all models)**: nutpie Rust NUTS sampler — single process, Rust threads for parallel chains (ADR-0051, ADR-0053). Graph-building functions (`build_per_chamber_graph()`, `build_joint_graph()`, `build_irt_graph()`) return PyMC models without sampling. Sampling functions compile with `nutpie.compile_pymc_model()` and sample with `nutpie.sample()`. PCA-informed init via `initial_points`; `jitter_rvs` excludes the PCA-initialized variable.
- **Apple Silicon (M3 Pro, 6P+6E)**: run bienniums sequentially; cap thread pools (`OMP_NUM_THREADS=6`); never use `taskpolicy -c background`. See ADR-0022.
- **PyTensor C compiler**: PyTensor requires `clang++`/`g++` for C-compiled kernels. Without it, falls back to pure Python (~18x slower). Common failure: Xcode update requires opening Xcode.app to accept license. Justfile exports `PATH` with `/usr/bin` to prevent stripped-PATH failures.
- **R (optional)**: Required for Phase 17 (W-NOMINATE/OC: `wnominate`, `oc`, `pscl`, `jsonlite`) and Phase 15 TSA enrichment (`changepoint`, `strucchange`). Install via `brew install r` then `install.packages()`. Not managed by uv. Core pipeline works without R. R CSV files use literal "NA" for missing values — always pass `null_values="NA"` to `pl.read_csv()` when reading R output (ADR-0073).
- **StepMix / scikit-learn shim (Phase 5b)**: StepMix 2.2.1 uses sklearn's private `_validate_data` and deprecated `force_all_finite` kwarg — both removed in scikit-learn 1.8. A monkey-patch in `analysis/05b_lca/lca.py` (lines 45-57) restores compatibility. **TODO: remove shim when StepMix ships a fix** (check `hasattr(StepMix, "_validate_data")` — shim is already guarded and will no-op once fixed upstream).

## Testing

```bash
just test                    # 1803 tests
just test-scraper            # scraper tests only (-m scraper)
just test-fast               # skip slow/integration tests (-m "not slow")
just check                   # full check (lint + typecheck + tests)
```

Pytest markers: `@pytest.mark.scraper` (scraper pipeline), `@pytest.mark.integration` (end-to-end/real-data), `@pytest.mark.slow` (>5s). Registered in `pyproject.toml`.

See `.claude/rules/testing.md` for test file inventory and conventions.
