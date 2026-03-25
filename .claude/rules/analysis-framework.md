---
paths:
  - "analysis/**/*.py"
---

# Analysis Framework

## Two Pipelines

**Single-biennium** (`just pipeline 2025-26`): phases 01-25 + 07b in order. Phases 06, 07b, 08, 16-23 gracefully skip when prerequisites are missing (no R, no bill texts, biennium out of range for SM/DIME).

EDA → PCA → MCA → Bill Text → IRT → 2D IRT → Hierarchical IRT → Hierarchical 2D IRT → PPC → UMAP → Clustering → LCA → Network → Bipartite → Indices → Beta-Binomial → Prediction → W-NOMINATE → External Validation → DIME → TSA → TBIP → Issue IRT → Model Legislation → Synthesis → Profiles

**Cross-biennium** (`just cross-pipeline`): phases 26-30. Requires data from multiple bienniums.

Cross-Session → Dynamic IRT → Common Space → W-NOMINATE Common Space

Hierarchical IRT's joint cross-chamber model is off by default (`--run-joint` to enable); Stocking-Lord linking is the production cross-chamber alignment method (ADR-0074). A flat pooled alternative exists in `analysis/experimental/joint_irt_experiment.py`.

## Technology Preferences

- **Polars over pandas** for all data manipulation (pandas only when downstream library requires it)
- **Python-first, best tool wins** — prefer Python; open to any open-source language (R, Rust, etc.) when it produces better results. Already using Rust (nutpie) for MCMC sampling.
- Tables: great_tables with polars DataFrames (no pandas conversion). Plots: base64-embedded PNGs. See ADR-0004.

## Directory Structure

Phases live in numbered subdirectories (`analysis/01_eda/`, `analysis/13_indices/`, etc.). A PEP 302 meta-path finder in `analysis/__init__.py` redirects `from analysis.eda import X` to `analysis/01_eda/eda.py` — zero import changes needed (ADR-0030). Shared infrastructure (`run_context.py`, `report.py`, `design/`) stays at the root.

## HTML Report System

Each phase produces a self-contained HTML report with SPSS/APA-style tables and embedded plots:

- `analysis/report.py` — Generic: `TableSection`, `FigureSection` (with `alt_text`), `TextSection`, `KeyFindingsSection`, `InteractiveTableSection`, `InteractiveSection` (with `aria_label`), `DownloadSection`, `ScrollySection`/`ScrollyStep`, `ReportBuilder`, `make_gt()`, `make_interactive_table()`, Jinja2 template + CSS (ADR-0069, ADR-0071, ADR-0079)
- `analysis/run_context.py` — `RunContext` context manager: structured output, elapsed timing, auto-primers, `export_csv()`, `strip_leadership_suffix()` utility, `generate_run_id()`, `resolve_upstream_dir()` (ADR-0052)
- `analysis/dashboard.py` — Pipeline dashboard index generator: sidebar nav + iframe embedding, elapsed time aggregation (ADR-0071)
- Phase-specific report builders: `*_report.py` in each subdirectory (e.g., `analysis/01_eda/eda_report.py`)
- Enhancement survey: `docs/report-enhancement-survey.md` (current inventory, gap analysis, 26 prioritized recommendations — R1-R13 implemented, ADR-0069)

## Key Data Modules (Pure Logic, No I/O)

- `analysis/15_prediction/nlp_features.py` — TF-IDF + NMF topic modeling on bill `short_title` text
- `analysis/24_synthesis/synthesis_detect.py` — Notable legislator detection (mavericks, bridge-builders, paradoxes)
- `analysis/25_profiles/profiles_data.py` — Profile targets, scorecards, bill-type breakdown, defections
- `analysis/26_cross_session/cross_session_data.py` — Legislator matching, IRT alignment, shift metrics, prediction transfer
- `analysis/17_external_validation/external_validation_data.py` — SM parsing, name normalization, matching, correlations, outlier detection
- `analysis/18_dime/external_validation_dime_data.py` — DIME parsing, name normalization, biennium filtering, CFscore matching
- `analysis/27_dynamic_irt/dynamic_irt_data.py` — Global roster, cross-biennium vote stacking, bridge coverage, emIRT interface
- `analysis/28_common_space/common_space_data.py` — Simultaneous affine alignment, bridge matrix, bootstrap uncertainty, quality gates, polarization trajectory
- `analysis/24_synthesis/coalition_labeler.py` — Auto-named coalitions from clusters (party composition, IRT ideal points)
- `analysis/phase_utils.py` — Cross-phase utilities: `load_horseshoe_status()` (reads `routing_manifest.json`), `horseshoe_warning_html()` (styled HTML banner), `drop_empty_optional_columns()` (prunes all-null columns for KanFocus data)

## Design Documents

Each phase has a design doc in `analysis/design/` — **read before interpreting results or adding a new phase:**

- `eda.md` — Binary encoding, filtering thresholds, agreement metrics, literature diagnostics (ADR-0026)
- `pca.md` — Imputation, standardization, sign convention, holdout design
- `mca.md` — Categorical encoding (Yea/Nay/Absent), prince library, Greenacre correction, horseshoe detection, PCA validation
- `irt.md` — Priors, MCMC settings, PCA-informed chain initialization, convergence diagnostics (R-hat, bulk/tail-ESS, E-BFMI)
- `clustering.md` — Three methods for robustness, k=2 finding
- `prediction.md` — XGBoost primary, IRT features dominate, NLP topic features
- `beta_binomial.md` — Empirical Bayes, per-party-per-chamber priors, shrinkage
- `synthesis.md` — Data-driven detection thresholds, graceful degradation
- `cross_session.md` — Affine IRT alignment, name matching, prediction transfer
- `external_validation.md` — SM name matching, correlation methodology, career-fixed vs session-specific
- `external_validation_dime.md` — DIME/CFscore matching, min-givers filter, incumbent-only, cycle-to-biennium mapping
- `tsa.md` — Rolling PCA drift, PELT changepoint detection, weekly Rice aggregation, CROPS penalty selection + Bai-Perron CIs (R enrichment). Deep dive: `docs/tsa-deep-dive.md`
- `dynamic_irt.md` — State-space IRT, random walk evolution, polarization decomposition, bridge coverage, post-hoc sign correction (ADR-0068). Deep dive: `docs/dynamic-ideal-points-deep-dive.md`
- `ppc.md` — PPC + LOO-CV model comparison, manual log-likelihood, Q3 local dependence, graceful degradation (ADR-0063)
- `bipartite.md` — BiCM backbone extraction, bill-side metrics, Newman projection, Leiden bill communities, Phase 6 comparison. Deep dive: `docs/bipartite-network-deep-dive.md`
- `wnominate.md` — Field-standard comparison (W-NOMINATE, Optimal Classification), R subprocess, validation-only design. Deep dive: `docs/w-nominate-deep-dive.md`
- `bill_text.md` — BERTopic topic modeling (FastEmbed + HDBSCAN + c-TF-IDF), CAP classification (Claude API, optional), bill similarity, caucus-splitting scores. Deep dive: `docs/bill-text-nlp-deep-dive.md`
- `tbip.md` — Text-based ideal points: embedding-vote approach, PCA on vote-weighted profiles, lower quality thresholds than Phase 14. ADR-0086.
- `issue_irt.md` — Issue-specific ideal points: topic-stratified flat IRT, two taxonomies (BERTopic/CAP), relaxed convergence thresholds, anchor strategy. ADR-0087.
- `model_legislation.md` — Model legislation detection: ALEC corpus matching, cross-state diffusion (MO/OK/NE/CO via OpenStates), cosine similarity thresholds, n-gram overlap. ADR-0089.
- `common_space.md` — Common space ideal points: simultaneous affine alignment (GLS 1999), bridge coverage, bootstrap uncertainty, quality gates. Deep dive: `docs/common-space-ideal-points.md`. ADR-0120.

## Key Data Structures

The **vote matrix** (legislators x roll calls, binary) is the foundation. Build from `votes.csv`: pivot `legislator_slug` x `vote_id`, Yea=1, Nay=0, absent=NaN.

**Critical preprocessing:**
- Filter near-unanimous votes (`CONTESTED_THRESHOLD`, default 2.5% — defined in `analysis/tuning.py`)
- Filter legislators with < `MIN_VOTES` (default 20 — defined in `analysis/tuning.py`)
- Analyze chambers separately

## Independent Party Handling

Scraper outputs empty string for non-R/D. Every analysis phase fills to "Independent" at load time. All modules define `PARTY_COLORS` with `"Independent": "#999999"`. Party-specific models exclude Independents. Plots iterate over parties present in data. See ADR-0021.

## Kansas-Specific Notes

- Republican supermajority (~72%) — intra-party variation more interesting than inter-party
- k=2 optimal (party split); intra-R variation is continuous
- 34 veto override votes are analytically rich (cross-party coalitions)
- Beta-Binomial and Bayesian IRT are the recommended Bayesian starting points

## Experiment Framework

Four components eliminate code duplication in MCMC experiments (ADR-0048):

- **`analysis/07_hierarchical/model_spec.py`** — `BetaPriorSpec` frozen dataclass + `PRODUCTION_BETA`, `JOINT_BETA`. Both `build_per_chamber_model()` and `build_joint_model()` accept `beta_prior=` parameter. Joint model uses `JOINT_BETA` (`lognormal_reparam`: exp(Normal(0,1)) for positive discrimination). `build_per_chamber_graph()` returns the PyMC model without sampling.
- **`analysis/07_hierarchical/irt_linking.py`** — IRT scale linking (Stocking-Lord, Haebara, Mean-Sigma, Mean-Mean) using shared anchor bills for cross-chamber alignment.
- **`analysis/experiment_monitor.py`** — `PlatformCheck` (validates Apple Silicon constraints), `ExperimentLifecycle` (PID lock, cleanup). `just monitor` checks experiment status.
- **`analysis/experiment_runner.py`** — `ExperimentConfig` frozen dataclass + `run_experiment()`. Orchestrates: platform check → data load → per-chamber models → optional joint → HTML report → metrics.json.

All hierarchical experiments produce full production HTML reports via `build_hierarchical_report()` (18-22 sections).

Standalone structural experiments (2D IRT, PC2-targeted IRT, joint pooled IRT) bypass `run_experiment()` but should still use `PlatformCheck` and `ExperimentLifecycle` directly for platform safety and process management (ADR-0105). The joint pooled IRT experiment (`analysis/experimental/joint_irt_experiment.py`) pools House+Senate into a single flat 1D IRT — succeeds where the hierarchical joint model (Phase 07) fails. See `docs/experiment-lab-code-review.md` for the full code review.

## Concurrency (MCMC)

- **MCMC (all models)**: nutpie Rust NUTS sampler — single process, Rust threads for parallel chains (ADR-0051, ADR-0053). Graph-building functions (`build_per_chamber_graph()`, `build_joint_graph()`, `build_irt_graph()`) return PyMC models without sampling. Sampling functions compile with `nutpie.compile_pymc_model()` and sample with `nutpie.sample()`. Initialization via `analysis/init_strategy.py`: `--init-strategy {auto,irt-informed,pca-informed,2d-dim1}` (ADR-0107). Auto prefers 1D IRT, falls back to PCA. **Phase 06 (2D IRT) defaults to `pca-informed`** to avoid horseshoe contamination from confounded 1D scores (see `docs/canonical-ideal-points.md`). `jitter_rvs` excludes the initialized variable. Robustness flags `--dim1-prior`, `--horseshoe-remediate` are research-only; production pipelines use **canonical ideal point routing** (2D Dim 1 for horseshoe-affected chambers, 1D for balanced chambers, following DW-NOMINATE standard), **validated by a W-NOMINATE cross-validation gate** (ADR-0123) that catches cases where the party-pooling prior distorts the selected dimension (6/28 sessions affected).
- **Apple Silicon (M3 Pro, 6P+6E)**: run bienniums sequentially; cap thread pools (`OMP_NUM_THREADS=6`); never use `taskpolicy -c background`. See ADR-0022.
- **PyTensor C compiler**: requires `clang++`/`g++` for C-compiled kernels. Without it, falls back to pure Python (~18x slower). Common failure: Xcode update requires opening Xcode.app to accept license.
- **R (optional)**: Required for Phase 16 (W-NOMINATE/OC) and Phase 19 TSA enrichment. Not managed by uv. R CSV files use literal "NA" — always pass `null_values="NA"` to `pl.read_csv()` when reading R output (ADR-0073).
- **StepMix / scikit-learn shim (Phase 10 LCA)**: StepMix 2.2.1 uses deprecated sklearn internals removed in scikit-learn 1.8. Monkey-patch in `analysis/10_lca/lca.py` (guarded, will no-op when StepMix fixes it).

## Known Issues

- **PCA axis instability in Senate sessions (RESOLVED, ADR-0118):** In 7/14 sessions (78th-83rd, 88th), PCA PC1 captures intra-Republican factionalism rather than the party divide. Detected and corrected via 7 party-separation quality gates (R1-R7): party-aware PCA init, 1D IRT party-d gate, Tier 2 party-d check (replacing circular PCA correlation), hierarchical minimum-separation guard, 2D IRT dimension swap detection, dynamic IRT canonical reference + per-period party-d. Full analysis in `docs/pca-ideology-axis-instability.md`.

## Analytics Method Docs

`Analytic_Methods/` has 29 documents (one per method). Naming: `NN_CAT_method_name.md`. Categories: DATA, EDA, IDX, DIM, BAY, CLU, NET, PRD, TSA.
