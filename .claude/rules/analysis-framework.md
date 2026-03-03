---
paths:
  - "analysis/**/*.py"
---

# Analysis Framework

## Pipeline

EDA -> PCA -> MCA -> IRT -> UMAP -> Clustering -> LCA -> Network -> Bipartite Network -> Indices -> Prediction -> Beta-Binomial -> Hierarchical IRT -> Synthesis -> Profiles -> TSA

Phase 04b (2D IRT) is available standalone (`just irt-2d`) but removed from the pipeline — Kansas voting is fundamentally 1D (ADR-0074). Hierarchical IRT's joint cross-chamber model is off by default (`--run-joint` to enable); Stocking-Lord linking is the production cross-chamber alignment method (ADR-0074). Cross-session validation compares across bienniums (separate from the per-session pipeline). External validation compares IRT ideal points against Shor-McCarty scores (84th-88th bienniums only). Phase 14b (DIME/CFscores) validates against campaign-finance ideology (84th-89th bienniums, ADR-0062). Dynamic IRT (Phase 16) is a cross-session phase using Martin-Quinn state-space IRT across all 8 bienniums (ADR-0058; post-hoc sign correction via static IRT correlation — ADR-0068). PPC + LOO-CV (Phase 4c) is a standalone validation phase running posterior predictive checks and model comparison across all three IRT variants (ADR-0063). W-NOMINATE + OC (Phase 17) is a standalone validation phase comparing IRT to field-standard legislative scaling methods via R subprocess (ADR-0059). Bill Text Analysis (Phase 18) is a standalone NLP phase — BERTopic topic modeling (FastEmbed + HDBSCAN), optional CAP classification via Claude API, bill similarity, and vote cross-reference. Runs with `just text-analysis`; not in the pipeline (requires BT1 bill text data + optional API key for CAP). Text-Based Ideal Points (Phase 18b) derives text-informed ideology from vote-weighted bill embeddings + PCA, validating against IRT. Runs with `just tbip`; not in pipeline (requires BT1 + IRT results, ADR-0086). Issue-Specific Ideal Points (Phase 19) runs Phase 04 flat IRT on per-topic vote subsets from Phase 18 BERTopic/CAP assignments — estimates per-policy-area ideal points. Runs with `just issue-irt`; not in pipeline (requires Phase 18 topics + Phase 04 IRT, ADR-0087). Model Legislation Detection (Phase 20) compares Kansas bill embeddings against ALEC model policy corpus + neighbor state bills (MO, OK, NE, CO via OpenStates API v3). Three-tier cosine similarity matching with n-gram overlap confirmation. Cross-references Phase 18 topics. Runs with `just model-legislation`; not in pipeline (requires BT1 bill texts + ALEC corpus from `just alec`, ADR-0089).

## Technology Preferences

- **Polars over pandas** for all data manipulation (pandas only when downstream library requires it)
- **Python-first, best tool wins** — prefer Python; open to any open-source language (R, Rust, etc.) when it produces better results. Already using Rust (nutpie) for MCMC sampling.
- Tables: great_tables with polars DataFrames (no pandas conversion). Plots: base64-embedded PNGs. See ADR-0004.

## Directory Structure

Phases live in numbered subdirectories (`analysis/01_eda/`, `analysis/07_indices/`, etc.). A PEP 302 meta-path finder in `analysis/__init__.py` redirects `from analysis.eda import X` to `analysis/01_eda/eda.py` — zero import changes needed (ADR-0030). Shared infrastructure (`run_context.py`, `report.py`, `design/`) stays at the root.

## HTML Report System

Each phase produces a self-contained HTML report with SPSS/APA-style tables and embedded plots:

- `analysis/report.py` — Generic: `TableSection`, `FigureSection` (with `alt_text`), `TextSection`, `KeyFindingsSection`, `InteractiveTableSection`, `InteractiveSection` (with `aria_label`), `DownloadSection`, `ScrollySection`/`ScrollyStep`, `ReportBuilder`, `make_gt()`, `make_interactive_table()`, Jinja2 template + CSS (ADR-0069, ADR-0071, ADR-0079)
- `analysis/run_context.py` — `RunContext` context manager: structured output, elapsed timing, auto-primers, `export_csv()`, `strip_leadership_suffix()` utility, `generate_run_id()`, `resolve_upstream_dir()` (ADR-0052)
- `analysis/dashboard.py` — Pipeline dashboard index generator: sidebar nav + iframe embedding, elapsed time aggregation (ADR-0071)
- Phase-specific report builders: `*_report.py` in each subdirectory (e.g., `analysis/01_eda/eda_report.py`)
- Enhancement survey: `docs/report-enhancement-survey.md` (current inventory, gap analysis, 26 prioritized recommendations — R1-R13 implemented, ADR-0069)

## Key Data Modules (Pure Logic, No I/O)

- `analysis/08_prediction/nlp_features.py` — TF-IDF + NMF topic modeling on bill `short_title` text
- `analysis/11_synthesis/synthesis_detect.py` — Notable legislator detection (mavericks, bridge-builders, paradoxes)
- `analysis/12_profiles/profiles_data.py` — Profile targets, scorecards, bill-type breakdown, defections
- `analysis/13_cross_session/cross_session_data.py` — Legislator matching, IRT alignment, shift metrics, prediction transfer
- `analysis/14_external_validation/external_validation_data.py` — SM parsing, name normalization, matching, correlations, outlier detection
- `analysis/14b_external_validation_dime/external_validation_dime_data.py` — DIME parsing, name normalization, biennium filtering, CFscore matching
- `analysis/16_dynamic_irt/dynamic_irt_data.py` — Global roster, cross-biennium vote stacking, bridge coverage, emIRT interface
- `analysis/11_synthesis/coalition_labeler.py` — Auto-named coalitions from clusters (party composition, IRT ideal points)

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

## Key Data Structures

The **vote matrix** (legislators x roll calls, binary) is the foundation. Build from `votes.csv`: pivot `legislator_slug` x `vote_id`, Yea=1, Nay=0, absent=NaN.

**Critical preprocessing:**
- Filter near-unanimous votes (minority < 2.5%)
- Filter legislators with < 20 votes
- Analyze chambers separately

## Independent Party Handling

Scraper outputs empty string for non-R/D. Every analysis phase fills to "Independent" at load time. All modules define `PARTY_COLORS` with `"Independent": "#999999"`. Party-specific models exclude Independents. Plots iterate over parties present in data. See ADR-0021.

## Kansas-Specific Notes

- Republican supermajority (~72%) — intra-party variation more interesting than inter-party
- k=2 optimal (party split); intra-R variation is continuous
- 34 veto override votes are analytically rich (cross-party coalitions)
- Beta-Binomial and Bayesian IRT are the recommended Bayesian starting points

## Analytics Method Docs

`Analytic_Methods/` has 29 documents (one per method). Naming: `NN_CAT_method_name.md`. Categories: DATA, EDA, IDX, DIM, BAY, CLU, NET, PRD, TSA.
