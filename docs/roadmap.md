# Roadmap

What's been done, what's next, and what's on the horizon for the Tallgrass analytics pipeline.

**Last updated:** 2026-03-07 (IRT sign flip fix roadmapped; 3 pipeline bug fixes; LOO-CV + sign flip docs)

---

## Completed Phases

| # | Phase | Date | Key Finding |
|---|-------|------|-------------|
| 1 | EDA | 2026-02-19 | 82% Yea base rate, 72% R supermajority, 34 veto overrides |
| 2 | PCA | 2026-02-19 | PC1 = party (57% variance), PC2 = contrarianism (Tyson/Thompson) |
| 3 | Bayesian IRT | 2026-02-20 | 1D ideal points converge cleanly; Tyson paradox identified |
| 4 | Clustering | 2026-02-20 | k=2 optimal (party split); intra-R variation is continuous, not factional |
| 5 | Network | 2026-02-20 | Zero cross-party edges at kappa=0.40; Schreiber sole bipartisan bridge |
| 6 | Prediction | 2026-02-20 | Vote AUC=0.98; IRT features do all the work; XGBoost adds nothing over logistic |
| 7 | Classical Indices | 2026-02-21 | Rice, CQ unity, ENP, weighted maverick; Schreiber/Dietrich top mavericks |
| 2b | UMAP | 2026-02-22 | Nonlinear ideological landscape; validates PCA/IRT; most accessible visualization |
| 6+ | NLP Bill Text Features | 2026-02-22 | NMF topics on short_title; House temporal AUC 0.90→0.96, Senate 0.86→0.96 |
| — | Synthesis Report | 2026-02-22 | 29-32-section narrative HTML; joins all 10 phases into one deliverable |
| — | Synthesis Deep Dive | 2026-02-25 | Code audit, field survey, 9 fixes (dynamic AUC, minority mavericks, data extraction, 47 tests). ADR-0034. |
| — | Legislator Profiles | 2026-02-22 | Per-legislator deep-dives: scorecard, bill-type breakdown, defections, neighbors, surprising votes. Name-based lookup (`--names`) added 2026-02-25. |
| 7b | Beta-Binomial Party Loyalty | 2026-02-22 | Bayesian shrinkage on CQ unity; empirical Bayes, closed-form posteriors, 4 plots per chamber |
| — | Cross-Biennium Portability | 2026-02-22 | Removed all hardcoded legislator names from general phases; full pipeline re-run validated |
| — | Visualization Improvement Pass | 2026-02-22 | All 6 phases (Network, IRT, Prediction, PCA, Clustering, EDA) retrofitted for nontechnical audience; plain-English titles, annotated findings, data-driven highlights |
| — | Missing Votes Visibility | 2026-02-22 | Auto-injected "Missing Votes" section in every HTML report + standalone `missing_votes.md` in data directory; close votes bolded |
| 8 | Hierarchical Bayesian IRT | 2026-02-22 | 2-level partial pooling by party, non-centered parameterization, ICC variance decomposition, shrinkage vs flat IRT |
| — | 90th Biennium Pipeline Run | 2026-02-22 | Full 11-phase pipeline on 2023-24 data (94K votes, 168 legislators); cross-biennium analysis now possible |
| — | ty Type Checker | 2026-02-22 | Two-tier policy: scraper strict (0 errors), analysis warnings-only; caught 2 real type bugs on first run (ADR-0018) |
| — | Historical Session Support | 2026-02-22 | 2011-2026 coverage: JS bill discovery fallback, ODT vote parser, pre-2015 party detection (ADR-0020) |
| — | Independent Party Handling | 2026-02-22 | Pipeline-wide null-party fill, PARTY_COLORS in all 12 modules, dynamic plots, Independent exclusion from party-specific models (ADR-0021) |
| — | 89th Biennium Pipeline Run | 2026-02-22 | Full 12-phase pipeline on 2021-22 data; Dennis Pyle (Independent) handled correctly across all phases |
| — | PCA-Informed IRT Init (Default) | 2026-02-23 | Prevents reflection mode-splitting; literature-backed (Jackman pscl::ideal); `--no-pca-init` to disable (ADR-0023). Combined with nutpie (ADR-0053), all 16/16 flat IRT sessions now converge. |
| — | Parallelism Performance Experiment | 2026-02-23 | `cores=n_chains` was already PyMC default; batch-job CPU contention was the real cause; sequential chains 1.8x slower due to thermal throttling (ADR-0022 addendum) |
| — | Analysis Primer | 2026-02-24 | `docs/analysis-primer.md`: plain-English guide to the 13-step pipeline for general audiences (journalists, policymakers, citizens) |
| — | Parallelism Experiment (Complete) | 2026-02-24 | 88th Legislature 4-run experiment: parallel 1.83-1.89x faster; convergence bit-identical; OMP_NUM_THREADS=6 cap applied (ADR-0022). Writeup: `docs/apple-silicon-mcmc-tuning.md` |
| — | Full 91st Pipeline with Joint Model | 2026-02-24 | 12/12 phases succeeded including hierarchical joint cross-chamber model; first complete run with joint model. Re-run 2026-02-26 with bill-matching: 39m 36s total (joint 31 min), Senate all checks passed. |
| — | Landscape Survey & Method Evaluation | 2026-02-24 | `docs/landscape-legislative-vote-analysis.md` and `docs/method-evaluation.md`: surveyed the field, evaluated all major methods, identified external validation as the priority gap |
| — | External Validation (Shor-McCarty) | 2026-02-24 | Shor-McCarty external validation phase: name matching, Pearson/Spearman correlations, scatter plots, outlier analysis. 5 overlapping bienniums (84th-88th). ADR-0025. |
| — | External Validation Results Article | 2026-02-24 | `docs/external-validation-results.md`: general-audience article explaining SM validation results (flat House r=0.981, flat Senate r=0.929, hierarchical Senate r=-0.541) |
| — | Hierarchical Shrinkage Deep Dive | 2026-02-24 | `docs/hierarchical-shrinkage-deep-dive.md`: literature-grounded analysis of J=2 over-shrinkage problem (Gelman 2006/2015, James-Stein, Peress 2009), 6 remedies proposed |
| — | IRT Deep Dive & Field Survey | 2026-02-25 | `docs/irt-deep-dive.md` and `docs/irt-field-survey.md`: field survey of IRT implementations, code audit, identification problem, unconstrained β contribution, Python ecosystem gap. Implemented: tail-ESS, shrinkage warning, sign-constraint removal, 28 new tests (853 total). |
| 2c | MCA (Multiple Correspondence Analysis) | 2026-02-25 | Categorical-data analogue of PCA using chi-square distance; Yea/Nay/Absent as 3 categories; prince library; Greenacre correction; PCA validation (Spearman r), horseshoe detection, biplot, absence map. 34 new tests. Deep dive: `docs/mca-deep-dive.md`, design: `analysis/design/mca.md`. |
| — | Hierarchical IRT Fixes (Bill-Matching + Adaptive Priors) | 2026-02-26 | Joint model bill-matching (ADR-0043): 71 shared bills bridge chambers via concurrent calibration. Group-size-adaptive priors fix Senate-D convergence. Joint model runtime 93 min → 31 min. ADR-0042, ADR-0043. |
| 9 | Cross-Session Validation (90th vs 91st) | 2026-02-26 | First post-fix run: ideology r=0.940 (House), 0.975 (Senate). Cross-session prediction AUC 0.967-0.976 (nearly matches within-session 0.975-0.984). 94 tests. IRT ideal points confirmed as stable traits; network centrality metrics confirmed session-specific. Tyson flagged as paradox in both bienniums. |
| — | PCA-Informed Init for Hierarchical IRT | 2026-02-26 | Experiment proved PCA init fixes House R-hat (1.0102→1.0026) with r=0.999996 agreement. Implemented as default in `build_per_chamber_model()`. Per-chain ESS reporting added. ADR-0044. Article: `docs/hierarchical-pca-init-experiment.md`. |
| — | 4-Chain Hierarchical IRT Experiment | 2026-02-26 | 4 chains resolve both ESS warnings (xi: 397→564, mu_party: 356→512) at +4% wall time. Discovered jitter mode-splitting: `jitter+adapt_diag` causes R-hat ~1.53 with 4 chains; fix is `adapt_diag` with PCA init. Run 3 unnecessary. Article: `docs/hierarchical-4-chain-experiment.md`. |
| 4b | 2D Bayesian IRT (Pipeline, Experimental) | 2026-02-26 (experiment), 2026-02-28 (pipeline) | M2PL model with PLT identification to resolve Tyson paradox. Pipeline phase 04b: both chambers, nutpie sampling, RunContext/HTML report, relaxed convergence thresholds. Deep dive: `docs/2d-irt-deep-dive.md`, design: `analysis/design/irt_2d.md`, ADR-0046, ADR-0054. |
| 15 | Time Series Analysis | 2026-02-28 | Rolling-window PCA ideological drift + PELT changepoint detection on weekly Rice. Per-chamber analysis, penalty sensitivity, veto override cross-reference. Uses `ruptures` library. Deep dive: `docs/tsa-deep-dive.md`, design: `analysis/design/tsa.md`, ADR-0057. |
| 16 | Dynamic Ideal Points (Martin-Quinn) | 2026-02-28 | State-space IRT across 8 bienniums (84th-91st). Non-centered random walk with per-party evolution SD. PyMC + nutpie. Conversion vs. replacement polarization decomposition. Bridge coverage analysis. Post-hoc sign correction via static IRT correlation (ADR-0068). 63 tests. Deep dive: `docs/dynamic-ideal-points-deep-dive.md`, design: `analysis/design/dynamic_irt.md`, ADR-0058. |
| 17 | W-NOMINATE + OC Validation | 2026-02-28 | Field-standard legislative scaling comparison. W-NOMINATE (Poole & Rosenthal) + Optimal Classification (Poole 2000) via R subprocess. 3×3 correlation matrix (IRT/WNOM/OC), per-chamber scatter plots, 2D W-NOMINATE space, eigenvalue scree, fit statistics. Validation-only (does not feed downstream). **All 8 bienniums validated** (2026-03-02, 6 R compatibility bugs fixed — ADR-0073). Deep dive: `docs/w-nominate-deep-dive.md`, design: `analysis/design/wnominate.md`, ADR-0059. |
| 4c | PPC + LOO-CV Model Comparison | 2026-02-28 | PPC battery (Yea rate, accuracy, GMP, APRE) + item/person fit + Yen's Q3 local dependence + LOO-CV model comparison across flat 1D, 2D IRT, and hierarchical IRT. Manual numpy log-likelihood (no PyMC rebuild). Graceful degradation for missing models. **6/8 bienniums** (87th/89th LOO mismatch — ADR-0073). 60 tests. Design: `analysis/design/ppc.md`, ADR-0063. |
| 5b | Latent Class Analysis | 2026-02-28 | Bernoulli mixture (LCA) on binary vote matrix via StepMix. BIC model selection (K=1..8), Salsa effect detection (profile correlations), IRT cross-validation, Phase 5 ARI comparison, within-party LCA. Correct generative model for binary data. Deep dive: `docs/latent-class-deep-dive.md`, design: `analysis/design/lca.md`. |
| 6b | Bipartite Bill-Legislator Network | 2026-02-28 | Two-mode network (legislators × bills). Bill polarization, bridge bills, bill communities (Leiden on Newman projection), BiCM backbone extraction (statistical validation via maximum-entropy null model), Phase 6 comparison. Deep dive: `docs/bipartite-network-deep-dive.md`, design: `analysis/design/bipartite.md`, ADR-0065. |
| — | 84th Biennium Pipeline Stress Test | 2026-03-01 | Full 17-phase pipeline + PPC + External Validation + DIME on 2011-12 data. 8 bug fixes (column naming, IRT sensitivity sign flip, arviz/matplotlib deprecations, DIME party type). LCA class membership tables added. ADR-0066. |
| — | Open-Source Readiness | 2026-03-01 | MIT LICENSE, README, CONTRIBUTING, pyproject metadata, CI expansion (lint+typecheck+test). 9 bug fixes (except syntax, Jinja2 autoescape, sign-flip DuplicateError, circular import, PID race). Phase 06 tests (16 new). 1701 total tests. ADR-0067. |
| — | Report Enhancements (R1-R13) | 2026-03-01 | 3 new section types (KeyFindings, InteractiveTable, Interactive), ITables for 10+ large tables, Plotly interactive scatters (IRT, PCA, indices), PyVis network graphs, key findings in all 22+ reports, party density + ICC in flat IRT, cutting lines + swing votes, BPI + plus-minus indices, coalition labeler, absenteeism analysis. 39 new tests (1716 total). ADR-0069. |
| — | Full Pipeline Audit | 2026-03-02 | 8-biennium × 17-phase review. 18 findings catalogued, 6 code fixes shipped (except syntax, prediction data leakage, sample threshold, logging). ADR-0072. |
| — | W-NOMINATE All-Biennium Run | 2026-03-02 | All 8 bienniums (84th-91st) validated against W-NOMINATE + OC. 6 R compatibility bugs fixed (rollcall codes, polarity vector, OC matrix access, CSV "NA" parsing, fit stat casting, slug rename). R installed via Homebrew. ADR-0073. |
| — | PPC All-Biennium Expansion | 2026-03-02 | Phase 08 expanded from 2 to 6 bienniums (85th, 86th, 88th, 90th added). 87th/89th excluded: ArviZ LOO observation mismatch between hierarchical and flat IRT vote matrices. ADR-0073. |
| — | Name Matcher District Tiebreaker | 2026-03-02 | Phase 14 (SM) + Phase 18 (DIME) name matching now uses district-based disambiguation for ambiguous last-name matches. 3 incorrect matches corrected. 6 new tests. |
| — | Shrinkage Null Investigation | 2026-03-02 | Deep dive confirmed `SHRINKAGE_MIN_DISTANCE=0.5` is statistically justified (24.8% null rate across all bienniums). Accepted as working-as-designed. |
| — | Report Enhancements (R14-R20) | 2026-03-02 | Folium district choropleths, full voting record in Profiles, iframe dashboard, CSV downloads (28+ exports), freshmen cohort analysis, bloc stability Sankey, scrollytelling synthesis. 4 new deps (folium, geopandas, scrollama via IntersectionObserver). ADR-0071. |
| — | WCAG Accessibility (R25) | 2026-03-02 | alt_text on 132 FigureSections, aria_label on 8 InteractiveSections across 23 report builders. WCAG 2.1 AA compliance. ADR-0079. |
| — | Scraper Refactoring (M2) | 2026-03-02 | Extracted `_extract_bill_title()`, `_extract_chamber_motion_date()`, `_parse_vote_categories()`, `_extract_party_and_district()` as static methods from `_parse_vote_page()` and `enrich_legislators()`. |
| — | Bill Lifecycle (M5) | 2026-03-02 | `BillAction` dataclass, KLISS HISTORY capture, `_bill_actions.csv` export. Sankey lifecycle diagram in EDA report. 17 new tests. |
| — | Ridgeline Plots (M6) | 2026-03-02 | `plot_ridgeline_ideology()` — stacked KDE curves by biennium in dynamic IRT report. Republicans/Democrats shown separately. 3 new tests (73 total in Phase 16). |
| — | Animated Scatter (M7) | 2026-03-02 | `plot_animated_scatter()` — Gapminder-style Plotly animation in dynamic IRT report. X=ideal point, Y=uncertainty, color=party, frame=biennium. 3 new tests (76 total in Phase 16). |
| — | Sponsor Slugs → Synthesis + Profiles | 2026-03-02 | `sponsor_slugs` (M8 scraper output) integrated into Phase 11 and Phase 12. Synthesis: `n_bills_sponsored` in unified scorecard. Profiles: per-legislator sponsorship section (primary/co-sponsor, passage rate), defection sponsor context. Graceful degradation for pre-89th data. 11 new tests (1952 total). ADR-0081. |
| — | OpenStates Legislator Identity | 2026-03-02 | OCD person IDs (`ocd-person/{uuid}`) from OpenStates for stable cross-biennium legislator identity. `roster.py` module: GitHub tarball download, YAML parsing, slug→ocd_id mapping cached as JSON. 3-phase matching in Phase 13 (OCD ID → name → fuzzy). Dynamic IRT roster groups by OCD ID. Correctly separates same-name legislators (two Mike Thompsons). Backward compatible with older CSVs. 22 new roster tests + 11 OCD matching tests. ADR-0085. |
| 18b | Text-Based Ideal Points (TBIP) | 2026-03-03 | Embedding-vote approach (not true TBIP due to ~92% committee sponsorship). Multiplies vote matrix by Phase 18 bill embeddings, PCA on legislator text profiles, PC1 = text-derived ideal point. Validates against IRT (flat + hierarchical). Standalone with `just tbip`; not in pipeline (requires BT1 + IRT results). Design: `analysis/design/tbip.md`, ADR-0086. |
| 20 | Model Legislation Detection (BT5) | 2026-03-03 | ALEC template matching + cross-state diffusion (MO, OK, NE, CO via OpenStates API). Cosine similarity on BGE embeddings (same vector space as Phase 18) with three-tier classification (near-identical >= 0.95, strong >= 0.85, related >= 0.70). 5-gram overlap confirmation for strong matches. ALEC corpus: ~1,057 model policies scraped via `just alec`. 9-section HTML report. 58 tests. Design: `analysis/design/model_legislation.md`, ADR-0089. |

---

## Next Up (Prioritized)

### ~~1. W-NOMINATE~~ — Done (Phase 17)

Completed 2026-02-28. See Completed Phases table above.

### 2. DIME/CFscores External Validation → Phase 18

Completed 2026-02-28. Campaign-finance-based ideology from Bonica's DIME project (V4.0, ODC-BY). Validates 84th-89th bienniums (6 bienniums, one beyond Shor-McCarty). Reuses Phase 14 infrastructure (correlations, outliers, name normalization). See `docs/dime-cfscore-deep-dive.md`, ADR-0062.

### ~~3. Standalone Posterior Predictive Checks~~ — Done (Phase 08)

Completed 2026-02-28. PPC battery (Yea rate, accuracy, GMP, APRE) + item/person fit + Yen's Q3 local dependence + LOO-CV model comparison across flat 1D, 2D IRT, and hierarchical IRT. Manual numpy log-likelihood (no PyMC rebuild). Graceful degradation for missing models. 60 tests. See ADR-0063, `analysis/design/ppc.md`.

### ~~4. Optimal Classification~~ — Done (Phase 17)

Completed 2026-02-28. Bundled with W-NOMINATE in Phase 17. See Completed Phases table above.

### ~~5. Latent Class Mixture Models~~ — Done (Phase 10)

Completed 2026-02-28. Bernoulli mixture model (LCA) on binary vote matrix using StepMix. BIC model selection, Salsa effect detection, IRT cross-validation, Phase 5 ARI comparison, within-party LCA. Confirms null result: correct generative model for binary votes. Deep dive: `docs/latent-class-deep-dive.md`, design: `analysis/design/lca.md`.

### ~~6. Bipartite Bill-Legislator Network~~ → Done (Phase 12)

Completed 2026-02-28. Bipartite bill-legislator network preserving two-mode structure for bill-centric analysis. Bill polarization scores, bridge bills (bipartite betweenness), Newman-weighted bill projection with Leiden community detection, BiCM backbone extraction (maximum-entropy null model with analytical p-values). Phase 6 comparison: edge Jaccard, community NMI/ARI, hidden alliances. 50 new tests. Deep dive: `docs/bipartite-network-deep-dive.md`, design: `analysis/design/bipartite.md`, ADR-0065.

### ~~7. TSA Hardening (Phase 15 Gaps)~~ — Done

**Completed** (2026-02-28). All seven improvements from the TSA deep dive are resolved.

| Gap | Status | Notes |
|-----|--------|-------|
| **Desposato small-group correction** | **Done** | `desposato_corrected_rice()` + `correct_size_bias=True` default. 6 new tests. |
| **Finer penalty grid** | **Done** | `np.linspace(1, 50, 25)` replaces 6-point grid. |
| **Short-session validation** | **Done** | `warnings.warn()` in 3 locations. 2 new tests. |
| **Imputation sensitivity check** | **Done** | `compute_imputation_sensitivity()` integrated into `main()`. 2 new tests. |
| **Variance-change detection test** | **Done** | `TestVarianceChangeDetection::test_detects_variance_change`. |
| **CROPS penalty selection** | **Done** | R `changepoint::cpt.mean(penalty="CROPS")` via subprocess. ADR-0061. |
| **Bai-Perron confidence intervals** | **Done** | R `strucchange::breakpoints()` + `confint()` via subprocess. ADR-0061. |

R enrichment is optional — `--skip-r` for Python-only mode. 21 new tests (85 total). Full analysis: [`docs/tsa-deep-dive.md`](tsa-deep-dive.md). Design doc: [`analysis/design/tsa.md`](../analysis/design/tsa.md). ADRs: 0057, 0061.

---

## ~~Bill Text NLP Pipeline~~ — All Complete

Full survey and technical design: [`docs/bill-text-nlp-deep-dive.md`](bill-text-nlp-deep-dive.md).

### ~~BT1. Bill Text Retrieval~~

**Completed (2026-03-02).** Separate `tallgrass-text` CLI (`just text 2025`) downloads bill PDFs and extracts text via `pdfplumber`. Multi-state-ready architecture: `StateAdapter` Protocol + `KansasAdapter` first implementation. Shared bill discovery module (`bills.py`) extracted from scraper — zero regressions. Output: 5th CSV (`{name}_bill_texts.csv`) with columns `session`, `bill_number`, `document_type`, `version`, `text`, `page_count`, `source_url`. Introduced + supplemental note document types. 108 new tests.

**Dependencies:** `pdfplumber`
**Prerequisite for:** BT2, BT3, BT4, BT5

### ~~BT2. Bill Text Analysis — Phase 18~~

**Completed (2026-03-02).** BERTopic topic modeling on full bill text (FastEmbed ONNX embeddings + HDBSCAN + c-TF-IDF). Optional CAP 20-category policy classification via Claude Sonnet API with content-hash caching. Bill similarity via cosine distance on 384-dim BGE embeddings. Vote cross-reference: Rice index per topic × party, caucus-splitting scores. 13-section HTML report with conditional CAP sections. No PyTorch — FastEmbed uses ONNX Runtime (~50-100 MB). 53 new tests (2113 total).

**Dependencies:** `bertopic`, `fastembed`, `hdbscan` (core); `anthropic` (optional `[classify]` extra)
**Output:** `bill_topics.csv`, `cap_classifications.csv`, topic distribution plots, policy-area heatmaps, similarity clusters
**Enriches:** Phase 08 (prediction features), Phase 11 (per-topic voting patterns), Phase 12 ("how did legislator X vote on education bills?"), Phase 07 (policy-area-specific indices)

### ~~BT3. Text-Based Ideal Points — Phase 21 (experimental)~~

**Completed (2026-03-03).** Embedding-vote approach: multiplies vote matrix by Phase 18 bill embeddings (384-dim BGE), PCA on legislator text profiles, PC1 = text-derived ideal point. Validates against IRT (flat + hierarchical). Not true TBIP (Vafa, Naidu & Blei, ACL 2020) due to ~92% committee sponsorship — no meaningful author mapping. Standalone with `just tbip`; not in pipeline. Design: `analysis/design/tbip.md`, ADR-0086.

### ~~BT4. Issue-Specific Ideal Points — Phase 19~~

**Completed (2026-03-03).** Topic-stratified flat IRT: runs Phase 04 2PL IRT model on per-topic vote subsets from Phase 18 BERTopic/CAP topic assignments. Answers "how conservative is each legislator on education vs taxes?" Two taxonomies: BERTopic (data-driven) + CAP (standardized). Relaxed convergence thresholds (R-hat < 1.05, ESS > 200) for smaller per-topic models. Cross-topic correlation heatmap, ideological profile matrix, outlier detection. Reuses `build_irt_graph()` / `build_and_sample()` — zero new model code, zero new dependencies. Standalone with `just issue-irt`; not in pipeline. Design: `analysis/design/issue_irt.md`, ADR-0087.

**Why not `issueirt`?** Shin 2024 R package estimates 2D ideal points per topic, but Phase 06 proved Kansas voting is fundamentally 1D. Package is GitHub-only (4 stars, pre-1.0, rstan dependency, uncertain maintenance).

### ~~BT5. Model Legislation Detection — Phase 23~~

**Completed (2026-03-03).** ALEC template matching + cross-state policy diffusion detection. Cosine similarity on BGE embeddings (same 384-dim vector space as Phase 18) with three-tier classification: near-identical (>= 0.95), strong match (>= 0.85), related (>= 0.70). 5-gram word overlap confirms genuine text reuse for strong matches. ALEC corpus (~1,057 model policies) scraped via `just alec` from alec.org/model-policy/. Cross-state bills from MO, OK, NE, CO via OpenStates API v3. 9-section HTML report with interactive tables, similarity distributions, topic heatmap, and match network. Gracefully skips when bill texts + ALEC corpus both missing. 58 tests. Design: `analysis/design/model_legislation.md`, ADR-0089.

---

## Completed (Formerly Deferred)

### ~~Joint Cross-Chamber IRT~~

**Completed (2026-02-24), fixed (2026-02-26).** The hierarchical joint cross-chamber model now runs with bill-matching (ADR-0043): shared `alpha`/`beta` parameters for 71 matched bills (91st) provide natural cross-chamber identification. Bill-matching reduced joint model runtime from 93 min to 31 min by shrinking the problem size (420 unified votes vs 491). Group-size-adaptive priors mitigate small-group convergence failures. Senate convergence now passes all checks. See ADR-0043 and `docs/joint-hierarchical-irt-diagnosis.md`.

---

## Pipeline Audit Findings (2026-03-02)

Full-pipeline review across all 8 bienniums (84th-91st), 17 phases each, plus cross-session and dynamic IRT. Prioritized by severity.

### Critical: Code Bugs

#### ~~A1. Python 2 `except` syntax across 9 call sites~~ — Done

**Fixed 2026-03-02.** Originally fixed with `# fmt: skip` workaround. Permanently resolved by Python 3.14.3 upgrade: PEP 758 makes bracketless `except A, B:` valid syntax meaning "catch both exceptions." All `# fmt: skip` comments removed; ruff now formats correctly.

### High: Systematic Issues (All Bienniums)

#### ~~A2. Joint hierarchical model fails convergence in all 8 bienniums~~ — Done

**Resolved 2026-03-02.** Joint model switched from default to opt-in (`--run-joint`). Stocking-Lord linking is the production cross-chamber alignment method. Report includes linking coefficients + linked ideal points. ADR-0074.

#### ~~A3. 2D IRT (Phase 06) fails convergence in most bienniums~~ — Done

**Resolved 2026-03-02.** Phase 06 removed from `just pipeline` and dashboard. Kansas voting is fundamentally 1D — Dim 2 is noise. Standalone `just irt-2d` preserved for research. ADR-0074.

#### ~~A4. Dynamic IRT Senate convergence failure (cross-session)~~ — Done

**Fixed 2026-03-02.** Root cause: double-standardization bug in informative prior (ADR-0070 regression). Static IRT values (already unit-scale) were re-standardized, destroying per-legislator sign info. Fix: remove re-standardization, use accumulator averaging, widen sigma 0.75→1.5. ADR-0074.

### Medium: Analysis Methodology

#### ~~A5. Prediction: bill passage surprises evaluated in-sample~~ — Done

**Fixed 2026-03-02.** `find_surprising_bills()` now evaluates on holdout test set only (via `test_indices` returned from `train_passage_models`).

#### ~~A6. Prediction: 90% false-positive asymmetry in surprising votes~~ — Done

**Fixed 2026-03-02.** Report now splits surprising votes into "Surprising Nay" (FP: predicted Yea, actual Nay) and "Surprising Yea" (FN: predicted Nay, actual Yea) with interpretation section explaining the base-rate mechanism. ADR-0076.

#### ~~A7. Prediction: per-legislator accuracy lacks minimum sample threshold~~ — Done

**Fixed 2026-03-02.** Added `MIN_VOTES_RELIABLE=10` constant and `reliable` boolean column to per-legislator accuracy output. `detect_hardest_legislators()` now filters to reliable-only before ranking.

#### ~~A8. LCA degenerate class probabilities (91st)~~ — Accepted

**Documented 2026-03-02.** Mathematically expected: ~30 discriminating binary indicators suffice for near-certain classification. Report now adds certainty note when all legislators have max P > 0.99. ADR-0076.

#### ~~A9. Clustering always recovers trivial party split (k=2)~~ — Accepted

**Documented 2026-03-02.** Added ARI-against-party diagnostic to `compare_methods()`. Report annotates when ARI > 0.95, confirming party is the only discrete structure. ADR-0076.

#### ~~A10. Network betweenness sparsity (66-73% zeros)~~ — Done

**Fixed 2026-03-02.** Added harmonic centrality + cross-party edge fraction to `compute_centralities()`. Bridge-builder detection now uses harmonic centrality when graph is disconnected (n_components >= 2), with role label "Within-Party Connector". Connected graphs retain betweenness (original behavior). ADR-0076.

#### ~~A11. House IRT sensitivity to minority threshold~~ — Done

**Documented 2026-03-02.** IRT report now includes ROBUST/SENSITIVE classification in sensitivity table + interpretation section explaining supermajority mechanism and field-standard convention. ADR-0076.

#### ~~A12. Beta-binomial parameter clamping not logged~~ — Done

**Fixed 2026-03-02.** `estimate_beta_params()` now emits a `warnings.warn()` when alpha or beta is clamped to 0.5, including the original and clamped values.

#### ~~A13. Hierarchical shrinkage_pct nulls in synthesis (all bienniums)~~ — Accepted

**Investigated 2026-03-02.** 24.8% null rate is consistent across all 8 bienniums (13.5%-35.7%). The `SHRINKAGE_MIN_DISTANCE=0.5` threshold is statistically justified — without it, values swing wildly (-2222% to +86%) due to near-zero denominators. Downstream impact is minimal: appears as blank cells in one interactive table in the hierarchical report. Not used in synthesis detection, profiles, cross-session, or any scoring logic. The `toward_party_mean` boolean (always non-null for non-anchors) already captures the actionable information. See backlog #2 below.

#### ~~A14. TSA imputation sensitivity silently returns None~~ — Done

**Fixed 2026-03-02.** Now prints "Imputation sensitivity: skipped (insufficient complete cases)" when the check returns None.

#### ~~A15. TSA penalty sensitivity not summarized in run log~~ — Done

**Fixed 2026-03-02.** Now prints "Penalty sweep [1-50]: max N changepoint(s) at penalty=X.X" to the run log.

### Low: Known Limitations

#### ~~A16. Small Senate Democrat groups (all bienniums)~~ — Done

**Documented 2026-03-02.** Hierarchical report key findings now surfaces small-group warning when any party has fewer than 20 legislators, recommending flat IRT for individual positions. ADR-0076.

#### ~~A17. Bipartite BiCM backbone extremely sparse (Senate)~~ — Done

**Fixed 2026-03-02.** Senate BiCM backbone now uses relaxed significance threshold (0.05 vs 0.01 for House) reflecting 10x fewer multiple comparisons. Report adds sparsity caveat when >50% of legislators are isolated. ADR-0076.

#### ~~A18. Bill communities mirror party split (all bienniums)~~ — Accepted

**Documented 2026-03-02.** Report now adds modularity quality gate: when best modularity < 0.10, notes weak community structure mirroring the party divide. ADR-0076.

---

## Code Audit Findings (2026-03-02)

Full-codebase audit: scraper, analysis infrastructure, all 17 phases, tests, config. Prioritized by value.

### Bugs

#### ~~B1. Dead if/else in retry wave logic~~ — Done

**Fixed.** Collapsed identical if/else branches to single `results[url] = future.result()`.

#### ~~B2. Division-by-zero in IRT linking~~ — Done

**Fixed.** Added `ValueError` guards in `link_mean_mean()` and `link_mean_sigma()` when denominators are zero.

#### ~~B3. Duplicate `_parse_vote_tally()` call~~ — Done

**Fixed.** Stored tally in sort tuple `(margin, failure, tally)` to avoid redundant re-parse.

### Dead Code

#### ~~D1. `_itables_init_html()` always returns empty string~~ — Done

**Fixed.** Removed dead function and call site.

#### ~~D2. Empty `NICKNAME_MAP` never populated~~ — Done

**Fixed.** Removed constant and dead conditional block.

### Refactoring: Cross-File Duplication

#### ~~R1. `print_header()` + `save_fig()` duplicated in 21 phase scripts~~ — Done

**Fixed.** Created `analysis/phase_utils.py` with canonical implementations. Replaced local definitions in all 21 files.

#### ~~R2. `load_metadata()` duplicated in 8 phases~~ — Done

**Fixed.** `load_metadata()` (tuple) and `load_legislators()` (single) in `phase_utils.py`. Replaced in pca, clustering, network, bipartite, umap, lca, irt, wnominate.

#### ~~R3. `normalize_name()` duplicated in 2 phases~~ — Done

**Fixed.** `normalize_name()` and `_LEADERSHIP_SUFFIX_RE` in `phase_utils.py`. Replaced in `cross_session_data.py` and `dynamic_irt_data.py`.

#### ~~R4. JS key unquoting duplicated in scraper~~ — Done

**Fixed.** Extracted `_parse_js_array()` static method, shared by `_parse_js_bill_data()` and `_parse_js_member_data()`.

#### R5. Chamber-from-slug extraction in 3 locations — Skipped

Only 2 occurrences in same file (`scraper.py`) with different case conventions. Not worth extracting.

#### ~~R6. Test helpers duplicated across test files~~ — Done

**Fixed.** Created `tests/factories.py` with shared `make_legislators()`, `make_votes()`, `make_rollcalls()` factories. `slug_column` parameter handles scraper/analysis schema split. Migrated 5 files: test_cross_session, test_dynamic_irt, test_mca, test_tsa, test_integration_pipeline. Domain-specific helpers (IRT data, vote matrices) kept local.

#### R7. `dict(zip(col.to_list(), col.to_list()))` pattern (14+ occurrences)

Clustering, bipartite, network, profiles all build lookup dicts the same way. Idiomatic enough to leave as-is.

### Refactoring: Large Functions

#### ~~R8. `_parse_vote_page()` — 191 lines~~ — Done

**Fixed.** Extracted `_extract_bill_title()`, `_extract_chamber_motion_date()`, `_parse_vote_categories()` as static methods. `_parse_vote_page()` is now a thin coordinator. `_parse_vote_categories()` returns `new_legislators` instead of mutating `self.legislators` as a side effect.

#### ~~R9. `enrich_legislators()` — 64 lines~~ — Done

**Fixed.** Extracted `_extract_party_and_district()` static method with post-2015/pre-2015 fallbacks. `enrich_legislators()` loop body reduced to a single method call.

### Efficiency

#### ~~E1. O(n²) bill polarization loop in bipartite~~ — Done

**Fixed.** Vectorized with numpy integer masks and dot products. Inner loop eliminated entirely.

#### ~~E2. Repeated `.to_list()` on same columns~~ — Done

**Fixed.** Cached `legislator_slug` list once, reused for 3 dict constructions in `plot_voting_blocs()`.

#### ~~E3. DataFrame created per number format in report~~ — Done

**Fixed.** Batched all format expressions into single `with_columns()` call.

### Error Handling

#### ~~H1. `irt_linking.py:64-87` — No KeyError guard on matched_bills~~ — Done

**Fixed.** Added membership check; unmatched vote IDs are skipped and counted as dropped.

#### ~~H2. `experiment_monitor.py:245` — `os.open()` unguarded~~ — Done

**Fixed.** Wrapped with `try/except OSError` providing error context.

#### ~~H3. `scraper.py:514` — KLISS API type assumption~~ — Done

**Fixed.** Added `isinstance(data, dict)` guard; non-list/non-dict data falls through to empty list.

### Tests

#### ~~T1. `@pytest.mark.slow` defined but unused~~ — Done

**Fixed.** 15 tests in `test_scraper_http.py` marked slow: `TestGetErrorClassification` (7), `TestGetRetries` (6), `TestFetchMany` retry-wave tests (2). `just test-fast` now skips them (0.17s vs 199s).

#### ~~T2. Two weak assertions~~ — Done

**Fixed.** `test_clustering.py`: added `0.0 <= avg_loyalty <= 1.0` range check. `test_dynamic_irt.py`: added `hasattr(model, "named_vars")` structure check.

### Estimated Impact (Realized)

| Category | Done | Total | Lines changed |
|----------|------|-------|---------------|
| Bugs (B1-B3) | 3/3 | 3 | ~15 lines fixed |
| Dead code (D1-D2) | 2/2 | 2 | ~20 lines removed |
| Cross-file dedup (R1-R4, R6) | 5/5 | 7 | ~400 lines removed, `phase_utils.py` + `factories.py` created |
| Large functions (R8-R9) | 2/2 | 2 | 4 static methods extracted |
| Efficiency (E1-E3) | 3/3 | 3 | E1: vectorized bipartite |
| Error handling (H1-H3) | 3/3 | 3 | Crash prevention |
| Tests (T1-T2) | 2/2 | 2 | T1: slow markers, T2: weak assertions |

2084 tests passing, lint clean, typecheck clean.

---

## Code Quality Backlog (Post-Audit Follow-Up)

Items from the 2026-03-02 code audit follow-up review. The 5 primary findings (A-E) are resolved in ADR-0080; these are the remaining lower-priority opportunities identified during the post-fix assessment.

### ~~CQ1. Manifest key regression test~~ — Done

Three tests in `TestManifestKeyConsistency` parse `manifests.get()` calls from source and assert every key is a valid `UPSTREAM_PHASES` entry. Covers `synthesis.py`, `synthesis_report.py`, and `load_all_upstream()` storage. Prevents ADR-0080 finding A from recurring.

**Files:** `tests/test_synthesis.py`

### ~~CQ2. Consolidate `_resolve_phase_dir()` into `resolve_upstream_dir()`~~ — Done

**Fixed 2026-03-02.** Replaced private `_resolve_phase_dir()` in `synthesis_data.py` with the public `resolve_upstream_dir()` from `run_context.py`. Single import swap, functionally identical (same 3-level fallback cascade).

### ~~CQ3. Leadership suffix regex deduplication~~ — Done

**Fixed 2026-03-02.** Consolidated `r"\s*-\s+.*$"` regex from 3 definitions to 1. `phase_utils.normalize_name()` and `external_validation_data.normalize_our_name()` now use `strip_leadership_suffix()` from `run_context.py`.

### ~~CQ4. EDA heatmap label lookup — precompute dict~~ — Done

**Fixed 2026-03-02.** Replaced double-filter list comprehension in `eda.py` with a precomputed `slug_to_name` dict lookup (matching the `slug_to_party` pattern already used nearby).

---

## ~~Backlog~~ — All Cleared

### ~~1. External Validation Name Matcher: District Tiebreaker~~ — Done

**Fixed 2026-03-02.** Both Phase 14 (Shor-McCarty) and Phase 18 (DIME) now use district-based disambiguation when multiple external candidates share a last name. SM version extracts year-specific `hdistrict{YYYY}`/`sdistrict{YYYY}` columns; DIME version parses the `district` string (e.g., `"KS-113"`). If multiple candidates match on last name: prefer the district match; if no district match, reject entirely (no match is better than a wrong match). Single-candidate matches are unaffected. 3 confirmed incorrect matches corrected (Bethell 84th, Dannebohm 86th, Weber 86th). 6 new tests across both test files.

### ~~2. Null `hier_shrinkage_pct` in Synthesis~~ — Accepted (Working as Designed)

**Investigated 2026-03-02.** Deep dive across all 8 bienniums found: 24.8% overall null rate (13.5%-35.7% per chamber-session), consistent across all bienniums — not 84th-specific. The `SHRINKAGE_MIN_DISTANCE=0.5` threshold is statistically justified: without it, shrinkage percentages swing from -2222% to +86% due to near-zero denominators (legislators already at their party mean). The `toward_party_mean` boolean (always non-null) captures the actionable direction. Downstream impact is minimal — one table column in the hierarchical report; not used in synthesis detection, profiles, cross-session, or any scoring logic. See also A13.

### ~~3. 84th Biennium Pipeline Re-run~~ — Done

**Re-run 2026-03-02.** Fresh unified pipeline run on current code. Picks up: district tiebreaker name matching, Python 2 `except` syntax fixes, prediction in-sample leakage fix, per-legislator min threshold, network edge weight KeyError fix, R1-R20 report enhancements, dashboard generation. Previous run (`84-260301.4`) already had the 3 originally-named bug fixes (Carey Unity, ARI ordering, IRT sign-flip).

### ~~4. Symlink Race: Pipeline-Only `latest` Updates~~ — Done

**Completed 2026-03-01.** `RunContext` now tracks `_explicit_run_id` to distinguish pipeline (explicit) from standalone (auto-generated) run IDs. Only explicit run IDs update `latest` and report convenience symlinks. 3 new tests + 1 updated test. ADR-0070.

### ~~5. Dynamic IRT Sign Identification~~ — Done

**Completed 2026-03-01.** Informative `xi_init` prior from static IRT: loads Phase 04 ideal points, maps to global roster, uses `Normal(xi_init_mu, 0.75)` instead of `Normal(0, 1)`. Transfers the well-identified sign convention. Post-hoc correction (ADR-0068) retained as diagnostic safety net. ADR-0070.

### ~~6. Dynamic IRT Senate Convergence~~ — Done

**Completed 2026-03-01.** Adaptive tau: small chambers (<80 legs) get `HalfNormal(0.15)` + global tau; large chambers keep `HalfNormal(0.5)` + per-party tau. MCMC budget increased to 2000/2000/4. `--tau-sigma` CLI override for experimentation. 5 new graph construction tests. ADR-0070.

---

## ~~Report Enhancement Backlog~~ — R1-R20 Done (ADR-0069, ADR-0071)

Prioritized improvements to HTML report output, based on a comprehensive survey of the open-source landscape, academic standards, and general-audience best practices. Full analysis: [`docs/report-enhancement-survey.md`](report-enhancement-survey.md).

**R1-R13 completed 2026-03-01** (ADR-0069). **R14-R20 completed 2026-03-02** (ADR-0071). Three new section types (`KeyFindingsSection`, `InteractiveTableSection`, `InteractiveSection`), seven new dependencies (itables, plotly, pyvis, folium, geopandas + IntersectionObserver JS), five new analytical features (BPI, plus-minus, cutting lines, swing votes, coalition labeler), all 22+ report builders enhanced with key findings. Tier 3 added: district choropleths, full voting records, iframe dashboard, CSV downloads, freshmen cohort, bloc stability, scrollytelling.

### ~~Tier 1: High-Impact, Lower Effort~~ — All Done

| # | Enhancement | Status |
|---|-------------|--------|
| R1 | **Key Findings section** at top of every report | **Done** — all 22+ report builders |
| R2 | **ITables for legislator tables** (sort, search, filter) | **Done** — 10+ large tables converted |
| R3 | **Party ideal point density overlay** | **Done** — `plot_party_density()` in Phase 04 |
| R4 | **Headline-style section titles** | **Done** — Synthesis data-driven, others use subtitle |
| R5 | **Absenteeism as analysis** | **Done** — EDA report, strategic absence flags |
| R6 | **ICCs in flat IRT report** | **Done** — `plot_icc_curves()` in Phase 04 |

### ~~Tier 2: High-Impact, Medium Effort~~ — All Done

| # | Enhancement | Status |
|---|-------------|--------|
| R7 | **Per-vote spatial visualization** (cutting lines) | **Done** — `plot_cutting_lines()` in Phase 04 |
| R8 | **Swing vote identification** | **Done** — `identify_swing_votes()` in Phase 04 |
| R9 | **Plotly for ideology scatter plots** | **Done** — IRT, PCA, Indices interactive scatters |
| R10 | **PyVis for network graphs** | **Done** — Phase 06 interactive network |
| R11 | **Predicted vs. actual ("plus-minus")** | **Done** — `compute_plus_minus()` + dumbbell chart |
| R12 | **Vote-based bipartisanship index** | **Done** — `compute_bipartisanship_index()` + scatter |
| R13 | **Named/described coalitions** | **Done** — `coalition_labeler.py` with auto-naming |

### ~~Tier 3: High-Impact, Higher Effort~~ — All Done

| # | Enhancement | Status |
|---|-------------|--------|
| R14 | **Folium district choropleth maps** | **Done** — `analysis/01_eda/geographic.py`, TIGER/Line GeoJSON, folium+geopandas |
| R15 | **Full voting record per legislator** | **Done** — `profiles_data.py:build_full_voting_record()`, `--full-record` flag, ITables |
| R16 | **Iframe dashboard** (lightweight, not Quarto) | **Done** — `analysis/dashboard.py`, sidebar nav + iframe, `just dashboard`, auto-called by pipeline |
| R17 | **Downloadable CSV alongside reports** | **Done** — `DownloadSection` in report.py, `RunContext.export_csv()`, 28+ export calls |
| R18 | **Freshmen cohort analysis** | **Done** — `cross_session_data.py:analyze_freshmen_cohort()`, KDE density + KS/t-tests |
| R19 | **Voting bloc stability tracking** | **Done** — `cross_session_data.py:compute_bloc_stability()`, Plotly Sankey, ARI, transition matrix |
| R20 | **Scrollytelling in Synthesis** | **Done** — `ScrollySection` in report.py, IntersectionObserver JS, `--scrolly` flag |

### Tier 4: Nice-to-Have — Milestoned

All Tier 4 items plus remaining code audit items have detailed implementation documents in [`docs/milestones/`](milestones/). Each milestone is self-contained with file paths, function signatures, test strategy, and documentation requirements.

| # | Enhancement | Milestone | Notes |
|---|-------------|-----------|-------|
| R21 | **Parliament/hemicircle charts** | **Done** — `analysis/viz_helpers.py`, hemicycle in EDA report per chamber | Plotly scatter on polar coords |
| R22 | **Sankey diagrams** for bill flow | [M5](milestones/m5-bill-lifecycle.md) | **Done** — `BillAction` dataclass, `_bill_actions.csv`, lifecycle Sankey in EDA report |
| R23 | **Ridgeline plots** for ideology | [M6](milestones/m6-ridgeline-plots.md) | **Done** — `plot_ridgeline_ideology()` in dynamic IRT report |
| R24 | **Animated scatter** (Gapminder) | [M7](milestones/m7-animated-scatter.md) | **Done** — `plot_animated_scatter()` in dynamic IRT report |
| R25 | **Descriptive alt text** | [M3](milestones/m3-accessibility-alt-text.md) | **Done.** WCAG 2.1 AA — alt_text on all FigureSections + aria_label on InteractiveSections across 23 report files |
| R26 | **Prediction enhancement** | [M8](milestones/m8-prediction-enhancement.md) | **Done** — `sponsor_slugs` in scraper, `sponsor_party_R` feature, passage SHAP/importance, stratified accuracy by bill prefix. Downstream: Phase 11 `n_bills_sponsored` scorecard, Phase 12 sponsorship section (ADR-0081). |

Additional milestones from code audit:

| Milestone | Scope | Document |
|-----------|-------|----------|
| M1 | ~~`@pytest.mark.slow` + test helper consolidation~~ | **Done** — slow markers on retry tests; shared `factories.py` for test data |
| M2 | ~~Extract `_parse_vote_page()` + `enrich_legislators()` helpers~~ | **Done** |

**All milestones completed.** M1, M2, M3, M5, M6, M7, M8 done.

### Key Library Additions (All Integrated)

| Library | Purpose | Integration Point |
|---------|---------|-------------------|
| [Plotly](https://plotly.com/python/) (v6) | Interactive scatter, bar, heatmap, Sankey | `InteractiveSection` in `report.py` |
| [PyVis](https://pyvis.readthedocs.io/) | Interactive network graphs | Phase 06, 06b |
| [ITables](https://github.com/mwouts/itables) (v2.6+) | Searchable/sortable tables | All phases with legislator tables |
| [Folium](https://python-visualization.github.io/folium/) | Interactive choropleth maps | EDA geographic sections |
| [GeoPandas](https://geopandas.org/) | GeoJSON/shapefile handling | District boundary loading |

---

## Other Backlog

### ~~Per-Phase Results Primers~~ — Done

All 18 phases define `*_PRIMER` strings (150-200 lines of Markdown each) that RunContext auto-writes to `README.md` in every phase output directory. Each primer covers: Purpose, Method, Inputs, Outputs, Interpretation Guide, and Caveats. The project-level primer (`docs/analysis-primer.md`) provides the general-audience overview.

### ~~Test Suite Expansion~~ — Done

2084 tests across scraper and analysis modules. All passing. Three gaps closed:
- **Integration tests**: `test_integration_pipeline.py` — synthetic data → EDA → PCA pipeline chain, RunContext lifecycle, upstream resolution (26 tests)
- **HTML report structural tests**: `test_report_structure.py` — TOC anchors, section ordering, numbering, container types, empty report, CSS embedding, make_gt integration (22 tests)
- **Pytest markers**: `@pytest.mark.scraper` (264 tests), `@pytest.mark.integration` (29 tests), `@pytest.mark.slow` (24 tests). Registered in `pyproject.toml`. Recipes: `just test-scraper`, `just test-fast`

---

## Infrastructure: PostgreSQL + Django

Long-term storage and web platform for multi-state legislative data. Full analysis: [`docs/data-storage-deep-dive.md`](data-storage-deep-dive.md).

**Current state:** CSVs in `data/kansas/` — adequate for single-state, becomes unwieldy at 50 states. KanFocus backfill (ADR-0088) is extending coverage to 78th-91st (1999-2026), adding ~14 bienniums of data.

### DB1. Django Project Scaffolding (COMPLETE — 2026-03-03)

Django project at `src/web/` with 8 models (State, Session, Legislator, RollCall, Vote, BillAction, BillText, ALECModelBill) mapping to the CSV schema. PostgreSQL 16 via Docker Compose. Django admin with list/filter/search on all models. 63 tests (model creation, unique constraints, nullable fields, FK cascades, `__str__`, admin registration). ADR-0090.

**Deliverables:**
- Django project with models, migrations, admin registration
- `docker-compose.yml` for local PostgreSQL
- Settings split: `base.py`, `local.py`, `test.py`
- Justfile recipes: `db-up`, `db-down`, `db-migrate`, `db-admin`, `db-shell`, `test-web`
- `web` dependency group (Django 5.2 LTS + psycopg3) — not a core dependency

### DB2. CSV-to-PostgreSQL Loader (COMPLETE — 2026-03-04)

Three management commands that bulk-load CSVs into PostgreSQL. Delete-and-reload per session inside `@transaction.atomic` — idempotent and safe. Legislators, rollcalls, bill actions, and bill texts use psycopg3 `COPY FROM STDIN`; votes use `bulk_create` for FK resolution (slug→Legislator.id, vote_id→RollCall.id). ADR-0094.

**Deliverables:**
- `manage.py load_session 91st_2025-2026` — loads all CSVs for one session
- `manage.py load_all` — discovers all session directories + loads ALEC corpus
- `manage.py load_alec` — loads ALEC model legislation corpus
- `--dry-run` validates CSVs without writing; `--skip-bill-text` skips large bill text files
- Justfile recipes: `just db-load`, `just db-load-all`, `just db-load-alec`

**Prerequisite:** DB1

### DB3. Scraper Post-Hook (COMPLETE — 2026-03-04)

**Completed.** `--auto-load` flag on `tallgrass`, `tallgrass-text`, and `tallgrass-kanfocus` CLIs. After a successful scrape, invokes `load_session` management command via subprocess. Scraper remains Django-free — `db_hook.py` shells out to `manage.py`. Fails soft if Django/PostgreSQL unavailable (prints warning, scrape output preserved). Shared helper at `src/tallgrass/db_hook.py`. ADR-0095.

```bash
just scrape 2025 --auto-load   # scrape + load into PostgreSQL
just text 2025 --auto-load     # fetch bill text + load into PostgreSQL
just kanfocus 2025 --auto-load # KanFocus scrape + load into PostgreSQL
```

**Prerequisite:** DB2

### DB4. REST API — COMPLETE (2026-03-04)

Read-only public API via [Django Ninja](https://django-ninja.dev/) (`>=1.5,<2`). Ninja chosen over DRF for type-hint alignment (Pydantic v2 schemas), built-in pagination/filtering/throttling/OpenAPI, and consistency with our BaseRally project. Full analysis: [`docs/rest-api-deep-dive.md`](rest-api-deep-dive.md). ADR-0096.

**Endpoints** (all `GET`, base path `/api/v1/`):

| Endpoint | Records | Key Filters |
|----------|---------|-------------|
| `/sessions/`, `/sessions/{id}/` | ~16 | `state`, `is_special` |
| `/legislators/`, `/legislators/{id}/` | ~2,000 | `session`, `chamber`, `party`, `search` |
| `/rollcalls/`, `/rollcalls/{id}/` | ~8,000 | `session`, `chamber`, `bill_number`, `passed`, `date_from/to`, `search` |
| `/votes/` | ~650,000 | `session`, `legislator_slug`, `rollcall`, `vote` |
| `/bill-actions/` | ~23,000 | `session`, `bill_number`, `chamber`, `action_code` |
| `/bill-texts/`, `/bill-texts/{id}/` | ~1,600 | `session`, `bill_number`, `document_type` |
| `/alec/`, `/alec/{id}/` | ~1,057 | `category`, `task_force`, `search` |
| `/health` | — | — |

**Delivered:**
- `legislature/api/` package (schemas, filters, pagination, throttling, 7 endpoint modules)
- 7 composite database indexes (new migration)
- `django-ninja>=1.5,<2` in `web` dependency group
- 64 tests (`@pytest.mark.web`) — endpoints, filtering, pagination, schema validation
- Auto-generated Swagger UI at `/api/v1/docs`
- IP-based rate limiting: 60/min lists, 120/min details
- List vs Detail schemas — lists omit large text fields; rollcall detail nests votes
- Semicolon-joined fields (`sponsor_slugs`, `committee_names`) parsed into JSON arrays

**Prerequisite:** DB1

### DB5. Analysis Pipeline Database Integration (COMPLETE)

PostgreSQL is the default data source for all 27 analysis phases. `analysis/db.py` uses raw SQL + psycopg3 + Polars `read_database()` — no Django ORM dependency. CSV is the automatic fallback when the DB is unavailable. `--csv` flag forces CSV-only mode.

**Deliverables (all complete):**
- `analysis/db.py` — centralized DB loading (5 SQL queries, 5 routing functions, connection caching)
- 9 call sites updated across 8 phase files + `phase_utils.py`
- `--csv` flag on all affected phase argparsers
- `psycopg[binary]>=3.2` in `dev` dependency group
- Unit tests (mocked) + integration tests (`@pytest.mark.web`)
- ADR-0099

**Prerequisite:** DB2

### DB6. Multi-State Adapter

Extend the `StateAdapter` Protocol (from bill text, ADR-0083) pattern to the vote scraper. Each state gets its own adapter producing the same CSV/DB schema. PostgreSQL's `state_code` foreign key and Django's `Session.state` field are already designed for this.

**Deliverables:**
- Second state adapter (candidate: Missouri, Nebraska, or Oklahoma — geographic neighbors with accessible legislature sites)
- Shared schema validation across states
- Cross-state queries in API and analysis pipeline

**Prerequisite:** DB3, one state beyond Kansas operational

### Implementation Order

```
DB1 (scaffolding) ─→ DB2 (loader) ─→ DB3 (post-hook)
       │                                      │
       └──→ DB4 (API)               DB5 (pipeline integration)
                                              │
                                    DB6 (multi-state)
```

DB1-DB5 are complete — data flows into PostgreSQL, is queryable via REST API, and is the default data source for the analysis pipeline. DB6 is the remaining future work and the long-term goal that motivates the entire effort.

---

## Explicitly Rejected

| Method | Why |
|--------|-----|
| emIRT (fast EM ideal points) | R-only; PyMC gives full posteriors; speed not a bottleneck for single-state |
| Vote-type-stratified IRT (IssueIRT) | Data doesn't support it: overrides are party-line (98%/1%), other types too few (N < 56) |
| Strategic absence modeling (idealstan) | 2.6% absence rate, 22 "Present and Passing" instances — negligible impact |
| Dynamic IRT within biennium | 2-year window too short; cross-session handles between-biennium |
| GGUM unfolding models | No extreme-alliance voting pattern in Kansas data |
| LLM legislative agents | Too experimental; XGBoost already at 0.98 AUC |
| TBIP text-based ideal points | ~~No full bill text available~~ — **Unrejected.** Bill text retrieval planned (BT1), TBIP planned as Phase 21 (BT3). See `docs/bill-text-nlp-deep-dive.md`. |

See `docs/method-evaluation.md` for detailed rationale on each rejection.

---

## Pending Operational Tasks

| Item | Command | Context |
|------|---------|---------|
| **Run 83rd pipeline** | `just pipeline 2009-10` | KanFocus data downloaded + DB-loaded (1,180 rollcalls, 66,575 votes, 170 legislators). Needs full 27-phase analysis. |
| ~~**Investigate KanFocus scraper duplicate output**~~ | — | **Resolved.** Deep dive confirmed the original duplicates were an operational issue (write interruption), not a reproducible code bug. A clean re-run from cache produces 1,186 rollcalls with 0 duplicates. Three defensive improvements shipped: (1) rollcall dedup by `vote_id` in `save_csvs()`, (2) early dedup + tally-mismatch warning in `convert_to_standard()`, (3) parser fix for `Result:` regex bleeding into "All Members" table header (8 affected records across 5 bienniums). 7 new tests. |

---

## IRT Sign Flip Fix (Supermajority Chambers)

**Status:** Documented, code fix pending.

PCA-based anchor selection produces sign flips in supermajority chambers where a rebel faction within the majority party votes with the minority. The horseshoe effect folds far-right rebels onto the same end of the latent dimension as Democrats, and `orient_pc1()` + `select_anchors()` lock in the wrong polarity. The model's shape is correct — only the sign is wrong.

**Evidence:** In the 79th Kansas Senate (2001-02), Tim Huelskamp (ultra-conservative, later Freedom Caucus) is placed at xi = -3.26 ("most liberal"), while Sandy Praeger (moderate R) is the conservative anchor at +1.00. Cross-party contested vote agreement confirms this is inverted: Huelskamp has low agreement with Democrats on contested votes, Praeger has high agreement.

**Proposed fix — post-hoc sign validation:**

1. After MCMC, identify contested votes (both parties split, ≥10% threshold)
2. Compute per-legislator cross-party agreement rate on contested votes
3. Correlate agreement rate with ideal point magnitude
4. If correlation is positive (extremes agree more with the other party), negate all ideal points and discrimination parameters — selecting the other posterior mode
5. Add supermajority diagnostic: flag when within-party variance > between-party gap

**Affected code:**
- `analysis/05_irt/irt.py` — `select_anchors()`, post-MCMC validation
- `analysis/07_hierarchical/hierarchical.py` — post-MCMC validation
- `analysis/02_pca/pca.py` — `orient_pc1()` (upstream, may need awareness)

**Documentation:**
- Deep dive: [`docs/irt-sign-identification-deep-dive.md`](irt-sign-identification-deep-dive.md)
- ADR-0101: Party-aware IRT anchor selection (needs update to reflect sign flip finding)

---

## Scraper Maintenance

| Item | When | Details |
|------|------|---------|
| **Resume KanFocus backfill** | **Next session** | 91st complete. Resume from 90th: `PYTHONUNBUFFERED=1 bash scripts/kanfocus_all.sh --delay 12`. Script auto-skips cached pages. 90th had 72 pages cached when stopped. Remaining: 90th→78th (13 bienniums). Use `--delay 12` during business hours. |
| Update `CURRENT_BIENNIUM_START` | 2027 | Change from 2025 to 2027 in `session.py` |
| Add special sessions | As needed | Add year to `SPECIAL_SESSION_YEARS` in `session.py`, scrape, then `just merge-special <year>` |
| Merge special sessions | After scraping | `just merge-special all` merges into parent bienniums (ADR-0082) |
| ~~Fix Shallenburger suffix~~ | ~~Done~~ | Fixed in analysis via `strip_leadership_suffix()` in `run_context.py` — applied at every CSV load point across all phases (ADR-0014). Scraper stores the raw name; analysis handles display. |

---

## All 29 Analytic Methods — Status

| # | Method | Category | Status |
|---|--------|----------|--------|
| 01 | Data Loading & Cleaning | DATA | Completed (EDA) |
| 02 | Vote Matrix Construction | DATA | Completed (EDA) |
| 03 | Descriptive Statistics | EDA | Completed (EDA) |
| 04 | Missing Data Analysis | EDA | Completed (EDA) |
| 05 | Rice Index | IDX | Completed (Indices) |
| 06 | Party Unity | IDX | Completed (Indices) |
| 07 | ENP | IDX | Completed (Indices) |
| 08 | Maverick Scores | IDX | Completed (Indices) |
| 09 | PCA | DIM | Completed (PCA) |
| 10 | MCA / Correspondence Analysis | DIM | Completed (MCA, Phase 2c) |
| 11 | UMAP / t-SNE | DIM | Completed (UMAP, Phase 2b) |
| 12 | W-NOMINATE | DIM | Completed (W-NOMINATE + OC, Phase 17) — all 8 bienniums |
| 13 | Optimal Classification | DIM | Completed (W-NOMINATE + OC, Phase 17) — all 8 bienniums |
| 14 | Beta-Binomial Party Loyalty | BAY | Completed (Beta-Binomial, Phase 7b) |
| 15 | Bayesian IRT (1D) | BAY | Completed (IRT) |
| 16 | Hierarchical Bayesian Model | BAY | Completed (Hierarchical IRT, Phase 8) |
| 17 | Posterior Predictive Checks | BAY | **Done** — Phase 4 basic + Phase 08 standalone (6/8 bienniums) |
| 18 | Hierarchical Clustering | CLU | Completed (Clustering) |
| 19 | K-Means / GMM Clustering | CLU | Completed (Clustering) |
| 20 | Co-Voting Network | NET | Completed (Network) |
| 21 | Bipartite Bill-Legislator Network | NET | Completed (Phase 12) |
| 22 | Centrality Measures | NET | Completed (Network) |
| 23 | Community Detection | NET | Completed (Network) |
| 24 | Vote Prediction | PRD | Completed (Prediction) |
| 25 | SHAP Analysis | PRD | Completed (Prediction) |
| 26 | Ideological Drift | TSA | Completed (TSA, Phase 15) |
| 27 | Changepoint Detection | TSA | Completed (TSA, Phase 15) |
| 28 | Latent Class Mixture Models | CLU | Completed (LCA, Phase 10) |
| 29 | Dynamic Ideal Points (Martin-Quinn) | TSA | Completed (Dynamic IRT, Phase 16) |
| 30 | DIME/CFscores External Validation | VAL | Completed — Phase 18 (ADR-0062) |
| 31 | Standalone Posterior Predictive Checks | BAY | **Done** — Phase 08 (ADR-0063), 6/8 bienniums (ADR-0073) |
| 32 | TSA Hardening (Desposato, CROPS, validation) | TSA | Completed — item #7 above |

| 33 | Bill Text Topic Modeling (BERTopic) | NLP | Completed — Phase 18 (BT2) |
| 34 | Bill Text Policy Classification (CAP) | NLP | Completed — Phase 18 (BT2) |
| 35 | Text-Based Ideal Points (TBIP) | NLP | Completed — Phase 21 (BT3, embedding-vote approach, ADR-0086) |
| 36 | Issue-Specific Ideal Points | BAY | Completed — Phase 19 (BT4, topic-stratified IRT, ADR-0087) |
| 37 | Model Legislation Detection | NLP | Completed — Phase 23 (BT5, ALEC + cross-state, ADR-0089) |

**Score: 37 completed, 0 planned, 6 rejected = 43 total**

Note: Methods 29-32 are additions beyond the original 28 (Dynamic Ideal Points, DIME/CFscores, Standalone PPC, TSA Hardening). Methods 33-37 are the bill text NLP pipeline. All 5 completed 2026-03-02/03.

---

## Key Architectural Decisions Still Standing

- **Polars over pandas** everywhere
- **Python-first, R where necessary** — R allowed via subprocess for field-standard methods with no Python equivalent (W-NOMINATE, Optimal Classification in Phase 17; emIRT in Phase 16; CROPS/Bai-Perron in Phase 15). Install: `brew install r` + `install.packages(c("wnominate", "oc", "pscl", "jsonlite", "changepoint", "strucchange"))`
- **Ruff + ty + uv** — all-Astral toolchain (lint, type check, package management)
- **IRT ideal points are the primary feature** — prediction confirmed this; everything else is marginal
- **Chambers analyzed separately** unless explicitly doing cross-chamber comparison
- **Tables never truncated** in analysis reports — full data for initial analysis
- **Nontechnical audience** — all visualizations self-explanatory, plain-English labels, annotated findings
