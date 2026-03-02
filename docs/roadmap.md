# Roadmap

What's been done, what's next, and what's on the horizon for the Tallgrass analytics pipeline.

**Last updated:** 2026-03-01 (report enhancement survey — 26 items added to backlog)

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
| 17 | W-NOMINATE + OC Validation | 2026-02-28 | Field-standard legislative scaling comparison. W-NOMINATE (Poole & Rosenthal) + Optimal Classification (Poole 2000) via R subprocess. 3×3 correlation matrix (IRT/WNOM/OC), per-chamber scatter plots, 2D W-NOMINATE space, eigenvalue scree, fit statistics. Validation-only (does not feed downstream). Deep dive: `docs/w-nominate-deep-dive.md`, design: `analysis/design/wnominate.md`, ADR-0059. |
| 4c | PPC + LOO-CV Model Comparison | 2026-02-28 | PPC battery (Yea rate, accuracy, GMP, APRE) + item/person fit + Yen's Q3 local dependence + LOO-CV model comparison across flat 1D, 2D IRT, and hierarchical IRT. Manual numpy log-likelihood (no PyMC rebuild). Graceful degradation for missing models. 60 tests. Design: `analysis/design/ppc.md`, ADR-0063. |
| 5b | Latent Class Analysis | 2026-02-28 | Bernoulli mixture (LCA) on binary vote matrix via StepMix. BIC model selection (K=1..8), Salsa effect detection (profile correlations), IRT cross-validation, Phase 5 ARI comparison, within-party LCA. Correct generative model for binary data. Deep dive: `docs/latent-class-deep-dive.md`, design: `analysis/design/lca.md`. |
| 6b | Bipartite Bill-Legislator Network | 2026-02-28 | Two-mode network (legislators × bills). Bill polarization, bridge bills, bill communities (Leiden on Newman projection), BiCM backbone extraction (statistical validation via maximum-entropy null model), Phase 6 comparison. Deep dive: `docs/bipartite-network-deep-dive.md`, design: `analysis/design/bipartite.md`, ADR-0065. |
| — | 84th Biennium Pipeline Stress Test | 2026-03-01 | Full 17-phase pipeline + PPC + External Validation + DIME on 2011-12 data. 8 bug fixes (column naming, IRT sensitivity sign flip, arviz/matplotlib deprecations, DIME party type). LCA class membership tables added. ADR-0066. |
| — | Open-Source Readiness | 2026-03-01 | MIT LICENSE, README, CONTRIBUTING, pyproject metadata, CI expansion (lint+typecheck+test). 9 bug fixes (except syntax, Jinja2 autoescape, sign-flip DuplicateError, circular import, PID race). Phase 04b tests (16 new). 1701 total tests. ADR-0067. |

---

## Next Up (Prioritized)

### ~~1. W-NOMINATE~~ — Done (Phase 17)

Completed 2026-02-28. See Completed Phases table above.

### 2. DIME/CFscores External Validation → Phase 14b

Completed 2026-02-28. Campaign-finance-based ideology from Bonica's DIME project (V4.0, ODC-BY). Validates 84th-89th bienniums (6 bienniums, one beyond Shor-McCarty). Reuses Phase 14 infrastructure (correlations, outliers, name normalization). See `docs/dime-cfscore-deep-dive.md`, ADR-0062.

### ~~3. Standalone Posterior Predictive Checks~~ — Done (Phase 4c)

Completed 2026-02-28. PPC battery (Yea rate, accuracy, GMP, APRE) + item/person fit + Yen's Q3 local dependence + LOO-CV model comparison across flat 1D, 2D IRT, and hierarchical IRT. Manual numpy log-likelihood (no PyMC rebuild). Graceful degradation for missing models. 60 tests. See ADR-0063, `analysis/design/ppc.md`.

### ~~4. Optimal Classification~~ — Done (Phase 17)

Completed 2026-02-28. Bundled with W-NOMINATE in Phase 17. See Completed Phases table above.

### ~~5. Latent Class Mixture Models~~ — Done (Phase 5b)

Completed 2026-02-28. Bernoulli mixture model (LCA) on binary vote matrix using StepMix. BIC model selection, Salsa effect detection, IRT cross-validation, Phase 5 ARI comparison, within-party LCA. Confirms null result: correct generative model for binary votes. Deep dive: `docs/latent-class-deep-dive.md`, design: `analysis/design/lca.md`.

### ~~6. Bipartite Bill-Legislator Network~~ → Done (Phase 6b)

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

## Completed (Formerly Deferred)

### ~~Joint Cross-Chamber IRT~~

**Completed (2026-02-24), fixed (2026-02-26).** The hierarchical joint cross-chamber model now runs with bill-matching (ADR-0043): shared `alpha`/`beta` parameters for 71 matched bills (91st) provide natural cross-chamber identification. Bill-matching reduced joint model runtime from 93 min to 31 min by shrinking the problem size (420 unified votes vs 491). Group-size-adaptive priors mitigate small-group convergence failures. Senate convergence now passes all checks. See ADR-0043 and `docs/joint-hierarchical-irt-diagnosis.md`.

---

## Next Up (Backlog)

### 1. External Validation Name Matcher: District Tiebreaker

Phase 14 (Shor-McCarty) uses last-name-only matching in `_phase2_last_name_match()` (`external_validation_data.py:294-331`). This produces an incorrect match for the 84th: `rep_bethell_lorene_1` (Lorene Bethell, District 113) is matched to SM's "Bob Bethell" — a different person. The code has a comment: "district tiebreaker not implemented — ambiguity is rare in KS data." Low impact (1/112 matches) but a genuine data quality bug. Fix: implement district-based disambiguation when multiple candidates share a last name and chamber.

### 2. Null `hier_shrinkage_pct` in Synthesis (84th)

In the 84th pipeline run, 28 House legislators (beyond the 2 expected IRT anchors) and 3 Senate legislators have null `hier_shrinkage_pct` in `legislator_df_{chamber}.parquet`. This is 26% of House and 14% of Senate. The calculation in `hierarchical.py:1123-1137` sets shrinkage to null when `abs(flat - party_mean) < SHRINKAGE_MIN_DISTANCE` (0.5) to avoid division-by-near-zero producing misleading percentages. Needs investigation: is 26% null rate normal for the 84th given its compressed ideological range, or is the threshold too aggressive? Compare null rates across all 8 bienniums.

### 3. 84th Biennium Pipeline Re-run

The current 84th results (`84-260228.5`) predate three code fixes: (1) Carey Unity `group_by` race condition (non-deterministic party member count swap), (2) clustering sensitivity ARI row-ordering mismatch (meaningless ARI values), (3) IRT sensitivity sign-flip handling (already fixed in code, results show stale negative correlations with `raw_pearson_r: null`). A fresh pipeline re-run (`just pipeline 2011-12`) would validate all fixes and produce clean results.

---

## Report Enhancement Backlog

Prioritized improvements to HTML report output, based on a comprehensive survey of the open-source landscape, academic standards, and general-audience best practices. Full analysis: [`docs/report-enhancement-survey.md`](report-enhancement-survey.md).

### Tier 1: High-Impact, Lower Effort

Presentation-layer enhancements — no pipeline changes required.

| # | Enhancement | Phases | Effort | Rationale |
|---|-------------|--------|--------|-----------|
| R1 | **Key Findings section** at top of every report | All 17 | Low | Every newsroom leads with findings. Academic papers have abstracts. Our reports jump straight to tables. Auto-generate 3-5 bullets from results data. |
| R2 | **ITables for legislator tables** (sort, search, filter, paginate) | All phases with full tables | Low | Every modern data site has searchable tables. Static HTML tables in 2026 feel dated. Zero-dependency library (v2.6+), Polars support. |
| R3 | **Party ideal point density overlay** | 04 IRT | Low | The single most published figure in IRT literature (Bafumi et al. 2005, Shor & McCarty 2011). Overlapping KDE curves show party separation and overlap at a glance. Missing from Phase 04; hierarchical has party posteriors but flat does not. |
| R4 | **Headline-style section titles** | All (Synthesis already partial) | Low | "Moderate Republicans Broke Ranks on 23 Key Votes" instead of "Cluster Analysis Results." Makes reports scannable and quotable. Follows FiveThirtyEight, Economist, Pew design. |
| R5 | **Absenteeism as analysis** (not just filtering) | 01 EDA, 07 Indices | Low | General audiences care deeply about who shows up. ProPublica and GovTrack use missed votes as a headline metric. Data already computed; present as finding, not filter. Rank legislators, correlate with vote closeness. |
| R6 | **ICCs in flat IRT report** | 04 IRT | Low | Code already exists in Phase 10 hierarchical. Item characteristic curves are standard in psychometric reporting. Show for top 5-10 most discriminating bills. |

### Tier 2: High-Impact, Medium Effort

New analysis code or visualization infrastructure changes.

| # | Enhancement | Phases | Effort | Rationale |
|---|-------------|--------|--------|-----------|
| R7 | **Per-vote spatial visualization** (cutting lines) | 04 IRT | Medium | VoteView's signature figure — legislators on ideology axis with cutting line showing Yea/Nay split. The most recognizable chart in legislative analysis. No equivalent in Tallgrass. Show for top-5 most discriminating votes per chamber. |
| R8 | **Swing vote identification** | 04 IRT or 07 Indices | Medium | "Whose vote decided the outcome?" — the most newsworthy question for close votes. Identify legislator(s) with ideal points nearest the cutting point on margin-≤5 votes. Aggregate to "swing vote frequency" ranking. |
| R9 | **Plotly for ideology scatter plots** | 02 PCA, 04 IRT, 05 Clustering | Medium | Interactive hover-to-identify on the most viewed figures. Plotly v6 has native Polars support, standalone HTML fragment export via `.to_html(full_html=False)`. Requires new section type in `report.py`. |
| R10 | **PyVis for network graphs** | 06 Network, 06b Bipartite | Medium | Interactive force-directed networks (drag, zoom, hover) from existing NetworkX objects. `from_nx(G)` one-liner. Standalone HTML output. Current static PNGs don't convey network structure well. |
| R11 | **Predicted vs. actual ("plus-minus") framing** | 07 Indices or 08 Prediction | Medium | FiveThirtyEight's most effective device: "Expected to vote with party 78%, actually 92%. Difference: +14." Dumbbell chart + ranked table. Uses existing prediction and party unity data. |
| R12 | **Vote-based bipartisanship index** | 07 Indices | Medium | Distinct from maverick score. Maverick = breaks from own party. Bipartisan = aligns with opposing party on party-line votes. Lugar Center BPI is the cosponsorship standard; vote-based analogue is straightforward. |
| R13 | **Named/described coalitions** | 05 Clustering, 11 Synthesis | Medium | Phases 05/05b/06 identify groups by number. Journalists need: "the moderate Republican faction (12 members, median IRT -0.3)." Auto-label from party composition, IRT range, and size. |

### Tier 3: High-Impact, Higher Effort

Substantial new features or infrastructure.

| # | Enhancement | Phases | Effort | Rationale |
|---|-------------|--------|--------|-----------|
| R14 | **Folium district choropleth maps** | New section in EDA or Profiles | High | Kansas legislative districts colored by ideology, party unity, or maverick score. Requires district GeoJSON (Census/redistricting). Interactive Leaflet maps with hover tooltips. |
| R15 | **Full voting record per legislator** | 12 Profiles | High | ProPublica, GovTrack, VoteView all show vote-by-vote detail. Profiles currently show defections and surprising votes but not the complete record. ITables with search/sort. |
| R16 | **Quarto unified dashboard** | All phases | High | Single navigable website combining all 17 phase reports with cross-linking. Supports Python, Plotly, Folium. Publishes to GitHub Pages. Replaces per-phase HTML files. |
| R17 | **Downloadable CSV alongside reports** | All phases | Medium | VoteView, ProPublica, GovTrack all offer data downloads. Add CSV export links for underlying data tables in each report. |
| R18 | **Freshmen cohort analysis** | 13 Cross-Session or new | Medium-High | Do newly elected legislators vote differently from incumbents? Convergence toward party mean over time? Comparative density plots, within-session drift. |
| R19 | **Voting bloc stability tracking** | 13 Cross-Session | Medium-High | Do clusters persist, split, or merge across bienniums? ARI of cluster assignments for returning legislators. Sankey diagram of bloc evolution. |
| R20 | **Scrollytelling in Synthesis** | 11 Synthesis | High | Progressive narrative reveal (D3.js + Scrollama.js). NYT/WaPo signature format. High engagement but requires JavaScript integration. |

### Tier 4: Nice-to-Have

| # | Enhancement | Notes |
|---|-------------|-------|
| R21 | **Parliament/hemicircle charts** for vote composition | Visually striking (Flourish/Plotly). Not analytically essential. |
| R22 | **Sankey diagrams** for bill flow (intro → committee → floor → passage) | Requires bill lifecycle data beyond roll calls. |
| R23 | **Ridgeline plots** for temporal ideology distributions | Alternative to existing density plots. More compact for multi-session views. |
| R24 | **Animated scatter** (Gapminder-style) for dynamic IRT | Engaging but complex. Plotly `animation_frame`. |
| R25 | **Descriptive alt text** for all figures | Accessibility improvement. Current `FigureSection` uses title only, not descriptive. |
| R26 | **Bill outcome prediction model** | Logistic regression on bill passage. Stanford CS229 achieves ~80% on congressional data. Requires bill-level features. |

### Key Library Additions

| Library | Purpose | Integration Point |
|---------|---------|-------------------|
| [Plotly](https://plotly.com/python/) (v6) | Interactive scatter, bar, heatmap, Sankey | New `InteractiveSection` in `report.py` |
| [PyVis](https://pyvis.readthedocs.io/) | Interactive network graphs | Phase 06, 06b |
| [ITables](https://github.com/mwouts/itables) (v2.6+) | Searchable/sortable tables | All phases with legislator tables |
| [Folium](https://python-visualization.github.io/folium/) | Interactive choropleth maps | New geographic sections |
| [Quarto](https://quarto.org/) | Unified multi-phase dashboard/website | Long-term publishing system |

---

## Other Backlog

### ~~Per-Phase Results Primers~~ — Done

All 18 phases define `*_PRIMER` strings (150-200 lines of Markdown each) that RunContext auto-writes to `README.md` in every phase output directory. Each primer covers: Purpose, Method, Inputs, Outputs, Interpretation Guide, and Caveats. The project-level primer (`docs/analysis-primer.md`) provides the general-audience overview.

### ~~Test Suite Expansion~~ — Done

1701 tests across scraper and analysis modules. All passing. Three gaps closed:
- **Integration tests**: `test_integration_pipeline.py` — synthetic data → EDA → PCA pipeline chain, RunContext lifecycle, upstream resolution (26 tests)
- **HTML report structural tests**: `test_report_structure.py` — TOC anchors, section ordering, numbering, container types, empty report, CSS embedding, make_gt integration (22 tests)
- **Pytest markers**: `@pytest.mark.scraper` (264 tests), `@pytest.mark.integration` (29 tests), `@pytest.mark.slow` (24 tests). Registered in `pyproject.toml`. Recipes: `just test-scraper`, `just test-fast`

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
| TBIP text-based ideal points | No full bill text available from scraper — revisit if bill text phase lands (see `docs/future-bill-text-analysis.md`) |

See `docs/method-evaluation.md` for detailed rationale on each rejection.

---

## Scraper Maintenance

| Item | When | Details |
|------|------|---------|
| Update `CURRENT_BIENNIUM_START` | 2027 | Change from 2025 to 2027 in `session.py` |
| Add special sessions | As needed | Add year to `SPECIAL_SESSION_YEARS` in `session.py` |
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
| 12 | W-NOMINATE | DIM | Completed (W-NOMINATE + OC, Phase 17) |
| 13 | Optimal Classification | DIM | Completed (W-NOMINATE + OC, Phase 17) |
| 14 | Beta-Binomial Party Loyalty | BAY | Completed (Beta-Binomial, Phase 7b) |
| 15 | Bayesian IRT (1D) | BAY | Completed (IRT) |
| 16 | Hierarchical Bayesian Model | BAY | Completed (Hierarchical IRT, Phase 8) |
| 17 | Posterior Predictive Checks | BAY | **Done** — Phase 4 basic + Phase 4c standalone |
| 18 | Hierarchical Clustering | CLU | Completed (Clustering) |
| 19 | K-Means / GMM Clustering | CLU | Completed (Clustering) |
| 20 | Co-Voting Network | NET | Completed (Network) |
| 21 | Bipartite Bill-Legislator Network | NET | Completed (Phase 6b) |
| 22 | Centrality Measures | NET | Completed (Network) |
| 23 | Community Detection | NET | Completed (Network) |
| 24 | Vote Prediction | PRD | Completed (Prediction) |
| 25 | SHAP Analysis | PRD | Completed (Prediction) |
| 26 | Ideological Drift | TSA | Completed (TSA, Phase 15) |
| 27 | Changepoint Detection | TSA | Completed (TSA, Phase 15) |
| 28 | Latent Class Mixture Models | CLU | Completed (LCA, Phase 5b) |
| 29 | Dynamic Ideal Points (Martin-Quinn) | TSA | Completed (Dynamic IRT, Phase 16) |
| 30 | DIME/CFscores External Validation | VAL | Completed — Phase 14b (ADR-0062) |
| 31 | Standalone Posterior Predictive Checks | BAY | **Done** — Phase 4c (ADR-0063) |
| 32 | TSA Hardening (Desposato, CROPS, validation) | TSA | Completed — item #7 above |

**Score: 32 completed, 7 rejected = 39 total**

Note: Methods 29-32 are additions beyond the original 28 (Dynamic Ideal Points, DIME/CFscores, Standalone PPC, TSA Hardening).

---

## Key Architectural Decisions Still Standing

- **Polars over pandas** everywhere
- **Python-first, R where necessary** — R allowed via subprocess for field-standard methods with no Python equivalent (W-NOMINATE, Optimal Classification in Phase 17; emIRT in Phase 16)
- **Ruff + ty + uv** — all-Astral toolchain (lint, type check, package management)
- **IRT ideal points are the primary feature** — prediction confirmed this; everything else is marginal
- **Chambers analyzed separately** unless explicitly doing cross-chamber comparison
- **Tables never truncated** in analysis reports — full data for initial analysis
- **Nontechnical audience** — all visualizations self-explanatory, plain-English labels, annotated findings
