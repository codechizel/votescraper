# Cross-Session Validation Design Choices

**Script:** `analysis/26_cross_session/cross_session.py`, `analysis/26_cross_session/cross_session_data.py`
**Report:** `analysis/cross_session_report.py`
**ADR:** `docs/adr/0019-cross-session-validation.md`

## Assumptions

1. **Both sessions have completed the full pipeline** (at minimum: EDA, PCA, IRT, clustering, network, prediction, indices, synthesis). Missing phases produce warnings but don't block the analysis.

2. **IRT ideal points are the canonical ideology measure.** All cross-session ideology comparisons use flat IRT `xi_mean`, not hierarchical or PCA scores. Flat IRT is available for both sessions and provides the most comparable baseline.

3. **Legislators are matched by name, not slug.** Slugs (`rep_schreiber`) encode a session-specific hash and are not stable across bienniums. Matching uses normalized `full_name` (lowercased, stripped of whitespace and leadership suffixes).

4. **Affine alignment is sufficient for two time points.** With 131 overlapping legislators (78% overlap), the linear mapping has abundant anchor data. Dynamic IRT (random walk) is reserved for when three or more sessions are available.

5. **Feature distributions are comparable after z-score standardization.** Cross-session prediction requires that features like `xi_mean`, `beta_mean`, etc. have similar distributional shapes across sessions, even if the absolute scales differ.

## Parameters & Constants

| Parameter | Value | Justification | Location |
|-----------|-------|---------------|----------|
| `MIN_OVERLAP` | 20 | Minimum returning legislators needed for meaningful comparison. Below this, affine alignment is unreliable. | `cross_session_data.py` |
| `SHIFT_THRESHOLD_SD` | 1.0 | A legislator who moved > 1 SD (in aligned scale) is flagged as a significant mover. Balances sensitivity vs false positives. | `cross_session_data.py` |
| `ALIGNMENT_TRIM_PCT` | 10 | Trim 10% most extreme residuals from affine fit. Prevents genuine movers from distorting the alignment. | `cross_session_data.py` |
| `CORRELATION_WARN` | 0.70 | If cross-session Pearson r < 0.70, warn that alignment may be unreliable. | `cross_session_data.py` |
| `FEATURE_IMPORTANCE_TOP_K` | 10 | Compare top 10 SHAP features across sessions. | `cross_session_data.py` |
| `SIGN_ARBITRARY_METRICS` | `{"PC1"}` | Metrics whose sign is conventional but can flip on edge-case data. `compute_metric_stability()` uses `abs()` on correlations for these so a sign flip is not misread as instability. ADR-0037. | `cross_session_data.py` |
| `XGBOOST_PARAMS` | n_estimators=200, max_depth=6, lr=0.1 | Fixed hyperparameters for cross-session prediction. Same for A→B and B→A. | `cross_session.py` |
| `UNITY_SKIP_THRESHOLD` | 0.95 | Maverick detection: skip if all party unity scores exceed this. | `synthesis_detect.py` |
| `BRIDGE_SD_TOLERANCE` | 1.0 | Bridge-builder: candidate must be within this many SDs of cross-party midpoint. | `synthesis_detect.py` |
| `PARADOX_RANK_GAP` | 0.5 | Paradox: minimum percentile rank gap between IRT and loyalty. | `synthesis_detect.py` |
| `PARADOX_MIN_PARTY_SIZE` | 5 | Paradox: minimum legislators in majority party for detection. | `synthesis_detect.py` |

### Stability Enrichment

`compute_metric_stability()` now includes five additional columns beyond Pearson/Spearman:

| Column | Source | Interpretation |
|--------|--------|----------------|
| `psi` | `compute_psi()` | Population Stability Index: < 0.10 stable, 0.10–0.25 investigate, > 0.25 significant drift |
| `psi_interpretation` | `interpret_psi()` | Human-readable PSI label |
| `icc` | `compute_icc()` | ICC(3,1) two-way mixed, single measures, consistency |
| `icc_interpretation` | `interpret_icc()` | Koo & Li 2016: < 0.50 poor, 0.50–0.75 moderate, 0.75–0.90 good, > 0.90 excellent |
| `stability_interpretation` | `interpret_stability()` | Spearman rho interpreted per Koo & Li 2016 thresholds |

### Fuzzy Matching

`match_legislators()` accepts an optional `fuzzy_threshold` parameter. When set, unmatched names go through a second pass with `fuzzy_match_legislators()` using `difflib.SequenceMatcher` (stdlib). This is a resilience feature — the default exact matching is preferred.

### Percentile-Based Detection

`detect_chamber_maverick()` accepts `percentile` (e.g., 0.10 for bottom 10%) and `detect_metric_paradox()` accepts `rank_gap_percentile`. Both default to `None` (original fixed-threshold behavior). These are additive — existing callers are unaffected.

## Methodological Choices

### 1. Legislator Matching

**Decision:** Match on `full_name` after normalization: lowercase, strip whitespace, remove known suffixes (leadership titles like " - House Minority Caucus Chair").

**Edge cases to handle:**
- **Chamber switches** (e.g., a House member who ran for Senate): same name, different slugs, different chamber. Include these — they're analytically interesting.
- **Party switches:** Same name, different party. Include and flag.
- **Name changes:** Rare in practice. No fuzzy matching — require exact normalized match. Manual override list as a constant if needed.

**Alternatives considered:**
- Match on district: rejected — redistricting between sessions makes district numbers unstable.
- Fuzzy matching (Levenshtein): rejected — introduces false positives; 131 exact matches is already strong.
- Match on slug: rejected — slugs are session-specific.

### 2. IRT Scale Alignment

**Decision:** Robust affine transformation using overlapping legislators as anchors.

**Algorithm:**
1. Extract `xi_mean` for all legislators in both sessions.
2. Inner-join on normalized `full_name` → ~131 matched pairs.
3. Fit `xi_b = A * xi_a + B` using ordinary least squares.
4. Compute residuals. Trim the top/bottom `ALIGNMENT_TRIM_PCT`% (genuine movers).
5. Re-fit on the trimmed set to get robust `A` and `B`.
6. Transform session A's ideal points onto session B's scale (session B is the reference).

**Why session B (2025-26) as reference:** The most recent session is the natural baseline — we're asking "where did legislators come from?" not "where did they go?"

**Validation:**
- Pearson/Spearman r between aligned points (expect r > 0.85).
- Residual distribution should be roughly normal with mean near 0.
- Party-level sanity check: Democrats should remain left of Republicans after alignment.

**Why not standardize to z-scores instead?** Z-scores destroy the scale's relationship to the data. The affine approach preserves relative distances while fixing location and scale. Z-scores would make session A and B look identical in spread, masking real distributional changes (e.g., if one session was more polarized).

### 3. Ideology Shift Quantification

**Decision:** Three complementary metrics per legislator:

1. **Point estimate shift:** `delta_xi = xi_b_aligned - xi_a_aligned` (simple, interpretable).
2. **Posterior overlap (deferred):** If MCMC traces are available, compute `wasserstein_distance(posterior_a_aligned, posterior_b_aligned)` for Bayesian shift quantification that accounts for uncertainty. Deferred — requires loading ArviZ InferenceData from both sessions and aligning full posterior chains. Revisit when 3+ sessions make dynamic IRT worthwhile.
3. **Rank shift:** Change in within-chamber percentile rank. More robust to scale issues than absolute shift.

**Significant mover threshold:** `|delta_xi| > SHIFT_THRESHOLD_SD * std(delta_xi)` among all returners. This adapts to the overall session-to-session variability.

### 4. Turnover Analysis

**Decision:** Classify all legislators into three cohorts:
- **Returning:** Present in both sessions (~131).
- **Departing:** In session A but not B (~41).
- **New:** In session B but not A (~43).

Compare ideology distributions (xi_mean) across cohorts:
- Are departing legislators more moderate or extreme than returners?
- Are new legislators pulling the chamber left or right?
- KS test for distributional differences between cohorts.

### 5. Cross-Session Prediction

**Decision:** Two prediction experiments:

**Experiment A — Vote prediction transfer:**
1. Load `vote_features_{chamber}.parquet` from both sessions.
2. Z-score normalize all numeric features within each session.
3. Train XGBoost on session A features → predict session B votes (returning legislators only).
4. Reverse: train on B → predict A.
5. Report cross-session AUC alongside within-session AUC.

**Experiment B — Feature importance stability:**
1. Train XGBoost on each session independently (already done in prediction phase).
2. Load SHAP values from both sessions.
3. Compare top-K feature rankings (Kendall's tau on importance ranks).
4. Stable rankings (tau > 0.7) indicate the model captures generalizable patterns.

**Why not bill passage prediction?** Bill passage has too few observations per chamber (~250-500 rollcalls) and the bills are completely different across sessions. Vote prediction has ~30K-60K observations and the features (legislator ideology, bill discrimination) are session-agnostic after standardization.

### 6. Detection Threshold Validation

**Decision:** Re-run synthesis detection on 2023-24 legislator DataFrames using the same thresholds calibrated on 2025-26.

**Metrics:**
- Does each detection function find a result? (If not, the threshold is too restrictive.)
- Are the detected roles plausible? (Manual check against Kansas political knowledge.)
- If both sessions produce reasonable results, the thresholds are generalizable. If not, propose adaptive thresholds (e.g., percentile-based instead of absolute).

## Report Structure

~15-20 section HTML report targeting a nontechnical audience. The "who moved?" scatter plot is the signature visualization.

### Sections (implemented)

Per-chamber sections repeat for each chamber (House, Senate):

1. **Overview** — What this analysis does, data summary, overlap statistics
2. **Matching Summary** — Returning legislators by chamber (table)
3. **Ideology Shift Scatter** (per chamber) — Previous xi vs current xi, colored by party, annotated movers
4. **Biggest Movers Figure** (per chamber) — Horizontal bar chart of |delta_xi|, colored by direction
5. **Biggest Movers Table** (per chamber) — Detailed table of significant movers with shift metrics
6. **Shift Distribution** (per chamber) — Histogram of delta_xi with significance threshold lines
7. **Turnover Impact** (per chamber) — KDE plots comparing departing, returning, new cohorts
8. **Metric Stability** (per chamber) — Table of Pearson/Spearman correlations for 8 metrics
9. **Detection Validation** — Table comparing flagged mavericks, bridges, paradoxes across sessions
10. **Methodology Notes** — Alignment coefficients, thresholds, matching method, references

Prediction sections (behind `--skip-prediction` flag):
11. **Prediction Transfer Summary** (per chamber) — Table: cross-session AUC, accuracy, F1 vs within-session
12. **Prediction AUC Comparison** (per chamber) — Grouped bar chart: within-session vs cross-session
13. **Feature Importance Comparison** (per chamber) — Side-by-side SHAP bar charts + Kendall's tau

## Downstream Implications

### For Synthesis
- Cross-session metrics (ideology shift, stability correlations) can be joined into the synthesis legislator DataFrame for legislators who appear in both sessions.
- A "Cross-Session" section can be added to synthesis reports when multiple sessions are available.

### For Profiles
- Per-legislator profiles can include a "Historical Comparison" section showing ideology trajectory across available sessions.

### For Future Sessions
- The phase is designed to be session-pair-agnostic. When 2027-28 is scraped, the same code runs with `--session-a 2025-26 --session-b 2027-28`.
- With three or more sessions, the affine alignment generalizes to a chain: align each pair sequentially, or use a common reference session.
