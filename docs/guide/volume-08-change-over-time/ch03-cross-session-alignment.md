# Chapter 3: Cross-Session Alignment: Putting Sessions on the Same Scale

> *Each biennium's IRT model is estimated independently. That means +1.0 in the 88th Legislature doesn't mean the same thing as +1.0 in the 90th. How do you put them on the same scale — without re-running the model?*

---

## The Comparability Problem

Chapter 2 solved the cross-session scale problem by fitting a single dynamic model across all bienniums simultaneously. That's the gold standard — one model, one scale, one set of estimates.

But sometimes you don't want (or can't afford) to run a 10,000-parameter Bayesian model. Sometimes you've already computed static IRT for two adjacent sessions and you want to compare them quickly. Sometimes you want to check whether the synthesis report's metrics (party unity, maverick rate, network centrality) are stable across sessions — and those metrics don't come from IRT.

**Cross-session alignment** is the lightweight alternative. It takes two independently estimated sets of ideal points and retroactively aligns their scales, using the legislators who appear in both sessions as calibration anchors.

The analogy: two thermometers that measure temperature in different units. One reads 20 (Celsius), the other reads 68 (Fahrenheit). They're measuring the same physical quantity — temperature — but on different scales. If you know that water freezes at 0°C and 32°F (two reference points), you can derive the conversion formula: F = 1.8 × C + 32. Cross-session alignment does the same thing with ideal points, using returning legislators as the reference points.

## The Affine Transformation

### The Formula

The alignment is an **affine transformation** — a fancy term for "stretch and shift":

```
xi_aligned = A × xi_original + B
```

- **A** (the slope): Stretches or compresses session A's scale to match session B's. If A = 1.5, session A's scores need to be 50% wider to match.
- **B** (the intercept): Shifts the center. If B = -0.3, session A's zero point is 0.3 units too high on session B's scale.
- **xi_original**: A legislator's ideal point from session A (the earlier session).
- **xi_aligned**: That same legislator's ideal point, transformed to session B's scale.

**A worked example:** Suppose the alignment produces A = 1.2 and B = -0.15. Representative Miller had xi = +0.8 in session A. On session B's scale, that becomes:

```
xi_aligned = 1.2 × 0.8 + (-0.15) = 0.96 - 0.15 = 0.81
```

Miller's position barely changed after alignment — session A's scale was slightly compressed and shifted, but not dramatically. Now you can directly compare Miller's 0.81 (on session B's scale) to her session B score. If her actual session B ideal point is +1.1, the difference (0.29 units rightward) represents genuine ideological movement, not a scale artifact.

### How A and B Are Estimated

The coefficients come from a simple linear regression on the returning legislators:

```
For each legislator who served in both sessions:
  x = their ideal point in session A
  y = their ideal point in session B

Fit: y = A × x + B via ordinary least squares
```

With 100+ returning legislators (typical for Kansas), this regression has abundant data. But there's a problem: some of those legislators genuinely changed their ideology between sessions. Including them in the regression would confuse scale differences with real movement.

### Robust Fitting: Handling Genuine Movers

The solution is **robust fitting** — a two-step process that isolates the scale calibration from the ideological drift:

**Step 1: Initial fit.** Run ordinary least squares on all returning legislators. This gives a preliminary A and B.

**Step 2: Trim the outliers.** Compute the residual (the difference between the predicted and observed session B score) for each legislator. The legislators with the largest residuals are the ones who moved the most — their session A score, even after alignment, doesn't match their session B score. Trim the top 10% by absolute residual.

**Step 3: Re-fit.** Run the regression again on the remaining 90%. This trimmed set contains legislators who *didn't* change much — the stable anchors who are best suited for scale calibration.

```
1. Fit initial regression: y = A_init × x + B_init (all returning legislators)
2. Compute residuals: r_i = y_i - (A_init × x_i + B_init)
3. Find the 90th percentile of |r_i|
4. Remove legislators above that threshold
5. Re-fit on trimmed set: y = A_final × x + B_final
6. Apply A_final, B_final to all session A scores
```

**Why 10%?** Conservative enough to keep most legislators (preserving statistical power) but aggressive enough to exclude the biggest movers. If trimming drops below the minimum overlap of 20 legislators, the phase falls back to the untrimmed fit — scale precision is more important than purity.

**Codebase:** `analysis/26_cross_session/cross_session_data.py` (`align_irt_scales()`, `ALIGNMENT_TRIM_PCT = 10`, `MIN_OVERLAP = 20`)

## Matching Legislators Across Sessions

### The Challenge

Before you can align scales, you need to know which legislators served in both sessions. This seems trivial — just match by name — but it has surprising edge cases:

- **Leadership suffixes:** "John Smith - President of the Senate" and "John Smith" are the same person.
- **Chamber switches:** A legislator who moved from the House to the Senate between sessions might appear with different metadata.
- **Party switches:** Rare but real. A legislator who switched parties between sessions needs to be matched to their earlier self.
- **Name variations:** "Bob" vs. "Robert," middle initials present or absent.

### The Matching Pipeline

Tallgrass uses a three-phase matching strategy:

**Phase 0: OCD ID join.** When available, the Open Civic Data ID (a national unique identifier for elected officials) provides the most reliable match. This handles name changes and disambiguates legislators with similar names.

**Phase 1: Normalized name join.** Normalize both sessions' names (lowercase, strip whitespace, remove leadership suffixes) and join on the normalized form. This catches the vast majority of returning legislators.

**Phase 2: Fuzzy matching (optional).** For unmatched remainders, a string similarity algorithm (SequenceMatcher) finds approximate matches above a threshold of 0.85 similarity. This catches minor spelling variations or formatting differences.

The result: a matched DataFrame with flags for chamber switches and party switches. In Kansas, typical overlap runs 70-80% of the smaller session's roster.

**Codebase:** `analysis/26_cross_session/cross_session_data.py` (`match_legislators()`, `fuzzy_match_legislators()`)

## Validation: Is the Alignment Good?

After alignment, Tallgrass checks the quality with two correlation measures:

- **Pearson r:** How well do the aligned session A scores predict session B scores? Values above 0.85 indicate a clean alignment where most of the remaining variance is genuine movement. Values below 0.70 trigger a warning — the alignment may be unreliable, possibly because the two sessions' IRT models captured fundamentally different dimensions.

- **Spearman rho:** How well does the *rank ordering* transfer? Even if the absolute scale is slightly off, do the most conservative legislators in session A remain the most conservative in session B? High Spearman rho with lower Pearson r suggests the rank structure is preserved but the scale mapping is nonlinear.

**Codebase:** `analysis/26_cross_session/cross_session_data.py` (`CORRELATION_WARN = 0.70`)

## Beyond Ideology: Metric Stability

Cross-session alignment isn't just about IRT scores. Tallgrass also measures whether eight behavioral metrics are stable across sessions:

| Metric | What It Measures |
|--------|-----------------|
| **Unity score** | How often a legislator votes with their party (Volume 6) |
| **Maverick rate** | How often they vote against their party on close votes |
| **Weighted maverick** | Maverick rate weighted by how close the vote was |
| **Betweenness centrality** | Their position as a bridge in the co-voting network (Volume 6) |
| **Eigenvector centrality** | Their influence in the network |
| **PageRank** | Their prestige in the directed voting network |
| **Loyalty rate** | Cluster-based voting consistency |
| **PC1** | First principal component score (Volume 3) |

For each metric, Tallgrass computes Pearson r, Spearman rho, ICC (Intraclass Correlation Coefficient), and PSI (Population Stability Index) between sessions.

**ICC** answers: "If I measure these legislators twice (in two sessions), how consistent are the measurements?" ICC above 0.75 is "good" — the metric ranks legislators similarly across sessions. ICC below 0.50 is "poor" — the metric is unstable or the underlying behavior changed.

**PSI** answers: "Has the *distribution* of this metric shifted?" PSI below 0.10 means stable; 0.10-0.25 means "investigate"; above 0.25 means significant drift. Unlike correlation (which measures individual-level consistency), PSI measures distributional shift (the overall shape changed, even if individuals kept their relative positions).

**Codebase:** `analysis/26_cross_session/cross_session_data.py` (`STABILITY_METRICS`)

## The Scatter Plot: Seeing Who Moved

The signature visualization is a **scatter plot** of aligned session A scores (x-axis) against session B scores (y-axis). Each point is a returning legislator, colored by party. The diagonal line represents perfect stability — legislators on the diagonal didn't change.

Points above the diagonal shifted rightward (more conservative in session B). Points below shifted leftward. The distance from the diagonal is the magnitude of the shift.

The top 5 movers are annotated with name labels. These are the legislators who changed the most — the ones whose shift exceeds 1.0 standard deviation of all shifts (the **significant mover threshold**).

A bar chart of the **top 15 movers** shows the direction and magnitude of each shift: red bars for rightward movement, blue for leftward. This quickly reveals whether the session-to-session shifts are predominantly in one direction (a systematic party shift) or scattered (individual-level noise).

**Codebase:** `analysis/26_cross_session/cross_session.py` (`TOP_MOVERS_N = 15`, `ANNOTATE_N = 5`), `analysis/26_cross_session/cross_session_data.py` (`SHIFT_THRESHOLD_SD = 1.0`)

## Prediction Transfer: Do Models Generalize?

A secondary analysis tests whether a vote prediction model trained in one session can predict votes in the next. If the legislature's dynamics are stable, a model trained on session A should perform nearly as well on session B as a model trained on session B itself.

The comparison:

| Scenario | Training Data | Test Data | Expected AUC |
|----------|--------------|-----------|-------------|
| Within-session | Session B (cross-validated) | Session B (held-out fold) | ~0.98 |
| Cross-session | Session A (all data) | Session B (all data) | ~0.90-0.95 |

A large drop from within-session to cross-session AUC means the two sessions have fundamentally different dynamics — different legislators, different issues, different coalition structures. A small drop means the political landscape is stable enough that last session's patterns predict this session's votes.

Tallgrass also compares **SHAP feature importance rankings** across sessions using Kendall's tau (a rank correlation). If the top 10 most important features are the same in both sessions (even if their exact magnitudes differ), the underlying model structure is generalizing. If the feature rankings shift dramatically, the dynamics changed.

**Codebase:** `analysis/26_cross_session/cross_session.py` (`XGBOOST_PARAMS`, `FEATURE_IMPORTANCE_TOP_K = 10`)

## What Can Go Wrong

### Insufficient Overlap

Cross-session alignment requires at least 20 returning legislators. This threshold is easily met in Kansas (130+ typically overlap), but could fail in states with term limits or after dramatic redistricting. If the overlap is too small, the affine regression has too few data points and the alignment becomes unreliable. The phase raises an error and stops rather than producing misleading results.

### Nonlinear Scale Differences

The affine transformation assumes the relationship between scales is linear: if a moderate in session A maps to a moderate in session B, then an extremist maps proportionally further out. If the true relationship is nonlinear (say, the extremes compressed), the affine fit will be slightly off at the tails. In practice, IRT scales are sufficiently well-behaved that this isn't a major concern, but the alignment works best for the middle 80% of the distribution.

### Confounding Real Change with Scale Artifacts

The alignment tries to separate scale differences from genuine movement, but the separation isn't perfect. A legislator who genuinely moved rightward will have a larger residual in the initial fit and might be trimmed as an "outlier." Their movement is then correctly preserved in the post-alignment shift calculation. But if *most* legislators moved in the same direction (a chamber-wide shift), the alignment might absorb part of that shift into the scale correction, underestimating the true change. This is an inherent limitation of any post-hoc alignment method — dynamic IRT (Chapter 2) handles it more naturally because it models the drift explicitly.

### Horseshoe-Affected Sessions

In the 7 Senate sessions where PC1 captured intra-Republican factionalism rather than the party divide (see Vol. 3), the IRT ideal points may not align cleanly with adjacent sessions that used the standard party axis. The alignment correlation will be lower, and the phase documents this with appropriate caveats.

---

## Key Takeaway

Cross-session alignment puts independently estimated IRT scores on a common scale using an affine transformation (stretch and shift) calibrated on returning legislators. Robust fitting trims the 10% biggest movers to prevent genuine ideological drift from contaminating the scale correction. The resulting alignment enables direct comparison of ideology across sessions, measurement of metric stability via ICC and PSI, and evaluation of whether predictive models generalize across legislative cycles.

---

*Terms introduced: cross-session alignment, affine transformation, scale correction, robust fitting, residual trimming, legislator matching (normalized name, OCD ID, fuzzy), minimum overlap, significant mover threshold, Intraclass Correlation Coefficient (ICC), Population Stability Index (PSI), prediction transfer, SHAP feature ranking stability, Kendall's tau*

*Previous: [Dynamic Ideal Points: The Martin-Quinn Model](ch02-dynamic-ideal-points.md)*

*Next: [Conversion vs. Replacement: Why Does Ideology Change?](ch04-conversion-vs-replacement.md)*
