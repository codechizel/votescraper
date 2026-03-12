# ADR-0115: PCA Multidimensional Interpretation

**Date:** 2026-03-12
**Status:** Accepted

## Context

Parallel analysis identifies 3-5 significant dimensions in Kansas legislative roll-call data, but the PCA report only interpreted PC1-2. The 79th Senate (30R/10D) has eigenvalue ratio 1.45 with 5 significant components — yet PC3-5 were invisible in the report. This gap told readers "you have 5 dimensions" then explained only two. All PC1-5 data was already computed and saved to parquet; the gap was purely in visualization and report interpretation.

The `pca-deep-dive.md` audit (Section 5.1) explicitly recommended extending multidimensional interpretation. The horseshoe effect (ADR-0114) makes PC2 diagnostics especially important for supermajority chambers where PC2 captures curvature rather than a genuine second ideological dimension.

## Decision

Add 5 new report sections and 3 new analysis functions to `analysis/02_pca/pca.py` and `analysis/02_pca/pca_report.py`. No changes to PCA computation — purely visualization and interpretation.

### New analysis functions (`pca.py`)

1. **`plot_score_scatter_matrix()`** — Pairwise scatter matrix of significant PCs (capped at 5) using `seaborn.pairplot()`. Diagonal = per-party KDE, off-diagonal = party-colored scatter with top-3 outlier labels. Guards on `n_significant < 2`.

2. **`plot_loading_heatmap()`** — Heatmap of top-5 absolute-loading bills per significant PC. `seaborn.heatmap()` with RdBu_r, center=0, annotated values. Row labels: `bill_number + short_title[:30]` when available; bill_number only for KanFocus data.

3. **`diagnose_pc2_horseshoe()`** — Fits `PC2 ~ PC1 + PC1²` via `np.polyfit`, returns R² and boolean horseshoe detection (threshold: R² > 0.30). Consumed by report builder, no plot.

### Data passthrough

Raw vote matrix (`X_raw`, with NaN values intact) passed through the result dict from `run_pca_for_chamber()`. Needed for the absence diagnostic in per-component interpretation.

### New report sections (`pca_report.py`)

1. **Score Scatter Matrix figure** — FigureSection for `scatter_matrix_{chamber}.png`.
2. **Loading Heatmap figure** — FigureSection for `loading_heatmap_{chamber}.png`.
3. **PC2 Horseshoe Diagnostic** — Yellow warning banner (same style as `horseshoe_warning_html` in `phase_utils`) if detected; brief text noting R² below threshold if not.
4. **Component Party Profile table** — Rows = PC1 through n_significant, columns = R Mean, R Std, D Mean, D Std, |R−D|, Interpretation. |R−D| > 1.0 → "Partisan", < 0.5 → "Within-party", else "Mixed". Single-party chambers handled with NaN fill.
5. **Per-Component Interpretation** (PC2+ only) — Auto-generated narrative for each significant component beyond PC1: party mean comparison, top 3 positive/negative loading bills, and an absence diagnostic warning when >30% of top-10 loading bills have >30% null rate in the raw vote matrix.

### Modified existing sections

- **Loading tables**: Loop PC1 through n_significant (was hardcoded PC1, PC2).
- **Legislator scores table**: Dynamic PC columns based on n_significant (was hardcoded `["PC1", "PC2"]`).

### Report section order (per chamber)

1. PCA Summary table *(existing)*
2. Dimensionality Diagnostics table *(existing)*
3. Scree Plot figure *(existing)*
4. Ideological Map figure *(existing)*
5. PC1 Distribution figure *(existing)*
6. Score Scatter Matrix figure *(new)*
7. Loading Heatmap figure *(new)*
8. PC2 Horseshoe Diagnostic *(new)*
9. Component Party Profile table *(new)*
10. Per-Component Interpretation *(new, one per PC2+)*
11. Extended Loading Tables *(modified, PC1 through n_significant)*
12. Extended Legislator Scores *(modified, all significant PCs)*
13. Reconstruction Error table *(existing, stays last)*

## Consequences

**Positive:**
- Closes the interpretation gap: readers now see narrative for every significant dimension, not just PC1-2.
- Horseshoe diagnostic on PC2 provides early warning before IRT investment (complements ADR-0114).
- Absence diagnostic warns when a component is driven by attendance patterns rather than ideology.
- KanFocus data gracefully degrades: heatmap uses bill_number only, interpretation omits titles.
- No new library dependencies (seaborn and numpy already imported).

**Negative:**
- Score scatter matrix can be large (5×5 grid for 79th Senate). File size ~500KB per chamber.
- Auto-generated narratives are formulaic. A human analyst would write richer interpretation.
- Horseshoe threshold (R² > 0.30) is empirically chosen, not derived from theory.

**Edge cases handled:**
- `n_significant = 1`: All new sections no-op. Loading tables show PC1 only.
- No Democrats (pure supermajority): Party profile fills D columns with NaN, interpretation says "single-party".
- PC3-5 not in DataFrame: All functions guard `if pc_col not in df.columns: return`.
