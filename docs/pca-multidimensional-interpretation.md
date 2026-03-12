# PCA Multidimensional Interpretation

## The Problem

Parallel analysis regularly identifies 3-5 significant dimensions in Kansas legislative roll-call data. The 79th Senate (30R/10D) shows an eigenvalue ratio of just 1.45 with 5 significant components; even well-separated modern sessions like the 91st retain 3-4 significant dimensions. Yet the PCA report only helps readers interpret PC1 (partisan ideology) and, to a lesser extent, PC2 (via a scatter plot and loading table). PC3-PC5 are computed and saved to parquet but never visualized or interpreted in the HTML report.

This means the report tells readers "you have 5 significant dimensions" in the dimensionality diagnostics table, then proceeds to explain only one of them. For a nontechnical audience — journalists, policymakers, engaged citizens — this is a gap that undermines confidence in the analysis.

## What the Literature Says

### PC1: Partisan Ideology (High Confidence)

Universally, the first principal component of roll-call voting captures the left-right partisan divide. In Kansas, this separates Republicans (positive) from Democrats (negative). Explains 15-55% of variance depending on polarization level. No interpretation challenge here.

### PC2: Cross-Cutting Intraparty Cleavage (Moderate Confidence)

Poole and Rosenthal's NOMINATE work shows PC2 captures whatever the dominant *intraparty* cleavage is at a given time: slavery (antebellum), civil rights (1940s-60s), social/lifestyle issues (post-1980s). In Kansas, this manifests as the **moderate Republican vs. conservative Republican** divide — well-documented in state politics.

**Horseshoe caveat:** In supermajority chambers (30R/10D Senate), PC2 is likely dominated by the horseshoe artifact — a mathematical consequence of distance saturation where minority-party members and extreme majority-party members curve toward each other. The PC2 loading table may show this as "contrarianism on routine legislation" rather than a substantive policy dimension. The report must distinguish genuine second dimensions from horseshoe artifacts.

### PC3-PC5: Diminishing Returns (Low Confidence)

The literature is thin for state legislatures beyond two dimensions. What evidence exists suggests:

- **PC3** may capture issue-specific cleavages (e.g., education funding, Medicaid expansion) or absence/participation patterns (legislators who miss many votes create a distinct dimension)
- **PC4-PC5** are likely artifacts in chambers with 40 members. Sahoo, Bhatt & Bhatt (PNAS 2023) demonstrate that PCA produces "phantom oscillations" — U-shaped or oscillatory patterns that do not exist in the data — as a mathematical consequence of how eigenvectors are orthogonalized. Higher components in small chambers are especially susceptible.
- Parallel analysis is conservative but not infallible: a component can exceed the random-data threshold while still being an artifact of the horseshoe or absence patterns rather than a substantive ideological axis.

### Key Insight: Interpretation Requires Bill Content

A principal component is defined by the bills that load heavily on it. Without examining those bills' policy content, a component is just a number. The primary interpretation method in the literature is:

1. Extract the k bills with highest absolute loadings on each component
2. Examine their policy area, subject tags, or legislative committee
3. Look for a pattern — if the top PC3 bills are all education or all tax policy, the dimension has a substantive label

Since KanFocus-era sessions (pre-2011) lack bill titles, this interpretation is partially blocked for historical data — an important limitation to document.

## What's Currently Computed but Hidden

The PCA pipeline already saves rich data to parquet that the report ignores:

| Data | Saved? | In Report? |
|------|--------|-----------|
| PC1-5 scores for all legislators | Yes (`pc_scores_*.parquet`) | PC1-2 only (scores table, scatter plot) |
| PC1-5 loadings for all roll calls | Yes (`pc_loadings_*.parquet`) | PC1-2 only (top 30 bills each) |
| PC1-5 explained variance | Yes (`explained_variance.parquet`) | Scree plot + summary table (all 5) |
| Parallel analysis thresholds | Yes (`filtering_manifest.json`) | Diagnostics table (all 5) |
| Reconstruction error per legislator | Yes (`reconstruction_error_*.parquet`) | High-error subset only |

The gap is clear: the data exists to support full multi-dimensional interpretation, but the report builder only renders PC1-2.

## Proposed New Report Sections

### 1. Component Score Scatter Matrix (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3)

**What:** A grid of pairwise scatter plots for all significant components (up to the number identified by parallel analysis), party-colored, with outlier labels.

**Why:** The current PC1-vs-PC2 scatter is the single most useful PCA visualization. Extending it to PC3+ reveals whether higher dimensions capture real political structure or are artifacts. If PC1-vs-PC3 shows a parabolic curve, that's the horseshoe contaminating PC3. If it shows distinct clusters, that's a genuine policy cleavage.

**Audience value:** A journalist can look at the PC2-vs-PC3 scatter and immediately see whether there are meaningful subgroups within the Republican majority.

### 2. Loading Heatmap (Bills x Significant Components)

**What:** A heatmap with rows = top-k contributing bills per component (union of top-5 per PC), columns = PC1 through PCk, cells colored by signed loading magnitude. Diverging RdBu colormap.

**Why:** The current loading tables show PC1 and PC2 independently. A heatmap reveals *cross-loading* — bills that define multiple dimensions simultaneously. For example, a bill that loads +0.8 on PC1 and +0.5 on PC3 tells you PC3 is partially aligned with ideology; a bill that loads 0.0 on PC1 but +0.9 on PC3 tells you PC3 is orthogonal to partisan ideology.

**Audience value:** A reader can scan the heatmap and see "the dark-red bills on PC3 are all education-related" without reading loading tables.

### 3. Per-Component Interpretation Summary

**What:** For each significant component beyond PC1, a structured interpretation block:
- Top 5 positive and negative loading bills (with titles if available)
- Party composition analysis: which party's members score high vs. low
- Correlation with PC1 (checking for horseshoe contamination)
- Artifact diagnostic: check whether top-loading bills are disproportionately absence-dominated or near-unanimous

**Why:** This is the core missing section. The report identifies significant dimensions but never explains what they mean. The per-component summary provides the narrative scaffolding a nontechnical reader needs.

**Audience value:** Instead of "Parallel analysis retains 4 significant dimensions in House," the reader gets "PC2 captures the moderate-vs-conservative Republican divide (top bills: HB 2545 education funding, SB 256 tax reform). PC3 is dominated by absence patterns and should not be interpreted as a policy dimension."

### 4. Component Party Profile

**What:** For each significant component, a table or bar chart showing mean score by party (and optionally by intra-party faction if clustering results are available from Phase 09).

**Why:** If PC3 has identical Republican and Democrat means, it's not a partisan dimension — it separates legislators on some other axis. If the Republican mean is split bimodally, it reveals an intra-GOP faction. This is the simplest diagnostic for whether a component is politically meaningful.

### 5. Horseshoe Diagnostic on PC2

**What:** When the eigenvalue ratio < 3 (meaningful second dimension), compute the quadratic correlation between PC1 and PC2 (fit PC2 ~ PC1 + PC1²). If R² > 0.30, flag PC2 as likely horseshoe-contaminated and add a warning banner explaining the artifact.

**Why:** The horseshoe effect on PC2 is the single biggest interpretation trap in legislative PCA. Readers (and downstream models) need to know whether PC2 reflects real structure or geometric distortion.

### 6. Loadings for PC3-5 (Tables)

**What:** Extend the existing `_add_top_loadings()` call to loop over all significant components, not just PC1-2.

**Why:** This is the simplest change — the function already accepts a `pc` parameter. Just call it for PC3, PC4, PC5 when parallel analysis says they're significant.

### 7. Absence-Loading Diagnostic

**What:** For each component, compute what fraction of top-10 loading bills are dominated by absence patterns (>30% null in the vote matrix). Report this fraction.

**Why:** Higher PCA components in legislative data frequently capture "who shows up" rather than "how they vote." If 8 of the top 10 PC4 bills have >30% absence rates, PC4 is a participation dimension, not a policy dimension. This is critical context that prevents misinterpretation.

## What NOT to Add

### Factor Rotation (Varimax/Promax)

The Poole-Rosenthal tradition does not rotate, and the downstream IRT models have their own identification constraints. Rotation would create a coordinate system inconsistent with IRT, complicating cross-phase interpretation. If rotation is ever needed, it should be a separate experimental analysis, not part of the standard pipeline.

### Sparse PCA

Interesting for interpretation but adds a second PCA variant to maintain. Better suited to an experimental analysis (`analysis/experimental/sparse_pca_experiment.py`) than the production pipeline. The loading heatmap achieves similar interpretive goals with simpler machinery.

### Automatic Dimension Labeling from Bill Text

Requires bill titles (unavailable for KanFocus sessions) and adds NLP complexity. Phase 22 (Issue IRT) already handles topic-based analysis where bill text exists. The PCA report should provide the raw materials (loadings, bill identifiers) and let human readers — or downstream phases — assign meaning.

## Implementation Considerations

### Data Flow

All proposed features use data already computed and saved by `run_pca_for_chamber()`. No changes to the PCA computation are needed. The work is entirely in:

1. `pca.py` — new plot functions (scatter matrix, loading heatmap)
2. `pca_report.py` — new report section builders

### Performance

- Scatter matrix: negligible (dozens of points per chamber)
- Loading heatmap: negligible (union of ~25 bills × 5 components)
- Horseshoe diagnostic: one quadratic fit per chamber, trivial
- Absence diagnostic: one null-fraction scan per component, trivial

### KanFocus Graceful Degradation

For sessions without bill titles (`short_title` all null/empty), the loading heatmap and per-component summaries should use bill numbers and vote types instead. The `drop_empty_optional_columns()` utility from `phase_utils.py` already handles this pattern.

## References

- Poole, K.T. & Rosenthal, H. (1997). *Congress: A Political-Economic History of Roll Call Voting.* Oxford University Press.
- Horn, J.L. (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika*, 30(2), 179-185.
- Sahoo, S., Bhatt, D., & Bhatt, P. (2023). Phantom oscillations in principal component analysis. *PNAS*, 120(48).
- Gerrish, S. & Blei, D.M. (2012). How they vote: Issue-adjusted models of legislative behavior. *NeurIPS*.
- Podani, J. & Miklós, I. (2002). Resemblance coefficients and the horseshoe effect in principal coordinates analysis. *Ecology*, 83(12).
