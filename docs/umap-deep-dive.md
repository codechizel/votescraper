# UMAP Deep Dive: Implementation Review and Literature Survey

**Date:** 2026-02-24
**Scope:** Research survey, code review, and recommendations for `analysis/04_umap/umap_viz.py`

## Executive Summary

Our UMAP implementation is well-aligned with community best practices. The core design choices — cosine metric, unsupervised mode, Procrustes sensitivity, Spearman validation — are all correct and match what leading open-source projects (scanpy, BERTopic) and the UMAP author's own legislative voting demo do. This review found **no correctness bugs** but identified several areas where the implementation could be strengthened: adding trustworthiness scores as a quantitative validation metric, improving the legend to include the Independent party when present, a minor refactoring opportunity in duplicated plot code, and a few test coverage gaps.

---

## 1. How UMAP Works

UMAP (Uniform Manifold Approximation and Projection) is grounded in Riemannian geometry and algebraic topology. The algorithm constructs a topological representation of high-dimensional data using fuzzy simplicial sets, then optimizes a low-dimensional analog to match.

### Algorithm Steps

1. **Local metric computation.** UMAP assumes data lives on a Riemannian manifold with locally varying distance. Each point gets its own metric calibrated so the k-th nearest neighbor is at unit distance. This adapts to local density — sparse regions get stretched, dense regions get compressed.

2. **Fuzzy simplicial set construction.** Using local metrics, UMAP builds a weighted k-nearest-neighbor graph. Edge weights represent fuzzy membership strengths (0 to 1), decaying exponentially with distance. A local connectivity constraint ensures each point connects to at least its nearest neighbor with full confidence.

3. **Graph union.** Each point's local fuzzy set may disagree with neighbors. UMAP resolves this probabilistically: `w_combined = a + b - a*b` (probability that at least one independent edge exists). The result is a single fuzzy topological representation.

4. **Low-dimensional optimization.** UMAP initializes a 2D embedding (spectral by default) and optimizes via stochastic gradient descent against a fuzzy set cross-entropy loss. Attractive forces pull neighbors together; repulsive forces (via negative sampling) push non-neighbors apart.

### Key Theoretical Insight (2025)

Damrich et al. (arXiv:2602.11662, Feb 2025) proved UMAP is mathematically equivalent to spectral clustering on the fuzzy k-nearest-neighbor graph. This unifies UMAP with contrastive learning and spectral methods, providing theoretical grounding for an algorithm previously justified primarily empirically.

### What UMAP Does and Does Not Preserve

| Preserved | Not preserved |
|-----------|---------------|
| Local neighborhoods (rank order of nearest neighbors) | Global distances (inter-cluster gaps are unreliable) |
| Cluster membership (points that belong together stay together) | Density (tight cluster ≠ homogeneous group without densMAP) |
| Topology (connected components, holes) | Axis meaning (UMAP1/UMAP2 are arbitrary coordinates) |

---

## 2. Python Implementations

### umap-learn (Canonical)

The original implementation by Leland McInnes. Fully scikit-learn compatible.

- **Version:** 0.5.11 (Jan 2026). Our dependency: `umap-learn>=0.5`.
- **Dependencies:** numba, numpy, pynndescent, scikit-learn, scipy, tqdm.
- **Recent changes:** 0.5.11 fixed deterministic sorting for fast_knn_indices (reproducibility edge case). 0.5.10 added Python 3.13 support and sklearn compatibility fixes.

### Alternatives (Not Relevant for Us)

| Implementation | Use case | Why not for us |
|---------------|----------|----------------|
| cuML UMAP (NVIDIA RAPIDS) | GPU-accelerated, 60x speedup | Apple Silicon, no NVIDIA GPU |
| Parametric UMAP | Neural network learned embedding, generalizes to unseen data | Keras dependency, overkill for ~170 legislators |
| torchdr | PyTorch-native GPU UMAP | Same GPU constraint |
| uwot (R) | R implementation | Python project, no rpy2 |

---

## 3. How Leading Projects Use UMAP

| Project | n_neighbors | n_components | min_dist | metric | Purpose |
|---------|------------|--------------|----------|--------|---------|
| scanpy (single-cell genomics) | 10-30 | 2 | 0.5 | euclidean | Continuous biological gradients |
| BERTopic (topic modeling) | 15 | 5-10 | 0.0 | cosine | Preprocessing for HDBSCAN clustering |
| AlignedUMAP politics demo | 20 | 2 | 0.1 | cosine | Congressional voting trajectories |
| **Our project** | **15** | **2** | **0.1** | **cosine** | **Legislative ideological landscape** |

Our parameters match the AlignedUMAP congressional voting demo almost exactly (the demo uses n_neighbors=20 for ~435 representatives; our 15 for ~125 House members is proportionally similar). The cosine metric choice, 2D components, and min_dist=0.1 are standard for legislative visualization.

**Key takeaway:** Different projects make very different min_dist choices depending on their downstream goal. Our 0.1 is the right balance between cluster visibility and internal structure for a visualization-only application.

---

## 4. Review of Our Implementation

### What We Get Right

**Parameter choices.** Cosine metric is correct for binary vote data (measures angle, not magnitude, so legislators with different participation rates are still comparable). n_neighbors=15 is well-calibrated for our dataset size. min_dist=0.1 gives readable clusters with visible internal structure.

**Validation methodology.** Spearman (not Pearson) for comparing UMAP1 vs PCA/IRT is correct — UMAP preserves rank order, not linear distances. Procrustes (not correlation) for sensitivity analysis is correct — UMAP embeddings have arbitrary rotation.

**Unsupervised mode.** We correctly use unsupervised UMAP and validate against party labels post-hoc. Using supervised UMAP with party labels would be circular (telling UMAP to separate parties, then concluding parties are separated).

**Spectral initialization.** We use the default spectral init, which the 2025 spectral clustering equivalence paper shows computes the optimal linear solution to the UMAP objective. This is better than random init for preserving global structure.

**Cross-party outlier detection.** Data-driven, no hardcoded names. Works across all sessions without modification.

**Sensitivity sweep.** Testing n_neighbors in [5, 15, 30, 50] with Procrustes comparison is exactly the approach recommended by the UMAP documentation.

**Self-contained imputation.** Duplicating row-mean imputation from PCA prevents silent coupling between phases. This is an intentional design choice documented in ADR-0011.

### Issues Found

#### Issue 1: Legend Hardcodes Republican/Democrat Only

`plot_umap_landscape()` (line 611-617) builds the legend with only Republican and Democrat patches:

```python
ax.legend(
    handles=[
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
    ],
    loc="best",
)
```

`PARTY_COLORS` includes `"Independent": "#999999"`, and `load_metadata()` fills empty/null parties to "Independent" (ADR-0021). But the legend never shows Independent. For the 89th biennium (which has an Independent legislator), this means a gray dot appears with no legend entry.

**Recommendation:** Build the legend dynamically from parties present in the data.

#### Issue 2: Cross-Party Outlier Detection Assumes Imputation Artifact

The cross-party annotation in `plot_umap_landscape()` (line 588) always labels outliers as "imputation artifact — {party}, low participation":

```python
ax.annotate(
    f"{name}\n(imputation artifact — {party},\nlow participation)",
    ...
)
```

This assumes every cross-party outlier is caused by imputation, but a legislator could genuinely vote against their party on most issues (a true maverick, not an imputation artifact). The annotation should be more neutral or should verify the imputation rate before labeling it as an artifact.

#### Issue 3: No Trustworthiness Score

The validation section computes Spearman correlations and Procrustes similarity, but does not compute sklearn's `trustworthiness()` metric — the standard quantitative measure of local neighborhood preservation. This is a missed opportunity for additional validation rigor. Adding it would be straightforward:

```python
from sklearn.manifold import trustworthiness
score = trustworthiness(X_high_dim, X_embedding, n_neighbors=15)
```

This directly answers: "what fraction of a point's nearest neighbors in the embedding were also nearest neighbors in the original vote matrix?" A score above 0.80 is good; above 0.95 is excellent.

#### Issue 4: Sensitivity Sweep Does Not Clamp n_neighbors for Senate

The sensitivity sweep tests n_neighbors=50, but the Senate has ~40 legislators. umap-learn silently clamps n_neighbors to n_samples-1 (39), so the n_neighbors=50 run is actually n_neighbors=39 without any log message explaining this. The sweep could either skip values exceeding the sample size or log the clamping.

#### Issue 5: Exception Handling Style

The `load_pca_scores()` and `load_irt_ideal_points()` functions use comma-separated exception syntax:

```python
except FileNotFoundError, OSError:
```

This works correctly in Python 3.14 (parsed as a tuple of exception types), but the conventional style is parenthesized: `except (FileNotFoundError, OSError):`. The comma form is unusual enough to cause double-takes during code review. Also, `FileNotFoundError` is a subclass of `OSError`, so catching both is redundant — `except OSError:` alone would suffice.

### What the Tests Cover

The test suite (`tests/test_umap_viz.py`, 21 tests) covers:

| Area | Tests | Coverage |
|------|-------|----------|
| `impute_vote_matrix` | Shape, no NaNs, values preserved, row-mean calculation, all-null case, slugs/vote_ids extraction | Good |
| `orient_umap1` | Flip when R negative, no flip when R positive, unknown party ignored | Good |
| `build_embedding_df` | Columns present, shape, metadata joined | Good |
| `compute_procrustes_similarity` | Identical, rotated, scaled, random embeddings | Good |
| `compute_validation_correlations` | No upstream, PCA correlation, too-few-shared | Good |
| `compute_umap` (requires umap-learn) | Output shape, deterministic seed, cosine metric | Good |

### Test Gaps

| Missing test | Why it matters |
|-------------|----------------|
| `run_sensitivity_sweep()` | The sensitivity sweep is a core feature but has no direct test. A unit test with a small synthetic dataset could verify that it produces the expected number of Procrustes pairs and handles edge cases (e.g., n_neighbors > n_samples). |
| `plot_umap_landscape()` with Independent legislators | The 89th biennium includes Independents. No test verifies the plot handles three parties correctly. |
| Cross-party outlier detection logic | The threshold `rep_mean * 0.5` / `dem_mean * 0.5` is untested. Edge cases: what if all legislators are one party? What if means are negative? |
| `save_filtering_manifest()` | No test that the manifest contains expected keys. |
| `load_metadata()` leadership suffix stripping | Not tested in the UMAP test file (tested elsewhere via `test_run_context.py`, but the integration path through `load_metadata` is uncovered). |

### Refactoring Opportunities

**1. Duplicated plot structure.** `plot_umap_colored_by_pc1()` and `plot_umap_colored_by_irt()` are nearly identical — they differ only in the column name, join source, and colorbar label. These could be unified into a single `plot_umap_colored_by()` function that takes the column name and label as parameters. This would eliminate ~50 lines of duplication.

**2. IRT column discovery is duplicated.** The `for candidate in ["xi_mean", "ideal_point", "theta", "xi"]` pattern appears in both `compute_validation_correlations()` (line 440) and `plot_umap_colored_by_irt()` (line 679). This could be extracted into a small helper.

**3. Slug column discovery is redundant.** Lines like `slug_col = "legislator_slug"; pca_slug_col = slug_col if slug_col in pca_scores.columns else "legislator_slug"` always assign the same value. This was likely a leftover from when the column name might have varied. It can be simplified.

### Dead Code

One dead branch found and removed in ADR-0037: `if not validation:` was unreachable because a `trustworthiness` key was always added immediately before the check. All remaining functions are called, all imports used, all constants referenced.

---

## 5. Comparison with the UMAP Congressional Voting Demo

The UMAP documentation includes a first-party demo analyzing U.S. House roll-call votes from 1990-2020 using AlignedUMAP. This is the closest external analog to our project.

| Aspect | Congressional demo | Our project |
|--------|-------------------|-------------|
| Data | ~435 reps x ~1000 votes | ~125 House, ~40 Senate x ~500 votes |
| Encoding | -1/0/1 (No/Absent/Aye) | 0/1/NaN (Nay/Yea/absent, imputed) |
| Metric | cosine | cosine |
| n_neighbors | 20 | 15 |
| Temporal | AlignedUMAP across decades | Independent per biennium |
| Dimensions | 3D | 2D |

The congressional demo uses AlignedUMAP to track legislator ideological trajectories across time. This is a natural extension for our cross-session analysis (currently done via IRT alignment in `analysis/cross_session.py`). AlignedUMAP could provide a visual complement to the quantitative IRT shift metrics.

---

## 6. The UMAP Criticism Debate (2024-2025)

A vigorous debate emerged in 2024-2025 about UMAP's role in scientific analysis:

**"Biologists, stop putting UMAP plots in your papers" (Simply Statistics, Dec 2024).** Argued UMAP plots are primarily decorative, can create artificial clusters from continuous distributions, and advocated for PCA instead. Key demonstration: UMAP separates a single multivariate normal into distinct clusters — structure that does not exist.

**"Stop Misusing t-SNE and UMAP for Visual Analytics" (arXiv:2506.08725, 2025).** Documented systematic misinterpretation of DR outputs by practitioners, particularly regarding inter-cluster distances and cluster shapes.

**"Is UMAP accurate? Addressing fair and unfair criticism" (SciLifeLab, Feb 2025).** Counter-argued that UMAP is useful for exploration but acknowledged it exaggerates class separation compared to ground truth.

### How Our Project Addresses These Criticisms

| Criticism | Our mitigation |
|-----------|---------------|
| UMAP creates artificial clusters | We validate against PCA and IRT — if UMAP shows structure not present in linear methods, we treat it skeptically |
| UMAP axes are uninterpretable | Documented in primer, design doc, and ADR. Reports explicitly state "only distances matter" |
| UMAP inter-cluster distances are unreliable | We never quantify the partisan gap using UMAP distances — IRT handles that |
| UMAP is stochastic | Fixed seed + Procrustes sensitivity sweep |
| Small datasets are unstable | Senate caveat documented; sensitivity sweep flags instability |
| UMAP as decoration | We use UMAP specifically as the nontechnical visualization layer; PCA and IRT are the quantitative methods |

Our project's framing — "UMAP is a map, not a measurement" — is exactly the right response to the criticism debate. UMAP serves a legitimate and specific purpose: providing an accessible ideological map for journalists and constituents, validated against rigorous quantitative methods.

---

## 7. Potential Enhancements

Listed by descending priority. None of these are bugs — the current implementation is correct. These are opportunities to strengthen it.

### Worth Considering

**Trustworthiness score.** Add `sklearn.manifold.trustworthiness()` as a quantitative validation metric. Cheap to compute, provides a concrete number for "how well did UMAP preserve local neighborhoods?" Would strengthen the filtering manifest and HTML report.

**Dynamic legend.** Build the legend from parties actually present in the data rather than hardcoding Republican/Democrat. Simple fix, improves correctness for the 89th biennium.

**Smarter cross-party annotation.** Check each outlier's imputation rate before labeling it an "imputation artifact." If a legislator has >90% real votes and still lands in the opposite party's territory, that is a genuine maverick, not an artifact.

**Clamp sensitivity sweep.** Skip or annotate n_neighbors values that exceed n_samples for the Senate.

### Nice to Have

**Unified gradient plot function.** Merge `plot_umap_colored_by_pc1()` and `plot_umap_colored_by_irt()` into a single parameterized function. Reduces duplication.

**Multi-seed stability.** Run UMAP with 3-5 different random_states, report pairwise Procrustes similarity range. This would directly address the "UMAP is stochastic" criticism.

**Additional tests.** Cover `run_sensitivity_sweep()`, cross-party outlier detection edge cases, and three-party scenarios.

### Future Exploration (Not Recommended Now)

**AlignedUMAP for cross-session analysis.** Could provide a visual complement to IRT shift metrics. However, our cross-session validation phase already handles this via quantitative IRT alignment. Adding AlignedUMAP would be a significant scope increase for primarily visual benefit.

**densMAP.** Density-preserving mode could reveal whether party factions have different cohesion levels. But this adds interpretive complexity ("is the moderate wing really more spread out, or is that a density artifact?") without clear analytical payoff given our PCA/IRT infrastructure.

---

## 8. Implementation Status

All recommendations from sections 7.1 ("Worth Considering") and 7.2 ("Nice to Have") were implemented in the same commit as this article. Summary of changes:

| Change | Files modified |
|--------|---------------|
| Exception syntax: `except OSError:` (was redundant `FileNotFoundError, OSError`) | `umap_viz.py` |
| Trustworthiness score via sklearn | `umap_viz.py`, `umap_report.py` |
| Dynamic legend (handles Independent party) | `umap_viz.py` |
| Smarter cross-party annotation (checks imputation rate) | `umap_viz.py` |
| Sensitivity sweep clamps n_neighbors to n_samples | `umap_viz.py` |
| Unified gradient plot function (`plot_umap_gradient`) | `umap_viz.py` |
| Extracted `find_irt_column()` helper | `umap_viz.py` |
| Simplified redundant slug column logic | `umap_viz.py` |
| Multi-seed stability analysis (5 seeds, Procrustes) | `umap_viz.py`, `umap_report.py` |
| `compute_imputation_pct()` for per-legislator rates | `umap_viz.py` |
| 19 new tests (40 total, up from 21) | `test_umap_viz.py` |

New constants: `STABILITY_SEEDS`, `HIGH_IMPUTATION_PCT`.

---

## 9. Conclusions

Our UMAP implementation is methodologically sound, well-documented, and follows community best practices. The design choices are defensible: cosine metric for binary vote data, unsupervised mode to avoid circular analysis, Procrustes for rotation-invariant sensitivity analysis, and Spearman for rank-order validation. The primer, ADR, and design doc provide clear interpretive guidance that addresses the 2024-2025 criticism debate.

With the implemented improvements, the UMAP phase now includes trustworthiness scores for quantitative neighborhood preservation validation, multi-seed stability analysis to address the "UMAP is stochastic" criticism, imputation-aware cross-party outlier detection, and proper three-party support for sessions with Independent legislators.

---

## References

- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
- Damrich, S., et al. (2025). UMAP Is Spectral Clustering. arXiv:2602.11662.
- Levitin, J. (2024). Biologists, stop putting UMAP plots in your papers. Simply Statistics.
- Nonato, L.G., & Aupetit, M. (2025). Stop Misusing t-SNE and UMAP for Visual Analytics. arXiv:2506.08725.
- Oskolkov, N. (2025). Is UMAP accurate? Addressing fair and unfair criticism. SciLifeLab/NBIS.
- umap-learn documentation: https://umap-learn.readthedocs.io/
- AlignedUMAP politics demo: https://umap-learn.readthedocs.io/en/latest/aligned_umap_politics_demo.html
- BERTopic parameter tuning: https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html
- scanpy UMAP: https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.umap.html
