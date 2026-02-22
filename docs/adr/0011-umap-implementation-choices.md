# ADR-0011: UMAP Implementation Choices

**Date:** 2026-02-22
**Status:** Accepted

## Context

UMAP (Uniform Manifold Approximation and Projection) is added as Phase 2b — a non-linear dimensionality reduction complement to PCA. While PCA finds the best linear projection, UMAP preserves local neighborhood structure and can reveal non-linear clusters and factions. Several implementation choices needed to be resolved:

1. **Distance metric.** UMAP supports many metrics (Euclidean, cosine, Hamming, etc.). The binary vote matrix has specific properties that favor some metrics over others.

2. **Comparison across hyperparameter settings.** UMAP embeddings have arbitrary rotation/reflection — standard correlation between UMAP1 at different settings is meaningless. Need a rotation-invariant comparison method.

3. **Validation against upstream methods.** UMAP preserves neighborhood structure (rank order), not distances. The validation metric must match this property.

4. **UMAP1 orientation.** Like PCA, UMAP axes have arbitrary sign. Need a convention.

5. **t-SNE inclusion.** The method doc (`Analytic_Methods/11_DIM_umap_tsne_visualization.md`) covers both UMAP and t-SNE. Decide whether to include both.

6. **File naming.** The Python `umap` package name conflicts with a module named `umap.py`.

7. **IRT column name.** IRT ideal points are stored under `xi_mean`, not a generic `ideal_point` column. The lookup must match the actual schema.

## Decision

1. **Cosine metric.** Cosine measures the angle between vote vectors, not magnitude. Two legislators who voted on different subsets of bills can still be compared meaningfully. Euclidean would penalize legislators with fewer votes. Cosine is explicitly recommended in the method doc for binary vote data.

2. **Procrustes analysis for sensitivity.** Procrustes optimally aligns two embeddings (rotate, reflect, scale) and measures residual disparity. The Procrustes similarity = 1 - disparity gives a rotation-invariant shape comparison. The sweep tests n_neighbors in [5, 15, 30, 50]. Similarity > 0.7 indicates stable structure.

3. **Spearman rank correlation for validation.** UMAP preserves rank ordering, not linear distances. Spearman measures rank-order agreement between UMAP1 and PCA PC1 or IRT ideal points. This is the correct test for "does UMAP produce the same ordering of legislators?"

4. **Orient UMAP1 so Republicans are positive.** Same convention as PCA PC1 — positive = conservative. Applied by checking party means and flipping if needed.

5. **UMAP only, no t-SNE.** UMAP is preferred because it preserves global structure (inter-cluster distances are meaningful), is faster, more deterministic, and produces continuous gradients (good for an ideological spectrum). t-SNE tends to create discrete blobs and loses inter-cluster distance information. The method doc explicitly recommends UMAP for legislative data.

6. **Named `umap_viz.py` to avoid shadowing.** A module named `umap.py` would shadow the `umap` package import. The `_viz` suffix is descriptive and avoids the conflict.

7. **Column lookup order: `xi_mean`, `ideal_point`, `theta`, `xi`.** The IRT phase stores ideal points as `xi_mean`. The lookup tries this first, then falls back to common alternatives for portability across IRT implementations.

## Consequences

**Benefits:**
- Cosine metric handles the binary vote matrix naturally, including legislators with varying participation rates.
- Procrustes sensitivity shows all n_neighbors pairs above 0.78 (House) and 0.91 (Senate), confirming structural robustness.
- Spearman validation against PCA (House rho=0.64) and IRT (House rho=0.57) shows moderate agreement — UMAP captures a related but distinct view of the data, as expected for a non-linear method.
- Self-contained phase: duplicated imputation logic from PCA means UMAP runs independently.
- 8 plots per run: landscape, PC1 gradient, IRT gradient, and sensitivity grid for each chamber.

**Trade-offs:**
- Spearman rho of 0.64 (House) and 0.29 (Senate) between UMAP1 and PCA PC1 is lower than the PCA-IRT correlation (r=0.97). This is expected — UMAP is non-linear and 2D, so UMAP1 alone doesn't capture the full embedding structure. The gradient validation plots (smooth color transitions) provide more convincing visual validation than the scalar correlation.
- Senate (n=42) is at the lower end of UMAP's effective range. The n_neighbors=50 setting in the sensitivity sweep is truncated to n=41 by umap-learn, and the small sample makes the embedding less stable than the House.
- Duplicated imputation means a bug fix must be applied in both `pca.py` and `umap_viz.py` independently.
- The `umap-learn` package adds `pynndescent` as a transitive dependency.
