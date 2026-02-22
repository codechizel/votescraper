# UMAP Design Choices

**Script:** `analysis/umap_viz.py`
**Constants defined at:** `analysis/umap_viz.py:131-137`
**ADR:** `docs/adr/0011-umap-implementation-choices.md`
**Method doc:** `Analytic_Methods/11_DIM_umap_tsne_visualization.md`

## Assumptions

1. **Local neighborhood preservation.** UMAP assumes that the high-dimensional vote matrix can be faithfully represented by preserving local neighborhoods in 2D. If a legislator's 15 nearest neighbors (by cosine distance on voting records) are the same legislators who appear nearby in the 2D plot, the embedding is faithful. This is a weaker assumption than PCA's linearity, making UMAP more flexible.

2. **Cosine metric is appropriate for binary vote data.** Binary vote vectors have a natural cosine interpretation: two legislators with identical voting records have cosine similarity 1.0, two with opposite records have cosine similarity -1.0. Euclidean distance is less appropriate because it conflates the direction of disagreement with the number of shared votes.

3. **2D is sufficient.** The Kansas Legislature's ideological structure is primarily one-dimensional (party), with a possible second dimension (faction/contrarian). 2D UMAP captures both. 3D would add complexity without clear interpretive benefit.

4. **Imputed data is acceptable input.** UMAP requires complete data. Row-mean imputation (same as PCA) fills absences with each legislator's base rate. This biases absent legislators toward moderation. For legislators with very high absence rates (e.g., Sen. Miller, 85% imputed), UMAP coordinates should be interpreted cautiously.

## Parameters & Constants

| Constant | Value | Justification | Code location |
|----------|-------|---------------|---------------|
| `DEFAULT_N_NEIGHBORS` | 15 | Per UMAP docs: good default for medium datasets (100-200 points). Balances local and global structure. | `umap_viz.py:131` |
| `DEFAULT_MIN_DIST` | 0.1 | Per UMAP docs: allows moderate cluster tightness without collapsing structure. | `umap_viz.py:132` |
| `DEFAULT_METRIC` | `"cosine"` | Recommended for binary vote data (see Assumption 2). | `umap_viz.py:133` |
| `RANDOM_STATE` | 42 | Fixed seed for reproducibility. UMAP is stochastic. | `umap_viz.py:134` |
| `SENSITIVITY_N_NEIGHBORS` | [5, 15, 30, 50] | Standard range from method doc. 5 = very local, 50 = global. | `umap_viz.py:135` |

## Methodological Choices

### Cosine over Euclidean

**Decision:** Use cosine distance as the UMAP metric for the binary vote matrix.

**Alternatives:** Euclidean, Hamming, Jaccard.

**Why cosine:** Cosine measures the angle between vote vectors, not their magnitude. Two legislators who voted on different subsets of bills (due to committee assignments or absences) can still be compared meaningfully. Euclidean distance would penalize legislators who simply had fewer votes. The method doc (`Analytic_Methods/11_DIM_umap_tsne_visualization.md`) explicitly recommends cosine for Kansas data.

### Procrustes for sensitivity (not Pearson)

**Decision:** Compare UMAP embeddings across n_neighbors settings using Procrustes analysis (rotation-invariant shape comparison), not Pearson correlation.

**Why:** UMAP embeddings have arbitrary rotation and reflection. The UMAP1 axis at n_neighbors=5 may be rotated 90 degrees relative to UMAP1 at n_neighbors=30. Pearson between UMAP1 across settings would be meaningless. Procrustes optimally aligns the two embeddings (rotate, reflect, scale) and then measures residual disparity, giving a true measure of structural similarity.

**Metric:** Procrustes similarity = 1 - disparity. Values > 0.7 indicate stable structure.

### Spearman for validation (not Pearson)

**Decision:** Validate UMAP1 against PCA PC1 and IRT ideal points using Spearman rank correlation, not Pearson.

**Why:** UMAP preserves neighborhood structure, not distances. The mapping from high-D to 2D is nonlinear, so the relationship between UMAP1 and PC1 is monotonic but not linear. Spearman measures rank-order agreement, which is the correct test for "does UMAP1 produce the same ordering of legislators as PCA?"

### UMAP1 orientation convention

**Decision:** Flip UMAP1 sign so Republicans have positive mean scores, matching the PCA PC1 convention (positive = conservative).

**Why:** UMAP axes have arbitrary sign. Without this convention, UMAP1 might place Democrats on the positive side in some runs, confusing comparisons with PCA and IRT. The flip is applied before saving embeddings.

### No t-SNE

**Decision:** Use UMAP only, do not include t-SNE.

**Alternatives:** Run both UMAP and t-SNE, or t-SNE only.

**Why UMAP preferred:**
- UMAP preserves global structure (inter-cluster distances are meaningful); t-SNE does not
- UMAP is faster and more deterministic
- UMAP produces continuous gradients (good for ideological spectrum); t-SNE tends to create discrete blobs
- The method doc explicitly recommends UMAP for legislative data

### Duplicated imputation

**Decision:** The UMAP script duplicates the row-mean imputation logic from PCA rather than importing it.

**Why:** Self-containment, following the PCA precedent. Changes to EDA or PCA imputation won't silently alter UMAP results. The duplication is intentional and documented.

### Data-driven annotation

**Decision:** Label the top 5 legislators by |UMAP1| and |UMAP2| extremes, not hardcoded names.

**Why:** No hardcoded legislator names. The same script works for any session without modification.

## Downstream Implications

### For Synthesis (Phase 8)
- UMAP embeddings provide an additional "ideological map" visualization for the narrative report.
- UMAP coordinates can be used to identify bridge legislators (between party clusters) or faction leaders (cluster centers).
- The Procrustes sensitivity metric provides a robustness statement for the synthesis narrative.

### For interpretation
- **UMAP axes are arbitrary.** Only relative positions and distances are meaningful. Do not interpret UMAP1 as a specific ideological dimension — it is merely oriented to correlate with PC1 for convenience.
- **Small chamber warning.** The Senate (~40 legislators) is at the lower end of UMAP's effective range. Sensitivity results for the Senate should be scrutinized more carefully than the House.
- **Imputation artifacts.** Legislators with many imputed votes may appear artificially moderate (pulled toward the center). Cross-reference with IRT, which handles missing data natively.
