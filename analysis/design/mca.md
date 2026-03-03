# MCA Design Choices

## Phase Placement

Phase 03, between PCA (02) and UMAP (03). Directory: `analysis/03_mca/`. Follows the UMAP "Phase 2b" naming precedent for additional dimensionality reduction views.

## Key Design Decisions

### Categorical encoding (not binary)

MCA on binary data is mathematically equivalent to PCA (Lebart et al. 1977). MCA only adds value by using the full categorical encoding: Yea / Nay / Absent as three distinct categories. This is the entire justification for the phase — it preserves the absence dimension.

### Absence category mapping

All absence-type categories (Absent and Not Voting, Not Voting, Present and Passing) are mapped to a canonical "Absent" label. Rationale: the Kansas Legislature has 5 vote categories but the distinctions between absence types are parliamentary procedure details, not ideological signals. Collapsing to 3 categories (Yea/Nay/Absent) gives MCA enough categorical structure to exceed PCA while avoiding sparse rare categories.

### prince library (not custom)

`prince` 0.16.5 is used rather than a custom numpy implementation. Rationale:
- FactoMineR-validated via rpy2 test suite
- Both Benzécri and Greenacre corrections implemented and tested
- Supplementary variable support
- The pandas dependency is acceptable at our matrix size (~170×500)

The Polars → pandas conversion happens at a single boundary (`polars_to_pandas_categorical`). All upstream and downstream work uses Polars.

### Greenacre correction (default)

Greenacre's inertia correction is preferred over Benzécri's because it's more conservative — it divides by average off-diagonal inertia rather than the sum of corrected eigenvalues. Benzécri tends to overestimate quality. Both are available via `--correction` CLI flag.

### Dim1 orientation convention

Republicans positive, matching PCA PC1. This makes cross-method comparison intuitive: positive = conservative on both MCA and PCA.

### Same filtering as PCA/EDA

Minority threshold = 2.5%, min votes = 20. Identical to PCA for fair comparison. Even though MCA's chi-square weighting partially handles near-unanimous votes, filtering still reduces the indicator matrix size and improves corrected inertia estimates.

### PCA validation

Spearman correlation between MCA Dim1 and PCA PC1 is computed and plotted. Expected r > 0.90 for binary-dominated data. Lower values indicate MCA found genuinely different structure (likely from the absence dimension).

### Horseshoe detection

Quadratic polynomial regression of Dim2 on Dim1. R² > 0.80 flags the horseshoe/arch effect, which is a known mathematical artifact of correspondence analysis on gradient-structured data, not a genuine second dimension.

## Data Flow

```
Raw votes CSV → build_categorical_vote_matrix() → Polars DataFrame
    → polars_to_pandas_categorical() → pandas DataFrame
    → prince.MCA.fit() → eigenvalues, row/column coordinates
    → orient_dim1() → extract_* functions → Polars DataFrames
    → save parquet, plots, report
```

## Outputs Consumed Downstream

MCA outputs are currently terminal (not consumed by later phases). Future integration points:
- Synthesis could incorporate MCA absence dimension findings
- Cross-session could compare MCA dimensions across bienniums
- Profiles could show per-legislator MCA coordinates

## References

- Greenacre (2017), *Correspondence Analysis in Practice*, 3rd ed.
- Le Roux & Rouanet (2010), *Multiple Correspondence Analysis*, SAGE
- Benzécri (1979), inertia correction formula
- Lebart et al. (1977), MCA-PCA equivalence on binary data
- See `docs/mca-deep-dive.md` for the full literature survey
