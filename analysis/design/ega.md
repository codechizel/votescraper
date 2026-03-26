# EGA Design Document

## Assumptions

- Roll-call votes are binary (Yea/Nay); absences are missing data.
- Tetrachoric correlations are the correct association measure for binary items (not Pearson).
- The number of latent dimensions (K) is unknown and should be estimated from data, not assumed.
- EGA's dimensionality estimate is **advisory** — canonical routing (Phase 06) makes the final 1D/2D decision.

## Parameters & Constants

All constants are centralized in `analysis/tuning.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `EGA_GLASSO_GAMMA` | 0.50 | EBIC hyperparameter for GLASSO sparsity |
| `EGA_BOOT_N` | 500 | Number of bootstrap replicates |
| `EGA_STABILITY_THRESHOLD` | 0.70 | Minimum item stability from bootEGA |
| `UVA_WTO_THRESHOLD` | 0.25 | wTO threshold for redundancy detection |

## Methodological Choices

### Tetrachoric over Pearson
Pearson correlation on binary data underestimates true associations. Tetrachoric correlation estimates the Pearson r between two latent continuous variables that underlie the binary observations. Algorithm: maximize the bivariate normal log-likelihood over ρ ∈ (-1, 1) for each 2×2 table. Fallback to Phi coefficient (Pearson on binary) when any cell is zero.

### GLASSO over raw correlation
The Kappa agreement network in Phase 11 uses observed (marginal) associations — two legislators who both vote conservative have strong Kappa even if they disagree on social issues. GLASSO estimates **conditional** dependencies via L1-penalized precision matrix. Edges represent associations after controlling for all other items. This is what IRT's local independence assumption says should be zero if the model is correct.

### GLASSO via direct covariance (ADR-0126)
Uses sklearn's `graphical_lasso()` function which accepts an empirical covariance matrix directly. Previous implementation generated synthetic data with `max(n_obs, p+1)` rows to work around the `GraphicalLasso` class's raw-data requirement, which inflated effective sample size when p > n (e.g., 227 synthetic observations for 40 real legislators). The direct-covariance approach eliminates this distortion.

### EBIC model selection (gamma=0.5)
100 lambda values log-spaced from lambda_max to lambda_max/100. EBIC with gamma=0.5 (Golino default) balances model fit with sparsity. Higher gamma → sparser networks; lower gamma → denser networks.

### Walktrap default (over Leiden)
Walktrap (random walks, step=4) is Golino's default and handles unequal community sizes better. Leiden is available via `--algorithm leiden` for comparison. Both are already used in Phase 11.

### Unidimensional check
When community detection finds K ≥ 2, Louvain on the zero-order (non-regularized) correlation matrix tests whether K=1 is more appropriate. If Louvain finds a single community, the GLASSO-based multidimensionality may be a regularization artifact.

### Fragmentation guard (ADR-0126)
When Walktrap/Leiden produces K > max(p/4, 10), the GLASSO network is too sparse for meaningful community detection (most nodes in singletons). The guard retries community detection on the largest connected component and assigns isolated nodes to a catch-all community. If even the largest component is fragmented, falls back to K=1. The `CommunityResult.fragmented` field records when this guard activated. Triggered by the 78th Senate (K=196 from 226 bills with only 40 legislators).

### Parametric bootstrap (default)
bootEGA defaults to parametric (generate from tetrachoric correlation matrix → threshold to binary). Non-parametric (resample rows) can produce degenerate columns in small chambers (Senate N~40).

### TEFI (Von Neumann entropy)
TEFI = sum of within-community VN entropies - full matrix VN entropy. Lower = better. Critical property: TEFI properly penalizes over-extraction, unlike RMSEA/CFI.

### UVA (weighted topological overlap)
wTO measures structural equivalence — items sharing the same neighbors with similar weights. Threshold 0.25 = moderate-to-large redundancy. For legislative data, catches procedural vote sequences and amendment cascades.

## Downstream Implications

- **Phase 05 (IRT)**: EGA's K estimate informs whether to run 2D IRT (Phase 06). If EGA finds K=1 for both chambers, 2D IRT can be skipped (saving ~20 min/chamber).
- **Phase 08 (PPC)**: EGA community assignments enable per-community Q3 analysis — Q3 violations within a community suggest the community is multidimensional.
- **Phase 11 (Network)**: Tetrachoric correlation matrix provides a principled alternative to Kappa for network construction.
- **Canonical routing**: EGA is advisory input, not a routing signal. Convergence diagnostics, manual PCA overrides, and W-NOMINATE cross-validation (diagnostic) remain authoritative.

## Kansas-Specific Notes

- **Senate (N~40)**: Small sample produces sparse GLASSO networks. The fragmentation guard (ADR-0126) handles pathological cases where Walktrap fragments the network into near-singleton communities. If zero edges, the chamber is reported as unidimensional.
- **Supermajority chambers**: High base rate (~82% Yea) means many bill pairs have degenerate 2×2 tables. Tetrachoric falls back to Pearson for these; monitor `n_fallback` in the summary JSON.
- **Contested-only filtering**: EGA uses the EDA-filtered vote matrix (already filtered to `CONTESTED_THRESHOLD`). Additional filtering is not recommended — EGA handles low-variance items via GLASSO regularization.
