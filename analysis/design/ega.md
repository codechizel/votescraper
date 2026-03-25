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

### EBIC model selection (gamma=0.5)
100 lambda values log-spaced from lambda_max to lambda_max/100. EBIC with gamma=0.5 (Golino default) balances model fit with sparsity. Higher gamma → sparser networks; lower gamma → denser networks.

### Walktrap default (over Leiden)
Walktrap (random walks, step=4) is Golino's default and handles unequal community sizes better. Leiden is available via `--algorithm leiden` for comparison. Both are already used in Phase 11.

### Unidimensional check
When community detection finds K ≥ 2, Louvain on the zero-order (non-regularized) correlation matrix tests whether K=1 is more appropriate. If Louvain finds a single community, the GLASSO-based multidimensionality may be a regularization artifact.

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
- **Canonical routing**: EGA is advisory input, not a routing signal. Convergence diagnostics and W-NOMINATE cross-validation remain authoritative.

## Kansas-Specific Notes

- **Senate (N~40)**: Small sample may produce very sparse GLASSO networks. Monitor edge count. If zero edges, the chamber may need TMFG (non-regularized alternative) in a future update.
- **Supermajority chambers**: High base rate (~82% Yea) means many bill pairs have degenerate 2×2 tables. Tetrachoric falls back to Pearson for these; monitor `n_fallback` in the summary JSON.
- **Contested-only filtering**: EGA uses the EDA-filtered vote matrix (already filtered to `CONTESTED_THRESHOLD`). Additional filtering is not recommended — EGA handles low-variance items via GLASSO regularization.
