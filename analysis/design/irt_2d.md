# 2D IRT Design Choices

**Script:** `analysis/06_irt_2d/irt_2d.py` (pipeline phase), `analysis/experimental/irt_2d_experiment.py` (original experiment)
**ADRs:** `docs/adr/0046-2d-irt-experimental.md`, `docs/adr/0054-2d-irt-pipeline-integration.md`
**Deep dive:** `docs/2d-irt-deep-dive.md`

## Status: Pipeline phase (experimental)

The 2D IRT model is an experimental extension of the canonical 1D baseline, integrated as pipeline phase 04b. It runs **both chambers** (House and Senate) with RunContext integration. It does NOT replace the 1D model. The 1D model remains primary for all downstream analyses. Relaxed convergence thresholds are used (R-hat < 1.05, ESS > 200, divergences < 50).

## Model Specification

### Multidimensional 2-Parameter Logistic (M2PL)

```
P(Yea | xi_i, alpha_j, beta_j) = logit^-1(sum_d(beta[j,d] * xi[i,d]) - alpha_j)

xi_i   ~ Normal(0, 1)     per dimension, per legislator   shape: (n_leg, 2)
alpha_j ~ Normal(0, 5)     per bill (difficulty)           shape: (n_votes,)
beta_j  ~ PLT-constrained  per bill, per dimension         shape: (n_votes, 2)
```

The dot product `beta_j · xi_i` replaces the 1D scalar product. Each bill discriminates on both dimensions simultaneously.

## Parameters & Constants

### MCMC Sampling

Sampler: nutpie (Rust NUTS with adaptive dual averaging), consistent with all production MCMC (ADR-0051, ADR-0053).

| Constant | Value | Justification |
|----------|-------|---------------|
| `N_SAMPLES` | 2000 | Same as 1D for comparability |
| `N_TUNE` | 2000 | Doubled from 1D (1000). Harder posterior geometry needs longer adaptation. |
| `N_CHAINS` | 4 | Following ADR-0045. PLT identification + PCA init should prevent mode-splitting. |
| `RANDOM_SEED` | 42 | Consistency with 1D model. |

### Convergence Thresholds (Relaxed for Experimental)

| Threshold | Value | 1D Value | Notes |
|-----------|-------|----------|-------|
| `RHAT_THRESHOLD` | 1.05 | 1.01 | Relaxed; 2D posterior is harder to mix |
| `ESS_THRESHOLD` | 200 | 400 | Relaxed; 2D model has more parameters |
| `MAX_DIVERGENCES` | 50 | 10 | Relaxed; expect some geometry issues |

### Priors

| Parameter | Prior | Shape | Notes |
|-----------|-------|-------|-------|
| xi (ideal points) | Normal(0, 1) | (n_leg, 2) | Standard normal per dimension |
| alpha (difficulty) | Normal(0, 5) | (n_votes,) | Identical to 1D |
| beta col 0 (Dim 1 disc.) | Normal(0, 1) | (n_votes,) | Unconstrained, same as 1D |
| beta[1,1] (Dim 2 anchor+) | HalfNormal(1) | scalar | Positive diagonal (PLT) |
| beta[2:,1] (Dim 2 rest) | Normal(0, 1) | (n_votes-2,) | Free to be positive or negative |
| beta[0,1] (Dim 2 anchor0) | Fixed at 0 | — | Zero constraint (PLT rotation fix) |

## Identification: PLT (Positive Lower Triangular)

### The Problem

For D=2, the likelihood has 2^D × D! = 8 equivalent posterior modes due to rotation, reflection, and label-switching invariance. Without constraints, MCMC will mix across modes, producing meaningless posterior summaries.

### The Solution

Constrain the discrimination matrix to be lower triangular with positive diagonal:

1. **Rotation fix:** Set `beta[0, 1] = 0` — the first item (anchor item) loads only on Dimension 1. This fixes the rotation angle.

2. **Dim 2 sign fix:** Set `beta[1, 1] > 0` via HalfNormal prior — the second item has a positive loading on Dimension 2. This fixes the Dim 2 sign convention.

3. **Dim 1 sign fix:** Post-hoc verification that Republican mean on Dim 1 is positive. Same approach as 1D IRT and hierarchical IRT.

### Anchor Item Selection

Item 0 (the rotation anchor, with `beta[0,1] = 0`) should be a bill that clearly loads only on the primary ideological dimension — a pure party-line vote. Selected as the bill with the highest absolute beta from the 1D IRT model (i.e., the most discriminating bill).

Item 1 (the positive diagonal anchor, with `beta[1,1] > 0`) should be a bill that plausibly loads on the second dimension. No strong constraint on selection — any non-anchor item works.

### Implementation in PyMC

```python
# Discrimination column 0: fully unconstrained (all bills load on Dim 1)
beta_col0 = pm.Normal("beta_col0", mu=0, sigma=1, shape=n_votes)

# Discrimination column 1: PLT-constrained
# Item 0: fixed at 0 (rotation anchor)
# Item 1: HalfNormal (positive diagonal)
# Items 2+: free Normal
beta_anchor_positive = pm.HalfNormal("beta_anchor_positive", sigma=1)
beta_col1_rest = pm.Normal("beta_col1_rest", mu=0, sigma=1, shape=n_votes - 2)

beta_col1 = pt.zeros(n_votes)
beta_col1 = pt.set_subtensor(beta_col1[1], beta_anchor_positive)
beta_col1 = pt.set_subtensor(beta_col1[2:], beta_col1_rest)

# Stack into (n_votes, 2) discrimination matrix
beta = pt.stack([beta_col0, beta_col1], axis=1)
```

## Initialization Strategy

### 2D PCA Initialization

```python
# xi[:, 0] initialized from PCA PC1 (standardized)
# xi[:, 1] initialized from PCA PC2 (standardized)
xi_init = np.column_stack([
    (pc1 - pc1.mean()) / pc1.std(),
    (pc2 - pc2.mean()) / pc2.std(),
])
```

- Passed via nutpie `initial_points={"xi": xi_init}` with `jitter_rvs` excluding xi
- PCA orientation provides mode identification for Dim 1
- PC2 orientation provides starting direction for Dim 2

### Why Not Random Initialization

The 1D IRT mode-splitting investigation (ADR-0023) showed that random initialization causes 5/16 convergence failures due to reflection invariance. With 8 modes instead of 2, the risk is much higher. PCA initialization places chains near the correct mode.

## Downstream Outputs

### Data Files

| File | Description |
|------|-------------|
| `ideal_points_2d_house.parquet` | House 2D ideal points with HDIs |
| `ideal_points_2d_senate.parquet` | Senate 2D ideal points with HDIs |
| `convergence_summary.json` | Per-chamber R-hat, ESS, divergences, correlations |

### Plots (per chamber)

| File | Description |
|------|-------------|
| `2d_scatter_{chamber}.png` | Dim 1 vs Dim 2, party-colored, Tyson/Thompson annotated |
| `dim1_vs_pc1_{chamber}.png` | 2D Dim 1 vs PCA PC1, with Pearson r |
| `dim2_vs_pc2_{chamber}.png` | 2D Dim 2 vs PCA PC2, with Pearson r |
| `2d_scatter_interactive_{chamber}.html` | Plotly interactive Dim 1 vs Dim 2 with hover details |
| `dim1_vs_pc1_interactive_{chamber}.html` | Plotly interactive Dim 1 vs PC1 with hover details |
| `dim2_vs_pc2_interactive_{chamber}.html` | Plotly interactive Dim 2 vs PC2 with hover details |

Interactive plots use Plotly (`fig.to_html(full_html=False, include_plotlyjs="cdn")`) and are embedded in the HTML report via `InteractiveSection`. Hover shows legislator name, party, coordinates, and HDIs. These are particularly useful for horseshoe diagnostics — hovering over the Dim 1 vs PC1 plot reveals which legislators are misplaced by PCA's dimension confounding (see `docs/horseshoe-effect-and-solutions.md`).

### Success Criteria

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| Convergence | R-hat < 1.05, ESS > 200 | Relaxed for experimental |
| Dim 1 ↔ 1D IRT | r > 0.90 | Primary axis should agree |
| Dim 2 ↔ PCA PC2 | Positive correlation | Second axis should match PCA finding |
| Tyson on Dim 2 | Most extreme | The whole point of the model |
| Thompson on Dim 2 | Second most extreme | Validates the PC2 pattern |

## Relationship to 1D Model

The 2D model does NOT replace the 1D model because:

1. **Most legislators are 1D.** For 95%+ of legislators, Dim 2 will be near zero with wide HDIs. The extra dimension adds noise, not signal.
2. **Prediction barely improves.** The 1D model already achieves 0.98 AUC. The second dimension captures behavior on low-discrimination bills.
3. **Interpretability decreases.** A single ideal point per legislator is easy to explain to policymakers. Two numbers per legislator is harder.
4. **Computational cost is 3-6x.** Not justified for routine pipeline runs.
5. **Identification is fragile.** PLT constraints work but add complexity. The 1D model's anchor identification is simpler and better-validated.

The 2D model is valuable as a **diagnostic tool**: it confirms that the Tyson paradox is a real multidimensional pattern (not a model artifact), estimates uncertainty on the second dimension, and identifies which bills drive the contrarianism pattern.

## Horseshoe Diagnostic Value

The 2D model also serves as the primary diagnostic for the horseshoe effect in supermajority chambers. When the 1D model conflates establishment-loyalty with ideology, the 2D Dim 1 vs PCA PC1 interactive plot makes the distortion directly visible — legislators whose PCA placement disagrees with their IRT placement are horseshoe victims. See `docs/horseshoe-effect-and-solutions.md` for the full explanation and experimental results.

The `--promote-2d` robustness flag (ADR-0104) cross-references 1D rankings with 2D Dim 1 rankings. A supermajority audit across 28 chamber-sessions (78th–91st) found 5 sessions with problematic 1D-2D disagreement, all in the Kansas Senate (79th, 80th, 81st, 83rd, 88th).
