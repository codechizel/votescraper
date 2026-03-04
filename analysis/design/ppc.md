# Phase 08: Posterior Predictive Checks + LOO-CV Model Comparison

## Assumptions

1. **Manual log-likelihood** — Bernoulli log-lik computed via numpy, not PyMC model rebuild. Avoids anchor reconstruction, party index rebuilding, and PyTensor compilation. The formula is trivial: `y * log(sigmoid(eta)) + (1-y) * log(1-sigmoid(eta))`.

2. **Standalone phase** — Not in `just pipeline`. Expensive, validation-only, does not feed downstream phases.

3. **Graceful degradation** — Runs on whatever models are available (1, 2, or 3). Missing models produce warnings, not errors. `az.compare()` requires 2+ models.

4. **Joint hierarchical excluded** — Known convergence issues, different legislator ordering. Phase 08 only compares flat 1D, 2D experimental, and per-chamber hierarchical.

## Parameters & Constants

| Parameter | Value | Source |
|-----------|-------|--------|
| `DEFAULT_N_REPS` | 500 | PPC replications per model |
| `DEFAULT_Q3_DRAWS` | 100 | Posterior draws for Q3 computation |
| `RANDOM_SEED` | 42 | Reproducibility (matches Phase 04) |
| Q3 violation threshold | 0.2 | Yen (1993) convention |
| Pareto k thresholds | 0.5 / 0.7 / 1.0 | Vehtari et al. (2017) |

## Methodological Choices

### PPC Statistics

- **Yea rate**: Basic calibration — does the model produce the right average?
- **Accuracy**: Classification — how often does the modal prediction match?
- **GMP** (Geometric Mean Probability): Penalizes confident wrong predictions more than accuracy. More robust for imbalanced data (82% Yea base rate).
- **APRE** (Aggregate Proportional Reduction in Error): Improvement over modal-category baseline. Controls for high Yea base rate — accuracy of 82% is no better than always guessing Yea.

### Q3 for Dimensionality

Yen's Q3 correlates item residuals after conditioning on ability. |Q3| > 0.2 indicates local dependence not explained by the latent trait. Key use: if 1D shows Q3 violations that 2D resolves, the second dimension is empirically justified.

### LOO-CV via PSIS

ArviZ `loo()` + `compare()`. PSIS-LOO estimates ELPD without refitting. Pareto k diagnostics identify observations where importance sampling is unreliable (k > 0.7). Stacking weights provide multi-model averaging.

## Known Limitations

### LOO observation mismatch (87th, 89th bienniums)

ArviZ `compare()` requires identical observation counts across all models. In the 87th and 89th bienniums, the hierarchical model uses a different vote matrix than flat IRT (different lopsided-vote filtering thresholds produce different observation sets). This causes `compare()` to raise a ValueError. PPC statistics (accuracy, GMP, APRE) still compute per-model; only the cross-model LOO comparison fails. These bienniums are excluded from LOO results. See ADR-0073.

## Downstream Implications

None. Phase 08 is terminal — results are human-interpretable validation, not pipeline inputs.
