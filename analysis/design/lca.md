# Latent Class Analysis (Phase 10) Design Choices

## Assumptions

1. **Local independence:** Within each latent class, votes are conditionally independent Bernoulli draws. This is the core LCA assumption. Violated if issue-specific coalitions create correlated voting patterns within a class, but BIC model selection partially accounts for this by preferring more classes if needed.

2. **Discrete latent structure:** LCA assumes the latent variable is categorical (K classes), not continuous. This is the test — if the data is better described by a continuum (as IRT assumes), LCA should select K=2 (the party split) and show high profile correlations between classes (Salsa effect).

3. **Bernoulli likelihood is correct for binary votes:** Unlike GMM (Phase 5) which assumes Gaussian data, LCA's Bernoulli mixture is the statistically correct generative model for Yea/Nay votes. This is why LCA exists as a separate phase.

4. **Same upstream filtering as EDA:** Uses the filtered vote matrix from Phase 1 (contested threshold < 2.5%, MIN_VOTES >= 20). Same data that feeds clustering, ensuring comparability.

## Parameters & Constants

| Parameter | Value | Justification |
|-----------|-------|---------------|
| K_MAX | 8 | Upper bound for class enumeration. Kansas has ~165 House + ~40 Senate; K=8 is generous. |
| N_INIT | 50 | Nylund-Gibson (2007) recommendation for EM stability. Multiple random starts prevent local optima. |
| MAX_ITER | 1000 | Convergence limit. Models that don't converge are flagged but still reported. |
| RANDOM_SEED | 42 | Project-wide reproducibility seed. |
| MIN_VOTES | 20 | Matches EDA/IRT threshold. |
| CONTESTED_THRESHOLD | 0.025 | Matches EDA threshold (defined in `analysis/tuning.py`; inherited from upstream vote matrix). |
| MIN_CLASS_FRACTION | 0.05 | Classes < 5% are flagged as potentially spurious. |
| SALSA_THRESHOLD | 0.80 | Spearman r above this = quantitative grading, not qualitative distinction. Based on convention for "strong" correlation. |
| TOP_DISCRIMINATING_BILLS | 30 | Bills shown in profile heatmap. Enough to see patterns without visual overload. |

## Methodological Choices

### Why BIC (not bootstrap LRT)

BIC is the primary model selection criterion (Nylund et al. 2007). Bootstrap LRT (BLRT) has better theoretical properties but is computationally expensive (requires fitting K models per bootstrap sample) and StepMix's BLRT implementation may not be stable for all K. BIC is sufficient for our purpose: confirming the null result (K=2) or detecting gross violations (K>>2).

AIC is reported but not used for selection — it's known to be too liberal for LCA (overfits K).

### Why Salsa effect detection

When K>2, we need to distinguish genuine multi-dimensional structure from quantitative grading. The "Salsa effect" (term from McLachlan & Peel 2000) occurs when classes have highly correlated P(Yea) profiles — they vote the same way on the same issues, just at different intensities. This maps exactly to the ideological continuum that IRT measures.

Spearman r > 0.80 between profile pairs = Salsa effect. This threshold is conservative — even r=0.7 would suggest substantial similarity.

### Why FIML (not listwise deletion)

StepMix's `measurement="binary_nan"` uses Full Information Maximum Likelihood for missing data. This is strictly superior to listwise deletion (which would discard legislators with any absences) and avoids imputation bias. Same principle as IRT's treatment of missing responses.

### Why not Bayesian LCA

Bayesian LCA (e.g., via PyMC) would provide posterior uncertainty on class assignments and parameters. However:
- Label switching makes MCMC inference on discrete latent variables fragile
- BIC-based frequentist LCA is the field standard in political science
- Our goal is confirmation of a null result, not parameter estimation
- StepMix is published in JSS (peer-reviewed implementation)

## Downstream Implications

- **Phase 11 (Synthesis):** LCA class assignments could be added to the narrative, but likely just confirms "two parties, continuous within-party variation."
- **No new features:** LCA is confirmatory — it validates Phase 5 findings with the statistically correct model. Does not produce new features for downstream phases.
- **Cross-validation with IRT:** If LCA classes are monotonically ordered in IRT space, both methods agree on the latent structure. If not, there's multi-dimensional structure that 1D IRT misses.
