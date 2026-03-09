# ADR-0054: 2D IRT Pipeline Integration

**Date:** 2026-02-28
**Status:** Accepted

## Context

The 2D Bayesian IRT experiment (`analysis/experimental/irt_2d_experiment.py`) successfully demonstrated that the Tyson paradox is a real multidimensional pattern, not a model artifact. Using a Positive Lower Triangular (PLT) identification strategy, the 2D M2PL model separates ideology (Dim 1, r=0.98 with PCA PC1) from contrarianism (Dim 2, r=0.81 with PCA PC2) on the 91st Senate.

However, the experiment only ran on a single chamber, used manual output directory management, and was not wired into the pipeline. We want to make this analysis reproducible and accessible as part of the standard pipeline while clearly marking it as experimental due to known Dim 2 convergence challenges.

## Decision

Integrate the 2D IRT model as pipeline phase `06_irt_2d` with these design choices:

1. **Both chambers**: Run House and Senate (not just Senate). Consistent with all other phases. House may reveal its own second-dimension patterns.

2. **RunContext integration**: Structured output, `--run-id` support, `--eda-dir`/`--pca-dir` overrides, HTML report, auto-primer. Standard pipeline infrastructure.

3. **nutpie sampling**: Uses `nutpie.compile_pymc_model()` + `nutpie.sample()` consistent with all other MCMC phases (ADR-0051, ADR-0053). The PLT beta construction with `pt.set_subtensor` compiles fine — nutpie operates on the PyTensor graph regardless of specific ops.

4. **Relaxed convergence thresholds**: R-hat < 1.05, ESS > 200, divergences < 50 (vs production 1.01/400/10). These are prominently documented in the experimental banner, primer, and report.

5. **No synthesis/profiles integration**: 2D scores are NOT fed into downstream phases. The convergence caveats make downstream consumption premature. Synthesis can check for 04b output in a future enhancement.

6. **Experimental banner**: The HTML report leads with a red-bordered TextSection explaining the relaxed thresholds and Dim 2 reliability caveats.

## Consequences

- The pipeline grows from 13 to 14 phases. `just pipeline` now includes `just irt-2d`.
- Pipeline runtime increases by ~20-40 minutes (two chambers of 2D MCMC).
- Researchers get reproducible 2D ideal points in the standard results structure.
- The experimental status is self-documenting — the report cannot be mistaken for production-grade analysis.
- Future work: nutpie migration (after `pt.set_subtensor` testing), synthesis integration (after convergence improvements), and cross-session 2D comparison.
- Interactive Plotly plots (2D scatter, Dim 1 vs PC1, Dim 2 vs PC2) embedded in HTML report via `InteractiveSection` (ADR-0069). Hover shows legislator name, party, coordinates, and HDIs.
- The 2D results serve as a horseshoe diagnostic: the `--promote-2d` robustness flag (ADR-0104) cross-references 1D rankings with 2D Dim 1 rankings. See `docs/horseshoe-effect-and-solutions.md`.
