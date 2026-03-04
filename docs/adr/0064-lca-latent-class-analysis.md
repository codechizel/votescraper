# ADR-0064: Latent Class Analysis (Phase 10)

**Date:** 2026-02-28
**Status:** Accepted

## Context

Phase 5 (Clustering) found that k=2 is optimal across all four methods tested (hierarchical Ward, k-means, HDBSCAN, Gaussian mixture). This means the dominant structure in Kansas roll call votes is the party split, with within-party variation appearing continuous rather than factional. However, all four methods operate on *derived* representations (IRT ideal points, Kappa distances) rather than the original binary vote matrix. A Bernoulli mixture model (Latent Class Analysis) operates directly on the binary data, providing a statistically principled test: if BIC selects K=2, we have model-based evidence from the correct generative model that discrete factions don't exist.

Additionally, LCA offers the "Salsa effect" diagnostic: when K>2 classes differ only in *intensity* (parallel P(Yea) profiles) rather than *kind* (qualitatively different voting patterns), the extra classes represent quantitative grading, not genuine factions.

## Decision

Create Phase 10 as a standalone LCA phase using StepMix 2.2.3 (Python, MIT, published in JSS 2025):

1. **Bernoulli mixture model** on the same binary vote matrix used by Phase 5 clustering. StepMix's `measurement="binary_nan"` handles missing votes via full-information maximum likelihood (FIML), avoiding listwise deletion bias.

2. **BIC model selection** (K=1..8). BIC is chosen over bootstrap likelihood ratio tests for computational tractability and because it penalizes complexity more conservatively than AIC. Entropy is reported as a descriptive metric (classification sharpness) but not used for selection.

3. **Salsa effect detection** via pairwise Spearman correlations between class P(Yea) profiles. If all pairwise correlations exceed 0.80, classes are flagged as quantitative grading rather than qualitative factions.

4. **IRT cross-validation**. LCA class assignments are compared against IRT ideal points: classes should be monotonic in IRT space if both methods recover the same latent dimension. Straddlers (max class probability < 0.80) are identified.

5. **Phase 5 ARI comparison**. Adjusted Rand Index between LCA labels and each Phase 5 clustering method (hierarchical, k-means, spectral, HDBSCAN, GMM) to quantify agreement.

6. **Within-party LCA** on Republican-only and Democrat-only subsets (K=1..4) to test for intra-party factional structure.

7. **StepMix/scikit-learn compatibility shim**. StepMix 2.2.1 uses sklearn's private `_validate_data` method, removed in scikit-learn 1.8.0. A monkey-patch shim translates the call, guarded by `if not hasattr(StepMix, "_validate_data")` so it auto-deactivates when StepMix ships a fix. **TODO:** Remove when StepMix releases a compatible version.

## Consequences

- Confirms (or challenges) the Phase 5 finding using the statistically correct generative model for binary data
- Salsa effect diagnostic provides vocabulary for distinguishing "more classes" from "more factions"
- IRT cross-validation strengthens the convergent validity story across methods
- StepMix shim is a temporary workaround with a self-deactivating guard — no long-term maintenance burden
- Phase runs in the pipeline after clustering (`just pipeline` includes `just lca`)
- ~37 new tests added to the suite
