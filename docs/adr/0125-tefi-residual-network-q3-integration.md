# ADR-0125: TEFI, Residual Network, and Q3 Per-Pair Integration (Phases 02, 08, 11)

**Date:** 2026-03-25
**Status:** Accepted

## Context

Three existing pipeline phases had specific gaps that EGA-adjacent methods could fill without requiring the full Phase 02b EGA infrastructure:

1. **Phase 02 (PCA):** Dimensionality assessed via scree plots (subjective) and parallel analysis. No information-theoretic metric.
2. **Phase 08 (PPC):** Q3 local dependence reported as aggregate statistics (violation count, mean |Q3|). No per-bill-pair visualization or top-violation identification.
3. **Phase 11 (Network):** Kappa agreement networks operate independently from IRT. No feedback loop showing which co-voting patterns IRT explains vs. doesn't.

## Decision

Add targeted enhancements to three existing phases:

### Phase 02 (PCA): TEFI Dimensionality Metric

- After PCA computation, compute TEFI (Von Neumann entropy) for K=1..5 using PCA loading-based bill assignments.
- Save TEFI scores to `data/tefi_pca_{chamber}.json`.
- Plot TEFI curve (K on x-axis, TEFI on y-axis, vertical line at best K).
- Add TEFI section to PCA HTML report between dimensionality diagnostics and scree plot.
- Gracefully skip if `analysis.ega.tefi` is unavailable.

### Phase 08 (PPC): Q3 Per-Pair Heatmap

- After Q3 computation, generate |Q3| heatmap plot per model per chamber.
- Save top 20 Q3 violation pairs to `data/q3_top_pairs_{model}_{chamber}.parquet`.
- Add heatmap figure and top violations table to PPC HTML report after Q3 summary.
- Bills with high |Q3| share variance unexplained by IRT — suggests missing dimensions or local dependencies.

### Phase 11 (Network): Residual Network

- After main network analysis, compute predicted pairwise agreement from IRT ideal points.
- Residual = observed Kappa − IRT-predicted Kappa.
- Build network on residual edges (|residual| > 0.15).
- Run Leiden community detection on residual network.
- Save top residual edges and plot colored by sign (green = more agreement than expected, red = less).
- Add residual network figure, edge table, and interpretation to network HTML report.

## Consequences

**Phase 02:** TEFI adds ~10 lines of computation (one eigendecomposition). Provides an objective, non-subjective dimensionality metric alongside the scree plot. Properly penalizes over-extraction.

**Phase 08:** Q3 heatmap makes local dependence structure visible. Previously, Q3 was a single number; now analysts can see *which* bill pairs violate local independence. Helps identify amendment cascades and procedural vote sequences.

**Phase 11:** Residual network bridges the IRT/network gap. Legislators with strong residual edges co-vote in ways ideology doesn't explain — geographic caucuses, committee effects, personal alliances. Communities in the residual network suggest missing dimensions or non-ideological voting blocs.

**No new dependencies.** All enhancements use existing libraries (numpy, matplotlib, igraph, leidenalg).

**Minimal runtime impact.** TEFI: ~0.1s. Q3 heatmap: ~1s (matplotlib). Residual network: ~2s (loop over pairs + Leiden).
