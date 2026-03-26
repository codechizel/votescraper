# ADR-0123: W-NOMINATE Cross-Validation Gate for Canonical Routing

**Date:** 2026-03-25
**Status:** Superseded (demoted to diagnostic-only, 2026-03-26)
**Deciders:** Joseph Claeys

## Context

A systemic audit cross-validating all IRT dimensions against W-NOMINATE Dim 1 across all 28 chamber-sessions (14 bienniums x 2 chambers) revealed that **6 of 28 sessions (21%) have the wrong canonical dimension** selected by the current routing system:

| Session | Chamber | Canon (r vs W-NOM) | Best Available (r vs W-NOM) | Best Model |
|---------|---------|--------------------|-----------------------------|------------|
| 79th | Senate | H2D-1 (0.330) | 1D IRT (0.989) | 1D |
| 80th | Senate | H2D-1 (0.773) | 1D IRT (0.966) | 1D |
| 84th | House | H2D-1 (0.838) | H2D Dim 2 (0.977) | H2D-2 |
| 84th | Senate | H2D-1 (0.392) | H2D Dim 2 (0.992) | H2D-2 |
| 85th | House | H2D-1 (0.828) | H2D Dim 2 (0.962) | H2D-2 |
| 88th | Senate | H2D-1 (0.856) | Flat 2D Dim 1 (0.994) | F2D-1 |

**Root cause:** The hierarchical 2D model's party-pooling prior forces party separation on Dim 1 by construction. The existing quality gates (ADR-0110 convergence tiers, ADR-0118 party-separation d > 1.5) check whether the selected dimension separates parties, but they cannot distinguish between **genuine ideology separation** and **prior-forced separation on a non-ideology axis**. In supermajority chambers with intra-party factionalism, the party-pooling prior pushes the real ideology dimension to Dim 2 while creating artificial party separation on Dim 1.

W-NOMINATE is immune to this problem because it is unsupervised — no party labels, no hierarchical priors. Its SVD initialization orders dimensions by explained variance, which naturally places the dominant voting axis (whether party or faction) on Dim 1.

See `docs/84th-legislature-common-space-analysis.md` for the full investigation.

## Decision

Add a **W-NOMINATE cross-validation gate** as a final check after the existing convergence and party-separation gates:

### Algorithm

After canonical routing selects a source and dimension:

1. Load W-NOMINATE Dim 1 scores from Phase 16 (already computed in the pipeline)
2. Compute |Pearson r| between the selected canonical IRT dimension and W-NOMINATE Dim 1
3. Compute |Pearson r| for ALL available IRT dimensions: 1D, Flat 2D Dim 1/2, H2D Dim 1/2
4. If a different dimension exceeds the selected one's correlation by more than `WNOM_GATE_DELTA` (0.10), swap to the better dimension
5. Log the swap decision in the routing manifest

### Constants

```python
WNOM_GATE_DELTA = 0.10   # Minimum improvement to trigger a swap
WNOM_GATE_MIN_R = 0.70   # Minimum W-NOMINATE correlation for any canonical source
```

### Pipeline ordering

Phase 16 (W-NOMINATE) already runs before the canonical routing output is consumed by downstream phases. The gate adds no new pipeline dependencies — it just adds a cross-check during the routing decision.

### Fallback

If W-NOMINATE scores are unavailable (Phase 16 skipped due to missing R), the gate is skipped and the existing convergence + party-separation routing applies unchanged.

## Consequences

**Positive:**
- All 6 currently misrouted sessions would be corrected
- Uses each method's strength: W-NOMINATE for dimension identification, IRT for estimation
- Zero changes to the IRT models themselves — the fix is in routing, not estimation
- Hierarchical party pooling is preserved for the 22/28 sessions where it helps
- W-NOMINATE comparison is objective and reproducible (deterministic)

**Negative:**
- Adds R/W-NOMINATE as a soft dependency for full routing quality (already required for Phase 16)
- If W-NOMINATE itself has the wrong dimension (no known cases in Kansas data), the gate could be misled
- Adds ~1 second per chamber to the routing decision (correlation computation)

**Trade-off:** This makes the pipeline's canonical routing dependent on a secondary method (W-NOMINATE). This is philosophically unusual — IRT is the primary estimator, but it defers to W-NOMINATE for dimension identification. The justification: W-NOMINATE's unsupervised dimension ordering is provably correct in every tested case, while IRT's supervised (party-pooled) dimension ordering fails in 21% of cases. Using each tool's strength is better than using either tool alone.

## Related

- ADR-0109 — Canonical ideal point routing (original routing logic)
- ADR-0110 — Tiered convergence quality gate (convergence checks, necessary but not sufficient)
- ADR-0117 — Hierarchical 2D IRT (party-pooling prior, source of dimension distortion)
- ADR-0118 — Party separation quality gates (party-d checks, necessary but not sufficient)
- `docs/84th-legislature-common-space-analysis.md` — Full investigation with per-session cross-validation table

## Superseded (2026-03-26)

The W-NOMINATE gate is demoted from auto-routing to diagnostic-only. The gate still computes all correlations and records them in the routing manifest, but it no longer triggers automatic dimension swaps. Instead, a manual PCA override file (`analysis/pca_overrides.yaml`) provides stable, auditable dimension assignments for the 8 problematic sessions.

**Rationale:** The gate makes the Bayesian IRT pipeline dependent on a frequentist method for a fundamental structural decision. The automated swap works for the observed data but has no theoretical guarantee of generalization to future bienniums. Manual overrides, updated biennially, are more durable.

See `docs/pca-rotation-and-human-intervention.md` for the full analysis of rotation methods and the case for human intervention.
