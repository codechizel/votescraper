# Implementation Plan: PCA Axis Instability Fixes

**Source:** `docs/pca-ideology-axis-instability.md`
**Date:** 2026-03-15
**Status:** COMPLETED + VALIDATED (2026-03-15, v2026.03.15.4 through v2026.03.15.16)

All 7 recommendations (R1-R7) implemented, deployed, and validated on the 79th (2001-2002) pipeline — the worst-case session. Three additional runtime fixes (v13-v15) and figure init labeling (v16) shipped during validation. See ADR-0118 for the full decision record and validation results.

This plan implements the seven recommendations (R1-R7) from the PCA axis instability deep dive. The fixes are ordered by dependency — later tasks consume outputs from earlier ones.

---

## Task 1: Party-Aware PCA Initialization (R1)

**Priority:** High — affects first pipeline run
**File:** `analysis/init_strategy.py`

### Changes

Add a `detect_ideology_pc()` function that computes point-biserial correlation between each PC and a binary party indicator (R=1, D=0). Return the PC column name with the strongest absolute party correlation.

Modify `resolve_init_source()` for the `pca-informed` strategy:
1. Call `detect_ideology_pc()` on the PCA scores DataFrame
2. If the best PC is not `PC1` and its |r| > 0.30, use that PC instead
3. Log a warning: `"PC swap detected: {best_pc} has stronger party correlation ({r:.3f}) than PC1 ({pc1_r:.3f}) — using {best_pc} for ideology init"`
4. Add a `pc_swap_detected` field to the return metadata

The `pca_column` parameter already exists in `resolve_init_source()` — this change selects it automatically when `strategy="pca-informed"` rather than hardcoding `"PC1"`.

### Tests

- Synthetic test: create PCA scores where PC2 has stronger party separation than PC1, verify init uses PC2
- Regression test: verify existing sessions where PC1 is correct continue to use PC1
- Edge case: single-party chamber (no Democrats) — should fall back to PC1

### Docs

- Update `analysis/design/pca.md` downstream implications
- Update `docs/pca-ideology-axis-instability.md` to mark R1 as implemented

**Commit + push**

---

## Task 2: PCA Report Axis-Swap Warning (R6)

**Priority:** Medium — improves human interpretability
**Files:** `analysis/02_pca/pca.py`, `analysis/02_pca/pca_report.py`

### Changes

In `pca.py`, after `orient_pc1()`, compute Cohen's d for party separation on PC1 and PC2. Store in the chamber results dict as `pc1_party_d` and `pc2_party_d`.

In `pca_report.py`, when `pc2_party_d > pc1_party_d` and `pc2_party_d > 2.0`, inject a warning banner (using `horseshoe_warning_html()` pattern from `phase_utils.py`):

> **Axis swap detected:** In this chamber, PC2 (d = {pc2_d:.1f}) separates parties more strongly than PC1 (d = {pc1_d:.1f}). PC1 captures intra-majority-party factional variation. Downstream IRT initialization should account for this — see `docs/pca-ideology-axis-instability.md`.

Add the party-d values to the existing PCA summary table in the report.

### Tests

- Test warning fires for 79th Senate (known PC2 > PC1)
- Test warning does NOT fire for 91st Senate (PC1 > PC2)
- Test warning HTML output format

**Commit + push**

---

## Task 3: 1D IRT Party Separation Quality Gate (R2)

**Priority:** High — directly addresses worst-case failure
**Files:** `analysis/05_irt/irt.py`

### Changes

After IRT sampling and extraction, compute Cohen's d between party mean ideal points. Add to the convergence summary JSON:

```python
party_d = compute_party_cohen_d(ideal_points)
convergence["party_separation_d"] = party_d
convergence["axis_uncertain"] = party_d < 1.5
```

When `axis_uncertain` is True, print a warning:

```
WARNING: Low party separation (d = {party_d:.2f} < 1.5). The 1D IRT may be
estimating intra-party factional variation rather than ideology.
See docs/pca-ideology-axis-instability.md
```

Add the `axis_uncertain` flag to `convergence_summary.json` so downstream phases (synthesis, profiles, canonical routing) can consume it.

### Downstream consumption

Update the canonical routing (`canonical_ideal_points.py`) to check `axis_uncertain` in the 1D convergence summary. When True, treat the 1D result as less trustworthy in the routing chain — prefer 2D even at Tier 2 rather than falling back to a 1D result flagged as axis-uncertain.

### Tests

- Test with 79th Senate data: verify d < 1.5 triggers flag
- Test with 91st Senate data: verify d > 1.5 does not trigger
- Test convergence_summary.json schema includes new fields

**Commit + push**

---

## Task 4: Fix Tier 2 Circular Dependency (R3)

**Priority:** High — existing gate actively rejects correct results
**File:** `analysis/canonical_ideal_points.py`

### Changes

In `assess_2d_convergence_tier()`, replace the PCA PC1 rank correlation check with a party separation check:

**Current (circular):**
```python
rank_corr = _compute_rank_correlation(ip_2d, pca_scores)
if rank_corr > TIER2_RANK_CORR_THRESHOLD:  # 0.70
    tier = 2
```

**Proposed (direct):**
```python
party_d = _compute_party_separation(ip_2d)  # Cohen's d
if party_d > TIER2_PARTY_D_THRESHOLD:  # 1.5
    tier = 2
```

Add `_compute_party_separation()` helper:
```python
def _compute_party_separation(ip: pl.DataFrame) -> float:
    """Cohen's d between R and D mean ideal points."""
    r = ip.filter(pl.col("party") == "Republican")["xi_mean"]
    d = ip.filter(pl.col("party") == "Democrat")["xi_mean"]
    if r.len() == 0 or d.len() == 0:
        return 0.0
    pooled_sd = sqrt((r.std()**2 + d.std()**2) / 2)
    return abs(r.mean() - d.mean()) / pooled_sd if pooled_sd > 0 else 0.0
```

Keep the PCA correlation as a secondary diagnostic (logged, not gating). Update the routing manifest schema to include `party_separation_d` alongside the existing `rank_corr`.

The `pca_dir` parameter remains optional for backward compatibility but is no longer required for Tier 2.

### Constants

```python
TIER2_PARTY_D_THRESHOLD = 1.5  # Minimum party separation for credible ideology estimate
```

### Tests

- Test that 79th 2D Dim 1 (d = 6.17) now passes Tier 2 despite low PCA correlation
- Test that a random dimension (d < 1.5) fails Tier 2
- Test backward compat: existing routing manifests still parse correctly

**Commit + push**

---

## Task 5: Hierarchical Minimum Separation Guard (R4)

**Priority:** Medium
**File:** `analysis/07_hierarchical/hierarchical.py`

### Changes

In `build_per_chamber_graph()`, after the `mu_party = pt.sort(mu_party_raw)` line, add a soft minimum-separation penalty:

```python
# Soft guard: penalize solutions where party means are < 0.5 apart
# The sort constraint ensures D < R, but allows near-zero separation.
# This penalty encourages meaningful party distinction without hard constraining.
MIN_PARTY_SEPARATION = 0.5
pm.Potential(
    "min_party_sep",
    pt.switch(mu_party[1] - mu_party[0] > MIN_PARTY_SEPARATION, 0.0, -100.0),
)
```

Also add the same guard to `build_hierarchical_2d_graph()` in Phase 07b for consistency.

After extraction, compute and log the gap/sigma ratio:
```python
gap = float(mu_party_R - mu_party_D)
ratio = gap / float(sigma_within_R) if sigma_within_R > 0 else 0
print(f"  Party gap: {gap:.3f} ({ratio:.2f} sigma_within_R)")
if ratio < 1.0:
    print(f"  WARNING: Weak party separation ({ratio:.2f}σ). Model may be on wrong axis.")
```

### Tests

- Verify the potential doesn't degrade convergence on clean sessions (91st)
- Verify it prevents near-zero party separation on the 79th

**Commit + push**

---

## Task 6: 2D IRT Dimension Swap Detection (R7)

**Priority:** Medium
**Files:** `analysis/06_irt_2d/irt_2d.py`

### Changes

After `extract_2d_ideal_points()` and `apply_dim1_sign_check()`, add a dimension-swap check:

```python
def check_dimension_swap(ideal_2d: pl.DataFrame) -> bool:
    """Check if Dim 2 separates parties better than Dim 1."""
    dim1_d = _compute_party_d(ideal_2d, "xi_dim1_mean")
    dim2_d = _compute_party_d(ideal_2d, "xi_dim2_mean")
    if dim2_d > dim1_d and dim2_d > 2.0:
        print(f"  DIMENSION SWAP DETECTED: Dim 2 (d={dim2_d:.2f}) separates "
              f"parties better than Dim 1 (d={dim1_d:.2f})")
        return True
    return False
```

When a swap is detected:
1. Swap columns in the ideal points DataFrame (xi_dim1 ↔ xi_dim2, including HDIs)
2. Swap columns in the posterior idata (xi[:,:,:,0] ↔ xi[:,:,:,1])
3. Log the swap prominently
4. Add `dimension_swap_corrected: true` to convergence_summary.json

Do the same in Phase 07b (`hierarchical_2d.py`).

### Tests

- Test with 79th Senate 2D data: verify swap is detected and corrected
- Test with 91st Senate: verify no swap triggered
- Test that corrected ideal points have Dim 1 d > Dim 2 d after swap

**Commit + push**

---

## Task 7: Dynamic IRT Axis-Swap Detection (R5)

**Priority:** High for data quality
**Files:** `analysis/27_dynamic_irt/dynamic_irt.py`, `analysis/27_dynamic_irt/dynamic_irt_data.py`

### Changes

**Sign correction reference:** Modify `fix_period_sign_flips()` to prefer canonical ideal points over raw Phase 05 output as the reference for sign correction. Canonical scores have already been routed through horseshoe detection and (after Task 4) party-separation validation.

```python
# Current: correlate dynamic xi with static 1D IRT xi
# Proposed: correlate dynamic xi with canonical xi (Phase 06 routing output)
```

**Per-period party separation check:** After sign correction, compute per-period party-d. For periods where d < 1.5, add an `axis_uncertain` flag to the period's metadata. These periods should be caveated in the dynamic IRT report.

**Cross-session correlation check:** After all periods are estimated, compute bridge-legislator Pearson correlation between adjacent periods. Log a warning when r < 0.50, which indicates a potential axis swap at that boundary.

### Docs

- Update `analysis/design/dynamic_irt.md` with the axis-uncertainty caveat

### Tests

- Test that canonical scores are used when available
- Test per-period party-d computation
- Test cross-period correlation warning

**Commit + push**

---

## Task 8: Documentation Finalization

**Priority:** Low (after all code changes)

### ADR

Create `docs/adr/0118-party-separation-quality-gates.md`:
- Decision: Add party separation (Cohen's d) as a quality metric at PCA init, 1D IRT output, Tier 2 routing, hierarchical model output, and dynamic IRT sign correction
- Context: PCA axis instability in 7/14 Senate sessions causes wrong-axis estimation cascade
- Consequences: Pipeline can detect and flag axis misalignment without depending on PCA axis ordering

### Article update

Update `docs/pca-ideology-axis-instability.md`:
- Mark each recommendation (R1-R7) as implemented with commit references
- Add a "Resolution" section summarizing what was fixed

### Design docs

- Update `analysis/design/irt_2d.md` with dimension swap detection
- Update `analysis/design/hierarchical.md` with minimum separation guard
- Update `analysis/design/hierarchical_2d.md` with same
- Update `docs/adr/README.md` with ADR-0118

**Commit + push**

---

## Verification

After all tasks complete:

1. `just lint` — clean
2. `just typecheck` — src/ clean, analysis/ warnings-only
3. `just test-fast` — no regressions
4. Re-run 79th Senate pipeline: `just pipeline 79th_2001-2002`
   - PCA report should show axis-swap warning for Senate
   - IRT should use PC2 for init (or best-party-PC)
   - 1D IRT convergence summary should flag `axis_uncertain: true` for Senate
   - 2D IRT should detect and correct dimension swap
   - Canonical routing should use party-d gate (not PCA correlation)
   - Hierarchical IRT should show minimum separation penalty active
5. Re-run 91st Senate pipeline: verify no regressions (no warnings, no swaps)
6. Spot-check 83rd and 88th Senate for correct behavior

---

## Dependency Graph

```
Task 1 (PCA init)          Task 2 (PCA report)
       ↓                          ↓
Task 3 (IRT gate) ──────→ Task 4 (Tier 2 fix)
       ↓                          ↓
Task 5 (Hier guard)        Task 6 (2D swap)
                                   ↓
                           Task 7 (Dynamic IRT)
                                   ↓
                           Task 8 (Docs + ADR)
```

Tasks 1, 2, 5, 6 are independent and can be parallelized.
Task 3 should precede Task 4 (IRT gate feeds routing).
Task 7 depends on Task 4 (canonical routing must be fixed first).
Task 8 is last (documents all changes).
