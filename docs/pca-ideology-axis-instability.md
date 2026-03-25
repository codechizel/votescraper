# PCA Ideology Axis Instability in the Kansas Senate

**Date:** 2026-03-14 (initial), 2026-03-15 (deep dive with literature review)
**Sparked by:** Reviewing the 79th (2001-2002) PCA report, sections 6 and 26 (House and Senate score scatter matrices)

---

## The Observation

In the 79th Senate PCA scatter matrix, PC1 shows no party separation whatsoever — Republicans and Democrats are completely intermixed. But PC2 shows textbook party separation, with Democrats clustered at the negative end and Republicans at the positive end. This is the opposite of the House, where PC1 is clearly the ideology axis.

The initial question: is PC2, not PC1, the ideology factor for the 79th Senate?

## The Answer: Yes, and It's Not a One-Off

Measuring party separation with Cohen's d (effect size: how many pooled standard deviations apart are the party means), we find that **ideology lands on PC2 in 7 of 14 sessions** for the Kansas Senate:

| Session | Senate R% | House PC1 d | Senate PC1 d | Senate PC2 d | Senate ideology on |
|---------|-----------|-------------|--------------|--------------|-------------------|
| 78th (1999-2000) | 68% | 4.97 | 2.11 | 2.56 | PC2 |
| **79th (2001-2002)** | **75%** | **5.37** | **0.28** | **4.98** | **PC2** |
| 80th (2003-2004) | 72% | 3.48 | 0.27 | 2.56 | PC2 |
| 81st (2005-2006) | 75% | 3.91 | 1.72 | 2.25 | PC2 |
| 82nd (2007-2008) | 75% | 3.88 | 0.89 | 2.41 | PC2 |
| 83rd (2009-2010) | 78% | 3.30 | 0.79 | 4.40 | PC2 |
| 84th (2011-2012) | 81% | 5.37 | 1.84 | 1.30 | ambiguous |
| 85th (2013-2014) | 79% | 4.77 | 6.83 | 0.19 | PC1 |
| 86th (2015-2016) | 80% | 3.84 | 5.69 | 0.09 | PC1 |
| 87th (2017-2018) | 71% | 2.66 | 2.10 | 1.87 | PC1 |
| 88th (2019-2020) | 73% | 8.74 | 1.72 | 3.10 | PC2 |
| 89th (2021-2022) | 70% | 7.20 | 7.21 | 0.60 | PC1 |
| 90th (2023-2024) | 72% | 6.69 | 6.75 | 0.38 | PC1 |
| 91st (2025-2026) | 75% | 7.30 | 7.49 | 0.14 | PC1 |

**The 79th is the most extreme case**: Senate PC1 d = 0.28 (essentially zero party separation) while PC2 d = 4.98 (massive separation). The House is completely normal in every session (PC1 d always > 2.5).

The pattern shifts around the 84th-85th (2011-2014). Before that, the Senate almost always has ideology on PC2. After, it's consistently on PC1. The 88th is a late-era exception.

## Why This Happens

PCA extracts components in order of **variance explained**, not in order of **substantive meaning**. This is not a bug — it is a mathematical property. As de Leeuw (2018) notes, PCA dimensions have no inherent substantive interpretation; labeling them "ideology" is an interpretive act that must be validated against external criteria.

In the Kansas Senate:

- **Before ~2013**: The largest source of variance in roll-call votes was intra-Republican factional disagreement (moderate establishment vs. conservative insurgent). With 70-78% Republicans, this intra-caucus axis dominates total variance. The party divide, while sharp, explains less variance because there are fewer Democrats to contribute to it. So PCA captures the intra-R axis as PC1 (19.6% variance in the 79th) and the party axis as PC2 (13.6%).

- **After ~2013**: The 2012 Kansas Republican primary ("the purge") swept out many moderate Republican senators. The surviving caucus was more ideologically homogeneous, reducing intra-R variance. The party divide re-emerged as the dominant axis, and PC1 recaptured ideology.

The eigenvalue ratios confirm this. In the 79th Senate, λ₁/λ₂ = 1.45 — barely above 1, meaning the first two components capture similar amounts of variance and could easily swap. In the 91st Senate, λ₁/λ₂ = 4.42 — PC1 dominates overwhelmingly.

**What PC1 actually captures in the 79th Senate:** Loading analysis shows PC1's top-10 highest-loading bills are all final action and conference committee reports with lopsided margins (31-8, 35-5, 25-3) — consensus-leaning votes where the far-right faction defects. The PC1 legislator rankings have Tim Huelskamp (R, far-right) at -31.0 and Sandy Praeger (R, moderate) at +9.3 — a 40-point spread among Republicans, with Democrats scattered between. The top-10 PC1 bills and top-10 PC2 bills have zero overlap.

### Literature Context

No published paper explicitly documents the PCA axis-swap in supermajority state legislatures, but the theoretical basis is straightforward from PCA's mathematical properties. Poole & Rosenthal's body of work on DW-NOMINATE shows that PC1 reliably captures the party divide in the modern US Congress because between-party variance dominates in a balanced, polarized chamber. That condition does not hold in Kansas's lopsided Senate.

Diaconis, Goel & Holmes (2008) showed that PCA on seriated data produces a "horseshoe" artifact where PC2 ≈ (PC1)², but the 79th Senate pattern is distinct: PC1 and PC2 capture genuinely different political cleavages, not an artifact of curvilinear projection. The horseshoe is a separate (though related) problem.

---

## Downstream Propagation: How Deep Does It Go?

### 1D IRT Ideal Points (Phase 05)

The 1D IRT model finds the latent dimension that **maximizes discrimination** — the axis along which vote outcomes are most predictable from legislator positions. When intra-Republican factional votes dominate the contested portion of the roll-call matrix, that axis is the factional one, not the partisan one.

Checking IRT ideal point correlation with PCA components and IRT party separation across all sessions:

| Session | S-PC1 d | S-PC2 d | IRT ρ(PC1) | IRT ρ(PC2) | IRT party d | Alignment | Assessment |
|---------|---------|---------|------------|------------|-------------|-----------|------------|
| 78th | 2.11 | 2.56 | 0.962 | 0.314 | 4.15 | 1.76 | OK — IRT found party divide |
| **79th** | **0.28** | **4.98** | **0.974** | **-0.067** | **0.86** | **0.17** | **IRT on wrong axis** |
| **80th** | **0.27** | **2.56** | **0.898** | **-0.331** | **0.59** | **0.17** | **IRT on wrong axis** |
| 81st | 1.72 | 2.25 | 0.961 | 0.355 | 2.13 | 0.81 | Marginal |
| **82nd** | **0.89** | **2.41** | **0.954** | **0.534** | **1.26** | **0.49** | **IRT on wrong axis** |
| **83rd** | **0.79** | **4.40** | **0.911** | **0.500** | **1.69** | **0.33** | **IRT on wrong axis** |
| 84th | 1.84 | 1.30 | 0.567 | 0.200 | 0.02 | 0.02 | Different pathology (anchor inversion) |
| 85th | 6.83 | 0.19 | 0.908 | 0.007 | 4.57 | 0.80 | Normal |
| 86th | 5.69 | 0.09 | 0.922 | 0.377 | 4.48 | 0.85 | Normal |
| 87th | 2.10 | 1.87 | 0.956 | 0.255 | 2.70 | 1.38 | Normal |
| 88th | 1.72 | 3.10 | 0.965 | 0.488 | 3.42 | 1.18 | OK — IRT found party divide |
| 89th | 7.21 | 0.60 | 0.941 | -0.032 | 4.60 | 0.70 | Normal |
| 90th | 6.75 | 0.38 | 0.937 | 0.146 | 4.35 | 0.72 | Normal |
| 91st | 7.49 | 0.14 | 0.751 | -0.335 | 6.04 | 0.67 | Normal |

**Alignment score** = IRT party d / max(PCA-PC1-d, PCA-PC2-d). Values near 1.0 mean the IRT captures the full available party signal; values near 0 mean it's on the wrong axis.

**In the 79th Senate, the IRT captures only 17% of the available party signal.** The IRT beta (discrimination) parameters correlate 0.816 with PC1 loadings and -0.006 with PC2 loadings — confirming the model's discrimination axis aligns with the factional axis, not the partisan one.

The ranking produced is not just noisy — it's substantively wrong:

```
79th Senate IRT "most liberal":       79th Senate IRT "most conservative":
  -3.259  Republican  Huelskamp         +2.712  Republican  Vratil
  -2.251  Republican  Lyon              +2.645  Republican  Oleen
  -2.223  Republican  Pugh              +2.610  Republican  Teichman
  -1.911  Republican  Tyson             +2.406  Republican  Adkins
  -1.746  Republican  O'Connor          +2.278  Republican  Kerr
```

The conservative insurgents (Huelskamp, Tyson) rank as "liberal" and the moderate establishment (Vratil, Oleen) rank as "conservative." The 10 Democrats are scattered through the middle.

This is not the horseshoe effect as typically understood (where 1D conflates two dimensions into a curve). This is the **wrong-axis problem**: the IRT model found a genuine latent dimension, but it's the factional dimension, not the partisan one.

### Hierarchical 1D IRT (Phase 07)

The hierarchical model's `sort(mu_party)` constraint is the strongest defense in the pipeline — it forces D < R ordering on the party means. But when the discrimination parameters align with the factional axis, the party means converge to near-zero separation:

| Session | mu_D | mu_R | Gap | sigma_within_R | Gap/sigma ratio |
|---------|------|------|-----|----------------|-----------------|
| **79th** | **-1.10** | **-0.51** | **0.59** | **2.70** | **0.22** |
| **80th** | -2.25 | -0.89 | 1.36 | 1.85 | 0.74 |
| **82nd** | -2.76 | +0.26 | 3.02 | 2.67 | 1.13 |
| **83rd** | -4.68 | -0.90 | 3.79 | 2.30 | 1.64 |
| 85th (control) | -8.30 | +2.61 | 10.90 | 2.09 | 5.21 |
| 91st (control) | -7.26 | +4.76 | 12.02 | 1.74 | 6.91 |

**The 79th hierarchical IRT places the party means 0.59 apart — only 0.22 within-party standard deviations.** The model literally cannot tell the parties apart. Clean sessions achieve gap/sigma ratios of 3-7.

### 2D IRT (Phase 06)

The 2D model tells an interesting story. Even in the 79th (where convergence failed, R-hat = 1.69, ESS = 6.4), the unconverged point estimates show:

- **Dim 1 party d = 6.17** — perfect party separation
- **Dim 2 party d = -0.12** — no party signal (captures the R factional axis)

The 2D model successfully disentangles the two cleavages. But Dim 1 correlates with PCA PC2 (r = 0.26 with PC1, strong with PC2), confirming the PLT identification anchored on the wrong axis. The dimensions are swapped relative to the labeling assumption (Dim 1 ≠ PC1).

For the 80th Senate, where 2D IRT converged better (R-hat = 1.06, ESS = 119), the same pattern holds: Dim 2 has the party signal (d = -4.18), not Dim 1 (d = -0.003).

### Cross-Session Comparability (Phase 27)

The Dynamic IRT assumes a single consistent latent dimension across sessions. Bridge legislator correlations reveal this assumption is violated:

| Transition | n_bridge | Pearson r | Category |
|-----------|----------|-----------|----------|
| 78th→79th | 25 | **-0.071** | Clean → Affected |
| 79th→80th | 38 | +0.696 | Affected → Affected |
| 80th→81st | 26 | +0.712 | Affected → Clean |
| 82nd→83rd | 30 | +0.792 | Affected → Affected |
| **83rd→84th** | **31** | **-0.017** | **Affected → Ambiguous** |
| 85th→86th | 38 | +0.933 | Clean → Clean |
| 88th→89th | 25 | +0.971 | Clean → Clean |
| 89th→90th | 37 | +0.907 | Clean → Clean |
| 90th→91st | 26 | +0.953 | Clean → Clean |

**Transitions between affected and clean sessions show near-zero or negative correlations** (78th→79th: r = -0.07, 83rd→84th: r = -0.02), while clean-to-clean transitions maintain r > 0.85. Within-affected transitions (79th→80th, 82nd→83rd) show moderate correlations (~0.70) because both sessions have IRT on the same (wrong) axis.

The Dynamic IRT's sign correction function (`fix_period_sign_flips`) compares each period's dynamic xi against the static 1D IRT from Phase 05 using Pearson correlation. This propagates axis misalignment: if the static reference is on the wrong axis, the dynamic model either aligns to the wrong axis or triggers a spurious sign correction.

### Canonical Routing and the Circular Dependency

The canonical routing (Phase 06) correctly detects the horseshoe in the 79th Senate. But the fallback chain has a critical flaw:

1. Horseshoe detected → try 2D IRT
2. 2D IRT convergence check → **Tier 2 validates by correlating 2D Dim 1 with PCA PC1**
3. If PCA PC1 is not ideology, a correct 2D Dim 1 (ideology) shows *low* correlation with the wrong PC1 → **Tier 2 rejects the correct result**
4. Conversely, a contaminated 2D Dim 1 (aligned with wrong PC1) shows *high* correlation → **Tier 2 accepts the wrong result**
5. When both tiers fail → fall back to 1D IRT, which is on the wrong axis

This circular dependency on PCA means the quality gate can fail in both directions: rejecting correct results and accepting wrong ones.

---

## Literature Context

### What the Field Knows

The legislative scaling literature has extensively studied identification in IRT models (Clinton, Jackman & Rivers 2004; Bafumi, Gelman, Park & Kaplan 2005) and dimensionality in roll-call voting (Poole & Rosenthal 1997/2007). The standard treatments assume balanced or moderately imbalanced chambers. The specific problem of PCA axis instability in state-level supermajority chambers appears to be **undocumented** — a known-but-unnamed practitioner issue.

Key relevant findings from the literature:

**Identification is fundamentally about the researcher's intent.** Lauderdale & Clark (2024, *APSR*) argue that standard IRT identification strategies (anchors, sign constraints, positive discrimination) resolve the mathematical identification problem but not the **substantive** identification problem — which dimension corresponds to which political concept. Their IRT-M framework encodes theoretical relationships between items and dimensions via constraint matrices, ensuring dimensions capture specified concepts (e.g., "economic ideology" vs. "social issues"). This requires at least d(d-1) zero constraints pre-coded by the researcher.

**L1 distance breaks rotational invariance.** Shin, Lim & Park (2024, *JASA*) propose using Manhattan distance instead of Euclidean distance for multidimensional ideal point estimation. Under L1, the likelihood is no longer invariant under arbitrary rotation — only under signed permutations of axes (2^D × D! discrete modes). This dramatically reduces the identification problem from infinite rotations to a finite, tractable set.

**PCA initialization is standard but underdocumented.** Imai, Lo & Olmsted (2016) use SVD-based starting values in their emIRT package. The idealstan package (Kubinec) uses Pathfinder variational inference for automatic anchor selection and initialization. No published work addresses the scenario where PCA initialization puts the sampler on the wrong axis.

**Cross-session comparability relies on bridge actors.** DW-NOMINATE (Poole & Rosenthal) uses bridge legislators; Shor & McCarty (2011) use state legislators who later serve in Congress to anchor state-level scores on a national scale. Martin & Quinn (2002) use random-walk priors for Supreme Court dynamics. None of these methods address the case where the underlying dimensional structure genuinely changes between sessions.

### What the Field Doesn't Know

1. **No published paper documents the PCA axis-swap in supermajority state legislatures.** The theoretical basis is sound, but it hasn't been written up as a named phenomenon.

2. **No standard method exists for detecting when PCA initialization has put an IRT model on the wrong axis.** The problem is well-defined; the solution is not.

3. **Cross-session comparability with dimension swaps remains an open problem.** Bridge legislators and Procrustes rotation assume the dimensional structure is stable. When it genuinely changes (as between the 83rd and 85th Kansas Senates), existing methods break.

---

## Pipeline Code Vulnerability Analysis

A systematic trace of every code path affected by the PC1 ≠ ideology problem:

| Code Path | Assumption | Defense | Severity |
|-----------|-----------|---------|----------|
| `init_strategy.py` (PCA-informed) | PC1 = ideology | None | **High** on first run |
| `irt.py` (1D IRT anchors) | Max-discrimination axis = party axis | Agreement-based anchors constrain sign, not orientation | **Medium** |
| `hierarchical.py` (sort constraint) | Party means can separate | `sort(mu_party)` forces ordering but not separation magnitude | **Low-Medium** |
| `irt_2d.py` (2D init) | PC1 → Dim 1, PC2 → Dim 2 | `apply_dim1_sign_check()` checks sign only, not axis swap | **High** |
| `canonical_ideal_points.py` (Tier 2 gate) | PCA PC1 validates 2D Dim 1 | Horseshoe detection fires, but PCA correlation is circular | **High** |
| `dynamic_irt.py` (sign correction) | Static IRT provides correct reference | HalfNormal beta + random walk smoothing | **High** |

### The Fundamental Gap

**The pipeline has no party-separation metric used as a quality gate.** The horseshoe detector checks for party *overlap* in 1D output (a symptom), but no phase validates that the estimated axis actually separates parties. Adding a party-separation check (Cohen's d or biserial correlation with party) at critical decision points would break the dependency on PCA axis stability.

---

## Recommendations

### R1. Party-Aware PCA Initialization (init_strategy.py)

**Problem:** `resolve_init_source()` with `strategy="pca-informed"` blindly uses PC1 regardless of whether it carries party signal.

**Fix:** Before returning PCA-informed values, compute the point-biserial correlation between each PC and the binary party indicator (R=1, D=0). Use the PC with the strongest absolute party correlation as the init source, not necessarily PC1.

```
pc1_party_r = biserial_correlation(PC1, party)
pc2_party_r = biserial_correlation(PC2, party)
if abs(pc2_party_r) > abs(pc1_party_r) and abs(pc2_party_r) > 0.30:
    use PC2 for ideology init, log warning
```

This is cheap to compute (no MCMC, just a correlation) and breaks the PC1-is-ideology assumption at the source.

**Priority:** High. Affects first pipeline run where no IRT/canonical data exists yet.

### R2. 1D IRT Party Separation Quality Gate (irt.py)

**Problem:** The 1D IRT can converge cleanly (good R-hat, good ESS) while measuring the wrong dimension. Convergence diagnostics don't detect axis misalignment.

**Fix:** After sampling, compute Cohen's d between party mean ideal points. If d < 1.5 (the minimum we observe in sessions where 1D works), flag the result as "axis-uncertain" in the convergence summary. Downstream phases (synthesis, profiles) should consume this flag and caveat their interpretations.

This is *not* a hard gate — the 1D model may be correct in measuring intra-party variance, which is itself substantively interesting. But it should not be silently consumed as "ideology" by downstream phases.

**Priority:** High. Directly addresses the worst-case failure mode.

### R3. Fix the Tier 2 Circular Dependency (canonical_ideal_points.py)

**Problem:** The Tier 2 quality gate validates 2D Dim 1 by correlating with PCA PC1. When PC1 is contaminated, this rejects correct results and accepts wrong ones.

**Fix:** Replace the PCA correlation check with a party-separation check on the 2D Dim 1 scores:

```
# Current (circular):
rank_corr = spearman(2d_dim1, pca_pc1)  # ← breaks when PC1 ≠ ideology

# Proposed (direct):
party_d = cohen_d(2d_dim1, by=party)     # ← validates the actual claim
```

If the 2D Dim 1 separates parties with d > 1.5, it's a credible ideology estimate regardless of which PCA component it correlates with. This breaks the circular dependency entirely.

**Priority:** High. The existing gate actively rejects correct results for affected sessions.

### R4. Hierarchical Minimum Separation Guard (hierarchical.py)

**Problem:** The `sort(mu_party)` constraint forces correct party *ordering* but allows party means to converge to near-zero separation (gap/sigma = 0.22 in the 79th).

**Fix:** Add a soft minimum-separation penalty:

```python
# After mu_party = pt.sort(mu_party_raw):
pm.Potential("min_sep", pt.switch(mu_party[1] - mu_party[0] > 0.5, 0.0, -100.0))
```

This is a weak guard — it penalizes solutions where party means are less than 0.5 apart, without rigidly constraining the posterior. The threshold of 0.5 is conservative (the minimum observed gap/sigma in clean sessions is 2.87).

**Priority:** Medium. The sort constraint already provides partial defense. This strengthens it.

### R5. Dynamic IRT Axis-Swap Detection (dynamic_irt.py)

**Problem:** The sign correction function uses static 1D IRT as the reference. When the static IRT is on the wrong axis, the dynamic model inherits the contamination.

**Fix:** Two changes:

1. Use canonical ideal points (which route through horseshoe detection) instead of raw Phase 05 output as the sign-correction reference.
2. Add a per-period party-separation check. If R_mean - D_mean for a period's dynamic xi is near zero or reversed, flag that period as axis-uncertain.

For the affected sessions (78th-83rd), the Dynamic IRT should probably treat these as a separate regime with a different latent dimension, rather than forcing them onto the same axis as later sessions. This is a design decision that warrants its own ADR.

**Priority:** High for data quality. Medium for implementation (requires careful thought about regime-change handling).

### R6. PCA Report Phase — Annotate the Axis Swap (pca.py)

**Problem:** The PCA report doesn't flag when PC1 ≠ ideology. A reader seeing the 79th Senate scatter matrix has to figure it out themselves.

**Fix:** In the PCA report builder, compute party-d for PC1 and PC2. If PC2 has stronger party separation than PC1, add a prominent warning banner:

> **Axis swap detected:** In this chamber, PC2 (not PC1) captures the party divide (d = 4.98 vs 0.28). PC1 captures intra-Republican factional variation. This affects downstream IRT initialization and should be accounted for when interpreting ideal points.

**Priority:** Medium. Improves interpretability for human readers.

### R7. 2D IRT Dimension Swap Detection (irt_2d.py)

**Problem:** When PCA PC1 ≠ ideology, the 2D IRT initializes with dimensions swapped. The PLT identification anchors rotation on the first bill, not on substantive meaning. Post-hoc sign check detects 180-degree flips but not 90-degree axis swaps.

**Fix:** After sampling, compute party-d on both Dim 1 and Dim 2. If Dim 2 separates parties better than Dim 1, the dimensions are likely swapped. Swap the columns in the posterior and re-label. This is the same "post-hoc dimension assignment" approach that is common in practice but rarely formalized.

**Priority:** Medium. Mainly matters for the 2D report interpretation and for the canonical routing chain.

### What Not to Fix

1. **PCA itself.** PC1 capturing intra-R variance is a genuine finding, not a bug. The PCA phase is correct.

2. **The House.** PC1 is consistently ideology across all 14 sessions. No changes needed.

3. **Sessions after 2013.** The standard pipeline works correctly for the 85th-91st (excluding the 88th's mild PC2 lean, where IRT still finds the party divide successfully with d = 3.42).

4. **Wholesale adoption of IRT-M or L1-based estimation.** Lauderdale & Clark's IRT-M (2024) and Shin et al.'s L1-based BMIM (2024) are principled solutions to the dimension-labeling and rotation problems, but they require substantial implementation effort and are very new. The targeted fixes above (R1-R7) address the immediate data quality issues without a framework rewrite. IRT-M and BMIM should be tracked for future consideration.

---

## Summary

The Kansas Senate's political structure changed fundamentally between 1999 and 2014. During the moderate-vs-conservative factional war (78th-83rd), intra-Republican variance dominated and PCA placed the party divide on PC2. After the 2012 primary purge, the party divide returned to PC1. This structural shift propagates through the entire pipeline: 1D IRT ideal points, hierarchical party means, 2D IRT dimension labels, canonical routing, and cross-session dynamics.

The pipeline's current defenses (horseshoe detection, sign checks, sort constraints) address symptoms but not the root cause. The root cause is that **no phase validates whether its estimated axis actually separates parties**. The seven recommendations above add party-separation checks at critical decision points, breaking the pipeline's implicit assumption that "maximum variance = ideology."

---

## Resolution (2026-03-15)

All seven recommendations have been implemented in commits v2026.03.15.4 through v2026.03.15.16:

| # | Fix | Commit |
|---|-----|--------|
| R1 | Party-aware PCA init (`detect_ideology_pc`) | v2026.03.15.4 |
| R6 | PCA report axis-swap warning banner | v2026.03.15.5, v2026.03.15.13 |
| R4 | Hierarchical minimum separation guard | v2026.03.15.6 |
| R2 | 1D IRT party separation quality gate | v2026.03.15.7 |
| R3 | Tier 2 quality gate — party-d replaces PCA correlation | v2026.03.15.8 |
| R7 | 2D IRT dimension swap detection and correction | v2026.03.15.9 |
| R5 | Dynamic IRT canonical reference + per-period party-d | v2026.03.15.10 |
| — | Figure init-source labels on 2D scatter + forest plots | v2026.03.15.16 |
| — | H2D report builder API fixes | v2026.03.15.14, v2026.03.15.15 |

Architecture decision: ADR-0118 (party separation quality gates across pipeline).

### Validation: 79th (2001-2002) Pipeline Run

Full pipeline completed successfully on the worst-case session (run `79-260314.3`). Every quality gate fired correctly:

| Gate | Result |
|------|--------|
| R6: PCA axis-swap warning | Senate PC2 d=5.21 > PC1 d=0.29 — **warning fired** |
| R1: PCA init PC swap | Senate init used PC2 (party r stronger than PC1) |
| R2: 1D IRT party-d gate | Senate d=1.19 — **axis_uncertain flagged** |
| R3: Tier 2 party-d gate | Senate 2D Dim 1 d=6.05 — **Tier 2 passed** (old PCA gate would have rejected) |
| R4: Hierarchical min-sep | Senate gap=1.308, ratio=0.48σ (improved from 0.22σ pre-R4) |
| R7: 2D dimension swap | Detected and corrected in 2D IRT |
| Canonical routing | Senate source: **`hierarchical_2d_dim1`** (d=4.36) — best available estimate |

The 79th Senate canonical ideal points now use the party-pooled 2D model (Phase 07b) instead of the wrong-axis 1D IRT. This is the exact failure mode the deep dive identified.

### Remaining Gap: Dimension Correctness (2026-03-25, ADR-0123)

A systemic audit cross-validating all IRT dimensions against W-NOMINATE Dim 1 across all 28 chamber-sessions revealed that the R1-R7 party-separation gates are **necessary but not sufficient**. In 6/28 sessions, the canonical dimension passes all party-separation gates but disagrees with W-NOMINATE Dim 1 — the hierarchical model's party-pooling prior creates artificial party separation on a non-ideology dimension. These include the 79th Senate (canonical H2D-1 r=0.330 vs W-NOMINATE, while 1D IRT r=0.989), the 84th House/Senate, 85th House, 80th Senate, and 88th Senate.

The fix: a W-NOMINATE cross-validation gate (ADR-0123) that checks the canonical IRT dimension against W-NOMINATE Dim 1 and swaps to a better IRT dimension if one exists. This uses W-NOMINATE's unsupervised dimension identification as an oracle, while retaining IRT's posterior uncertainty for the actual ideal point estimates.

See `docs/84th-legislature-common-space-analysis.md` for the full per-session cross-validation table.

### Key References

- Bafumi, J., Gelman, A., Park, D. K., & Kaplan, N. (2005). Practical issues in implementing and understanding Bayesian ideal point estimation. *Political Analysis*, 13(2), 171-187.
- Clinton, J., Jackman, S., & Rivers, D. (2004). The statistical analysis of roll call data. *APSR*, 98(2), 355-370.
- de Leeuw, J. (2018). Estimating ideal points from roll-call data: Explore PCA, especially for more than one dimension? *Social Sciences*, 7(1), 12.
- Diaconis, P., Goel, S., & Holmes, S. (2008). Horseshoes in multidimensional scaling and local kernel methods. *Annals of Applied Statistics*, 2(3), 777-807.
- Imai, K., Lo, J., & Olmsted, J. (2016). Fast estimation of ideal points with massive data. *APSR*, 110(4), 631-656.
- Lauderdale, B. E., & Clark, T. S. (2024). Measurement that matches theory: Theory-driven identification in IRT models. *APSR*.
- Martin, A. D., & Quinn, K. M. (2002). Dynamic ideal point estimation via MCMC for the US Supreme Court, 1953-1999. *Political Analysis*, 10(2), 134-153.
- Poole, K. T., & Rosenthal, H. (2007). *Ideology and Congress* (2nd ed.). Transaction.
- Shin, M., Lim, D., & Park, J. (2024). L1-based Bayesian ideal point model for multidimensional politics. *JASA*, 120(550).
- Shor, B., & McCarty, N. (2011). The ideological mapping of American legislatures. *APSR*, 105(3), 530-551.
