# PCA Rotation Methods and the Case for Human Intervention

**Date:** 2026-03-26
**Context:** Deep investigation into whether PCA rotation methods (varimax, promax, oblimin, Procrustes) can solve the axis instability documented in `docs/pca-ideology-axis-instability.md`, or whether human-in-the-loop intervention is the durable solution.

---

## The Problem Restated

In 7 of 14 Kansas Senate sessions (78th-83rd, 88th), PCA PC1 captures intra-Republican factionalism rather than the party divide. The pipeline's automated correction layers — seven quality gates (ADR-0118), tiered convergence (ADR-0110), and a W-NOMINATE cross-validation gate (ADR-0123) — catch most failures but still misroute 6/28 chamber-sessions. Each new gate is a patch on the previous gate's failure mode, and the architecture has become fragile.

The question: can PCA rotation methods provide a more durable fix, or should the pipeline embrace human intervention for the problematic sessions?

---

## Rotation Methods Evaluated

### Varimax (Orthogonal)

Varimax maximizes the variance of squared loadings *within each factor*, pushing each variable to load strongly on one factor and weakly on others ("simple structure"). It is the most common rotation in psychometrics and social science.

**Why it fails here:** Varimax is designed to *break up* a dominant general factor, distributing its variance across components. In the Kansas Senate, the general factor (total voting disagreement) is already split between intra-R factionalism and the party divide. Varimax would sharpen this split further, not resolve it. It would make the axis ordering problem *worse* by more cleanly separating the two cleavages into distinct, equally-weighted components.

### Quartimax (Orthogonal)

Quartimax maximizes the variance of squared loadings *within each variable* (row-wise rather than column-wise). It better preserves a dominant general factor — useful when theory says one dimension dominates.

**Why it fails here:** Quartimax preserves the dominant factor, but in supermajority sessions the dominant factor IS intra-R factionalism, not the party divide. Quartimax would anchor PC1 on the factional axis even more firmly.

### Promax and Oblimin (Oblique)

Oblique rotations allow factors to correlate, which is realistic for political data — economic and social conservatism are not independent. Promax first applies varimax then sharpens loadings via a power transformation before rotating obliquely. Oblimin minimizes cross-products of loadings directly.

**Why they fail here:** Oblique rotation produces a richer model (pattern matrix + structure matrix + factor correlation matrix), but it does not change which axis captures the most variance. The correlated-factor model would show that the factional and party dimensions are partially correlated (they are — moderate Republicans align with Democrats on some votes), but the fundamental ordering by explained variance is unchanged.

The factor correlation matrix from oblique rotation would be *diagnostically useful* — it quantifies the R-D overlap that drives the horseshoe effect. But it does not solve the routing problem.

### Procrustes / Target Rotation

Target rotation rotates a factor solution to best match a researcher-specified target matrix. This is the one method that *could* force Dim 1 to align with the party divide: define a target where legislators load on Factor 1 proportionally to their party affiliation, then Procrustes-rotate toward it.

**Why it's overkill here:** Target rotation is a mathematically sophisticated way of saying "I want Dim 1 to be the party dimension." The existing `detect_ideology_pc()` function in `analysis/init_strategy.py` achieves the same result by computing point-biserial correlation between each PC and party and selecting the best match. Procrustes adds implementation complexity and conceptual opacity without adding information.

Target rotation is valuable when aligning factor solutions across time (cross-temporal Procrustes), but for within-session dimension identification, the simpler party-correlation approach is equivalent and more transparent.

### Summary Table

| Rotation | Solves axis ordering? | Adds diagnostic value? | Complexity |
|----------|----------------------|----------------------|------------|
| Varimax | No — makes it worse | Minimal | Medium |
| Quartimax | No — preserves wrong dominant axis | Minimal | Medium |
| Promax/Oblimin | No — different model, same ordering | Yes (factor correlations) | Medium |
| Procrustes | Technically, but just imposes prior | Useful for cross-temporal | High |

---

## The Deeper Issue: Pearson Correlations on Binary Data

There is a genuine PCA improvement worth making, though it does not solve the axis-ordering problem.

### Difficulty Factors

PCA currently uses Pearson correlations on binary yea/nay data. The psychometric literature (Wirth & Edwards 2007, Pearson & Mundform 2010) is clear that Pearson correlations between binary variables (phi coefficients) create **spurious "difficulty factors"** when items have heterogeneous marginal distributions. In a supermajority chamber where most votes pass 70-30 or 80-20, items with similar pass rates cluster together regardless of substantive content.

This does not cause the axis-swap (intra-R factionalism is real signal, not artifact), but it degrades loading estimates and makes parallel analysis less reliable.

### Tetrachoric Correlations

The tetrachoric correlation estimates what the Pearson correlation between two underlying continuous preferences would be, given only the observed 2x2 binary table. It corrects for the attenuation and difficulty-factor bias inherent in phi coefficients.

The EGA pipeline (`analysis/ega/tetrachoric.py`) already computes tetrachoric correlations for its GLASSO network. Running PCA on the tetrachoric matrix alongside the standard Pearson PCA would produce cleaner loadings and reduce spurious difficulty factors, though the intra-R factional axis would still dominate in supermajority sessions because it IS the real dominant axis.

**Recommendation:** Add tetrachoric PCA as a diagnostic comparison in Phase 02. When the two PCA variants disagree on component ordering, that disagreement is itself diagnostic information.

---

## Why the Automated Gates Keep Failing

The pipeline currently has three layers of automated correction:

1. **Party-separation quality gates (ADR-0118, R1-R7):** Detect when PC1 doesn't separate parties and auto-swap to the best PC. Works for PCA init, but the party-d threshold (1.5) is arbitrary and session-dependent.

2. **Tiered convergence gate (ADR-0110):** Validates 2D IRT convergence and falls back to 1D when MCMC fails. Necessary but orthogonal to the axis-ordering problem.

3. **W-NOMINATE cross-validation gate (ADR-0123):** Correlates canonical IRT dimension with W-NOMINATE Dim 1 and auto-swaps when a different dimension correlates better by > 0.10.

The W-NOMINATE gate is the most effective — it catches all 6 misrouted sessions. But it has a structural flaw: **IRT defers to W-NOMINATE for dimension identification, making the Bayesian pipeline dependent on a frequentist method for a fundamental structural decision.** If W-NOMINATE itself fails in a future biennium (e.g., a competitive third party, or a session where both dimensions explain similar variance), the gate becomes a liability.

More fundamentally, each gate is a heuristic threshold. The party-d threshold of 1.5 was chosen because it works for the observed data. The W-NOMINATE delta of 0.10 was chosen because it catches the 6 known failures. These thresholds encode the specific structure of 2001-2026 Kansas politics. They have no theoretical basis that guarantees generalization.

### The Circular Dependency

The deepest problem is the circular dependency identified in the original instability analysis (R3): the pipeline validates its own outputs against its own inputs. Tier 2 originally checked PCA PC1 correlation — but when PC1 is wrong, the check rejects correct answers and accepts wrong ones. The W-NOMINATE gate breaks this specific circle by using an external reference, but it creates a new dependency.

---

## What the Literature Says

The political science literature on legislative scaling converges on a clear message:

> PCA extracts components in order of **variance explained**, not **substantive meaning**. When PC1 captures intra-party factionalism in a supermajority chamber, that is the statistically correct answer. The decision to treat the "party dimension" as the primary dimension is a **substantive/normative choice**, not a statistical correction.

Key references:

- **Poole & Rosenthal** establish that PC1 reliably captures the party divide in the modern US Congress because between-party variance dominates in a balanced, polarized chamber. That condition does not hold in the Kansas Senate.
- **Shin (2024, L1 ideal points)** warns explicitly: "practitioners should resist retrofitting ideological labels" when dimensions don't align with party.
- **de Leeuw (2018)** notes that PCA dimensions have no inherent substantive interpretation; labeling them "ideology" is an interpretive act that must be validated against external criteria.
- **Armstrong, Bakker, Carroll, Hare, Poole & Rosenthal (2014)** emphasize that the ordering of dimensions by variance is mathematical, not substantive — there is no guarantee that "most important" in a variance sense means "most politically meaningful."

The implication: no algorithm can reliably distinguish "PC1 = intra-party factionalism" from "PC1 = ideology" without substantive political knowledge. A party-d threshold is a proxy for that knowledge, but it is a brittle one.

---

## The Case for Human Intervention

### The Honest Architecture

The pipeline should distinguish between sessions where automation is reliable and sessions where it is not:

- **Unambiguous sessions (22/28):** λ₁/λ₂ > 2.0, PC1 party d > 2.0, no horseshoe detected. Automation runs untouched.
- **Ambiguous sessions (6/28):** λ₁/λ₂ < 2.0, party separation split across PCs, horseshoe effects. These need human review.

The key insight is that the **diagnostic information for human review already exists** — the pipeline computes Cohen's d, eigenvalue ratios, party-correlation metrics, and W-NOMINATE comparisons. What's missing is a structured way to surface these diagnostics and accept human overrides.

### The Override System

A manual override file (`analysis/pca_overrides.yaml`) would provide stable, auditable, human-vetted dimension assignments:

```yaml
# Manual PCA dimension overrides for sessions where automated detection is ambiguous.
# Each entry specifies which PC to use as the ideology axis and why.
# The pipeline reads this file during init strategy resolution and canonical routing.
# Sessions not listed here use automated detect_ideology_pc() logic.

79th_Senate:
  ideology_pc: PC2
  rationale: >
    PC1 captures moderate-vs-conservative Republican factionalism (d=0.28);
    PC2 is the party divide (d=4.98). λ₁/λ₂=1.45.
    79th Senate was the epicenter of the Kansas R civil war (Frank 2004).

80th_Senate:
  ideology_pc: PC2
  rationale: >
    PC1 d=0.27, PC2 d=2.56. Factional axis dominates.

81st_Senate:
  ideology_pc: PC2
  rationale: >
    PC1 d=1.72, PC2 d=2.25. Moderate R still more ideologically diverse than D caucus.

82nd_Senate:
  ideology_pc: PC2
  rationale: >
    PC1 d=0.89, PC2 d=2.41. Strong factional structure pre-purge.

83rd_Senate:
  ideology_pc: PC2
  rationale: >
    PC1 d=0.79, PC2 d=4.40. Last session before 2012 primary purge.

84th_Senate:
  ideology_pc: PC2
  rationale: >
    PC1 d=1.84, PC2 d=1.30. Transitional session — purge in progress.
    Neither PC cleanly separates parties. W-NOMINATE validates PC2 (r=0.992 vs 0.392).

88th_Senate:
  ideology_pc: PC2
  rationale: >
    PC1 d=1.72, PC2 d=3.10. Late-era exception — post-Brownback moderate resurgence.
```

### What This Changes

1. **PCA init** (`init_strategy.py`): Before calling `detect_ideology_pc()`, check the override file. If an override exists for this session+chamber, use it.

2. **Canonical routing** (`canonical_ideal_points.py`): When the override specifies PC2, the routing logic knows that H2D Dim 2 (or 1D IRT, depending on the session) is the ideology axis without relying on the W-NOMINATE gate.

3. **Diagnostic dashboard**: For flagged sessions, the PCA report includes an annotated biplot, per-component party profile, and override status, making the human decision transparent and auditable.

4. **W-NOMINATE gate**: Demoted from an automatic routing override to a diagnostic metric. The correlation is still computed and reported, but it no longer triggers auto-swaps. Sessions where W-NOMINATE disagrees with the selected canonical dimension are flagged for human review.

### Maintenance Cost

The override file changes only when new bienniums are added — once every two years. For the vast majority of future sessions, automated detection will work because the modern Kansas Legislature is highly polarized (PC1 d > 5.0). The overrides are exclusively for the pre-2013 factional era and the occasional post-2013 exception (88th).

---

## Recommendations

### Do Now

1. **Create `analysis/pca_overrides.yaml`** with overrides for the 7 known problematic sessions.
2. **Wire override loading** into `init_strategy.py` and `canonical_ideal_points.py`.
3. **Demote W-NOMINATE gate** from auto-routing to diagnostic-only.
4. **Enhance PCA report** with an axis-ambiguity diagnostic section for flagged sessions.

### Do Later (Optional)

5. **Add tetrachoric PCA** as a diagnostic comparison in Phase 02, using the existing `analysis/ega/tetrachoric.py` implementation.
6. **Add oblique rotation** as a diagnostic (factor correlation matrix reveals the R-D overlap structure).
7. **Cross-temporal Procrustes** for common-space alignment (separate from the axis-ordering problem).

### Don't Do

- Varimax or quartimax rotation — they don't help and add complexity.
- More automated quality gates — the gate architecture has reached diminishing returns.
- Backward-compatibility shims for the W-NOMINATE gate — remove the auto-routing cleanly.

---

## References

- Armstrong, D. A., Bakker, R., Carroll, R., Hare, C., Poole, K. T., & Rosenthal, H. (2014). *Analyzing Spatial Models of Choice and Judgment*. Routledge.
- Clinton, J., Jackman, S., & Rivers, D. (2004). The Statistical Analysis of Roll Call Data. *American Political Science Review*, 98(2), 355-370.
- de Leeuw, J. (2018). Computing and Using Principal Components. In *Encyclopedia of Statistics in Behavioral Science*.
- Diaconis, P., Goel, S., & Holmes, S. (2008). Horseshoes in multidimensional scaling and local kernel methods. *Annals of Applied Statistics*, 2(3), 777-807.
- Gill, J. (2018). Estimating Ideal Points from Roll-Call Data: Explore PCA. *Social Sciences*, 7(1), 12.
- Pearson, R. H., & Mundform, D. J. (2010). Recommended Sample Size for Conducting Exploratory Factor Analysis on Dichotomous Data. *Journal of Modern Applied Statistical Methods*, 9(2), 359-368.
- Poole, K. T., & Rosenthal, H. (1997). *Congress: A Political-Economic History of Roll Call Voting*. Oxford University Press.
- Shin, S. (2024). L1-based Bayesian Ideal Point Model. Working paper.
- Shor, B., & McCarty, N. (2011). The Ideological Mapping of American Legislatures. *American Political Science Review*, 105(3), 530-551.
- Wirth, R. J., & Edwards, M. C. (2007). Item factor analysis: Current approaches and future directions. *Psychological Methods*, 12(1), 58-79.
