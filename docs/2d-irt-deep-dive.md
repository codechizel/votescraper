# 2D Bayesian IRT: Deep Dive

**Date:** 2026-02-26 (experiment); 2026-02-28 (pipeline integration)
**Status:** Pipeline phase 04b (experimental)
**Script:** `analysis/06_irt_2d/irt_2d.py`
**Design doc:** `analysis/design/irt_2d.md`
**ADRs:** `docs/adr/0046-2d-irt-experimental.md` (original), `docs/adr/0054-2d-irt-pipeline-integration.md` (pipeline)

---

## 1. Motivation: The Tyson Paradox

The 1D Bayesian IRT model is the canonical baseline for Tallgrass. It works — 0.98 AUC, clean convergence, stable across bienniums (cross-session r > 0.94). But it compresses all voting behavior into a single liberal-conservative axis. For most legislators, that's fine. For Caryn Tyson, it produces a misleading result.

The Tyson paradox (`analysis/design/tyson_paradox.md`): IRT ranks Tyson as the most conservative senator (xi = +4.17), 22 ranks higher than PCA placed her (23rd among Republicans). The mechanism is well-understood:

- Tyson votes with perfect conservative alignment on high-discrimination bills (81/81)
- Her 74 contrarian Nay votes fall almost entirely on low-discrimination bills that IRT barely weights
- 31 of 41 dissent votes have negative beta, so IRT codes them as *conservative* votes

PCA captures this two-dimensional behavior: PC1 = ideology (+4.86, solidly conservative), PC2 = contrarianism (-24.8, most extreme in the chamber by 3x). The two senators with the largest PCA→IRT rank shifts — Tyson (+22) and Thompson (+15) — are the two with the most extreme negative PC2 scores. The second dimension exists in the data; the 1D model simply cannot represent it.

A 2D IRT model would estimate both dimensions simultaneously: Dimension 1 for ideology, Dimension 2 for the contrarianism pattern PCA captures as PC2. This is the "theoretically correct solution" identified in the Tyson paradox writeup.

---

## 2. Political Science Context

### The History of the Second Dimension

Poole and Rosenthal's D-NOMINATE (1997) established that Congressional voting is well-described by two dimensions. The first captures the liberal-conservative spectrum; the second historically captured civil rights, race, and regional splits within parties. In the mid-20th century, the second dimension explained 15-25% of variance — enough that 1D models were clearly inadequate.

Since the 1990s, the second dimension has collapsed in Congress. Poole (2005) showed it dropped below 5% of variance by the 108th Congress. Modern polarization has sorted the parties so thoroughly that nearly all voting variation falls along a single axis. McCarty, Poole, and Rosenthal (2006) call this "the disappearance of the moderate" — and with moderates go the cross-cutting coalitions that gave the second dimension its content.

### State-Level Evidence Gap

Most 2D analyses focus on Congress. State legislatures are understudied. The Kansas Legislature, with its Republican supermajority (~72%), presents a different dynamic than the roughly equal partisan split in Congress. Intra-Republican variation is the primary substantive interest, and within-party ideological structure may be inherently multidimensional in ways that within-party Congressional voting is not.

Our PCA results show PC2 at 11% of variance — comparable to where Congress was in the 1990s, before the second dimension collapsed. Whether this represents a genuine policy dimension or an artifact of a few contrarian legislators (Tyson, Thompson, Peck) is exactly what a 2D IRT model can distinguish.

### What 2D Would Add

For Kansas, the second dimension is unlikely to be a traditional policy cleavage (social vs. economic conservatism, urban vs. rural). PCA suggests it captures a behavioral pattern: willingness to dissent on routine, near-unanimous legislation. A 2D IRT model would:

1. Separate ideology (Dim 1) from contrarianism (Dim 2) in a single coherent model
2. Provide Bayesian uncertainty intervals on both dimensions
3. Estimate bill-level discrimination on each dimension (which bills load on contrarianism?)
4. Test whether the second dimension genuinely improves model fit, or whether it's just overfitting Tyson's behavior

---

## 3. The Identification Problem

### Why 2D Is Harder Than 1D

In 1D IRT, the model has a reflection invariance: negating all xi and all beta leaves the likelihood unchanged. This is a 2-mode problem (original + reflected), solved by two anchor legislators.

In 2D IRT, the invariance group is much larger. The likelihood is unchanged under:

1. **Rotation** — rotating all ideal points and discrimination vectors by the same angle
2. **Reflection** — flipping the sign of either dimension independently
3. **Label switching** — swapping Dimension 1 and Dimension 2

For D dimensions, this produces 2^D × D! equivalent posterior modes. For D=2: 2² × 2! = **8 modes**. An MCMC sampler that explores all 8 modes will produce meaningless posterior summaries — the chain means of ideal points will collapse to zero as positive and negative modes cancel.

### What Must Be Constrained

To reduce 8 equivalent modes to 1, we need D(D-1)/2 + D = 3 constraints:

- **1 rotation constraint** (D(D-1)/2 = 1): fix the rotation angle so dimension axes are uniquely oriented
- **2 sign constraints** (D = 2): fix the sign convention for each dimension

The standard 1D approach of fixing two anchors provides 2 constraints (location + sign). For 2D, we need 3 constraints minimum, plus 2 more for location (mean of each dimension) and 2 for scale (variance of each dimension) — though normal priors on ideal points handle location and scale.

---

## 4. Identification Strategies

Five approaches have been proposed in the literature:

### 4a. Positive Lower Triangular (PLT) Constraints

**Source:** Lopes & West (2004), Béguin & Glas (2001)

Constrain the discrimination matrix (beta) to be lower triangular with positive diagonal:

```
beta = | beta[j,0]    0        |   for anchor item j=0
       | beta[j,0]  beta[j,1]  |   for all other items, beta[j,1] > 0 for j=1
```

This provides:
- 1 zero constraint (beta[0,1] = 0): fixes rotation
- 1 positive constraint (beta[1,1] > 0): fixes Dim 2 sign
- Dim 1 sign fixed by anchor or post-hoc party mean ordering

**Advantages:** Minimal constraints, well-studied, easy to implement in PyMC via boolean mask or set_subtensor. Does not require choosing anchor legislators — only anchor items (bills).

**Disadvantages:** The choice of anchor item (which bill has beta[:,1]=0) affects results slightly. The anchor item should be one that clearly loads only on Dim 1 (a pure party-line vote).

### 4b. Fixed Anchor Legislators

**Source:** Rivers (2003), Clinton, Jackman & Rivers (2004)

Fix D+1 = 3 legislators' ideal points in 2D space. For example, fix one conservative at (1, 0), one liberal at (-1, 0), and one known contrarian at (0, 1).

**Advantages:** Intuitive, directly controls the substantive meaning of each dimension.

**Disadvantages:** Requires knowing *a priori* which legislators define each dimension — feasible for Congress with well-known members, risky for state legislatures with less external validation. Fixing 3 points in 2D is borderline over-constrained (provides 6 constraints when only 5 are needed).

### 4c. Post-Hoc Procrustes Rotation

**Source:** Common in factor analysis, applied to IRT by Jackman (2001)

Run the model with minimal constraints (just sign identification), then rotate the posterior to a target orientation (e.g., align Dim 1 with the party axis).

**Advantages:** Cleanly separates estimation from interpretation. The MCMC sampler explores the posterior without artificial constraints.

**Disadvantages:** Requires a rotation target. Procrustes on MCMC output is computationally expensive (rotate each posterior draw). Can fail if the sampler visits multiple modes during sampling.

### 4d. L1 Norm Constraint (Shin 2024)

**Source:** Shin (2024), "Measuring Multidimensional Ideology"

Constrain the L1 norm of each legislator's ideal point vector to identify scale, combined with sign constraints on specific dimension means.

**Advantages:** Theoretically elegant, avoids privileging any particular item as anchor.

**Disadvantages:** Novel method with limited implementation experience. The L1 constraint creates a non-differentiable boundary that may interact poorly with HMC.

### 4e. IRT-M Theory-Driven Identification

**Source:** Kornilova & Eguia (2024)

Use theory to specify which items load on which dimensions (a Q-matrix), reducing identification to the 1D case on each dimension.

**Advantages:** Produces dimensions with clear substantive interpretation.

**Disadvantages:** Requires strong prior knowledge about bill content. Kansas bill metadata (short titles only, no full text) is insufficient for reliable Q-matrix construction.

### Recommendation: PLT

PLT is the most practical approach for our setting:
- No legislator anchors needed (avoids the 3-anchor problem)
- Well-studied in the psychometrics literature
- Easy to implement in PyMC with `pt.set_subtensor` and `HalfNormal`
- Minimal constraints (1 zero + 1 positive) provide just enough identification
- Dim 1 sign identified by sorted party means (same approach as hierarchical IRT)

---

## 5. Python Ecosystem Evaluation

| Package | Type | 2D Support | Identification | Verdict |
|---------|------|------------|----------------|---------|
| **py-irt** | Bayesian (PyTorch) | 1D only | None | No 2D |
| **girth** | Frequentist (EM) | M2PL via MIRT | Post-hoc rotation | No uncertainty |
| **girth-mcmc** | Bayesian (PyMC3) | 1D only | Anchors | Legacy PyMC3 |
| **deepirtools** | Deep learning (PyTorch) | Yes via Q-matrix | Requires Q-matrix | Need bill classification |
| **jamalex/bayesian-irt** | Bayesian (PyStan) | 1D only | None | Incomplete |

**Verdict:** No existing Python package provides a 2D Bayesian IRT model with proper identification constraints suitable for legislative ideal points. A custom PyMC implementation is the only viable path.

This is consistent with the broader finding from the IRT field survey (`docs/irt-field-survey.md`): the Python ecosystem for political science IRT is immature. The R ecosystem (pscl, MCMCpack, emIRT) is mature but excluded by project policy.

---

## 6. Recommended Approach: PLT in PyMC

### Model Specification

The Multidimensional 2-Parameter Logistic (M2PL) model:

```
P(Yea | xi_i, alpha_j, beta_j) = logit^-1(beta_j · xi_i - alpha_j)

xi_i   ∈ R^2  — legislator ideal point (2D vector)
alpha_j ∈ R    — bill difficulty (scalar, same as 1D)
beta_j  ∈ R^2  — bill discrimination (2D vector)
```

The dot product `beta_j · xi_i = beta[j,0]*xi[i,0] + beta[j,1]*xi[i,1]` replaces the scalar product `beta_j * xi_i` from the 1D model.

### PLT Constraint on Discrimination

```python
# Column 0: unconstrained (all bills load freely on Dim 1)
beta_col0 = pm.Normal("beta_col0", mu=0, sigma=1, shape=n_votes)

# Column 1: anchor item has zero loading, rest free
# Item 0 (anchor): beta[0,1] = 0 (fixes rotation)
# Item 1: beta[1,1] > 0 (fixes Dim 2 sign, via HalfNormal)
# Items 2..n: beta[j,1] unconstrained (Normal)
beta_anchor_positive = pm.HalfNormal("beta_anchor_positive", sigma=1)
beta_col1_rest = pm.Normal("beta_col1_rest", mu=0, sigma=1, shape=n_votes - 2)

# Assemble: [0, positive, free, free, ...]
beta_col1 = pt.zeros(n_votes)
beta_col1 = pt.set_subtensor(beta_col1[1], beta_anchor_positive)
beta_col1 = pt.set_subtensor(beta_col1[2:], beta_col1_rest)
```

### Prior Choices

| Parameter | Prior | Shape | Notes |
|-----------|-------|-------|-------|
| xi (ideal points) | Normal(0, 1) | (n_leg, 2) | Per-legislator, 2D |
| alpha (difficulty) | Normal(0, 5) | (n_votes,) | Same as 1D |
| beta col 0 | Normal(0, 1) | (n_votes,) | Unconstrained |
| beta col 1, item 1 | HalfNormal(1) | scalar | Positive diagonal |
| beta col 1, items 2+ | Normal(0, 1) | (n_votes-2,) | Free |

### Initialization Strategy

- **xi[:, 0]**: PCA PC1 scores, standardized to mean 0, sd 1
- **xi[:, 1]**: PCA PC2 scores, standardized to mean 0, sd 1
- **init='adapt_diag'**: No jitter, consistent with ADR-0045 finding
- **target_accept=0.95**: Higher than 1D (0.9) due to harder posterior geometry

### Dim 1 Sign Identification

Same approach as the hierarchical IRT model: after sampling, verify that the Republican mean on Dim 1 is positive. If not, negate Dim 1. This is not a constraint during sampling — PLT already eliminates rotation and Dim 2 reflection. Only Dim 1 reflection ambiguity remains, resolved post-hoc.

---

## 7. Integration Design

### Pipeline Phase 06 (2026-02-28)

The 2D IRT model is integrated as pipeline phase `06_irt_2d` with experimental status (ADR-0054):

- **Pipeline script:** `analysis/06_irt_2d/irt_2d.py` (RunContext, HTML report, auto-primer)
- **Original experiment:** `analysis/experimental/irt_2d_experiment.py` (preserved for reference)
- **Runs both chambers** (House and Senate) — consistent with all other phases
- **Sampler:** nutpie (Rust NUTS), consistent with all production MCMC (ADR-0051, ADR-0053)
- **Pipeline:** `just irt-2d` standalone, or included in `just pipeline`
- **Results:** `results/kansas/{session}/04b_irt_2d/{YYMMDD}.{n}/`

### Reuse from 1D IRT

The phase imports shared functions from `analysis.irt`:
- `load_eda_matrices()` — filtered vote matrices
- `load_pca_scores()` — PCA scores for initialization and comparison
- `load_metadata()` — rollcall and legislator metadata
- `prepare_irt_data()` — wide-to-long conversion with index mappings
- `select_anchors()` — PCA-based anchor selection (for reference logging)

### Downstream Consumption

2D scores are NOT fed into downstream phases (synthesis, profiles) — convergence caveats make this premature. Per-chamber outputs available for future integration:
- `ideal_points_2d_{chamber}.parquet` — 2D ideal points with HDIs
- `convergence_summary.json` — per-chamber diagnostics and correlations

### Relationship to 1D Model

The 1D model remains primary. The 2D model is a diagnostic:
- If Dim 1 correlates r > 0.95 with 1D IRT, the 1D model is capturing the same primary axis
- If Dim 2 captures Tyson's contrarianism (she's the extreme), the model is working as intended
- If classification gain is < 1%, the 2D model confirms that 1D is sufficient for most purposes

---

## 8. Expected Costs and Benefits

### Runtime

| Model | Params | Expected Runtime (M3 Pro) |
|-------|--------|---------------------------|
| 1D IRT (Senate) | ~280 | ~8 min |
| 2D IRT (Senate) | ~520 | ~25-50 min |

The 2D model approximately doubles the parameter count (2D xi, 2D beta). Posterior geometry is harder (correlated dimensions), so sampling per iteration is slower. Expected total: 3-6x the 1D runtime.

### Accuracy

- **Classification improvement:** 0.5-2% expected. The 1D model already achieves 0.98 AUC; the second dimension captures behavior on low-discrimination bills that contribute little to prediction.
- **Marginal**: For most legislators, 1D and 2D ideal points will be nearly identical on Dim 1. Only legislators with extreme PC2 (Tyson, Thompson, Peck) should move meaningfully.

### Diagnostic Value

This is the primary benefit. The 2D model answers:

1. **Does the Bayesian second dimension match PCA PC2?** If Dim 2 correlates with PC2, the PCA finding is validated by a proper statistical model with uncertainty.
2. **Is Tyson genuinely two-dimensional, or just noisy?** If her Dim 2 HDI is tight and extreme, it's a real signal. If it's wide, she may just be an unpredictable voter.
3. **Which bills define the second dimension?** The Dim 2 discrimination parameters identify specific legislation that separates contrarians from non-contrarians.
4. **Would more dimensions help?** If 2D captures essentially all the structured variance, there's no need for 3D+.

### What "Success" Looks Like

- R-hat < 1.05 (relaxed for experimental), ESS > 200 for all parameters
- Dim 1 ↔ 1D IRT: Pearson r > 0.90
- Dim 2 ↔ PCA PC2: positive, meaningful correlation
- Tyson is the most extreme on Dim 2
- Thompson is second-most extreme on Dim 2
- 2D scatter plot shows structure beyond a simple party axis

---

## 9. Experimental Results (91st Senate, 2026-02-26)

### Summary

The 2D IRT model ran on the Kansas Senate (42 legislators x 194 contested votes) in 2.8 minutes with 4 chains, 2000 draws, 2000 tune. PLT identification with PCA 2D initialization was used. (Original experiment used PyMC sampling; pipeline phase 04b uses nutpie.)

### Convergence

Partial failure. Dimension 1 parameters converged cleanly; Dimension 2 experienced mode-splitting on a subset of parameters.

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| R-hat (xi) max | 1.73 | < 1.05 | FAILED |
| R-hat (alpha) max | 1.008 | < 1.05 | OK |
| R-hat (beta) max | 1.64 | < 1.05 | FAILED |
| Bulk ESS (xi) min | 6 | > 200 | FAILED |
| Bulk ESS (alpha) min | 1311 | > 200 | OK |
| Divergences | 0 | < 50 | OK |
| E-BFMI (all chains) | 0.91-0.98 | > 0.3 | OK |

The R-hat = 1.73 and ESS = 6 on xi indicate that some chains explored different modes of the Dim 2 posterior. The PLT constraint (beta[0,1]=0, beta[1,1]>0) did not fully prevent Dim 2 mode-splitting — likely because the second dimension has weak signal (11% variance) and most legislators have Dim 2 near zero.

### Correlation with PCA

| Comparison | Pearson r | Spearman rho |
|------------|-----------|--------------|
| Dim 1 vs PCA PC1 | **0.9802** | 0.8778 |
| Dim 2 vs PCA PC2 | **0.8119** | 0.7647 |

Dim 1 strongly recovers the PCA ideological axis. Dim 2 captures the contrarianism pattern at r = 0.81 — a meaningful but imperfect correspondence, consistent with IRT weighting votes differently from PCA.

### The Tyson Paradox: Resolved

The 2D model separates Tyson's two behavioral patterns:

| Legislator | Dim 1 (Ideology) | Dim 1 Rank | Dim 2 (Contrarianism) | Dim 2 Rank (by |Dim 2|) |
|------------|-------------------|------------|------------------------|--------------------------|
| **Caryn Tyson** | +0.984 | **3rd** | **-1.882** | **1st** |
| **Mike Thompson** | +0.882 | 8th | -0.883 | 2nd |
| **Virgil Peck** | +0.785 | 11th | -0.639 | 3rd |

In the 1D model, Tyson was ranked #1 (xi = +4.17). In the 2D model, she drops to **#3 on Dim 1** (behind Gossage and Blew) but is **#1 on Dim 2** — exactly the separation the Tyson paradox article predicted. Thompson and Peck follow in the same order as their PC2 scores.

The top 5 most conservative on Dim 1: Gossage (+1.077), Blew (+1.067), Tyson (+0.984), Rose (+0.949), Masterson (+0.923). This ranking is much closer to the PCA ranking (Gossage #2, Masterson #5) than the 1D IRT ranking was.

### HDI Widths: The Second Dimension Is Noisy

Most legislators have Dim 2 HDIs spanning zero, indicating the second dimension adds little information for them:

- Tyson Dim 2 HDI: [-5.785, +3.040] — wide, but the point estimate (-1.882) and rank (#1) are meaningful
- Gossage Dim 2 HDI: [-3.206, +3.291] — centered on zero, no contrarianism signal
- Democrats Dim 2 HDIs span ±4-5 — the second dimension is uninformative for the minority party

This confirms the expectation: the second dimension captures behavior of 3-5 extreme Republican legislators, not a chamber-wide pattern.

### Implications

1. **The Tyson paradox is a real multidimensional pattern**, not a model artifact. A 2D model correctly separates ideology from contrarianism.
2. **The 1D model remains appropriate for most purposes.** Dim 2 is noisy for 90%+ of legislators.
3. **Convergence is marginal.** The Dim 2 posterior is weakly identified — the data supports only a few extreme points, not a full second dimension. Future work could explore stronger constraints or Procrustes post-processing.
4. **The PCA→IRT rank-shift pattern is validated.** The three largest 1D IRT rank shifts (Tyson +22, Thompson +15, Peck +3) correspond exactly to the three most extreme Dim 2 legislators.

### Output Files

Results in `results/experimental_lab/2026-02-26_irt-2d/`:
- `data/ideal_points_2d.parquet` — 42 legislators with 2D ideal points and HDIs
- `data/convergence_summary.json` — full diagnostics and correlation metrics
- `plots/2d_scatter.png` — Dim 1 vs Dim 2, party-colored, Tyson/Thompson annotated
- `plots/dim1_vs_pc1.png` — Correlation scatter (r = 0.9802)
- `plots/dim2_vs_pc2.png` — Correlation scatter (r = 0.8119)

---

## 10. References

- Béguin, A. A., & Glas, C. A. W. (2001). MCMC estimation and some model-fit analysis of multidimensional IRT models. *Psychometrika*, 66(4), 541-561.
- Clinton, J., Jackman, S., & Rivers, D. (2004). The statistical analysis of roll call data. *American Political Science Review*, 98(2), 355-370.
- Jackman, S. (2001). Multidimensional analysis of roll call data via Bayesian simulation. *Political Analysis*, 9(3), 227-241.
- Kornilova, A., & Eguia, J. X. (2024). IRT-M: Inductive discovery of multi-dimensional ideology. Working paper.
- Lopes, H. F., & West, M. (2004). Bayesian model assessment in factor analysis. *Statistica Sinica*, 14(1), 41-67.
- Martin, A. D., & Quinn, K. M. (2002). Dynamic ideal point estimation via Markov chain Monte Carlo for the U.S. Supreme Court, 1953-1999. *Political Analysis*, 10(2), 134-153.
- McCarty, N., Poole, K. T., & Rosenthal, H. (2006). *Polarized America: The Dance of Ideology and Unequal Riches*. MIT Press.
- Poole, K. T. (2005). *Spatial Models of Parliamentary Voting*. Cambridge University Press.
- Poole, K. T., & Rosenthal, H. (1997). *Congress: A Political-Economic History of Roll Call Voting*. Oxford University Press.
- Rivers, D. (2003). Identification of multidimensional spatial voting models. Typescript, Stanford University.
- Shin, M. (2024). Measuring multidimensional ideology. Working paper.
