# Factoring Out Establishment: Fisher's LDA for Ideology Extraction

**Date:** 2026-03-26

---

## The Problem

PCA on roll-call voting data extracts components in order of variance explained. In a supermajority chamber, the largest source of variance can be intra-party factionalism rather than the party divide, causing PC1 to capture the "establishment vs. rebel" axis instead of ideology. This is the documented axis instability in 4 of 14 Kansas Senate sessions (79th-80th, 82nd-83rd; see `docs/pca-dimension-audit-2026.md`).

The pipeline currently manages this with manual overrides (`analysis/pca_overrides.yaml`) that tell downstream phases which PC to treat as ideology. This works, but raises a natural question: can we use the party labels we already have to automatically find the ideology dimension, factoring out establishment and other non-partisan axes?

The answer is yes, with caveats. This document evaluates Fisher's Linear Discriminant Analysis (LDA) as a party-supervised projection, presents empirical results across all 28 chamber-sessions, and assesses whether it should replace, complement, or remain separate from the existing PCA pipeline.

---

## What LDA Does

Fisher's Linear Discriminant Analysis finds the linear combination of features that maximizes the ratio of between-group variance to within-group variance. For two groups (Republican and Democrat), it reduces to a single discriminant direction:

```
w = S_w^(-1) (mu_R - mu_D)
```

where `S_w` is the pooled within-group covariance matrix, and `mu_R`, `mu_D` are the party centroids in PCA space. Projecting each legislator onto this direction gives a continuous "ideology" score.

The key distinction from simply projecting onto the line between party centroids: the `S_w^(-1)` term warps the space so that dimensions with large within-party variance (like the moderate-vs-conservative Republican axis) get **downweighted**, while dimensions with small within-party variance (like a clean party split on education funding) get **amplified**. This is precisely the mathematical operation needed to factor out establishment.

Applied after PCA, the workflow is:

1. Run standard PCA to get PC1...PC5 scores
2. Compute LDA on those scores using party as the grouping variable
3. The LDA projection is the "ideology" score
4. The orthogonal complement is the "establishment" score (free)

---

## Empirical Results

### LDA vs. Best Single PC (Senate, All 14 Bienniums)

| Session | Best PC | Best PC d | LDA d | Improvement | LDA Weights |
|---------|---------|-----------|-------|-------------|-------------|
| 78th (1999-2000) | PC1 | 6.39 | 15.76 | +147% | PC1 37%, PC3 28%, PC2 18% |
| **79th (2001-2002)** | **PC2** | **4.07** | **9.85** | **+142%** | **PC2 43%, PC4 26%, PC1 20%** |
| **80th (2003-2004)** | **PC2** | **3.96** | **~9** | **+127%** | **PC2 dominant** |
| 81st (2005-2006) | PC1 | 2.44 | ~6 | +146% | Mixed |
| **82nd (2007-2008)** | **PC2** | **2.41** | **8.51** | **+253%** | **PC3 43%, PC2 29%** |
| **83rd (2009-2010)** | **PC2** | **4.53** | **9.12** | **+101%** | **PC2 53%, PC1 19%, PC3 18%** |
| 84th (2011-2012) | PC1 | 1.89 | 5.03 | **+166%** | PC4 35%, PC2 22%, PC3 22% |
| 85th (2013-2014) | PC1 | 6.74 | 13.23 | +96% | PC3 53%, PC1 27% |
| 86th (2015-2016) | PC1 | 5.79 | 13.32 | +130% | PC3 54%, PC1 23% |
| 87th (2017-2018) | PC1 | 2.47 | 4.49 | +82% | PC2 36%, PC1 25%, PC3 22% |
| 88th (2019-2020) | PC1 | 4.87 | ~11 | +126% | Mixed |
| 89th (2021-2022) | PC1 | 8.46 | ~12 | +42% | PC1 dominant |
| 90th (2023-2024) | PC1 | 7.44 | 15.09 | +103% | PC1 40%, PC3 43% |
| 91st (2025-2026) | PC1 | 10.00 | 12.03 | +20% | PC1 53%, PC3 19% |

LDA improves party separation in every single session. The improvement ranges from +20% (91st, where PC1 already dominates) to +253% (82nd, where ideology is spread across PC2 and PC3). Leave-one-out cross-validated accuracy is 97.3-100% across all sessions, indicating the improvement is not overfitting.

### The 84th Senate: Solving the Unsolvable

The 84th (2011-2012) was the one session we labeled "ambiguous" — neither PC1 (d=1.89) nor PC2 (d=1.61) cleanly separated parties. This is the transitional session where the 2012 primary purge happened mid-biennium, creating a voting record that mixes two different political configurations.

LDA finds ideology spread across PC2 (22%), PC3 (22%), and PC4 (35%), with PC1 contributing only 12%. No single PC captures the party divide because it's distributed across the lower-variance components. LDA recovers a d=5.03 score — strong, clean party separation that neither PC alone could achieve.

### Hidden Signal in PC3+

A surprising finding: even in "clean" sessions (85th-91st), PCs 3-5 contain substantial party-relevant signal. In the 90th Senate (PC1 d=7.44), PC3 carries **43%** of the LDA weight, boosting the score to d=15.09. This signal is invisible to the current pipeline, which only considers PC1 and PC2 for ideology.

Using all 5 PCs adds a mean of +2.71 Cohen's d beyond using just PC1+PC2. The improvement is especially large in Senate sessions where ideology is fragmented across multiple low-variance dimensions.

### The Establishment Dimension (Free)

The orthogonal complement of the LDA ideology direction captures intra-party factionalism — the "establishment vs. rebel" axis. Within-Republican quartile d on this axis averages 6.75, confirming it captures real factional structure.

Legislators at the extremes tell a consistent story across sessions:

| Pole | Recurring Names |
|------|----------------|
| Anti-establishment | Tim Huelskamp, Dennis Pyle, Caryn Tyson, Mary Pilcher Cook, Trevor Jacobs, Randy Garber |
| Establishment | John Vratil, Pete Brungardt, Vicki Schmidt, Jean Schodorf, Brenda Dietrich, Mark Schreiber |

This is the axis that the 79th-83rd Senate PCA was capturing on PC1 — it's genuine political signal, not noise. LDA doesn't destroy it; it separates it from the ideology axis so both can be examined independently.

---

## The Circularity Problem

The strongest argument against LDA is circularity: if we use party labels to find the dimension that best separates parties, we've built party into the ideology score by construction. This concern has three layers, each with a different answer.

### Layer 1: Does LDA collapse to binary party membership?

**No.** LDA produces continuous scores. Legislators with voting patterns closer to the opposing party's centroid receive intermediate scores. A moderate Republican who votes with Democrats on 40% of contested bills will project between the two party centroids, not at the Republican centroid. The scores preserve within-party ordering — LDA guarantees the *axis* separates parties, not that individual legislators land on the "correct" side.

Empirically, cross-party overlap occurs in the Kansas data: John Doll (R, 90th Senate) projects at -7.47 on PC1, well into Democrat territory. LDA would preserve this.

### Layer 2: Does LDA guarantee the party dimension is primary?

**Yes, and this is the real concern.** PCA can discover that the factional axis dominates the party axis — that intra-Republican variance exceeds inter-party variance. This is a genuine finding about Kansas politics. LDA cannot make this discovery, because its objective function *defines* the party axis as primary. Aldrich, Montgomery & Sparks (2014) show that even unsupervised methods struggle with this distinction in highly polarized chambers; supervised methods cannot make it at all.

### Layer 3: Is "party" the same as "ideology"?

**Not necessarily.** Poole & Rosenthal have consistently argued that the first NOMINATE dimension captures "liberal-conservative ideology," which *correlates* with party but is not defined by it. In the modern U.S. Congress, the correlation exceeds 0.95, making the distinction academic. In state legislatures with weaker party discipline — like Kansas — the distinction is substantive.

The key question: does LDA find "party" or "ideology"? The honest answer is that it finds **the direction that best separates party labels**, which is ideology to the extent that ideology drives party-line voting, and party to the extent that other factors (leadership loyalty, committee assignments, campaign finance) also drive party-line voting.

For the Kansas pipeline's purpose — initializing IRT models and routing canonical ideal points — this distinction is acceptable. The IRT model's likelihood will override the initialization for legislators where the LDA-based starting point is wrong. But for **interpreting** the scores, users should know that LDA-derived ideology scores are party-oriented by construction.

---

## The Small-Sample Problem

Kansas Senate sessions typically have 29-32 Republicans and 8-12 Democrats, with 5 PCA features. This creates two statistical concerns.

### Unstable covariance estimation

The pooled within-class covariance `S_w` combines contributions from both groups, but the Democrat contribution is based on only 8-12 observations for a 5x5 matrix. The standard recommendation (Tabachnick & Fidell 2007) is at least 20 observations per group; the Democrat group fails this. Small eigenvalues of `S_w` will be poorly estimated, and `S_w^(-1)` amplifies exactly the directions where estimation is worst.

**Mitigation: Shrinkage LDA.** Ledoit & Wolf (2004) provide an optimal shrinkage estimator: `S_shrunk = alpha * I + (1-alpha) * S_sample`, where `alpha` is determined analytically. This stabilizes the inverse by pulling the covariance toward spherical. scikit-learn implements this as `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')`.

### Overfitting to idiosyncratic Democrats

With 5 features and 10 Democrats, LDA has enough degrees of freedom to achieve near-perfect separation even when the true separation is modest. The 97-100% LOO accuracy is reassuring but not conclusive — LOO can overestimate accuracy when the separation is large relative to noise.

**Mitigation: Reduce dimensionality.** Using only PC1+PC2 (p=2) drops the LDA improvement somewhat but dramatically improves stability. The tradeoff: PC3+ contains real party signal (especially in the 85th-90th), but estimating its contribution from 10 Democrats is unreliable. For sessions with fewer than 10 Democrats, restricting to PC1+PC2 is the safer choice.

### Empirical assessment

Despite these concerns, the empirical results are strong. LOO accuracy never drops below 97.3% (84th Senate, the hardest case). The LDA weights are substantively interpretable — they emphasize PCs where party signal exists and downweight PCs where it doesn't. The scores produce sensible legislator orderings (verified against known voting patterns). The small-sample risk is real but does not appear to dominate in practice for this dataset.

---

## The Contested Threshold Connection

A key insight: the pipeline's `CONTESTED_THRESHOLD` is already performing a crude form of supervised dimension reduction.

By removing votes where fewer than 10% of members dissented, the contested threshold eliminates votes where party is not a meaningful predictor. These are the same votes that drive the factional axis — a 35-5 vote in a 40-member Senate typically features 5 conservative Republican defectors against a bipartisan majority. Removing them suppresses the factional signal and lets the party signal dominate.

This is conceptually identical to **Supervised PCA** (Bair, Hastie, Paul & Tibshirani 2006), which screens features by their association with a target variable before running PCA. The contested threshold screens votes by their party-association (indirectly) before PCA sees them.

The progression from crude to refined:

| Approach | How it selects the party axis | Granularity |
|----------|------------------------------|-------------|
| Low contested threshold (2.5%) | No selection — all votes enter PCA | None |
| High contested threshold (10%) | Remove lopsided votes (crude party filter) | Vote-level |
| `detect_ideology_pc()` | Pick the PC with strongest party correlation | Component-level |
| Manual PCA overrides | Human judgment on which PC is ideology | Component-level |
| Fisher's LDA | Optimal weighted combination of all PCs | Continuous |

Each step is more supervised than the last. The contested threshold is supervision at the vote level; `detect_ideology_pc()` is supervision at the component level; LDA is supervision at the projection level.

This means the pipeline is already on a supervised-unsupervised spectrum, not at one extreme. Adding LDA would move further toward the supervised end, but the philosophical leap is smaller than it appears.

---

## What LDA Cannot Do

### Discover when party isn't the dominant dimension

In the 79th Senate, the most important finding is that intra-Republican factionalism explains more voting variance than the party divide. PCA discovers this; LDA suppresses it. If the pipeline moved entirely to LDA, this finding — which is the most interesting thing about pre-2012 Kansas politics — would be invisible.

### Distinguish ideology from party discipline

A legislator who votes the party line because of genuine ideological conviction and one who votes the party line because of leadership pressure produce identical roll-call records. LDA scores them identically. PCA does too — but PCA doesn't *claim* to measure ideology. LDA's framing as an "ideology" score implies a substantive interpretation that the method cannot validate.

### Handle Independents

The Kansas data includes occasional Independents (John Doll in the 87th, Dennis Pyle in the 89th). LDA's two-class framework has no natural way to include them in the training set. They can be projected onto the LDA axis after fitting (and they land in sensible positions), but they don't contribute to defining the axis.

### Replace IRT

LDA is a linear projection of PCA scores. It has no model of bill difficulty, no discrimination parameters, no posterior uncertainty. IRT is still needed for everything beyond initialization and rough ordering. LDA is a preprocessing step, not an estimation method.

---

## Alternatives Considered

| Method | Relationship to LDA | Advantage | Disadvantage |
|--------|-------------------|-----------|--------------|
| **Party centroid projection** | LDA with S_w = I (no covariance correction) | Simplest possible. No matrix inversion. | Suboptimal when within-party covariance is non-spherical |
| **Canonical Correlation Analysis** | Mathematically identical to LDA for 2 groups | None — same result | Same concerns |
| **Partial Least Squares (PLS-DA)** | Similar direction with implicit regularization | Handles multicollinearity | No advantage at p=5 |
| **Sliced Inverse Regression** | Closely related to LDA for binary Y | Extends to non-linear relationships | Reduces to ~LDA for this case |
| **Supervised PCA** (Bair et al. 2006) | Screen votes by party association before PCA | Operates at vote level, avoids small-sample covariance problem | Coarser than LDA |
| **Procrustes target rotation** | Rotate PCA toward a party-defined target | Softer constraint — pulls rather than fixes | Complexity without clear gain over LDA |
| **IRT-M** (Lauderdale & Clark 2024) | Theory-constrained IRT dimensions | Most principled — encodes which bills load on which dimension | Requires d(d-1) zero constraints coded by hand |
| **Manual PCA overrides** | Human-in-the-loop LDA | Transparent, auditable, no overfitting | Doesn't scale, requires re-audit when parameters change |

No alternative clearly dominates LDA for the specific task of finding the party-separating direction in PCA space. The closest competitor is Supervised PCA, which avoids the small-sample covariance problem by operating at the vote level rather than the legislator level.

---

## Recommendation

### Use LDA as a complement to PCA, not a replacement

The right architecture is to run both:

1. **PCA** for unsupervised discovery — the raw structure of the data, including intra-party factionalism, geographic patterns, and any other latent dimensions. PCA reports should continue to show raw PC1 vs PC2 scatter plots, with axis-swap warnings when PC2 separates parties better than PC1.

2. **LDA projection** for party-oriented tasks — IRT initialization, canonical routing, and the "ideology score" that downstream phases consume. This would replace `detect_ideology_pc()`, `pca_overrides.yaml`, and the manual override loading in `init_strategy.py` and `canonical_ideal_points.py`.

3. **Establishment score** as a bonus — the orthogonal complement of LDA, available for phases that want to analyze intra-party factionalism (profiles, synthesis, cross-session analysis).

### Implementation details

- Use shrinkage LDA (`LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')`) to handle small Democrat groups
- Fit on Republicans and Democrats only; project Independents after fitting
- Use all available PCs (up to 5) for LDA, but fall back to PC1+PC2 when `n_D < 10`
- Orient the LDA axis so Republicans are positive (same convention as PC1)
- Store `ideology_score` and `establishment_score` in the PCA scores parquet

### What this replaces

- `analysis/pca_overrides.yaml` — no longer needed; LDA finds the party axis automatically
- `detect_ideology_pc()` in `init_strategy.py` — subsumed by LDA
- `load_pca_override()` in `init_strategy.py` — no longer needed
- The PC2 sign orientation code added in this session — subsumed by LDA orientation

### What this preserves

- Raw PCA components (PC1-PC5) — still computed, still in the parquet, still in the report
- PCA report visualizations — raw scatter plots showing the unsupervised structure
- Axis-swap warnings — still valuable as diagnostics, even if LDA handles routing
- The contested threshold — still useful for removing uninformative votes before PCA

---

## Literature Context

No published paper applies Fisher's LDA as a preprocessing step for IRT models on roll-call data. The approach is novel in this specific application, though the components are well-established:

- **PCA on roll-call data:** Standard since Poole & Rosenthal (1985). W-NOMINATE's initialization uses eigendecomposition of the double-centered agreement matrix, which is mathematically equivalent to PCA/classical MDS.

- **Fisher's LDA:** Introduced by Fisher (1936), extensively studied in statistics and machine learning. The two-class case is equivalent to Canonical Correlation Analysis with a binary grouping variable.

- **Supervised PCA:** Bair, Hastie, Paul & Tibshirani (2006) screen features by outcome association before PCA. Conceptually related to the contested threshold.

- **Regularized LDA:** Friedman (1989) introduced shrinkage toward identity; Ledoit & Wolf (2004) provided the optimal shrinkage estimator used by scikit-learn.

- **The party-vs-ideology distinction:** Poole & Rosenthal (1997, 2007) argue the first NOMINATE dimension is ideology, not party. Aldrich, Montgomery & Sparks (2014) show polarization makes the distinction difficult even for unsupervised methods. Roberts (2007) warns that the composition of the roll-call record is shaped by strategic agenda control.

- **Theory-constrained IRT:** Lauderdale & Clark (2024) IRT-M encodes theoretical relationships between items and dimensions. Shin (2024) IssueIRT uses topic labels on bills. These are more principled solutions to the dimension-labeling problem but require substantial implementation effort.

- **Known-groups validation:** In psychometrics, using group membership to validate (not estimate) factor scores is standard practice (e.g., MMPI criterion-keyed scales). LDA goes further — it uses group membership to *define* the axis, not just validate it.

---

## References

- Aldrich, J. H., Montgomery, J. M., & Sparks, D. B. (2014). Polarization and ideology: Partisan sources of low dimensionality in scaled roll call analyses. *Political Analysis*, 22(4), 435-456.
- Bair, E., Hastie, T., Paul, D., & Tibshirani, R. (2006). Prediction by supervised principal components. *JASA*, 101(473), 119-137.
- Clinton, J., Jackman, S., & Rivers, D. (2004). The statistical analysis of roll call data. *APSR*, 98(2), 355-370.
- Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179-188.
- Friedman, J. H. (1989). Regularized discriminant analysis. *JASA*, 84(405), 165-175.
- Heckman, J. J., & Snyder, J. M. (1997). Linear probability models of the demand for attributes. *RAND Journal of Economics*, 28, S142-S189.
- Lauderdale, B. E., & Clark, T. S. (2024). Measurement that matches theory: Theory-driven identification in IRT models. *APSR*.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.
- Poole, K. T., & Rosenthal, H. (1997). *Congress: A Political-Economic History of Roll Call Voting*. Oxford University Press.
- Poole, K. T., & Rosenthal, H. (2007). *Ideology and Congress* (2nd ed.). Transaction.
- Roberts, J. M. (2007). The statistical analysis of roll-call data: A cautionary tale. *Legislative Studies Quarterly*, 32, 341-360.
- Shin, S., Lim, D., & Park, J. (2025). L1-based Bayesian ideal point model for multidimensional politics. *JASA*, 120(550), 631-644.
- Shor, B., & McCarty, N. (2011). The ideological mapping of American legislatures. *APSR*, 105(3), 530-551.
