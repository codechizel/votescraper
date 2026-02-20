# IRT Design Choices

**Script:** `analysis/irt.py`
**Constants defined at:** `analysis/irt.py:181-205`
**ADR:** `docs/adr/0006-irt-implementation-choices.md`

## Assumptions

1. **Unidimensional ideology.** The 2PL model assumes a single latent dimension explains all voting behavior. Legislators who deviate on a second dimension (e.g., Tyson's contrarianism on routine bills) will have their 1D ideal point estimated as a compromise between their positions on both dimensions. This is by design for the canonical baseline — a 2D model is a future extension.

2. **Yea/Nay only.** The model's likelihood is Bernoulli (binary). "Present and Passing," absences, and non-votes are excluded entirely — they do not enter the likelihood. This is a strength: no imputation artifacts.

3. **Missing at random (MAR).** Absences are assumed uninformative about ideology conditional on the observed votes. If legislators strategically avoid recorded votes on contentious bills, their ideal-point estimates may be too moderate (same concern as PCA, but IRT is less affected because it doesn't impute).

4. **Bills have fixed ideological content.** Each bill's difficulty (alpha) and discrimination (beta) are treated as fixed characteristics. In reality, legislators may interpret the same bill differently, or the bill's ideological content may not map cleanly onto a single dimension.

5. **Exchangeability of non-anchor legislators.** All non-anchor legislators share the same prior: Normal(0, 1). There is no hierarchical structure by party, region, or seniority. The data alone drives posterior differences.

## Parameters & Constants

### MCMC Sampling

| Constant | Value | Justification | Code location |
|----------|-------|---------------|---------------|
| `DEFAULT_N_SAMPLES` | 2000 | Posterior draws per chain. Standard for IRT — sufficient for stable HDIs. | `irt.py:181` |
| `DEFAULT_N_TUNE` | 1000 | NUTS adaptation period (discarded). Allows the sampler to learn the posterior geometry. | `irt.py:182` |
| `DEFAULT_N_CHAINS` | 2 | Two independent chains. Sufficient for R-hat computation. See ADR-0006 for trade-off discussion. | `irt.py:183` |
| `TARGET_ACCEPT` | 0.9 | Higher than the PyMC default (0.8). Reduces divergences in the complex IRT posterior at a ~20% speed cost. | `irt.py:184` |
| `RANDOM_SEED` | 42 | For MCMC reproducibility. Same seed used consistently across phases. | `irt.py:185` |

### Priors

| Prior | Distribution | Justification |
|-------|-------------|---------------|
| xi (ideal point) | Normal(0, 1) + two anchors fixed at +1/-1 | Standard normal center with unit scale. Anchors fix location and scale. |
| alpha (difficulty) | Normal(0, 5) | Diffuse prior. Allows difficulty to range widely — some bills are easy, some are hard. The SD of 5 means 95% prior mass covers [-10, +10], which is much wider than the ideal-point scale. |
| beta (discrimination) | Normal(0, 1) | **Unconstrained.** Anchors provide sign identification; positive β = conservative Yea, negative β = liberal Yea. See `beta_prior_investigation.md` for rationale. |

### Filtering & Validation

| Constant | Value | Justification | Code location |
|----------|-------|---------------|---------------|
| `MINORITY_THRESHOLD` | 0.025 | Inherited from EDA. IRT reads pre-filtered matrices. | `irt.py:187` |
| `SENSITIVITY_THRESHOLD` | 0.10 | Full MCMC re-run at 10% threshold. | `irt.py:188` |
| `MIN_VOTES` | 20 | Inherited from EDA. | `irt.py:189` |
| `HOLDOUT_FRACTION` | 0.20 | In-sample prediction on 20% of observed cells. | `irt.py:190` |
| `HOLDOUT_SEED` | 42 | For holdout reproducibility. | `irt.py:191` |
| `MIN_PARTICIPATION_FOR_ANCHOR` | 0.50 | Anchors must have voted on >= 50% of contested bills. Ensures tight estimation. | `irt.py:192` |

### Convergence Thresholds

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `RHAT_THRESHOLD` | 1.01 | R-hat must be < 1.01 for all parameters. Standard in Bayesian literature. |
| `ESS_THRESHOLD` | 400 | Effective sample size > 400 for reliable inference. Vehtari et al. recommendation. |
| `MAX_DIVERGENCES` | 10 | < 10 divergent transitions across all chains. More indicates posterior geometry problems. |

## Methodological Choices

### Discrimination prior (beta)

**Current decision:** Normal(0, 1) — unconstrained. Anchors provide sign identification.

**What this does:** Allows beta to be positive (conservative Yea) or negative (liberal Yea). The sign of beta encodes which end of the spectrum favors Yea; the magnitude encodes how partisan the vote is. The two hard anchors (xi = +1 and -1) break the sign ambiguity, making a positive constraint unnecessary.

**Why Normal(0, 1):**
- **Uses all data.** With unconstrained beta, both R-Yea bills (beta > 0) and D-Yea bills (beta < 0) contribute to ideal point estimation.
- **Anchors handle identification.** The sign-switching problem that motivates positive-constrained priors is already solved by the hard anchors.
- **Best convergence.** In a 3-way experiment (LogNormal(0.5,0.5), Normal(0,2.5), Normal(0,1)), Normal(0,1) had the highest ESS, fastest sampling, and zero divergences.
- **Best accuracy.** Holdout accuracy +3.5%, AUC +0.025, PCA correlation +0.022 vs LogNormal.

**History:** The initial implementation used LogNormal(0.5, 0.5), following the standard recommendation for soft-identified IRT models. Investigation revealed this silenced 12.5% of bills (all D-Yea votes assigned beta near zero). See `analysis/design/beta_prior_investigation.md` for the full investigation, experiment results, and mathematical proof that alpha cannot compensate for constrained beta.

**Alternatives considered:**
- LogNormal(0.5, 0.5) — rejected: positive constraint silences D-Yea bills. Was standard advice but assumes soft identification, not hard anchors.
- Normal(0, 2.5) — viable but wider prior produces slightly slower convergence than Normal(0, 1) with nearly identical accuracy.
- Half-Normal — viable but lacks negative support needed for D-Yea bills.

### PCA-based anchor selection

**Decision:** Fix the most-conservative legislator (highest PCA PC1 score) at xi=+1 and the most-liberal (lowest PC1) at xi=-1. Both must have >= 50% participation.

**Why:** Automates anchor selection using existing PCA results. No manual knowledge of Kansas politics required. The 50% participation guard ensures anchors have enough data for tight estimation (their ideal points are fixed, so their data directly informs bill parameters).

**Alternatives considered:**
- Manual anchor selection (e.g., "pick a Freedom Caucus member and a Lawrence Democrat") — rejected for reproducibility; requires human judgment that changes across sessions
- Soft identification via N(0,1) prior + post-hoc sign correction — rejected because it's fragile with 2 chains
- Three anchors (conservative, liberal, moderate at 0) — rejected as unnecessary for 1D; two anchors fix location, scale, and sign

**Impact:** If PCA scores are wrong (e.g., the PC1 sign convention flipped incorrectly), the IRT anchors will be wrong. Validated by the PCA-IRT correlation check (r > 0.95 expected).

### Native missing data handling

**Decision:** Absences are handled by simply excluding those (legislator, vote) pairs from the likelihood. No imputation.

**Why:** This is a key advantage of IRT over PCA. The Bernoulli likelihood is defined only over observed cells. A legislator who was absent for 70% of votes still contributes information from their 30%, and the model correctly widens their credible interval to reflect the uncertainty.

**Impact:** Legislators with few votes (e.g., Miller with 30/194 Senate votes) will have wide HDIs but unbiased point estimates. No imputation artifacts.

### Two chains (not four)

**Decision:** Default to 2 MCMC chains instead of the textbook 4.

**Why:** Runtime. Each chain takes ~5-10 minutes per chamber. 2 chains = ~15-20 min total; 4 chains = ~30-40 min. The model is well-identified (anchored, positive-constrained discrimination), so 2 chains are typically sufficient.

**Trade-off:** Less power to detect multi-modal posteriors. If R-hat > 1.01 or ESS < 400, re-run with `--n-chains 4`.

### In-sample holdout (not true out-of-sample)

**Decision:** Use posterior means from the full model to predict a random 20% of observed cells. This is documented as in-sample prediction.

**Why:** A true out-of-sample holdout requires masking cells before fitting and running a second MCMC — doubling runtime. The posterior predictive check (PPC) provides the proper Bayesian validation: it samples from the full posterior and compares replicated data to observed.

**Impact:** The holdout accuracy overstates predictive performance (model saw all data during fitting). The PPC Bayesian p-value is the more reliable validation metric.

### Per-chamber models (not joint)

**Decision:** House and Senate are fitted as completely independent IRT models.

**Why:** Standard approach in the literature (NOMINATE, Clinton et al. 2004). Avoids the complexity of cross-chamber identification.

**Trade-off:** Cannot leverage bridging legislators (e.g., Miller, who served in both chambers). A joint model could use Miller's ~300+ House votes to tightly constrain his Senate ideal point. This is deferred to a future enhancement and documented in `docs/analytic-flags.md`.

## Downstream Implications

### For Clustering (Phase 5)
- **Use IRT ideal points (xi_mean) as the primary clustering input**, not PCA scores. IRT accounts for vote difficulty and discrimination; PCA does not.
- **HDI width is a confidence measure.** Consider weighting legislators by 1/xi_sd or excluding those with xi_sd above a threshold.
- Sen. Miller's wide HDI means his cluster assignment is low-confidence. Flag in cluster interpretation.

### For Network Analysis (Phase 6)
- Bill discrimination parameters (beta) identify the most ideologically informative votes. Consider building networks from high-discrimination votes only (beta > 1.5) for a cleaner signal.
- The difficulty parameter (alpha) identifies where on the spectrum each bill "cuts." This can annotate network edges with substantive meaning.

### For Prediction (Phase 7)
- The IRT posterior can predict future votes: P(Yea) = logit^-1(beta * xi - alpha). For a new bill, estimate alpha and beta from similar past bills; for a new legislator, use party-average xi as a prior.

### For interpretation
- **Ideal points are on an arbitrary scale** (anchored at +1/-1, but the units are not "degrees of conservatism"). Compare legislators to each other, not to absolute values.
- **HDI overlap = indistinguishable.** Two legislators whose 95% HDIs overlap cannot be reliably ranked. The forest plot visualizes this.
- **Discrimination sign indicates direction.** Positive β = conservative position is Yea. Negative β = liberal position is Yea. |β| measures discriminating power. A bill with |β| > 1.5 is highly discriminating; |β| < 0.5 is weakly discriminating.
- **The 1D model is a simplification.** Tyson's contrarianism on routine bills, Thompson's mild version of the same, and any other multi-dimensional pattern will be compressed into a single number. If a legislator's ideal point seems surprising, check whether they have unusual PC2 behavior (see `docs/analytic-flags.md`).
