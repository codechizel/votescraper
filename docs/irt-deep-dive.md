# Ideal Point Estimation Deep Dive

A comprehensive audit of our Bayesian IRT implementation — field survey, code review, correctness assessment, and recommendations.

**Date:** 2026-02-25

---

## What This Article Covers

We stepped back and looked at our IRT implementation with fresh eyes. This meant:

1. Surveying every Python implementation and open source project we could find
2. Reviewing the canonical methodology (Clinton-Jackman-Rivers, Martin-Quinn, NOMINATE)
3. Auditing our code line-by-line against field best practices
4. Assessing test coverage and identifying gaps
5. Looking for dead code, refactoring opportunities, and correctness issues

The short version: the implementation is sound and in several areas exceeds what's available in the open source Python ecosystem. But there are specific test gaps worth addressing and a few minor improvements worth making.

---

## The Field: Python IRT for Legislative Analysis

### There Isn't Much

The most surprising finding from the survey is how sparse the Python landscape is for legislative ideal point estimation. The field is dominated by R and Stan:

| Package | Language | Method | Notes |
|---------|----------|--------|-------|
| `pscl::ideal()` | R | Gibbs MCMC | Canonical CJR reference; `startvals="eigen"` since 2001 |
| `emIRT` | R | EM algorithm | 100-1000x faster than MCMC; Imai et al. (2016) |
| `idealstan` | R + Stan | HMC | Most comprehensive: time-varying, informative missingness, ordinal |
| `wnominate` | R | MLE | Poole-Rosenthal NOMINATE; Gaussian utility, not Bayesian |
| `hbamr` | R + Stan | HMC | Hierarchical Bayesian Aldrich-McKelvey scaling; active (Feb 2026) |
| `pgIRT` | R | Polya-Gamma EM | Data augmentation for fast IRT; active (Oct 2024) |
| `py-irt` | Python (Pyro) | Variational inference | Educational/NLP focus; no anchors, no missing data model |
| `tbip` | Python (TF/NumPyro) | VI | Text-based ideal points; outdated TF 1.14 dependencies |
| `Jnotype` | Python (JAX) | Various | WIP, unstable API, 3 stars |

On PyMC specifically, there is no production-grade IRT package for legislative analysis. The PyMC Discourse has scattered examples, most with critical issues. A notable April 2025 thread where Bob Carpenter evaluated AI-generated IRT code found all attempts had identification failures. His key insight: "Soft identification is usually insufficient for these models."

Our implementation fills a genuine gap in the Python ecosystem.

### What the Canonical References Say

**Clinton-Jackman-Rivers (2004)** established the Bayesian IRT framework for roll call data:

```
P(Yea) = logit⁻¹(β_j · ξ_i - α_j)

ξ_i  = legislator ideal point
α_j  = bill difficulty
β_j  = bill discrimination
```

Three properties made CJR the standard: simultaneous estimation of all parameters, full posterior uncertainty via credible intervals, and native missing data handling.

**Jackman (2001)** provided practical guidance: PCA-based starting values (eigendecomposition of the vote matrix), anchor constraints for identification, and the importance of auditing bill-level parameters — not just legislator scores.

**Martin-Quinn** extended CJR to dynamic (time-varying) ideal points using a random walk prior: `ξ_{i,t} ~ Normal(ξ_{i,t-1}, Δ_i)`. Relevant for our roadmap item on cross-session temporal modeling, but not applicable within a single biennium.

---

## How We Compare to Best Practices

### Identification Strategy

The literature documents three canonical approaches:

1. **Fix legislator positions (our approach):** Pin two legislators at known extremes. Resolves additive aliasing, multiplicative aliasing, and sign ambiguity simultaneously.
2. **Standardize + signed discrimination:** Constrain ξ to mean 0, sd 1; use skew-normal on β to set direction. More robust to bad anchor choices.
3. **Standardize + single discrimination sign:** Constrain ξ and restrict the sign of one β parameter.

Our implementation uses Strategy 1 with automated, party-aware anchor selection via PCA PC1 extremes: the most extreme Republican (highest PC1) as conservative anchor, and the most extreme Democrat (lowest PC1) as liberal anchor. This is consistent with `pscl::ideal()` and `idealstan`. The automation is an improvement over most implementations, which require manual anchor specification. The party-aware selection prevents sign flip in supermajority chambers where intra-party variation dominates PC1. See `docs/irt-sign-identification-deep-dive.md`.

**Hard vs. soft constraints:** The PyMC Discourse documents a case where overly tight anchor priors (`sd=0.001`) caused 455 divergences. Our approach — fixing values directly via `pt.set_subtensor()` rather than using spike priors — is cleaner and avoids this numerical issue entirely.

**Assessment:** Sound. Matches or exceeds field practice.

### Prior Choices

| Parameter | Our Prior | Field Standard | Assessment |
|-----------|-----------|---------------|------------|
| ξ (ideal points) | Normal(0, 1) | Normal(0, 1) | Standard |
| α (difficulty) | Normal(0, 5) | Normal(0, 5) or Normal(0, 10) | Slightly tighter than some; appropriate for ±1 anchor scale |
| β (discrimination) | Normal(0, 1) | LogNormal or Normal, depending on identification | Correct for hard-anchored models |

The β prior is the most interesting. Standard advice says use LogNormal to constrain discrimination positive, but that advice assumes soft identification. With hard anchors, unconstrained Normal is both theoretically correct and empirically superior — our beta prior investigation documented a 3.5% accuracy improvement, 10x ESS gain, and 18% faster sampling when switching from LogNormal to Normal. This finding is well-supported by the theory (anchors break the sign symmetry, making positive-constraint redundant and harmful) but rarely discussed in the literature because most implementations either use soft identification or don't audit bill-level parameters.

**Assessment:** Our Normal(0,1) β prior is a genuine methodological contribution. Well-documented in `analysis/design/beta_prior_investigation.md`.

### Non-Centered Parameterization

The flat IRT model uses centered parameterization (`xi_free ~ Normal(0, 1)` directly). The hierarchical model uses non-centered (`xi = mu_party + sigma_within * xi_offset`). Both are correct.

Non-centered parameterization addresses the "funnel of hell" geometry (Betancourt & Girolami, 2013) that arises when hyperparameters control group-level variance. Without hierarchy on ideal points, centered is appropriate. With the party-level hierarchy, non-centered is essential.

**Assessment:** Correct usage in both contexts.

### PCA-Informed Initialization

Jackman's `pscl::ideal()` has used `startvals="eigen"` as the default since 2001. Our PCA initialization is the same idea in PyMC. The experimental validation (5/16 chamber-sessions failing without it, 0/16 with it) provides stronger evidence than most implementations offer.

**Assessment:** Standard practice, well-validated. The `--no-pca-init` escape hatch is appropriate.

### Convergence Diagnostics

Our thresholds match Vehtari, Gelman, et al. (2021):

| Diagnostic | Our Threshold | Vehtari et al. | Status |
|------------|--------------|----------------|--------|
| R-hat | < 1.01 | < 1.01 | Matches |
| ESS | > 400 | > 400 (100 per chain) | Matches |
| Divergences | < 10 | 0 ideally | Slightly permissive but practical |
| E-BFMI | > 0.3 | > 0.3 | Matches |

One subtle point: Vehtari et al. distinguish between bulk-ESS and tail-ESS. Our implementation uses ArviZ's `az.ess()` which computes bulk-ESS by default. Adding tail-ESS would be a minor enhancement — it catches poor mixing in the posterior tails that bulk-ESS can miss.

**Assessment:** Strong. The only gap is tail-ESS, which is minor for well-behaved posteriors.

### Missing Data Handling

We exclude absences from the likelihood (MAR assumption). This is the standard approach in `pscl::ideal()`, `emIRT`, and most implementations.

The alternative — modeling informative missingness — is available in `idealstan` but requires evidence of strategic abstention. Kansas state legislature data shows a 2.6% absence rate with no systematic pattern, making MAR appropriate.

**Assessment:** Correct for our data.

### Cross-Chamber Test Equating

Our test equating approach (scale factor A from shared bill discrimination SDs, location shift B from bridging legislators) is a sophisticated addition that most implementations lack. The method is mathematically sound and follows measurement theory conventions.

**Known fragility:** Only 3 bridging legislators for the location shift in the 91st session. The code correctly falls back to B=0 if no bridging legislators are found.

**Assessment:** Novel and well-implemented. The fragility is inherent to the problem (few legislators serve in both chambers), not a code issue.

---

## Code Audit Findings

### Model Specification: Correct

The model graph in `build_irt_graph()` (extracted in ADR-0053; compiled and sampled via nutpie in `build_and_sample()`) implements the 2PL correctly:

```python
xi_free = pm.Normal("xi_free", mu=0, sigma=1, shape=n_leg - n_anchors)
# Anchors inserted via pt.set_subtensor
alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")
beta = pm.Normal("beta", mu=0, sigma=1, shape=n_votes, dims="vote")
eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")
```

The hierarchical model graph in `build_per_chamber_graph()` (compiled and sampled via nutpie in `build_per_chamber_model()`) is also correct:

```python
mu_party_raw = pm.Normal("mu_party_raw", mu=0, sigma=2, shape=n_parties)
mu_party = pm.Deterministic("mu_party", pt.sort(mu_party_raw), dims="party")
sigma_within = pm.HalfNormal("sigma_within", sigma=1, shape=n_parties, dims="party")
xi_offset = pm.Normal("xi_offset", mu=0, sigma=1, shape=n_leg, dims="legislator")
xi = pm.Deterministic("xi", mu_party[party_idx] + sigma_within[party_idx] * xi_offset)
```

The joint model's per-chamber ordering constraint (`hierarchical.py:451-453`) correctly sorts House and Senate pairs independently:

```python
house_pair = pt.sort(group_offset_raw[:2])
senate_pair = pt.sort(group_offset_raw[2:])
```

### Dead Code: None Found

No unused imports, no TODO/FIXME/HACK markers, no commented-out code. All constants are referenced. The codebase is clean.

### The Optional Soft Sign Constraint

`irt.py:1163-1170` implements an experimental `pm.Potential("sign_constraint", -(mean_d - mean_r))` that penalizes mean(ξ_D) > mean(ξ_R). The design doc notes this had zero effect in testing — anchors already handle sign identification completely. This code is harmless (disabled by default), but it's dead weight from an experiment that's concluded.

**Recommendation:** Consider removing the `--sign-constraint` flag and associated code. It's an experiment that proved unnecessary and adds parameter surface without benefit. Low priority — it's not hurting anything.

### Shrinkage Rescaling Silent Fallback

In `hierarchical.py:643-679`, `extract_hierarchical_ideal_points()` rescales flat IRT estimates to the hierarchical scale via `np.polyfit`. If fewer than 3 legislators match between flat and hierarchical models, it silently falls back to `slope=1.0`:

```python
else:
    slope = 1.0
    df = df.with_columns(
        pl.col("flat_xi_mean").alias("flat_xi_rescaled"),
    )
```

This is safe but should log a warning. If this fallback triggers, the shrinkage comparison becomes less accurate.

**Recommendation:** Add a print statement when the fallback triggers. One-line fix.

### Hardcoded Plot Constant

`irt.py:1989` uses `n = min(4, len(slugs))` for convergence summary plots — hardcoded 4 where other similar constants are explicitly named. Minor style inconsistency.

**Recommendation:** Extract to a named constant (`N_CONVERGENCE_SUMMARY = 4`). Low priority.

---

## Test Coverage Assessment

### What's Well-Tested (45 tests in `test_irt.py`, 25 in `test_hierarchical.py`)

**Data preparation and transformations** — thorough coverage:
- `prepare_irt_data()`: wide-to-long conversion, null handling, index preservation
- `select_anchors()`: PCA-based selection with 50% participation guard
- `filter_vote_matrix_for_sensitivity()`: minority threshold filtering
- `build_joint_vote_matrix()`: cross-chamber bill matching, bridging legislator detection
- `unmerge_bridging_legislators()`: expansion of merged legislators

**Mathematical foundations** — validated:
- Logistic 2PL equation: P(Yea) = logit⁻¹(β·ξ - α)
- Parameter role tests: positive β favors conservatives, negative favors liberals
- Alpha as threshold shift (cannot flip ordering)
- Zero β produces constant probability

**Edge cases** — covered:
- Single-party legislatures, small legislator counts, low-participation legislators
- Missing votes in joint models, bridging legislator presence/absence
- Paradox legislator detection, uncertainty detection

### Critical Gaps (as of initial audit — now partially addressed)

The test suite covered data transformation logic thoroughly but had zero coverage of the Bayesian inference itself. **Update (2026-02-25):** 28 new tests were added to address the high-priority gaps marked below.

| Component | Coverage | Risk | Status |
|-----------|----------|------|--------|
| MCMC sampling (`build_irt_graph` + `build_and_sample`) | graph: **3 tests** | ~~High~~ Medium | Graph-builder tests added (ADR-0053); nutpie integration test Tier 3 |
| Convergence diagnostics (`check_convergence`) | ~~0%~~ **9 tests** | ~~High~~ Low | **ADDRESSED** — R-hat, bulk-ESS, tail-ESS, divergences, E-BFMI, mode-split |
| Posterior extraction (`extract_ideal_points`, `extract_bill_parameters`) | ~~0%~~ **12 tests** | ~~High~~ Low | **ADDRESSED** — schema, sorting, HDI, metadata |
| PCA-informed initialization | 0% | Medium | Requires integration test (Tier 3) |
| Holdout validation & PPC | 0% | Medium | Requires integration test (Tier 3) |
| Test equating (`equate_chambers`) | ~~0%~~ **7 tests** | ~~Medium~~ Low | **ADDRESSED** — transformation, bridging, fallback |
| Visualization functions | 0% | Low | Not prioritized |
| Report builders | 0% | Low | Not prioritized |

### Why This Matters

The data preparation tests ensure we're feeding the right data to PyMC. But nothing validates that:
1. Convergence diagnostics correctly flag failed runs
2. Posterior extraction produces correct HDIs
3. The holdout prediction logic computes AUC-ROC accurately
4. Test equating handles edge cases (negative scale factor, no bridging legislators)

These functions are where subtle bugs would live — the kind where output looks plausible but is quietly wrong.

### Recommended New Tests (Prioritized)

**Tier 1 — Mock inference tests (high impact, moderate effort):**

Create a synthetic `InferenceData` object with known properties (e.g., two chains with identical distributions, or two chains with deliberately different means) and test:
- `check_convergence()` returns correct R-hat/ESS/divergence status
- `extract_ideal_points()` produces correct posterior means and HDIs
- `extract_bill_parameters()` handles positive and negative discrimination

**Tier 2 — Equating and validation tests (medium impact, moderate effort):**

- `equate_chambers()` with known scale factors and bridging legislators
- Holdout prediction logic with a pre-computed posterior
- HDI bound swap when scale factor A < 0

**Tier 3 — Integration tests (high impact, high effort):**

- End-to-end on a tiny synthetic dataset (5 legislators, 10 bills, 2 chains, 50 draws)
- Verify the full pipeline: data prep → sample → diagnose → extract → report

---

## Comparison to the Most Complete Implementation: idealstan

`idealstan` (R + Stan) is the most comprehensive ideal point package in any language. Comparing feature-for-feature:

| Feature | idealstan | Our Implementation | Gap |
|---------|-----------|-------------------|-----|
| Binary response (Yea/Nay) | Yes | Yes | None |
| Ordinal response | Yes | No | Not needed (data is binary) |
| Time-varying ideal points | Yes | No | Roadmap item (Martin-Quinn) |
| Informative missingness | Yes (hurdle model) | No (MAR) | Not needed (2.6% absence rate) |
| Hard anchors | Yes | Yes | None |
| Soft anchors | Yes | No | Not needed (hard anchors work) |
| PCA initialization | No (manual) | Yes (automated) | We're ahead |
| Unconstrained discrimination | Configurable | Yes | None |
| Non-centered hierarchical | Yes | Yes | None |
| Cross-chamber equating | No | Yes | We're ahead |
| Joint cross-chamber model | No | Yes (3-level hierarchy) | We're ahead |
| Convergence diagnostics | Via Stan output | Automated (R-hat, bulk/tail-ESS, E-BFMI, divergences) | None |
| External validation | No | Yes (Shor-McCarty) | We're ahead |
| Report generation | No | Yes (self-contained HTML) | We're ahead |

We lack idealstan's time-varying and informative missingness features, but those aren't relevant for our data. We surpass idealstan in automation (PCA anchor selection, convergence checking), cross-chamber analysis, and reporting.

---

## Interpretability Caveat

John Myles White (2019) raises a fundamental question: do ideal points measure ideology or party loyalty? His argument: IRT assumes legislators "vote in favor of a motion if it moves policy outcomes closer to their most preferred policy." If legislators vote based on caucus pressure rather than policy preferences, ideal points measure intensity of loyalty, not ideology.

For Kansas, this is partially mitigated by our external validation. Correlation with Shor-McCarty scores (r=0.93-0.98), which are estimated from a completely different methodology and different data (including bill text and cosponsorship patterns), suggests our ideal points capture something stable and meaningful — whether we call it "ideology" or "revealed preference."

But the caveat stands: a legislator who votes with their party 100% of the time could be a true believer or an obedient caucus member. IRT cannot distinguish the two.

---

## Summary of Recommendations

> **Update (2026-02-25):** All "Do" and "Consider" recommendations below have been implemented. See ADR-0006 revision history and the `test_irt.py` changelog for details.

### Do (concrete improvements) — IMPLEMENTED

1. **Add mock inference tests** for `check_convergence()`, `extract_ideal_points()`, and `extract_bill_parameters()`. 21 new tests added (9 convergence, 6 ideal point extraction, 6 bill parameter extraction).
2. **Add tail-ESS** to convergence diagnostics alongside bulk-ESS. Now checks `az.ess(idata, method="tail")` for xi, alpha, and beta parameters.
3. **Add a warning** when shrinkage rescaling falls back to slope=1.0 in `extract_hierarchical_ideal_points()`.

### Consider (nice-to-have) — IMPLEMENTED

4. **Removed the `--sign-constraint` flag** and all associated code (`pm.Potential`, CLI argument, parameter threading).
5. **Extracted the hardcoded 4** to `N_CONVERGENCE_SUMMARY = 4` named constant.
6. **Added equating tests** for `equate_chambers()` — 7 new tests covering transformation values, concordance, bridging fallback, and correlations.

### Don't (not worth the effort)

7. Switching to Stan/CmdStanPy. PyMC's NUTS sampler produces equivalent results and the Python-native workflow is simpler.
8. Adding informative missingness modeling. Kansas absence rates don't justify the complexity.
9. Adding variational inference (py-irt approach). Full MCMC posteriors are essential for credible intervals and downstream analyses.

---

## References

### Canonical Methodology
- Clinton, Jackman, Rivers (2004). "The Statistical Analysis of Roll Call Data." *American Political Science Review* 98: 355-370. [PDF](https://www.cs.princeton.edu/courses/archive/fall09/cos597A/papers/ClintonJackmanRivers2004.pdf)
- Jackman (2001). "Practical Issues in Implementing and Understanding Bayesian Ideal Point Estimation." *Political Analysis* 9(3): 227-242.
- Martin & Quinn (2002). "Dynamic Ideal Point Estimation via MCMC for the U.S. Supreme Court." *Political Analysis* 10(2): 134-153.
- Carroll, Lewis, Lo, Poole, Rosenthal (2013). "Comparing NOMINATE and IDEAL." [PDF](https://scholar.princeton.edu/sites/default/files/jameslo/files/lsq_nomvsideal.pdf)

### Convergence and Diagnostics
- Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021). "Rank-Normalization, Folding, and Localization: An Improved R-hat." *Bayesian Analysis* 16(2): 667-718. [arXiv](https://arxiv.org/abs/1903.08008)
- Betancourt (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." [arXiv](https://arxiv.org/abs/1701.02434)
- Betancourt & Girolami (2013). "Hamiltonian Monte Carlo for Hierarchical Models." [arXiv](https://arxiv.org/pdf/1312.0906)

### Priors and Identification
- Gelman (2018). "What Prior to Use for Item-Response Parameters?" [Blog post](https://statmodeling.stat.columbia.edu/2018/03/01/prior-use-item-response-parameters/)
- Stan User's Guide, Section 1.11: Item Response Models. [Link](https://mc-stan.org/docs/2_20/stan-users-guide/item-response-models-section.html)
- Jeffrey Arnold's "BUGS Examples in Stan" — Legislators chapter. [Link](https://jrnold.github.io/bugs-examples-in-stan/legislators.html)

### Python Implementations
- py-irt: [GitHub](https://github.com/nd-ball/py-irt), [PyPI](https://pypi.org/project/py-irt/)
- TBIP: [GitHub](https://github.com/keyonvafa/tbip)
- Jnotype: [GitHub](https://github.com/cbg-ethz/Jnotype)
- RobertMyles/IRT (Stan): [GitHub](https://github.com/RobertMyles/IRT)

### R Implementations (for comparison)
- idealstan: [GitHub](https://github.com/saudiwin/idealstan)
- emIRT: [GitHub](https://github.com/kosukeimai/emIRT), [Paper](https://imai.fas.harvard.edu/research/files/fastideal.pdf)
- hbamr: [GitHub](https://github.com/jbolstad/hbamr)
- pgIRT: [GitHub](https://github.com/vkyo23/pgIRT)

### Interpretability
- John Myles White (2019). "Interpretational Challenges with Ideal Point Models." [Blog post](https://www.johnmyleswhite.com/notebook/2019/01/20/interpretational-challenges-with-ideal-point-models/)

### Our Prior Work
- Beta prior investigation: `analysis/design/beta_prior_investigation.md`
- Hierarchical shrinkage deep dive: `docs/hierarchical-shrinkage-deep-dive.md`
- PCA-informed init: ADR-0023, `docs/adr/0023-pca-informed-irt-initialization.md`
- External validation: `docs/external-validation-results.md`
- IRT design doc: `analysis/design/irt.md`
- Hierarchical design doc: `analysis/design/hierarchical.md`
