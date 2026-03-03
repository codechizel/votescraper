# Hierarchical IRT Deep Dive

A code audit, ecosystem survey, and fresh-eyes evaluation of the Hierarchical Bayesian IRT implementation (Phase 10).

**Date:** 2026-02-25

---

## Executive Summary

The implementation is mathematically correct and well-engineered. The 2-level per-chamber model uses textbook non-centered parameterization with an ordering constraint for identification — both standard best practices confirmed by the literature (Gelman BDA3, Betancourt 2017, Papaspiliopoulos et al. 2007). External validation against Shor-McCarty scores confirms the House model (r=0.984) and the flat IRT it extends (r=0.98). The code is clean and thoroughly documented (ADR-0017, design doc, two prior deep dives).

This deep dive identified **nine issues** (two substantive, one code defect, six code quality), **eight test gaps** (including one defective existing test), and **one refactoring opportunity**. All issues have been fixed and all test gaps addressed — test count went from 26 to 35. It also surveys the Python ecosystem for hierarchical IRT — a survey that reveals our implementation fills a genuine gap, as no Python library offers party-level hierarchical IRT for legislative analysis.

The critical known limitation — Senate hierarchical failure due to J=2 groups with small N — was already discovered and documented via external validation. A code-level mitigation (small-group warning) has been implemented.

---

## 1. Python Ecosystem Survey

### 1.1 The Landscape: R Dominates, Python Has Gaps

Legislative ideal point estimation is an R-first field. Python has scattered implementations but no production-grade hierarchical IRT for legislative analysis. We surveyed every relevant package to understand the state of the art and confirm that building a custom PyMC implementation was the right call.

**This project uses Python exclusively** (no R, no rpy2). The R packages below are listed only for context — they represent the field's reference implementations and help validate that our model math is correct. They are not dependencies, recommendations, or candidates for adoption.

| Package | Language | Models | Estimation | Hierarchical | Legislative Focus | Status |
|---------|----------|--------|------------|-------------|-------------------|--------|
| **idealstan** | R + Stan | 2PL binary, ordinal, counts, continuous | HMC (Stan) | Time-varying, informative missingness | Yes (primary use case) | Active (Kubinec 2024) |
| **pscl::ideal()** | R | 2PL binary | Gibbs MCMC | None (flat only) | Yes (CJR reference) | Maintained |
| **emIRT** | R | 1PL/2PL binary, ordinal, dynamic | Variational EM | `hierIRT()` covariates + `dynIRT()` dynamic | Yes | Active (Sep 2025) |
| **MCMCpack** | R | 1PL/2PL binary, K-dim, dynamic | Gibbs MCMC | `MCMCirtHier1d()` subject covariates | Yes | Active (Aug 2024) |
| **wnominate** | R | NOMINATE | MLE (Gaussian utility) | None | Yes | Maintained (Jun 2024) |
| **hbamr** | R + Stan | Aldrich-McKelvey scaling | HMC (Stan) | Yes (hierarchical BAM) | Survey-based scaling | Active (Bolstad 2024) |
| **pgIRT** | R | 2PL binary, dynamic | Polya-Gamma EM | Dynamic | Yes (Goplerud 2019) | Active |
| **hIRT** | R | 2PL binary, graded response | EM | Covariate-dependent mean + variance | No (public opinion) | Archived from CRAN 2026-01-30 |
| **py-irt** | Python (Pyro) | 1PL, 2PL, 4PL | Variational inference | Global hyperprior only | No (NLP/education) | v0.6.6, **requires Python <3.12** |
| **girth** | Python | 1PL-3PL, graded, partial credit | MLE | None | No (psychometrics) | Last release 2022 |
| **girth_mcmc** | Python (PyMC3) | Same as girth | MCMC or VI | None | No | v0.6.0, Nov 2021 (stale) |
| **deepirtools** | Python (PyTorch) | Deep learning IRT | Amortized VI | None | No (research code) | Urban 2024 |
| **pynominate** | Python | DW-NOMINATE | MLE | None | Yes (Voteview port) | ~30 commits, minimal maintenance |
| **NumPyro** | Python (JAX) | Manual specification | HMC (NUTS) | Whatever you code | N/A (PPL) | Active |
| **PyMC** | Python | Manual specification | HMC (NUTS) | Whatever you code | N/A (PPL) | Active |

### 1.2 What py-irt Actually Offers

py-irt (Lalor 2022, INFORMS Journal on Computing) is the most prominent Python IRT library. We examined it closely:

- **Models:** 1PL, 2PL, 4PL (no 3PL yet). Supports "vague" or "hierarchical" priors on ability parameters.
- **Estimation:** Variational inference via Pyro (not MCMC). GPU-accelerated via PyTorch.
- **Identification:** No anchor constraints. No ordering constraints. Relies on VI's mode-seeking behavior.
- **Missing data:** No explicit missing data model.
- **Application focus:** Educational testing and NLP evaluation (LLM benchmarking). The documentation examples use SQuAD dataset evaluation, not legislative voting.
- **Hierarchical extension:** The "hierarchical prior" option adds a global hyperprior on ability, not party-level partial pooling. There is no way to specify group structure.

**Python version blocker:** py-irt v0.6.6 declares `python_requires=">=3.8,<3.12"` due to its Pyro/PyTorch dependency chain. Our project requires Python 3.14+, making py-irt unusable even if its feature set were appropriate.

**Verdict:** py-irt solves a different problem (educational assessment at scale via VI) and lacks the identification, group structure, and legislative-specific features we need. The Python <3.12 requirement is an independent hard blocker.

### 1.3 The NumPyro Alternative

NumPyro (JAX-backed NUTS) is the most plausible alternative backend to PyMC. A peer-reviewed benchmark (Nishio et al. 2023, PeerJ Computer Science) comparing NumPyro and PyStan for Bayesian IRT found comparable parameter estimates with shorter sampling times in NumPyro, particularly on GPU. A separate benchmark (Ingram 2023) found PyMC+JAX GPU sampling ~11x faster than standard PyMC and ~4x faster than JAX on CPU.

**For our use case (~40K observations, 91st session, ~90 min on M3 Pro CPU):**

- The model specification would be nearly identical (same math, different syntax)
- We'd lose ArviZ integration (convergence diagnostics, NetCDF export, HDI computation) — or need to bridge it
- PyMC itself can use the NumPyro NUTS sampler as a backend (`pm.sampling_jax.sample_numpyro_nuts()`) without rewriting the model
- The migration cost doesn't justify the speedup for a pipeline that runs infrequently
- **Apple Silicon GPU (M3 Pro):** JAX-Metal (Apple's JAX GPU backend) is experimental and unstable as of early 2026. GPU-accelerated sampling via NumPyro is not currently viable on our hardware. CPU-only NumPyro would still provide ~2-4x speedup from JAX JIT compilation.

**Recommendation:** Stay with PyMC. If speed becomes critical, try `sample_numpyro_nuts()` as a drop-in first (CPU-only on Apple Silicon). Full NumPyro migration is not warranted.

### 1.4 Variational Inference for IRT

Recent work demonstrates VI can approximate IRT posteriors orders of magnitude faster than MCMC:
- Wu et al. (2020), "Variational Item Response Theory: Fast, Accurate, and Expressive"
- Wu et al. (2021), "Modeling IRT with Stochastic Variational Inference"
- emIRT's `dynIRT` uses variational EM for Martin-Quinn scores (Imai et al. 2016)

**For hierarchical IRT specifically:** VI's mode-seeking behavior (it finds one mode) is problematic when the posterior is multimodal — exactly the scenario with J=2 groups and small N that causes our Senate failure. MCMC at least reports the problem (R-hat > 1, low ESS). VI would silently converge to one mode and report false confidence.

**Recommendation:** MCMC remains the right choice for our sample sizes and group structure.

### 1.5 Recent Methodological Advances

**L1 Bayesian Ideal Points (Shin 2024, JASA):** Proposes L1-distance-based ideal point estimation for multidimensional politics. The L1 formulation transforms rotational invariance into a signed permutation problem, enabling principled multidimensional estimation. Applied to the U.S. House during the Gilded Age (1891-1899). Interesting but not relevant to our 1D model.

**Bob Carpenter's PyMC IRT Evaluation (April 2025):** Carpenter tested AI-generated PyMC IRT code on the PyMC Discourse and found all attempts failed on identification. Key finding: IRT 2PL models face additive non-identifiability (solvable via sum-to-zero constraints) and multiplicative non-identifiability (solvable via pinning discriminativeness or anchoring). Carpenter noted the models "fit way better with sum-to-zero constraints" and that "soft identification is usually insufficient." Our implementation uses hard constraints (anchors in flat, ordering in hierarchical), avoiding this pitfall.

---

## 2. Code Audit

### 2.1 Mathematical Correctness

The model specification is correct and matches the literature:

**Per-chamber model (lines 321-344):**
```
mu_party_raw ~ Normal(0, sigma=2)           [2 params]
mu_party = sort(mu_party_raw)               [identification: D < R]
sigma_within ~ HalfNormal(sigma=1)          [2 params, per-party]

xi_offset_i ~ Normal(0, sigma=1)            [n_legislators params]
xi_i = mu_party[party_i] + sigma_within[party_i] * xi_offset_i

alpha_j ~ Normal(0, sigma=5)                [n_votes params]
beta_j ~ Normal(0, sigma=1)                 [n_votes params]

P(Yea) = logit^-1(beta_j * xi_i - alpha_j)
```

Validated against:

| Reference | Requirement | Implementation | Status |
|-----------|-------------|----------------|--------|
| Clinton-Jackman-Rivers (2004) | P(Yea) = logit^-1(beta * xi - alpha) | Lines 343-344 | Correct |
| Gelman BDA3 Ch 5 | Non-centered for hierarchical | Lines 331-335 | Correct |
| Papaspiliopoulos et al. (2007) | Non-centered avoids funnel | xi = mu + sigma * offset | Correct |
| Betancourt (2017) | HMC-friendly parameterization | Non-centered + HalfNormal | Correct |

**Joint model (lines 434-477):** Correctly extends to 3 levels (global -> chamber -> group -> legislator) with per-chamber-pair ordering constraints. The fix applied 2026-02-23 after label-switching discovery in the 90th session correctly sorts each chamber's pair independently.

**Identification strategy:** The ordering constraint `pt.sort(mu_party_raw)` is a valid identification strategy. It's cleaner than the alternatives:
- Hard anchors (flat IRT's approach) — prevents anchored legislators from being shrunk toward party mean
- `pm.Potential` soft constraints — numerically fragile, can produce gradient discontinuities (Carpenter 2025 specifically criticized this approach)
- Post-hoc label alignment — requires checking all posterior draws, error-prone

**ICC computation (lines 744-793):** Correct. Between-group variance computed per posterior draw, within-group variance weighted by group size for unbalanced groups, total variance used as denominator. Edge case (total_var = 0) handled with `np.where`. **One naming inconsistency:** the ICC credible interval is computed via `np.percentile([2.5, 97.5])` (equal-tailed interval) but columns are named `icc_hdi_*` (suggesting a highest-density interval via `az.hdi()`). All other posterior summaries in the module use `az.hdi()`. For symmetric posteriors this distinction is immaterial; for skewed ICC distributions near 0 or 1 the difference could matter. See Issue 7.

**Shrinkage rescaling (lines 643-689):** Mathematically sound. Linear rescaling via `np.polyfit` is the standard approach in test equating (educational measurement theory). IRT ideal points are ordinal-scale, so linear transforms preserve all meaningful relationships.

### 2.2 Issue 1: No Small-Group Warning (Substantive)

**File:** `analysis/07_hierarchical/hierarchical.py`, `prepare_hierarchical_data()`, lines 279-282

The code prints party composition but never warns when a group is dangerously small:

```python
for i, name in enumerate(PARTY_NAMES):
    count = int((party_idx == i).sum())
    print(f"  {name}: {count} legislators")
```

The James-Stein estimator (1961) dominates the MLE only when the number of groups J >= 3. With J=2, shrinkage can *increase* mean squared error for the smaller group. The hierarchical shrinkage deep dive documented this producing r=-0.541 (inverted) for Senate Democrats with ~11 legislators, while the flat model achieved r=0.929.

**Impact:** Users running on new sessions with small minority parties will get no warning that the hierarchical model may be unreliable for that chamber. Convergence diagnostics (R-hat, ESS) will eventually flag the problem, but a proactive warning at data preparation time is much clearer.

**Recommendation:** Add a warning threshold:

```python
MIN_GROUP_SIZE_WARN = 15  # Below this, hierarchical may be unreliable

for i, name in enumerate(PARTY_NAMES):
    count = int((party_idx == i).sum())
    print(f"  {name}: {count} legislators")
    if count < MIN_GROUP_SIZE_WARN:
        print(
            f"  WARNING: {name} has only {count} legislators. "
            "Hierarchical shrinkage may be unreliable for groups this small "
            "(J=2 groups with small N; flat IRT may be more trustworthy). "
            "See docs/hierarchical-shrinkage-deep-dive.md"
        )
```

### 2.3 Issue 2: Shrinkage Scatter Uses Raw Scales (Code Quality)

**File:** `analysis/07_hierarchical/hierarchical.py`, `plot_shrinkage_scatter()`, lines 978-979

The scatter plot uses raw flat and hierarchical ideal points, which are on different scales (flat ~ [-4, 3], hier ~ [-11, 9]). The Pearson r in the title is scale-invariant and correct, but the identity line (line 998) is visually misleading — points cluster far from the diagonal because of the scale difference, not because of shrinkage.

The `extract_hierarchical_ideal_points` function computes `flat_xi_rescaled` (via `np.polyfit`) but drops it (line 689) before the plotting stage can use it.

**Recommendation:** Keep `flat_xi_rescaled` in the output parquet (remove the `.drop()` on line 689) and use it for the scatter plot x-axis.

### 2.4 Issue 3: `flat_xi_rescaled` Dropped from Output (Code Quality)

**File:** `analysis/07_hierarchical/hierarchical.py`, line 689

```python
df = df.drop("flat_xi_rescaled")
```

The rescaled flat values are computed for `delta_from_flat` and `shrinkage_pct` calculations, then dropped. Keeping them costs negligible storage and benefits the scatter plot (Issue 2), downstream phases wanting a common scale, and debugging.

**Recommendation:** Remove the `.drop()` call.

### 2.5 Issue 4: Hardcoded `0.5` Threshold for Shrinkage Stability (Code Quality)

**File:** `analysis/07_hierarchical/hierarchical.py`, line 681

```python
pl.when((pl.col("flat_xi_sd") > 0.01) & (flat_dist > 0.5))
```

The `0.5` threshold for `flat_dist` (distance from party mean below which `shrinkage_pct` is null) is a magic number. It's a reasonable heuristic — shrinkage ratios become noisy when the denominator is small — but it should be a named constant.

**Recommendation:** Extract to `SHRINKAGE_MIN_DISTANCE = 0.5`.

### 2.6 Issue 5: Convergence Variable Lists Hardcoded (Code Quality)

**File:** `analysis/07_hierarchical/hierarchical.py`, `check_hierarchical_convergence()`, line 526

```python
var_names = ["xi", "mu_party", "sigma_within", "alpha", "beta"]
```

Joint-specific variables are added dynamically (lines 530-532). Both lists could be module-level constants for clarity.

**Recommendation:** Low priority. Extract to `HIER_CONVERGENCE_VARS` and `JOINT_EXTRA_VARS`.

### 2.7 Issue 6: No PCA-Informed Init for Hierarchical (Substantive Gap)

**File:** `analysis/07_hierarchical/hierarchical.py`, `build_per_chamber_graph()` + `build_per_chamber_model()`

**Resolved:** PCA-informed initialization was implemented (ADR-0044) and both per-chamber and flat models now use nutpie with PCA init (ADR-0051, ADR-0053). The joint model accepts optional `xi_offset_initvals` for future PCA init.

**Recommendation:** Low priority. The hierarchical structure provides sufficient initialization guidance for most sessions. Test on one failing session before committing.

### 2.8 Issue 7: ICC Credible Interval Labeled as HDI (Code Quality)

**File:** `analysis/07_hierarchical/hierarchical.py`, `compute_variance_decomposition()`, line 781

```python
icc_lower, icc_upper = np.percentile(icc_samples, [2.5, 97.5])
```

The result columns are named `icc_hdi_2.5` and `icc_hdi_97.5`, implying a highest-density interval. But the computation uses `np.percentile` (equal-tailed interval), while all other posterior summaries in the module use `az.hdi()`. For symmetric posteriors the distinction is negligible; for skewed ICC distributions near 0 or 1 the difference could matter.

**Recommendation:** Either rename to `icc_ci_*` or switch to `az.hdi()` for consistency.

### 2.9 Issue 8: `extract_group_params` Crashes on Joint Model Data (Code Defect)

**File:** `analysis/07_hierarchical/hierarchical.py`, `extract_group_params()`, line 712

The function hardcodes `mu_party` as the variable to extract:

```python
mu_post = idata.posterior["mu_party"]
```

The joint model uses `mu_group` (4 groups: House-D, House-R, Senate-D, Senate-R) instead of `mu_party` (2 groups). Calling `extract_group_params` on joint model InferenceData raises a `KeyError`. This is both a code defect (no graceful handling) and a test gap (no coverage).

**Recommendation:** Either dispatch on model type or raise a clear error: `raise ValueError("extract_group_params only supports per-chamber models (mu_party)")`.

### 2.10 Issue 9: Shrinkage Scatter Mixed Scale Methodology (Code Quality)

**File:** `analysis/07_hierarchical/hierarchical.py`, `plot_shrinkage_scatter()`, lines 978-1008

The scatter plot axes use raw flat and hierarchical ideal points (different scales), but the "top 5 movers" annotation labels at lines 1001-1008 are ranked by `delta_from_flat`, which uses the scale-corrected `flat_xi_rescaled`. This means the point positions are raw-scale but the labels highlight scale-corrected outliers — an internal inconsistency within the same plot.

**Recommendation:** Fix alongside Issue 2 (use rescaled flat values for the x-axis).

---

## 3. Dead Code and Refactoring

### 3.1 No Dead Code Found

The audit found no unused imports, no commented-out code, no TODO/FIXME/HACK comments, no unused functions, and no unused constants. The codebase is clean.

### 3.2 Internal Documentation Inaccuracies

Two minor docstring/primer inaccuracies were found:

1. **`compute_variance_decomposition` docstring** (line 753) says `sigma_within_pooled is the mean of the per-party sigma_within`, but the actual computation (lines 771-773) is a *weighted* mean (weighted by group size). The docstring should say "group-size-weighted mean."

2. **`HIERARCHICAL_PRIMER` string** (lines 84-184) describes the joint model formula as `mu_group = mu_chamber[c] + sigma_party * offset_party` but omits the ordering constraint (`group_offset_sorted` is the sorted version of `group_offset_raw`). Minor omission in the inline primer.

### 3.3 Refactoring Opportunity: Extract Shrinkage Comparison

`extract_hierarchical_ideal_points()` (lines 588-701, 113 lines) mixes three concerns:

1. **Posterior summary extraction** (lines 599-626): Mean, SD, HDI from ArviZ
2. **Metadata join** (lines 630-632): Legislator name/party/district
3. **Shrinkage comparison** (lines 635-689): Rescaling, delta, toward_party_mean, shrinkage_pct

Concern #3 is 54 lines of intricate rescaling logic that could be a standalone function:

```python
def compute_shrinkage_comparison(
    hier_ip: pl.DataFrame,
    flat_ip: pl.DataFrame,
) -> pl.DataFrame:
    """Compare hierarchical and flat IRT ideal points, computing shrinkage metrics."""
    ...
```

**Benefits:** Independently testable, reusable for cross-session comparison.

**Recommendation:** Medium priority. Worth doing when cross-session integration (Section 8.3) is implemented.

---

## 4. Test Gaps

The existing 26 tests cover data preparation (7), model structure (4), result extraction (5), group params (3), variance decomposition (4), and flat-vs-hier comparison (3). All pass. The test infrastructure (`_make_fake_idata()`) creates realistic synthetic posteriors without MCMC sampling — appropriate for unit tests.

### 4.1 No Test for Small-Group Warning

No test verifies a warning when a party group is small. This is the most important gap because it's tied to the most serious known limitation.

```python
def test_small_group_warning(
    self, house_matrix: pl.DataFrame, legislators: pl.DataFrame, capsys
) -> None:
    """Small party groups should trigger a warning."""
    small_matrix = house_matrix.filter(
        pl.col("legislator_slug").is_in([
            "rep_a_a_1", "rep_b_b_1", "rep_c_c_1",  # 3 R
            "rep_f_f_1", "rep_g_g_1",                 # 2 D
        ])
    )
    prepare_hierarchical_data(small_matrix, legislators, "House")
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
```

### 4.2 No Test for Joint Model Ordering Constraint

The per-chamber ordering constraint is tested, but the joint model's per-chamber-pair sorting (the 2026-02-23 label-switching fix) has no dedicated test.

```python
def test_joint_ordering_per_chamber(self) -> None:
    """Joint model should sort each chamber's pair independently."""
    raw = np.array([1.5, -0.5, 0.8, -1.2])  # Unsorted pairs
    house_pair = pt.sort(pt.as_tensor_variable(raw[:2])).eval()
    senate_pair = pt.sort(pt.as_tensor_variable(raw[2:])).eval()
    assert house_pair[0] < house_pair[1]
    assert senate_pair[0] < senate_pair[1]
```

### 4.3 No Test for Shrinkage Rescaling Fallback

When fewer than 3 legislators match, the rescaling falls back to `slope=1.0` with a warning. No test covers this path.

### 4.4 No Test for Variance Decomposition with Highly Unequal Groups

All ICC test fixtures use 5R/3D. Kansas data has ~90R/30D (House) and ~30R/10D (Senate). A test with highly unequal groups would verify the weighted pooling.

### 4.5 No Test for `extract_group_params` with Joint Model Data

`extract_group_params` reads `mu_party` (2 parties) but the joint model uses `mu_group` (4 groups). The function is silently inapplicable to joint results, and no test verifies this boundary. See also Issue 8.

### 4.6 Defective Assertion in `test_correlation_handles_missing`

**File:** `tests/test_hierarchical.py`, line 560

```python
assert not np.isnan(r) or True
```

This is a tautology — it always evaluates to `True` regardless of the value of `r`. The test description says "With < 3 overlap, nan is acceptable," but the assertion never actually tests anything. If the code incorrectly returned a non-NaN value with only 2 overlap points, this test would still pass.

**Recommendation:** Replace with `assert np.isnan(r)` if NaN is the expected result, or `assert np.isnan(r) or isinstance(r, float)` if both NaN and a valid float are acceptable.

### 4.7 No Test for Independent Legislator Exclusion

`prepare_hierarchical_data` (lines 253-261) excludes non-major-party legislators, but no test passes a matrix containing an Independent legislator and verifies they are excluded from the output.

### 4.8 Weak Assertion in `test_shrinkage_toward_party_mean`

**File:** `tests/test_hierarchical.py`, lines 576-577

The assertion `assert non_null.height > 0` only checks that at least one row has a non-null `toward_party_mean` value. It does not verify that the majority of legislators shrink toward their party mean (the expected behavior). A stronger test would verify that `toward_party_mean == True` for most rows.

---

## 5. Comparison to Field Standard

### 5.1 Our Implementation vs idealstan

idealstan (Kubinec 2024, "Generalized Ideal Point Models for Robust Measurement with Dirty Data in the Social Sciences") is the most feature-complete IRT package for legislative analysis. It's R + Stan only.

| Feature | Our Implementation | idealstan | Assessment |
|---------|-------------------|-----------|------------|
| Binary response (Yea/Nay) | Yes | Yes | Parity |
| Ordinal response | No | Yes | Not needed (Kansas data binary) |
| Time-varying ideal points | No | Yes (random walk) | Would need Martin-Quinn extension |
| Informative missingness | No (MAR assumed) | Yes | Low priority (2.6% absence rate) |
| Hierarchical (party pooling) | Yes (2-level + 3-level) | No | **We're ahead** |
| Hard anchors | Yes (flat IRT) | Yes | Parity |
| Ordering constraint | Yes (hierarchical) | No | **We're ahead** |
| PCA-informed init | Yes (flat IRT, ADR-0023) | No | **We're ahead** |
| Non-centered parameterization | Yes | Yes | Parity |
| Cross-chamber equating | Yes (flat IRT) | No | **We're ahead** |
| Joint cross-chamber model | Yes (3-level) | No | **We're ahead** |
| Convergence reporting | Yes (automated R-hat/ESS/BFMI) | Via Stan output | **We're ahead** |
| External validation | Yes (Shor-McCarty) | No | **We're ahead** |
| Variance decomposition (ICC) | Yes | No | **We're ahead** |
| Shrinkage comparison | Yes | No | **We're ahead** |
| HTML report generation | Yes | No | **We're ahead** |
| GPU acceleration | No (could use PyMC+JAX) | No | Parity |
| Variational inference | No | Partial | Not needed (Section 1.4) |

**Summary:** For our specific use case — Kansas binary roll-call votes with party structure — we have the more complete solution. idealstan's advantages (time-varying, informative missingness, ordinal) address problems we don't have.

### 5.2 Our Implementation vs the PyMC Community

Bob Carpenter's April 2025 evaluation of AI-generated IRT code on PyMC Discourse found all attempts failed on identification. He identified five specific failures: no true hierarchy, missing simulated data, unsolved identification, no validation code, and incomplete SBC. His key critique: even advanced LLMs couldn't produce correct PyMC IRT code.

Our implementation handles identification correctly in both models:
- **Flat IRT:** Hard anchors (conservative at +1, liberal at -1) selected via PCA
- **Hierarchical:** Ordering constraint via `pt.sort` (D < R) — a hard, deterministic constraint

This puts our implementation well ahead of what the PyMC community has produced.

---

## 6. Known Limitations and Their Status

### 6.1 J=2 Groups / Small Senate Democrats (CRITICAL, DOCUMENTED)

**Status:** Discovered via external validation. Documented in `docs/hierarchical-shrinkage-deep-dive.md`. No code-level mitigation yet.

**Evidence:** Senate hierarchical r=-0.541 vs Shor-McCarty (inverted), while flat Senate r=0.929. Convergence diagnostics: R-hat 1.83, ESS 3 (catastrophic).

**Root cause:** The James-Stein estimator guarantees shrinkage dominance only for J >= 3 groups (James & Stein 1961). With J=2 parties and ~11 Senate Democrats, the model cannot reliably separate `mu_party[D]` from `sigma_within[D]`. The funnel geometry produces bimodal posteriors despite non-centered parameterization.

**Recommendation:** Issue 1 (add small-group warning) addresses the code-level gap. The deeper fix — group-size-adaptive priors or a minimum-group-size gate — should be considered but is lower priority since the flat model works well for Senate.

### 6.2 Joint Model Fragility (KNOWN, HANDLED)

**Status:** Gracefully handled with try/except and `--skip-joint` flag.

The joint model combines ~170 legislators across ~500+ unique votes with only ~70 shared cross-chamber bills. When it converges, results are excellent (91st: R-hat 1.004, 0 divergences, 93 min). When it doesn't, the exception handler catches it cleanly.

### 6.3 IRT Convergence Failures in Historical Sessions (RESOLVED)

Previously, 5 of 16 chamber-sessions failed convergence with PyMC's default NUTS sampler: 84th House, 85th Senate, 86th House, 87th Senate, 89th House. **All 16/16 now pass** with the nutpie Rust NUTS sampler migration (ADR-0053, 2026-02-28). All sessions show R-hat < 1.01, ESS > 400, zero divergences.

---

## 7. Priors Assessment

### 7.1 Current Priors

| Parameter | Prior | Purpose |
|-----------|-------|---------|
| `mu_party_raw` | `Normal(0, 2)` | Weakly informative on party means |
| `sigma_within` | `HalfNormal(1)` | Regularizes within-party spread toward 0 |
| `xi_offset` | `Normal(0, 1)` | Standard normal for non-centered form |
| `alpha` (difficulty) | `Normal(0, 5)` | Diffuse — allows wide range of difficulty |
| `beta` (discrimination) | `Normal(0, 1)` | Allows positive and negative discrimination |

### 7.2 Assessment

**`mu_party_raw ~ Normal(0, 2)`:** Appropriate. Allows party means up to ~±6, well beyond observed values (typically ±3). Not so diffuse that it fails to regularize.

**`sigma_within ~ HalfNormal(1)`:** The most consequential prior choice. With `sigma=1`, ~95% of the prior mass falls below sigma_within=2. For the Senate Democrat group (~11 legislators), this prior contributes meaningful information — the data alone can't pin down sigma_within with only 11 observations. The hierarchical shrinkage deep dive identified this as a potential lever: tightening to `HalfNormal(0.5)` for small groups would reduce over-shrinkage.

**`beta ~ Normal(0, 1)` (unconstrained):** Unlike some implementations that constrain beta > 0 (all items discriminate in the same direction), our model allows negative discrimination. This is correct for legislative data — some votes separate parties in the expected direction, others inversely. The ordering constraint on `mu_party` handles the global sign. Carpenter (2025) endorsed this approach when combined with hard identification constraints.

**`alpha ~ Normal(0, 5)`:** Diffuse, appropriate for a nuisance parameter. The `sigma=5` is wider than `beta`'s `sigma=1`, reflecting that difficulty has a wider natural range.

---

## 8. Downstream Integration Assessment

### 8.1 Synthesis (Phase 11) — Used

Synthesis reads `hierarchical_ideal_points_{chamber}.parquet` and uses `xi_mean`, `xi_sd`, `shrinkage_pct`, `toward_party_mean`. Integration is optional and graceful.

### 8.2 Profiles (Phase 12) — Not Used

Despite the design doc stating profiles could "show credible intervals alongside point estimates," profiles uses flat IRT only. **Missed opportunity:** flagging legislators whose hierarchical and flat estimates diverge significantly.

### 8.3 Cross-Session (Phase 13) — Not Used

Despite the design doc noting "Cross-session comparison should use posterior means for legislators with few votes in one session," cross-session uses flat IRT only. This is the most significant integration gap — shrinkage is designed for exactly this use case.

### 8.4 External Validation (Phase 14) — Used

External validation compares both flat and hierarchical ideal points against Shor-McCarty scores. This is how the Senate over-shrinkage was discovered.

---

## 9. Summary of Recommendations

### Must Fix (Substantive / Code Defect)

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 1 | Add small-group warning in `prepare_hierarchical_data` | Prevents silent unreliable results for small party groups | ~10 lines + 1 test |
| 8 | `extract_group_params` crashes on joint model data | KeyError on `mu_party` when called with joint InferenceData | ~5 lines (guard or dispatch) |

### Should Fix (Code Quality)

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 2 | Keep `flat_xi_rescaled` in output parquet | Enables better scatter plot + downstream use | ~1 line (remove `.drop()`) |
| 3 | Fix shrinkage scatter to use rescaled flat values | Makes scatter plot visually meaningful for non-technical readers | ~5 lines |
| 4 | Extract `SHRINKAGE_MIN_DISTANCE = 0.5` constant | Eliminates magic number | ~3 lines |
| 5 | Extract convergence variable lists to constants | Minor clarity improvement | ~5 lines |
| 7 | Rename ICC interval columns from `icc_hdi_*` to `icc_ci_*` (or switch to `az.hdi()`) | Naming consistency with other posterior summaries | ~3 lines |
| 9 | Fix scatter plot mixed scale methodology | Axes use raw values but "top 5 movers" use scale-corrected deltas | Fix with Issue 2/3 |

### Should Add / Fix (Tests)

| # | Test | What It Catches |
|---|------|-----------------|
| 10 | Small-group warning test | Regression on Issue 1 |
| 11 | Joint model ordering constraint test | Regression on 2026-02-23 label-switching fix |
| 12 | Shrinkage rescaling fallback test | Regression on silent identity-transform fallback |
| 13 | Highly unequal groups ICC test | Weighted pooling correctness |
| 14 | Joint model group params boundary test | Verifies `extract_group_params` applicability (Issue 8) |
| 15 | Fix tautological assertion in `test_correlation_handles_missing` | `assert not np.isnan(r) or True` always passes — defective test |
| 16 | Independent legislator exclusion test | Verifies non-major-party legislators are correctly excluded |
| 17 | Strengthen `test_shrinkage_toward_party_mean` assertion | Currently only checks `height > 0`, should verify majority shrink toward mean |

### Won't Fix (Correct As-Is)

- **Non-centered parameterization:** Standard best practice (Papaspiliopoulos et al. 2007). Centered would be slightly more efficient for House Republicans (N~90) but the safety margin is worth it.
- **Sampler choice:** All models now use nutpie Rust NUTS (ADR-0051, ADR-0053). NumPyro migration not warranted.
- **No PCA-informed init for hierarchical:** Party structure provides sufficient initialization guidance.
- **ICC with `ddof=0`:** Correct — 2 parties is the population, not a sample.
- **`PARTY_COLORS["Independent"]`** imported but unused in this phase: Intentional (pulled in with other IRT constants).

### Deferred (Future Work)

- **Group-size-adaptive priors:** Scale `sigma_within` prior tightness by group size. Promising but needs experimentation.
- **Tail-ESS diagnostics:** Complement bulk-ESS with tail-ESS per Vehtari et al. (2021). Flagged in IRT deep dive too.
- **Cross-session integration:** Use hierarchical posterior means for returning legislators with few votes.
- **Dynamic IRT extension:** Martin-Quinn random-walk priors for time-varying ideal points across sessions. This would be a substantial new phase. No Python implementation exists — emIRT (R) uses variational EM, idealstan (R+Stan) uses HMC.

---

## 10. Key References

- Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
- Bolstad, J. (2024). "Hierarchical Bayesian Aldrich-McKelvey Scaling." *Political Analysis*, 32(1), 50-64.
- Clinton, J., Jackman, S., & Rivers, D. (2004). "The Statistical Analysis of Roll Call Data." *American Political Science Review*, 98(2), 355-370.
- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press. Chapter 5 (hierarchical models).
- Imai, K., Lo, J., & Olmsted, J. (2016). "Fast Estimation of Ideal Points with Massive Data." *American Political Science Review*, 110(4), 631-656.
- James, W. & Stein, C. (1961). "Estimation with Quadratic Loss." *Proceedings of the Fourth Berkeley Symposium*, 1, 361-379.
- Kubinec, R. (2024). "Generalized Ideal Point Models for Robust Measurement with Dirty Data in the Social Sciences."
- Lalor, J.P. (2022). "py-irt: A Scalable Item Response Theory Library for Python." *INFORMS Journal on Computing*, 35(1).
- Martin, A.D. & Quinn, K.M. (2002). "Dynamic Ideal Point Estimation via Markov Chain Monte Carlo for the U.S. Supreme Court, 1953-1999." *Political Analysis*, 10(2), 134-153.
- Nishio, M. et al. (2023). "Comparison between PyStan and NumPyro in Bayesian Item Response Theory." *PeerJ Computer Science*, 9, e1620.
- Papaspiliopoulos, O., Roberts, G.O., & Skold, M. (2007). "A General Framework for the Parametrization of Hierarchical Models." *Statistical Science*, 22(1), 59-73.
- Shin, S. (2024). "L1-based Bayesian Ideal Point Model for Multidimensional Politics." *JASA*, 120(550), 631-644.
- Vehtari, A. et al. (2021). "Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC." *Bayesian Analysis*, 16(2), 667-718.
- Wu, M. et al. (2020). "Variational Item Response Theory: Fast, Accurate, and Expressive." arXiv:2002.00276.
- Zhou, X. (2019). "Hierarchical Item Response Models for Analyzing Public Opinion." *Political Analysis*, 27(4), 481-502.
