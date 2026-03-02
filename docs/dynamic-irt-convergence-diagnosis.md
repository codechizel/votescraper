# Dynamic IRT Convergence Diagnosis

Research into three issues surfaced during the full 8-biennium pipeline run (2026-03-01): the persistent 87th sign flip, Senate convergence failure, and `latest` symlink race condition. Findings inform the roadmap for Phase 16 improvements.

## Table of Contents

1. [87th Sign Flip: Root Cause and Identification Strategies](#87th-sign-flip)
2. [Senate Convergence Failure: Diagnosis and Fixes](#senate-convergence-failure)
3. [Symlink Race Condition: Engineering Fix](#symlink-race-condition)
4. [Prioritized Recommendations](#prioritized-recommendations)

---

## 87th Sign Flip

### Problem Statement

The 87th biennium (2017-18) ideal points are inverted relative to static IRT. This persists across both dynamic IRT runs:

| Run | 88th Data? | House r | Senate r |
|-----|-----------|---------|----------|
| 260301.1 | Missing | -0.937 | unaffected |
| 260301.2 | Present (94 bridges, 74%) | -0.959 | -0.657 |

The sign flip is **intrinsic to the 86th→87th transition**, not a data gap artifact.

### Why It Happens

In static IRT (Phase 04), sign identification uses **hard anchors**: two legislators are fixed at +1 and -1, completely resolving the reflection invariance. The dynamic model switched to **soft identification** via `HalfNormal(2.5)` on beta (bill discrimination). This is weaker because:

1. Positive beta forces all bills' discrimination positive, but the random walk can still find a reflected mode for *ideal points* at specific time periods
2. The innovation term `xi[t] = xi[t-1] + tau * innovation` is symmetric around zero — if the sampler explores a mode where xi at period t is negated, the innovations absorb the sign difference
3. Bridge legislators constrain adjacent periods to have the same sign, but this constraint is probabilistic (not hard), and it fails when the likelihood landscape has a competing mode

### Literature Review

| Source | Identification Strategy |
|--------|------------------------|
| Martin & Quinn (2002) | `theta.constraints`: fix known justices' signs (Stevens < 0, Thomas > 0) |
| Clinton, Jackman & Rivers (2004) | Hard anchors: fix two legislators at ±1 (static IRT gold standard) |
| Bailey (2007) | Bridge observations for cross-institutional comparison; still needs sign constraints |
| emIRT / Imai et al. (2016) | Informative priors on initial ideal points (`x.mu0`, `x.sigma0`); EM cannot mode-switch |
| idealstan / Kubinec (2019) | `restrict_ind_high`/`restrict_ind_low`, item-based identification ("very useful with time-varying models") |
| Erosheva & Bhattacharyya (2017) | Post-hoc relabeling; warns that positivity constraints can produce "loadings with opposite polarity" |

**Key insight:** Every established implementation uses either hard anchor constraints or informative priors on initial ideal points. Our `HalfNormal(2.5)` beta-only approach is the weakest identification strategy in the literature.

### Candidate Fixes

#### Strategy A: Informative Priors from Static IRT (HIGH impact, LOW effort)

Use Phase 04 static IRT posterior means as informative priors for `xi_init`:

```python
# Current: uninformative
xi_init = pm.Normal("xi_init", mu=0, sigma=1, shape=n_leg)

# Fix: transfer static IRT sign convention
static_xi = load_static_irt_means(biennium=0)
xi_init = pm.Normal("xi_init", mu=static_xi, sigma=0.75, shape=n_leg)
```

This is essentially what emIRT does with `x.mu0` / `x.sigma0`. The static IRT uses hard anchors and has no sign ambiguity. By propagating its posterior into the dynamic model's prior, the correct sign convention flows through the entire random walk. We already load static IRT results in `fix_period_sign_flips()` — moving this upstream from post-hoc correction to model construction is minimal code change.

#### Strategy B: Anchor Legislator Sign Constraints (HIGH impact, MODERATE effort)

Add truncated normal priors on 2 known legislators per chamber:

```python
# Conservative anchor: truncated to positive
xi_init_cons = pm.TruncatedNormal("xi_init_cons", mu=1.5, sigma=1, lower=0)
# Liberal anchor: truncated to negative
xi_init_lib = pm.TruncatedNormal("xi_init_lib", mu=-1.5, sigma=1, upper=0)
```

This mirrors MCMCpack's `theta.constraints`. Select the longest-serving legislator from each party with the most extreme PCA PC1 score. The `build_global_roster()` function already has `n_periods` per legislator.

#### Strategy C: Item-Based Identification (HIGH impact, MODERATE effort)

Fix the discrimination of 2-4 known partisan bills per biennium (e.g., clean party-line votes or veto overrides). idealstan explicitly recommends this for dynamic models: "very useful with time-varying models where fixing person ideal points can be very challenging."

#### Strategy D: Continuity Penalty (MEDIUM impact, LOW effort)

Tighten the tau prior from `HalfNormal(0.5)` to `HalfNormal(0.2)` to penalize large sign-reversing jumps. This is a bias-variance trade-off — tighter tau suppresses genuine drift.

### Current Mitigation

Post-hoc per-period sign correction (ADR-0068) handles this robustly for ideal points. However, it cannot undo tau inflation caused by the sign flip during sampling — the corrected posterior is not a true posterior from a correctly-identified model.

### Recommendation

**Strategy A (informative priors from static IRT)** is the highest-impact, lowest-effort fix. Combined with Strategy B (anchor constraints) as belt-and-suspenders, this should eliminate sign flips entirely. Post-hoc correction (ADR-0068) should be retained as a diagnostic — if it triggers, it means in-model identification failed.

---

## Senate Convergence Failure

### Problem Statement

Dynamic IRT Senate (Run 260301.2): R-hat 1.84, ESS 3, 0 divergences. Democrat tau CI: [0.1023, 1.9584]. House converges well (R-hat 1.02, ESS 789, 0 divergences).

### Diagnosis: Mode-Splitting

The pattern **high R-hat + very low ESS + zero divergences** is the textbook signature of mode-splitting. Each chain samples coherently within its own mode (no divergences), but the two chains are in different modes (R-hat 1.84, ESS ≈ number of modes explored).

The causal chain:
1. 87th-88th sign flip creates an artificial ~3-unit jump in ideal points
2. One chain absorbs this with large tau; the other sees no flip and estimates small tau
3. Democrat tau (estimated from ~10-12 Democrats × 7 transitions ≈ 70-84 data points) is too weakly identified to resolve the disagreement
4. The tau disagreement propagates to all xi estimates through the random walk

### Why the Senate Fails but the House Doesn't

| Factor | House | Senate | Impact |
|--------|-------|--------|--------|
| Legislators per biennium | 113-130 | 37-42 | 3× less data per biennium |
| Total unique legislators | 381 | 112 | 3.4× fewer trajectories |
| Bridge coverage (min) | 58 (51%) | 21 (57%) | Thinner at worst case |
| Obs/param ratio | ~17:1 | ~12:1 | Lower information density |

The 2PL IRT literature recommends ≥500 respondents for standard calibration. At 37-42, the Senate is ~10× below. The Bayesian approach can work with smaller samples, but the priors do much more work and posterior geometry becomes pathological.

### Literature Context

Martin & Quinn (2002) designed the model for SCOTUS (9 justices but 47+ terms and ~600+ cases/term). For state legislatures, no published dynamic ideal point analysis exists for chambers this small. The critical difference: all three established implementations (MCMCpack, emIRT, idealstan) either fix tau or provide mechanisms to strongly constrain it. Our `HalfNormal(0.5)` per-party tau is the loosest specification in the literature.

### Prioritized Fixes

#### Fix 1: Tighten Tau Prior (HIGH impact, LOW effort)

```python
# Current: allows tau up to ~1.5 (97.5th percentile)
tau = pm.HalfNormal("tau", sigma=0.5, shape=n_parties)

# Fix: chamber-adaptive prior
tau_sigma = 0.15 if n_leg < 80 else 0.5
tau = pm.HalfNormal("tau", sigma=tau_sigma, shape=n_parties)
```

Legislator ideal point drift per biennium is typically 0.1-0.3 SD in the literature. `HalfNormal(0.15)` has a 97.5th percentile of ~0.45, still allowing moderate drift but making sign-flip jumps (~3 units) essentially impossible.

#### Fix 2: Global Tau for Senate (HIGH impact, LOW effort)

Switch from per-party to global tau for small chambers:

```python
if n_leg < 80 and evolution_structure == "per_party":
    evolution_structure = "global"
```

Instead of ~70 Democrat observations constraining `tau_Democrat`, all ~200+ observations constrain a single tau. The `--evolution global` CLI flag already exists.

#### Fix 3: Increase to 4 Chains (MODERATE impact, LOW effort)

The hierarchical IRT 4-chain experiment showed +4.6% wall time with 42-83% ESS improvement on the M3 Pro. With 4 chains, mode-splitting is easier to diagnose (3-vs-1 pattern) and less likely to be symmetric.

#### Fix 4: Increase Draws to 2000/2000 (MODERATE impact, LOW effort)

More warmup for mass matrix adaptation, more draws for ESS. Only effective after fixing the mode-splitting root cause (Fixes 1-2). Without those, 10,000 draws per chain still produces ESS ≈ 2.

#### Fix 5: emIRT Initialization (HIGH impact, MODERATE effort)

Run `emIRT::dynIRT` as preprocessing (infrastructure already exists in `dynamic_irt_data.py` lines 432-458). Use emIRT's point estimates to initialize ALL xi periods:

```python
emirt_xi = load_emirt_results(emirt_output_path)
# Initialize xi_innovations from differences: (xi[t] - xi[t-1]) / tau_init
```

emIRT runs in 2-5 minutes and produces r = 0.93-0.96 vs full MCMC on SCOTUS data. Starting all chains from emIRT's solution prevents mode exploration. Requires R subprocess (already used in Phase 15 TSA and Phase 17 W-NOMINATE).

#### Fix 6: nutpie Low-Rank Mass Matrix (MODERATE impact, LOW effort)

```python
idata = nutpie.sample(
    compiled,
    low_rank_modified_mass_matrix=True,
    mass_matrix_eigval_cutoff=3,
    mass_matrix_gamma=1e-5,
)
```

Captures correlations between tau and xi_innovations that the diagonal mass matrix misses.

#### Fix 7: Fixed Tau / Last Resort (DEFINITIVE impact)

If tau cannot be reliably estimated, fix it to a constant calibrated from House results:

```python
tau_fixed = 0.1  # from House posterior
tau_leg = pt.ones(n_leg) * tau_fixed
```

This follows the Martin-Quinn (2002) approach of a priori specified evolution variances. Eliminates mode-splitting entirely but requires sensitivity analysis.

### Recommended Experiment Plan

| Phase | Fixes | Effort | Expected Outcome |
|-------|-------|--------|-----------------|
| A: Quick fixes | Tau prior (0.15) + global tau + 4 chains + 2000/2000 | 2 hours | Likely resolves convergence |
| B: emIRT init | Add emIRT preprocessing | 4-6 hours | Anchors chains in correct mode |
| C: Sampler tuning | Low-rank mass matrix, target_accept=0.95 | 2-4 hours | Better mixing |
| D: Model simplification | Fixed tau, explicit anchor constraints | 2 hours | Guaranteed convergence |

Try Phase A first. If R-hat drops below 1.05 and ESS exceeds 400, stop.

---

## Symlink Race Condition

### Problem Statement

Independent analyses (DIME, external-validation) overwrite the pipeline's `latest` symlink when run against individual bienniums. Dynamic IRT then reads `latest/01_eda/data/` from the wrong run directory (one without EDA data), silently skipping the biennium.

### Root Cause

`RunContext.finalize()` (line 413-427) unconditionally updates the session-level `latest` symlink for any successful biennium run. When standalone phases auto-generate a run_id (because no `--run-id` was passed), they create a new run directory and overwrite `latest`.

The bug sequence:
1. `just pipeline 2019-20` → `latest → 88-260301.1` (complete pipeline data)
2. `just dime --all-sessions` → auto-generates `88-260301.3` → `latest → 88-260301.3` (DIME only)
3. `just dynamic-irt` reads `latest/01_eda/data/` → `88-260301.3/01_eda/` → not found → skip

### Design Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A: `--no-latest` flag | New parameter on `RunContext` | Minimal change | Fails open; each consumer must remember |
| B: Separate `latest-pipeline` | Second symlink for pipeline runs | Fully separates concerns | Invasive; must audit every `latest` reader |
| C: Explicit run IDs | Pass `--run-id` to dynamic IRT | Most precise | Terrible ergonomics (8 run IDs) |
| D: Lock file | Atomicity mechanism | Eliminates TOCTOU | Wrong abstraction (semantic problem, not race) |
| **E: Pipeline-only updates** | Only explicit `--run-id` updates `latest` | Automatic, minimal change | Standalone re-runs don't become `latest` (desirable) |

### Recommended Fix: Option E (Pipeline-Only Symlink Updates)

Distinguish between pipeline runs (explicit `--run-id` from `just pipeline`) and standalone runs (auto-generated run_id). Only the former updates `latest`.

**Change 1:** Track whether run_id was explicitly provided in `RunContext.__init__()`:

```python
self._explicit_run_id = run_id is not None
```

**Change 2:** Guard symlink update in `finalize()`:

```python
if not failed:
    if self.run_id is not None and self._explicit_run_id:
        # Pipeline run: update session-level latest
        latest = self._session_root / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(self.run_id)
    elif self.run_id is None:
        # Flat mode (cross-session): update phase-level latest
        latest = self._analysis_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(self._run_label)
    # else: auto-generated run_id — do NOT update latest
```

**What doesn't change:**
- No consumer phases need modification — `resolve_upstream_dir()` still follows `latest`
- No `Justfile` changes — pipeline recipe already passes explicit `--run-id`
- Cross-session / special sessions unchanged (flat mode)
- If someone explicitly runs `just eda --run-id 91-260301.1`, the explicit `--run-id` triggers an update (correct behavior)

**Edge case:** Standalone re-runs (e.g., `just eda` without `--run-id`) no longer update `latest`. This is desirable — if you re-run EDA without the full pipeline, downstream phases would see new EDA data but stale IRT data, which is equally broken.

This warrants a new ADR documenting the decision.

---

## Prioritized Recommendations

### Summary Table

| # | Fix | Issue | Impact | Effort | Dependencies |
|---|-----|-------|--------|--------|-------------|
| 1 | Symlink race: pipeline-only `latest` updates | Symlink | HIGH | 30 min | None |
| 2 | Informative xi_init priors from static IRT | Sign flip | HIGH | 2 hours | None |
| 3 | Tighter tau prior for small chambers | Senate convergence | HIGH | 30 min | None |
| 4 | Global tau for Senate | Senate convergence | HIGH | 15 min | None |
| 5 | Anchor legislator sign constraints | Sign flip | HIGH | 4 hours | #2 |
| 6 | 4 chains + 2000/2000 draws | Senate convergence | MODERATE | 15 min | None |
| 7 | nutpie low-rank mass matrix | Senate convergence | MODERATE | 30 min | None |
| 8 | emIRT initialization | Both | HIGH | 6 hours | R subprocess |
| 9 | Fixed tau as last resort | Senate convergence | DEFINITIVE | 1 hour | Sensitivity analysis |
| 10 | Item-based identification | Sign flip | HIGH | 4 hours | Anchor bill selection |

### Immediate Wins (do first)

Fixes 1, 3, 4, 6: combined effort ~1.5 hours. The symlink fix prevents the race condition from ever recurring. Tighter tau + global tau + 4 chains addresses the Senate convergence root cause.

### High-Value Next Steps

Fixes 2, 5: combined effort ~6 hours. Informative priors + anchor constraints should eliminate sign flips entirely, making post-hoc correction a diagnostic-only safety net.

### Optional / Research

Fixes 7, 8, 10: for experimentation if the immediate fixes are insufficient.
