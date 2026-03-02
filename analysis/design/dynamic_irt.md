# Dynamic Ideal Point Estimation Design Choices

**Script:** `analysis/16_dynamic_irt/dynamic_irt.py`
**Data module:** `analysis/16_dynamic_irt/dynamic_irt_data.py`
**Constants defined at:** top of each module

## Assumptions

1. **Legislators are matched by normalized name** across bienniums. Leadership suffixes stripped. Same matching logic as cross-session validation (Phase 13).

2. **Chambers analyzed separately**, consistent with all upstream phases. House and Senate are independent models.

3. **Random walk evolution is Markovian**: ideal points evolve via `xi[t] = xi[t-1] + tau * innovation`. No mean-reversion or drift term.

4. **Bridge legislators anchor cross-biennium scale**: legislators serving multiple bienniums connect the ideal point scales across time. No explicit anchor constraints needed — the random walk prior + shared likelihood provides identification.

5. **Positive beta provides sign identification**: with `HalfNormal(2.5)` on discrimination, all beta > 0, fixing the direction of the latent scale. Higher xi = more conservative. **Caveat:** positive beta alone is insufficient when the random walk chain is broken (e.g., missing biennium data creating 0 bridge legislators). A post-hoc sign correction step (ADR-0068) validates each period against static IRT and negates xi if the correlation is negative.

6. **Absent periods are interpolated, not estimated**: for legislators not serving in a biennium, the random walk prior carries their ideal point forward. The posterior is wide (prior-dominated) — this is correct behavior, not a bug.

## Parameters & Constants

| Constant | Value | Justification |
|----------|-------|---------------|
| `DEFAULT_N_SAMPLES` | 1000 | Sufficient for posterior summaries. Increase for publication-quality HDIs. |
| `DEFAULT_N_TUNE` | 1000 | Standard tuning length. May need 2000 for convergence. |
| `DEFAULT_N_CHAINS` | 2 | Memory constraint. 4 chains preferred if system allows. |
| `RANDOM_SEED` | 42 | Reproducibility. Same as all other phases. |
| `RHAT_THRESHOLD` | 1.05 | Relaxed from standard 1.01 — dynamic IRT has ~10K params. |
| `ESS_THRESHOLD` | 400 | Standard. Same as flat and hierarchical IRT. |
| `MAX_DIVERGENCES` | 50 | Higher tolerance for large state-space model. |
| `TOP_MOVERS_N` | 20 | Number of top movers to display. |
| `MIN_BRIDGE_OVERLAP` | 5 | Minimum shared legislators for a valid bridge. |
| `PARTY_COLORS` | R=#E81B23, D=#0015BC | Consistent with all prior phases. |

## Model Specification

### State-Space 2PL IRT (Non-Centered Random Walk)

```
tau ~ HalfNormal(0.5, dims="party")
xi_init ~ Normal(0, 1, dims="legislator")
xi_innovations ~ Normal(0, 1, shape=(T-1, N_leg))

xi[0] = xi_init
xi[t] = xi[t-1] + tau[party_idx] * xi_innovations[t-1]

alpha ~ Normal(0, 5, dims="bill")       # per-biennium bills
beta ~ HalfNormal(2.5, dims="bill")     # positive for sign ID

eta = beta[bill_idx] * xi[time_idx, leg_idx] - alpha[bill_idx]
obs ~ Bernoulli(logit_p=eta)
```

### Non-Centered Parameterization

The random walk is implemented manually rather than using `GaussianRandomWalk`:
- Allows per-party `tau` broadcasting to each legislator
- Consistent with hierarchical IRT's `xi_offset` pattern
- Each innovation is individually addressable for diagnostics

### Scale

| Chamber | Params (approx) | Explanation |
|---------|-----------------|-------------|
| House | ~10,000 | 250 legs × 8 times (xi) + ~8,000 bills (alpha, beta) + 2 tau |
| Senate | ~6,000 | 80 legs × 8 times + ~5,000 bills + 2 tau |

### PCA-Informed Initialization

First biennium's PCA PC1 scores initialize `xi_init`. Standardized and oriented so Republicans are positive. All other variables jittered (including `xi_innovations`, `tau`, `alpha`, `beta`). Critical: do NOT set `jitter_rvs=set()` — HalfNormal variables need jitter to avoid support boundary (`log(0) = -inf`).

## Post-Hoc Sign Correction

After sampling, each period's dynamic xi is correlated with the corresponding static IRT (Phase 04). If the Pearson r is negative, xi for that period is negated across all chains and draws.

**When it triggers:** r < 0 with static IRT for a given period. This indicates the sampler found a sign-flipped mode.

**Root cause:** Positive beta is insufficient for sign identification when the random walk chain is broken — e.g., missing biennium data creates 0 bridge legislators on both sides of a gap, severing the Markov chain. The Senate is less susceptible due to 4-year staggered terms providing higher continuity.

**What it does:** `xi_post[:, :, t, :] *= -1` for the affected period. The corrected InferenceData is saved to NetCDF before any post-processing, so all downstream quantities (trajectories, decomposition, movers, tau extraction, correlation tables) use the corrected posterior.

**Transparency:** Every correction is documented in the HTML report with named reference legislators (the 3 highest-|xi| matches showing dynamic vs static values). A dedicated Model Priors section displays all priors and tuning parameters.

**Precedent:** Mirrors `fix_joint_sign_convention()` in hierarchical IRT (ADR-0042). Documented case: 87th House sign flip in run 260301.1 (r = -0.937 with static IRT) caused by missing 88th biennium data.

## Post-Processing

### Polarization Decomposition

For adjacent biennium pairs (t, t+1), per party:

```
total_shift = mean(xi[t+1]) - mean(xi[t])    # party mean shift

returning = legislators serving both t and t+1
conversion = mean(xi_returning[t+1]) - mean(xi_returning[t])

replacement = total_shift - conversion
```

This decomposes polarization into:
- **Conversion**: existing members moving
- **Replacement**: new members are more extreme than departing ones

### Top Movers

```
total_movement = sum(|xi[t+1] - xi[t]|)  for all consecutive served periods
net_movement = xi[last] - xi[first]
```

Total captures oscillation; net captures directional drift.

## Data Pipeline

1. **Session enumeration**: 8 bienniums (`"2011-12"` through `"2025-26"`)
2. **Load upstream**: per-biennium EDA matrices, PCA scores, legislator CSVs
3. **Prepare IRT data**: reuses `prepare_irt_data()` from Phase 04
4. **Build global roster**: name-match across bienniums, assign global indices
5. **Stack bienniums**: remap local indices to global, offset bill indices
6. **Bridge coverage**: count shared legislators between adjacent periods
7. **Build model**: state-space 2PL IRT graph
8. **Sample**: nutpie Rust NUTS, PCA-informed init
9. **Load static IRT**: per-biennium ideal points from Phase 04
10. **Sign correction**: compare dynamic xi with static IRT, negate if r < 0 (ADR-0068)
11. **Post-process**: trajectories, decomposition, top movers, static correlation
12. **Report**: ~16-section HTML report (includes sign corrections and model priors)

## Comparison with Phase 04 (Static IRT)

| Feature | Phase 04 (Static) | Phase 16 (Dynamic) |
|---------|-------------------|---------------------|
| Scope | Single biennium | All 8 bienniums jointly |
| Identification | Anchor constraints (±1 on PCA extremes) | Positive beta (HalfNormal) |
| Sign correction | Anchors fix sign at model level | Post-hoc correlation check (ADR-0068) |
| Temporal | None | Random walk across bienniums |
| Scale linking | Not needed | Bridge legislators |
| Runtime | ~10 min/chamber | ~1–3 hours/chamber |
| Output | Point + interval per legislator | Trajectory per legislator across time |

## emIRT Exploration Tier

Optional: `emIRT::dynIRT` in R provides fast EM point estimates. Used purely for validation (scatter plot vs PyMC). Auto-skipped if R or emIRT is unavailable. CLI flag: `--skip-emirt`.

## Known Limitations

1. **84th biennium data quality**: ~30% committee-of-the-whole votes (tally-only, no individual votes). Ideal points for 84th-only legislators will have wide posteriors.
2. **84th→85th bridge weakness**: post-2012 redistricting reduced legislator overlap.
3. **Independent party**: excluded from party-level tau. Their evolution uses a default party assignment. With ≤2 Independents in any biennium, this has minimal impact.
4. **Absent periods**: random walk interpolation produces uncertain but not absent estimates. Report clearly marks which periods each legislator actually served.
5. **Sign flip with broken chain**: When biennium data is missing, 0 bridge legislators sever the Markov chain, enabling sign flips in isolated periods. Documented case: 87th House in run 260301.1 (r = -0.937). Post-hoc correction (ADR-0068) handles this, but the root fix is ensuring all biennium data is available before running.
6. **Scale drift from sign flip**: A sign flip inflates tau to absorb artificial ±3-unit jumps at the boundary, causing the dynamic/static range ratio to grow (e.g., 0.66x at 84th to 1.51x at 91st). Re-running with complete data should eliminate both the flip and the drift.
