# Why the Joint Hierarchical IRT Model Produces Distorted Cross-Chamber Estimates

**Date:** 2026-02-26
**Status:** Fixed (ADR-0043) — bill-matching + group-size-adaptive priors implemented

## Terminology

This document discusses two distinct IRT models in the tallgrass pipeline:

- **Flat IRT** (Phase 04, `analysis/05_irt/irt.py`): Standard 2PL Bayesian IRT estimated per chamber, with test equating to place both chambers on a common scale. This is the canonical baseline model.
- **Hierarchical IRT** (Phase 10, `analysis/07_hierarchical/hierarchical.py`): Extends the flat IRT with partial pooling by party. Produces three model variants:
  - **Per-chamber hierarchical**: House-only and Senate-only models. These work correctly.
  - **Joint hierarchical**: A single model combining both chambers. **This is the broken model.**

## The Symptom

When running the joint hierarchical IRT model on Kansas Legislature data, Senate Republicans consistently appear much more moderate than House Republicans — not just slightly, but dramatically. In the 91st Legislature (2025-26):

- The joint hierarchical places the most conservative House member (Barrett) at +9.75 and the most conservative Senator (Tyson) at +8.29, even though Tyson is widely known as the most ideologically extreme member of the entire legislature.
- 31 of 32 Senate Republicans fall below the House Republican median.
- The mean Senate Republican ideal point (+3.84) is barely half the mean House Republican ideal point (+7.15).

The flat IRT, which estimates chambers separately and then rescales via test equating, produces a sensible picture: House Rs at +2.01 and Senate Rs at +1.81 — roughly comparable, with Tyson as the clear overall outlier at +4.43.

## Root Cause: The Joint Hierarchical Discards Shared Bills

Kansas is a bicameral legislature. A bill must pass both the House and the Senate in identical form before the governor can sign it. This means many bills are voted on by both chambers — they are the same legislation, producing the same law. In the 91st session, **174 bill numbers** appear in both chambers' roll call records.

These shared bills are natural bridging items: if both chambers vote on the same bill, and the bill discriminates between liberal and conservative legislators the same way in both chambers, then the bill's discrimination parameter (`beta`) anchors the two chambers to a common ideological scale. This is exactly how every successful cross-chamber scaling method works — from Poole and Rosenthal's Common Space scores to the flat IRT's test equating.

The problem is that `build_joint_model()` in `hierarchical.py` **throws away this bridging information** by combining data using `vote_id` instead of `bill_number`.

### The bug (hierarchical.py, line 418)

```python
all_vote_ids = list(dict.fromkeys(house_data["vote_ids"] + senate_data["vote_ids"]))
```

A `vote_id` is a unique identifier for each roll call event (e.g., `je_20250320203513`). When the House votes on HB 2001 and the Senate later votes on that same HB 2001, they produce two different vote_ids — because they are two separate roll call events, even though they are voting on identical legislation.

Deduplicating by `vote_id` treats every roll call as a unique item. For the 91st session: 297 House vote_ids + 194 Senate vote_ids = **491 total, zero shared**. Each gets its own `alpha` and `beta` parameters. Since no `beta` is shared between chambers, no bill parameter constrains the relative scale. The two chambers' likelihoods are **completely separable**.

### How the flat IRT handles this correctly (irt.py, lines 310-508)

The flat IRT's `build_joint_vote_matrix()` does exactly what the joint hierarchical should do. It matches bills across chambers in three steps:

**Step 1: Match by bill_number.** Group each chamber's vote_ids by their `bill_number` from the rollcalls table. Find bill numbers appearing in both chambers:

```python
shared_bills = set(house_bill_vids.keys()) & set(senate_bill_vids.keys())
# 91st: 71 shared bills found (after filtering)
# 90th: 133 shared bills found
```

**Step 2: Pick the best vote_id per chamber.** When a bill has multiple roll calls in a chamber (committee vote, first vote, final action, etc.), the flat IRT prefers the "Final Action" or "Emergency Final Action" motion — the vote most likely to be on identical text in both chambers. If no final action exists, it picks the latest chronologically.

```python
def _pick_best_vid(vids: list[str]) -> str:
    final_vids = [v for v in vids
        if (vid_to_motion.get(v) or "").lower() in ("final action", "emergency final action")]
    candidates = final_vids if final_vids else vids
    return sorted(candidates)[-1]
```

**Step 3: Create shared columns.** For each matched bill, a single column `matched_{bill_number}` is created in the joint vote matrix. House members get their House vote value, Senators get their Senate vote value, and the IRT model estimates a single `beta` for this column. That shared `beta` is the bridge: it must explain both chambers' votes simultaneously, forcing the xi values onto a common scale.

The result:

```
Joint matrix: 169 legislators x 420 votes
  71 matched (shared beta — bridges the chambers)
  226 house-only (House members only, Senate members have NaN)
  123 senate-only (Senate members only, House members have NaN)
```

**Step 4: Test equating.** The flat IRT then uses the shared bill parameters to compute an explicit affine transformation (`equate_chambers()`, lines 726-890):

- **Scale factor A** from the ratio of standard deviations of bill discrimination parameters on concordant shared bills: `A = SD(beta_senate) / SD(beta_house)`. Only bills where both chambers' betas have the same sign are used (51/71 in the 91st, 100/133 in the 90th).
- **Location shift B** from bridging legislators (individuals who served in both chambers): `B = mean(xi_house) - A * mean(xi_senate)`. Falls back to `B = 0` if no bridging legislators exist.
- Senate scores are transformed: `xi_equated = A * xi_senate + B`.

This produces the equated flat IRT scores that correctly place both chambers on a common scale.

### What the joint hierarchical has instead

With zero shared bill parameters, the only connection between House and Senate is the **prior hierarchy**:

```
mu_global → mu_chamber[House], mu_chamber[Senate]
                ↓                    ↓
         mu_group[HD, HR]    mu_group[SD, SR]
              ↓                    ↓
          xi[House members]    xi[Senate members]
```

This soft hierarchical pooling says House and Senate party means should be "somewhat similar." But a prior is not a bridge. It imposes no hard constraint on the relative scale. The model is free to find completely different mappings from xi to vote probability in each chamber — because the two likelihoods are independent.

## The Cascade of Failures

### 1. House dominates the scale

130 House members voting on 297 bills contribute ~75% of the likelihood. Beta parameters for House-only bills are estimated based on 130 legislators' responses; Senate betas are based on only 42. Without shared items to force the scales to agree, the larger body's scale wins.

### 2. The Senate's sign convention flips

The multiplicative sign symmetry (`beta * xi = (-beta) * (-xi)`) is identified within each chamber by the ordering constraint `sort(mu_group_raw)` ensuring Democrat < Republican. But across chambers, nothing prevents the Senate from converging to the opposite sign convention — and it does, consistently. In both the 91st and 90th sessions, the Senate converges with Democrats positive and Republicans negative.

### 3. The group-level posterior degenerates

The sign flip creates a contradictory model state. The sorting constraint forces `mu_group[Senate_D] < mu_group[Senate_R]`, but the actual Senate xi values (pre-flip) have Republicans as the smaller values. The model resolves this by:

- Squashing both Senate mu_group values into a narrow, meaningless range (91st: `[-3.72, -3.47]`)
- Inflating `sigma_within` for Senate Democrats to an absurd 5.66 (vs ~2.0 for other groups)

The hierarchy has collapsed: `sigma_within` carries all the variance that `mu_group` should capture.

**91st session posteriors:**

| Parameter | House D | House R | Senate D | Senate R |
|---|---|---|---|---|
| mu_group | -5.60 | +7.11 | **-3.72** | **-3.47** |
| sigma_within | 2.28 | 1.96 | **5.66** | 1.65 |
| Empirical xi mean | -5.68 | +7.15 | -7.58 | +3.84 |

House parameters are healthy. Senate parameters are degenerate. The same pattern appears in the 90th session: `mu_group[SD] = -1.26, mu_group[SR] = -1.00` with `sigma_within[SD] = 5.65`.

### 4. Post-hoc correction can't recover the damage

Negating Senate xi values (our `fix_joint_sign_convention()`) corrects the direction, but the xi magnitudes were estimated in a degenerate model state. Within-chamber rankings are preserved (r = 0.999 vs per-chamber hierarchical), but the cross-chamber scale is distorted:

| Chamber | Equated flat IRT sd | Joint hierarchical sd | Scale ratio |
|---|---|---|---|
| House | 2.08 | 6.17 | 2.96x |
| Senate | 2.35 | 5.18 | 2.21x |
| **Differential** | | | **1.34x** |

The joint hierarchical inflates House ideal points 34% more than Senate ideal points relative to the flat IRT's equated scale.

## The Fix: Match Bills by Bill Number

This is an engineering bug, not a fundamental modeling limitation. The bridging data exists — 71 to 174 bills per session are voted on by both chambers. The joint hierarchical model just needs to use it.

The fix is to refactor `build_joint_model()` to match bills across chambers the same way `build_joint_vote_matrix()` already does in the flat IRT:

1. **Match vote_ids across chambers by `bill_number`**, preferring Final Action motions
2. **Create shared `beta`/`alpha` indices** for matched bills — both chambers' observations on the same bill point to the same `beta[j]` parameter
3. **Keep separate indices** for chamber-only bills
4. **The shared betas become the bridge**: the model must satisfy both chambers' observations simultaneously, forcing the xi values onto a common scale

```
Before (current — broken):
  House votes → beta[0..296]    (297 House-only betas)
  Senate votes → beta[297..490] (194 Senate-only betas)
  Zero overlap. Likelihoods separable.

After (fixed):
  House-only votes → beta[0..225]      (226 House-only betas)
  Senate-only votes → beta[226..348]   (123 Senate-only betas)
  Matched votes → beta[349..419]       (71 shared betas, both chambers)
  Shared betas bridge the scale.
```

With shared bill parameters, the sign indeterminacy across chambers should resolve naturally (shared bills constrain the relative sign) and the scale distortion should disappear (shared bills constrain the relative magnitudes).

## Why the Per-Chamber Hierarchical Models Work Fine

The per-chamber hierarchical models (House-only and Senate-only) don't have this problem:

- House: ICC = 0.92, r = 0.987 vs flat IRT, sensible group parameters
- Senate: ICC = 0.88, r = 0.975 vs flat IRT, sensible group parameters

They work because each model only has two groups (Democrat and Republican) within a single chamber. The ordering constraint is sufficient for identification, there's no cross-chamber scale to establish, and all bills are observed by all legislators.

## Literature Context

Cross-chamber scaling is a well-studied problem. Every successful method relies on shared data:

| Method | Bridge mechanism | Key reference |
|---|---|---|
| DW-NOMINATE Common Space | Chamber-switching legislators (650 over 200 years) | Poole & Rosenthal |
| Shor-McCarty | NPAT survey responses as common items | Shor & McCarty (2011) |
| Bailey bridge scores | Presidential position-taking on Court cases | Bailey (2007) |
| Flat IRT + test equating | Shared bill discrimination parameters | Psychometric test equating |
| Joint Bayesian IRT | Shared item parameters on matched bills | Clinton, Jackman & Rivers (2004) |

Our joint hierarchical is the only one that attempts cross-chamber scaling with **no shared data**. A hierarchical prior is a regularization device, not a bridging mechanism. As Poole and Rosenthal documented: even with 650 chamber-switching members over 200 years, "the larger number of House members will drive the fit." With zero shared items, this asymmetry becomes overwhelming.

Shor, McCarty, and Berry (2011) studied the thin-bridge problem via Monte Carlo simulation and found that even a few bridge items are sufficient — but zero is qualitatively different from a few.

## Current Status

**Fixed in ADR-0043** (2026-02-26). Two changes:

1. **Bill-matching in `build_joint_model()`**: The `rollcalls` DataFrame is now passed to the joint model builder. A new `_match_bills_across_chambers()` function (extracted from the flat IRT's proven logic) matches vote_ids across chambers by `bill_number`, preferring Final Action motions. Matched bills share a single `alpha`/`beta` parameter pair — the mathematical bridge for cross-chamber identification. Expected: 71-174 shared bills per session.

2. **Group-size-adaptive priors**: Groups with fewer than 20 members (e.g. Senate Democrats) get `sigma_within ~ HalfNormal(0.5)` instead of `HalfNormal(1.0)`. This follows Gelman (2015) on informative priors for small J.

The `fix_joint_sign_convention()` safety net is retained but should no longer trigger for sessions with sufficient shared bills.

Previously:

- **Per-chamber hierarchical models** remain valid for within-chamber analysis (shrinkage, ICC, variance decomposition)
- **Flat IRT + test equating** remains the gold standard for cross-chamber comparisons
- **Joint hierarchical results** should now produce sensible cross-chamber placement with shared bills providing natural identification

## References

- Clinton, J., Jackman, S., & Rivers, D. (2004). The statistical analysis of roll call data. *American Political Science Review*, 98(2), 355-370.
- Shor, B., & McCarty, N. (2011). The ideological mapping of American legislatures. *American Political Science Review*, 105(3), 530-551.
- Shor, B., McCarty, N., & Berry, C. (2011). Methodological issues in bridging ideal points in disparate institutions in a data sparse environment. SSRN Working Paper.
- Bailey, M. (2007). Comparable preference estimates across time and institutions for the Court, Congress, and Presidency. *American Journal of Political Science*, 51(3), 433-448.
- Gray, T. (2020). A bridge too far? Examining bridging assumptions in common-space estimations. *Legislative Studies Quarterly*, 45(1), 131-149.
- Poole, K., & Rosenthal, H. DW-NOMINATE joint House and Senate scaling. legacy.voteview.com.
