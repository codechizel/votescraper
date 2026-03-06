# IRT Sign Identification in Supermajority Legislatures

A deep dive into reflection invariance, dimension collapse, and why far-right rebels
can appear "liberal" in ideal point estimates.

## The Problem

In the 79th Kansas Senate (2001-2002), our 1-D Bayesian IRT model places Tim Huelskamp
— a staunch fiscal conservative who later served as a Freedom Caucus member in Congress
— as the *most liberal* legislator, more negative than any Democrat. The hierarchical
IRT model (with a sort constraint ensuring Democrat mean < Republican mean) produces the
same pattern: Huelskamp at -1.90 while the most moderate Republican (Sandy Praeger) sits
at +6.45.

This is not a coding error. It is a well-documented artifact of spatial voting models in
supermajority chambers.

## Background: Sign Indeterminacy in IRT

The two-parameter IRT model for legislative voting suffers from three fundamental
non-identifiabilities:

1. **Translation invariance**: adding a constant to all ideal points while adjusting item
   parameters preserves the likelihood. Solved by a Normal(0, 1) prior on ideal points.

2. **Scale invariance**: scaling ideal points by a factor while inversely scaling
   discrimination produces identical voting probabilities. Solved by fixing the variance
   of the ability distribution.

3. **Reflection invariance**: negating *both* ideal points (xi) and discrimination
   parameters (beta) yields an identical likelihood. In a D-dimensional model, this
   creates 2^D x D! symmetric posterior modes — so a 1-D model has 2 modes, and a 2-D
   model has 8.

The third invariance — reflection — is the one that bites us. Without a constraint to
break the symmetry, the sampler can settle into either mode, and the "sign" of ideal
points is arbitrary.

### Standard Identification Strategies

**Anchoring (hard constraints).** Clinton, Jackman, and Rivers (2004) fix two
legislators at known positions: a known liberal at -1 and a known conservative at +1.
This is the approach used in our Phase 05 flat IRT. The `pscl` R package uses
near-degenerate "spike" priors (precision = 1e12) to pin legislators.

**Sort constraint (ordering).** Our Phase 07 hierarchical IRT uses
`pt.sort(mu_party_raw)` to enforce Democrat mean < Republican mean. This constrains the
party-level sign without pinning individual legislators.

**Positive discrimination.** In educational IRT, requiring all discrimination parameters
alpha > 0 (via LogNormal priors) resolves sign. This does *not* work for legislative
voting because roll calls can cut the ideological space in either direction — a "yea"
vote may indicate either liberal or conservative positions depending on the bill.

**Post-hoc sign correction.** After MCMC, check party means and flip if needed. Simple
and effective for balanced chambers, but does not fix the deeper problem in
supermajority chambers.

## Why Supermajority Chambers Break the First Dimension

When one party controls 75%+ of seats, the dominant source of voting variation shifts
from *between* parties to *within* the majority party. The first principal component —
and therefore the first IRT dimension — captures the largest source of variation in the
vote matrix, which in a supermajority chamber is the split between:

- **Establishment/leadership-loyal majority members** (voting Yea on leadership bills)
- **Dissident majority members** (voting Nay from the ideological extreme)
- **Minority party** (also voting Nay, but for opposite ideological reasons)

The IRT model sees rebel conservatives and Democrats voting the **same way** on many
roll calls. Since the model has no concept of *why* someone votes — only the pattern of
votes — it cannot distinguish "voting Nay because the bill is too conservative" from
"voting Nay because the bill is not conservative enough."

The result: far-right rebels are estimated near Democrats on the first dimension. This
is sometimes called **dimension collapse** — the single dimension conflates two distinct
phenomena (ideological direction and anti-establishment posture) that produce identical
voting patterns.

### When This Occurs

The pattern is most severe when:

1. One party holds a supermajority (75%+ seats)
2. A faction within the majority routinely votes with the minority
3. The faction's defections are driven by ideological extremism rather than moderation
4. The roll call agenda is dominated by establishment vs. dissident votes rather than
   partisan bills

### Evidence from the 79th Kansas Senate

The 79th Kansas Senate (2001-2002) is a textbook case. With 30 Republicans and 10
Democrats, the PCA eigenvalue ratio is λ1/λ2 = 1.45 — the second dimension carries
nearly as much information as the first. The party unity standard deviation for
Republicans (9.2%) is 2.4x that of Democrats (3.8%), reflecting the deep
moderate-conservative factional split.

Both our flat IRT and hierarchical IRT models produce consistent results:

| Model | R mean | D mean | Huelskamp | Praeger |
|-------|--------|--------|-----------|---------|
| Flat IRT | +0.65 | -0.49 | -3.26 | +1.00 (anchor) |
| Hierarchical IRT | +3.10 | +1.19 | -1.90 | +6.45 |

In both models, the sign is correctly identified (R mean > D mean). But Huelskamp's
voting pattern places him at the negative extreme — more extreme than any Democrat. The
hierarchical model's sigma_within_R = 2.48 spans a range of ~10 points, completely
engulfing the inter-party gap of ~1.9 points.

## The Kansas Three-Party System

Kansas political scientists describe the state as having "three-party politics":
moderate Republicans, conservative Republicans, and Democrats. This split intensified
after the 1991 "Summer of Mercy" anti-abortion protests in Wichita, which fractured the
Kansas GOP.

Key figures in the 79th Senate illustrate the pattern:

- **Tim Huelskamp** (R-38th): Leader of the conservative faction. Removed from the
  Ways and Means Committee for clashes with leadership. Later a Freedom Caucus member
  in Congress.
- **Sandy Praeger** (R-23rd): Moderate Republican. Later served as Kansas Insurance
  Commissioner. Praised by both parties for bipartisan work.
- **David Haley** (D-4th): Reliable Democratic partisan. Used as liberal anchor.

The IRT dimension correctly captures the *dominant cleavage* in the Kansas Senate: the
moderate Republican establishment (Praeger, Oleen, Vratil, Adkins) versus everyone else
(Huelskamp's conservative faction + Democrats). This is the axis along which most close
votes split. The traditional left-right axis is the *second* dimension.

Thomas Frank's *What's the Matter with Kansas?* (2004) provides broader context: Kansas
is a state where "the only remaining political division [is] between the moderate and
more-extreme right wings of the same party."

## Solutions from the Literature

### 1. IRT-M: Theory-Driven Identification (Morucci et al. 2024)

The most promising recent development. Analysts pre-specify how each roll call relates
to theoretical dimensions using a constraint matrix of {+1, -1, 0, NA} values. This
creates synthetic "anchor units" at extreme positions along each dimension, solving
identification without fixing real legislators. Robust even with 50-75% misspecified
constraints. Available as an R package (`IRTM`).

**Applicability to Tallgrass:** High. We have bill text and subject classifications from
Phase 20 (BERTopic) and Phase 22 (issue-specific IRT) that could inform constraint
matrices. Would require a 2-D model and roll call classification infrastructure.

### 2. Multi-Dimensional IRT

A 2-D IRT model *should* separate the ideological dimension from the establishment
dimension. Poole and Rosenthal's formulation: the first dimension captures the dominant
left-right split, the second captures "cross-cutting, salient issues of the day."

**Challenges:** 2-D models have 8 symmetric modes (2^2 x 2!), making identification
much harder. Rotational invariance means recovered dimensions may not align with
theoretically meaningful axes without external constraints.

**Applicability to Tallgrass:** Phase 06 already runs 2-D IRT. The question is whether
its identification strategy (2 anchors per dimension) correctly separates establishment
from ideology in supermajority chambers.

### 3. External Validation via Bridge Actors (Shor & McCarty 2011)

The gold standard for cross-state scaling. Uses legislators who served in both state
and federal legislatures as "bridge actors" to place state-level ideal points on a
common scale. Shor-McCarty scores cover 24,716 state legislators from 1993-2018.

**Applicability to Tallgrass:** Phase 17 (External Validation) already correlates with
Shor-McCarty scores where available. Could be extended to use SM scores as informative
priors for anchor selection.

### 4. Party-Constrained Priors

Set prior means for ideal points conditional on party membership (e.g., Republicans
centered at +0.5, Democrats at -0.5). This soft-constrains the scale but does not
prevent within-party reordering.

**Applicability to Tallgrass:** The hierarchical IRT already does this via the sort
constraint on party means. The issue is that the constraint is necessary but not
sufficient — it constrains relative ordering of party *means* but cannot prevent
individual legislators from crossing.

### 5. Supermajority Diagnostic

For chambers where one party holds 70%+ seats, compute:
- Within-party variance of ideal points for the dominant party
- Between-party difference in means
- Ratio: if within-party variance exceeds between-party difference, flag the
  session as potentially capturing intra-party rather than inter-party variation

**Applicability to Tallgrass:** Easy to add as a diagnostic in Phase 05 and Phase 07
reports. Already have the data — just need the ratio computation and interpretive text.

## What Our Pipeline Does

### Phase 05: Flat IRT

**Anchor selection** (`select_anchors()` in `analysis/05_irt/irt.py`): After PCA
orients PC1 so that Republican mean > Democrat mean, we select anchors by party:

- **Conservative anchor**: Republican with highest PC1 (most party-typical R)
- **Liberal anchor**: Democrat with lowest PC1 (most party-typical D)

This ensures anchors come from opposite parties and represent the ideological mainstream
of each party. The previous approach — raw PC1 extremes without party filtering — caused
sign flip in supermajority chambers where intra-party variation dominated PC1.

### Phase 07: Hierarchical IRT

**Sort constraint**: `pt.sort(mu_party_raw)` with `dims="party"` where party =
`["Democrat", "Republican"]` enforces Democrat mean < Republican mean. This correctly
identifies the sign at the party level.

**PCA initialization**: xi_offset is initialized from standardized PCA PC1 scores.
This prevents reflection mode-splitting during MCMC sampling.

## Interpretation Guidance

When reviewing IRT results from supermajority chambers:

1. **The first dimension captures voting behavior, not ideology.** A legislator's ideal
   point reflects how they *vote* relative to others, not what they *believe*. In
   supermajority chambers, voting against the establishment — whether from the left or
   the right — produces similar voting patterns.

2. **Check the eigenvalue ratio.** When λ1/λ2 < 2.0 (as in the 79th Senate at 1.45),
   the second dimension carries substantial information. A 1-D model may not adequately
   represent the ideological space.

3. **Look at sigma_within for the majority party.** When sigma_within for the majority
   exceeds the between-party gap in means, individual majority-party legislators will
   span the entire ideological space, and some will appear "more extreme" than minority
   party members.

4. **Cross-reference with party unity scores.** Phase 01 EDA computes party unity for
   each legislator. Low-unity majority-party members (like Huelskamp at 72.7%) are the
   ones most likely to appear at the "wrong" end of the ideal point scale.

5. **The 2-D IRT (Phase 06) may be more informative.** For supermajority sessions, the
   second dimension often separates ideological rebels from the minority party, revealing
   the three-group structure that 1-D models collapse.

## Key References

- Clinton, J., Jackman, S., & Rivers, D. (2004). "The Statistical Analysis of Roll Call
  Data." *American Political Science Review*, 98(2).
- Frank, T. (2004). *What's the Matter with Kansas?* Metropolitan Books.
- Martin, A. & Quinn, K. (2002). "Dynamic Ideal Point Estimation via Markov Chain Monte
  Carlo for the U.S. Supreme Court, 1953-1999." *Political Analysis*, 10(2).
- McCarty, N. (2010). "Measuring Legislative Preferences." Princeton manuscript.
- Morucci, M., Foster, J., Webster, S., Lee, D., & Siegel, A. (2024). "Measurement
  that Matches Theory: Theory-Driven Identification in IRT Models." *American Political
  Science Review*.
- Poole, K. & Rosenthal, H. (1985). "A Spatial Model for Legislative Roll Call
  Analysis." *American Journal of Political Science*, 29(2).
- Shor, B. & McCarty, N. (2011). "The Ideological Mapping of American Legislatures."
  *American Political Science Review*, 105(3).
- Spirling, A. & Quinn, K. (2010). "UK-OC OK? Interpreting Optimal Classification
  Scores for the UK House of Commons." *Political Analysis*, 18(1).

## Related ADRs

- ADR-0006: IRT implementation choices (anchor selection, priors, identification)
- ADR-0023: PCA-informed IRT chain initialization
- ADR-0044: PCA initialization for hierarchical model
- ADR-0051: nutpie per-chamber IRT
- ADR-0053: nutpie flat + joint IRT
- ADR-0074: Stocking-Lord linking for cross-chamber alignment
- ADR-0101: Party-aware IRT anchor selection (this article's companion ADR)
