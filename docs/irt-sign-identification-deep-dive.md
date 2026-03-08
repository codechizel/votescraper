# IRT Sign Identification in Supermajority Legislatures

A deep dive into reflection invariance, the horseshoe effect, and how PCA-based anchor
selection produces sign flips in supermajority chambers — with a diagnostic and proposed fix.

## The Problem

In the 79th Kansas Senate (2001-2002), our 1-D Bayesian IRT model places Tim Huelskamp
— a staunch fiscal conservative who later served as a Freedom Caucus member in Congress
— as the *most liberal* legislator, more negative than any Democrat. The hierarchical
IRT model (with a sort constraint ensuring Democrat mean < Republican mean) produces the
same pattern: Huelskamp at -1.90 while the most moderate Republican (Sandy Praeger) sits
at +6.45.

**This is a sign flip.** The model has correctly recovered the latent dimension's shape
but has assigned the wrong ideological polarity to the endpoints. The fix is a post-hoc
validation step that detects and corrects these flips.

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
supermajority chambers where sign is "correct" at the party-mean level but wrong at the
individual level.

## The Horseshoe Effect

The core mechanism behind the sign flip is the **horseshoe effect** — a well-known
artifact of dimension reduction in spatial voting models. When a single dimension is
extracted from a two-dimensional (or higher) voting space, groups that are far apart
on the true ideological spectrum but vote similarly get folded onto the same end of the
recovered dimension.

In a supermajority chamber, the dominant source of voting variation shifts from *between*
parties to *within* the majority party. The first principal component captures the split
between:

- **Establishment/leadership-loyal majority members** (voting Yea on leadership bills)
- **Dissident majority members** (voting Nay from the ideological extreme)
- **Minority party** (also voting Nay, but for opposite ideological reasons)

The IRT model sees rebel conservatives and Democrats voting the **same way** on many
roll calls. Since the model has no concept of *why* someone votes — only the pattern of
votes — it cannot distinguish "voting Nay because the bill is too conservative" from
"voting Nay because the bill is not conservative enough."

The result: far-right rebels and Democrats are placed on the same end of the dimension.
When PCA orients this dimension using party means (R mean > D mean), the rebel
Republicans — who vote *against* their own party — end up on the Democratic side.

### Why This Is a Sign Flip, Not Just "Dimension Collapse"

Previous analysis framed this as "dimension collapse — a data feature, not a model
failure." That framing is incorrect. Here's why:

1. **The model correctly identifies the cleavage structure.** The dimension separates
   high-defection legislators from party-loyal ones. The shape is right.

2. **But PCA-based orientation assigns the wrong polarity.** `orient_pc1()` flips PC1
   so that R mean > D mean. In a supermajority chamber, the R mean is dominated by the
   moderate establishment majority. The rebel conservatives pull toward the D end. So
   the orientation is "correct" at the aggregate level (R mean > D mean) but wrong at
   the individual level (Huelskamp, the most conservative senator, appears most liberal).

3. **Anchor selection inherits the PCA error.** `select_anchors()` picks the R with
   the highest PC1 as the conservative anchor. In a supermajority chamber, that's the
   most *establishment* Republican (Praeger), not the most *conservative*. The liberal
   anchor (most extreme Democrat) is correct. So the model anchors on one correct and
   one incorrect reference point, locking in the sign flip.

4. **The mirror image of the recovered dimension is the correct solution.** If you
   negate all ideal points, Huelskamp moves to the far-right extreme and Praeger moves
   toward the center — exactly matching their known ideological positions. The
   hierarchical model's sort constraint (D mean < R mean) would need to be re-evaluated
   after the flip, but the individual placements would be correct.

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

Both our flat IRT and hierarchical IRT models produce the sign flip:

| Model | R mean | D mean | Huelskamp | Praeger |
|-------|--------|--------|-----------|---------|
| Flat IRT | +0.65 | -0.49 | -3.26 | +1.00 (anchor) |
| Hierarchical IRT | +3.10 | +1.19 | -1.90 | +6.45 |

In both models, the sign is "correctly" identified at the party-mean level (R mean >
D mean). But the individual placements are inverted: Huelskamp appears more liberal than
any Democrat, while Praeger (the most moderate Republican) appears most conservative.

## Diagnosing the Sign Flip

### Cross-Party Contested Vote Agreement

The key diagnostic is **cross-party contested vote agreement**: on roll calls where both
parties had members voting on both sides (genuinely contested, bipartisan votes), which
legislators from opposite parties vote together most often?

The logic:

- On contested votes, legislators near the center of the ideological spectrum should
  cross party lines and vote with moderates from the other party.
- Ultra-conservative Republicans should *rarely* agree with Democrats on contested votes.
- Moderate Republicans should agree with Democrats *more often* on contested votes.

If the IRT dimension is correctly oriented:
- Legislators with ideal points near the center (moderate Republicans, conservative
  Democrats) should have the highest cross-party agreement rates.
- Legislators at the extremes (liberal Democrats, conservative Republicans) should have
  the lowest cross-party agreement rates.

If the sign is flipped:
- The legislator placed at the most "liberal" position (Huelskamp) would have *low*
  agreement with Democrats on contested votes — because he's actually the most
  conservative, not the most liberal.
- The legislator placed at the most "conservative" position (Praeger) would have *high*
  agreement with Democrats — because she's actually a moderate.

### Applying the Diagnostic to the 79th Senate

In the 79th Kansas Senate, cross-party contested vote agreement reveals the flip:

- **Huelskamp** (IRT: most liberal at -3.26): Low agreement with Democrats on contested
  votes. He votes Nay on the same bills as Democrats, but on *different* bills — he
  defects on conservative-vs-establishment fights, not on bipartisan issues.
- **Praeger** (IRT: most conservative at +1.00): High agreement with Democrats on
  contested votes. As a genuine moderate, she frequently crosses party lines on social
  and fiscal issues.

This is definitive evidence of a sign flip. The contested-vote agreement pattern is the
inverse of what correctly oriented ideal points would predict.

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

Thomas Frank's *What's the Matter with Kansas?* (2004) provides broader context: Kansas
is a state where "the only remaining political division [is] between the moderate and
more-extreme right wings of the same party."

## Post-Hoc Sign Validation (Implemented)

**Status:** Implemented in `analysis/05_irt/irt.py` as `validate_sign()` (2026-03-07). 6 tests in `tests/test_irt.py`. ADR-0101 addendum.

### Algorithm

After MCMC sampling produces ideal points, `validate_sign()` applies:

1. **Identify contested votes**: roll calls where both parties had members voting Yea
   and Nay (minimum threshold: at least 10% of each party on each side).

2. **Compute cross-party agreement**: for each legislator, calculate what fraction of
   contested votes they agreed with the median voter of the opposite party.

3. **Correlate with ideal points**: if the ideal points are correctly oriented,
   legislators nearer the center (closer to 0) should have *higher* cross-party
   agreement, and those at the extremes should have *lower* agreement.

4. **Check the correlation sign**: a negative correlation (higher agreement = more
   central ideal point) confirms correct orientation. A positive correlation (higher
   agreement = more extreme ideal point) indicates a sign flip.

5. **If flipped, negate all ideal points and discrimination parameters.** This is
   mathematically equivalent — both solutions are equally valid posterior modes. The
   validation step selects the mode that matches the known ideological structure.

### Why This Works

The cross-party agreement metric is *external* to the IRT model. It uses raw vote
matrices, not model-estimated parameters. This makes it robust to the horseshoe
effect — it doesn't matter that Huelskamp and Democrats vote Nay on many of the same
bills, because the diagnostic only looks at *contested* bills where both parties are
split. On those bills, Huelskamp's agreement pattern with Democrats is very different
from Praeger's.

### Edge Cases

- **Chambers with no contested votes**: if party-line voting is near-total, there are
  no contested votes to validate against. In these chambers, the sign flip problem is
  also unlikely (no rebel faction to trigger the horseshoe effect).
- **Three-way splits**: when both parties have factions, the correlation may be weak
  but still directionally correct.
- **Very small chambers**: with fewer than 20 legislators, sample sizes for
  cross-party agreement rates may be too small for reliable correlation.

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

### 4. Supermajority Diagnostic

For chambers where one party holds 70%+ seats, compute:
- Within-party variance of ideal points for the dominant party
- Between-party difference in means
- Ratio: if within-party variance exceeds between-party difference, flag the
  session as likely affected by the horseshoe effect

**Applicability to Tallgrass:** Easy to add as a diagnostic in Phase 05 and Phase 07
reports. Already have the data — just need the ratio computation and interpretive text.

## What Our Pipeline Does (Current State)

### Phase 05: Flat IRT

**Anchor selection** (`select_anchors()` in `analysis/05_irt/irt.py`): After PCA
orients PC1 so that Republican mean > Democrat mean, we select anchors by party:

- **Conservative anchor**: Republican with highest PC1 (most party-typical R)
- **Liberal anchor**: Democrat with lowest PC1 (most party-typical D)

This ensures anchors come from opposite parties, but in supermajority chambers the
"most party-typical R" is the most establishment-aligned moderate, not the most
ideologically conservative.

**Post-hoc sign validation** (`validate_sign()`, implemented 2026-03-07): After MCMC,
correlates cross-party contested vote agreement with Republican ideal points. If the
Spearman correlation is positive (p < 0.10), negates xi, xi_free, and beta posteriors.
This catches the sign flip that party-aware anchor selection alone cannot prevent.

### Phase 07: Hierarchical IRT

**Sort constraint**: `pt.sort(mu_party_raw)` with `dims="party"` where party =
`["Democrat", "Republican"]` enforces Democrat mean < Republican mean. This correctly
identifies the sign at the party level but does not prevent individual-level horseshoe
placement. Negation cannot fix the hierarchical model because it would violate the
sort constraint — the individual-level issue is genuine dimension collapse, not a
sign flip.

**PCA initialization**: xi_offset is initialized from standardized PCA PC1 scores.

### Remaining Opportunities

1. **Supermajority diagnostic** in IRT reports: flag sessions where within-party
   variance exceeds between-party gap. (Not yet implemented.)

2. **External anchor validation**: in supermajority chambers, consider using
   Shor-McCarty scores or party unity rankings to validate anchor selection.
   (Phase 17 already correlates with SM scores where available.)

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
- ADR-0103: IRT identification strategy system (7 strategies, auto-detection)
- ADR-0104: IRT robustness flags (horseshoe diagnostic, contested-only refit, 2D cross-reference)

## Related Documentation

- `docs/irt-identification-strategies.md` — All 7 identification strategies and auto-detection logic
- `docs/79th-horseshoe-robustness-analysis.md` — Empirical robustness analysis of the 79th using all three diagnostic flags
