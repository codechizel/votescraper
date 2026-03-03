# Classical Indices Design Choices

**Script:** `analysis/13_indices/indices.py`
**Constants defined at:** `analysis/13_indices/indices.py` (top of file)

## Assumptions

1. **Indices computed on ALL roll calls** (unfiltered), matching Congressional Quarterly convention. EDA filters (minority < 2.5%, min 20 votes) are not applied to the primary computations. A sensitivity analysis reruns on EDA-filtered data to measure the impact.

2. **Party vote = CQ standard.** A roll call is a "party vote" if and only if the majority of Republicans oppose the majority of Democrats, with each party having at least 2 Yea+Nay voters. This is stricter than EDA's 2.5% minority filter and produces fewer qualifying votes.

3. **Chambers are analyzed separately**, consistent with all upstream phases.

4. **Legislator identity is stable within session.** Mid-session replacements (e.g., Miller) are included and joined on `legislator_slug`, which is a stable identifier across the pipeline.

5. **50/50 party splits are Nay-majority.** When a party's Yea and Nay counts are equal, the party has no clear majority position. We treat this as Nay majority (no affirmative consensus).

## Parameters & Constants

| Constant | Value | Justification |
|----------|-------|---------------|
| `PARTY_VOTE_THRESHOLD` | 0.50 | CQ standard: >50% of Yea+Nay = majority |
| `RICE_FRACTURE_THRESHOLD` | 0.50 | Rice < 0.5 = party more divided than united on that vote |
| `MIN_PARTY_VOTERS` | 2 | Need 2+ voters from a party for Rice to be meaningful |
| `MAVERICK_WEIGHT_FLOOR` | 0.01 | Floor on closeness weight to prevent division by near-zero |
| `CO_DEFECTION_MIN` | 3 | Minimum shared defections for co-defection heatmap entry |
| `ENP_MULTIPARTY_THRESHOLD` | 2.5 | ENP > 2.5 = multiparty-like voting behavior on that roll call |
| `ROLLING_WINDOW` | 15 | Rolling window in roll calls (not calendar days) for time series |
| `TOP_DEFECTORS_N` | 20 | Number of top defectors shown in co-defection heatmap |
| `PARTY_COLORS` | R=#E81B23, D=#0015BC | Consistent with all prior phases |

## Methodological Choices

### Rice Index: per-vote per-party cohesion

**Formula:** `Rice = |Yea - Nay| / (Yea + Nay)` for each party on each roll call.

**Range:** 0 (50-50 split) to 1 (unanimous). A Rice of 0.50 means 75-25 or 25-75 — the party is more united than divided but not strongly so.

**Decision:** Compute Rice on ALL votes, not just party votes. Rice is meaningful even on non-party votes (it measures intra-party cohesion regardless of what the other party does). Party votes are a subset used for unity scores.

**Null handling:** If a party has fewer than `MIN_PARTY_VOTERS` Yea+Nay voters on a roll call, Rice is null for that party on that vote.

### Party Unity: CQ-standard legislator score

**Formula:** `Unity = (votes with party majority on party votes) / (party votes where legislator was present)`

**Decision:** Only computed on "party votes" (majority-vs-majority). This is the standard Congressional Quarterly definition and differs from clustering's "party loyalty" metric (which uses contested-vote threshold = 10% dissent). Both are valid; they answer different questions.

**Comparison with clustering loyalty:** CQ unity uses a binary majority threshold (>50%). Clustering loyalty uses a 10% dissent threshold. On votes where 51% vote Yea and 49% Nay, CQ sees a party majority; clustering sees a contested vote. CQ unity will generally be higher than clustering loyalty for the same legislator.

### Effective Number of Parties (ENP)

**Seat-based:** `ENP_seats = 1 / sum(p_i^2)` where p_i is each party's seat share. This is the Laakso-Taagepera index.

**Vote-based (per roll call):** Blocs are (party, vote_direction) pairs. A roll call where Rs split 60-40 and Ds vote 100% Nay has 3 blocs: (R, Yea), (R, Nay), (D, Nay). ENP is computed on bloc sizes.

**Decision:** Include absent/not-voting legislators in the denominator for seat-based ENP (total seats, not just present). For vote-based ENP, only count legislators who voted Yea or Nay (present and voting).

### Maverick Scores: unweighted and weighted

**Unweighted maverick:** Fraction of party votes where the legislator voted against their party's majority. This is `1 - unity`.

**Weighted maverick:** Same as unweighted, but each defection is weighted by how close the overall chamber vote was. Weight = `1 / max(margin, MAVERICK_WEIGHT_FLOOR)` where margin = |Yea - Nay| / (Yea + Nay) at the chamber level. Close votes get higher weight because defecting on a close vote is more consequential than defecting on a blowout.

**Decision:** Use chamber-level margin (not within-party margin) because the question is "how much did this defection matter to the outcome?" Within-party margin is already captured by Rice.

### Carey UNITY: stricter Rice variant

**Formula:** `Carey UNITY = |Yea - Nay| / (total party members in chamber)` per vote per party.

Unlike Rice (denominator = Yea + Nay), Carey includes absent and not-voting legislators in the denominator. This penalizes parties for low turnout, capturing strategic abstention — legislators who skip votes rather than openly defect.

**Reference:** Carey, J.M. (2007). "Competing Principals, Political Institutions, and Party Unity in Legislative Voting." *American Journal of Political Science* 51(1).

**Decision:** Carey UNITY is computed alongside Rice and saved to a separate parquet file. The relationship `Carey ≤ Rice` always holds (larger denominator) and the gap `Rice - Carey` measures the abstention penalty.

### Co-defection matrix

**Definition:** For the top N defectors in a party, count how many times each pair defected together (both voted against their party majority on the same party vote).

**Minimum threshold:** Pairs with fewer than `CO_DEFECTION_MIN` shared defections are zeroed out to reduce noise from single coincidences.

## Downstream Implications

- **Multi-session stacking:** All parquet outputs include a `session` column, enabling future concatenation across sessions without schema conflicts.
- **Comparison with clustering:** CQ unity and clustering loyalty answer different questions. Unity is the standard political science metric; loyalty captures within-contested-vote behavior. Both should be reported side-by-side.
- **Rice Index provides per-vote data** that no prior phase computed. Rice timeseries can reveal session dynamics (e.g., increasing fracture as session progresses).
