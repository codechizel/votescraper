# Chapter 4: Conversion vs. Replacement: Why Does Ideology Change?

> *The Republican caucus shifted rightward between the 87th and 88th Legislatures. But why? Did the existing members become more conservative — or did moderate Republicans lose their seats to more conservative challengers?*

---

## Two Ways a Party Changes

When a party's average ideology shifts between sessions, one of two things happened. Maybe both.

**Conversion:** Returning legislators changed their voting behavior. The same people came back and voted differently. Perhaps a moderate Republican started voting with the conservative wing. Perhaps a liberal Democrat moved to the center. The roster didn't change — the people did.

**Replacement:** Different legislators showed up. Moderates retired or lost primaries and were replaced by more ideologically extreme newcomers. Or extremists were voted out and replaced by pragmatists. The people changed — their behavior upon arrival was simply different from their predecessors'.

Understanding which mechanism drove the shift matters enormously for interpreting Kansas politics. Conversion suggests that the political environment changed — perhaps a new party leader demanded loyalty, or a high-profile issue pulled members in a new direction. Replacement suggests that the electorate changed — voters chose different kinds of candidates, or redistricting reshaped the competitive landscape.

Tallgrass decomposes every session-to-session shift into these two components.

## The Three Cohorts

The decomposition starts by dividing legislators into three groups based on whether they served in the earlier session (A), the later session (B), or both:

| Cohort | Definition | What They Tell Us |
|--------|-----------|-------------------|
| **Returning** | Served in both A and B | Their shift measures *conversion* — same people, different behavior |
| **Departing** | Served in A but not B | Their ideology represents what was *lost* when they left |
| **New** | Served in B but not A | Their ideology represents what was *added* when they arrived |

Every legislator falls into exactly one cohort. The three cohorts are exhaustive and mutually exclusive.

### Getting Everyone on the Same Scale

To compare departing legislators' ideology (measured on session A's scale) with new legislators' ideology (measured on session B's scale), you need a common scale. Tallgrass uses the affine transformation from Chapter 3:

```
xi_departing_aligned = A × xi_departing_original + B
```

After alignment, all three cohorts' ideal points are on session B's scale, making direct comparison meaningful.

## The Decomposition

### The Math

The total shift in a party's average ideology equals the sum of conversion and replacement:

```
Total shift = Conversion effect + Replacement effect
```

Each component:

```
Conversion = mean(xi_returning in B) - mean(xi_returning in A, aligned)
```

**Plain English:** "How much did the returning members' average ideology change between sessions?"

```
Replacement = Total shift - Conversion
```

**Plain English:** "Whatever the total shift was that conversion didn't explain — that's replacement."

Or, thinking about it from the cohort perspective:

```
Replacement ≈ (mean(xi_new) - mean(xi_departing_aligned)) × (cohort proportion)
```

**Plain English:** "How different are the newcomers from the people they replaced, scaled by how many seats turned over?"

### A Worked Example

Suppose the Kansas House Republican caucus in the 87th Legislature has a mean ideal point of +1.20 (on the aligned scale). In the 88th Legislature, the mean is +1.45. The total rightward shift is +0.25.

Now break it down:

- **Returning Republicans** (70 members who served in both): Their mean went from +1.22 (87th, aligned) to +1.30 (88th). Conversion = +0.08. These members individually shifted a small amount rightward.

- **Departing Republicans** (15 members who left): Their aligned mean was +0.95 — they were, on average, the party's moderates. **New Republicans** (15 members who arrived): Their mean is +1.50 — they're more conservative than the caucus average.

- Replacement = +0.25 - 0.08 = +0.17. The newcomers were substantially more conservative than the departing moderates, accounting for about two-thirds of the total rightward shift.

**Interpretation:** The party moved right primarily through **replacement** — moderate Republicans were replaced by more conservative successors. Conversion contributed, but the bigger driver was the changing roster.

## The KS Test: Is the Difference Real?

The cohort means tell a story, but with 15 departing and 15 new legislators, the sample sizes are small. Is the difference between their ideology distributions statistically significant, or could it be noise?

The **Kolmogorov-Smirnov (KS) test** answers this question. It compares two entire distributions — not just their means — and asks: "Could these two groups plausibly have been drawn from the same underlying population?"

The test works by computing the maximum vertical distance between the two groups' cumulative distribution functions (CDFs). Imagine plotting a step function for each group, where each step represents one legislator's ideal point. The KS statistic is the widest gap between the two step functions.

```
KS statistic = max|CDF_new(x) - CDF_departing(x)| for all x
```

**Plain English:** "At the point where the two groups differ the most, how different are they?"

- **Large KS statistic + small p-value (< 0.05):** The distributions are significantly different. New members really do come from a different ideological population than departing members.
- **Small KS statistic + large p-value:** No significant difference. The turnover was ideologically neutral — the newcomers look like the people they replaced.

The KS test is non-parametric: it doesn't assume the ideology distributions are bell-shaped. This matters because legislative ideology often isn't normally distributed (bimodal in polarized chambers, skewed in supermajority ones).

Tallgrass runs two KS tests per chamber:

1. **Departing vs. Returning:** Did the people who left differ ideologically from those who stayed?
2. **New vs. Returning:** Do the newcomers differ ideologically from the incumbents they joined?

If both tests are significant and point the same direction (departing members were moderate, new members are extreme), the case for replacement-driven change is strong.

**Codebase:** `analysis/26_cross_session/cross_session_data.py` (`compute_turnover_impact()`, using `scipy.stats.ks_2samp()`)

## Freshmen Cohort Analysis

Beyond the aggregate decomposition, Tallgrass profiles the freshmen class (new members) as a group and compares them to the incumbents they joined:

| Comparison | Method | What It Reveals |
|-----------|--------|----------------|
| **Ideology** | KS test on ideal point distributions | Are freshmen more extreme or more moderate than incumbents? |
| **Party unity** | t-test on CQ unity scores | Do freshmen toe the party line more or less than veterans? |
| **Maverick rate** | Comparison of means | Are freshmen more willing to cross party lines? |

The freshmen analysis produces a density plot showing two overlapping distributions — freshmen in one color, returning members in another — making the ideological overlap (or separation) visible at a glance.

**Codebase:** `analysis/26_cross_session/cross_session_data.py` (`analyze_freshmen_cohort()`)

## Voting Bloc Stability: Did the Coalitions Survive?

Chapter 4 of Volume 6 identified voting blocs using cluster analysis. But do those blocs persist across sessions? A "moderate Republican" bloc in the 87th Legislature might dissolve, merge with the conservative wing, or reconstitute with different members in the 88th.

Tallgrass tracks bloc transitions with a **Sankey diagram** — a flow visualization where:

- **Left side:** Blocs from session A (each colored band represents a cluster)
- **Right side:** Blocs from session B
- **Flows:** Lines connecting where each legislator ended up

A legislator who was in the "moderate Republican" cluster in session A and the "conservative" cluster in session B appears as a flow from left to right. Thick flows represent many legislators making the same transition; thin flows represent individual defections.

The Sankey reveals:

- **Stable blocs:** Thick, horizontal flows (most members stay in the same cluster type)
- **Dissolving blocs:** A left-side band that fragments into multiple right-side destinations
- **Absorbing blocs:** Multiple left-side sources flowing into a single right-side cluster
- **Individual migrations:** Thin diagonal lines showing specific legislators who switched factions

## What Can Go Wrong

### Small Cohort Sizes

If only 5 legislators departed and 5 arrived, the decomposition has extremely wide uncertainty. The KS test will have low power (it can't detect real differences with tiny samples), and the cohort means are volatile. Tallgrass reports the cohort sizes prominently so readers can calibrate their confidence.

### Retirement vs. Defeat

The decomposition treats all departing legislators the same — they're gone. But a moderate who retired voluntarily (choosing not to run) has a different political interpretation than one who lost a primary to a more conservative challenger. The analysis can't distinguish between these scenarios from the vote data alone; interpreting the replacement effect requires external knowledge of why members left.

### Alignment Uncertainty

The decomposition depends on the affine alignment from Chapter 3. If the alignment is imprecise (few returning legislators, nonlinear scale differences), the departing cohort's aligned scores may be slightly off, biasing the replacement effect estimate. This is most likely when comparing sessions separated by redistricting.

### Party Switches

A legislator who switches parties between sessions complicates the decomposition. Are they a "departing" Republican and a "new" Democrat, or a "returning" legislator who changed labels? Tallgrass flags party switches and includes these legislators in the returning cohort (since they serve in both sessions), but their ideological shift might reflect genuine conversion or might just reflect a label change that was always coming.

---

## Key Takeaway

A party's ideological shift decomposes into conversion (returning members changed their voting) and replacement (departing members were replaced by ideologically different newcomers). The KS test formally evaluates whether the newcomer and departing cohorts come from different ideological populations. Freshmen cohort analysis profiles the incoming class, and Sankey diagrams track voting bloc transitions across sessions. Together, these tools answer *why* the legislature changed — not just *that* it changed.

---

*Terms introduced: conversion effect, replacement effect, turnover decomposition, cohort (returning, departing, new), KS test (Kolmogorov-Smirnov), cumulative distribution function (CDF), freshmen cohort, Sankey diagram, bloc transition, party switch*

*Previous: [Cross-Session Alignment: Putting Sessions on the Same Scale](ch03-cross-session-alignment.md)*

*Next: [Fifteen Years of Kansas Politics (2011-2026)](ch05-fifteen-years.md)*
