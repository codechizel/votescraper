# Chapter 1: Time Series: Detecting When the Legislature Shifts

> *Is the Kansas Legislature more polarized today than it was at the start of the session? If so, when did the shift happen — and can we pinpoint the moment?*

---

## Why Time Matters Within a Session

Everything in Volumes 1-7 treated a biennium as a single snapshot. The IRT model saw 500 roll calls and estimated one ideology score per legislator. PCA compressed the entire vote matrix into two dimensions. Clustering found one set of groups. But a two-year session isn't a single event — it's a story with a beginning, middle, and end.

Legislative sessions have rhythms. The first weeks are calm: routine committee reports, naming resolutions, and noncontroversial bills. As the session progresses, contentious legislation emerges. By the final weeks — what the Kansas Statehouse calls "the vise" — controversial bills that have been deferred all session come to the floor under enormous time pressure. Party discipline tightens. Bipartisan cooperation evaporates.

This chapter watches that story unfold in real time, using two complementary lenses:

1. **Ideological drift** — Does the pattern of who agrees with whom shift over the course of the session?
2. **Party cohesion** — Do parties vote together more (or less) tightly as the session progresses?

And when the pattern breaks, it finds the exact moment.

## Ideological Drift: Rolling-Window PCA

### The Idea

Volume 3 applied PCA to the full vote matrix and extracted a single ideological dimension per legislator. But PCA on the full session averages across 500+ votes, smoothing over any within-session dynamics. What if a legislator who starts the session voting with the moderates drifts toward the party base by the end?

**Rolling-window PCA** applies PCA repeatedly to overlapping subsets of consecutive votes, like watching the legislature through a sliding window:

```
Window 1: Votes 1-75
Window 2: Votes 16-90
Window 3: Votes 31-105
...
```

Each window produces its own PCA dimension — its own snapshot of the ideological landscape at that point in the session. String the snapshots together, and you get a time series of ideology.

The analogy: imagine photographing a group of people standing in a room. A single photo captures where everyone is at that moment. But if you take a photo every five minutes, you can watch people migrate — some drift toward the door, others cluster around the refreshment table, a few move from one conversation to another. That's what rolling PCA does with legislators.

### The Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| **Window size** | 75 consecutive roll calls | Large enough for PCA to find stable patterns, small enough to detect within-session shifts |
| **Step size** | 15 roll calls between windows | 75% overlap ensures smooth transitions (each vote appears in ~5 windows) |
| **Minimum votes per legislator** | 10 within each window | Legislators who barely voted in a window are excluded from that snapshot |
| **Minimum legislators** | 20 per window | A window needs enough people for PCA to find meaningful structure |

With a typical session of 400-600 roll calls, this produces 20-35 overlapping windows — enough to trace the session's ideological arc without creating noise from tiny windows.

### Sign Alignment: Making the Windows Comparable

Each window's PCA is fit independently, which means PC1 might point "left" in one window and "right" in the next. The signs are arbitrary — PCA doesn't know which direction is "conservative."

Tallgrass aligns every window using the same convention: Republican mean PC1 must be positive. In most sessions, this is straightforward. But in sessions with the **horseshoe effect** (where intra-Republican factionalism dominates PC1 — see Volume 3), the simple party-mean rule can oscillate. For those sessions, Tallgrass uses a safer approach: correlate each window's PC1 with the full-session scores, and flip if the correlation is negative. This prevents the time series from appearing to oscillate when it's actually the sign convention that's unstable.

**Codebase:** `analysis/19_tsa/tsa.py` (`rolling_window_pca()`, `align_pc_signs()`)

### What the Drift Plot Shows

The primary output is a **party trajectory plot**: two lines (Republican red, Democrat blue) showing the mean PC1 score of each party across windows. In a typical session:

- **Early windows** show the parties closer together (bipartisan early-session votes pull everyone toward the center).
- **Middle windows** show gradual divergence as partisan legislation reaches the floor.
- **Late windows** show maximum separation as the session enters its final push.

A secondary plot shows the **polarization gap** — the absolute distance between the two party means — as a single line that rises or falls over time. When it rises, the parties are moving apart. When it falls, something is pulling them together.

### Top Movers: Who Drifted the Most?

For each legislator, Tallgrass computes an **early vs. late comparison**: their average PC1 in the first half of the session versus the second half. The difference is their drift. The top 10 legislators by absolute drift are highlighted.

A large drift doesn't necessarily mean the legislator changed their mind. It might mean the *issues* changed — early-session votes were on topics where this legislator votes with their party, while late-session votes were on topics where they cross the aisle. But the measurement is real: this legislator's voting pattern shifted relative to their colleagues.

**Codebase:** `analysis/19_tsa/tsa.py` (`WINDOW_SIZE = 75`, `STEP_SIZE = 15`, `MIN_WINDOW_VOTES = 10`, `MIN_WINDOW_LEGISLATORS = 20`, `TOP_MOVERS_N = 10`)

## Party Cohesion: The Rice Index Over Time

### Building the Time Series

The **Rice Index** (introduced in Volume 6) measures how unified a party is on a single vote:

```
Rice Index = |Yea - Nay| / (Yea + Nay)
```

**Plain English:** "What fraction of the party voted together, beyond what you'd expect by chance?"

A Rice Index of 1.0 means the party was unanimous. A Rice Index of 0.0 means it split exactly 50-50.

Tallgrass computes the Rice Index for every roll call, for each party. This creates a raw time series: one value per vote per party. But individual votes are noisy — a single unexpected vote doesn't mean the party is falling apart. To see the trend, Tallgrass aggregates into **weekly means**: the average Rice Index across all votes in each calendar week, per party.

### The Desposato Correction

Small caucuses (like the Kansas Senate Democratic caucus, often 8-12 members) have a systematic problem: even if members vote randomly, the Rice Index will be inflated just because small groups produce extreme splits by chance. Flip a coin 10 times and you'll often get 7-3 or 8-2 — a "Rice Index" of 0.40 or 0.60 — even though there's no coordination.

The **Desposato (2005) correction** removes this bias:

```
1. Compute the observed Rice Index for the vote
2. Simulate 10,000 random votes (coin flips) with the same group size
3. Compute the average Rice Index of the random votes (the "expected" baseline)
4. Subtract: Corrected Rice = max(0, Observed - Expected)
```

**Plain English:** "How much more cohesive is the party than a group of the same size voting randomly?"

For a 40-member caucus, the correction is small (random voting produces Rice near 0.10). For a 10-member caucus, it's substantial (random voting produces Rice near 0.25). Without the correction, you'd overestimate the cohesion of small parties.

**Codebase:** `analysis/19_tsa/tsa.py` (`desposato_corrected_rice()`, 10,000 simulations, seed=42)

## Changepoint Detection: Finding the Breaks

### The Analogy

Imagine you're monitoring a patient's heart rate. For an hour, it's steady at 70 beats per minute. Then it suddenly jumps to 90 and stays there. Then later it drops to 60. Those jumps are **changepoints** — moments when the underlying state changed.

The party cohesion time series works the same way. For several weeks, the majority party's Rice Index hovers around 0.85. Then something happens — a leadership challenge, a controversial bill, a committee shake-up — and it drops to 0.65 and stays there. That transition is a changepoint.

### PELT: Pruned Exact Linear Time

**PELT** is the algorithm that finds these breaks. It scans the weekly Rice time series and identifies the moments where the mean and/or variance of party cohesion shifts abruptly.

PELT uses an **RBF (Radial Basis Function) kernel**, which means it can detect changes in both the average level and the variability of cohesion. A party that goes from "consistently unified" (high mean, low variance) to "unpredictably split" (lower mean, high variance) would trigger a changepoint even if the average didn't drop much.

The critical parameter is the **penalty** — the cost of adding a changepoint. Think of it like the thermostat on a furnace:

- **Low penalty** (sensitive): The algorithm splits the series at every wiggle. You get many changepoints, most of which are noise.
- **High penalty** (conservative): The algorithm only detects the most dramatic shifts. You might miss real but moderate breaks.
- **Right penalty**: Captures the genuine structural breaks without overfitting to noise.

The default penalty is 10.0, with a minimum segment length of 5 weeks (to prevent splitting the series into week-long fragments).

**Joint detection** runs PELT on both parties simultaneously, stacking the Republican and Democrat Rice series into a 2D signal. This finds changepoints that affect the *entire chamber* — events that shifted both parties, not just one.

**Codebase:** `analysis/19_tsa/tsa.py` (`detect_changepoints_pelt()`, `detect_changepoints_joint()`, `PELT_PENALTY_DEFAULT = 10.0`, `PELT_MIN_SIZE = 5`)

### Sensitivity Analysis: How Many Breaks Are Real?

A single penalty gives a single answer. But how do you know the answer is right? Tallgrass runs a **sensitivity analysis** across 25 penalty values from 1 to 50:

```
Penalties: [1.0, 3.0, 5.0, ..., 48.0, 50.0]
```

For each penalty, PELT reports the number of changepoints. The result is a **step function**: at low penalties you might find 8 changepoints; as the penalty increases, the count drops to 5, then 3, then 1, then 0. The **elbow** — the penalty where the count drops most dramatically — suggests the natural number of breaks in the data.

The analogy: imagine you're adjusting the resolution on a satellite image. At maximum resolution, you see every pebble (too much detail). At minimum resolution, you see only the continents (too little). The useful resolution is somewhere in between — detailed enough to see the cities but not so detailed that you're counting individual buildings. The elbow is that useful resolution.

### CROPS: Automated Penalty Selection

For sessions where the R statistical environment is available, Tallgrass runs **CROPS** (Changepoints for a Range of Penalties) — a more principled version of the sensitivity analysis.

CROPS scans the penalty range [1.0, 50.0] and identifies the exact penalty thresholds where the optimal segmentation changes. Instead of evaluating 25 evenly spaced penalties, it finds the precise values where the number of changepoints increases or decreases. The elbow detection algorithm then identifies the penalty of maximum change — the boundary between "too many breaks" and "too few."

**Codebase:** `analysis/19_tsa/tsa_strucchange.R` (R subprocess), `analysis/19_tsa/tsa_r_data.py` (`find_crops_elbow()`, `CROPS_PEN_MIN = 1.0`, `CROPS_PEN_MAX = 50.0`)

### Bai-Perron: Confidence Intervals on the Breaks

PELT tells you *where* the breaks are but not *how certain* you should be about their locations. The **Bai-Perron structural break test** fills that gap by computing **95% confidence intervals** for each breakpoint.

The method:

```
1. Search for up to 5 structural breaks in the weekly Rice series
2. For each break, compute a 95% confidence interval:
   - Lower bound: the earliest week the break could plausibly be
   - Upper bound: the latest week
3. Cross-reference with PELT: if a Bai-Perron CI contains a PELT changepoint,
   that PELT detection is "confirmed"
```

A PELT changepoint that falls within a Bai-Perron confidence interval has two independent methods agreeing on roughly the same moment. A PELT changepoint with no Bai-Perron confirmation might be a false positive — a wiggle in the data that PELT found with a particular penalty but that doesn't survive formal testing.

The cross-reference uses a tolerance of 14 days (2 calendar weeks) to account for the different granularity of the two methods.

**Codebase:** `analysis/19_tsa/tsa_strucchange.R`, `analysis/19_tsa/tsa_r_data.py` (`parse_bai_perron_result()`, `merge_bai_perron_with_pelt()`, `BAI_PERRON_MAX_BREAKS = 5`)

## Putting It Together: A Kansas Example

In a typical Kansas House session, the analysis might find:

**Drift:** Party trajectories diverge steadily from January through March, plateau through April, then diverge further during the May vise. The top mover might be a moderate Republican whose early-session votes align with the bipartisan center but whose late-session votes track the conservative wing — pulled rightward by the increasingly partisan floor agenda.

**Changepoints:** PELT detects 2-3 changepoints in majority party cohesion. One falls in late February (when the budget bill reaches the floor and splits rural from suburban Republicans). Another falls in late April (when a controversial social policy bill forces members to choose between the party leadership and their district). The sensitivity analysis shows these breaks persist across a wide range of penalties. Bai-Perron confirms both with 95% intervals spanning 2-3 weeks each.

**Joint changepoints** identify moments when *both* parties shifted simultaneously — often corresponding to a major procedural event (leadership change, rules suspension) or an external shock (court ruling, federal action) that restructured the political dynamics of the chamber.

## What Can Go Wrong

### Too Few Votes for Rolling PCA

If a chamber has fewer than 75 roll calls (the window size), rolling PCA can't compute even a single window. The analysis reports this and skips the drift section. This is more likely in the Senate (fewer bills, fewer contested votes) than the House.

### Changepoints in Noise

With only 20-30 weekly observations, the Rice time series is short. PELT can find "changepoints" that are really just noise in a short series. The sensitivity analysis and Bai-Perron confirmation mitigate this, but for very short sessions, even confirmed breaks should be interpreted cautiously.

### Sign Oscillation in Horseshoe Chambers

In the 7 Senate sessions where PC1 captures intra-Republican factionalism rather than the party divide (see Vol. 3), the rolling PCA drift plots may show spurious oscillations. The horseshoe-safe sign alignment (correlating with full-session scores) handles this, but the drift scores themselves are measuring a different dimension in those windows — faction membership rather than ideology.

### R Availability

The CROPS and Bai-Perron analyses require R with the `changepoint` and `strucchange` packages. If R isn't installed, the analysis runs Python-only (PELT and sensitivity analysis), which provides the changepoint locations but not the formal confidence intervals. The report notes which enrichments were available.

---

## Key Takeaway

Rolling-window PCA tracks how the ideological landscape shifts within a session, while weekly Rice Index time series measure party cohesion over time. PELT changepoint detection finds the moments when cohesion breaks — confirmed by Bai-Perron confidence intervals and calibrated by CROPS penalty selection. The combination reveals the legislative session not as a static snapshot but as a sequence of political regimes, each triggered by an identifiable event.

---

*Terms introduced: rolling-window PCA, sign alignment (horseshoe-safe), ideological drift, party trajectory, polarization gap, Desposato correction, PELT (Pruned Exact Linear Time), changepoint, RBF kernel, penalty (changepoint sensitivity), sensitivity analysis, elbow, CROPS (Changepoints for a Range of Penalties), Bai-Perron structural break test, confidence interval (on changepoints), joint changepoint detection*

*Next: [Dynamic Ideal Points: The Martin-Quinn Model](ch02-dynamic-ideal-points.md)*
