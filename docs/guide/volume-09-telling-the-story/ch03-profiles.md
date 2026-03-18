# Chapter 3: Legislator Profiles: Individual Deep Dives

> *The synthesis report tells you who the mavericks and bridge-builders are. The profile report tells you why — bill by bill, vote by vote, neighbor by neighbor.*

---

## From Summary to Deep Dive

The synthesis report (Chapter 2) identifies the session's most interesting legislators and summarizes their behavior in a sentence or two. But a sentence isn't enough if you're a journalist writing a story about Representative Davis's voting pattern, or a constituent wondering exactly which bills your senator broke from the party on.

Phase 25 produces **individual legislator profiles** — standalone deep dives that include a scorecard of six normalized metrics, a breakdown of how they vote on partisan versus routine bills, a ranked list of their most consequential defections, their closest and most-different voting neighbors, and the votes where the prediction model was most surprised by their behavior.

The analogy: the synthesis report is like a team roster that lists each player's batting average. The profile report is the player card — complete with stats, highlights, and comparisons to the rest of the league.

## Who Gets Profiled?

### Automatic Detection

Phase 25 reuses the same detection algorithms from Phase 24 (Chapter 2). The mavericks, bridge-builders, and paradox legislators identified by `detect_all()` are automatically queued for deep-dive profiles. In a typical session, this produces 3-6 profiles — enough to cover the most interesting behavior in both chambers without overwhelming the report.

### User-Requested Profiles

The automatic detection finds the statistically unusual. But sometimes a reader cares about a specific legislator who isn't unusual at all — their own representative, a committee chair, a candidate in an upcoming election.

Phase 25 accepts two CLI flags for manual additions:

**`--names "Masterson,Blake Carpenter"`** — Name-based lookup. The algorithm resolves names through three stages:

1. **Exact match:** Case-insensitive comparison against full names in the legislature. "John Alcala" matches "John Alcala" directly.

2. **Last-name match:** If no exact match, try last name only. "Carpenter" might match "Blake Carpenter" and "Tory Marie Arnberger-Blew" if there were a Carpenter — but in practice, last names are usually unique enough.

3. **First-name disambiguation:** If the last name matches multiple legislators (two "Smiths"), the algorithm uses the first name from the query to narrow it down.

The name resolution uses the same `normalize_name()` function from the cross-session alignment (Volume 8, Chapter 3) — stripping leadership suffixes, normalizing whitespace, and handling case differences.

**`--slugs rep_masterson_ty_1,sen_alcala_john_1`** — Direct slug lookup. Unambiguous, no resolution needed.

### The Cap

A maximum of **8 profiles** per report. Automatic detections fill the first slots; user requests fill the remainder. If more than 8 are requested, the least ideologically extreme (by `|xi_mean|`) are dropped first — the assumption being that extreme legislators have more interesting stories.

**Codebase:** `analysis/25_profiles/profiles_data.py` (`gather_profile_targets()`, `resolve_names()`, `MAX_PROFILE_TARGETS = 8`)

## The Scorecard: Six Metrics at a Glance

Each profile starts with a **scorecard** — a horizontal bar chart showing six normalized metrics on a 0-to-1 scale, with the party average marked for comparison.

| Metric | Source Phase | What It Measures | 0 Means | 1 Means |
|--------|-------------|------------------|---------|---------|
| **Ideological Rank** | IRT (Phase 05/06) | Percentile position in the chamber's ideology distribution | Most liberal | Most conservative |
| **Party Unity (CQ)** | Indices (Phase 13) | Fraction of party-line votes where they voted with their party | Always defects | Always loyal |
| **Clustering Loyalty** | Clustering (Phase 09) | How consistently they vote with their assigned cluster (k=2) | Frequent cross-overs | Perfect cluster member |
| **Maverick Rate** | Indices (Phase 13) | Fraction of close votes where they voted against their party | Never rebels | Always rebels |
| **Network Influence** | Network (Phase 11) | Percentile of betweenness centrality — how many connections pass through them | Peripheral | Central hub |
| **Prediction Accuracy** | Prediction (Phase 15) | Fraction of their votes the XGBoost model predicted correctly | Unpredictable | Perfectly predictable |

The choice of these six metrics is deliberate. They span four analytical methods (IRT, clustering, network analysis, prediction) and capture three dimensions of legislative behavior (ideology, loyalty, structural position). All are on a 0-1 scale, making direct comparison meaningful.

The scorecard visualization shows the legislator's value as a colored bar (party color) overlaid on the full 0-1 range. A small marker shows the party average. At a glance, you can see: "This legislator has high ideology rank and network influence, but low party unity and loyalty — they're an ideologically extreme but independent voter."

**Codebase:** `analysis/25_profiles/profiles_data.py` (`build_scorecard()`, `SCORECARD_METRICS`)

## Bill Type Breakdown: Partisan vs. Routine

Not all votes are created equal. Some bills sharply divide the parties (a tax bill, a gun regulation, an education funding measure). Others pass near-unanimously (naming a highway, a procedural motion, a noncontroversial technical fix). A legislator who votes with their party 95% of the time on routine bills but only 70% on partisan bills is telling a different story than one who's at 95% across the board.

The **bill type breakdown** classifies every bill by its IRT discrimination parameter (beta, from Volume 4):

| Category | Threshold | What It Means |
|----------|-----------|---------------|
| **High discrimination** | \|beta\| > 1.5 | The bill sharply separates legislators along the ideological dimension — a partisan vote |
| **Low discrimination** | \|beta\| < 0.5 | The bill barely discriminates — most legislators vote the same way regardless of ideology |

For each category, the profile computes the legislator's Yea rate and compares it to the party average:

```
Legislator Yea rate on partisan bills: 72%
Party average Yea rate on partisan bills: 91%
Gap: -19 percentage points
```

If the gap exceeds 10 percentage points, the visualization annotates it: "Representative Davis votes Yea 19% less often than Republicans on partisan bills."

A legislator who matches their party on routine bills but diverges on partisan ones is making strategic choices — they're loyal on easy votes and independent on hard ones. A legislator who diverges on both is genuinely cross-pressured.

The analysis requires at least 3 bills per category. If a session has fewer than 3 high-discrimination bills (unlikely in Kansas, but possible in a short special session), the breakdown is skipped.

**Codebase:** `analysis/25_profiles/profiles_data.py` (`compute_bill_type_breakdown()`, `HIGH_DISC_THRESHOLD = 1.5`, `LOW_DISC_THRESHOLD = 0.5`, `MIN_BILLS_PER_TIER = 3`)

## Defection Votes: When They Broke from the Party

The defection analysis answers the most specific question a constituent might ask: "Which bills did my representative vote against the party on?"

### How It Works

1. For every roll call, compute the **party majority vote** — did more than 50% of the party vote Yea? If so, the party's position is Yea.

2. Identify every vote where the profiled legislator disagreed with the party majority. These are the defections.

3. Rank the defections by **margin closeness** — how close the party's internal vote was to 50-50. A defection on a vote where the party split 51-49 is more dramatic (and arguably more consequential) than a defection on a vote where the party was at 90-10 and the legislator was one of only a handful of dissenters.

4. Return the top 15, with bill metadata (bill number, title, motion, sponsor) so the reader can identify the specific legislation.

The analogy: imagine a list of every time an employee disagreed with their boss, ranked by how controversial the disagreement was. The disagreements where the whole team was evenly split are at the top — those are the moments that defined whether the employee was a team player or an independent thinker.

The visualization is a horizontal bar chart. Each bar represents one defection bill, showing the party's Yea percentage (the gray bar) with a diamond marker showing where the profiled legislator voted (Yea = right edge, Nay = left edge). You can immediately see: on these 15 bills, the party was at various levels of agreement, and the profiled legislator was always on the other side.

**Codebase:** `analysis/25_profiles/profiles_data.py` (`find_defection_bills()`)

## Voting Neighbors: Who They Vote Like

The voting neighbor analysis answers: "If you took away the party labels and just looked at who votes the same way, who would you group this legislator with?"

### The Computation

1. Build a **vote matrix** — rows are vote IDs, columns are legislator slugs, values are 1 (Yea) or 0 (Nay). Absent legislators have null entries.

2. For each pair of legislators who share at least 5 votes, compute **simple agreement** — the fraction of shared votes where both voted the same way.

```
Agreement(A, B) = (votes where A and B both voted Yea + votes where both voted Nay)
                  ÷ (total votes where both A and B cast a vote)
```

3. Rank all legislators by agreement with the profiled target. The top 5 are the **closest neighbors** — the five legislators whose voting record most closely mirrors the target's. The bottom 5 are the **most different** — the legislators whose behavior is most opposite.

The 5-vote minimum prevents spurious results. Two legislators who both voted on only 3 bills might agree on all 3 by chance; requiring 5 shared votes provides a minimal baseline for meaningful comparison.

The visualization is a two-panel bar chart. The left panel shows the 5 closest neighbors with their agreement percentages and party colors. The right panel shows the 5 most different. If the profiled legislator is a maverick Republican, their closest neighbors might include a Democrat or two — confirming the maverick finding from a completely different angle.

**Codebase:** `analysis/25_profiles/profiles_data.py` (`find_voting_neighbors()`)

## Surprising Votes: Where the Model Was Wrong

The prediction model (Phase 15, Volume 7) assigns a probability to every vote: "There is a 95% chance Representative Davis votes Yea on HB 2032." When she votes Nay, the model was surprised — and the degree of surprise is measured by the **confidence error**: how confident the model was in the wrong answer.

The surprising votes table shows the 10 votes where the model was most wrong about this legislator, sorted by confidence error. Each row includes:

| Column | What It Shows |
|--------|--------------|
| **Bill** | The bill number |
| **Motion** | What was being voted on (passage, amendment, veto override) |
| **Actual** | How the legislator actually voted |
| **Predicted** | What the model expected |
| **Model Confidence** | How sure the model was (as a percentage) |
| **Surprise Score** | The confidence error — high means the model was very wrong |

A vote where the model predicted Yea with 97% confidence but the legislator voted Nay gets a surprise score near 0.97. That's a vote worth investigating — something about that bill made this legislator break from the pattern the model learned from their other 500+ votes.

Surprising votes often cluster around specific policy areas. A Republican maverick might surprise the model consistently on education funding bills but never on tax bills. The pattern reveals which issues drive the legislator's independence.

**Codebase:** `analysis/25_profiles/profiles_data.py` (`find_legislator_surprising_votes()`)

## Sponsorship and Full Voting Record

### Sponsorship Stats

If the scraper captured bill sponsor data (the `sponsor_slugs` column in rollcalls), the profile includes a sponsorship summary: how many bills the legislator sponsored, how many as primary sponsor (listed first), and what fraction passed.

A legislator who sponsors 25 bills with a 40% passage rate tells a different story than one who sponsors 3 bills that all passed. Volume of sponsorship suggests legislative activity; passage rate suggests effectiveness (or at least alignment with the majority's agenda).

### Full Voting Record

When a user requests a specific legislator by name (`--names`), the profile includes their **complete voting record** — every Yea and Nay vote they cast, with bill metadata, the party majority position, a flag for whether they voted with the party, and the bill's outcome.

This is the most data-intensive section: a searchable, sortable interactive table with potentially 500+ rows. It's designed for the reader who wants to audit a specific legislator's record in detail — a journalist fact-checking a claim, a constituent tracking their representative, or a researcher building a dataset.

The full record is optional (suppressed by default) because it significantly increases report size. The `--names` flag auto-enables it, on the assumption that name-based lookups indicate deep-dive interest.

**Codebase:** `analysis/25_profiles/profiles_data.py` (`compute_sponsorship_stats()`, `build_full_voting_record()`)

## The Position-in-Context Plot

One final visualization places the profiled legislator in the context of their same-party colleagues. It's a **forest plot** — a vertical list of every same-party member's IRT ideal point, sorted from most moderate to most extreme, with 95% credible intervals shown as horizontal lines.

The profiled legislator is highlighted with a diamond marker and a yellow background bar. You can immediately see: "Representative Davis is the 5th most conservative Republican out of 85, with a credible interval that doesn't overlap with the moderate wing."

This answers the question: "Is this legislator unusual among their own party, or are they in the mainstream?" A maverick who is ideologically centrist but low on party unity is a different case than one who is ideologically extreme and low on unity.

**Codebase:** `analysis/25_profiles/profiles.py` (`plot_position_in_context()`)

## Putting It Together: A Profile Example

Imagine the synthesis report identified Representative Taylor as the House maverick. Her profile would include:

1. **Header:** "Taylor, Representative (R-42) — House Maverick. Voted with Republicans on 71% of party-line votes — 21 points below the caucus average."

2. **Scorecard:** High ideological rank (she's conservative), low party unity (71%), low clustering loyalty (she frequently crosses cluster boundaries), moderate maverick rate, low network influence (she's on the periphery), and moderate prediction accuracy (the model gets her right about 85% of the time).

3. **Bill Type Breakdown:** She votes Yea 68% of the time on partisan bills (party average: 89%) but 94% on routine bills (party average: 96%). Her independence is concentrated on the hard votes.

4. **Top Defections:** Her 15 most consequential defections, ranked by how close the party was to splitting. Three are education funding bills, two are tax bills, four are social policy — her independence has a pattern.

5. **Voting Neighbors:** Her closest neighbor is a moderate Republican; her 2nd-closest is a Democrat. Her most-different colleague is the most conservative Republican in the chamber.

6. **Surprising Votes:** The prediction model's 10 biggest misses on Taylor. Seven are education-related — the model hasn't learned that Taylor breaks from the party specifically on education, because the IRT model treats all votes equally.

7. **Position in Context:** Among 85 Republicans, Taylor is the 12th most conservative. She's ideologically in the mainstream but behaviorally independent — the paradox that synthesis detected.

## What Can Go Wrong

### Insufficient Vote Data

A legislator who only voted on 20 bills (perhaps appointed mid-session) will have thin data for every analysis. The scorecard metrics may be unreliable, the bill-type breakdown may lack enough bills per tier, and the voting neighbor computation may fail the 5-shared-vote minimum with some colleagues. The profile still runs but should be interpreted cautiously.

### Name Resolution Ambiguity

If a user requests "Smith" and three legislators share that last name, the resolution returns all three as "ambiguous" — the user sees a message listing the matches and can re-run with a more specific name or a slug. The algorithm never guesses.

### Missing Upstream Phases

Profiles need IRT (for ideology and bill parameters), Indices (for party unity and maverick rates), and the raw vote CSVs. If any of these are missing, the profile skips the affected sections. A profile with only IRT data still shows the position-in-context plot but lacks the scorecard, defections, and neighbors.

### Disconnected Networks

When the co-voting network has no edges between parties (a common pattern in highly polarized sessions), the voting neighbor analysis may not find cross-party neighbors. The 5 "most different" legislators will all be from the other party with agreement rates near 0% — accurate but uninformative. The profile handles this by noting the partisan separation rather than presenting it as a finding.

---

## Key Takeaway

Legislator profiles (Phase 25) produce individual deep dives for 3-8 notable legislators, combining six normalized scorecard metrics, a bill-type breakdown (partisan vs. routine using IRT discrimination), ranked defection votes (sorted by party margin closeness), pairwise voting neighbor analysis, and the prediction model's most surprising misses. Profile targets are selected automatically (mavericks, bridge-builders, paradoxes) and can be supplemented by user request via name matching or slug lookup. Every section degrades gracefully when data is unavailable — the report adapts to what the pipeline produced.

---

*Terms introduced: legislator profile, scorecard (6 normalized metrics), bill discrimination tier (high > 1.5, low < 0.5), defection vote (disagreement with party majority), margin closeness, voting neighbor (pairwise simple agreement), surprising vote (confidence error), full voting record, name resolution (exact, last-name, first-name disambiguation), profile target cap*

*Previous: [Synthesis: The Session Story](ch02-synthesis.md)*

*Next: [How to Read a Tallgrass Report](ch04-how-to-read-report.md)*
