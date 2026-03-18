# Chapter 5: Classical Indices: Rice, Party Unity, and the Maverick Score

> *Before Bayesian models and graph theory, political scientists had simpler tools — and some of them date back a century. The Rice Index was published in 1925. These measures lack the sophistication of IRT or network analysis, but they answer direct questions in direct ways: How unified is each party? Who defects the most? Is this a two-party legislature or something more complicated?*

---

## The Value of Simple Measures

The previous four chapters used sophisticated statistical machinery: hierarchical clustering with cophenetic validation, Bernoulli mixture models with BIC selection, network centrality with Leiden community detection, and BiCM backbone extraction with maximum-entropy null models.

These methods are powerful, but they're also black boxes to most readers. A journalist covering the Kansas Legislature doesn't need eigenvector centrality. They need to know: "How often does Senator Smith vote with her party?" That question has a simple, transparent answer — and it's been answered the same way for a hundred years.

Classical political science indices provide **interpretable, comparable, well-established** measures of legislative behavior. They don't replace the Bayesian models; they complement them. When a Tallgrass report says a legislator has a party unity score of 72%, any reader immediately understands that this person breaks ranks 28% of the time. No statistical training required.

## The Rice Index (1925)

### The Oldest Measure

**Stuart A. Rice** introduced the Rice Index in 1925 while a graduate student at Columbia University. His paper, "The Behavior of Legislative Groups: A Method of Measurement" in *Political Science Quarterly*, proposed the first systematic metric for how cohesively a party votes.

The formula is disarmingly simple:

```
Rice Index = |Yea − Nay| / (Yea + Nay)
```

Where Yea and Nay are the counts of legislators from one party on one roll call.

**Step-by-step example:**

On a particular bill, 30 House Republicans voted Yea and 10 voted Nay:

```
Rice = |30 − 10| / (30 + 10) = 20 / 40 = 0.50
```

A Rice Index of 0.50 means 75% of the party voted one way and 25% the other. The party was divided, but not evenly.

Now consider a party that voted 38 Yea and 2 Nay:

```
Rice = |38 − 2| / (38 + 2) = 36 / 40 = 0.90
```

Nearly unanimous — high cohesion.

And a perfectly split party: 20 Yea, 20 Nay:

```
Rice = |20 − 20| / (20 + 20) = 0 / 40 = 0.00
```

Zero cohesion. The party provides no predictive signal on this vote.

| Rice Index | What it means |
|------------|--------------|
| 0.00 | 50/50 split — no party cohesion |
| 0.50 | 75/25 split — divided but leaning |
| 0.80 | 90/10 split — strongly unified |
| 1.00 | Unanimous — perfect party cohesion |

### Rice Over Time

The Rice Index is computed per party per vote. By plotting it as a rolling average across the session's roll calls (in chronological order), you can see how party cohesion changes over time.

In Kansas, Rice Index patterns tell a seasonal story. Early in the session (January-February), many bills are noncontroversial and both parties show high Rice scores (everyone agrees). As the session progresses and more partisan legislation reaches the floor (March-May), the minority party's Rice Index can drop — Democrats sometimes split on which losing battles to fight. The majority party's Rice tends to stay high because they control the agenda and only bring bills to the floor when they have the votes.

**Codebase:** `analysis/13_indices/indices.py` (`compute_rice_index()`, `compute_rice_summary()`, `plot_rice_over_time()`, `ROLLING_WINDOW = 15`)

### Rice by Vote Type

Not all roll calls are alike. The Kansas Legislature holds votes on several types of motions:

- **Final Action** — the up-or-down vote on a bill
- **Committee of the Whole** — a procedural vote in the full chamber acting as a committee
- **Emergency Final Action** — bypasses the normal scheduling delay (same simple-majority threshold as Final Action)
- **Veto Override** — requires 2/3 to overturn the governor's veto
- **Concurrence** — agreeing to the other chamber's amendments

Tallgrass computes the Rice Index separately for each vote type. Final Action and Emergency Final Action votes tend to be the most partisan (lower Rice for the minority party). Consent Calendar votes are nearly unanimous (Rice close to 1.0 for both parties). Veto overrides show unusual patterns because they sometimes create bipartisan coalitions.

**Codebase:** `analysis/13_indices/indices.py` (`compute_rice_by_vote_type()`)

### Fractured Votes

When the majority party's Rice Index drops below 0.50 on a bill, that's a **fractured vote** — the ruling party split roughly evenly. These votes are politically significant because they reveal internal fault lines. A bill that fractures the Republican caucus 45-40 passed only because some Democrats joined the "wrong" faction of Republicans.

Tallgrass flags all votes where the majority party's Rice falls below the fracture threshold and reports them in the analysis.

**Codebase:** `analysis/13_indices/indices.py` (`find_fractured_votes()`, `RICE_FRACTURE_THRESHOLD = 0.50`)

## The Carey UNITY Index

The Rice Index has a blind spot: **it ignores absences.** If a 40-member party has 30 vote Yea, 5 vote Nay, and 5 are absent, the Rice Index is |30 − 5| / (30 + 5) = 0.71. But should those 5 absences count as disunity? The party only got 30 out of 40 members to show up and vote the party line.

Political scientist **John Carey** (of the Legislative Voting and Decisions project) proposed the **Carey UNITY** variant that uses total party members in the denominator:

```
Carey UNITY = |Yea − Nay| / total_party_members
```

For the same example:

```
Carey UNITY = |30 − 5| / 40 = 25 / 40 = 0.625
```

The Carey version is always less than or equal to Rice, and the gap measures how much absences reduce unity. When the two diverge sharply, it suggests that some party members are dodging controversial votes — a pattern the strategic absence diagnostic (Volume 3, Chapter 2) is designed to detect.

**Codebase:** `analysis/13_indices/indices.py` (`compute_carey_unity()`)

## Party Unity Score (CQ Standard)

### The Legislator-Level View

The Rice Index is per-vote, per-party. The **party unity score** is per-legislator: "How often does this legislator vote with their party?"

Tallgrass uses the **Congressional Quarterly (CQ) standard**, which has been the benchmark in American politics since CQ began tracking congressional voting in the 1950s:

1. **Identify party votes:** Roll calls where a majority of Republicans opposed a majority of Democrats. Both parties must have at least 2 members voting for this to count.

2. **For each legislator:** Count how many party votes they participated in, and how many times they voted with their party's majority.

```
Party Unity = votes_with_party / party_votes_present
```

**Step-by-step example:**

Suppose there were 200 party votes in the session. Representative Jones was present for 180 of them and voted with her party's majority on 162:

```
Unity = 162 / 180 = 0.90
```

She votes with her party 90% of the time on contested votes. The other 10% are her moments of independence — or defection, depending on who you ask.

### Why Only Party Votes?

The CQ standard doesn't count *every* roll call — only the ones where the parties opposed each other. This is deliberate. If a bill passes 120-5 with bipartisan support, voting Yea doesn't tell you anything about party loyalty. Party unity only matters when the parties disagree.

This is a more restrictive definition than the one used in clustering (Chapter 1), which counted any vote with at least 10% party dissent as "contested." The CQ standard requires *majority-vs.-majority* conflict, which typically captures about 30-50% of all roll calls in Kansas.

**Codebase:** `analysis/13_indices/indices.py` (`identify_party_votes()`, `compute_unity_and_maverick()`, `PARTY_VOTE_THRESHOLD = 0.50`)

## Maverick Scoring: Who Breaks Ranks?

### Unweighted Mavericks

The simplest maverick score is just the complement of party unity:

```
Maverick Rate = 1 − Unity Score
```

A legislator with 90% unity has a 10% maverick rate. One with 72% unity has a 28% maverick rate. The higher the maverick rate, the more often they defect from their party.

### Weighted Mavericks: Close Calls vs. Blowouts

Not all defections are created equal. A legislator who defects on a 52-48 vote is doing something politically significant — their vote could have swung the outcome. A legislator who defects on a 90-10 blowout is making a symbolic gesture that changes nothing.

The **weighted maverick score** accounts for this by giving more weight to defections on close votes:

```
closeness_weight = 1 / max(|Yea − Nay| / (Yea + Nay), 0.01)
```

**Plain English:** The closer the vote, the higher the weight. A 60-60 tie gets maximum weight. A 100-20 blowout gets minimal weight.

Then:

```
Weighted Maverick = sum(closeness_weight × defection) / sum(closeness_weight)
```

This separates **strategic defectors** (who break ranks on close votes where it matters) from **performative defectors** (who make symbolic gestures on votes their party would win anyway).

### The Maverick Landscape

Plotting unweighted maverick (x-axis) against weighted maverick (y-axis) creates a landscape with four quadrants:

| Quadrant | Interpretation |
|----------|---------------|
| Low unweighted, low weighted | Party loyalist — rarely defects |
| High unweighted, low weighted | Performative defector — breaks ranks on blowouts |
| Low unweighted, high weighted | Strategic defector — rarely defects, but does so on close votes |
| High unweighted, high weighted | Genuine maverick — frequently defects, especially when it matters |

The upper-right quadrant is where the truly independent legislators live. The lower-right is where the "show voters" hide — they want to look independent without actually changing outcomes.

**Codebase:** `analysis/13_indices/indices.py` (`compute_unity_and_maverick()`, `MAVERICK_WEIGHT_FLOOR = 0.01`)

## Co-Defection Analysis

When a legislator defects from their party, do they defect alone or with company? **Co-defection analysis** looks at the majority party's defectors and asks which ones tend to break ranks together.

Tallgrass builds a **co-defection matrix** for the top 20 defectors: a square table where the value in row *i*, column *j* is the number of party votes where both legislator *i* and legislator *j* defected together.

Pairs with high co-defection counts are likely part of an informal subcaucus — a group that shares a policy orientation distinct from the party mainstream. In Kansas, co-defection clusters in the Republican caucus often correspond to moderate Republicans from suburban districts who cooperate on education and healthcare votes.

**Codebase:** `analysis/13_indices/indices.py` (`compute_co_defection_matrix()`, `CO_DEFECTION_MIN = 3`, `TOP_DEFECTORS_N = 20`)

## Effective Number of Parties

### The Question

Kansas has two major parties. But does the legislature *behave* like a two-party system?

If one party holds 90% of the seats, it's effectively a one-party legislature — the minority can't block anything. If three parties each hold a third, it's effectively three parties. The **Effective Number of Parties** (ENP), introduced by **Markku Laakso and Rein Taagepera** in 1979, captures this with a single number.

### The Formula

```
ENP = 1 / sum(share_i²)
```

Where share_i is each party's proportion of seats.

**Step-by-step example:**

The Kansas House has 85 Republicans and 40 Democrats (approximate):

```
share_R = 85/125 = 0.68
share_D = 40/125 = 0.32

ENP = 1 / (0.68² + 0.32²) = 1 / (0.4624 + 0.1024) = 1 / 0.5648 = 1.77
```

An ENP of 1.77 means the Kansas House behaves like a system with fewer than two equally sized parties. The Republican supermajority makes it more like a 1.8-party system — the Democrats exist but have limited influence.

For comparison:

| Seat distribution | ENP | Interpretation |
|-------------------|-----|---------------|
| 100/0 | 1.00 | One-party system |
| 80/20 | 1.47 | Dominant-party system |
| 68/32 | 1.77 | Weakly competitive two-party |
| 50/50 | 2.00 | Fully competitive two-party |
| 33/33/33 | 3.00 | Three-party system |

The formula is mathematically identical to the **inverse of the Herfindahl-Hirschman Index** used in economics to measure market concentration. A monopoly has HHI = 1 (and ENP = 1). A perfectly competitive market has HHI near 0 (and ENP near infinity).

### Per-Vote ENP

Seat-based ENP is a static, structural measure. But voting behavior can be more or less fragmented than seat counts suggest. Tallgrass also computes **per-vote ENP** by treating each (party, direction) combination as a "bloc":

On a given vote, the blocs might be:
- Republicans voting Yea (70 members)
- Republicans voting Nay (15 members)
- Democrats voting Yea (5 members)
- Democrats voting Nay (35 members)

That's 4 blocs with shares 0.56, 0.12, 0.04, 0.28, giving ENP = 2.9 — higher than the seat-based ENP of 1.77. This vote *behaves* like a multi-party vote because the Republican caucus split internally.

Averaging per-vote ENP across the session gives a behavioral measure of fragmentation that complements the structural seat-based measure.

**Codebase:** `analysis/13_indices/indices.py` (`compute_enp_seats()`, `compute_enp_per_vote()`, `ENP_MULTIPARTY_THRESHOLD = 2.5`)

## Bipartisanship Index

The **bipartisanship index** (modeled on the Lugar Center's methodology) asks a different question than party unity: not "how often do you vote with *your* party?" but "how often do you vote with the *other* party?"

```
BPI = votes_with_opposition_majority / party_votes_present
```

A legislator who votes with the opposing party's majority on 25% of party votes has a BPI of 0.25. This is *not* the same as a 25% maverick rate — a maverick votes against their own party, which doesn't necessarily mean they voted *with* the other party (they might have been absent or voted a third way).

BPI identifies legislators who actively collaborate across the aisle, as opposed to those who merely fail to show up for their own party's position.

**Codebase:** `analysis/13_indices/indices.py` (`compute_bipartisanship_index()`)

## Plus-Minus: Above or Below Average?

The simplest way to contextualize a legislator's unity score: compare it to their party's average.

```
Plus-Minus = unity_score − party_mean_unity
```

A Plus-Minus of +0.05 means this legislator is 5 percentage points more loyal than the average party member. A Plus-Minus of −0.12 means they're 12 points less loyal.

This is like plus-minus in basketball: it doesn't tell you how good you are in absolute terms, but it tells you how you compare to the baseline. A legislator in a very cohesive party (average unity 95%) who scores 90% has a Plus-Minus of −0.05 — they look moderate by comparison. The same 90% in a fractured party (average 80%) would be +0.10 — relatively loyal.

**Codebase:** `analysis/13_indices/indices.py` (`compute_plus_minus()`)

---

## Key Takeaway

Classical indices strip legislative behavior down to its simplest, most interpretable components. The Rice Index (1925) tells you how unified a party was on each vote. Party unity (CQ standard) tells you how loyal each legislator is overall. The maverick score — especially the weighted version — distinguishes genuine mavericks from performative defectors. Co-defection analysis reveals informal subcaucuses. ENP quantifies whether the legislature behaves like a one-party, two-party, or multi-party system. These measures lack the nuance of Bayesian models but gain transparency and comparability: any reader can understand a party unity score, and it can be compared directly across decades and across states.

---

*Terms introduced: Rice Index, cohesion, fractured vote, Carey UNITY, party vote (CQ standard), party unity score, maverick rate, weighted maverick, closeness weight, co-defection matrix, Effective Number of Parties (ENP), Herfindahl-Hirschman Index, per-vote ENP, bipartisanship index (BPI), Plus-Minus*

*Next: [Empirical Bayes: Shrinkage Estimates of Party Loyalty](ch06-empirical-bayes.md)*
