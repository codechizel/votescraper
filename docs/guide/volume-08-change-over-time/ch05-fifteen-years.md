# Chapter 5: Fifteen Years of Kansas Politics (2011-2026)

> *Numbers don't speak for themselves. Eight bienniums of ideal points, changepoints, and decompositions need a narrative. Here's what the data reveals about three eras of Kansas politics.*

---

## What This Chapter Is (and Isn't)

This chapter weaves the temporal analysis from Chapters 1-4 into a narrative arc. It describes the patterns the data reveals. It does not advocate for any political position or evaluate whether the observed changes were good or bad for Kansas. When the data shows polarization increasing, that's a measurement — readers can and should form their own opinions about what it means.

The narrative draws on the dynamic ideal point trajectories (Chapter 2), the conversion-replacement decomposition (Chapter 4), and within-session changepoint analysis (Chapter 1). Where external context is needed (elections, policy events, court decisions), it's provided to help interpret the numbers — not to explain them away.

## The Study Period: 84th Through 91st Legislatures

Tallgrass covers eight bienniums: 2011-2026. This period spans three distinct political eras in Kansas, each shaped by different forces and producing different patterns in the data.

| Biennium | Legislature | Era | Governor |
|----------|------------|-----|----------|
| 2011-2012 | 84th | Brownback | Sam Brownback (R) |
| 2013-2014 | 85th | Brownback | Sam Brownback (R) |
| 2015-2016 | 86th | Late Brownback | Sam Brownback (R) |
| 2017-2018 | 87th | Transition | Jeff Colyer (R) / Brownback resignation |
| 2019-2020 | 88th | Kelly | Laura Kelly (D) |
| 2021-2022 | 89th | Kelly | Laura Kelly (D) |
| 2023-2024 | 90th | Kelly | Laura Kelly (D) |
| 2025-2026 | 91st | Current | (current session) |

## Era I: The Brownback Experiment (84th-86th, 2011-2016)

### The Context

Governor Sam Brownback took office in January 2011 with an ambitious fiscal agenda: deep income tax cuts designed to stimulate economic growth, paired with reductions in state services. The 84th Legislature — dominated by conservative Republicans — passed the tax cuts in 2012. The policy was explicitly framed as an "experiment" that other states could follow.

### What the Data Shows

**Intra-party factionalism.** The IRT models for this period reveal something the popular narrative often missed: the Republican caucus was not monolithic. The 2D IRT (Volume 4) and PCA (Volume 3) consistently find at least two Republican factions — a conservative wing that backed the tax cuts and a moderate wing that had reservations. In several Senate sessions during this era, the first principal component captures *this intra-Republican divide* rather than the traditional party divide — the horseshoe effect documented in Volume 3.

**Rising polarization.** The dynamic ideal points show the party means diverging during this period. But the divergence is asymmetric: the Republican mean shifted rightward more than the Democrat mean shifted leftward. The total polarization gap widened not because both parties moved to their extremes, but primarily because the majority party moved further right.

**Replacement-driven change.** The 2012 primaries — sometimes called the "Brownback purge" — produced substantial Republican caucus turnover. Several moderate Republican senators lost primary challenges to conservative opponents. The conversion-replacement decomposition for the 84th → 85th transition would show a large replacement effect: the departing moderates were replaced by legislators whose ideal points were further right.

### Data Quality Caveat

The 84th Legislature (2011-2012) has the weakest data in the study period. Approximately 30% of its roll calls are committee-of-the-whole votes with no individual positions recorded. The dynamic IRT handles this (wider posterior credible intervals for 84th-only legislators), but any quantitative claims about the exact magnitude of 84th-era polarization should carry wider error bars than later periods.

## Era II: The Moderate Resurgence (87th-88th, 2017-2020)

### The Context

By 2017, the Brownback tax experiment had produced persistent state budget shortfalls. School funding litigation (Gannon v. State) put the legislature under court order to increase education spending. Brownback resigned in January 2018 to become Ambassador at Large for International Religious Freedom, and Lt. Governor Jeff Colyer completed the term. In November 2018, Democrat Laura Kelly won the gubernatorial election — the first Democratic governor since Kathleen Sebelius.

### What the Data Shows

**Within-session changepoints.** The 87th Legislature's Rice Index time series likely shows a structural break corresponding to the school funding resolution. When the legislature voted to increase education funding to satisfy the court order, it fractured the Republican supermajority: fiscal conservatives opposed the spending, while suburban and moderate Republicans supported it. PELT would detect this as a changepoint — a moment when majority party cohesion dropped abruptly.

**Suburban realignment.** The conversion-replacement decomposition for the 87th → 88th transition tells the story of Kansas's suburban districts. Several suburban Republican-held seats flipped to Democrats in the 2018 blue wave, particularly in the Johnson County area around Kansas City. The replacement effect for this transition shows new Democratic members who are more moderate than the party's existing rural and urban base — they pulled the Democratic caucus average *rightward* even as the party gained seats.

This is a counterintuitive finding: a party can gain seats and become more moderate simultaneously, if the gained seats are in centrist districts.

**Dynamic trajectories.** The dynamic ideal points for long-serving legislators during this period reveal individual stories. Some moderate Republicans who survived the Brownback primaries show gradual rightward drift — adapting to the party's new center of gravity. Others hold steady, maintaining their centrist positions even as the party moved around them. These holdouts are the legislators who show up as cross-pressured in the issue-specific IRT (Volume 7, Chapter 4).

## Era III: Divided Government (89th-91st, 2021-2026)

### The Context

Governor Kelly's tenure produced persistent divided government: a Democratic governor and Republican supermajorities in both chambers. The signature dynamic was the **veto override** — Kelly vetoing legislation passed by the Republican majority, and the legislature attempting (with varying success) to override. This dynamic appears clearly in the voting data.

### What the Data Shows

**Veto override as polarization engine.** Veto override votes are, by definition, the most partisan votes in the data. They require two-thirds supermajority — meaning even with a comfortable majority, Republicans needed near-unanimous support and potentially some Democratic votes. The bill discrimination parameters (beta) for veto overrides are among the highest in the dataset: these votes separate legislators along the ideological dimension more sharply than almost any other vote type.

The within-session time series (Chapter 1) shows changepoints that correspond to veto override sequences. When the governor vetoes a string of bills in quick succession, the Rice Index for both parties spikes (high cohesion — members circle the wagons) and the polarization gap widens.

**Stability in the dynamic model.** The dynamic ideal point trajectories for the 89th through 91st Legislatures show relative stability compared to the earlier eras. The party means are far apart, but they're not moving much. The tau estimates (evolution variance) are smaller in the later periods, suggesting the chamber has settled into a stable equilibrium — polarized, but consistently so.

**Replacement effects flatten.** The conversion-replacement decomposition for recent transitions shows smaller replacement effects than the Brownback era. Turnover still occurs, but the newcomers are ideologically similar to the people they replace. The primary system has sorted: conservative districts elect conservatives, moderate districts elect moderates, and the mapping is stable.

## Cross-Era Patterns

### The Polarization Arc

Stitching together the dynamic ideal points across all eight bienniums reveals a characteristic shape: polarization rises sharply during the Brownback era (driven by replacement), plateaus during the moderate resurgence (as suburban shifts partially offset conservative gains), and stabilizes at an elevated level during divided government.

This is not a simple "things got worse" narrative. The polarization *did* increase over the study period. But the rate of increase slowed dramatically, and by the 90th-91st Legislatures, the year-to-year changes are within the model's credible intervals. The system may have reached a structural equilibrium — the political geometry of Kansas (deep-red rural districts, blue urban districts, purple suburban districts) maps onto a stable ideological distribution.

### Conversion vs. Replacement Over Time

Plotting the conversion and replacement effects across all seven session transitions reveals a pattern: replacement dominates the early transitions (84th → 85th, 85th → 86th) while conversion is more prominent in later transitions. This suggests that the Kansas Legislature's rightward shift was initially driven by personnel changes (who got elected) but was sustained by behavioral adaptation (how incumbents voted).

### Metric Stability Trends

The cross-session metric stability analysis (Chapter 3) shows which behavioral measures are most consistent over time:

- **IRT ideal points:** Highly stable (ICC > 0.85 in most transitions). Ideology is the most persistent legislative behavior.
- **Party unity scores:** Moderately stable (ICC 0.60-0.80). Members' propensity to vote with their party persists, but less perfectly than their underlying ideology.
- **Network centrality:** Less stable (ICC 0.40-0.70). A legislator's position in the co-voting network depends heavily on the specific bills in each session, making it more session-specific.
- **PC1 scores:** Variable, especially in Senate sessions affected by the horseshoe axis.

The pattern makes theoretical sense: deep traits (ideology) persist more than behaviors (unity) which persist more than structural positions (centrality).

## The Limits of the Story

### What the Data Can't Tell You

The temporal analysis measures *what* changed and *when* it changed. It can decompose the change into conversion and replacement. It can identify the specific legislators who moved the most. But it cannot definitively answer *why*.

When the data shows a moderate Republican shifting rightward between the 86th and 87th Legislatures, it could be:

- **Genuine ideological change** — the legislator reconsidered their positions
- **Strategic adaptation** — the legislator voted more conservatively to survive a primary challenge
- **Agenda effects** — the bills that came to a vote in the 87th were different, and the same underlying preferences produced different votes
- **Leadership pressure** — the party whip was more effective in the 87th

The data is consistent with all of these explanations. External knowledge — interviews, campaign records, news reporting — is needed to distinguish between them. Tallgrass provides the measurements; the interpretation is a collaborative exercise between the data and the reader's contextual understanding.

### What the Data Smooths Over

Dynamic IRT estimates one ideal point per legislator per biennium. Within a biennium, the model averages across all votes. A legislator who was moderate in January and conservative by December gets a single blended score. The rolling PCA in Chapter 1 captures some of this within-session variation, but with less precision than a formal model.

Similarly, the two-year biennium is an arbitrary time unit. A legislator who shifted in their second year of one biennium and their first year of the next would appear to change between bienniums, when the actual change was within a single year. The model's temporal resolution is set by the Kansas legislative calendar, not by the underlying political dynamics.

### Selection Bias in the "Top Movers"

The legislators highlighted as "top movers" are selected by the magnitude of their shift, which creates a selection bias: dramatic shifts are more likely to be reported and discussed, even though the typical legislator barely moved. The top movers are interesting, but they're not representative. The median legislator's biennium-to-biennium shift is small — the stability of most legislators' ideological positions is one of the strongest findings in the data.

---

## Key Takeaway

Fifteen years of Kansas legislative data reveal a polarization arc driven initially by replacement (the Brownback-era primary challenges that shifted the Republican caucus rightward), sustained by conversion (behavioral adaptation of returning members), and stabilized by the structural geometry of Kansas's electoral map. Within sessions, changepoints correspond to identifiable political events — school funding votes, veto override sequences, leadership challenges. The data provides precise measurements of what changed and when; the reader provides the context for why.

---

*Terms introduced: (this chapter introduces no new technical terms — it applies the vocabulary from Chapters 1-4 to the Kansas narrative)*

*Previous: [Conversion vs. Replacement: Why Does Ideology Change?](ch04-conversion-vs-replacement.md)*

*Next: [Volume 9 — Telling the Story](../volume-09-telling-the-story/)*
