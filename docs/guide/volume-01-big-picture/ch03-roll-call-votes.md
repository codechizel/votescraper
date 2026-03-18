# Chapter 3: What Roll Call Votes Tell Us (and What They Don't)

> *Roll call votes are the gold standard of legislative data. But like any measurement tool, they have blind spots. Understanding both the power and the limits of this data is essential to interpreting everything that follows.*

---

## The Record

Every time the Kansas Legislature takes a roll call vote, the moment is recorded with precision. For each of the 125 House members or 40 Senators, the official record shows exactly one of five outcomes:

| Category | Meaning | Frequency |
|----------|---------|-----------|
| **Yea** | The legislator voted in favor | ~82% of all records |
| **Nay** | The legislator voted against | ~12% |
| **Absent and Not Voting** | The legislator was not present | ~5% |
| **Not Voting** | The legislator was present but did not vote | ~0.5% |
| **Present and Passing** | The legislator was present and formally abstained | ~0.02% |

These five categories collapse into a simple structure. For statistical purposes, we care mainly about the first two: Yea and Nay. These are the "substantive" votes — the moments when a legislator went on the record for or against a specific piece of legislation. The other three categories (absent, not voting, present and passing) represent missing data — the legislator didn't register a position, for whatever reason.

This five-category record is the raw material of the entire Tallgrass pipeline. Everything else — ideal points, party unity scores, coalition maps, predictive models — is derived from this foundation.

## What Votes Reveal

Roll call votes are valuable precisely because they are *actions*, not *words*. When a legislator votes Yea on a tax cut, they have spent real political capital. When they vote Nay on their own party's priority bill, they face real consequences — loss of committee assignments, primary challenges, leadership disfavor. The costs of voting are high enough that roll calls are generally regarded as honest reflections of a legislator's priorities, or at least of the pressures they face.

### Revealed Preferences

Economists have a concept called *revealed preference*: instead of asking people what they want, watch what they actually choose. If someone says they prefer healthy food but eats pizza every night, their revealed preference is pizza.

Roll call votes work the same way. A legislator may give a floor speech praising fiscal responsibility, but if they vote Yea on every spending bill, their revealed preference is spending. Tallgrass works entirely with revealed preferences. It never consults campaign materials, floor speeches, press releases, or interviews. Just votes.

This is both a strength and a limitation. The strength is objectivity: the vote record is the same regardless of who's reading it, and it can be processed computationally without subjective judgment calls. The limitation is depth: we know *what* the legislator did, but not *why*.

### The Information Gradient

Not all votes carry equal information. Consider three votes:

1. **A vote that passes 120-5.** Almost everyone voted Yea. This tells you very little about any individual legislator — they were going with the overwhelming consensus. The five Nay voters are mildly interesting, but even they might have been casting protest votes on an outcome they knew was guaranteed.

2. **A vote that passes 85-40 along party lines.** Nearly all Republicans voted Yea, nearly all Democrats voted Nay. This tells you something — mainly, who's a Republican and who's a Democrat — but it's not very discriminating within each party.

3. **A vote that passes 70-55, with 15 Republicans joining all the Democrats.** Now we're learning something. Those 15 crossover Republicans are revealing that, on this particular issue, they're closer to the Democratic position than to their own party's. These are the votes that give the statistical model the most traction.

Tallgrass encodes this intuition formally. Its Item Response Theory (IRT) model automatically weights each vote by how much it "discriminates" between legislators of different ideologies. Unanimous votes get near-zero weight. Contested, cross-party votes get high weight. This is not a manual choice — it falls out of the mathematics, as we'll see in Volume 4.

### Patterns Across Hundreds of Votes

A single vote reveals very little about any individual legislator. Maybe they had a constituent concern. Maybe they were trading favors. Maybe they misunderstood the bill. Any one vote could be noise.

But across hundreds of votes, patterns emerge that cannot be explained by idiosyncrasy. If a legislator consistently votes with the conservative wing on tax bills, criminal justice bills, education bills, and healthcare bills, the simplest explanation is that they hold conservative views. The IRT model formalizes this: it estimates each legislator's position on a latent (unobserved) ideological dimension that best explains their overall pattern of Yea and Nay votes.

Think of it this way. If you watched a single hand of poker, you couldn't tell a good player from a bad one. But if you watched them play a thousand hands, their skill level would become apparent — not from any single decision, but from the accumulated pattern. Roll call analysis works the same way. A single vote is a single hand. Five hundred votes across a two-year session is a full tournament.

## What Votes Don't Reveal

### The Invisible Legislature

The roll call record captures only the final, public stage of the legislative process. Everything that happens before a vote — the drafting, the committee negotiations, the backroom deals, the leadership arm-twisting — is invisible. This creates a selection effect: the votes we observe are not a random sample of possible legislative actions. They are the survivors of a filtering process that we cannot directly observe.

Consider: a moderate Republican might privately lobby leadership to kill a far-right bill in committee. If they succeed, the bill never reaches the floor and no roll call is recorded. This legislator's moderation is real but invisible in the data. Conversely, if leadership does bring the bill to the floor, the same moderate might vote Yea under pressure — making them look more conservative than they actually are.

This is not a flaw in the data; it's a fundamental feature of the legislative institution. Tallgrass works with what is publicly recorded. The private legislature remains private.

### Strategic Absences

When a legislator is recorded as "Absent and Not Voting," there are at least three possible explanations:

1. **Genuine absence** — They were physically elsewhere (illness, travel, family emergency).
2. **Strategic absence** — They deliberately avoided the vote because taking a position would be politically costly. Voting Yea would anger their base; voting Nay would anger leadership; not voting lets them avoid the dilemma.
3. **Scheduling conflict** — They were in a committee hearing, a meeting, or caught in traffic.

Tallgrass treats all absences the same way: as missing data. The IRT model simply excludes absent observations from the likelihood calculation for that vote. This is statistically sound under the assumption that absences are "missing at random" — that the decision to be absent is unrelated to what the legislator would have voted, conditional on their other votes.

In practice, this assumption is sometimes violated. Strategic absences are most common on the most politically charged votes — exactly the votes that carry the most statistical weight. But there is no reliable way to distinguish strategic absences from genuine ones in the data, so the pipeline treats them all equally. This is a known limitation, noted in the analysis reports.

### The Base Rate Problem

Here is the single most important statistical fact about Kansas roll call data:

> **Roughly 82% of all recorded votes are Yea.**

This is not peculiar to Kansas. Most state legislatures have Yea rates between 75% and 90%. The reason is structural: bills that reach a floor vote have already passed through multiple filters (committee hearings, leadership approval, informal whip counts). The floor vote is the last step, not the first. By the time a bill gets there, it usually has the votes to pass.

But this high base rate creates a trap for naive analysis. If someone tells you that two legislators "agree 90% of the time," that sounds impressive — until you realize that two legislators who voted randomly (flipping a coin weighted 82% toward Yea) would agree about 70% of the time just by chance. The 90% agreement is only 20 percentage points above what you'd expect from pure chance.

Let's make this concrete with an analogy. Imagine two weather forecasters in Phoenix. Forecaster A says "sunny" every day. Forecaster B also says "sunny" every day. They agree 100% of the time — but you wouldn't conclude they're brilliant forecasters. Phoenix is sunny about 300 days a year. Any idiot who just says "sunny" will be right 82% of the time. To learn who's actually good at forecasting, you need to look at the days when the weather is *uncertain* — the monsoon season, the rare winter storms.

Roll call analysis faces exactly the same challenge. The "sunny days" are the near-unanimous votes. The "monsoon season" is the contested votes — the ones where the outcome is in doubt and legislators' choices actually reveal something. The pipeline's first step is to filter out the unanimous votes and focus on the contested ones.

### Voice Votes and Committee Kills

Not all legislative action produces roll call data. Two major categories are invisible:

**Voice votes** — Instead of recording each legislator's position, the presiding officer calls for "ayes" and "nays" by voice, and makes a judgment call about which side was louder. Voice votes are used for non-controversial measures and procedural motions. They are fast but produce no individual-level data.

**Committee kills** — A committee chair can simply decline to schedule a hearing on a bill, effectively killing it without any recorded vote. This is the most common fate for introduced legislation — the majority of bills introduced in any session never receive a committee vote, let alone a floor vote. The bills that *do* make it to the floor are a curated subset, not a random sample.

These gaps mean that Tallgrass sees only the tip of the legislative iceberg. The votes we analyze represent the public, recorded portion of a much larger and more complex process.

## The Vote Matrix

All of this raw data — 95,000 individual vote records in a typical session — gets reorganized into a structure called the **vote matrix**. This is the fundamental data object that feeds the entire analysis pipeline.

The vote matrix is a table where:
- Each **row** is a legislator
- Each **column** is a roll call vote
- Each **cell** contains 1 (Yea), 0 (Nay), or nothing (absent/abstain)

Here's what a tiny slice might look like (using made-up legislators and bills):

|  | SB 12 (Tax cut) | HB 45 (School funding) | SB 78 (Gun rights) | HB 102 (Medicaid) |
|--|:---:|:---:|:---:|:---:|
| **Sen. Anderson (R)** | 1 | 0 | 1 | 0 |
| **Sen. Baker (D)** | 0 | 1 | 0 | 1 |
| **Sen. Clark (R)** | 1 | 1 | 1 | 0 |
| **Sen. Diaz (D)** | 0 | 1 | 0 | — |
| **Sen. Evans (R)** | 1 | 0 | 0 | 0 |

Even in this tiny example, you can start to see structure. Anderson and Baker are mirror images — they disagree on everything. Clark is a Republican who crossed over on the school funding bill. Diaz missed the Medicaid vote. Evans is a Republican who broke ranks on the gun rights bill. The analysis pipeline's job is to extract this kind of structure from a matrix that's 125 rows by 600 columns, where the patterns are too complex for the human eye to detect.

## From Votes to Understanding

The journey from raw vote records to statistical understanding follows a logical progression:

1. **Collect** — Scrape every recorded vote from the legislature's website (Volume 2)
2. **Clean** — Filter out uninformative votes, handle missing data, build the vote matrix (Volume 3)
3. **Compress** — Use dimensionality reduction to find the main axes of variation (Volume 3)
4. **Estimate** — Use Bayesian inference to assign each legislator an ideal point (Volume 4)
5. **Validate** — Check the estimates against independent data (Volume 5)
6. **Enrich** — Add clustering, network analysis, classical indices, and prediction (Volumes 6-7)
7. **Track** — Follow changes across sessions and years (Volume 8)
8. **Report** — Turn the numbers into readable narratives (Volume 9)

Each step builds on the one before it. You can't estimate ideal points without first building the vote matrix. You can't validate without first estimating. The pipeline is a chain, and the vote matrix is its first link.

## A Word About Imperfection

No dataset is perfect, and this one is no exception.

Some vote records are ambiguous — a motion that reads "motion prevailed" without specifying what the motion was. Some historical sessions (2011-2014) stored their votes in ODT (OpenDocument) format rather than HTML, requiring a separate parser. Some legislator names are inconsistent across sessions (marriages, name changes, suffix conventions). Special sessions use different URL patterns than regular sessions.

The scraper handles all of these cases, and the test suite verifies that it handles them correctly (~2,960 tests at last count). But the data is only as good as the source, and the source is a government website maintained over 27 years by people who had no idea anyone would someday run Bayesian statistics on it.

Where the data is imperfect, the pipeline is transparent about it. Filtering decisions are logged. Missing data is tracked. Quality metrics are reported. The goal is not perfection — it's honest, reproducible analysis with known limitations.

---

## Key Takeaway

Roll call votes are the best available window into legislative ideology: they are public, precise, and costly enough to be meaningful. But they capture only the visible tip of the legislative process, they're dominated by an 82% Yea base rate that inflates naive agreement measures, and they can't distinguish strategic behavior from sincere belief. The statistical pipeline is designed to extract real signal from this imperfect data — and to be honest about where the signal runs out.

---

*Terms introduced: revealed preference, information gradient, discrimination (preview), selection effect, strategic absence, base rate, vote matrix, Cohen's Kappa (preview)*

*Next: [The Pipeline: From Raw Votes to Insight](ch04-the-pipeline.md)*
