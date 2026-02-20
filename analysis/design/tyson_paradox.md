# The Tyson Paradox: When the Biggest Nay-Voter Looks Like the Biggest Conservative

**Date:** 2026-02-20
**Author context:** Joseph Claeys is a Kansas state senator (District 27). His brother J.R. Claeys serves in District 24. Both are Republicans. Joseph's qualitative knowledge of the chamber is extensive and immediate.

## The Surprise

Our IRT model — a Bayesian statistical model that estimates where each legislator falls on an ideological spectrum from liberal to conservative — places Sen. Caryn Tyson as **the single most conservative member of the Kansas Senate**. Not just slightly: her ideal point is +4.17, a full 0.73 standard deviations above the next senator (Thompson at +3.44) and more than 1.4 above the Republican cluster median.

If you follow the Kansas Senate, this should make you do a double-take.

Caryn Tyson votes **Nay more than any other Republican senator.** She cast 74 Nay votes on contested bills — more than double the Republican median of ~27. Her overall Yea rate is 61.9%, versus a Republican average north of 85%. She regularly casts lone or near-lone dissenting votes on bills that pass 37-1 or 38-2. Bond validation? Nay. Waterfowl hunting regulations? Nay. National Guard education benefits? Nay. Newborn screening? Nay. An antisemitism resolution that passed 28-3? Nay.

To anyone who watches the chamber, Tyson isn't "the most conservative senator." She's a renegade — unpredictable, frequently at odds with her own caucus, as likely to vote against a Republican-backed bill as a Democratic one. The model says she's the ideological pole of the chamber. The qualitative read says she's something different.

**So is the model wrong?**

The short answer: no. The model is answering a different question than the one your instincts are asking. The longer answer takes some explaining.

## What PCA Said (and Why It Made More Sense)

Before running IRT, we ran PCA (Principal Component Analysis) — a simpler method that treats every vote equally and finds patterns. PCA told a story much closer to the qualitative intuition:

| Senator | PCA rank among Rs | IRT rank among Rs | Rank shift |
|---|---|---|---|
| Michael Murphy | 1st | 29th (anchor) | -28 |
| Beverly Gossage | 2nd | 6th | -4 |
| Doug Shane | 3rd | 4th | -1 |
| Renee Erickson | 4th | 5th | -1 |
| Ty Masterson | 5th | 9th | -4 |
| Caryn Tyson | **23rd** | **1st** | **+22** |
| Mike Thompson | **17th** | **2nd** | **+15** |

PCA ranked Tyson 23rd out of 32 Republican senators. Solidly Republican, but not extreme. Her PC1 score (+4.86) is well below the leaders (Murphy at +6.60, Gossage at +6.53). But PCA also found something unusual about her: she had the most extreme PC2 score of any senator, at -24.8 — more than double the next most extreme. PC2 captures a pattern we labeled "contrarianism on routine legislation."

IRT flipped the script entirely. Tyson jumped 22 ranks, from middle-of-the-pack to the top.

The two biggest rank jumps in the entire Senate — Tyson (+22) and Thompson (+15) — are also the two senators with the most extreme negative PC2 scores (-24.8 and -8.0 respectively). That's not a coincidence. It's the mechanism.

## The Mechanism: IRT Doesn't Count Votes — It Weighs Them

The fundamental difference between PCA and IRT is how they treat each vote.

**PCA** counts every vote equally. A lone dissent on a waterfowl regulation counts the same as a party-line fight on the budget. Tyson's 74 Nay votes all drag her PC1 score toward center, regardless of what those bills were about.

**IRT** weights each vote by how well it separates liberals from conservatives. Each bill gets a "discrimination" parameter (beta) that measures how ideological it is. A classic party-line fight where all Republicans vote Yea and all Democrats vote Nay gets a high beta — the model learns a lot about ideology from that vote. A routine bill that passes 38-2 gets a near-zero beta — it tells the model almost nothing about ideology.

Here's what that weighting does to Tyson:

| Bill type | Tyson's votes | Her Nay rate | IRT weight |
|---|---|---|---|
| High discrimination (\|beta\| > 1.5) | 81 votes | **22%** (18 Nays) | **78% of total signal** |
| Low discrimination (\|beta\| <= 1.5) | 113 votes | **50%** (56 Nays) | **22% of total signal** |

Her contrarian behavior is concentrated almost entirely on low-discrimination bills — the ones IRT barely listens to. On high-discrimination bills — the partisan fights that define ideology — she is perfectly conservative:

- **63 R-Yea high-discrimination bills (beta > 1.5): Tyson votes Yea on 63/63 (100%)**
- **18 D-Yea high-discrimination bills (beta < -1.5): Tyson votes Nay on 18/18 (100%)**

No other senator has a perfect record on high-discrimination votes. Murphy, who PCA ranked as the most conservative, went 62/63 on R-Yea and 17/18 on D-Yea. Gossage: 63/63 and 17/18. Peck: 63/63 and 17/18.

When the chamber divides along ideological lines, Tyson is always — literally always — on the conservative side. IRT sees this and concludes she's the most conservative senator. It's not wrong. It's just measuring something narrower than "reliable Republican."

## The Twist: Her Dissent Actually *Reinforces* Her Conservative Score

This is the part that's genuinely counterintuitive.

Tyson voted Nay on 41 bills where more than 80% of Republicans voted Yea. You'd think these dissenting votes would pull her IRT score toward center. But they don't — most of them push her *further right*.

Here's why: 31 of those 41 dissent bills have **negative beta**. In IRT's vocabulary, negative beta means "the liberal direction on this bill is Yea." These are bipartisan bills where Democrats all vote Yea, most Republicans also vote Yea, and a handful of dissenters (often including Tyson) vote Nay. The model classifies the Yea side as "liberal" and the Nay side as "conservative" — because the Yea coalition looks more like the average Democrat's behavior.

So when Tyson votes Nay on the antisemitism resolution (beta = -1.53), the nursing workforce bill (beta = -1.42), or the waterfowl regulation (beta = -1.13), the model sees each of those as a conservative vote. She's voting opposite to the direction Democrats vote.

Only 10 of her 41 dissent bills have positive beta (genuinely R-Yea legislation she voted against). Those are the only votes pulling her left — and they're overwhelmed by the 31 pulling right.

| Dissent votes | Count | Effect on IRT score |
|---|---|---|
| Nay on negative-beta bills (bipartisan, D-Yea direction) | 31 | Pushes **right** (conservative) |
| Nay on positive-beta bills (R-Yea direction) | 10 | Pushes **left** (moderate) |

Her contrarianism on bipartisan legislation, far from moderating her score, is one of the things pushing it to the extreme.

## The Head-to-Head: Tyson vs. Murphy

Murphy is our conservative anchor — his ideal point is fixed at +1.0 by design. We chose him because PCA ranked him as the most conservative senator. But what happens if we compare him to Tyson vote-by-vote?

Out of 194 contested Senate votes, Tyson and Murphy both cast Yea/Nay on all 194. They agreed 149 times (77%) and disagreed 45 times (23%).

On those 45 disagreements:
- **Tyson took the more conservative position 31 times (69%)**
- Murphy took the more conservative position 14 times (31%)

But the pattern within those disagreements is revealing:
- Murphy's 14 "more conservative" votes are almost all on **positive-beta bills** — actual Republican legislation that Tyson voted against. HB 2291 (regulatory relief), SB 413 (tort reform), HB 2228 (judiciary). These are the contrarian votes that make Joseph say "she's not reliable."
- Tyson's 31 "more conservative" votes are almost all on **negative-beta bills** — bipartisan legislation that Murphy supported but Tyson opposed. SB 44 (antisemitism), SB 334 (nursing), SB 2 (bond validation). These are the lone-dissent votes that make Joseph say "she votes against random stuff."

The model sees both types. It just weights them differently.

## So Is the Model Wrong?

No. But it's answering a specific question, and you might have a different question in mind.

**What IRT measures:** "If I drew a single line from liberal to conservative and placed every senator on it, where would each one go to best explain their voting pattern on ideologically informative bills?"

**What your instincts measure:** "How reliably does this senator vote with the conservative governing coalition?"

These are genuinely different things. A senator who votes with the Republican caucus on 90% of bills, including routine ones, but occasionally crosses party lines on the big fights would score **lower** on IRT (because IRT only cares about the big fights) but **higher** on your instinct metric (because she's a reliable vote).

Tyson is the reverse. She's unreliable on routine legislation — she'll vote Nay on things everyone supports. But on the bills that actually divide the chamber along ideological lines, she is the most consistently conservative voter in the Senate. There is no recorded instance of her voting with Democrats on a high-discrimination bill.

To use a baseball analogy: IRT measures batting average in clutch situations. Your instinct measures overall plate discipline. Tyson has terrible plate discipline (she swings at — or rather, against — everything) but a perfect batting average when it matters.

## What This Means for the Analysis

**The IRT result is not a bug.** It's a mathematically correct answer to a well-defined question. But it highlights a real limitation of 1D modeling:

1. **Tyson's behavior is genuinely two-dimensional.** She has a strong ideological position (very conservative on partisan fights) AND a strong contrarian tendency (votes Nay on routine legislation regardless of ideology). A 1D model can only capture one of these. It captures the one that's most informative about the liberal-conservative spectrum, which is the partisan votes.

2. **PC2 captures what IRT can't.** Her extreme PC2 score (-24.8, more than 3x the next senator) is the statistical fingerprint of her contrarianism. In a 2D IRT model (a future extension), she would likely appear as: very conservative on Dimension 1, extreme outlier on Dimension 2.

3. **The anchor is part of the story.** Murphy, whom PCA ranked as the most conservative, is the conservative anchor (fixed at xi = +1.0). He can't "compete" with Tyson in the rankings because his position is fixed. If Murphy's ideal point were freely estimated, he would likely land around +2.0 to +2.5 — still well below Tyson's +4.17, because he genuinely has a less extreme voting pattern on high-discrimination bills. But the gap would be smaller.

4. **Neither PCA nor IRT is capturing what a floor whip cares about.** If you're counting votes for a bill, Tyson's IRT score tells you nothing useful. Her contrarianism means she's as likely to dissent on your conservative bill as on a Democratic one. A party loyalty index or a simple intra-party agreement rate would better capture "reliability." IRT captures ideology, not reliability.

## The Broader Pattern

Tyson isn't the only senator affected by this. The three biggest rank shifts from PCA to IRT are:

| Senator | PCA rank | IRT rank | Shift | PC2 |
|---|---|---|---|---|
| Caryn Tyson | 23 | 1 | +22 | -24.8 |
| Mike Thompson | 17 | 2 | +15 | -8.0 |
| Virgil Peck | 6 | 3 | +3 | -3.2 |

All three have the most negative PC2 scores in the Republican caucus. The more contrarian a senator is on routine bills (large negative PC2), the more IRT inflates their conservative ranking (because their routine dissent on bipartisan/D-Yea bills registers as conservative, and their routine dissent on R-Yea bills is downweighted).

This is a feature of the model, not a bug — but it's a feature that systematically mischaracterizes a specific behavioral type. The 1D IRT model treats "votes Nay on everything" and "votes Nay only on liberal bills" as similar, because on the high-discrimination bills both patterns look the same.

## What We Could Do About It (Future Work)

1. **2D IRT.** A two-dimensional model would estimate Tyson's position on both the ideology axis and the contrarianism axis. This is the theoretically correct solution but significantly more complex to identify and interpret.

2. **Party loyalty index.** A simple metric — "fraction of votes agreeing with the party median" — would immediately distinguish Tyson (low loyalty, high ideology) from Murphy (high loyalty, high ideology). This could supplement IRT rankings.

3. **Weighted IRT with contrarianism adjustment.** Downweight votes where a legislator is one of only 1-3 dissenters against a supermajority. This would reduce the influence of lone-dissent votes without fully discarding them.

4. **Report both metrics.** The most honest approach may be to report IRT ideal points alongside a party agreement rate and flag legislators (like Tyson) where the two diverge significantly. The divergence itself is informative.

For now, we document this as a known limitation of the 1D model and flag Tyson's ranking with appropriate context in all downstream analysis.

---

## Technical Reference

**Data:** Kansas Senate, 2025-26 session, 42 legislators x 194 contested votes
**Model:** 2PL Bayesian IRT with Normal(0, 1) discrimination prior, hard anchors at xi = +/- 1

### Key numbers

- Tyson contested votes: 194 (100% participation)
- Tyson Yea rate on contested: 61.9% (R average: ~86%)
- Tyson Nay count: 74 (R median: ~27)
- On |beta| > 1.5 bills: 100% conservative (81/81)
- On |beta| <= 1.5 bills: 50% Yea, 50% Nay
- Fisher information from high-disc votes: 78% of total
- Fisher information from her dissent votes (Nay where >80% Rs = Yea): 9% of total
- PCA PC1: +4.86 (rank 23 of 32 R senators)
- PCA PC2: -24.8 (most extreme in entire chamber by 3x)
- IRT xi: +4.17 (rank 1 of 32 R senators)

### Related files

- `analysis/design/irt.md` — IRT design choices (Assumption 1 discusses 1D limitation)
- `analysis/design/pca.md` — PCA design, PC2 interpretation
- `docs/analytic-flags.md` — Tyson and Thompson flagged as contrarian outliers
- `docs/lessons-learned.md` — Lesson 6 (beta prior) is unrelated but contextually adjacent
