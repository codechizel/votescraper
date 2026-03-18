# Chapter 2: Who Agrees with Whom? (Cohen's Kappa)

> *If two legislators agree on 85% of votes, is that a lot? In a legislature where 82% of all votes are Yea, the answer is: not really. Cohen's Kappa tells you how much agreement is real and how much is just the base rate talking.*

---

## The Problem with Raw Agreement

The most natural way to measure whether two legislators vote alike is to count: out of all the votes they both participated in, what fraction did they vote the same way?

This is called **raw agreement**, and it's intuitive. If Representatives Adams and Baker both voted on 500 roll calls, and they cast the same vote (both Yea or both Nay) on 425 of them, their raw agreement is 425/500 = 85%.

But as Chapter 1 explained, the Kansas Legislature has an 82% Yea base rate on contested votes. Two legislators who vote completely at random — flipping a coin weighted 82% toward Yea — would agree about 70% of the time by pure chance.

So that 85% figure isn't as impressive as it sounds. It's only 15 percentage points above the chance baseline. The real question is: *how much do Adams and Baker agree beyond what we'd expect from two random legislators with the same voting tendencies?*

This question was answered in 1960 by a psychologist named Jacob Cohen.

## Jacob Cohen and the Kappa Statistic

**Jacob Cohen** (1923–1998) was a quantitative psychologist at New York University who spent his career studying how to measure agreement. In 1960, he published a paper titled "A Coefficient of Agreement for Nominal Scales" in the journal *Educational and Psychological Measurement*. The paper introduced a simple statistic — which he modestly labeled κ (the Greek letter kappa) — that corrects agreement for chance.

Cohen's original problem wasn't about legislatures. He was studying how well two doctors agreed when diagnosing patients. If two doctors both diagnose 90% of patients as "healthy" (because most patients are healthy), their raw agreement will be high even if they're guessing randomly. Kappa strips out the portion of agreement that's attributable to the base rate and reports only the agreement that's *above and beyond chance*.

The same logic applies perfectly to legislative voting.

## The Formula

Cohen's Kappa is defined as:

**Plain English:** "How much more do these two legislators agree than we'd expect by chance?"

**Equation:**

```
κ = (p_observed − p_expected) / (1 − p_expected)
```

Where:
- **p_observed** = the fraction of votes where both legislators voted the same way (raw agreement)
- **p_expected** = the fraction of votes where we'd *expect* them to agree by chance, based on each legislator's individual voting tendencies

**Step-by-step walkthrough:**

Let's say Representative Adams votes Yea 85% of the time, and Representative Baker votes Yea 78% of the time.

1. **What's the chance they both vote Yea on the same bill?**
   - P(both Yea) = 0.85 × 0.78 = 0.663

2. **What's the chance they both vote Nay?**
   - P(both Nay) = (1 − 0.85) × (1 − 0.78) = 0.15 × 0.22 = 0.033

3. **Expected agreement by chance:**
   - p_expected = 0.663 + 0.033 = **0.696** (about 70%)

4. **If their observed agreement is 85%:**
   - κ = (0.85 − 0.696) / (1 − 0.696)
   - κ = 0.154 / 0.304
   - κ = **0.51**

5. **Interpretation:** Their agreement beyond chance is "moderate" — they agree more than random, but there's substantial room for disagreement.

Now compare: if two legislators from the same party agree 95% of the time:
   - κ = (0.95 − 0.696) / (1 − 0.696) = 0.254 / 0.304 = **0.84** — "almost perfect" agreement

And if two legislators from opposite parties agree only 60% of the time:
   - κ = (0.60 − 0.696) / (1 − 0.696) = −0.096 / 0.304 = **−0.32** — they agree *less* than chance, meaning they're actively voting against each other

## The Interpretation Scale

In 1977, J. Richard Landis and Gary Koch proposed a scale for interpreting Kappa values that has become the standard reference:

| Kappa Range | Interpretation |
|-------------|----------------|
| < 0.00 | Worse than chance — active disagreement |
| 0.00 – 0.20 | Slight agreement |
| 0.21 – 0.40 | Fair agreement |
| 0.41 – 0.60 | Moderate agreement |
| 0.61 – 0.80 | Substantial agreement |
| 0.81 – 1.00 | Almost perfect agreement |

Landis and Koch acknowledged these thresholds are somewhat arbitrary — they didn't derive them from any mathematical principle. But they've been cited thousands of times and remain the most widely used benchmarks.

**A negative Kappa is meaningful.** It means two legislators agree *less* often than chance would predict. In a legislature, this usually means they're from opposite parties and are actively voting against each other on contested bills. A Kappa of −0.5 between a Democrat and a Republican means they're on opposite sides more often than not, even accounting for the base rate.

## Why Kappa Matters for Legislative Data

The 82% Yea base rate makes raw agreement almost useless for comparing legislators. Consider three pairs:

| Pair | Raw Agreement | Kappa | Interpretation |
|------|--------------|-------|----------------|
| Two same-party allies | 95% | 0.84 | Almost perfect — real ideological alignment |
| Two moderates from opposite parties | 85% | 0.51 | Moderate — they cross the aisle sometimes |
| Random baseline | 70% | 0.00 | No signal — just base rate noise |

Raw agreement says the first two pairs differ by only 10 percentage points (95% vs. 85%). Kappa says they differ by 33 points (0.84 vs. 0.51) — the true gap is much larger once you account for chance.

**Think of it this way.** Imagine you're comparing two weather forecasters in Phoenix, where it's sunny 82% of the time. Forecaster A is right 95% of the time. Forecaster B is right 85% of the time. Both look good — but Forecaster A is actually *far* better, because being right 85% in Phoenix is barely above guessing "sunny" every day. Kappa is like a skill score for forecasters: it tells you how much better they are than the naive strategy of always predicting the most common outcome.

## The Agreement Matrix

Tallgrass computes Kappa between every pair of legislators in each chamber. For the Kansas House (about 120 legislators after filtering), that's roughly 120 × 119 / 2 = **7,140 pairwise comparisons**. For the Senate (~40 legislators), it's about 780 pairs.

The result is an **agreement matrix** — a square table where the value in row *i*, column *j* is the Kappa between legislator *i* and legislator *j*. The diagonal is always 1.0 (everyone agrees perfectly with themselves). The matrix is symmetric (the Kappa between Adams and Baker is the same as between Baker and Adams).

A small-sample safeguard: any pair of legislators who shared fewer than **10 contested votes** gets no Kappa score (marked as missing). This prevents unreliable statistics from tiny samples — if two legislators only overlapped on 5 votes, any agreement measure would be dominated by noise.

### What the Matrix Reveals

When you plot the agreement matrix as a heatmap — with legislators sorted by party — the structure of the legislature jumps out immediately:

- **Same-party blocks** glow warm (high Kappa, typically 0.4–0.8): Republicans agree with Republicans, Democrats agree with Democrats.
- **Cross-party blocks** glow cool (negative Kappa, typically −0.2 to −0.5): Democrats and Republicans actively vote against each other.
- **The diagonal boundary** between parties is sharp and clear — the single most visible feature of the plot.

Even without any statistical model, the raw agreement matrix makes the two-party structure obvious. This is the first clue that a one-dimensional ideology score (liberal ↔ conservative) might work well for this data.

**Codebase:** `analysis/01_eda/eda.py` (`compute_agreement_matrices()`, constant `MIN_SHARED_VOTES = 10`)

## Beyond Agreement: Party Unity and Strategic Absence

While Kappa is the headline measure, the EDA phase also computes two additional diagnostics that help characterize the legislature:

### Party Unity Scores

For each legislator, the pipeline computes how often they vote with the majority of their party on **party-line votes** — roll calls where a majority of Democrats voted one way and a majority of Republicans voted the other. This is called the **party unity score**, and it follows a method developed by political scientist John Carey and the Legislative Voting and Decisions (LVD) project.

A party unity score of 95% means the legislator voted with their party on 95% of contested, party-line votes. A score of 70% means they broke ranks 30% of the time — a maverick.

### Strategic Absence

Some legislators don't just vote against their party — they avoid the vote entirely. The strategic absence diagnostic (based on Rosas & Shomer, 2008) compares each legislator's absence rate on party-line votes to their overall absence rate. If a legislator misses party-line votes at **twice the rate** they miss other votes, that's flagged as potentially strategic.

Strategic absence is particularly interesting because it's invisible in the vote matrix. A legislator who skips a controversial vote doesn't show up as a "Nay" — they simply have a blank cell. Without this diagnostic, strategic avoidance would go undetected.

**Codebase:** `analysis/01_eda/eda.py` (`compute_party_unity_scores()`, `compute_strategic_absence()`)

## A Practical Example

Here's how agreement looks in the 91st Kansas Senate (2025–2026), which has 30 Republicans and 10 Democrats:

**Within the Republican caucus:** Kappa values typically range from 0.30 to 0.75. The wide range reflects the fact that Kansas Republicans are not monolithic — there are moderates, mainstream conservatives, and more right-leaning members. Even within one party, there's a spectrum.

**Within the Democratic caucus:** Kappa values are typically higher, ranging from 0.50 to 0.85. With only 10 members, the Democratic caucus is smaller and more ideologically cohesive — there's less room for internal disagreement.

**Between parties:** Kappa values are typically negative, ranging from −0.10 to −0.50. The most partisan pairings (a very conservative Republican and a very liberal Democrat) have Kappas around −0.5, meaning they vote against each other far more often than chance would predict.

This three-layer structure — tight within-party agreement, loose within-party variation, and strong between-party disagreement — is the signature of a polarized two-party legislature. Every Kansas session from 2011 to 2026 shows this pattern.

---

## Key Takeaway

Cohen's Kappa corrects pairwise agreement for the base rate, revealing how much legislators truly agree beyond chance. In a legislature with an 82% Yea rate, raw agreement is inflated by about 70 percentage points. Kappa strips that away, exposing the real structure: high within-party agreement, wide within-party variation, and strong between-party disagreement. The agreement matrix is the first clear picture of the legislature's ideological landscape.

---

*Terms introduced: raw agreement, Cohen's Kappa, p_observed, p_expected, Landis-Koch scale, agreement matrix, party unity score, strategic absence, party-line vote*

*Next: [Compressing the Data: PCA Explained](ch03-pca-explained.md)*
