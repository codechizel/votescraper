# Chapter 1: The Vote Matrix: Ones, Zeros, and Missing Data

> *Imagine a spreadsheet with 125 rows (one per legislator) and 600 columns (one per vote). Each cell says Yea, Nay, or nothing at all. How do you make sense of it?*

---

## From Thousands of Records to One Table

At the end of Volume 2, the scraper has produced five CSV files for each legislative session. The most important of these is the **individual votes** file — a list of every recorded vote by every legislator on every roll call. For the 91st Legislature (2025–2026), this file contains over 95,000 rows. Each row says something like: *Senator Smith voted Yea on roll call #347*.

But a list of 95,000 individual records isn't something a human — or a statistical model — can work with directly. To analyze voting patterns, we need to reorganize the data into a **vote matrix**: a single table where every row is a legislator, every column is a roll call, and each cell records how that legislator voted on that roll call.

Think of it like a gradebook. The teacher has a stack of 600 graded assignments from 125 students. Right now, the grades are in a big pile — one slip per student per assignment. The first step is to organize them into a grid: students down the left side, assignments across the top, grades in the cells. Only then can the teacher start asking questions like "who's doing well?" or "which assignments were the hardest?"

## Building the Matrix

### The Encoding

Each cell in the vote matrix needs a number. The Kansas Legislature records five possible vote categories:

| Category | Encoding | Meaning |
|----------|----------|---------|
| Yea | 1 | Voted in favor |
| Nay | 0 | Voted against |
| Absent and Not Voting | missing | Not present for the vote |
| Not Voting | missing | Present but did not vote |
| Present and Passing | missing | Present, explicitly abstained |

Yea becomes 1, Nay becomes 0, and everything else becomes **missing** — a blank cell in the spreadsheet. This binary encoding (1 or 0) is the standard approach in political science, used by Poole and Rosenthal's NOMINATE and every major roll call analysis system.

Why not encode absences as their own number — say, 0.5? Because absence can mean many things. A legislator might be absent because they're sick, attending a committee meeting, strategically avoiding a controversial vote, or simply having lunch. Lumping all of these together and calling them "half a vote" would add noise, not signal. It's better to admit we don't know and leave the cell blank.

The rare "Present and Passing" category (about 22 instances out of 95,000 in the 91st Legislature — less than 0.03%) also becomes missing. These are genuine abstentions, but they're too rare to analyze as a separate category.

### Two Matrices, Not One

The Kansas Legislature has two chambers — the House (125 members) and the Senate (40 members) — and they vote separately. A House member never votes on a Senate roll call and vice versa. If we put everyone in one matrix, half the cells would be blank by design.

So the pipeline builds two matrices: one for the House and one for the Senate. Each is analyzed independently through the rest of the pipeline.

**Codebase:** `analysis/01_eda/eda.py` (`build_vote_matrix()`)

## Why Filtering Matters

Here's the problem with using the raw vote matrix: most legislative votes are boring.

In a typical Kansas session, about 82% of all votes are Yea. Most bills pass with overwhelming majorities. A resolution honoring the state's high school wrestling champions passes 123–0. A technical correction to the tax code passes 118–2. A routine appropriations bill passes 105–15.

These near-unanimous votes tell us almost nothing about ideology. If everyone votes the same way, the vote doesn't help us distinguish liberals from conservatives. It's like a test question that every student gets right — it doesn't help you figure out who studied and who didn't.

Tallgrass applies two sequential filters to remove the noise and keep the signal:

### Filter 1: Remove Near-Unanimous Votes

Any roll call where the minority side has fewer than **2.5% of voters** is dropped. On a 125-member House, that means any vote where fewer than about 3 legislators dissented. On a 40-member Senate, it's any vote where only 1 person voted differently from everyone else.

This 2.5% threshold comes from VoteView, the gold standard roll call database maintained by political scientists at UCLA and Georgia. It's a deliberate choice: aggressive enough to remove ceremonial and procedural votes, conservative enough to keep genuinely contested ones.

**What gets dropped:** Unanimous votes, near-unanimous resolutions, procedural motions where party discipline isn't a factor. Typically, about 25-40% of roll calls are filtered out.

**What survives:** Votes on substantive legislation where at least a handful of legislators disagreed — exactly the votes that reveal ideological differences.

### Filter 2: Remove Low-Participation Legislators

After removing near-unanimous votes, any legislator who participated in fewer than **20 contested votes** is dropped. This catches two cases:

- **Mid-session replacements:** A legislator who was appointed in April to replace someone who resigned might have only voted on 15 contested bills. That's not enough data to estimate their ideology reliably.
- **Chronic absentees:** A legislator who missed most of the session doesn't provide enough information for analysis.

The threshold of 20 is deliberately modest. In a typical session, the House has about 400–700 contested roll calls after filtering. A legislator with only 20 is barely present — but 20 is enough for basic statistical measurement.

### The Numbers

For the 91st Legislature (2025–2026), here's what filtering looks like. The House and Senate vote separately, so their roll call counts differ — the House, with 125 members and more procedural business, produces roughly twice as many roll calls as the 40-member Senate:

| Metric | House | Senate |
|--------|-------|--------|
| Roll calls before filtering | ~725 | ~380 |
| Roll calls after 2.5% filter | ~500 | ~270 |
| Legislators before filtering | 125 | 40 |
| Legislators after 20-vote filter | ~120 | ~40 |

The filtered vote matrix — about 120 legislators by 500 roll calls in the House, or 40 legislators by 270 roll calls in the Senate — is what every downstream analysis phase works with.

**Codebase:** `analysis/01_eda/eda.py` (`filter_vote_matrix()`, constants `MINORITY_THRESHOLD = 0.025`, `MIN_VOTES = 20`)

## The Base Rate Problem

Even after filtering, there's a subtlety that trips up many analysts: the **base rate**.

In the 91st Kansas Legislature, about 82% of recorded votes on contested bills are Yea. That's not because Kansas legislators are pushovers — it's because the legislature is designed to consider bills that have a reasonable chance of passing. Bills that would fail catastrophically are usually killed in committee before reaching a floor vote.

This 82% Yea rate has a surprising consequence: **two legislators who vote completely randomly would still appear to agree about 70% of the time.**

Here's the math. If Legislator A votes Yea 82% of the time and Legislator B votes Yea 82% of the time (both randomly, with no ideological signal), the probability that both vote Yea on the same bill is 0.82 × 0.82 = 0.67. The probability both vote Nay is 0.18 × 0.18 = 0.03. Total expected agreement: 0.67 + 0.03 = **70%**.

So when you see that two Kansas legislators agree on 85% of votes, that sounds impressive — but it's only 15 percentage points above what pure chance would produce. The next chapter introduces a statistic designed to correct for exactly this problem.

**Think of it this way.** Imagine a weather forecaster in Phoenix, Arizona, where it's sunny 300 days a year. If the forecaster simply predicts "sunny" every day, they'll be right 82% of the time. That doesn't make them a good forecaster — they're just exploiting the base rate. Similarly, two legislators who both tend to vote Yea will "agree" most of the time even if they have nothing in common ideologically. The interesting question isn't how often they agree, but how much they agree *beyond what the base rate would predict*.

## Handling Missing Data

The vote matrix always has blank cells. No legislator votes on every single roll call — people get sick, attend committee meetings, or miss the occasional floor vote. In the Kansas House, a typical legislator participates in about 90–95% of contested roll calls. That means 5–10% of their cells are missing.

Missing data isn't a crisis, but it does require decisions. Different analysis methods handle it differently:

| Method | Missing Data Strategy |
|--------|----------------------|
| **Agreement (Kappa)** | Only count votes where both legislators participated. Pairs with fewer than 10 shared votes get no score. |
| **PCA** | Impute missing values with each legislator's average Yea rate (their personal base rate). |
| **MCA** | Treat absence as a third category alongside Yea and Nay. |
| **UMAP** | Same imputation as PCA (row-mean). |
| **IRT** | Missing values are simply skipped — the model only uses observed votes. |

The PCA imputation strategy deserves explanation. If a legislator votes Yea on 75% of their non-missing votes, their missing cells are filled with 0.75. The logic: our best guess for a missing vote, in the absence of any other information, is that the legislator would have voted according to their overall tendency. This is called **row-mean imputation**, and while it's not perfect, it's the standard approach in roll call analysis.

**Codebase:** `analysis/02_pca/pca.py` (`impute_vote_matrix()`)

## Data Integrity Checks

Before any analysis begins, the pipeline runs nine automated integrity checks on the raw data:

1. **Seat counts:** Does the data have the right number of legislators? (House = 125, Senate = 40)
2. **Mid-session replacements:** Are there more names than seats? (Happens when a legislator resigns and is replaced.)
3. **Referential integrity:** Does every vote reference a real roll call and a real legislator?
4. **Duplicate roll calls:** Are any roll calls recorded twice?
5. **Tally consistency:** Does the number of Yea + Nay + Absent votes match the expected total?
6. **Chamber-size bounds:** Are vote tallies within the physical limits of each chamber?
7. **Chamber-slug consistency:** Do House slugs appear only in House votes?
8. **Vote category validation:** Are all vote values one of the five legal categories?
9. **Party validation:** Do legislators have recognized party affiliations?

These checks catch scraper bugs, data corruption, and edge cases before they can contaminate the analysis. A failing check halts the pipeline with a descriptive error message.

**Codebase:** `analysis/01_eda/eda.py` (`check_data_integrity()`)

## What the EDA Phase Produces

The exploratory data analysis phase (Phase 01 of the pipeline) takes the raw CSV files from the scraper and produces:

- **Filtered vote matrices** for House and Senate (Parquet files)
- **Agreement matrices** — pairwise raw agreement and Cohen's Kappa (next chapter)
- **Party unity scores** — how often each legislator votes with their party
- **Strategic absence analysis** — whether any legislators miss party-line votes more often than other votes
- **Item-total correlations** — which roll calls are most discriminating
- **A filtering manifest** — a record of exactly how many rows and columns were removed and why

This output becomes the input for every subsequent phase. PCA reads the filtered matrices. IRT reads the filtered matrices. The clustering phase reads the agreement matrices. Everything starts here.

```bash
# Run the EDA phase
just eda 2025-26
```

**Codebase:** `analysis/01_eda/eda.py` (the complete EDA phase)

---

## Key Takeaway

The vote matrix is the foundation of the entire analysis pipeline. Building it means encoding votes as 1 (Yea), 0 (Nay), or missing, then filtering out near-unanimous votes (below 2.5% minority) and low-participation legislators (fewer than 20 votes). The 82% Yea base rate means raw agreement percentages are misleading — a problem the next chapter solves with Cohen's Kappa.

---

*Terms introduced: vote matrix, binary encoding, contested vote, near-unanimous vote, base rate, row-mean imputation, filtering manifest, data integrity checks*

*Next: [Who Agrees with Whom? (Cohen's Kappa)](ch02-cohens-kappa.md)*
