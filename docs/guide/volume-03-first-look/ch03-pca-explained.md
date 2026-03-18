# Chapter 3: Compressing the Data: PCA Explained

> *You have 600 votes. Can you summarize a legislator's entire voting record with just one or two numbers?*

---

## The Compression Problem

After filtering, the Kansas House vote matrix has about 120 rows (legislators) and 500 columns (contested roll calls). That's 60,000 cells of data. You can't visualize 500 dimensions. You can't plot 500 numbers per person. And yet the agreement matrix from Chapter 2 showed that most of the action in the data comes from a single dimension: party.

What if we could find a way to collapse those 500 columns down to just one or two — capturing the essential pattern while discarding the noise?

That's exactly what **Principal Component Analysis** does.

## A Brief History

PCA has roots going back more than a century. The British mathematician **Karl Pearson** first described the idea in 1901: given a cloud of data points in space, find the line (or plane) that fits the data most closely. Pearson was thinking geometrically — literally looking for the best-fitting line through a scatter of points.

Three decades later, the American statistician **Harold Hotelling** independently developed the algebraic machinery in 1933, framing the problem in terms of eigenvalues and eigenvectors. Hotelling's formulation is essentially the one used today.

PCA has since become one of the most widely used techniques in all of statistics. It appears in genetics (where it identifies population structure from DNA), in image compression (where it reduces file sizes), in finance (where it identifies market factors), and in political science (where it identifies ideological dimensions from voting records).

## The Analogy: Finding the Best Angle

Imagine you're a photographer trying to take a picture of a group of people standing in a room. You want the photo to show as much difference between the people as possible — you want to be able to tell them apart.

If you photograph them from directly above, they all look like circles on the floor. Not helpful — everyone looks the same. If you photograph them from the side, you can see differences in height. Better. But the best angle might be slightly diagonal, capturing both height differences *and* the fact that some people are standing further forward than others.

**PCA is like finding the perfect camera angle.** It looks at your high-dimensional data (500 votes) and finds the single direction — the single "angle" — along which the data varies the most. That direction becomes **PC1** (the first principal component). Then it finds the next-best direction, perpendicular to the first, and calls it **PC2**. And so on.

The key insight: in legislative voting data, PC1 almost always captures the party divide. It's the single direction along which legislators differ the most — and that direction runs from liberal to conservative.

## How PCA Works (Step by Step)

Let's walk through PCA on a tiny example before applying it to the full vote matrix.

### Step 1: Start with the Data

Imagine three legislators and three votes:

|  | Vote A | Vote B | Vote C |
|--|--------|--------|--------|
| **Rep. Red** | 1 (Yea) | 1 (Yea) | 0 (Nay) |
| **Rep. Purple** | 1 (Yea) | 0 (Nay) | 0 (Nay) |
| **Sen. Blue** | 0 (Nay) | 0 (Nay) | 1 (Yea) |

### Step 2: Center the Data

Subtract the average of each column so the data is centered at zero. The average of Vote A is (1 + 1 + 0)/3 = 0.67.

|  | Vote A | Vote B | Vote C |
|--|--------|--------|--------|
| **Rep. Red** | +0.33 | +0.67 | −0.67 |
| **Rep. Purple** | +0.33 | −0.33 | −0.67 |
| **Sen. Blue** | −0.67 | −0.33 | +0.33 |

Centering doesn't change the relationships between legislators — it just repositions the data so the average is at the origin.

### Step 3: Find the Direction of Maximum Spread

PCA now looks for the single direction — a line through the centered data — along which the three legislators are as spread out as possible. Mathematically, it finds the eigenvector of the covariance matrix with the largest eigenvalue. In plain English: it finds the "spine" of the data cloud.

In this toy example, that direction runs roughly from "votes like Rep. Red" to "votes like Sen. Blue" — the party divide.

### Step 4: Project onto the Line

Each legislator gets a score on PC1 — their position when projected onto that best-fit line. Rep. Red might land at +1.0, Rep. Purple at +0.3, and Sen. Blue at −1.2.

Those three numbers — (+1.0, +0.3, −1.2) — are the **PC1 scores**. Three votes have been compressed into a single number per legislator, and that number captures the dominant pattern: the party divide.

### Step 5: Repeat for PC2

PCA finds the next direction of maximum spread, but with a constraint: it must be **perpendicular** (orthogonal) to PC1. This ensures PC2 captures a genuinely different pattern, not a rehash of PC1.

In legislative data, PC2 often captures **within-party variation** — the difference between, say, moderate Republicans and conservative Republicans. It's a second dimension of disagreement that's independent of the first.

## The Equation Behind the Score

What PCA actually computes for each legislator is a **weighted sum** of their votes:

**Equation:**

```
PC1 score for legislator i = w₁ · vote₁ + w₂ · vote₂ + ... + w₆₀₀ · vote₆₀₀
```

where the weights (w₁, w₂, ..., w₆₀₀) are chosen to maximize the spread of scores across all legislators.

**Plain English:** "Take every vote the legislator cast, multiply each one by how important that vote is for distinguishing people, and add them all up."

A highly partisan vote (like a party-line tax bill) gets a large weight — it's very useful for distinguishing liberals from conservatives. A near-unanimous vote (like a resolution honoring veterans) gets a weight near zero — it doesn't help distinguish anyone. PCA figures out these weights automatically from the data.

## Eigenvalues and the Scree Plot

Each principal component comes with a number: its **eigenvalue**. The eigenvalue measures how much of the data's total variation that component captures. If the first eigenvalue is large and the rest are small, it means one dimension explains most of the data — the legislature is fundamentally one-dimensional (party is everything).

The standard way to visualize this is the **scree plot** — a bar chart of eigenvalues in descending order.

```
Eigenvalue
  │
  │ ████
  │ ████
  │ ████
  │ ████ ██
  │ ████ ██ █ █ █ █ █ █ ...
  └──────────────────────────
    PC1  PC2 PC3 ...
```

A steep drop from PC1 to PC2 means the data is strongly one-dimensional. If PC1 and PC2 are similar in size, the data has two important dimensions.

### Explained Variance

Eigenvalues are often converted to **percentages of explained variance**: what fraction of the total variation does each component capture?

For the 91st Kansas Senate:

| Component | Eigenvalue | Variance Explained | Cumulative |
|-----------|------------|-------------------|------------|
| PC1 | ~3.4 | ~37% | 37% |
| PC2 | ~1.1 | ~12% | 49% |
| PC3 | ~0.8 | ~9% | 58% |
| PC4 | ~0.6 | ~7% | 65% |
| PC5 | ~0.5 | ~6% | 71% |

PC1 alone captures 37% of all variation — the single most important pattern. The eigenvalue ratio (PC1/PC2 = 3.4/1.1 ≈ 3.1) tells us PC1 is about three times as important as PC2.

### How Many Components Matter?

Tallgrass uses **Horn's parallel analysis** (1965) to decide how many components are meaningful versus noise. The method works by generating 100 random matrices of the same size as the real data. Each random matrix has no structure — it's pure noise. PCA is run on each, and the eigenvalues are recorded.

The rule: a real component is "significant" if its eigenvalue exceeds the 95th percentile of the random eigenvalues. If PC1's eigenvalue is 3.4 but random data never produces an eigenvalue above 1.8, then PC1 is definitely real. If PC5's eigenvalue is 0.5 and random data regularly produces eigenvalues of 0.6, then PC5 is just noise.

In Kansas data, Horn's analysis typically identifies 2–4 significant components — the party divide plus a few secondary dimensions.

**Codebase:** `analysis/02_pca/pca.py` (`parallel_analysis()`, `PARALLEL_ANALYSIS_N_ITER = 100`)

## What PC1 and PC2 Mean

### PC1: The Party Divide

In nearly every Kansas legislative session, PC1 separates Democrats from Republicans. Legislators with negative PC1 scores are Democrats; those with positive scores are Republicans. The convention (Republicans positive) matches the standard in political science literature.

For the 91st Senate, the party separation on PC1 (measured by Cohen's d) is **7.49** — an enormous gap. This means the average Republican is 7.49 standard deviations away from the average Democrat on PC1. For context, a d of 0.8 is considered "large" in the social sciences. A d of 7.49 means the parties are almost completely non-overlapping on this dimension.

### PC2: Within-Party Variation

PC2 captures the second-biggest source of disagreement — and in most sessions, that's *within-party* variation. For example:

- In the Republican caucus, PC2 might separate establishment-aligned moderates from more conservative members.
- In the Democratic caucus, PC2 might separate urban progressives from rural moderates.

The key point: PC2's party separation (Cohen's d) is typically very low — around 0.1 to 0.3. That confirms it's capturing a dimension that doesn't align with party, which makes it useful for understanding intra-party dynamics.

### The Sign Convention

PCA doesn't inherently know which direction is "liberal" and which is "conservative." The math could just as easily put Democrats on the positive side and Republicans on the negative side — the patterns would be identical, just mirrored.

Tallgrass enforces a **sign convention**: PC1 is oriented so that the Republican mean is positive. This is done after PCA runs, by simply flipping the sign of all scores if necessary. It's like deciding whether to read a ruler left-to-right or right-to-left — a convention, not a mathematical result.

**Codebase:** `analysis/02_pca/pca.py` (`orient_pc1()`)

## Validating PCA

How do we know PCA is capturing something real and not just noise? Tallgrass runs three validation checks:

### Holdout Validation

The pipeline randomly masks 20% of the non-missing cells in the vote matrix, re-runs PCA on the remaining 80%, and then predicts the held-out votes using the PCA reconstruction. If PCA is capturing real structure, the predictions should be better than simply guessing the base rate.

For the 91st Senate:
- **Base rate accuracy** (always guess Yea): ~82%
- **PCA reconstruction accuracy**: ~93%
- **AUC-ROC**: 0.97 (out of 1.0)

PCA's predictions are far better than guessing — confirming it's capturing genuine structure in the data.

### Sensitivity Analysis

The pipeline re-runs PCA with a stricter filter (10% minority threshold instead of 2.5%), which drops more near-unanimous votes. If the PC1 scores are robust, they should barely change. The correlation between default and sensitivity PC1 scores is typically above 0.95 — the results are stable across filtering choices.

### Reconstruction Error

For each legislator, the pipeline measures how well PCA reconstructs their full voting record from just the first few components. Legislators with unusually high reconstruction error — more than 2 standard deviations above the mean — may be unusual voters whose records don't fit the dominant patterns.

**Codebase:** `analysis/02_pca/pca.py` (`run_holdout_validation()`, `run_sensitivity()`)

## The Loadings: Which Bills Define Each Axis?

Each principal component isn't just a score for legislators — it also has **loadings** for each roll call. A loading tells you how much a particular vote contributes to that component.

Bills with high PC1 loadings are the most partisan — they sharply separate Democrats from Republicans. Bills with loadings near zero are non-partisan (everyone voted the same way). Bills with high PC2 loadings capture within-party disagreements.

The pipeline identifies the top-5 highest-loading bills for each significant component and reports their bill numbers, titles, and motions. This tells you what *substantive policy questions* define each ideological axis — a direct bridge from statistics to political interpretation.

## Running PCA

```bash
# Run PCA on the 91st Legislature
just pca 2025-26
```

The output includes Parquet data files (PC scores, loadings, eigenvalues), PNG plots (scree plot, ideological map, score distributions), and an HTML report with 14+ sections covering every diagnostic.

**Codebase:** `analysis/02_pca/pca.py` (the complete PCA phase)

---

## Key Takeaway

PCA compresses 600+ votes into a small number of components, each capturing a distinct pattern in the data. PC1 — the dominant component — almost always captures the party divide, explaining about 35–40% of all variation in a typical Kansas session. PC2 captures within-party variation. Together, these two numbers per legislator provide the first quantitative "ideology score" — a preliminary version that the IRT models in Volume 4 will refine with full statistical rigor.

---

*Terms introduced: Principal Component Analysis (PCA), principal component, eigenvalue, scree plot, explained variance, Horn's parallel analysis, PC1 (party divide), PC2 (within-party variation), loadings, sign convention, holdout validation, reconstruction error*

*Next: [Alternative Views: MCA and UMAP](ch04-mca-and-umap.md)*
