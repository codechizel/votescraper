# Chapter 4: Alternative Views: MCA and UMAP

> *PCA gives us the most common angle. But what if the data isn't best viewed from a single angle? MCA handles the categorical nature of votes more rigorously, and UMAP unfolds nonlinear structure that PCA can't see.*

---

## Why Multiple Methods?

If PCA works well — and it does — why bother with alternatives?

Because every statistical method makes assumptions, and different assumptions reveal different things. PCA assumes the data is continuous and relationships are linear. Legislative votes aren't continuous (they're Yea or Nay), and the relationships between legislators may not be perfectly linear (a moderate who sometimes crosses the aisle doesn't fit neatly on a straight line).

Running multiple methods on the same data is like looking at a sculpture from three different angles. If all three views agree on the basic shape — two parties, one dominant dimension — that's strong evidence the pattern is real. If they disagree, the disagreement itself is informative.

Tallgrass runs three dimensionality reduction methods: PCA (Chapter 3), **MCA** (this chapter), and **UMAP** (this chapter). The first two are closely related — different answers to the same question. UMAP is something entirely different.

---

## MCA: PCA's Categorical Cousin

### The Problem with Binary Numbers

When PCA works with votes, it treats Yea as 1 and Nay as 0. That's a reasonable simplification, but it throws away a piece of the puzzle: **absence**. PCA treats missing votes as "I don't know" (filled in with the legislator's average). But a legislator who was absent 40% of the time on party-line votes might be telling us something — maybe they're strategically avoiding tough choices.

**Multiple Correspondence Analysis (MCA)** takes a different approach. Instead of encoding votes as numbers (1 and 0), it treats each vote as a **category**: Yea, Nay, or Absent. Each of these categories is a first-class citizen in the analysis — absence isn't an inconvenience to be papered over but a signal to be measured.

### A Brief History

MCA comes from the French school of data analysis (*analyse des données*), developed largely by **Jean-Paul Benzécri** in the 1960s and 1970s. While American and British statisticians were developing methods for continuous data (like PCA), French statisticians were developing parallel methods for categorical data — the kind collected in surveys, censuses, and (it turns out) legislative votes.

MCA is a member of the same family as PCA — both are "factorial methods" that find the most important dimensions in data. But they use different tools under the hood:

| Aspect | PCA | MCA |
|--------|-----|-----|
| **Input** | Numbers (1s and 0s) | Categories (Yea, Nay, Absent) |
| **Distance** | Euclidean (straight-line) | Chi-squared (accounts for category frequency) |
| **Missing data** | Imputed (row-mean) | Treated as its own category |
| **Underlying math** | Eigendecomposition of covariance matrix | Correspondence analysis of indicator matrix |

### How MCA Works (Conceptual)

Think of it this way: PCA asks "how do legislators differ in the *amount* they vote Yea?" MCA asks "how do legislators differ in the *pattern* of their categorical responses?"

MCA builds an **indicator matrix** — a much wider table where each vote is expanded into separate columns for each category:

| | Vote A: Yea | Vote A: Nay | Vote A: Absent | Vote B: Yea | Vote B: Nay | Vote B: Absent | ... |
|--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Rep. Red** | 1 | 0 | 0 | 1 | 0 | 0 | ... |
| **Sen. Blue** | 0 | 1 | 0 | 0 | 0 | 1 | ... |

Each legislator now has a profile of which categories they belong to. MCA finds the dimensions along which these profiles differ the most — using chi-squared distance, which naturally accounts for the fact that Yea is much more common than Nay.

The result is a set of **dimensions** (Dim1, Dim2, etc.) that are analogous to PCA's principal components. In legislative data, Dim1 almost always captures the party divide — just like PC1.

### The Greenacre Correction

MCA has a known quirk: the raw **inertia** (MCA's version of explained variance) looks pessimistically low. A Dim1 that clearly separates parties might be reported as explaining only 8% of the total inertia. This happens because the indicator matrix introduces artificial dimensionality — the extra columns for each category inflate the denominator.

**Michael Greenacre** (1993) proposed a correction that fixes this problem. The corrected inertia focuses only on the meaningful variation in the data (the off-diagonal part of what's called the Burt table) and discards the artificial component. After correction, Dim1 might jump from 8% to 60% — a much more honest representation of its explanatory power.

Tallgrass uses the Greenacre correction by default. There's an alternative correction by Benzécri (1979), but Greenacre's is more conservative — it's less likely to overstate how well the analysis fits.

One caveat: after Greenacre correction, cumulative inertia can sometimes exceed 100%. This is a known artifact of the correction formula, not a bug.

**Codebase:** `analysis/03_mca/mca.py` (`fit_mca()`, constant `CORRECTION = "greenacre"`)

### What MCA Adds

In practice, MCA and PCA agree closely on the main dimension — Spearman correlations between PC1 and Dim1 are typically above 0.95. The party divide looks the same from both angles.

Where MCA adds value is in two specific situations:

1. **Strategic absence patterns:** If a group of legislators systematically avoids certain votes, MCA can detect this as a pattern. PCA, which fills in absences with the average, smooths this signal away.

2. **The biplot:** MCA produces a powerful visualization called a **biplot**, which shows both legislators *and* vote categories in the same space. A Yea marker near a cluster of Republicans means "Republicans tended to vote Yea on this bill." A Nay marker between parties means "voting Nay on this bill was bipartisan." This joint visualization doesn't have a natural equivalent in PCA.

### Horseshoe Detection

MCA also provides a second check for the horseshoe effect (covered in detail in Chapter 5). The pipeline fits a quadratic curve to the relationship between Dim2 and Dim1. If the R² exceeds 0.80, it means Dim2 is just a curved version of Dim1 — a horseshoe artifact, not a real second dimension. This confirms that the data is fundamentally one-dimensional.

```bash
# Run MCA on the 91st Legislature
just mca 2025-26
```

**Codebase:** `analysis/03_mca/mca.py` (the complete MCA phase)

---

## UMAP: The Nonlinear Map

### Why Go Nonlinear?

PCA and MCA are **linear** methods — they find the best straight-line projections of the data. That works beautifully when the underlying structure is simple (one dominant dimension, party). But what if there's more going on? What if some legislators cluster in unexpected ways that a straight-line projection would miss?

**UMAP** (Uniform Manifold Approximation and Projection) is a **nonlinear** dimensionality reduction technique introduced by **Leland McInnes, John Healy, and James Melville** in 2018. It produces intuitive 2D "maps" where legislators who vote alike end up near each other — regardless of whether the underlying structure is a line, a curve, or something more complex.

### The Analogy: Uncrumpling a Map

Imagine you have a flat map of Kansas — all the cities laid out on a flat piece of paper. Now crumple that paper into a ball. If you flatten the ball with a rolling pin (what PCA does), you'll get *a* flat picture, but the geography will be distorted — cities that were far apart might end up overlapping because the paper folded.

UMAP is like carefully uncrumpling the paper, smoothing it out while keeping the neighborhood relationships intact. Cities that were close together on the original map end up close together in the uncrumpled version. The result is a faithful reconstruction of the original layout — not a forced flattening.

In mathematical terms: PCA looks for straight lines, UMAP looks for curved surfaces (called **manifolds**). Legislative ideology might not be perfectly straight. UMAP can handle curves.

### How UMAP Works (Conceptual)

UMAP works in two phases:

**Phase 1: Build a neighborhood graph.**
For each legislator, UMAP identifies their nearest neighbors in the high-dimensional voting space. "Nearest" is measured by **cosine distance** — a metric that compares the *angle* between two voting records rather than their magnitude. Two legislators who vote Yea on the same bills and Nay on the same bills will have a small cosine distance, even if one participated in more votes than the other.

The result is a weighted graph: each legislator is connected to their closest voting partners, with stronger connections for more similar voting records.

**Phase 2: Lay out the graph in 2D.**
UMAP then arranges the legislators on a 2D plane such that the neighborhood relationships are preserved as faithfully as possible. Legislators who are connected by strong edges (similar voting) end up close together. Legislators with weak or no connections end up far apart.

### Key Parameters

UMAP has three important settings:

| Parameter | Tallgrass Default | What It Controls |
|-----------|------------------|------------------|
| **n_neighbors** | 15 | How many nearby legislators to consider. Low values emphasize fine-grained local structure (small cliques). High values emphasize the big picture (party-level separation). |
| **min_dist** | 0.1 | How tightly points pack together. Low values produce tight, distinct clusters. High values produce softer, more diffuse clouds. |
| **metric** | cosine | How distance is measured. Cosine distance works well for binary voting data because it focuses on *patterns* rather than *counts*. |

### The Critical Warning: Axes Are Meaningless

Here is the single most important thing to know about UMAP: **the axes don't mean anything.**

In PCA, PC1 has a clear interpretation (the direction of maximum variance, which in our case is the party divide). In UMAP, the x-axis (UMAP1) and y-axis (UMAP2) have no inherent meaning. A legislator at UMAP1 = +3 is not "more conservative" than one at UMAP1 = +1. The only thing that matters is **relative position** — who is close to whom.

This means:
- **You CAN interpret:** clusters, neighborhoods, gaps between groups, isolated outliers
- **You CANNOT interpret:** axis values, distances between distant clusters, coordinates across different UMAP runs

Think of UMAP as a seating chart, not a ruler. It tells you who's sitting near whom, but the table numbers are arbitrary.

### Stability and Robustness

Because UMAP involves randomness (the initial layout is random), Tallgrass runs multiple checks to ensure the results are stable:

**Multi-seed stability:** UMAP is run 5 times with different random seeds (42, 123, 456, 789, 1337). The results are compared using **Procrustes analysis** — a technique from shape analysis that finds the best rotation, reflection, and scaling to overlay two point clouds. If the similarity is high (above 0.7 on a 0-to-1 scale), the structure is stable across random starts.

Why Procrustes instead of simple correlation? Because UMAP axes are rotation-invariant — one run might put Republicans on the right, another might put them on the left. Procrustes handles this by allowing rotation before comparing.

**Sensitivity sweep:** UMAP is run with four different n_neighbors values (5, 15, 30, 50) to check whether the picture depends heavily on this parameter choice. If the party structure appears consistently across all four settings, it's a real feature of the data, not an artifact of the parameter.

**Trustworthiness score:** A quantitative measure (from scikit-learn) of how many neighbors in the 2D map were also neighbors in the original high-dimensional space. Scores above 0.80 are good; above 0.95 is excellent. Typical Kansas results: 0.85–0.95.

**Codebase:** `analysis/04_umap/umap_viz.py` (`run_stability_sweep()`, `run_sensitivity_sweep()`, `compute_trustworthiness()`)

### What UMAP Reveals

In Kansas data, the UMAP landscape typically shows:

- **Two clear clusters** — Republicans on one side, Democrats on the other. The gap between clusters is the partisan divide.
- **Within-cluster variation** — Within the Republican cluster, legislators spread from moderate (near the center) to conservative (far from center). The Democratic cluster is smaller but shows similar internal variation.
- **Cross-party outliers** — Occasionally, a legislator from one party appears on the other party's side of the map. This could mean they're a genuine cross-party voter (a moderate Democrat who votes with Republicans on many issues) or an **imputation artifact** (a legislator with many missing votes whose imputed values landed them in the wrong cluster). The pipeline flags any legislator with more than 50% imputed votes as a potential artifact.
- **Isolated points** — A legislator floating away from both clusters has a voting record unlike anyone else's — a genuine independent, a contrarian, or someone who missed most of the session.

### Validation Against PCA and IRT

To confirm that UMAP is capturing the same ideological structure as the other methods, the pipeline computes Spearman correlations:

- **UMAP1 vs. PCA PC1:** typically ρ = 0.90–0.98
- **UMAP1 vs. IRT ideal points** (from Phase 5): typically ρ = 0.85–0.95

These high correlations confirm that UMAP, PCA, and IRT all agree on the ordering of legislators from liberal to conservative — even though they use completely different mathematical machinery.

```bash
# Run UMAP on the 91st Legislature
just umap 2025-26
```

**Codebase:** `analysis/04_umap/umap_viz.py` (the complete UMAP phase)

---

## Comparing the Three Methods

| Feature | PCA | MCA | UMAP |
|---------|-----|-----|------|
| **Best for** | Quantitative overview | Categorical structure | Visual exploration |
| **Data type** | Binary (1/0) | Categorical (Yea/Nay/Absent) | Binary (1/0) |
| **Distance metric** | Euclidean | Chi-squared | Cosine |
| **Axes meaningful?** | Yes (PC1 = most variance) | Yes (Dim1 = most inertia) | No (only positions matter) |
| **Handles absence** | Imputation (fills with average) | As a category | Imputation |
| **Linear/nonlinear** | Linear | Linear | Nonlinear |
| **Speed** | Fast | Fast | Moderate |
| **Key output** | Eigenvalues, loadings | Inertia, biplot | Neighborhood map |

**When the three methods agree** (which they almost always do in Kansas data), you can be confident that the patterns are real. The party divide, the within-party spread, the isolation of particular legislators — all three views confirm the same structure.

**When they disagree**, the disagreement is itself useful:
- If UMAP shows a cluster that PCA doesn't, there may be a nonlinear grouping (a faction that votes together on specific issues but not on the general left-right spectrum).
- If MCA's Dim2 shows more structure than PCA's PC2, the absence patterns may be carrying meaningful information.

---

## Key Takeaway

MCA extends PCA by treating votes as categories (Yea/Nay/Absent) rather than numbers (1/0/missing), and the Greenacre correction ensures honest reporting of explained variance. UMAP takes a fundamentally different approach — building a neighborhood graph and laying it out in 2D — to produce an intuitive map where proximity means agreement. Together with PCA, these three methods provide three independent views of the same data. Where they agree, the patterns are trustworthy. Where they disagree, the difference teaches us something.

---

*Terms introduced: Multiple Correspondence Analysis (MCA), indicator matrix, chi-squared distance, inertia, Greenacre correction, biplot, UMAP, manifold, cosine distance, Procrustes analysis, trustworthiness, sensitivity sweep, imputation artifact*

*Next: [The Horseshoe Problem (When the Map Lies)](ch05-horseshoe-problem.md)*
