# Agreement Matrix and Hierarchically Clustered Heatmap

**Category:** Exploratory Data Analysis
**Prerequisites:** `01_DATA_vote_matrix_construction`
**Complexity:** Low-Medium
**Feeds into:** `18_CLU_hierarchical_clustering`, `20_NET_covoting_network`

## What It Measures

The agreement matrix computes pairwise voting agreement between every pair of legislators — the fraction of roll calls where both were present and voted the same way. When visualized as a heatmap with hierarchical clustering (a "clustermap"), it reveals the full structure of legislative alliances: party blocs, intra-party factions, cross-party moderates, and isolated mavericks. This is often the single most informative visualization in a legislative analysis.

## Questions It Answers

- How sharply do the parties separate in voting behavior?
- Are there visible factions within the majority party?
- Which legislators are most similar to members of the opposite party?
- Is the partisan divide equally sharp across both chambers?
- Do any unexpected cross-party alliances appear?

## Mathematical Foundation

### Pairwise Agreement Rate

For legislators $i$ and $j$, let $V_{ij}$ be the set of roll calls where both voted Yea or Nay (not absent):

$$\text{Agreement}(i, j) = \frac{|\{v \in V_{ij} : \text{vote}_i(v) = \text{vote}_j(v)\}|}{|V_{ij}|}$$

This ranges from 0 (always vote opposite) to 1 (always vote the same). Random chance between two independent coin-flippers would produce ~0.5, but in practice the high Yea rate inflates agreement. On contested votes only, the baseline is closer to 0.5.

### Cohen's Kappa (Chance-Corrected Agreement)

Raw agreement is inflated when the base rate is extreme. In Kansas, ~82% of all votes are Yea, so two random legislators would agree ~70% of the time by pure chance. **Cohen's Kappa** corrects for this:

$$\kappa(i, j) = \frac{p_o - p_e}{1 - p_e}$$

Where:
- $p_o$ = observed agreement rate (same as Agreement above)
- $p_e$ = expected agreement by chance = $p_i^{Yea} \cdot p_j^{Yea} + p_i^{Nay} \cdot p_j^{Nay}$

**Properties:**
- $\kappa = 1$: Perfect agreement beyond chance
- $\kappa = 0$: Agreement no better than chance
- $\kappa < 0$: Agreement worse than chance (actively opposing)

**When to use which:**
- Use **raw agreement** for visualization (intuitive, familiar scale)
- Use **Cohen's Kappa** for analysis where the high Yea base rate would inflate similarity (clustering, network thresholding)
- Use Kappa when comparing across chambers or sessions with different base rates

### Phi Correlation and Mutual Information

Two additional chance-corrected metrics worth considering:

- **Phi coefficient**: Pearson correlation on the binary vote vectors. Equivalent to Kappa when both legislators have the same marginal vote distribution. Range: [-1, 1].
- **Mutual Information**: Information-theoretic measure of dependence. Unlike correlation/Kappa, captures non-linear relationships. Range: [0, ∞), or normalize to [0, 1].

### Distance Matrix

For clustering and network analysis, convert agreement to distance:

$$\text{Distance}(i, j) = 1 - \text{Agreement}(i, j)$$

Or, for the chance-corrected version:

$$\text{Distance}_\kappa(i, j) = 1 - \kappa(i, j)$$

## Python Implementation

### Computing the Agreement Matrix

```python
import pandas as pd
import numpy as np

def compute_agreement_matrix(vote_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise agreement rate between all legislators.

    Args:
        vote_matrix: Binary vote matrix (1=Yea, 0=Nay, NaN=absent).
                     Rows are legislators, columns are roll calls.

    Returns:
        Square DataFrame of agreement rates, indexed by legislator_slug.
    """
    n = len(vote_matrix)
    values = vote_matrix.values  # Shape: (n_legislators, n_rollcalls)

    # Mask for non-missing votes
    present = ~np.isnan(values)

    # Number of votes where both legislators were present
    # Shape: (n, n) — matrix multiplication of presence masks
    both_present = present.astype(float) @ present.astype(float).T

    # Number of votes where both voted the same way
    # For binary data: agree when both 1 or both 0
    # Same = both_yea + both_nay
    both_yea = values @ values.T  # dot product counts matching 1s (handles NaN via 0)
    # Need to handle NaN: replace NaN with a value that won't match
    values_filled = np.where(present, values, -999)
    both_nay = (1 - values_filled) @ (1 - values_filled).T
    # Fix: both_nay overcounts when either is NaN
    # Better approach: use nan-aware computation
    agree = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            mask = present[i] & present[j]
            if mask.sum() == 0:
                agree[i, j] = agree[j, i] = np.nan
            else:
                agree[i, j] = agree[j, i] = (
                    (values[i, mask] == values[j, mask]).sum() / mask.sum()
                )

    agreement = pd.DataFrame(agree, index=vote_matrix.index, columns=vote_matrix.index)
    return agreement
```

### Computing Cohen's Kappa Matrix

```python
def compute_kappa_matrix(vote_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Cohen's Kappa (chance-corrected agreement).

    This is strictly better than raw agreement when the Yea base rate is high,
    because it accounts for the fact that two legislators who both vote Yea 90%
    of the time will agree 82% of the time by pure chance.
    """
    n = len(vote_matrix)
    values = vote_matrix.values.astype(float)
    present = ~np.isnan(values)

    kappa = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            mask = present[i] & present[j]
            n_shared = mask.sum()
            if n_shared == 0:
                kappa[i, j] = kappa[j, i] = np.nan
                continue

            vi, vj = values[i, mask], values[j, mask]

            # Observed agreement
            p_o = (vi == vj).mean()

            # Expected agreement by chance
            p_i_yea, p_j_yea = vi.mean(), vj.mean()
            p_e = p_i_yea * p_j_yea + (1 - p_i_yea) * (1 - p_j_yea)

            if p_e == 1.0:
                kappa[i, j] = kappa[j, i] = 1.0  # Both vote identically always
            else:
                kappa[i, j] = kappa[j, i] = (p_o - p_e) / (1 - p_e)

    np.fill_diagonal(kappa, 1.0)
    return pd.DataFrame(kappa, index=vote_matrix.index, columns=vote_matrix.index)
```

**Optimized version** using vectorized operations for larger datasets:

```python
def compute_agreement_matrix_fast(vote_matrix: pd.DataFrame) -> pd.DataFrame:
    """Vectorized agreement matrix computation."""
    vals = vote_matrix.values.astype(float)
    present = ~np.isnan(vals)

    # Replace NaN with -999 (won't match anything)
    safe = np.nan_to_num(vals, nan=-999.0)

    n_legislators = len(vals)
    n_rollcalls = vals.shape[1]

    # Both present count
    both_present = present.astype(float) @ present.astype(float).T

    # Agreement count: use broadcasting
    # For each pair (i,j), count roll calls where both present AND same vote
    # Trick: for binary votes, same = (a == b) = (a*b) + (1-a)*(1-b) when both present
    yea_vals = np.where(present, vals, 0)
    nay_vals = np.where(present, 1 - vals, 0)

    both_yea = yea_vals @ yea_vals.T
    both_nay = nay_vals @ nay_vals.T
    same_vote = both_yea + both_nay

    agreement = np.where(both_present > 0, same_vote / both_present, np.nan)
    np.fill_diagonal(agreement, 1.0)

    return pd.DataFrame(agreement, index=vote_matrix.index, columns=vote_matrix.index)
```

### Creating the Clustermap

```python
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def plot_agreement_clustermap(
    agreement: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    chamber: str | None = None,
    figsize: tuple = (16, 14),
    save_path: str | None = None,
):
    """Plot a hierarchically clustered heatmap of legislator agreement."""
    # Filter to chamber if specified
    if chamber:
        slugs = legislator_meta[legislator_meta["chamber"] == chamber].index
        agreement = agreement.loc[
            agreement.index.isin(slugs),
            agreement.columns.isin(slugs),
        ]

    # Create distance matrix for clustering
    distance = 1 - agreement.fillna(0.5)  # NaN -> 0.5 (neutral)
    np.fill_diagonal(distance.values, 0)

    # Party color mapping for row/column annotations
    meta_aligned = legislator_meta.loc[agreement.index]
    party_colors = meta_aligned["party"].map({
        "Republican": "#E81B23",
        "Democrat": "#0015BC",
    })

    # Create clustermap
    g = sns.clustermap(
        agreement,
        method="ward",
        metric="euclidean",
        cmap="RdYlGn",
        vmin=0.4,
        vmax=1.0,
        figsize=figsize,
        row_colors=party_colors,
        col_colors=party_colors,
        xticklabels=False,
        yticklabels=True if len(agreement) <= 50 else False,
        dendrogram_ratio=(0.15, 0.15),
    )

    title = f"Legislator Voting Agreement — {chamber or 'All'}"
    g.fig.suptitle(title, y=1.02, fontsize=14)

    if save_path:
        g.savefig(save_path, dpi=150, bbox_inches="tight")

    return g
```

### Usage

```python
# Build vote matrix (binary encoding)
vote_matrix, legislator_meta = build_vote_matrix(
    "ks_2025_26_votes.csv",
    "ks_2025_26_legislators.csv",
    encoding="binary",
)

# Filter to contested votes
vote_matrix_contested = filter_lopsided_votes(vote_matrix, min_minority_pct=0.025)

# Split by chamber
chambers = split_by_chamber(vote_matrix_contested, legislator_meta)

# Compute and plot for each chamber
for chamber_name, chamber_matrix in chambers.items():
    agreement = compute_agreement_matrix_fast(chamber_matrix)
    plot_agreement_clustermap(
        agreement,
        legislator_meta,
        chamber=chamber_name,
        save_path=f"agreement_heatmap_{chamber_name.lower()}.png",
    )
```

## Reading the Heatmap

The clustermap has four key visual elements:

1. **Color intensity**: Green = high agreement (>80%), Yellow = moderate (~60-70%), Red = low agreement (<50%). The strongest partisan legislatures show a clear green diagonal blocks with red off-diagonal blocks.

2. **Dendrograms** (tree structures on the sides): Show the hierarchical clustering. The first major branch typically separates the two parties. Subsequent branches reveal factions within parties.

3. **Row/column color bars**: Colored by party (red = Republican, blue = Democrat). If the clustering aligns perfectly with the color bars, partisanship explains voting. If clusters break across party colors, something more complex is happening.

4. **Off-diagonal patterns**: Cross-party agreement blocks (green patches between a group of Republicans and a group of Democrats) indicate bipartisan coalitions on specific issues.

## Interpretation Guide

- **Clean two-block structure** (green diagonal blocks, red off-diagonal): Strong partisanship. The legislature votes along party lines most of the time.
- **Multiple blocks within a party**: Intra-party factions. In Kansas, look for moderate Republican blocks that agree more with Democrats than with conservative Republicans.
- **Diffuse structure** (no clear blocks): Weak partisanship. Votes are driven by issue-specific coalitions rather than party affiliation.
- **Agreement range 0.6-0.9** for within-party pairs is typical. Below 0.6 within-party suggests serious internal divisions. Cross-party agreement above 0.7 suggests bipartisan moderates.

## Kansas-Specific Considerations

- **Analyze chambers separately.** House and Senate members vote on different roll calls, so cross-chamber agreement is undefined for most votes.
- **Republican factions are the main story.** With a supermajority, the Republican party likely has internal divisions (moderate vs. conservative) that show up as sub-blocks in the heatmap.
- **Filter to contested votes** before computing agreement. On near-unanimous votes everyone agrees, which inflates agreement scores and masks real differences.
- The Senate (42 members) produces a much more readable heatmap than the House (130 members). For the House, consider showing legislator names only for outliers.

## Key References

- Andris, Clio, et al. "The Rise of Partisanship and Super-Cooperators in the US House of Representatives." *PLOS ONE* 10(4), 2015. (Uses agreement networks and their visualization to show increasing partisan polarization.)
- Seaborn clustermap documentation: https://seaborn.pydata.org/generated/seaborn.clustermap.html
