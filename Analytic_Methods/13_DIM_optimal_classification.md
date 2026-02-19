# Optimal Classification (OC)

**Category:** Dimensionality Reduction / Ideal Point Estimation
**Prerequisites:** `01_DATA_vote_matrix_construction`
**Complexity:** Medium-High (requires R)
**Related:** `12_DIM_w_nominate`, `09_DIM_principal_component_analysis`

## What It Measures

Optimal Classification is a non-parametric method for placing legislators in a low-dimensional ideological space. Unlike W-NOMINATE (which assumes a specific Gaussian utility function), OC makes no assumptions about the *shape* of the utility function — it only assumes legislators have ideal points and vote for the closer outcome. OC minimizes the total number of classification errors: legislator-vote pairs where a legislator voted on the "wrong" side of the estimated cutting line.

This makes OC more robust than NOMINATE when the data doesn't fit the Gaussian utility assumption. It is especially valuable for small chambers (like the Kansas Senate with 42 members) where parametric assumptions may be too strong.

## Questions It Answers

- Same as W-NOMINATE: where does each legislator sit ideologically?
- What is the minimum classification error achievable with a spatial model?
- Does the non-parametric model reveal different patterns than the parametric NOMINATE?
- Which votes are misclassified even with the best spatial model? (These votes may involve dimensions not captured by the 1D/2D model.)

## Mathematical Foundation

### The Optimization Problem

Given $n$ legislators and $m$ roll calls, find:
- Ideal points $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^d$
- Cutting lines (hyperplanes) $\mathbf{h}_1, \ldots, \mathbf{h}_m$

That minimize:

$$\sum_{i=1}^{n} \sum_{j=1}^{m} \mathbb{1}[\text{vote}_{ij} \text{ misclassified by } (\mathbf{x}_i, \mathbf{h}_j)]$$

A vote is correctly classified if the legislator's ideal point is on the correct side of the cutting line for that roll call.

### Algorithm

OC uses an alternating algorithm:
1. **Fix legislators, optimize cutting lines**: For each roll call, find the optimal cutting line (a line in $d$-D space that best separates Yea from Nay voters). This is a linear classification problem.
2. **Fix cutting lines, optimize legislators**: For each legislator, find the point in $d$-D space that is on the correct side of the most cutting lines. This is solved by a method related to linear programming.
3. **Repeat** until convergence.

### Key Difference from NOMINATE

NOMINATE assumes votes follow a probabilistic model (Gaussian utility + logistic error). OC is deterministic — a vote is either correctly classified or not. This means:
- OC has no probabilistic interpretation (no "probability of voting Yea")
- OC cannot produce measures of uncertainty
- OC tends to produce slightly higher classification rates than NOMINATE
- OC is less sensitive to model misspecification

## Python/R Implementation

### R Implementation (via rpy2)

```python
import pandas as pd
import numpy as np

def run_optimal_classification(
    vote_matrix: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    dims: int = 2,
    polarity_slug: str | None = None,
) -> dict:
    """Run Optimal Classification via R's oc package.

    Args:
        vote_matrix: Binary vote matrix (1=Yea, 0=Nay, NaN=missing).
        legislator_meta: DataFrame with party info.
        dims: Number of dimensions (1 or 2).
        polarity_slug: Slug of a known conservative for sign identification.

    Returns:
        Dict with 'ideal_points', 'fit_stats', 'cutting_lines'.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    pandas2ri.activate()

    pscl = importr("pscl")
    oc_pkg = importr("oc")

    # Convert to NOMINATE format (1=Yea, 6=Nay, 9=Missing)
    r_matrix = vote_matrix.copy()
    r_matrix = r_matrix.map(lambda x: 1 if x == 1 else (6 if x == 0 else 9))

    rc = pscl.rollcall(
        pandas2ri.py2rpy(r_matrix),
        yea=1, nay=6, missing=9,
    )

    # Set polarity
    if polarity_slug:
        polarity_idx = vote_matrix.index.get_loc(polarity_slug) + 1
    else:
        rep_slugs = legislator_meta.loc[
            legislator_meta["party"] == "Republican"
        ].index.intersection(vote_matrix.index)
        polarity_idx = vote_matrix.index.get_loc(rep_slugs[0]) + 1

    polarity = ro.IntVector([polarity_idx] * dims)

    # Run OC
    result = oc_pkg.oc(rc, dims=dims, polarity=polarity)

    # Extract results
    legislators = pandas2ri.rpy2py(result.rx2("legislators"))

    fit = {
        "dimensions": dims,
        "correct_class": float(result.rx2("fits").rx2("percent.correctly.classified")[dims - 1]),
    }

    return {
        "ideal_points": legislators,
        "fit_stats": fit,
    }
```

### Pure R Script

```r
# optimal_classification.R
library(pscl)
library(oc)

votes <- read.csv("vote_matrix_binary.csv", row.names=1)

vote_data <- as.matrix(votes)
vote_data[vote_data == 1] <- 1
vote_data[vote_data == 0] <- 6
vote_data[is.na(vote_data)] <- 9

rc <- rollcall(vote_data, yea=1, nay=6, missing=9)

# Run Optimal Classification (2 dimensions)
result <- oc(rc, dims=2, polarity=c(1, 1))

# Save results
write.csv(result$legislators, "oc_ideal_points.csv")

# Summary and plots
summary(result)
plot(result)
```

## Interpretation Guide

### Comparing OC and NOMINATE Results

| Aspect | NOMINATE | OC |
|--------|----------|-----|
| Correct classification | 85-90% typical | 87-93% typical (slightly higher) |
| Ideal point scale | Bounded (-1, 1) | Bounded (-1, 1) |
| Correlation between methods | > 0.95 on first dimension |
| Uncertainty estimates | No (point estimates) | No (point estimates) |
| Handling of noise | Probabilistic (absorbs noise) | Deterministic (every error counts) |

### When OC and NOMINATE Disagree

If OC and NOMINATE produce noticeably different ideal points for certain legislators, it usually means:
1. Those legislators' voting patterns don't fit the Gaussian utility assumption
2. They may be "strategic" voters whose behavior depends on context rather than pure ideology
3. The legislator has unusual patterns of absence that affect the two methods differently

### Classification Accuracy Benchmarks

| Correct Classification | Interpretation |
|-----------------------|----------------|
| > 95% | Extremely well-structured legislature (rare) |
| 90-95% | Strong partisan structure |
| 85-90% | Normal for a two-party legislature |
| 80-85% | Moderate partisanship or complex multi-dimensional conflict |
| < 80% | Spatial model is a poor fit; consider more dimensions or non-spatial methods |

## Kansas-Specific Considerations

- **Particularly useful for the Senate.** Armstrong et al. (2014) showed that OC outperforms NOMINATE in small chambers because it doesn't rely on parametric assumptions that need large samples to estimate well.
- **Run both OC and NOMINATE** and compare. Agreement validates the spatial model; disagreement points to interesting legislators or votes.
- **OC is faster than NOMINATE** for the same dataset, making it practical for sensitivity analyses (e.g., testing different vote filters).

## Feasibility Assessment

- **Data size**: Excellent for 170 legislators
- **Compute time**: Under 1 minute
- **Libraries**: R (`oc`, `pscl`), optionally `rpy2`
- **Difficulty**: Same as NOMINATE (requires R setup)

## Key References

- Poole, Keith T. "Non-Parametric Unfolding of Binary Choice Data." *Political Analysis* 8(3), 2000.
- Armstrong, David A. II, et al. "Analyzing Spatial Models of Choice and Judgment with R." CRC Press, 2014. Chapter 6.
- `oc` R package: https://cran.r-project.org/web/packages/oc/index.html
