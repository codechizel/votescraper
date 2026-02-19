# W-NOMINATE (Weighted Nominal Three-step Estimation)

**Category:** Dimensionality Reduction / Ideal Point Estimation
**Prerequisites:** `01_DATA_vote_matrix_construction`
**Complexity:** High (requires R)
**Related:** `09_DIM_principal_component_analysis`, `13_DIM_optimal_classification`, `15_BAY_irt_ideal_points`

## What It Measures

W-NOMINATE is the gold standard method for estimating legislator ideal points (ideological positions) in political science. Developed by Keith Poole and Howard Rosenthal, it models each vote as a choice between two points in a low-dimensional policy space: the Yea outcome location and the Nay outcome location. Each legislator has an ideal point, and they vote for the outcome closer to their ideal point, with some probabilistic error.

W-NOMINATE is the single-session variant of DW-NOMINATE (the "D" stands for "Dynamic," used for cross-session comparisons). For a single Kansas session, W-NOMINATE is the appropriate choice.

Every political science paper on legislative ideology references NOMINATE. The [Voteview](https://voteview.com) project publishes NOMINATE scores for every member of every US Congress going back to 1789.

## Questions It Answers

- Where does each legislator sit on the ideological spectrum (with a well-calibrated spatial model)?
- How many dimensions of conflict does the legislature have?
- Which bills best discriminate between legislators (high "discrimination" parameters)?
- What is the "cutting line" for each vote — the line in ideological space that separates Yea from Nay voters?
- How well does a 1D or 2D spatial model explain voting behavior (classification accuracy)?

## Mathematical Foundation

### Spatial Voting Model

Each legislator $i$ has an ideal point $\mathbf{x}_i \in \mathbb{R}^d$ (typically $d = 1$ or $2$).

Each roll call $j$ has two outcome points: $\mathbf{y}_j$ (Yea outcome) and $\mathbf{z}_j$ (Nay outcome).

Legislator $i$ votes Yea on roll call $j$ if the Yea outcome is closer to their ideal point:

$$U_i(\mathbf{y}_j) > U_i(\mathbf{z}_j)$$

Where utility follows a Gaussian kernel:

$$U_i(\mathbf{y}_j) = \beta \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{y}_j\|^2}{2}\right) + \epsilon_{ij}$$

The $\beta$ parameter controls the signal-to-noise ratio, and $\epsilon_{ij}$ is a random error term.

### NOMINATE Parameters (Per Roll Call)

Rather than estimating the full outcome points $\mathbf{y}_j$ and $\mathbf{z}_j$, NOMINATE parameterizes each roll call by:
- **Midpoint** $\mathbf{m}_j = (\mathbf{y}_j + \mathbf{z}_j)/2$ — the center of the cutting line
- **Normal vector** $\mathbf{n}_j = \mathbf{y}_j - \mathbf{z}_j$ — the direction that separates Yea from Nay

The probability that legislator $i$ votes Yea on roll call $j$:

$$P(\text{Yea}_{ij}) = F\left(\beta \cdot \mathbf{n}_j \cdot (\mathbf{x}_i - \mathbf{m}_j)\right)$$

Where $F$ is a logistic or normal CDF link function.

### Estimation

W-NOMINATE uses an alternating optimization:
1. Fix roll call parameters, optimize legislator ideal points
2. Fix ideal points, optimize roll call parameters
3. Repeat until convergence

This is a maximum likelihood estimation procedure.

## Python/R Implementation

W-NOMINATE has no mature Python implementation. The standard approach is to use R via `rpy2`.

### R Implementation (via rpy2)

```python
import pandas as pd
import numpy as np

def run_wnominate(
    vote_matrix: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    dims: int = 2,
    polarity_slugs: tuple[str, str] | None = None,
) -> dict:
    """Run W-NOMINATE via R's wnominate package.

    Args:
        vote_matrix: Binary vote matrix (1=Yea, 0=Nay, NaN=missing).
        legislator_meta: DataFrame with party, chamber info.
        dims: Number of dimensions (1 or 2).
        polarity_slugs: Tuple of (conservative_slug, liberal_slug) for sign identification.

    Returns:
        Dict with 'ideal_points', 'rollcall_params', 'fit_stats'.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    pandas2ri.activate()

    # Import R packages
    pscl = importr("pscl")
    wnominate = importr("wnominate")

    # Convert vote matrix to R format
    # NOMINATE convention: 1=Yea, 6=Nay, 9=Missing
    r_matrix = vote_matrix.copy()
    r_matrix = r_matrix.map(lambda x: 1 if x == 1 else (6 if x == 0 else 9))

    # Create rollcall object
    r_votes = pandas2ri.py2rpy(r_matrix)
    legis_names = ro.StrVector(vote_matrix.index.tolist())
    vote_names = ro.StrVector(vote_matrix.columns.tolist())

    # Create party vector for the rollcall object
    meta_aligned = legislator_meta.loc[vote_matrix.index]
    party_codes = meta_aligned["party"].map({"Republican": "R", "Democrat": "D"})

    rc = pscl.rollcall(
        r_votes,
        yea=1,
        nay=6,
        missing=9,
        legis_names=legis_names,
        vote_names=vote_names,
    )

    # Identify polarity (which direction is "conservative")
    if polarity_slugs:
        polarity = [
            vote_matrix.index.get_loc(polarity_slugs[0]) + 1,  # R is 1-indexed
        ] * dims
    else:
        # Auto-detect: use a known conservative Republican
        rep_slugs = meta_aligned[meta_aligned["party"] == "Republican"].index
        # Use the first Republican as polarity anchor
        polarity = [vote_matrix.index.get_loc(rep_slugs[0]) + 1] * dims

    # Run W-NOMINATE
    result = wnominate.wnominate(rc, dims=dims, polarity=ro.IntVector(polarity))

    # Extract results
    legislators_r = result.rx2("legislators")
    ideal_points = pandas2ri.rpy2py(legislators_r)

    rollcalls_r = result.rx2("rollcalls")
    rollcall_params = pandas2ri.rpy2py(rollcalls_r)

    fit = {
        "dimensions": dims,
        "correct_class_1d": float(result.rx2("fits").rx2("percent.correctly.classified")[0]),
    }
    if dims >= 2:
        fit["correct_class_2d"] = float(result.rx2("fits").rx2("percent.correctly.classified")[1])

    return {
        "ideal_points": ideal_points,
        "rollcall_params": rollcall_params,
        "fit_stats": fit,
    }
```

### Pure R Script (Alternative)

If `rpy2` is problematic, write an R script and call it from Python:

```r
# wnominate_analysis.R
library(pscl)
library(wnominate)

# Load vote matrix (exported from Python as CSV)
votes <- read.csv("vote_matrix_binary.csv", row.names=1)

# Create rollcall object
# Recode: 1=Yea, 0=Nay, NA=missing
vote_data <- as.matrix(votes)
vote_data[vote_data == 1] <- 1
vote_data[vote_data == 0] <- 6
vote_data[is.na(vote_data)] <- 9

rc <- rollcall(vote_data, yea=1, nay=6, missing=9)

# Run W-NOMINATE (2 dimensions)
result <- wnominate(rc, dims=2, polarity=c(1, 1))

# Save ideal points
write.csv(result$legislators, "wnominate_ideal_points.csv")
write.csv(result$rollcalls, "wnominate_rollcall_params.csv")

# Summary
summary(result)
plot(result)
```

```python
import subprocess

subprocess.run(["Rscript", "wnominate_analysis.R"], check=True)
ideal_points = pd.read_csv("wnominate_ideal_points.csv", index_col=0)
```

## Output and Interpretation

### Legislator Ideal Points

W-NOMINATE produces, for each legislator:
- `coord1D`: First dimension score (left-right ideology)
- `coord2D`: Second dimension score (if 2D model)
- `GMP`: Geometric Mean Probability — how well the model predicts this legislator's votes (0 to 1)
- `CC`: Correct Classification rate for this legislator

### Roll Call Parameters

For each roll call:
- `midpoint1D`, `midpoint2D`: Where the cutting line sits in ideological space
- `spread1D`, `spread2D`: How steeply the vote probability changes across the cutting line
- `PRE`: Proportional Reduction in Error — how much better the spatial model predicts this vote compared to always guessing the majority outcome

### Classification Accuracy

- **1D correct classification 85-90%**: Typical for a partisan legislature. Means 85-90% of all votes can be predicted from a single ideological dimension.
- **2D correct classification 90-95%**: The second dimension captures additional structure (5-10% of votes depend on it).
- **If 1D and 2D are nearly the same** (e.g., 87% vs. 88%): The legislature is effectively one-dimensional.

### Cutting Line Visualization

```python
def plot_cutting_lines(
    ideal_points: pd.DataFrame,
    rollcall_params: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    n_votes: int = 10,
    save_path: str | None = None,
):
    """Plot legislator ideal points with cutting lines for selected votes."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot ideal points
    meta = legislator_meta.loc[ideal_points.index]
    colors = meta["party"].map({"Republican": "#E81B23", "Democrat": "#0015BC"})
    ax.scatter(ideal_points["coord1D"], ideal_points["coord2D"],
              c=colors, s=40, alpha=0.6, edgecolors="black", linewidth=0.3)

    # Plot cutting lines for the most discriminating votes
    top_votes = rollcall_params.nlargest(n_votes, "PRE")
    for _, vote in top_votes.iterrows():
        # Cutting line is perpendicular to the normal vector
        # passing through the midpoint
        mid1 = vote["midpoint1D"]
        mid2 = vote["midpoint2D"]
        spread1 = vote["spread1D"]
        spread2 = vote["spread2D"]

        # Draw cutting line
        normal = np.array([spread1, spread2])
        tangent = np.array([-spread2, spread1])
        tangent = tangent / np.linalg.norm(tangent) * 2

        ax.plot(
            [mid1 - tangent[0], mid1 + tangent[0]],
            [mid2 - tangent[1], mid2 + tangent[1]],
            "k-", alpha=0.2, linewidth=0.5,
        )

    ax.set_xlabel("First Dimension (left-right)")
    ax.set_ylabel("Second Dimension")
    ax.set_title("W-NOMINATE Ideal Points with Cutting Lines")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

## Advantages and Limitations

### Advantages
- **Gold standard** in political science — results are directly comparable to the vast NOMINATE literature
- **Well-calibrated** spatial model with a principled probabilistic foundation
- **Classification accuracy** provides a clear measure of model fit
- **Cutting lines** provide intuitive visualization of how each vote divides the chamber

### Limitations
- **Requires R** — no mature Python implementation
- **Parametric assumptions**: Gaussian utility kernel may not fit all legislators equally well
- **Cannot handle ordinal or multi-category votes** (only binary Yea/Nay)
- **No uncertainty quantification**: Unlike Bayesian IRT, NOMINATE produces point estimates without credible intervals
- **Identification requires polarity constraints**: Must specify which direction is "conservative"

## Kansas-Specific Considerations

- **Run per chamber.** W-NOMINATE works best within a single chamber where all legislators vote on the same roll calls.
- **Senate (42 members) may have identification issues.** Small chambers can produce unstable estimates. The `13_DIM_optimal_classification` may be more appropriate for the Senate. See: Armstrong et al., "Small Chamber Ideal Point Estimation" (*Political Analysis*, 2014).
- **Polarity**: Use a known conservative Republican (e.g., a Freedom Caucus-aligned member) as the polarity anchor.
- **Compare with PCA**: PC1 scores should correlate > 0.95 with NOMINATE 1D scores. If they don't, investigate why.
- **Compare with Bayesian IRT**: The key advantage of `15_BAY_irt_ideal_points` over NOMINATE is posterior distributions (uncertainty) for each ideal point.

## Feasibility Assessment

- **Data size**: 170 legislators x ~400 contested roll calls = standard NOMINATE problem
- **Compute time**: 1-5 minutes for 2D
- **Libraries**: R (`wnominate`, `pscl`), optionally `rpy2` for Python bridge
- **Difficulty**: High (requires R setup). If you want to stay in Python, PCA and Bayesian IRT produce very similar results.

## Key References

- Poole, Keith T., and Howard Rosenthal. "A Spatial Model for Legislative Roll Call Analysis." *American Journal of Political Science* 29(2), 1985.
- Poole, Keith T., and Howard Rosenthal. *Congress: A Political-Economic History of Roll Call Voting*. Oxford University Press, 1997.
- Poole, Keith T., Jeffrey Lewis, Howard Rosenthal, James Lo, and Royce Carroll. "Scaling Roll Call Votes with wnominate in R." *Journal of Statistical Software* 42(14), 2011. https://www.jstatsoft.org/article/view/v042i14
- Armstrong, David A. II, et al. "Small Chamber Ideal Point Estimation." *Political Analysis* 22(3), 2014.
- Voteview project: https://voteview.com
