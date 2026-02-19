# Principal Component Analysis (PCA) on Vote Matrix

**Category:** Dimensionality Reduction / Ideal Point Estimation
**Prerequisites:** `01_DATA_vote_matrix_construction`
**Complexity:** Medium
**Related:** `10_DIM_correspondence_analysis`, `12_DIM_w_nominate`, `15_BAY_irt_ideal_points`

## What It Measures

PCA on the vote matrix extracts the principal axes of variation in legislators' voting behavior. The first principal component (PC1) almost always corresponds to the left-right ideological spectrum — it is the single dimension that explains the most variance in how legislators vote differently from each other. PC2 typically captures a second, cross-cutting dimension (e.g., urban vs. rural, establishment vs. insurgent, or specific policy cleavages like fiscal vs. social).

PCA is the simplest Python-native method for estimating legislator "ideal points" — positions on a latent ideological spectrum. Research shows that PC1 scores are highly correlated (r > 0.95) with NOMINATE scores, making PCA an excellent and accessible approximation.

## Questions It Answers

- What is the primary dimension of legislative conflict? (Almost always partisan)
- How much of the variance in voting does the partisan dimension explain?
- Is there a meaningful second dimension? What does it represent?
- Where does each legislator sit on the ideological spectrum?
- Which legislators are in the ideological "middle" between parties?
- How well does a low-dimensional model explain voting behavior?

## Mathematical Foundation

### Standard PCA

Given a vote matrix $\mathbf{X}$ with $n$ legislators (rows) and $m$ roll calls (columns), coded as 1 (Yea) and 0 (Nay) with missing values imputed:

1. **Center the matrix**: $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{X}}$ (subtract column means)
2. **Compute covariance matrix**: $\mathbf{C} = \frac{1}{n-1} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}}$
3. **Eigendecomposition**: $\mathbf{C} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$
4. **Project**: $\mathbf{Z} = \tilde{\mathbf{X}} \mathbf{V}_k$ where $\mathbf{V}_k$ is the first $k$ eigenvectors

The projection $\mathbf{Z}$ gives each legislator a score on each of the $k$ principal components.

### Explained Variance

$$\text{Explained Variance Ratio}_k = \frac{\lambda_k}{\sum_{i=1}^{m} \lambda_i}$$

If PC1 explains 40% of variance, it means that 40% of all the differences in how legislators vote can be captured by a single number (their PC1 score).

### Relationship to SVD and NOMINATE

PCA via SVD: $\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$. The left singular vectors $\mathbf{U}$ (scaled by $\mathbf{\Sigma}$) give legislator scores; the right singular vectors $\mathbf{V}$ give bill/roll-call loadings.

NOMINATE is effectively PCA with a nonlinear (logistic) link function and a specific spatial utility model. For most purposes, linear PCA gives nearly identical first-dimension rankings.

## Python Implementation

### Basic PCA

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def compute_vote_pca(
    vote_matrix: pd.DataFrame,
    n_components: int = 5,
    impute_method: str = "row_mean",
) -> tuple[pd.DataFrame, PCA]:
    """Run PCA on the vote matrix.

    Args:
        vote_matrix: Binary vote matrix (legislators x roll calls).
        n_components: Number of components to extract.
        impute_method: How to handle NaN ('row_mean', 'column_mean', 'zero').

    Returns:
        scores: DataFrame of PC scores per legislator.
        pca: Fitted PCA object (for explained variance, loadings, etc.).
    """
    # Impute missing values
    if impute_method == "row_mean":
        filled = vote_matrix.T.fillna(vote_matrix.mean(axis=1)).T
    elif impute_method == "column_mean":
        filled = vote_matrix.fillna(vote_matrix.mean(axis=0))
    elif impute_method == "zero":
        filled = vote_matrix.fillna(0)
    else:
        raise ValueError(f"Unknown impute method: {impute_method}")

    # Standardize (center and scale)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(filled)

    # Fit PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    scores_df = pd.DataFrame(
        scores,
        index=vote_matrix.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    return scores_df, pca
```

### Scree Plot (How Many Dimensions?)

```python
def plot_scree(pca: PCA, save_path: str | None = None):
    """Plot explained variance by component."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Individual explained variance
    axes[0].bar(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        edgecolor="black",
    )
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Scree Plot")

    # Cumulative explained variance
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(
        range(1, len(cumulative) + 1),
        cumulative,
        "bo-",
    )
    axes[1].axhline(0.9, color="red", linestyle="--", label="90% threshold")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title("Cumulative Variance Explained")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

### Ideological Map (PC1 vs PC2)

```python
def plot_ideological_map(
    scores: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    chamber: str | None = None,
    label_outliers: bool = True,
    save_path: str | None = None,
):
    """Plot legislators in PC1-PC2 space, colored by party."""
    meta = legislator_meta.loc[scores.index]
    if chamber:
        mask = meta["chamber"] == chamber
        scores = scores[mask]
        meta = meta[mask]

    colors = meta["party"].map({"Republican": "#E81B23", "Democrat": "#0015BC"})

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        scores["PC1"],
        scores["PC2"],
        c=colors,
        s=60,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # Label outliers (extreme PC1 or PC2 values)
    if label_outliers:
        for pc in ["PC1", "PC2"]:
            extremes = scores[pc].abs().nlargest(5).index
            for slug in extremes:
                last_name = slug.split("_")[1].title()
                ax.annotate(
                    last_name,
                    (scores.loc[slug, "PC1"], scores.loc[slug, "PC2"]),
                    fontsize=7,
                    ha="left",
                    va="bottom",
                )

    ax.axhline(0, color="gray", linestyle="-", alpha=0.2)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.2)
    ax.set_xlabel("PC1 (primary ideological dimension)")
    ax.set_ylabel("PC2 (secondary dimension)")
    title = f"Ideological Map — {chamber or 'All'}"
    ax.set_title(title)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E81B23", label="Republican"),
        Patch(facecolor="#0015BC", label="Democrat"),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

### Bill Loadings Analysis (What Does Each PC Represent?)

```python
def analyze_pc_loadings(
    pca: PCA,
    vote_matrix: pd.DataFrame,
    rollcalls: pd.DataFrame,
    pc: int = 1,
    top_n: int = 15,
) -> pd.DataFrame:
    """Find the roll calls with highest loadings on a given PC.

    These are the votes that most strongly define the dimension.
    """
    loadings = pd.Series(
        pca.components_[pc - 1],
        index=vote_matrix.columns,
        name=f"PC{pc}_loading",
    )

    # Get roll call metadata
    loading_df = loadings.reset_index()
    loading_df.columns = ["vote_id", f"PC{pc}_loading"]
    loading_df = loading_df.merge(
        rollcalls[["vote_id", "bill_number", "bill_title", "motion", "vote_type"]],
        on="vote_id",
        how="left",
    )

    # Top positive and negative loadings
    top_positive = loading_df.nlargest(top_n, f"PC{pc}_loading")
    top_negative = loading_df.nsmallest(top_n, f"PC{pc}_loading")

    return pd.concat([top_positive, top_negative])
```

### Sign Correction

PCA components have arbitrary sign. By convention, orient PC1 so that positive values correspond to the conservative/Republican direction:

```python
def orient_pca_scores(
    scores: pd.DataFrame,
    legislator_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Orient PC1 so positive = Republican direction."""
    meta = legislator_meta.loc[scores.index]
    rep_mean = scores.loc[meta["party"] == "Republican", "PC1"].mean()
    dem_mean = scores.loc[meta["party"] == "Democrat", "PC1"].mean()

    if rep_mean < dem_mean:
        # Flip sign so Republicans are positive
        scores["PC1"] = -scores["PC1"]

    return scores
```

## Interpretation Guide

### PC1 (First Principal Component)

- **Almost always maps to the partisan dimension.** Legislators from different parties will separate cleanly on PC1.
- **Explained variance of 30-50%** for PC1 is typical for a partisan legislature. Higher values (>50%) indicate extreme partisanship.
- **Overlap region**: Legislators from different parties whose PC1 scores overlap are the moderates/swing voters. In Kansas, look for moderate Republicans whose PC1 scores approach the Democratic range.
- **PC1 score magnitude**: Farther from zero = more ideologically extreme. Legislators near zero are centrists.

### PC2 (Second Principal Component)

- **Interpretation requires examining bill loadings.** Look at which roll calls have the highest PC2 loadings to understand what this dimension captures.
- **Common second dimensions in state legislatures:**
  - Urban vs. rural (policy preferences related to agriculture, infrastructure, education)
  - Establishment vs. insurgent (support for leadership vs. backbench rebellion)
  - Fiscal vs. social (some legislators are fiscally conservative but socially moderate, or vice versa)
- **Explained variance of 5-15%** for PC2 is typical. If PC2 explains less than 5%, the legislature is essentially one-dimensional.

### Scree Plot Reading

- **Sharp elbow after PC1**: The legislature is effectively one-dimensional (partisan).
- **Gradual decline**: Multiple dimensions of conflict coexist.
- **90% cumulative variance**: The number of components needed to reach 90% tells you the effective dimensionality.

## Advantages and Limitations

### Advantages
- Pure Python implementation — no R dependency
- Fast: PCA on a 170x500 matrix takes milliseconds
- Produces results highly correlated with NOMINATE (the gold standard)
- Easy to visualize and interpret
- Natural starting point before more complex methods

### Limitations
- Linear model: assumes voting is a linear function of ideology. NOMINATE uses a nonlinear (logistic) link function that better matches the binary nature of votes.
- Requires imputation for missing data (absences). The imputation method can affect results.
- Arbitrary sign and rotation of components.
- No uncertainty quantification: PCA gives a point estimate, not a posterior distribution. See `15_BAY_irt_ideal_points` for the Bayesian version with full uncertainty.

## Kansas-Specific Considerations

- **Analyze chambers separately.** Cross-chamber PCA is problematic because House and Senate members vote on different roll calls, creating a block-diagonal missing data pattern.
- **Filter lopsided votes.** Near-unanimous votes add noise, not signal. Filter to votes where the minority side is ≥2.5% (NOMINATE convention) or ≥10% (more aggressive).
- **PC2 may reveal the moderate-conservative Republican split.** In Kansas, the second dimension often captures the factional divide within the Republican majority.
- **Compare with Bayesian IRT.** After running PCA, compare the PC1 scores with the Bayesian ideal point estimates from `15_BAY_irt_ideal_points`. High correlation validates both methods.

## Feasibility Assessment

- **Data size**: 170 legislators x ~400 contested roll calls = trivially small for PCA
- **Compute time**: Sub-second
- **Libraries**: sklearn (likely already installed)
- **Difficulty**: Straightforward. The main challenge is interpreting PC2 and higher components.

## Key References

- Heckman, James J., and James M. Snyder. "Linear Probability Models of the Demand for Attributes with an Empirical Application to Estimating the Preferences of Legislators." *RAND Journal of Economics* 28, 1997.
- Rosenthal, Howard, and Erik Voeten. "Measuring Legal Systems." *Journal of Comparative Economics* 35(4), 2007.
- Rosenthal, Howard, and Erik Voeten. "Analyzing Roll Calls with Ideal Points, with an Application to the European Parliament." *Social Science* 7(1), 2018. https://www.mdpi.com/2076-0760/7/1/12
- Poole, Keith T. "Recovering a Basic Space from a Set of Issue Scales." *American Journal of Political Science* 42(3), 1998.
