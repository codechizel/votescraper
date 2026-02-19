# Correspondence Analysis and Multiple Correspondence Analysis

**Category:** Dimensionality Reduction / Ideal Point Estimation
**Prerequisites:** `01_DATA_vote_matrix_construction`
**Complexity:** Medium
**Related:** `09_DIM_principal_component_analysis`

## What It Measures

Correspondence Analysis (CA) and Multiple Correspondence Analysis (MCA) are dimensionality reduction techniques designed specifically for categorical data. Unlike PCA (which assumes continuous variables), CA/MCA respects the categorical nature of votes — Yea, Nay, Present and Passing, Absent and Not Voting are qualitative categories, not points on a continuous scale.

The key advantage: CA simultaneously maps both legislators AND roll calls into the same low-dimensional space. This means you can see not just which legislators are similar, but which *bills* are similar, and which legislators are associated with which bills — all in a single plot.

## Questions It Answers

- Which legislators and which bills cluster together in ideological space?
- Does the categorical structure of votes (including abstentions) reveal different patterns than binary PCA?
- Are there bills that bridge the partisan divide (positioned between party clusters)?
- Do "Present and Passing" votes form a meaningful cluster distinct from Yea and Nay?
- What is the optimal low-dimensional representation of the full categorical vote data?

## Mathematical Foundation

### Simple Correspondence Analysis (CA)

Given a two-way contingency table $\mathbf{N}$ (legislators x vote categories aggregated across bills):

1. Compute the correspondence matrix: $\mathbf{P} = \frac{\mathbf{N}}{\sum \mathbf{N}}$
2. Row and column profiles: $r_i = \sum_j P_{ij}$, $c_j = \sum_i P_{ij}$
3. Standardized residuals: $S_{ij} = \frac{P_{ij} - r_i c_j}{\sqrt{r_i c_j}}$
4. SVD of $\mathbf{S}$: $\mathbf{S} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$
5. Row (legislator) coordinates: $\mathbf{F} = \mathbf{D}_r^{-1/2} \mathbf{U} \mathbf{\Sigma}$
6. Column (category) coordinates: $\mathbf{G} = \mathbf{D}_c^{-1/2} \mathbf{V} \mathbf{\Sigma}$

The total inertia (analogous to total variance) is the chi-squared statistic divided by $n$.

### Multiple Correspondence Analysis (MCA)

MCA generalizes CA to multi-way categorical data. For our vote matrix where each roll call is a categorical variable with levels {Yea, Nay, Absent, etc.}:

1. Construct the indicator matrix $\mathbf{Z}$: for each roll call, create dummy columns for each vote category
2. Compute the Burt matrix $\mathbf{B} = \mathbf{Z}^T \mathbf{Z}$
3. Apply CA to $\mathbf{B}$ (or equivalently, apply PCA to the standardized indicator matrix)

MCA maps both legislators and vote categories into the same space, allowing joint visualization.

### Chi-Square Distance

CA/MCA use chi-square distance rather than Euclidean distance. Chi-square distance weights each variable inversely by its frequency, so rare categories (like "Present and Passing") are given proportionally more weight. This is important for legislative data where category frequencies are highly unbalanced.

## Python Implementation

### Using the `prince` Library

```python
import pandas as pd
import numpy as np
import prince
import matplotlib.pyplot as plt

def run_mca_on_votes(
    vote_matrix_categorical: pd.DataFrame,
    n_components: int = 5,
) -> prince.MCA:
    """Run Multiple Correspondence Analysis on categorical vote matrix.

    Args:
        vote_matrix_categorical: Vote matrix with categorical values
            ('Yea', 'Nay', 'Present and Passing', 'Absent and Not Voting').
        n_components: Number of dimensions to extract.

    Returns:
        Fitted MCA object.
    """
    # Fill NaN with a category (MCA treats all categories equally)
    filled = vote_matrix_categorical.fillna("Absent")

    # Convert to string type (required by prince)
    filled = filled.astype(str)

    mca = prince.MCA(
        n_components=n_components,
        n_iter=3,  # Number of iterations for randomized SVD
        random_state=42,
    )
    mca = mca.fit(filled)

    return mca


def build_categorical_vote_matrix(
    votes_path: str,
    legislators_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build vote matrix preserving all vote categories."""
    votes = pd.read_csv(votes_path)
    legislators = pd.read_csv(legislators_path)

    vote_matrix = votes.pivot_table(
        index="legislator_slug",
        columns="vote_id",
        values="vote",
        aggfunc="first",
    )

    meta = legislators.set_index("slug")[["full_name", "chamber", "party", "district"]]
    return vote_matrix, meta
```

### Extracting and Visualizing Results

```python
def plot_mca_biplot(
    mca: prince.MCA,
    vote_matrix: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    dim_x: int = 0,
    dim_y: int = 1,
    save_path: str | None = None,
):
    """Plot MCA biplot showing legislators and vote categories."""
    # Row (legislator) coordinates
    row_coords = mca.row_coordinates(vote_matrix.fillna("Absent").astype(str))
    meta = legislator_meta.loc[row_coords.index]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot legislators colored by party
    colors = meta["party"].map({"Republican": "#E81B23", "Democrat": "#0015BC"})
    ax.scatter(
        row_coords.iloc[:, dim_x],
        row_coords.iloc[:, dim_y],
        c=colors,
        s=40,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.3,
        zorder=5,
    )

    # Plot column (category) coordinates
    col_coords = mca.column_coordinates(vote_matrix.fillna("Absent").astype(str))

    # Color by category type
    category_colors = {
        "Yea": "green",
        "Nay": "orange",
        "Present and Passing": "purple",
        "Absent and Not Voting": "gray",
        "Absent": "gray",
    }

    for idx, row in col_coords.iterrows():
        # idx format is "vote_id_CategoryName"
        category = str(idx).split("_")[-1] if "_" in str(idx) else str(idx)
        color = category_colors.get(category, "gray")
        ax.scatter(
            row.iloc[dim_x],
            row.iloc[dim_y],
            c=color,
            marker="x",
            s=10,
            alpha=0.3,
        )

    ax.set_xlabel(f"Dimension {dim_x + 1} ({mca.percentage_of_variance_[dim_x]:.1f}%)")
    ax.set_ylabel(f"Dimension {dim_y + 1} ({mca.percentage_of_variance_[dim_y]:.1f}%)")
    ax.set_title("MCA Biplot: Legislators and Vote Categories")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#E81B23", label="Republican"),
        Patch(facecolor="#0015BC", label="Democrat"),
        Line2D([0], [0], marker="x", color="green", label="Yea", linestyle="None"),
        Line2D([0], [0], marker="x", color="orange", label="Nay", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
```

### Inertia Plot (MCA equivalent of Scree Plot)

```python
def plot_inertia(mca: prince.MCA, save_path: str | None = None):
    """Plot explained inertia by dimension."""
    fig, ax = plt.subplots(figsize=(10, 5))

    dims = range(1, len(mca.percentage_of_variance_) + 1)
    ax.bar(dims, mca.percentage_of_variance_, edgecolor="black")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Percentage of Inertia (%)")
    ax.set_title("MCA Inertia by Dimension")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

## Interpretation Guide

### Reading the Biplot

In an MCA biplot:
- **Legislators (dots)** that are close together vote similarly.
- **Category points (x markers)** show where each response category falls in the space.
- **Proximity of a legislator to a category** indicates that the legislator frequently chose that category.
- **The origin (0,0)** represents the "average" profile — legislators near the origin have typical voting patterns.

### Explained Inertia

MCA's explained inertia percentages are typically much lower than PCA's explained variance percentages. An MCA first dimension explaining 10% of inertia might capture as much structure as a PCA first component explaining 40% of variance. This is a known property of MCA, not a sign of poor fit. Use Benzécri's correction or Greenacre's adjustment for more interpretable percentages:

$$\tilde{\lambda}_k = \left(\frac{K}{K-1}\right)^2 \left(\lambda_k - \frac{1}{K}\right)^2 \quad \text{for } \lambda_k > \frac{1}{K}$$

Where $K$ is the number of variables (roll calls) and $\lambda_k$ is the $k$-th eigenvalue.

### CA vs MCA vs PCA: When to Use Which

| Method | Data Type | Missing Data | Joint Mapping | Best For |
|--------|-----------|-------------|---------------|----------|
| PCA | Continuous (binary imputed) | Requires imputation | No (legislators only) | Quick ideal point proxy |
| CA | Two-way contingency table | Excludes naturally | Yes (row + column) | Aggregated vote profiles |
| MCA | Multi-way categorical | Treats as a category | Yes (all in one space) | Full categorical structure |

## Advantages Over PCA

1. **Handles categorical data natively.** No need to pretend Yea/Nay are continuous.
2. **Handles "Present and Passing" and "Absent" as first-class categories** rather than as missing data to impute.
3. **Joint mapping** of legislators and bills in the same space.
4. **Chi-square distance** gives appropriate weight to rare categories.

## Limitations

1. **Lower interpretability** of explained inertia percentages without correction.
2. **Computationally more expensive** than PCA for large datasets (but still fast for our 170x865 matrix).
3. **The prince library** is less mature than sklearn's PCA implementation.
4. **Biplots get cluttered** with hundreds of roll calls. May need to show only a subset of category points.

## Kansas-Specific Considerations

- **"Present and Passing" votes (22 instances) are analytically interesting** in MCA because they get their own position in the space. In PCA, they're imputed away.
- **Run per-chamber** for the same reasons as PCA.
- **The biplot can reveal bill clusters** — groups of bills that are voted on similarly. This can suggest latent policy dimensions.
- **Consider filtering to contested votes** before MCA. On near-unanimous votes, the Yea and Nay categories have extreme frequency imbalance that can dominate the inertia.

## Feasibility Assessment

- **Data size**: 170 x 865 categorical matrix = fast for MCA
- **Compute time**: 1-5 seconds
- **Libraries**: `prince` (pip install prince)
- **Difficulty**: Moderate. MCA interpretation requires understanding of chi-square distance and inertia.

## Key References

- Greenacre, Michael. *Correspondence Analysis in Practice*. 3rd ed. CRC Press, 2017. (The definitive textbook.)
- Benzécri, Jean-Paul. *L'Analyse des Données*. Dunod, 1973. (The original formulation of CA.)
- Le Roux, Brigitte, and Henry Rouanet. *Multiple Correspondence Analysis*. SAGE, 2010.
- `prince` library documentation: https://github.com/MaxHalford/prince
