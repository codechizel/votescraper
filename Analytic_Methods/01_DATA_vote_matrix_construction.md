# Vote Matrix Construction

**Category:** Data Preparation
**Prerequisite for:** Every other analysis in this directory

## What This Is

The vote matrix is the foundational data structure for legislative vote analysis. It transforms the long-format votes CSV (one row per legislator per roll call) into a wide-format matrix where rows are legislators, columns are roll calls, and cells contain vote codes. Nearly every method documented in this directory operates on this matrix or a derivative of it.

## The Question It Answers

How do we reshape our raw scraped data into a form suitable for matrix-based analysis (PCA, IRT, clustering, networks, etc.)?

## Input Data

From the scraper's output:
- `ks_{session}_votes.csv` — columns: `session`, `bill_number`, `bill_title`, `vote_id`, `vote_datetime`, `vote_date`, `chamber`, `motion`, `legislator_name`, `legislator_slug`, `vote`
- `ks_{session}_legislators.csv` — columns: `name`, `full_name`, `slug`, `chamber`, `party`, `district`, `member_url`

## Vote Encoding Schemes

Different analyses require different encodings. Build the matrix once, then recode as needed.

### Binary Encoding (for PCA, IRT, NOMINATE, clustering)

| Vote Category | Code | Rationale |
|---------------|------|-----------|
| Yea | 1 | Affirmative vote |
| Nay | 0 | Negative vote |
| Present and Passing | NaN | Deliberate abstention — not a position signal |
| Absent and Not Voting | NaN | Missing data — legislator not present |
| Not Voting | NaN | Missing data |

### Ternary Encoding (for some clustering methods)

| Vote Category | Code |
|---------------|------|
| Yea | +1 |
| Nay | -1 |
| Present and Passing | 0 |
| Absent and Not Voting | NaN |
| Not Voting | NaN |

### Full Categorical Encoding (for Correspondence Analysis / MCA)

Keep the original five categories as categorical labels. MCA is designed for this.

## Python Implementation

```python
import pandas as pd
import numpy as np

def build_vote_matrix(
    votes_path: str,
    legislators_path: str,
    encoding: str = "binary",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the legislator x roll-call vote matrix.

    Args:
        votes_path: Path to votes CSV.
        legislators_path: Path to legislators CSV.
        encoding: One of "binary", "ternary", "categorical".

    Returns:
        vote_matrix: DataFrame with legislator_slug as index, vote_id as columns.
        legislator_meta: DataFrame with party, chamber, district per legislator.
    """
    votes = pd.read_csv(votes_path)
    legislators = pd.read_csv(legislators_path)

    # Map vote categories to numeric codes
    if encoding == "binary":
        vote_map = {"Yea": 1, "Nay": 0}
    elif encoding == "ternary":
        vote_map = {"Yea": 1, "Nay": -1, "Present and Passing": 0}
    elif encoding == "categorical":
        vote_map = None  # Keep as-is
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    if vote_map is not None:
        votes["vote_code"] = votes["vote"].map(vote_map)
    else:
        votes["vote_code"] = votes["vote"]

    # Pivot to wide format
    vote_matrix = votes.pivot_table(
        index="legislator_slug",
        columns="vote_id",
        values="vote_code",
        aggfunc="first",  # Should be one vote per legislator per roll call
    )

    # Merge legislator metadata
    legislator_meta = legislators.set_index("slug")[["full_name", "chamber", "party", "district"]]

    # Align indices
    vote_matrix = vote_matrix.loc[vote_matrix.index.isin(legislator_meta.index)]
    legislator_meta = legislator_meta.loc[legislator_meta.index.isin(vote_matrix.index)]

    return vote_matrix, legislator_meta
```

## Filtering Decisions

### Removing Near-Unanimous Votes

Near-unanimous roll calls provide no ideological information — everyone voted the same way. The standard practice (from NOMINATE methodology) is to remove votes where the minority side is less than 2.5% of votes cast.

```python
def filter_lopsided_votes(
    vote_matrix: pd.DataFrame,
    min_minority_pct: float = 0.025,
) -> pd.DataFrame:
    """Remove roll calls where the minority side is too small."""
    yea_pct = vote_matrix.mean(axis=0)  # NaN-aware: only counts 0s and 1s
    minority_pct = np.minimum(yea_pct, 1 - yea_pct)
    contested = minority_pct >= min_minority_pct
    return vote_matrix.loc[:, contested]
```

**Impact on Kansas data:** With ~82% of all votes being Yea, a significant fraction of roll calls are near-unanimous. Filtering at 2.5% typically retains 40-60% of roll calls. At 10%, you retain only the genuinely contested votes.

### Removing Low-Participation Legislators

Legislators who voted on very few roll calls produce unreliable estimates. Standard practice is to exclude legislators with fewer than 20 non-missing votes.

```python
def filter_low_participation(
    vote_matrix: pd.DataFrame,
    min_votes: int = 20,
) -> pd.DataFrame:
    """Remove legislators with too few votes."""
    vote_counts = vote_matrix.notna().sum(axis=1)
    active = vote_counts >= min_votes
    return vote_matrix.loc[active]
```

### Chamber Separation

Many analyses should be run separately per chamber because House and Senate members vote on different roll calls.

```python
def split_by_chamber(
    vote_matrix: pd.DataFrame,
    legislator_meta: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split vote matrix by chamber."""
    chambers = {}
    for chamber in ["House", "Senate"]:
        slugs = legislator_meta[legislator_meta["chamber"] == chamber].index
        chamber_matrix = vote_matrix.loc[vote_matrix.index.isin(slugs)]
        # Drop columns (roll calls) with all NaN (other chamber's votes)
        chamber_matrix = chamber_matrix.dropna(axis=1, how="all")
        chambers[chamber] = chamber_matrix
    return chambers
```

## Handling Missing Data

Missing votes (NaN) arise from absences, abstentions, or legislators joining/leaving mid-session. Different methods handle missing data differently:

| Method | Missing Data Strategy |
|--------|----------------------|
| PCA | Impute with row mean (legislator's average vote) or column mean (roll call average) |
| Bayesian IRT | Naturally handled — simply omit from likelihood |
| NOMINATE | Built-in handling in the R package |
| Clustering | Compute pairwise distances using only shared non-missing votes |
| Network | Compute agreement using only votes where both legislators were present |
| Rice/Unity indices | Use only present votes in the denominator |

### Imputation for PCA

```python
def impute_vote_matrix(vote_matrix: pd.DataFrame, method: str = "row_mean") -> pd.DataFrame:
    """Impute missing votes for methods that require complete data."""
    if method == "row_mean":
        return vote_matrix.T.fillna(vote_matrix.mean(axis=1)).T
    elif method == "column_mean":
        return vote_matrix.fillna(vote_matrix.mean(axis=0))
    elif method == "zero":
        return vote_matrix.fillna(0)
    else:
        raise ValueError(f"Unknown imputation method: {method}")
```

## Output Artifacts

The vote matrix construction step should produce:

1. **Full vote matrix** — `vote_matrix_full.parquet` (all legislators, all roll calls, NaN preserved)
2. **Filtered vote matrix** — `vote_matrix_contested.parquet` (lopsided votes and low-participation legislators removed)
3. **Chamber-specific matrices** — `vote_matrix_house.parquet`, `vote_matrix_senate.parquet`
4. **Legislator metadata** — `legislator_meta.parquet` (party, chamber, district aligned to matrix index)

Using Parquet format preserves dtypes (including NaN) and is much faster than CSV for matrix data.

## Diagnostic Checks

After building the matrix, verify:

1. **Dimensions**: Expect ~170 rows x ~865 columns for the full matrix
2. **Sparsity**: What percentage of cells are NaN? (Expect ~5-10% from absences, plus ~50% from cross-chamber votes if not split)
3. **Balance**: After filtering lopsided votes, what's the Yea/Nay distribution? (Should be closer to 50/50)
4. **Party verification**: Spot-check that known Democrats and Republicans cluster as expected with a simple correlation check

```python
# Quick sanity check: average vote by party
meta = legislator_meta.loc[vote_matrix.index]
party_means = vote_matrix.groupby(meta["party"]).mean()
# Democrats and Republicans should differ on contested votes
```

## Kansas-Specific Notes

- The legislature has two distinct chambers that rarely vote on the same roll calls. Cross-chamber analysis requires careful handling — typically analyze chambers separately, then compare results.
- With ~82% Yea rate overall, aggressive filtering is needed to get at the interesting contested votes.
- The `legislator_slug` column is the stable identifier linking votes to legislators. It encodes chamber (`sen_` vs `rep_`), making chamber separation straightforward.
- Some bills receive multiple roll calls (Final Action, Conference Committee, Veto Override). The `vote_id` is unique per roll call, not per bill.
