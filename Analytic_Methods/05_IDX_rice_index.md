# Rice Index (Party Cohesion)

**Category:** Index-Based Measures
**Prerequisites:** `01_DATA_vote_matrix_construction`
**Complexity:** Low
**Related:** `06_IDX_party_unity_scores`

## What It Measures

The Rice Index measures how cohesively a party votes on a single roll call. Named after Stuart Rice (1925), it is the oldest and most widely used measure of party cohesion in political science. A Rice Index of 1.0 means the party voted unanimously; 0.0 means the party was evenly split.

## Questions It Answers

- How cohesive is each party on a given vote?
- Which votes caused the most internal party conflict?
- Is one party more cohesive than the other overall?
- How does cohesion vary by vote type (Final Action vs. Conference Committee vs. Veto Override)?
- On which bills did the majority party fracture?

## Mathematical Foundation

For party $p$ on roll call $j$:

$$\text{Rice}_j^p = \left| \frac{Y_j^p - N_j^p}{Y_j^p + N_j^p} \right|$$

Where:
- $Y_j^p$ = number of party $p$ members voting Yea on roll call $j$
- $N_j^p$ = number of party $p$ members voting Nay on roll call $j$

**Properties:**
- Range: [0, 1]
- Rice = 1: unanimous party vote (all Yea or all Nay)
- Rice = 0: party evenly split (50% Yea, 50% Nay)
- Absent members are excluded from the denominator

**Session-level average:**

$$\overline{\text{Rice}}^p = \frac{1}{J} \sum_{j=1}^{J} \text{Rice}_j^p$$

Where $J$ is the total number of roll calls.

### Relationship to Other Measures

The Rice Index is related to party unity scores:
- Rice measures the *party's* cohesion on a vote
- Party Unity Score measures the *legislator's* loyalty across votes

If a party's average Rice Index is 0.85, it means that on average, 92.5% of the party votes together (since Rice = |2p - 1| where p is the majority fraction).

## Python Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_rice_index(
    votes: pd.DataFrame,
    legislators: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Rice Index for each party on each roll call.

    Args:
        votes: Individual votes DataFrame with legislator_slug, vote_id, vote columns.
        legislators: Legislators DataFrame with slug, party columns.

    Returns:
        DataFrame with columns: vote_id, party, yea_count, nay_count, rice_index.
    """
    # Merge votes with party
    merged = votes.merge(
        legislators[["slug", "party"]],
        left_on="legislator_slug",
        right_on="slug",
        how="left",
    )

    # Keep only substantive votes
    substantive = merged[merged["vote"].isin(["Yea", "Nay"])]

    # Count Yea and Nay per party per roll call
    party_counts = (
        substantive.groupby(["vote_id", "party", "vote"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Compute Rice Index
    party_counts["total"] = party_counts.get("Yea", 0) + party_counts.get("Nay", 0)
    party_counts["rice_index"] = np.where(
        party_counts["total"] > 0,
        abs(party_counts.get("Yea", 0) - party_counts.get("Nay", 0)) / party_counts["total"],
        np.nan,
    )

    return party_counts[["vote_id", "party", "Yea", "Nay", "total", "rice_index"]]


def rice_summary(rice_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize Rice Index by party."""
    return rice_df.groupby("party")["rice_index"].agg(["mean", "median", "std", "min", "count"])
```

### Visualization: Rice Index Distribution

```python
def plot_rice_distribution(rice_df: pd.DataFrame, save_path: str | None = None):
    """Plot distribution of Rice Index by party."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for i, party in enumerate(["Republican", "Democrat"]):
        party_data = rice_df[rice_df["party"] == party]["rice_index"].dropna()
        color = "#E81B23" if party == "Republican" else "#0015BC"

        axes[i].hist(party_data, bins=30, color=color, edgecolor="black", alpha=0.7)
        axes[i].axvline(party_data.mean(), color="black", linestyle="--",
                       label=f"Mean: {party_data.mean():.3f}")
        axes[i].set_xlabel("Rice Index")
        axes[i].set_ylabel("Number of Roll Calls")
        axes[i].set_title(f"{party} Party Cohesion")
        axes[i].legend()
        axes[i].set_xlim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

### Identifying Low-Cohesion Votes

```python
def find_fractured_votes(
    rice_df: pd.DataFrame,
    rollcalls: pd.DataFrame,
    party: str = "Republican",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Find roll calls where a party was most internally divided."""
    low_cohesion = rice_df[
        (rice_df["party"] == party) & (rice_df["rice_index"] < threshold)
    ]
    return low_cohesion.merge(
        rollcalls[["vote_id", "bill_number", "bill_title", "motion", "vote_type", "result"]],
        on="vote_id",
    ).sort_values("rice_index")
```

### Rice Index Over Time

```python
def plot_rice_over_time(
    rice_df: pd.DataFrame,
    rollcalls: pd.DataFrame,
    save_path: str | None = None,
):
    """Plot Rice Index over the session timeline."""
    rice_dated = rice_df.merge(
        rollcalls[["vote_id", "vote_datetime"]],
        on="vote_id",
    )
    rice_dated["date"] = pd.to_datetime(rice_dated["vote_datetime"]).dt.date

    fig, ax = plt.subplots(figsize=(14, 5))

    for party in ["Republican", "Democrat"]:
        party_data = rice_dated[rice_dated["party"] == party]
        daily_rice = party_data.groupby("date")["rice_index"].mean()
        # 7-day rolling average for smoothing
        rolling = daily_rice.rolling(7, min_periods=1).mean()
        color = "#E81B23" if party == "Republican" else "#0015BC"
        ax.plot(rolling.index, rolling.values, color=color, label=party, alpha=0.8)

    ax.set_xlabel("Date")
    ax.set_ylabel("Rice Index (7-day rolling mean)")
    ax.set_title("Party Cohesion Over the Session")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

## Interpretation Guide

- **Rice > 0.9**: Party voted nearly unanimously. This is common for routine bills.
- **Rice 0.7-0.9**: Moderate cohesion. Some internal dissent but the party still has a clear majority position.
- **Rice 0.5-0.7**: Significant internal division. The party is split on this issue.
- **Rice < 0.5**: Major party fracture. The party is more internally divided than united.
- **Rice = 0**: Perfect 50/50 split. Extremely rare.

### What Typical Values Look Like

| Legislature Type | Avg Rice Index |
|-----------------|---------------|
| US Congress (modern, strong partisanship) | 0.85-0.95 |
| State legislatures (moderate partisanship) | 0.70-0.90 |
| European parliaments (party discipline) | 0.90-0.99 |
| Non-partisan or weak-party bodies | 0.30-0.60 |

### Caution: Base Rate Inflation

When one party has a supermajority and most bills pass, the Rice Index for the majority party is mechanically inflated — they're voting "Yea" together on everything, including non-controversial bills. Always compare Rice on *contested* votes (where the minority had at least some opposing votes) vs. all votes.

## Kansas-Specific Considerations

- **Republican Rice will be high** because the supermajority means they win nearly every vote. The interesting analysis is identifying the roll calls where Republican Rice drops below ~0.7.
- **Democrat Rice may be lower** not because Democrats are less cohesive, but because with fewer members (~28% of seats), individual absences or crossovers have a larger effect on the index.
- **Veto overrides** are particularly interesting — they require 2/3 majority, so the governor's vetoes likely targeted bills where Democratic support was needed, producing lower Rice for both parties.
- **Compare Rice by vote type**: Final Action vs. Conference Committee vs. Veto Override. Conference Committee votes (post-negotiation) may show higher cohesion than initial Final Action votes.

## Key References

- Rice, Stuart A. "The Behavior of Legislative Groups: A Method of Measurement." *Political Science Quarterly* 40(1), 1925.
- Desposato, Scott W. "Parties for Rent? Ambition, Ideology, and Party Switching in Brazil's Chamber of Deputies." *American Journal of Political Science* 50(1), 2006.
- Rice Index entry, Wikipedia: https://en.wikipedia.org/wiki/Rice_index
