# Party Unity Scores

**Category:** Index-Based Measures
**Prerequisites:** `01_DATA_vote_matrix_construction`, `05_IDX_rice_index`
**Complexity:** Low
**Related:** `08_IDX_loyalty_and_maverick_scores`, `14_BAY_beta_binomial_party_loyalty`

## What It Measures

Party Unity Scores measure how often an individual legislator votes with the majority of their own party on "party votes" — roll calls where the two parties' majorities opposed each other. Unlike the Rice Index (which measures the party's cohesion), the Unity Score measures the legislator's loyalty to their party.

This is the measure that Congressional Quarterly (CQ) and similar outlets publish when they rank legislators as "most loyal" or "most independent."

## Questions It Answers

- Which legislators are the most loyal to their party?
- Which legislators are the most independent or "maverick"?
- Is party loyalty stronger in the House or Senate?
- Do certain policy areas produce more party defection?
- How does Kansas compare to national norms for party unity?

## Mathematical Foundation

### Step 1: Identify Party Votes

A roll call $j$ is a "party vote" if the majority of Republicans and the majority of Democrats voted on opposite sides:

$$\text{Party Vote}_j = \begin{cases} \text{True} & \text{if } \text{Rep majority} \neq \text{Dem majority} \\ \text{False} & \text{otherwise} \end{cases}$$

Where "Rep majority" is Yea if >50% of Republicans voted Yea, and Nay otherwise (similarly for Democrats).

### Step 2: Compute Per-Legislator Unity Score

For legislator $i$ with party $p$, across party votes where $i$ was present:

$$\text{Unity}_i = \frac{|\{j : j \text{ is a party vote, } i \text{ voted with } p\text{'s majority}\}|}{|\{j : j \text{ is a party vote, } i \text{ voted Yea or Nay}\}|}$$

**Properties:**
- Range: [0, 1]
- Unity = 1: Never broke with party on party votes
- Unity = 0.5: Voted with party exactly half the time (coin flip)
- Unity = 0: Always voted against party (switched parties in practice)

## Python Implementation

```python
import pandas as pd
import numpy as np

def identify_party_votes(
    votes: pd.DataFrame,
    legislators: pd.DataFrame,
) -> pd.Series:
    """Identify roll calls where party majorities opposed each other.

    Returns:
        Series indexed by vote_id, True if it's a party vote.
    """
    merged = votes.merge(
        legislators[["slug", "party"]],
        left_on="legislator_slug",
        right_on="slug",
    )
    substantive = merged[merged["vote"].isin(["Yea", "Nay"])]

    party_votes = {}
    for vote_id, group in substantive.groupby("vote_id"):
        rep_votes = group[group["party"] == "Republican"]
        dem_votes = group[group["party"] == "Democrat"]

        if len(rep_votes) == 0 or len(dem_votes) == 0:
            party_votes[vote_id] = False
            continue

        rep_majority_yea = (rep_votes["vote"] == "Yea").mean() > 0.5
        dem_majority_yea = (dem_votes["vote"] == "Yea").mean() > 0.5

        # Party vote: majorities on opposite sides
        party_votes[vote_id] = rep_majority_yea != dem_majority_yea

    return pd.Series(party_votes, name="is_party_vote")


def compute_party_unity_scores(
    votes: pd.DataFrame,
    legislators: pd.DataFrame,
    party_vote_ids: pd.Series,
) -> pd.DataFrame:
    """Compute party unity score for each legislator.

    Returns:
        DataFrame with columns: legislator_slug, party, chamber,
        party_votes_present, votes_with_party, unity_score.
    """
    # Filter to party votes only
    pv_ids = party_vote_ids[party_vote_ids].index
    party_votes = votes[
        votes["vote_id"].isin(pv_ids) & votes["vote"].isin(["Yea", "Nay"])
    ]

    merged = party_votes.merge(
        legislators[["slug", "party", "chamber"]],
        left_on="legislator_slug",
        right_on="slug",
    )

    # Determine party majority position on each party vote
    party_positions = {}
    for vote_id, group in merged.groupby("vote_id"):
        for party_name in ["Republican", "Democrat"]:
            party_group = group[group["party"] == party_name]
            if len(party_group) > 0:
                majority_yea = (party_group["vote"] == "Yea").mean() > 0.5
                party_positions[(vote_id, party_name)] = "Yea" if majority_yea else "Nay"

    # Score each legislator
    results = []
    for slug, group in merged.groupby("legislator_slug"):
        party = group["party"].iloc[0]
        chamber = group["chamber"].iloc[0]

        votes_with_party = sum(
            1 for _, row in group.iterrows()
            if (row["vote_id"], party) in party_positions
            and row["vote"] == party_positions[(row["vote_id"], party)]
        )

        results.append({
            "legislator_slug": slug,
            "party": party,
            "chamber": chamber,
            "party_votes_present": len(group),
            "votes_with_party": votes_with_party,
            "unity_score": votes_with_party / len(group) if len(group) > 0 else np.nan,
        })

    return pd.DataFrame(results).sort_values("unity_score")
```

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_unity_scores(unity_df: pd.DataFrame, save_path: str | None = None):
    """Plot party unity scores as a strip/swarm plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, chamber in enumerate(["House", "Senate"]):
        chamber_data = unity_df[unity_df["chamber"] == chamber]

        colors = {"Republican": "#E81B23", "Democrat": "#0015BC"}
        sns.stripplot(
            data=chamber_data,
            x="party",
            y="unity_score",
            hue="party",
            palette=colors,
            ax=axes[i],
            jitter=0.3,
            alpha=0.6,
            size=6,
        )
        axes[i].set_title(f"{chamber} Party Unity Scores")
        axes[i].set_ylabel("Unity Score")
        axes[i].set_ylim(0, 1.05)
        axes[i].axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        axes[i].get_legend().remove()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)


def plot_unity_ranking(
    unity_df: pd.DataFrame,
    chamber: str,
    party: str,
    top_n: int = 20,
    save_path: str | None = None,
):
    """Bar chart of most/least loyal legislators within a party."""
    subset = unity_df[
        (unity_df["chamber"] == chamber) & (unity_df["party"] == party)
    ].sort_values("unity_score")

    # Show bottom and top N
    extremes = pd.concat([subset.head(top_n), subset.tail(top_n)]).drop_duplicates()

    fig, ax = plt.subplots(figsize=(10, max(8, len(extremes) * 0.3)))
    color = "#E81B23" if party == "Republican" else "#0015BC"

    ax.barh(
        range(len(extremes)),
        extremes["unity_score"],
        color=color,
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_yticks(range(len(extremes)))
    ax.set_yticklabels(extremes["legislator_slug"])
    ax.set_xlabel("Party Unity Score")
    ax.set_title(f"{chamber} {party} Party Unity Scores")
    ax.axvline(extremes["unity_score"].median(), color="gray", linestyle="--",
              label=f"Median: {extremes['unity_score'].median():.3f}")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

## Interpretation Guide

- **Unity > 0.95**: Extremely loyal. Votes with party on nearly every contested vote. Common for party leadership.
- **Unity 0.85-0.95**: Reliable party voter with occasional independence. This is the norm for most legislators.
- **Unity 0.70-0.85**: Moderate or swing voter. Worth investigating which issues cause defection.
- **Unity < 0.70**: Significant maverick. In a two-party system, scoring below 0.70 means voting with the opposing party nearly as often as your own.

### Historical Benchmarks (US Congress)

| Era | Average Unity Score |
|-----|-------------------|
| 1970s (low partisanship) | 0.65-0.75 |
| 1990s (rising partisanship) | 0.80-0.85 |
| 2010s-present (high partisanship) | 0.90-0.95 |

State legislatures vary widely, but Kansas (with strong party organizations) likely falls in the 0.85-0.92 range.

## Relationship to Bayesian Methods

Party Unity Scores are frequentist point estimates — they don't account for sample size. A legislator who voted on only 10 party votes and was loyal 9/10 times gets a unity score of 0.90, the same as one who voted on 200 party votes and was loyal 180/200 times. The `14_BAY_beta_binomial_party_loyalty` analysis addresses this by using Bayesian shrinkage to produce more reliable estimates for low-participation legislators.

## Kansas-Specific Considerations

- **Fewer party votes than you'd expect.** In a supermajority legislature, many bills pass with bipartisan support (Democrats voting Yea with Republicans), so the denominator (number of "party votes") may be smaller than in a closely divided legislature.
- **Republican intra-party variation is the story.** With 92 House Republicans and 32 Senate Republicans, the Unity Score distribution within the Republican party reveals the moderate-conservative spectrum.
- **Veto overrides complicate interpretation.** On veto overrides, some Democrats may vote with Republicans to override a Democratic governor's veto (or vice versa). These votes count as "party votes" and can lower unity scores for legislators who voted based on the bill's merits rather than party pressure.
- **Consider computing unity separately by vote type.** Unity on Final Action vs. Conference Committee vs. Veto Override may tell different stories about party discipline.

## Key References

- Cox, Gary W., and Keith T. Poole. "On Measuring Partisanship in Roll-Call Voting: The U.S. House of Representatives, 1877-1999." *American Journal of Political Science* 46(3), 2002.
- Congressional Quarterly methodology: CQ publishes annual party unity scores for every member of Congress.
- FiveThirtyEight Trump Score: https://projects.fivethirtyeight.com/congress-trump-score/ (a modern variant that measures unity with the president's position rather than party)
