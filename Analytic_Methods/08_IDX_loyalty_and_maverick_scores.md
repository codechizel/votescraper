# Loyalty and Maverick Scores

**Category:** Index-Based Measures
**Prerequisites:** `06_IDX_party_unity_scores`
**Complexity:** Low
**Related:** `14_BAY_beta_binomial_party_loyalty` (Bayesian version)

## What It Measures

Loyalty and Maverick scores are extensions of Party Unity that go beyond simple party-line frequency. They measure: (a) how often a legislator's vote was decisive in maintaining or breaking party cohesion, (b) how often a legislator crossed party lines specifically on close or consequential votes, and (c) relative positioning within their party's loyalty distribution. The "maverick" framing identifies legislators whose voting behavior stands out from their partisan peers.

## Questions It Answers

- Who are the most reliable party soldiers and who are the biggest rebels?
- When legislators defect from their party, which direction do they go (toward the opposition, or toward abstention)?
- Which legislators are "pivotal mavericks" — their defection actually changes outcomes?
- Within the Republican majority, who are the moderate outliers that sometimes side with Democrats?
- Is there a "maverick caucus" — a group of legislators who consistently defect together?

## Mathematical Foundation

### Basic Maverick Score

$$\text{Maverick}_i = 1 - \text{Unity}_i$$

A legislator with Unity = 0.85 has Maverick = 0.15 (defects 15% of the time on party votes).

### Weighted Maverick Score

Not all defections are equal. Defecting on a close vote is more consequential than defecting on a blowout. Weight each defection by how close the vote was:

$$\text{Weighted Maverick}_i = \frac{\sum_{j \in D_i} w_j}{\sum_{j \in P} w_j}$$

Where:
- $D_i$ = set of party votes where legislator $i$ defected
- $P$ = set of all party votes where $i$ was present
- $w_j = 1 - \text{margin}_j$ (weight inversely proportional to margin; close votes weighted more)

### Cross-Party Agreement Score

How often does a legislator vote with the *opposition* party's majority?

$$\text{Cross-Party}_i = \frac{|\{j : i \text{ voted with other party's majority}\}|}{|\{j : j \text{ is a party vote, } i \text{ present}\}|}$$

Note: $\text{Cross-Party}_i + \text{Unity}_i \leq 1$ (the gap is votes where neither party has a clear majority position, which doesn't apply under our definition of party votes).

### Loyalty Z-Score (Relative Positioning)

How many standard deviations a legislator's unity score is from their party mean:

$$z_i = \frac{\text{Unity}_i - \mu_{\text{party}(i)}}{\sigma_{\text{party}(i)}}$$

This identifies outliers relative to their own party, which is more meaningful than raw scores in a legislature where both parties have high baseline unity.

## Python Implementation

```python
import pandas as pd
import numpy as np

def compute_maverick_scores(
    votes: pd.DataFrame,
    legislators: pd.DataFrame,
    rollcalls: pd.DataFrame,
    unity_df: pd.DataFrame,
    party_vote_ids: pd.Series,
) -> pd.DataFrame:
    """Compute comprehensive loyalty/maverick metrics."""
    pv_ids = party_vote_ids[party_vote_ids].index

    # Get vote margins for weighting
    margins = rollcalls.set_index("vote_id")[["yea_count", "nay_count", "total_votes"]]
    margins["margin"] = abs(margins["yea_count"] - margins["nay_count"]) / margins["total_votes"]
    margins["closeness_weight"] = 1 - margins["margin"]

    # Merge everything
    merged = votes[votes["vote_id"].isin(pv_ids) & votes["vote"].isin(["Yea", "Nay"])].merge(
        legislators[["slug", "party", "chamber"]],
        left_on="legislator_slug",
        right_on="slug",
    )

    # Determine party majority position per vote
    party_positions = {}
    for vote_id, group in merged.groupby("vote_id"):
        for party_name in ["Republican", "Democrat"]:
            pg = group[group["party"] == party_name]
            if len(pg) > 0:
                majority_yea = (pg["vote"] == "Yea").mean() > 0.5
                party_positions[(vote_id, party_name)] = "Yea" if majority_yea else "Nay"

    # Other party's majority position
    other_party = {"Republican": "Democrat", "Democrat": "Republican"}

    results = []
    for slug, group in merged.groupby("legislator_slug"):
        party = group["party"].iloc[0]
        chamber = group["chamber"].iloc[0]
        n_party_votes = len(group)

        defections = 0
        weighted_defections = 0
        weighted_total = 0
        cross_party_votes = 0

        for _, row in group.iterrows():
            vid = row["vote_id"]
            my_party_pos = party_positions.get((vid, party))
            opp_party_pos = party_positions.get((vid, other_party[party]))
            weight = margins.loc[vid, "closeness_weight"] if vid in margins.index else 0.5

            weighted_total += weight
            voted_with_party = row["vote"] == my_party_pos

            if not voted_with_party:
                defections += 1
                weighted_defections += weight

            if opp_party_pos and row["vote"] == opp_party_pos:
                cross_party_votes += 1

        results.append({
            "legislator_slug": slug,
            "party": party,
            "chamber": chamber,
            "party_votes_present": n_party_votes,
            "defections": defections,
            "maverick_score": defections / n_party_votes if n_party_votes > 0 else np.nan,
            "weighted_maverick": weighted_defections / weighted_total if weighted_total > 0 else np.nan,
            "cross_party_rate": cross_party_votes / n_party_votes if n_party_votes > 0 else np.nan,
        })

    result_df = pd.DataFrame(results)

    # Add Z-scores within party
    for party_name in ["Republican", "Democrat"]:
        mask = result_df["party"] == party_name
        scores = result_df.loc[mask, "maverick_score"]
        result_df.loc[mask, "maverick_zscore"] = (scores - scores.mean()) / scores.std()

    # Merge with unity scores
    result_df = result_df.merge(
        unity_df[["legislator_slug", "unity_score"]],
        on="legislator_slug",
        how="left",
    )

    return result_df.sort_values("maverick_score", ascending=False)
```

### Co-Defection Analysis (Maverick Caucus Detection)

```python
def find_co_defectors(
    votes: pd.DataFrame,
    legislators: pd.DataFrame,
    party_vote_ids: pd.Series,
    party: str = "Republican",
    min_co_defections: int = 5,
) -> pd.DataFrame:
    """Find pairs of legislators who frequently defect together."""
    pv_ids = party_vote_ids[party_vote_ids].index
    party_slugs = legislators[legislators["party"] == party]["slug"].tolist()

    # Find defection votes per legislator
    # (votes where they voted against party majority)
    # ... (build on party_positions from above)

    # For each pair, count roll calls where both defected
    from itertools import combinations

    co_defection_counts = {}
    for slug_a, slug_b in combinations(defecting_legislators, 2):
        shared_defections = defections[slug_a] & defections[slug_b]
        if len(shared_defections) >= min_co_defections:
            co_defection_counts[(slug_a, slug_b)] = len(shared_defections)

    return pd.DataFrame([
        {"legislator_a": a, "legislator_b": b, "co_defections": n}
        for (a, b), n in co_defection_counts.items()
    ]).sort_values("co_defections", ascending=False)
```

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_maverick_landscape(maverick_df: pd.DataFrame, chamber: str, save_path: str | None = None):
    """Scatter plot: maverick score vs. weighted maverick score, sized by participation."""
    data = maverick_df[maverick_df["chamber"] == chamber]
    colors = data["party"].map({"Republican": "#E81B23", "Democrat": "#0015BC"})

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        data["maverick_score"],
        data["weighted_maverick"],
        c=colors,
        s=data["party_votes_present"] * 2,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )

    # Label top mavericks
    top_mavericks = data.nlargest(5, "maverick_score")
    for _, row in top_mavericks.iterrows():
        ax.annotate(
            row["legislator_slug"].split("_")[1],  # Last name
            (row["maverick_score"], row["weighted_maverick"]),
            fontsize=8,
            ha="left",
        )

    ax.set_xlabel("Maverick Score (unweighted)")
    ax.set_ylabel("Weighted Maverick Score (close votes weighted more)")
    ax.set_title(f"{chamber} Maverick Landscape")
    ax.plot([0, 0.5], [0, 0.5], "k--", alpha=0.3, label="1:1 line")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

## Interpretation Guide

- **Weighted maverick > unweighted maverick**: This legislator defects disproportionately on close votes. Their defections are strategically consequential.
- **Weighted maverick < unweighted maverick**: This legislator defects on lopsided votes (where it doesn't matter). Their rebellion is performative rather than pivotal.
- **High cross-party rate**: Not just defecting from own party, but actively voting with the opposition. Stronger signal of ideological distance from party.
- **Positive maverick Z-score**: More maverick than the party average. Z > 2 is a clear outlier.
- **Co-defection clusters**: Groups of legislators who defect together on the same votes likely represent an organized faction (e.g., a moderate caucus within the Republican party).

## Kansas-Specific Considerations

- **The moderate Republican caucus** is the primary story. In Kansas politics, the divide between moderate and conservative Republicans is often more consequential than the Republican-Democrat divide. Maverick scores within the Republican party identify these moderates.
- **Pivotal mavericks on veto overrides**: Since overrides require 2/3 of elected members, a handful of Republican defectors can sustain a governor's veto. Identifying who these pivotal mavericks are on override votes is politically significant.
- **Don't over-interpret Democratic maverick scores.** With only 38 House Democrats and 10 Senate Democrats, a single defector can produce a large maverick score, and the sample of party votes is smaller.
- **Consider issue-specific mavericks.** A legislator might be a maverick on fiscal votes but a loyalist on social issues. The aggregate maverick score averages over all issues.

## Key References

- Poole, Keith T., and Howard Rosenthal. "The Polarization of American Politics." *Journal of Politics* 46(4), 1984.
- McCarty, Nolan, Keith T. Poole, and Howard Rosenthal. *Polarized America: The Dance of Ideology and Unequal Riches*. MIT Press, 2006.
- FiveThirtyEight methodology for "maverick" identification in Congress.
