# Effective Number of Parties (ENP) and Herfindahl Index

**Category:** Index-Based Measures
**Prerequisites:** `01_DATA_vote_matrix_construction`
**Complexity:** Low

## What It Measures

The Effective Number of Parties (ENP) quantifies how fragmented or concentrated a legislature is, accounting for party size. A legislature with two equally-sized parties has ENP = 2. One where a single party holds 90% of seats has ENP close to 1. The related Herfindahl-Hirschman Index (HHI) measures concentration (the inverse concept).

While Kansas formally has two parties, the *voting behavior* may reveal that the effective number of *voting blocs* differs from 2 — if the Republican majority frequently splits into moderate and conservative factions, the effective number of voting blocs could be 3 or more.

## Questions It Answers

- How concentrated is legislative power in Kansas?
- Does the effective number of voting blocs differ from the formal party count?
- On which votes does the legislature behave as if it has more than two parties?
- How does Kansas compare to other state legislatures in fragmentation?

## Mathematical Foundation

### Seat-Based ENP (Laakso-Taagepera Index)

Given $k$ parties with seat shares $s_1, s_2, \ldots, s_k$ where $\sum s_i = 1$:

$$\text{ENP}_{\text{seats}} = \frac{1}{\sum_{i=1}^{k} s_i^2} = \frac{1}{\text{HHI}}$$

**Herfindahl-Hirschman Index:**

$$\text{HHI} = \sum_{i=1}^{k} s_i^2$$

### Vote-Based ENP

More interesting than the static seat-based measure: compute ENP *per roll call* using the vote shares of each party's position:

For roll call $j$ with parties voting in different proportions, define "blocs" based on the Yea coalition and Nay coalition. If the Yea coalition has members from parties with shares $y_1, y_2, \ldots$ and the Nay coalition similarly:

$$\text{ENP}_j = \frac{1}{\sum_b s_{b,j}^2}$$

Where $s_{b,j}$ is the share of voters in bloc $b$ on roll call $j$.

### Effective Number of Voting Blocs (ENVB)

A more sophisticated approach: use clustering (from `18_CLU_hierarchical_clustering`) to identify $k$ voting blocs, then compute ENP using bloc sizes instead of party sizes. This captures intra-party factions that formal party labels miss.

## Python Implementation

### Static (Seat-Based) ENP

```python
import pandas as pd
import numpy as np

def compute_enp_seats(legislators: pd.DataFrame) -> dict:
    """Compute ENP based on seat shares."""
    party_counts = legislators["party"].value_counts()
    total = party_counts.sum()
    shares = party_counts / total
    hhi = (shares ** 2).sum()
    enp = 1 / hhi

    return {
        "party_counts": party_counts.to_dict(),
        "seat_shares": shares.to_dict(),
        "hhi": hhi,
        "enp": enp,
    }


# By chamber
for chamber in ["House", "Senate"]:
    chamber_legs = legislators[legislators["chamber"] == chamber]
    result = compute_enp_seats(chamber_legs)
    print(f"{chamber}: ENP = {result['enp']:.2f}, HHI = {result['hhi']:.3f}")
```

For the 2025-26 Kansas Legislature:
- House (92R, 38D): HHI = (92/130)^2 + (38/130)^2 = 0.501 + 0.085 = 0.586, ENP = 1.71
- Senate (32R, 10D): HHI = (32/42)^2 + (10/42)^2 = 0.580 + 0.057 = 0.637, ENP = 1.57

### Per-Vote ENP (Dynamic)

```python
def compute_enp_per_vote(
    votes: pd.DataFrame,
    legislators: pd.DataFrame,
) -> pd.DataFrame:
    """Compute effective number of voting blocs per roll call.

    Defines blocs as (party, vote_direction) combinations.
    """
    merged = votes.merge(
        legislators[["slug", "party"]],
        left_on="legislator_slug",
        right_on="slug",
    )
    substantive = merged[merged["vote"].isin(["Yea", "Nay"])]

    results = []
    for vote_id, group in substantive.groupby("vote_id"):
        total = len(group)
        # Define blocs: (party, vote) combinations
        bloc_counts = group.groupby(["party", "vote"]).size()
        bloc_shares = bloc_counts / total
        hhi = (bloc_shares ** 2).sum()
        enp = 1 / hhi if hhi > 0 else np.nan

        results.append({
            "vote_id": vote_id,
            "total_voters": total,
            "n_blocs": len(bloc_counts),
            "hhi": hhi,
            "enp": enp,
        })

    return pd.DataFrame(results)
```

### Visualization

```python
import matplotlib.pyplot as plt

def plot_enp_distribution(enp_per_vote: pd.DataFrame, save_path: str | None = None):
    """Plot distribution of per-vote ENP."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(enp_per_vote["enp"].dropna(), bins=40, edgecolor="black", alpha=0.7)
    ax.axvline(2.0, color="red", linestyle="--", label="ENP = 2 (pure two-party)")
    ax.axvline(enp_per_vote["enp"].mean(), color="blue", linestyle="--",
              label=f"Mean: {enp_per_vote['enp'].mean():.2f}")
    ax.set_xlabel("Effective Number of Voting Blocs")
    ax.set_ylabel("Number of Roll Calls")
    ax.set_title("Per-Vote ENP Distribution")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

## Interpretation Guide

| ENP Value | Interpretation |
|-----------|---------------|
| 1.0 | Perfect one-party state (unanimous votes) |
| 1.0-1.5 | Dominant party with weak opposition |
| 1.5-2.0 | Two parties but one dominant |
| 2.0 | Perfect two-party system (equal-sized parties on opposite sides) |
| 2.0-3.0 | Two main parties with a third faction (or one party splitting) |
| > 3.0 | Multiparty dynamics or severe party fragmentation |

For Kansas:
- **Seat-based ENP ~1.6** reflects the formal party imbalance.
- **Per-vote ENP near 1.0** = near-unanimous vote (everyone on the same side).
- **Per-vote ENP near 2.0** = classic party-line vote.
- **Per-vote ENP > 2.0** = one or both parties splitting, creating 3+ voting blocs. These are the most analytically interesting votes.

## Kansas-Specific Considerations

- The static seat-based ENP (~1.6) understates the effective competition because it doesn't capture the moderate Republican faction that sometimes votes with Democrats.
- Per-vote ENP reveals when the legislature effectively functions as a three-bloc body (conservative Republicans, moderate Republicans, Democrats).
- Combining per-vote ENP with the Rice Index gives a richer picture: low Rice + high ENP = true multi-party dynamics on that vote.
- This measure is most useful as a *filter* to identify interesting votes for deeper analysis, not as an end in itself.

## Key References

- Laakso, Markku, and Rein Taagepera. "Effective Number of Parties: A Measure with Application to West Europe." *Comparative Political Studies* 12(1), 1979.
- Golosov, Grigorii V. "The Effective Number of Parties: A New Approach." *Party Politics* 16(2), 2010. (Proposes modifications for dominant-party systems.)
- Effective Number of Parties, Wikipedia: https://en.wikipedia.org/wiki/Effective_number_of_parties
