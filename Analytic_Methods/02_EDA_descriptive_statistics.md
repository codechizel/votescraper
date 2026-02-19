# Descriptive Statistics

**Category:** Exploratory Data Analysis
**Prerequisites:** `01_DATA_vote_matrix_construction`
**Complexity:** Low — pure pandas/numpy, no modeling

## What It Measures

Descriptive statistics provide the baseline understanding of the legislative session. Before applying any sophisticated model, you need to know: how many bills were voted on, how often they pass, how participation varies, and whether the basic partisan structure shows up in raw counts.

## Questions It Answers

- How many roll calls occurred, and what types (Final Action, Conference Committee, Veto Override, etc.)?
- What is the overall passage rate? Does it differ by chamber, vote type, or time period?
- How do vote margins distribute? Are most votes near-unanimous or closely contested?
- What is the basic partisan split on contested votes?
- How does legislative activity vary over the session calendar?

## Input Data

All three CSVs directly — no vote matrix needed for most of these.

## Analyses and Implementation

### 1. Session Summary Statistics

```python
import pandas as pd

rollcalls = pd.read_csv("ks_2025_26_rollcalls.csv")
votes = pd.read_csv("ks_2025_26_votes.csv")
legislators = pd.read_csv("ks_2025_26_legislators.csv")

summary = {
    "total_roll_calls": len(rollcalls),
    "unique_bills": rollcalls["bill_number"].nunique(),
    "total_individual_votes": len(votes),
    "total_legislators": len(legislators),
    "house_members": len(legislators[legislators["chamber"] == "House"]),
    "senate_members": len(legislators[legislators["chamber"] == "Senate"]),
    "republicans": len(legislators[legislators["party"] == "Republican"]),
    "democrats": len(legislators[legislators["party"] == "Democrat"]),
    "date_range": f'{rollcalls["vote_date"].min()} to {rollcalls["vote_date"].max()}',
}
```

### 2. Vote Type Distribution

```python
vote_type_counts = rollcalls["vote_type"].value_counts()
vote_type_by_chamber = rollcalls.groupby(["chamber", "vote_type"]).size().unstack(fill_value=0)
```

This reveals the legislative workload structure. Kansas typically has more Final Action votes than procedural ones. Conference Committee votes indicate bills that required reconciliation between chambers.

### 3. Passage Rate Analysis

```python
# Overall passage rate
passage_rate = rollcalls["passed"].mean()  # Excludes NaN (procedural votes)

# By chamber
passage_by_chamber = rollcalls.groupby("chamber")["passed"].mean()

# By vote type
passage_by_type = rollcalls.groupby("vote_type")["passed"].mean()
```

### 4. Vote Margin Distribution

The vote margin (Yea% - Nay%) reveals how contested the legislature is. A bimodal distribution (peaks near 0% and 100%) indicates a mix of contested and unanimous votes.

```python
import matplotlib.pyplot as plt

rollcalls["yea_pct"] = rollcalls["yea_count"] / rollcalls["total_votes"]
rollcalls["margin"] = abs(rollcalls["yea_count"] - rollcalls["nay_count"]) / rollcalls["total_votes"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Yea percentage distribution
axes[0].hist(rollcalls["yea_pct"], bins=50, edgecolor="black")
axes[0].set_xlabel("Yea Percentage")
axes[0].set_ylabel("Number of Roll Calls")
axes[0].set_title("Distribution of Yea Percentage Across Roll Calls")
axes[0].axvline(0.5, color="red", linestyle="--", label="50% threshold")
axes[0].legend()

# Margin distribution
axes[1].hist(rollcalls["margin"], bins=50, edgecolor="black")
axes[1].set_xlabel("Vote Margin (|Yea - Nay| / Total)")
axes[1].set_ylabel("Number of Roll Calls")
axes[1].set_title("Distribution of Vote Margins")

plt.tight_layout()
plt.savefig("descriptive_vote_margins.png", dpi=150)
```

### 5. Temporal Activity Pattern

```python
rollcalls["vote_date_parsed"] = pd.to_datetime(rollcalls["vote_date"], format="%m/%d/%Y")
rollcalls["month"] = rollcalls["vote_date_parsed"].dt.to_period("M")

monthly_activity = rollcalls.groupby("month").size()

fig, ax = plt.subplots(figsize=(12, 5))
monthly_activity.plot(kind="bar", ax=ax)
ax.set_xlabel("Month")
ax.set_ylabel("Number of Roll Calls")
ax.set_title("Legislative Activity by Month")
plt.tight_layout()
plt.savefig("descriptive_temporal_activity.png", dpi=150)
```

Kansas legislatures typically show a burst of activity near session deadlines (turnaround day, sine die adjournment). This plot reveals those patterns.

### 6. Vote Category Breakdown

```python
vote_category_counts = votes["vote"].value_counts()
vote_category_pct = votes["vote"].value_counts(normalize=True) * 100

# By party (requires merge with legislators)
votes_with_party = votes.merge(
    legislators[["slug", "party"]],
    left_on="legislator_slug",
    right_on="slug",
)
party_vote_dist = votes_with_party.groupby(["party", "vote"])["vote"].count().unstack()
```

### 7. Party-Line vs. Bipartisan Votes

A critical initial question: how often do the parties oppose each other?

```python
def classify_vote_partisanship(rollcall_votes: pd.DataFrame, legislators: pd.DataFrame) -> str:
    """Classify a roll call as party-line, bipartisan, or mixed."""
    merged = rollcall_votes.merge(legislators[["slug", "party"]], left_on="legislator_slug", right_on="slug")
    substantive = merged[merged["vote"].isin(["Yea", "Nay"])]

    for party in ["Republican", "Democrat"]:
        party_votes = substantive[substantive["party"] == party]
        if len(party_votes) == 0:
            continue
        party_yea_pct = (party_votes["vote"] == "Yea").mean()
        # Store for comparison

    # A "party vote" is one where majorities of each party voted opposite ways
    rep_majority_yea = rep_yea_pct > 0.5
    dem_majority_yea = dem_yea_pct > 0.5

    if rep_majority_yea != dem_majority_yea:
        return "party_line"  # Parties on opposite sides
    else:
        return "bipartisan"  # Parties on same side
```

```python
# Count party-line vs bipartisan for each roll call
vote_types_classified = []
for vote_id, group in votes.groupby("vote_id"):
    classification = classify_vote_partisanship(group, legislators)
    vote_types_classified.append({"vote_id": vote_id, "partisanship": classification})

partisanship_df = pd.DataFrame(vote_types_classified)
print(partisanship_df["partisanship"].value_counts())
```

In a supermajority legislature like Kansas, expect a large fraction of bipartisan votes (where the opposition doesn't have enough members to oppose meaningfully) and a smaller set of genuinely contested party-line votes.

### 8. Sponsor Analysis

```python
# Which sponsors have the most roll calls?
sponsor_counts = rollcalls["sponsor"].value_counts().head(20)

# Passage rate by sponsor party (requires parsing sponsor party from name)
# Sponsors are formatted as "Senator Name" or "Representative Name"
```

## Visualization Recommendations

| Chart | Library | Purpose |
|-------|---------|---------|
| Bar chart of vote types | `matplotlib` | Legislative workload structure |
| Histogram of vote margins | `matplotlib` | Contestedness distribution |
| Stacked bar by party x vote | `seaborn` | Partisan voting patterns |
| Calendar heatmap of activity | `matplotlib` + custom | Session rhythm |
| Treemap of bill subjects | `squarify` | Policy area distribution |

## Interpretation Guide

- **High passage rate** (>80%) is normal for state legislatures where the majority party controls the floor calendar.
- **Bimodal margin distribution** (many near-unanimous + some close votes) is the typical pattern. The close votes are where the interesting analysis happens.
- **Temporal clustering** of votes near deadlines reflects procedural rules (turnaround day, committee deadlines).
- **Few party-line votes** in a supermajority legislature doesn't mean parties don't matter — it means the majority party can afford internal dissent while still winning.

## Kansas-Specific Considerations

- With a ~72/28 Republican/Democrat split, the majority party can lose up to ~22% of its members on any vote and still win. This makes intra-party dynamics more important than inter-party ones.
- The 34 veto override votes are a natural case study — these require 2/3 supermajority and often have unusual coalition patterns.
- Conference Committee votes (14% of roll calls) represent the final negotiated version of bills and may have different partisan dynamics than initial passage votes.

## Key References

- No specific paper — descriptive statistics are the universal starting point for any data analysis.
- For Kansas-specific legislative context: [Kansas Legislature website](https://kslegislature.gov)
