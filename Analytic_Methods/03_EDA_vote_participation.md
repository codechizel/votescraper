# Vote Participation Analysis

**Category:** Exploratory Data Analysis
**Prerequisites:** `01_DATA_vote_matrix_construction`
**Complexity:** Low

## What It Measures

Vote participation analysis examines absenteeism patterns — which legislators miss votes, how often, and whether the pattern is random or systematic. In a legislature, absences are not random noise. They can signal strategic behavior (avoiding a controversial vote), health issues, competing obligations, or low engagement. The distinction between "Absent and Not Voting" (physical absence) and "Present and Passing" (deliberate abstention while present) is analytically important.

## Questions It Answers

- Which legislators have the highest and lowest participation rates?
- Are absences concentrated in certain periods, vote types, or topics?
- Do absences correlate with party, chamber, or seniority?
- Is there a difference between strategic abstention ("Present and Passing") and simple absence?
- Are certain roll calls systematically missed (e.g., late-night votes, procedural votes)?

## Input Data

- `votes.csv` — need the vote category column to distinguish Yea/Nay from absence types
- `legislators.csv` — party and chamber for grouping
- `rollcalls.csv` — vote datetime and type for temporal and procedural analysis

## Mathematical Foundation

### Participation Rate

For legislator $i$ across all roll calls in their chamber:

$$\text{Participation}_i = \frac{\text{Number of Yea or Nay votes cast by } i}{\text{Number of roll calls in } i\text{'s chamber}}$$

"Present and Passing" votes count as present but are analytically distinct — the legislator was there but chose not to take a side.

### Absence Rate by Category

$$\text{Absence Rate}_i = \frac{|\text{Absent and Not Voting}| + |\text{Not Voting}|}{\text{Total roll calls in chamber}}$$

$$\text{Abstention Rate}_i = \frac{|\text{Present and Passing}|}{\text{Total roll calls in chamber}}$$

## Python Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

votes = pd.read_csv("ks_2025_26_votes.csv")
legislators = pd.read_csv("ks_2025_26_legislators.csv")
rollcalls = pd.read_csv("ks_2025_26_rollcalls.csv")

# Count roll calls per chamber
chamber_rollcall_counts = rollcalls.groupby("chamber")["vote_id"].nunique()

# Count vote categories per legislator
legislator_vote_counts = votes.pivot_table(
    index="legislator_slug",
    columns="vote",
    values="vote_id",
    aggfunc="count",
    fill_value=0,
)

# Merge with legislator metadata
leg_meta = legislators.set_index("slug")[["full_name", "chamber", "party", "district"]]
participation = legislator_vote_counts.join(leg_meta)

# Compute participation rate
def calc_participation(row):
    total_rollcalls = chamber_rollcall_counts[row["chamber"]]
    substantive_votes = row.get("Yea", 0) + row.get("Nay", 0)
    return substantive_votes / total_rollcalls

participation["participation_rate"] = participation.apply(calc_participation, axis=1)
participation["absence_count"] = participation.get("Absent and Not Voting", 0)
participation["abstention_count"] = participation.get("Present and Passing", 0)
```

### Visualization: Participation Rate Distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, chamber in enumerate(["House", "Senate"]):
    chamber_data = participation[participation["chamber"] == chamber]
    colors = chamber_data["party"].map({"Republican": "#E81B23", "Democrat": "#0015BC"})

    axes[i].barh(
        range(len(chamber_data)),
        chamber_data["participation_rate"].sort_values(),
        color=colors.loc[chamber_data["participation_rate"].sort_values().index],
    )
    axes[i].set_xlabel("Participation Rate")
    axes[i].set_title(f"{chamber} Participation Rates")
    axes[i].axvline(0.95, color="gray", linestyle="--", alpha=0.5, label="95% threshold")
    axes[i].legend()

plt.tight_layout()
plt.savefig("participation_rates.png", dpi=150)
```

### Temporal Absence Patterns

```python
# Merge votes with rollcall dates
votes_dated = votes.merge(
    rollcalls[["vote_id", "vote_datetime", "vote_type"]],
    on="vote_id",
)
votes_dated["date"] = pd.to_datetime(votes_dated["vote_datetime"]).dt.date
votes_dated["is_absent"] = votes_dated["vote"].isin(["Absent and Not Voting", "Not Voting"])

# Daily absence rate
daily_absences = votes_dated.groupby("date")["is_absent"].mean()

fig, ax = plt.subplots(figsize=(14, 4))
daily_absences.plot(ax=ax)
ax.set_ylabel("Absence Rate")
ax.set_title("Daily Absence Rate Across Session")
plt.tight_layout()
```

### Absence by Vote Type

```python
absence_by_type = votes_dated.groupby("vote_type")["is_absent"].mean().sort_values(ascending=False)
```

This reveals whether legislators are more likely to miss procedural votes vs. final action votes.

### Strategic Absence Detection

A legislator who has high attendance overall but is absent for specific controversial votes may be strategically avoiding a recorded position.

```python
# Find contested votes (close margins)
rollcalls["margin_pct"] = abs(rollcalls["yea_count"] - rollcalls["nay_count"]) / rollcalls["total_votes"]
contested_votes = rollcalls[rollcalls["margin_pct"] < 0.2]["vote_id"]

# Compare absence rate on contested vs. non-contested votes
votes_dated["is_contested"] = votes_dated["vote_id"].isin(contested_votes)

strategic = votes_dated.groupby(["legislator_slug", "is_contested"])["is_absent"].mean().unstack()
strategic.columns = ["absence_rate_noncontested", "absence_rate_contested"]
strategic["strategic_absence_delta"] = (
    strategic["absence_rate_contested"] - strategic["absence_rate_noncontested"]
)

# Legislators with high delta may be strategically absent on contested votes
strategic_suspects = strategic.nlargest(10, "strategic_absence_delta")
```

## Interpretation Guide

- **Participation rate > 95%** is normal for most legislators. Below 90% is notable; below 80% warrants investigation.
- **"Present and Passing"** is rare (22 instances in the 2025-26 session, ~0.03%) and almost always strategic — the legislator is in the chamber but refuses to record a Yea or Nay.
- **Temporal spikes** in absences near session end or during veto overrides can indicate political maneuvering.
- **Party-correlated absence** is worth flagging — if one party's members are systematically more absent, it affects vote outcomes.
- **Higher absences on contested votes** (vs. overall) for specific legislators is a strong signal of strategic behavior.

## Kansas-Specific Considerations

- Kansas is a part-time legislature. Members have other jobs, which can affect attendance.
- Late-session votes (after sine die is extended) may show higher absence rates as members leave.
- With the Republican supermajority, strategic absence by Republicans on controversial votes can be a way to avoid going on record without changing the outcome.
- Veto override votes require 2/3 majority of the *elected* membership (not just those present), so absences effectively count as "No" votes on veto overrides. This makes participation analysis especially important for those 34 override votes.

## Key References

- Forgette, Richard. "Party Caucuses and Coordination: Assessing Caucus Activity and Party Effects." *Legislative Studies Quarterly* 29(3), 2004.
- Cohen, Linda R., and Matthew L. Spitzer. "Term Limits and Representation." *Journal of Law, Economics, and Organization*, 1996.
- Practical reference: [FiveThirtyEight's "Tracking Congress" methodology](https://projects.fivethirtyeight.com/congress-trump-score/)
