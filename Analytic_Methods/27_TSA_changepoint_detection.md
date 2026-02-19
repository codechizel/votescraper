# Changepoint Detection in Voting Patterns

**Category:** Time Series Analysis
**Prerequisites:** `05_IDX_rice_index`, `01_DATA_vote_matrix_construction`
**Complexity:** Medium
**Caveat:** Requires sufficient temporal resolution (~500 roll calls ordered chronologically)

## What It Measures

Changepoint detection identifies moments when the statistical properties of legislative voting structurally change. Unlike ideological drift (which tracks gradual evolution), changepoint detection looks for abrupt shifts: a party suddenly becoming less cohesive, a new coalition forming, or a controversial event fracturing existing alliances.

## Questions It Answers

- Are there dates when voting patterns structurally changed?
- Did a specific bill, controversy, or event cause a lasting shift in party cohesion?
- Is the session better described as one regime or multiple regimes with different dynamics?
- When did partisan polarization peak?

## Mathematical Foundation

### Offline Changepoint Detection (Ruptures)

Given a time series $y_1, y_2, \ldots, y_T$ (e.g., daily Rice Index), find the set of changepoints $\{t_1, t_2, \ldots, t_k\}$ that minimizes:

$$\sum_{i=0}^{k} \text{cost}(y_{t_i+1:t_{i+1}})$$

Subject to a penalty for the number of changepoints:

$$\min_k \left[\sum_{i=0}^{k} \text{cost}(y_{t_i+1:t_{i+1}}) + \lambda \cdot k\right]$$

Common cost functions:
- **L2 (mean shift)**: $\text{cost}(y) = \sum (y_i - \bar{y})^2$ — detects shifts in the average level
- **Normal (mean + variance shift)**: Detects changes in both mean and variance
- **Rank-based (non-parametric)**: Robust to outliers

### Bayesian Online Changepoint Detection (BOCPD)

BOCPD processes data sequentially, maintaining a probability distribution over the "run length" (number of observations since the last changepoint). At each new observation, it updates:

$$P(r_t | y_{1:t}) \propto \sum_{r_{t-1}} P(y_t | r_{t-1}, y_{t-r_{t-1}:t-1}) \cdot P(r_t | r_{t-1}) \cdot P(r_{t-1} | y_{1:t-1})$$

Where $r_t$ = run length at time $t$. A spike in $P(r_t = 0)$ indicates a changepoint.

## Python Implementation

### Using Ruptures (Offline)

```python
import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

def detect_changepoints_in_cohesion(
    rice_over_time: pd.DataFrame,
    party: str = "Republican",
    n_changepoints: int | None = None,
    penalty: float = 10.0,
) -> tuple[list[int], np.ndarray]:
    """Detect changepoints in party cohesion time series.

    Args:
        rice_over_time: DataFrame with date index and party-specific Rice Index.
        party: Which party's cohesion to analyze.
        n_changepoints: Exact number of changepoints (if known). If None, use penalty.
        penalty: Penalty for number of changepoints (higher = fewer changepoints).

    Returns:
        changepoints: List of changepoint indices.
        signal: The time series analyzed.
    """
    signal = rice_over_time[party].values.astype(float)

    # Remove NaN
    valid = ~np.isnan(signal)
    signal_clean = signal[valid]

    # Detect changepoints
    algo = rpt.Pelt(model="rbf", min_size=5).fit(signal_clean)

    if n_changepoints is not None:
        changepoints = rpt.Binseg(model="l2", min_size=5).fit(signal_clean).predict(n_bkps=n_changepoints)
    else:
        changepoints = algo.predict(pen=penalty)

    return changepoints, signal_clean


def build_cohesion_timeseries(
    votes: pd.DataFrame,
    legislators: pd.DataFrame,
    rollcalls: pd.DataFrame,
) -> pd.DataFrame:
    """Build a time series of per-vote Rice Index for each party."""
    merged = votes.merge(
        legislators[["slug", "party"]],
        left_on="legislator_slug",
        right_on="slug",
    )
    substantive = merged[merged["vote"].isin(["Yea", "Nay"])]

    results = []
    for vote_id, group in substantive.groupby("vote_id"):
        rc = rollcalls[rollcalls["vote_id"] == vote_id].iloc[0]
        date = pd.to_datetime(rc["vote_datetime"])

        for party_name in ["Republican", "Democrat"]:
            party_group = group[group["party"] == party_name]
            if len(party_group) == 0:
                continue
            yea_pct = (party_group["vote"] == "Yea").mean()
            rice = abs(2 * yea_pct - 1)

            results.append({
                "vote_id": vote_id,
                "date": date,
                "party": party_name,
                "rice_index": rice,
                "n_voters": len(party_group),
            })

    df = pd.DataFrame(results)
    # Pivot to wide format with rolling average
    pivot = df.pivot_table(index="date", columns="party", values="rice_index", aggfunc="mean")
    pivot = pivot.sort_index()

    # Apply rolling average for smoother changepoint detection
    smoothed = pivot.rolling(7, min_periods=1).mean()

    return smoothed


def plot_changepoints(
    signal: np.ndarray,
    changepoints: list[int],
    dates: pd.DatetimeIndex | None = None,
    party: str = "Republican",
    save_path: str | None = None,
):
    """Visualize detected changepoints on the cohesion time series."""
    fig, ax = plt.subplots(figsize=(14, 5))

    x_axis = dates if dates is not None else range(len(signal))
    color = "#E81B23" if party == "Republican" else "#0015BC"

    ax.plot(x_axis, signal, color=color, alpha=0.7, linewidth=1)

    # Mark changepoints
    for cp in changepoints:
        if cp < len(signal):
            x_val = dates[cp] if dates is not None else cp
            ax.axvline(x_val, color="red", linestyle="--", alpha=0.7, linewidth=1.5)

    # Color segments between changepoints
    cps = [0] + changepoints
    cmap = plt.cm.Pastel1
    for i in range(len(cps) - 1):
        start, end = cps[i], min(cps[i + 1], len(signal))
        x_range = x_axis[start:end]
        y_range = signal[start:end]
        ax.fill_between(x_range, y_range, alpha=0.15, color=cmap(i % 8))

    ax.set_xlabel("Date")
    ax.set_ylabel("Rice Index (7-day rolling avg)")
    ax.set_title(f"{party} Party Cohesion with Detected Changepoints")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

### Multi-Signal Changepoint Detection

```python
def detect_multivariate_changepoints(
    cohesion_ts: pd.DataFrame,
    penalty: float = 20.0,
) -> list[int]:
    """Detect changepoints using both parties' cohesion simultaneously.

    Changepoints in the joint signal indicate moments when the
    *overall* legislative dynamics changed, not just one party.
    """
    signal = cohesion_ts[["Republican", "Democrat"]].dropna().values

    algo = rpt.Pelt(model="rbf", min_size=5).fit(signal)
    changepoints = algo.predict(pen=penalty)

    return changepoints
```

## Interpretation Guide

### What Changepoints Mean

- **Cohesion drop (Rice decreases)**: The party became less unified. Could be due to a controversial bill, internal conflict, or leadership challenge.
- **Cohesion increase**: The party consolidated around an issue (e.g., response to a governor's veto, election-year rallying).
- **Simultaneous changepoints in both parties**: A structural shift in the legislative environment (new session phase, leadership change, external event).
- **Changepoint in one party only**: An internal party event (faction conflict, defection of key members).

### Validation

Cross-reference detected changepoints with:
- Kansas legislative calendar (turnaround day, committee deadlines, sine die)
- Major bills or vetoes near the changepoint date
- News coverage of the Kansas Legislature around those dates

### Limitations

- **Sensitivity to penalty/smoothing**: Different penalty values produce different numbers of changepoints. Report results across a range of penalties.
- **Agenda effects**: Changes in which bills are voted on can create apparent changepoints in cohesion even if legislators' behavior didn't change.
- **Small sample per time point**: Each day may have only 1-10 roll calls, making daily Rice Index noisy. Smoothing helps but delays detection.

## Kansas-Specific Considerations

- **Look for changepoints around veto override periods.** The 34 veto overrides likely cluster in time and represent a distinct regime where cross-party dynamics differ.
- **Turnaround day** (deadline for bills to pass their chamber of origin) is a natural structural changepoint.
- **The early-session/late-session transition** often shows a changepoint as the agenda shifts from routine bills to the contentious ones.
- **Use per-week or per-two-week aggregation** rather than per-day to reduce noise.

## Feasibility Assessment

- **Data size**: ~500 roll calls over ~11 months = ~2 roll calls per day on average. Marginal for fine-grained detection, adequate for weekly aggregation.
- **Compute time**: Sub-second for ruptures, seconds for BOCPD
- **Libraries**: `ruptures` (offline), `bayesian-changepoint-detection` or custom (BOCPD)
- **Difficulty**: Medium

## Key References

- Adams, Ryan P., and David J.C. MacKay. "Bayesian Online Changepoint Detection." arXiv:0710.3742, 2007.
- Killick, Rebecca, Paul Fearnhead, and Idris A. Eckley. "Optimal Detection of Changepoints with a Linear Computational Cost." *Journal of the American Statistical Association* 107(500), 2012.
- `ruptures` documentation: https://centre-borelli.github.io/ruptures-docs/
- Gundersen, Gregory. "Implementing Bayesian Online Changepoint Detection." https://gregorygundersen.com/blog/2020/10/20/implementing-bocd/
