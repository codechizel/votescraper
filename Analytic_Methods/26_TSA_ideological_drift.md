# Ideological Drift Within a Session

**Category:** Time Series Analysis
**Prerequisites:** `09_DIM_principal_component_analysis` or `15_BAY_irt_ideal_points`
**Complexity:** Medium-High
**Caveat:** Limited signal from a single session

## What It Measures

Ideological drift analysis tracks how legislators' voting positions change over the course of a legislative session. While the canonical application (DW-NOMINATE) tracks drift across decades of congressional history, within-session analysis asks: do legislators shift their voting behavior between the beginning and end of session? Do specific events (leadership challenges, controversial bills, election-year pressures) cause measurable shifts?

## Questions It Answers

- Do legislators' ideal points change between early-session and late-session votes?
- Do certain events (veto overrides, leadership elections) coincide with shifts?
- Is within-session drift larger for some legislators or parties?
- Does the partisan gap widen or narrow over the session?

## Mathematical Foundation

### Rolling-Window Ideal Points

Divide the session into $T$ overlapping time windows. For each window $t$, estimate ideal points using only the roll calls within that window.

$$\hat{\xi}_i^{(t)} = \text{PCA}(\mathbf{X}^{(t)})_1 \quad \text{or} \quad \hat{\xi}_i^{(t)} = \text{IRT}(\mathbf{X}^{(t)})$$

Where $\mathbf{X}^{(t)}$ is the vote matrix restricted to roll calls in time window $t$.

**Window parameters:**
- Window size: 50-100 roll calls (enough for stable estimation)
- Overlap: 50-75% (smooth trajectory)
- Step size: 10-25 roll calls per step

### Dynamic IRT (Bayesian Random Walk)

A more principled approach: model ideal points as evolving over time with a random walk prior:

$$\xi_i^{(t+1)} | \xi_i^{(t)} \sim \text{Normal}(\xi_i^{(t)}, \sigma_{\text{drift}}^2)$$

The drift variance $\sigma_{\text{drift}}^2$ is estimated from the data. Small $\sigma_{\text{drift}}$ = stable ideal points; large = rapidly evolving positions.

### Statistical Test for Drift

Compare early-session and late-session ideal points using a paired test:

$$H_0: \mu_{\text{early}} = \mu_{\text{late}}$$

Using either a paired t-test (on PCA scores) or a posterior comparison (for Bayesian IRT).

## Python Implementation

### Rolling-Window PCA

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def rolling_window_ideal_points(
    vote_matrix: pd.DataFrame,
    rollcall_dates: pd.Series,
    window_size: int = 75,
    step_size: int = 15,
) -> pd.DataFrame:
    """Compute ideal points in rolling time windows.

    Args:
        vote_matrix: Binary vote matrix with columns ordered chronologically.
        rollcall_dates: Series mapping vote_id to datetime, aligned with vote_matrix columns.
        window_size: Number of roll calls per window.
        step_size: Roll calls between window starts.

    Returns:
        DataFrame with columns: legislator_slug, window_start, window_end, pc1_score.
    """
    # Sort columns chronologically
    dates = rollcall_dates.loc[vote_matrix.columns].sort_values()
    sorted_cols = dates.index.tolist()
    vm_sorted = vote_matrix[sorted_cols]

    results = []
    for start in range(0, len(sorted_cols) - window_size + 1, step_size):
        end = start + window_size
        window_cols = sorted_cols[start:end]
        window_matrix = vm_sorted[window_cols]

        # Drop legislators with too few votes in this window
        has_votes = window_matrix.notna().sum(axis=1) >= 10
        window_matrix = window_matrix.loc[has_votes]

        if len(window_matrix) < 20:
            continue

        # Impute and run PCA
        filled = window_matrix.T.fillna(window_matrix.mean(axis=1)).T
        scaler = StandardScaler()
        X = scaler.fit_transform(filled)

        pca = PCA(n_components=1)
        scores = pca.fit_transform(X)

        window_start_date = dates[window_cols[0]]
        window_end_date = dates[window_cols[-1]]
        window_midpoint = window_start_date + (window_end_date - window_start_date) / 2

        for i, slug in enumerate(window_matrix.index):
            results.append({
                "legislator_slug": slug,
                "window_start": window_start_date,
                "window_end": window_end_date,
                "window_midpoint": window_midpoint,
                "pc1_score": scores[i, 0],
                "n_votes": window_matrix.loc[slug].notna().sum(),
            })

    return pd.DataFrame(results)
```

### Sign Alignment Across Windows

PCA components can flip sign between windows. Align them:

```python
def align_pc_signs(
    rolling_df: pd.DataFrame,
    legislator_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure consistent sign convention across windows."""
    for window, group in rolling_df.groupby("window_midpoint"):
        # Convention: Republicans should have positive PC1
        merged = group.merge(
            legislator_meta[["party"]],
            left_on="legislator_slug",
            right_index=True,
        )
        rep_mean = merged.loc[merged["party"] == "Republican", "pc1_score"].mean()
        if rep_mean < 0:
            rolling_df.loc[group.index, "pc1_score"] *= -1

    return rolling_df
```

### Visualization

```python
def plot_ideological_drift(
    rolling_df: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    slugs: list[str] | None = None,
    save_path: str | None = None,
):
    """Plot ideal point trajectories over time."""
    if slugs is None:
        # Show party averages
        merged = rolling_df.merge(
            legislator_meta[["party"]],
            left_on="legislator_slug",
            right_index=True,
        )
        party_avg = merged.groupby(["window_midpoint", "party"])["pc1_score"].mean().unstack()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(party_avg.index, party_avg["Republican"], color="#E81B23", linewidth=2,
               label="Republican (avg)")
        ax.plot(party_avg.index, party_avg["Democrat"], color="#0015BC", linewidth=2,
               label="Democrat (avg)")

        # Shade between parties (polarization gap)
        ax.fill_between(
            party_avg.index,
            party_avg["Republican"],
            party_avg["Democrat"],
            alpha=0.1, color="gray",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Average Ideal Point (PC1)")
        ax.set_title("Partisan Polarization Over the Session")
        ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        for slug in slugs:
            leg_data = rolling_df[rolling_df["legislator_slug"] == slug]
            party = legislator_meta.loc[slug, "party"]
            color = "#E81B23" if party == "Republican" else "#0015BC"
            name = slug.split("_")[1].title()
            ax.plot(leg_data["window_midpoint"], leg_data["pc1_score"],
                   color=color, alpha=0.7, label=name)

        ax.set_xlabel("Date")
        ax.set_ylabel("Ideal Point (PC1)")
        ax.set_title("Individual Ideological Trajectories")
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)


def plot_polarization_gap(
    rolling_df: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    save_path: str | None = None,
):
    """Plot the partisan gap (distance between party means) over time."""
    merged = rolling_df.merge(
        legislator_meta[["party"]],
        left_on="legislator_slug",
        right_index=True,
    )
    party_avg = merged.groupby(["window_midpoint", "party"])["pc1_score"].mean().unstack()
    gap = abs(party_avg["Republican"] - party_avg["Democrat"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(gap.index, gap.values, "k-", linewidth=2)
    ax.fill_between(gap.index, gap.values, alpha=0.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Partisan Gap (|Rep mean - Dem mean|)")
    ax.set_title("Partisan Polarization Gap Over the Session")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

### Early vs. Late Comparison

```python
def test_early_vs_late_drift(
    vote_matrix: pd.DataFrame,
    legislator_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Compare ideal points from first half vs. second half of session."""
    mid = vote_matrix.shape[1] // 2
    cols = vote_matrix.columns.tolist()

    early_matrix = vote_matrix[cols[:mid]]
    late_matrix = vote_matrix[cols[mid:]]

    def get_pc1(vm):
        has_votes = vm.notna().sum(axis=1) >= 10
        vm = vm.loc[has_votes]
        filled = vm.T.fillna(vm.mean(axis=1)).T
        X = StandardScaler().fit_transform(filled)
        pca = PCA(n_components=1)
        return pd.Series(pca.fit_transform(X).flatten(), index=vm.index)

    early_pc1 = get_pc1(early_matrix)
    late_pc1 = get_pc1(late_matrix)

    # Align signs
    common = early_pc1.index.intersection(late_pc1.index)
    if np.corrcoef(early_pc1[common], late_pc1[common])[0, 1] < 0:
        late_pc1 = -late_pc1

    comparison = pd.DataFrame({
        "early_pc1": early_pc1[common],
        "late_pc1": late_pc1[common],
        "drift": late_pc1[common] - early_pc1[common],
    })
    comparison = comparison.join(legislator_meta[["party", "chamber"]])

    return comparison
```

## Interpretation Guide

- **Stable trajectories** (flat lines): Legislators' positions don't change much within a session. This is the normal finding.
- **Widening partisan gap**: The parties are voting more differently as the session progresses (often happens as easy/bipartisan bills are exhausted and controversial ones remain).
- **Narrowing gap at session end**: May reflect end-of-session deal-making and compromise.
- **Individual spikes**: A legislator whose trajectory jumps may have experienced a specific event (committee assignment change, leadership challenge, constituent pressure).

## Limitations

- **Single-session signal is weak.** True ideological drift is a multi-year phenomenon. Within-session "drift" may actually be changes in the *agenda* (which bills are voted on) rather than changes in *legislators*.
- **PCA sign flips** between windows can create artificial discontinuities. Always align signs.
- **Window size matters.** Too small = noisy estimates. Too large = blurs real changes. Sensitivity analysis is essential.

## Kansas-Specific Considerations

- **Expect limited drift** within a single session. The more interesting finding may be agenda effects: the composition of bills changes across the session, and the *apparent* ideal points shift accordingly.
- **The veto override period** (typically late-session) may show a distinctive pattern as coalitions form around governor vetoes.
- **Compare with time-varying Rice Index** (from `05_IDX_rice_index`). If party cohesion drops at the same time ideal point drift increases, it confirms real behavioral change rather than agenda effects.

## Feasibility Assessment

- **Data size**: ~500 roll calls across ~11 months = marginal for drift detection
- **Compute time**: Seconds per window, minutes total
- **Libraries**: `scikit-learn`, `matplotlib`
- **Difficulty**: Medium (computation), High (interpretation)

## Key References

- Martin, Andrew D., and Kevin M. Quinn. "Dynamic Ideal Point Estimation via Markov Chain Monte Carlo for the U.S. Supreme Court, 1953-1999." *Political Analysis* 10(2), 2002.
- Bailey, Michael A. "Comparable Preference Estimates across Time and Institutions for the Court, Congress, and Presidency." *American Journal of Political Science* 51(3), 2007.
