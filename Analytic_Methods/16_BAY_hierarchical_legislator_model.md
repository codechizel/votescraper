# Hierarchical Bayesian Model: Legislator Nested in Party Nested in Chamber

**Category:** Bayesian Methods
**Prerequisites:** `14_BAY_beta_binomial_party_loyalty`, `15_BAY_irt_ideal_points`
**Complexity:** High
**Related:** `17_BAY_posterior_predictive_checks`

## What It Measures

The hierarchical Bayesian model places legislators within a nested structure: individual legislators are drawn from party-level distributions, which are themselves drawn from chamber-level distributions. This architecture decomposes the total variation in voting behavior into three interpretable levels:

1. **Between-chamber variation**: Do House and Senate members differ ideologically?
2. **Between-party variation (within chamber)**: How far apart are Republicans and Democrats in each chamber?
3. **Within-party variation**: How much do individual legislators deviate from their party?

The model quantifies each level's contribution and produces ideal point estimates that benefit from *partial pooling* at every level — a legislator with few votes is pulled toward their party mean, which is itself pulled toward the chamber mean.

## Questions It Answers

- How much of the variation in voting is explained by party vs. individual ideology?
- Are the parties more polarized in the House or the Senate?
- Is within-party variation larger for Republicans (who have a supermajority and thus more room for internal disagreement)?
- What fraction of the legislature's ideological landscape is captured by the party label alone?
- How do hierarchical ideal points compare to flat (non-hierarchical) IRT estimates?

## Mathematical Foundation

### Three-Level Hierarchical IRT

**Level 1 — Observation (Vote)**

$$Y_{ij} | \xi_i, \alpha_j, \beta_j \sim \text{Bernoulli}(\text{logit}^{-1}(\beta_j \xi_i - \alpha_j))$$

Same 2PL IRT likelihood as `15_BAY_irt_ideal_points`.

**Level 2 — Legislator (Nested in Party)**

$$\xi_i | \mu_{p[i]}, \sigma_{p[i]} \sim \text{Normal}(\mu_{p[i]}, \sigma_{p[i]})$$

Each legislator's ideal point is drawn from a party-specific distribution. The party mean $\mu_p$ represents the "typical" ideological position for that party, and $\sigma_p$ represents within-party dispersion.

**Level 3 — Party (Nested in Chamber)**

$$\mu_{p} | \mu_{c[p]}, \sigma_c \sim \text{Normal}(\mu_{c[p]}, \sigma_c)$$

Each party's mean is drawn from a chamber-level distribution. This captures systematic differences between chambers.

**Hyperpriors:**

$$\mu_c \sim \text{Normal}(0, 2) \quad \text{(chamber-level mean)}$$
$$\sigma_c \sim \text{HalfNormal}(1) \quad \text{(between-party spread)}$$
$$\sigma_p \sim \text{HalfNormal}(1) \quad \text{(within-party spread)}$$

### Group Structure

For Kansas:

| Group Index | Chamber | Party | Expected $n$ |
|-------------|---------|-------|---------------|
| 0 | House | Republican | ~92 |
| 1 | House | Democrat | ~38 |
| 2 | Senate | Republican | ~32 |
| 3 | Senate | Democrat | ~10 |

### Variance Decomposition

After fitting, compute the fraction of total variance at each level:

$$\text{ICC}_{\text{party}} = \frac{\sigma_c^2}{\sigma_c^2 + \sigma_p^2}$$

$$\text{ICC}_{\text{legislator}} = \frac{\sigma_p^2}{\sigma_c^2 + \sigma_p^2}$$

High $\text{ICC}_{\text{party}}$ means party labels explain most of the variance (strong partisanship).

## Python Implementation with PyMC

```python
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd

def fit_hierarchical_irt(
    data: dict,
    n_samples: int = 2000,
    n_tune: int = 1000,
) -> az.InferenceData:
    """Fit three-level hierarchical IRT model.

    Args:
        data: Output from prepare_irt_data() with additional fields:
            'party_idx': party index per legislator (0-3 for 4 party-chamber groups)
            'chamber_idx': chamber index per party-chamber group (0 or 1)
            'group_labels': list of group labels
    """
    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y = data["y"]
    n_leg = data["n_legislators"]
    n_votes = data["n_votes"]

    # Group indices
    party_idx = data["party_idx"]  # Shape: (n_legislators,), values 0-3
    chamber_idx = data["chamber_idx"]  # Shape: (4,), maps each party-group to chamber

    n_groups = len(np.unique(party_idx))  # 4 (House-R, House-D, Senate-R, Senate-D)
    n_chambers = len(np.unique(chamber_idx))  # 2

    with pm.Model() as model:
        # === Level 3: Chamber-level priors ===
        mu_chamber = pm.Normal("mu_chamber", mu=0, sigma=2, shape=n_chambers)

        # === Level 2: Party-within-chamber ===
        sigma_between_parties = pm.HalfNormal("sigma_between_parties", sigma=1)
        mu_group = pm.Normal(
            "mu_group",
            mu=mu_chamber[chamber_idx],
            sigma=sigma_between_parties,
            shape=n_groups,
        )

        sigma_within_party = pm.HalfNormal("sigma_within_party", sigma=1, shape=n_groups)

        # === Level 1: Individual legislators ===
        xi = pm.Normal(
            "xi",
            mu=mu_group[party_idx],
            sigma=sigma_within_party[party_idx],
            shape=n_leg,
        )

        # === Roll call parameters (same as flat IRT) ===
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes)
        beta = pm.Normal("beta", mu=0, sigma=2.5, shape=n_votes)

        # === Identification constraint ===
        # Use an ordering constraint: mu_group[republican] > mu_group[democrat]
        # Or fix one legislator — here we use soft constraint via prior
        order = pm.Potential(
            "order_constraint",
            pm.math.switch(mu_group[0] > mu_group[1], 0, -1000)
            + pm.math.switch(mu_group[2] > mu_group[3], 0, -1000),
        )

        # === Likelihood ===
        eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
        obs = pm.Bernoulli("obs", logit_p=eta, observed=y)

        # === Derived quantities ===
        party_spread = pm.Deterministic(
            "party_spread",
            mu_group[0] - mu_group[1],  # House: Rep - Dem
        )
        senate_spread = pm.Deterministic(
            "senate_spread",
            mu_group[2] - mu_group[3],  # Senate: Rep - Dem
        )

        # === Sample ===
        trace = pm.sample(
            n_samples,
            tune=n_tune,
            cores=2,
            target_accept=0.95,  # Higher for hierarchical models
            random_seed=42,
        )

    return az.from_pymc3(trace=trace, model=model)
```

### Data Preparation Extension

```python
def prepare_hierarchical_data(
    vote_matrix: pd.DataFrame,
    legislator_meta: pd.DataFrame,
) -> dict:
    """Extend IRT data preparation with hierarchical group indices."""
    # Start with standard IRT data preparation
    data = prepare_irt_data(vote_matrix, legislator_meta)

    meta = data["legislator_meta"]

    # Create party-chamber group indices
    # 0: House-Republican, 1: House-Democrat, 2: Senate-Republican, 3: Senate-Democrat
    group_map = {
        ("House", "Republican"): 0,
        ("House", "Democrat"): 1,
        ("Senate", "Republican"): 2,
        ("Senate", "Democrat"): 3,
    }

    party_idx = np.array([
        group_map[(meta.loc[slug, "chamber"], meta.loc[slug, "party"])]
        for slug in data["leg_slugs"]
    ])

    # Map groups to chambers
    chamber_idx = np.array([0, 0, 1, 1])  # House=0, Senate=1

    data["party_idx"] = party_idx
    data["chamber_idx"] = chamber_idx
    data["group_labels"] = ["House-R", "House-D", "Senate-R", "Senate-D"]

    return data
```

### Visualization: Group-Level Parameters

```python
import matplotlib.pyplot as plt

def plot_group_posteriors(
    idata: az.InferenceData,
    group_labels: list[str],
    save_path: str | None = None,
):
    """Plot posterior distributions of group-level means."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Group means
    mu_group = idata.posterior["mu_group"]
    group_colors = ["#E81B23", "#0015BC", "#E81B23", "#0015BC"]  # R, D, R, D
    group_styles = ["-", "-", "--", "--"]  # House solid, Senate dashed

    for g in range(4):
        samples = mu_group.sel(mu_group_dim_0=g).values.flatten()
        axes[0].hist(
            samples, bins=50, alpha=0.5,
            color=group_colors[g], linestyle=group_styles[g],
            label=group_labels[g], density=True,
        )

    axes[0].set_xlabel("Group Mean Ideal Point")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Posterior Distributions of Party-Chamber Means")
    axes[0].legend()

    # Within-party standard deviations
    sigma_wp = idata.posterior["sigma_within_party"]
    for g in range(4):
        samples = sigma_wp.sel(sigma_within_party_dim_0=g).values.flatten()
        axes[1].hist(
            samples, bins=50, alpha=0.5,
            color=group_colors[g],
            label=group_labels[g], density=True,
        )

    axes[1].set_xlabel("Within-Party Standard Deviation")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Within-Party Ideological Dispersion")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)


def plot_variance_decomposition(
    idata: az.InferenceData,
    save_path: str | None = None,
):
    """Pie chart / bar chart of variance at each hierarchical level."""
    sigma_between = idata.posterior["sigma_between_parties"].values.flatten()
    sigma_within = idata.posterior["sigma_within_party"].mean(dim="sigma_within_party_dim_0").values.flatten()

    # Variance proportions
    var_between = sigma_between ** 2
    var_within = sigma_within ** 2
    total_var = var_between + var_within

    pct_between = (var_between / total_var).mean()
    pct_within = (var_within / total_var).mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        ["Between Parties\n(within chamber)", "Within Party\n(individual variation)"],
        [pct_between, pct_within],
        color=["#ff7f0e", "#2ca02c"],
        edgecolor="black",
    )
    ax.set_ylabel("Fraction of Total Variance")
    ax.set_title("Variance Decomposition: How Much Does Party Explain?")
    ax.set_ylim(0, 1)

    for i, pct in enumerate([pct_between, pct_within]):
        ax.text(i, pct + 0.02, f"{pct:.1%}", ha="center", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

### Shrinkage Comparison

```python
def compare_flat_vs_hierarchical(
    flat_summary: pd.DataFrame,
    hier_summary: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    save_path: str | None = None,
):
    """Compare ideal points from flat vs. hierarchical IRT."""
    merged = flat_summary[["mean"]].rename(columns={"mean": "flat_mean"}).join(
        hier_summary[["mean"]].rename(columns={"mean": "hier_mean"}),
    )
    merged = merged.join(legislator_meta[["party"]])

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = merged["party"].map({"Republican": "#E81B23", "Democrat": "#0015BC"})

    ax.scatter(merged["flat_mean"], merged["hier_mean"], c=colors, s=30, alpha=0.6)
    ax.plot([-2, 2], [-2, 2], "k--", alpha=0.3, label="1:1 line")

    ax.set_xlabel("Flat IRT Ideal Point")
    ax.set_ylabel("Hierarchical IRT Ideal Point")
    ax.set_title("Effect of Hierarchical Shrinkage on Ideal Points")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

## Interpretation Guide

### Group-Level Parameters

- **$\mu_{\text{group}}$ (party-chamber means)**: The four group means show the average ideological position of each party-chamber combination. Expect: House-R > Senate-R > Senate-D > House-D (or similar ordering depending on sign convention).
- **$\sigma_{\text{between parties}}$**: How far apart the parties are (controlling for chamber). Large = polarized. Small = overlapping parties.
- **$\sigma_{\text{within party}}$**: How much individual legislators vary within their party. Large = ideologically diverse party. Small = disciplined party.

### Variance Decomposition

| Partition | High (>70%) | Low (<30%) |
|-----------|-------------|------------|
| Between-party | Strong partisanship; party label is highly predictive | Weak parties; individual ideology dominates |
| Within-party | Diverse/fractured parties | Disciplined parties |

For Kansas, expect between-party variance to dominate (60-80%), but with meaningful within-party variation for Republicans due to the moderate-conservative divide.

### Hierarchical Shrinkage Effects

Compared to flat IRT:
- **Small-group members shrink more**: Senate Democrats (n=10) will be pulled more toward the chamber mean than House Republicans (n=92).
- **Mavericks shrink toward party mean**: A Republican who votes like a Democrat will have their ideal point pulled back toward the Republican mean. The amount of pull depends on how many votes they cast.
- **This is desirable**: Shrinkage reflects our genuine prior uncertainty about legislators with limited or unusual voting records.

## Advantages Over Flat IRT

1. **Principled shrinkage**: Legislators with few votes get sensible estimates instead of noisy ones
2. **Variance decomposition**: Quantifies the contribution of party vs. individual ideology
3. **Better calibrated uncertainty**: Credible intervals account for group-level uncertainty
4. **Interpretable group parameters**: Direct estimates of party positions and within-party dispersion

## Limitations

1. **More complex model**: More parameters, harder to diagnose convergence
2. **Stronger assumptions**: Assumes normal distributions at each level
3. **Slower MCMC**: More parameters to sample. Expect 15-30 minutes.
4. **Identification**: More constraints needed than flat IRT
5. **Only 4 groups**: With only 4 party-chamber combinations, the chamber-level hyperparameters are weakly identified. Consider dropping the chamber level and using just 4 group-specific priors.

## Kansas-Specific Considerations

- **The model structure perfectly matches the legislature**: 2 chambers x 2 parties = 4 natural groups, each with enough members (10-92) for stable group-level estimation.
- **The 10 Senate Democrats are the most affected by shrinkage.** Their ideal points will be pulled toward the Senate mean more than other groups. This may underestimate how liberal they actually are.
- **Republican within-party sigma is the key finding.** If it's significantly larger than Democrat within-party sigma, it confirms the moderate-conservative split within the Republican majority.
- **Compare party_spread (House) vs. senate_spread (Senate)**: If the Senate is less polarized (smaller spread), it may reflect the Senate's more deliberative culture or different district compositions.

## Feasibility Assessment

- **Data size**: 170 legislators in 4 groups = ideal for hierarchical modeling
- **Compute time**: 15-30 minutes for full MCMC
- **Libraries**: `pymc`, `arviz`
- **Difficulty**: High (requires understanding of hierarchical models, identification, MCMC diagnostics)

## Key References

- Gelman, Andrew, and Jennifer Hill. *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press, 2006.
- Goet, Niels D. "Explaining Differences in Voting Patterns Across Voting Domains Using Hierarchical Bayesian Models." *Political Analysis*, 2025. https://arxiv.org/abs/2312.15049
- Jackman, Simon. "Multidimensional Analysis of Roll Call Data via Bayesian Simulation." *Political Analysis* 9(3), 2001.
- Martin, Andrew D., and Kevin M. Quinn. "Dynamic Ideal Point Estimation via Markov Chain Monte Carlo for the U.S. Supreme Court, 1953-1999." *Political Analysis* 10(2), 2002.
