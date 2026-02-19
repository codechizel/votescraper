# Beta-Binomial Bayesian Party Loyalty Model

**Category:** Bayesian Methods
**Prerequisites:** `01_DATA_vote_matrix_construction`, `06_IDX_party_unity_scores`
**Complexity:** Medium
**Related:** `16_BAY_hierarchical_legislator_model`

## What It Measures

The Beta-Binomial model estimates each legislator's "true" party loyalty rate using Bayesian inference. Unlike raw party unity scores (which can be noisy for legislators with few votes), the Bayesian model uses *partial pooling* — legislators with limited voting records are "shrunk" toward their party's average loyalty rate. This produces more reliable estimates, especially for legislators who missed many votes or joined mid-session.

This is the natural first Bayesian model for legislative data because the Beta-Binomial conjugacy allows closed-form posterior computation without MCMC sampling.

## Questions It Answers

- What is the posterior distribution of each legislator's "true" loyalty rate?
- How much uncertainty do we have about each legislator's loyalty?
- Which legislators have unusually low or high loyalty after accounting for sample size?
- How cohesive is each party in terms of the distribution of loyalty across members?
- Are there meaningful differences between House and Senate loyalty patterns?

## Mathematical Foundation

### The Model

For each legislator $i$ in party $p$:

**Likelihood:**
$$y_i | \theta_i \sim \text{Binomial}(n_i, \theta_i)$$

Where:
- $y_i$ = number of party-line votes (votes matching party majority position)
- $n_i$ = total party votes where legislator $i$ was present and voted Yea/Nay
- $\theta_i$ = legislator's true loyalty probability

**Prior (Party-Level):**
$$\theta_i | \alpha_p, \beta_p \sim \text{Beta}(\alpha_p, \beta_p)$$

The Beta distribution parameterizes the party-level distribution of loyalty rates. $\alpha_p$ and $\beta_p$ encode the party's average loyalty and the spread of loyalty across members.

**Hyperpriors:**
$$\alpha_p, \beta_p \sim \text{prior}$$

### Conjugate Posterior (Closed Form)

Because the Beta distribution is conjugate to the Binomial likelihood, the posterior is available in closed form:

$$\theta_i | y_i, n_i, \alpha_p, \beta_p \sim \text{Beta}(\alpha_p + y_i, \beta_p + n_i - y_i)$$

**Posterior mean (point estimate):**
$$\hat{\theta}_i = \frac{\alpha_p + y_i}{\alpha_p + \beta_p + n_i}$$

This is a weighted average of the prior mean $\frac{\alpha_p}{\alpha_p + \beta_p}$ and the observed rate $\frac{y_i}{n_i}$, with the weight depending on sample size $n_i$.

### Shrinkage

The key insight: when $n_i$ is small (few votes), the posterior is dominated by the prior (shrinks toward party average). When $n_i$ is large, the posterior approaches the observed rate. This is exactly what we want — uncertain estimates get pulled toward the group mean.

**Shrinkage factor:**
$$\lambda_i = \frac{n_i}{n_i + \alpha_p + \beta_p}$$

$$\hat{\theta}_i = \lambda_i \cdot \frac{y_i}{n_i} + (1 - \lambda_i) \cdot \frac{\alpha_p}{\alpha_p + \beta_p}$$

### Reparameterization

It's often more intuitive to parameterize the Beta in terms of mean $\mu$ and concentration $\kappa$:

$$\alpha = \mu \cdot \kappa, \quad \beta = (1 - \mu) \cdot \kappa$$

Where:
- $\mu = \frac{\alpha}{\alpha + \beta}$ = average party loyalty rate
- $\kappa = \alpha + \beta$ = concentration (higher = less within-party variance)

## Python Implementation

### Empirical Bayes (Method of Moments)

The simplest approach: estimate $\alpha_p, \beta_p$ from the data, then compute individual posteriors.

```python
import pandas as pd
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.optimize import minimize_scalar

def estimate_beta_params(
    success_counts: np.ndarray,
    total_counts: np.ndarray,
) -> tuple[float, float]:
    """Estimate Beta distribution parameters from observed rates using method of moments.

    Args:
        success_counts: Array of y_i (votes with party).
        total_counts: Array of n_i (total party votes present).

    Returns:
        (alpha, beta) parameters of the Beta prior.
    """
    rates = success_counts / total_counts
    mu = rates.mean()
    var = rates.var()

    # Method of moments for Beta(alpha, beta)
    # mu = alpha / (alpha + beta)
    # var = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))
    if var >= mu * (1 - mu):
        # Variance too high for Beta — use weakly informative prior
        return (1.0, 1.0)

    common = mu * (1 - mu) / var - 1
    alpha = mu * common
    beta = (1 - mu) * common

    return (max(alpha, 0.5), max(beta, 0.5))


def compute_bayesian_loyalty(
    unity_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Bayesian party loyalty estimates with shrinkage.

    Args:
        unity_data: DataFrame with columns: legislator_slug, party, chamber,
                    votes_with_party, party_votes_present.

    Returns:
        DataFrame with posterior mean, credible intervals, and shrinkage.
    """
    results = []

    for party in ["Republican", "Democrat"]:
        party_data = unity_data[unity_data["party"] == party].copy()

        # Filter to legislators with at least a few votes
        party_data = party_data[party_data["party_votes_present"] >= 3]

        y = party_data["votes_with_party"].values
        n = party_data["party_votes_present"].values

        # Estimate Beta prior from data (empirical Bayes)
        alpha_prior, beta_prior = estimate_beta_params(y, n)

        for _, row in party_data.iterrows():
            yi = row["votes_with_party"]
            ni = row["party_votes_present"]

            # Posterior parameters
            alpha_post = alpha_prior + yi
            beta_post = beta_prior + ni - yi

            # Posterior statistics
            post_mean = alpha_post / (alpha_post + beta_post)
            post_median = beta_dist.median(alpha_post, beta_post)
            ci_low, ci_high = beta_dist.ppf([0.025, 0.975], alpha_post, beta_post)

            # Raw rate for comparison
            raw_rate = yi / ni if ni > 0 else np.nan

            # Shrinkage
            shrinkage = (alpha_prior + beta_prior) / (alpha_prior + beta_prior + ni)

            results.append({
                "legislator_slug": row["legislator_slug"],
                "party": party,
                "chamber": row.get("chamber", ""),
                "raw_loyalty": raw_rate,
                "posterior_mean": post_mean,
                "posterior_median": post_median,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "credible_interval_width": ci_high - ci_low,
                "shrinkage": shrinkage,
                "n_party_votes": ni,
                "alpha_prior": alpha_prior,
                "beta_prior": beta_prior,
            })

    return pd.DataFrame(results).sort_values("posterior_mean")
```

### Full Bayesian with PyMC (Hierarchical)

For a fully Bayesian treatment where hyperparameters also have posteriors:

```python
import pymc as pm
import arviz as az

def fit_beta_binomial_pymc(
    unity_data: pd.DataFrame,
    party: str = "Republican",
) -> az.InferenceData:
    """Fit hierarchical Beta-Binomial model with PyMC.

    This gives full posterior distributions for all parameters,
    including the party-level hyperparameters.
    """
    party_data = unity_data[
        (unity_data["party"] == party) & (unity_data["party_votes_present"] >= 3)
    ].reset_index(drop=True)

    y = party_data["votes_with_party"].values.astype(int)
    n = party_data["party_votes_present"].values.astype(int)
    n_legislators = len(party_data)

    with pm.Model() as model:
        # Hyperpriors on party-level loyalty distribution
        # Parameterize as mean and concentration
        mu = pm.Beta("mu", alpha=2, beta=2)  # Party mean loyalty
        kappa = pm.Gamma("kappa", alpha=2, beta=0.1)  # Concentration

        # Individual legislator loyalty rates
        theta = pm.Beta(
            "theta",
            alpha=mu * kappa,
            beta=(1 - mu) * kappa,
            shape=n_legislators,
        )

        # Likelihood
        obs = pm.Binomial("obs", n=n, p=theta, observed=y)

        # Sample
        trace = pm.sample(2000, tune=1000, cores=2, random_seed=42)
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)

    # Combine into InferenceData
    idata = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model)

    # Add legislator slugs to coordinates
    idata.posterior = idata.posterior.assign_coords(
        theta_dim_0=party_data["legislator_slug"].values
    )

    return idata
```

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_shrinkage(loyalty_df: pd.DataFrame, party: str, save_path: str | None = None):
    """Plot raw vs. Bayesian loyalty estimates to show shrinkage."""
    data = loyalty_df[loyalty_df["party"] == party].copy()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Arrows from raw to posterior
    for _, row in data.iterrows():
        ax.annotate(
            "",
            xy=(row["posterior_mean"], row["n_party_votes"]),
            xytext=(row["raw_loyalty"], row["n_party_votes"]),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )

    # Raw estimates
    ax.scatter(data["raw_loyalty"], data["n_party_votes"],
              marker="o", s=30, color="red", alpha=0.5, label="Raw rate")
    # Posterior estimates
    ax.scatter(data["posterior_mean"], data["n_party_votes"],
              marker="o", s=30, color="blue", alpha=0.5, label="Bayesian estimate")

    ax.set_xlabel("Loyalty Rate")
    ax.set_ylabel("Number of Party Votes (sample size)")
    ax.set_title(f"{party} Party Loyalty: Raw vs. Bayesian Estimates")
    ax.legend()

    # Note: shrinkage arrows should point toward the party mean
    party_mean = data["posterior_mean"].mean()
    ax.axvline(party_mean, color="green", linestyle="--", alpha=0.3, label="Party mean")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)


def plot_posterior_distributions(
    loyalty_df: pd.DataFrame,
    slugs: list[str],
    save_path: str | None = None,
):
    """Plot posterior Beta distributions for selected legislators."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.linspace(0, 1, 200)

    for slug in slugs:
        row = loyalty_df[loyalty_df["legislator_slug"] == slug].iloc[0]
        a = row["alpha_prior"] + row["n_party_votes"] * row["raw_loyalty"]
        b = row["alpha_prior"] + row["beta_prior"] + row["n_party_votes"] - a + row["alpha_prior"]
        # Recalculate properly
        a_post = row["alpha_prior"] + int(row["n_party_votes"] * row["raw_loyalty"])
        b_post = row["beta_prior"] + row["n_party_votes"] - int(row["n_party_votes"] * row["raw_loyalty"])

        y = beta_dist.pdf(x, a_post, b_post)
        label = slug.split("_")[1].title()
        ax.plot(x, y, label=f"{label} (n={int(row['n_party_votes'])})")

    ax.set_xlabel("Loyalty Rate")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Distributions of Party Loyalty")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

## Interpretation Guide

### Reading the Results

- **Posterior mean vs. raw rate**: The difference is the shrinkage effect. Legislators with few votes are pulled more toward the party mean.
- **Credible interval width**: Reflects uncertainty. Wide intervals = unreliable estimate (few votes). Narrow intervals = reliable (many votes).
- **Shrinkage factor**: 0 = no shrinkage (many votes), 1 = fully shrunk to prior (no votes). Values around 0.1-0.3 are typical.

### What Shrinkage Looks Like

| Legislator | Raw Rate | n Votes | Posterior Mean | Interpretation |
|-----------|----------|---------|----------------|----------------|
| A | 0.60 | 5 | 0.82 | Few votes, heavily shrunk toward party mean |
| B | 0.90 | 50 | 0.89 | Many votes, minimal shrinkage |
| C | 0.95 | 100 | 0.94 | Lots of data, posterior ≈ raw rate |
| D | 1.00 | 8 | 0.91 | Perfect record, but shrunk due to small sample |

### When Empirical Bayes vs. Full Bayes Matters

- **Empirical Bayes** (method of moments): Faster, no MCMC needed, adequate for most purposes. Underestimates uncertainty in the hyperparameters.
- **Full Bayes** (PyMC): Propagates uncertainty through all levels. Better for formal inference and model comparison. Takes a few minutes to run.

For exploratory analysis, empirical Bayes is sufficient. For publishable results, use full Bayes.

## Kansas-Specific Considerations

- **Particularly valuable for the Senate.** With only 42 senators voting on ~300-400 party votes, the per-senator sample sizes can vary widely. Bayesian shrinkage stabilizes estimates.
- **The Republican prior will have high mean loyalty** (probably 0.88-0.92) with moderate concentration. The Democratic prior may have slightly lower concentration due to fewer members and more variable voting.
- **Shrinkage reveals "true" mavericks.** After shrinkage, legislators who still have low posterior loyalty are reliably independent — not just victims of small samples.
- **Consider chamber-specific priors.** Run the model separately for House Republicans, Senate Republicans, House Democrats, Senate Democrats. The four groups may have meaningfully different loyalty distributions.

## Feasibility Assessment

- **Data size**: 172 legislators = trivial
- **Compute time**: Empirical Bayes — milliseconds. Full Bayes (PyMC) — 1-3 minutes.
- **Libraries**: `scipy.stats` for empirical Bayes, `pymc` + `arviz` for full Bayes
- **Difficulty**: Low (empirical Bayes), Medium (full Bayes)

## Key References

- Gelman, Andrew, et al. *Bayesian Data Analysis*. 3rd ed. CRC Press, 2013. Chapter 5 (hierarchical models).
- Johnson, Alicia A., Miles Q. Ott, and Mine Dogucu. *Bayes Rules! An Introduction to Applied Bayesian Modeling*. CRC Press, 2022. Chapter 3 (Beta-Binomial). https://www.bayesrulesbook.com/chapter-3
- Efron, Bradley, and Carl Morris. "Stein's Estimation Rule and Its Competitors — An Empirical Bayes Approach." *Journal of the American Statistical Association* 68(341), 1973.
