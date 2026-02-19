# Posterior Predictive Checks and Bayesian Model Comparison

**Category:** Bayesian Methods
**Prerequisites:** `15_BAY_irt_ideal_points` and/or `16_BAY_hierarchical_legislator_model`
**Complexity:** Medium-High

## What It Measures

Posterior predictive checks (PPCs) answer the question: *does this model generate data that looks like the real data?* After fitting a Bayesian model, we simulate new datasets from the posterior and compare them to the observed data. If the simulated data systematically differs from reality, the model is misspecified.

Bayesian model comparison (using WAIC, LOO-CV, or Bayes factors) determines which model best balances fit and complexity. Is a 1D IRT model sufficient, or does 2D add meaningful information? Is the hierarchical model better than the flat model?

These are not separate analyses — they are essential validation steps for any Bayesian model fitted in `15_BAY_irt_ideal_points` or `16_BAY_hierarchical_legislator_model`.

## Questions It Answers

- Does the fitted IRT model reproduce the observed patterns of partisanship?
- Can the model reproduce the observed distribution of vote margins?
- Does the model correctly predict party unity and Rice index distributions?
- Is a 1D model sufficient, or does adding a second dimension meaningfully improve fit?
- Is the hierarchical model better than the flat model (after accounting for complexity)?
- Which individual votes or legislators are poorly captured by the model?

## Mathematical Foundation

### Posterior Predictive Distribution

Given the posterior $p(\theta | y)$, the posterior predictive distribution for new data $\tilde{y}$ is:

$$p(\tilde{y} | y) = \int p(\tilde{y} | \theta) p(\theta | y) d\theta$$

In practice, we approximate this by:
1. Drawing $\theta^{(s)} \sim p(\theta | y)$ for $s = 1, \ldots, S$ (MCMC samples)
2. For each draw, simulating $\tilde{y}^{(s)} \sim p(\tilde{y} | \theta^{(s)})$

The collection $\{\tilde{y}^{(s)}\}$ is the posterior predictive sample.

### Test Statistics for Legislative Voting Models

A test statistic $T(y)$ summarizes a pattern in the data. We compare $T(y_{\text{observed}})$ to the distribution of $T(\tilde{y}^{(s)})$:

| Test Statistic | What It Checks | Formula |
|---------------|----------------|---------|
| Overall Yea rate | Base rate calibration | $\text{mean}(y)$ |
| Per-legislator Yea rate | Individual-level calibration | $\text{mean}(y_i)$ for each $i$ |
| Per-roll-call Yea rate | Vote-level calibration | $\text{mean}(y_j)$ for each $j$ |
| Rice Index distribution | Party cohesion patterns | See `05_IDX_rice_index` |
| Party Unity Score dist. | Individual loyalty patterns | See `06_IDX_party_unity_scores` |
| Vote margin distribution | Contestedness patterns | $|\text{Yea}_j - \text{Nay}_j| / \text{Total}_j$ |
| Classification accuracy | Predictive power | $\text{mean}(\hat{y} = y)$ |

### Bayesian p-value

$$p_B = P(T(\tilde{y}) \geq T(y_{\text{obs}}) | y_{\text{obs}})$$

Values near 0 or 1 indicate model misspecification. Values near 0.5 indicate the model generates data consistent with the observation.

### WAIC (Widely Applicable Information Criterion)

$$\text{WAIC} = -2 \left[\text{lppd} - p_{\text{WAIC}}\right]$$

Where:
- $\text{lppd}$ = log pointwise predictive density (model fit)
- $p_{\text{WAIC}}$ = effective number of parameters (complexity penalty)

Lower WAIC is better. The difference in WAIC between models, relative to the standard error, determines whether one model is significantly better.

### LOO-CV (Leave-One-Out Cross-Validation)

Approximate LOO-CV using Pareto-smoothed importance sampling (PSIS):

$$\text{elpd}_{\text{loo}} = \sum_{i=1}^{n} \log p(y_i | y_{-i})$$

ArviZ provides `az.loo()` which implements PSIS-LOO. High Pareto $k$ values (>0.7) indicate observations that are influential and may need exact LOO computation.

## Python Implementation

### Generating Posterior Predictive Samples

```python
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_ppc(model, trace) -> az.InferenceData:
    """Generate posterior predictive samples."""
    with model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)
    return az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model)
```

### PPC: Overall Yea Rate

```python
def check_overall_yea_rate(idata: az.InferenceData, y_obs: np.ndarray, save_path: str | None = None):
    """Check if model reproduces the overall Yea rate."""
    ppc_samples = idata.posterior_predictive["obs"].values  # Shape: (chains, draws, n_obs)
    simulated_rates = ppc_samples.mean(axis=-1).flatten()
    observed_rate = y_obs.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(simulated_rates, bins=50, density=True, alpha=0.7, label="Posterior predictive")
    ax.axvline(observed_rate, color="red", linewidth=2, label=f"Observed: {observed_rate:.3f}")
    ax.set_xlabel("Overall Yea Rate")
    ax.set_ylabel("Density")
    ax.set_title("PPC: Overall Yea Rate")
    ax.legend()

    p_value = (simulated_rates >= observed_rate).mean()
    ax.text(0.05, 0.95, f"Bayesian p = {p_value:.3f}", transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

### PPC: Party Unity Scores

```python
def check_party_unity_ppc(
    idata: az.InferenceData,
    data: dict,
    y_obs: np.ndarray,
    save_path: str | None = None,
):
    """Check if model reproduces the distribution of party unity scores."""
    ppc_samples = idata.posterior_predictive["obs"].values
    meta = data["legislator_meta"]
    leg_idx = data["leg_idx"]

    # Compute unity scores for observed and simulated data
    def compute_unity_from_votes(y, leg_idx, meta, leg_slugs):
        """Simplified unity computation from vote array."""
        # Group votes by legislator
        df = pd.DataFrame({"leg_idx": leg_idx, "vote": y})
        leg_rates = df.groupby("leg_idx")["vote"].mean()

        parties = [meta.loc[leg_slugs[i], "party"] for i in leg_rates.index]
        unity_by_party = {}
        for party in ["Republican", "Democrat"]:
            party_mask = [p == party for p in parties]
            if any(party_mask):
                party_rates = leg_rates[[i for i, m in zip(leg_rates.index, party_mask) if m]]
                unity_by_party[party] = party_rates.mean()
        return unity_by_party

    # Compute for many posterior predictive draws
    n_draws = min(200, ppc_samples.shape[0] * ppc_samples.shape[1])
    flat_ppc = ppc_samples.reshape(-1, ppc_samples.shape[-1])[:n_draws]

    sim_unity_r = [
        compute_unity_from_votes(draw, leg_idx, meta, data["leg_slugs"]).get("Republican", np.nan)
        for draw in flat_ppc
    ]

    obs_unity_r = compute_unity_from_votes(y_obs, leg_idx, meta, data["leg_slugs"]).get("Republican", np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sim_unity_r, bins=40, density=True, alpha=0.7, label="Simulated")
    ax.axvline(obs_unity_r, color="red", linewidth=2, label=f"Observed: {obs_unity_r:.3f}")
    ax.set_title("PPC: Republican Average Yea Rate")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

### PPC: Vote Margin Distribution

```python
def check_vote_margins_ppc(
    idata: az.InferenceData,
    data: dict,
    y_obs: np.ndarray,
    save_path: str | None = None,
):
    """Check if model reproduces the distribution of vote margins."""
    ppc_samples = idata.posterior_predictive["obs"].values
    vote_idx = data["vote_idx"]
    n_votes = data["n_votes"]

    # Observed margins
    obs_margins = []
    for j in range(n_votes):
        mask = vote_idx == j
        if mask.sum() > 0:
            yea_pct = y_obs[mask].mean()
            obs_margins.append(abs(2 * yea_pct - 1))

    # Simulated margins (average over draws)
    flat_ppc = ppc_samples.reshape(-1, ppc_samples.shape[-1])[:100]
    sim_margins_all = []
    for draw in flat_ppc:
        for j in range(n_votes):
            mask = vote_idx == j
            if mask.sum() > 0:
                yea_pct = draw[mask].mean()
                sim_margins_all.append(abs(2 * yea_pct - 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(obs_margins, bins=30, density=True, alpha=0.5, label="Observed", color="red")
    ax.hist(sim_margins_all, bins=30, density=True, alpha=0.3, label="Simulated (PPD)", color="blue")
    ax.set_xlabel("Vote Margin")
    ax.set_ylabel("Density")
    ax.set_title("PPC: Vote Margin Distribution")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

### Model Comparison

```python
def compare_models(
    models: dict[str, az.InferenceData],
    save_path: str | None = None,
) -> pd.DataFrame:
    """Compare models using LOO-CV.

    Args:
        models: Dict mapping model names to InferenceData objects.
            E.g., {"1D_flat": idata_1d, "2D_flat": idata_2d, "1D_hier": idata_hier}
    """
    comparison = az.compare(models, ic="loo")
    print(comparison)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    az.plot_compare(comparison, ax=ax)
    ax.set_title("Model Comparison (LOO-CV)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)

    return comparison
```

### Identifying Problematic Observations

```python
def find_surprising_votes(
    idata: az.InferenceData,
    data: dict,
    threshold: float = 0.1,
) -> pd.DataFrame:
    """Find votes where the model strongly disagrees with reality.

    These are votes the model assigns low probability to the observed outcome.
    """
    # Log-likelihood per observation
    log_lik = idata.log_likelihood["obs"].values  # (chains, draws, n_obs)
    mean_ll = log_lik.mean(axis=(0, 1))  # (n_obs,)

    # Low log-likelihood = model surprised by observation
    surprising = np.where(mean_ll < np.log(threshold))[0]

    results = []
    for idx in surprising:
        results.append({
            "obs_idx": idx,
            "legislator": data["leg_slugs"][data["leg_idx"][idx]],
            "vote_id": data["vote_ids"][data["vote_idx"][idx]],
            "observed_vote": data["y"][idx],
            "mean_log_lik": mean_ll[idx],
            "predicted_prob": np.exp(mean_ll[idx]),
        })

    return pd.DataFrame(results).sort_values("mean_log_lik")
```

## Interpretation Guide

### PPC Failures and What They Mean

| PPC Check | Fails If | Implies |
|-----------|----------|---------|
| Overall Yea rate off | Model systematically biased | Intercept/difficulty parameters misspecified |
| Per-legislator rates wrong | Individual ideal points poorly estimated | Need more data or different priors |
| Vote margin dist. wrong | Model can't reproduce contestedness | Missing dimension or non-spatial dynamics |
| Rice Index wrong | Party cohesion poorly captured | Party structure not adequately modeled |
| Classification accuracy low | Spatial model poor fit | Consider more dimensions or non-spatial model |

### Model Comparison Rules of Thumb

| ELPD Difference | SE of Difference | Interpretation |
|----------------|-----------------|----------------|
| > 2 SE | — | Clearly better model |
| 1-2 SE | — | Probably better, but not definitive |
| < 1 SE | — | Models are essentially equivalent (prefer simpler) |

### Expected Results for Kansas

- **1D flat IRT** should achieve 85-90% classification accuracy
- **2D flat IRT** should add 2-5% accuracy (marginal improvement)
- **Hierarchical IRT** should have better WAIC/LOO than flat IRT, especially for legislators with few votes
- **PPCs should show**: Good overall calibration, slightly too little variance in simulated margins (a common issue with 1D models in multi-dimensional legislatures)

## Kansas-Specific Considerations

- **Run all three model variants** (1D flat, 1D hierarchical, 2D flat) and compare. The comparison itself is informative about the legislature's dimensionality.
- **Focus PPCs on the Republican party**: Since this is where the interesting intra-party variation exists, check whether the model correctly reproduces within-Republican variation.
- **Veto override votes may be model outliers**: These votes have unusual dynamics (cross-party coalitions) that a simple ideological model may not capture.

## Feasibility Assessment

- **Compute time**: PPCs add a few minutes on top of the model fitting time. Model comparison requires fitting multiple models.
- **Libraries**: `pymc`, `arviz` (both already needed for the IRT models)
- **Difficulty**: Medium. The concepts are straightforward; the interpretation requires judgment.

## Key References

- Gelman, Andrew, Xiao-Li Meng, and Hal Stern. "Posterior Predictive Assessment of Model Fitness Via Realized Discrepancies." *Statistica Sinica* 6, 1996.
- Vehtari, Aki, Andrew Gelman, and Jonah Gabry. "Practical Bayesian Model Evaluation Using Leave-One-Out Cross-Validation and WAIC." *Statistics and Computing* 27(5), 2017.
- ArviZ documentation on model comparison: https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/model_comparison.html
- ArviZ posterior predictive checks: https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html
