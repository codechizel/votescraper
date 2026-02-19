# Bayesian IRT Ideal Point Estimation

**Category:** Bayesian Methods
**Prerequisites:** `01_DATA_vote_matrix_construction`, ideally after `09_DIM_principal_component_analysis`
**Complexity:** High
**Related:** `12_DIM_w_nominate`, `14_BAY_beta_binomial_party_loyalty`, `17_BAY_posterior_predictive_checks`

## What It Measures

Bayesian Item Response Theory (IRT) models are the Bayesian counterpart to NOMINATE for estimating legislator ideal points. Instead of maximum likelihood estimation, IRT uses MCMC sampling to produce full posterior distributions for each legislator's ideological position. This means you get not just a point estimate ("this legislator is at position 0.3 on the liberal-conservative spectrum") but a complete probability distribution reflecting uncertainty ("we're 95% sure this legislator is between 0.15 and 0.45").

The Clinton, Jackman, and Rivers (2004) paper establishing this method is one of the most cited in quantitative political science. It demonstrated that Bayesian IRT produces ideal points nearly identical to NOMINATE while adding principled uncertainty quantification.

## Questions It Answers

- Where does each legislator sit on the ideological spectrum, *with uncertainty*?
- For which legislators are we most uncertain about their position?
- How strongly does each roll call discriminate between ideological positions?
- Which bills are "easy" (nearly unanimous) vs. "hard" (highly discriminating)?
- How does the Bayesian posterior compare to PCA and NOMINATE estimates?

## Mathematical Foundation

### The 2-Parameter Logistic (2PL) IRT Model

For legislator $i$ and roll call $j$:

$$P(\text{Yea}_{ij} = 1 | \xi_i, \alpha_j, \beta_j) = \text{logit}^{-1}(\beta_j \xi_i - \alpha_j)$$

Where:
- $\xi_i$ = legislator $i$'s ideal point (the latent "ability" in IRT terminology)
- $\beta_j$ = roll call $j$'s discrimination parameter (how sharply the vote separates liberals from conservatives)
- $\alpha_j$ = roll call $j$'s difficulty parameter (the ideological location of the "tipping point" between Yea and Nay)

**Intuition:**
- High $|\beta_j|$: A strongly partisan vote (e.g., tax policy). Small changes in ideology produce large changes in vote probability.
- Low $|\beta_j|$: A weakly partisan vote (e.g., post office naming). Ideology barely predicts the vote.
- $\alpha_j$: Where on the ideological spectrum the vote "flips" from mostly Nay to mostly Yea.

### Priors

$$\xi_i \sim \text{Normal}(0, 1)$$
$$\alpha_j \sim \text{Normal}(0, \sigma_\alpha^2)$$
$$\beta_j \sim \text{Normal}(0, \sigma_\beta^2)$$

With typical choices:
- $\sigma_\alpha = 5$ (diffuse prior on difficulty)
- $\sigma_\beta = 2.5$ (moderately informative on discrimination)

### Identification Constraints

The model is not identified without constraints because you can:
1. Flip the sign of all $\xi_i$ and all $\beta_j$ (sign invariance)
2. Shift all $\xi_i$ and all $\alpha_j$ by a constant (location invariance)
3. Scale all $\xi_i$ and divide all $\alpha_j, \beta_j$ (scale invariance)

**Standard solutions:**
- **Fix two legislators**: Set a known conservative's ideal point to +1 and a known liberal's to -1
- **Ordered constraint**: $\xi_{\text{conservative}} > \xi_{\text{liberal}}$
- **Prior identification**: Use $\xi_i \sim N(0, 1)$ with a sign constraint on one legislator

### Joint Posterior

$$p(\boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\beta} | \mathbf{Y}) \propto \left[\prod_{i,j: \text{observed}} P(Y_{ij} | \xi_i, \alpha_j, \beta_j)\right] \cdot p(\boldsymbol{\xi}) \cdot p(\boldsymbol{\alpha}) \cdot p(\boldsymbol{\beta})$$

This posterior is not analytically tractable. We sample from it using MCMC (specifically, the No-U-Turn Sampler / NUTS in PyMC).

### Missing Data Handling

One of the key advantages over PCA: missing votes (absences) are handled naturally by simply omitting those $(i, j)$ pairs from the likelihood. No imputation needed.

## Python Implementation with PyMC

### Data Preparation

```python
import pandas as pd
import numpy as np

def prepare_irt_data(
    vote_matrix: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    min_minority_pct: float = 0.025,
    min_votes: int = 20,
) -> dict:
    """Prepare vote data in long format for IRT model.

    Returns dict with arrays for PyMC model.
    """
    # Filter lopsided votes
    yea_pct = vote_matrix.mean(axis=0)
    minority_pct = np.minimum(yea_pct, 1 - yea_pct)
    contested = minority_pct >= min_minority_pct
    vm = vote_matrix.loc[:, contested]

    # Filter low-participation legislators
    vote_counts = vm.notna().sum(axis=1)
    active = vote_counts >= min_votes
    vm = vm.loc[active]

    # Convert to long format (only observed votes)
    long = vm.stack().reset_index()
    long.columns = ["legislator_slug", "vote_id", "vote"]

    # Create integer indices
    leg_slugs = vm.index.tolist()
    vote_ids = vm.columns.tolist()

    long["leg_idx"] = long["legislator_slug"].map({s: i for i, s in enumerate(leg_slugs)})
    long["vote_idx"] = long["vote_id"].map({v: i for i, v in enumerate(vote_ids)})

    return {
        "leg_idx": long["leg_idx"].values.astype(int),
        "vote_idx": long["vote_idx"].values.astype(int),
        "y": long["vote"].values.astype(int),
        "n_legislators": len(leg_slugs),
        "n_votes": len(vote_ids),
        "n_obs": len(long),
        "leg_slugs": leg_slugs,
        "vote_ids": vote_ids,
        "legislator_meta": legislator_meta.loc[leg_slugs],
    }
```

### The PyMC Model

```python
import pymc as pm
import arviz as az

def fit_irt_model(
    data: dict,
    anchor_conservative: str | None = None,
    anchor_liberal: str | None = None,
    n_samples: int = 2000,
    n_tune: int = 1000,
) -> az.InferenceData:
    """Fit 2PL IRT model for ideal point estimation.

    Args:
        data: Output from prepare_irt_data().
        anchor_conservative: Slug of a known conservative (fixed at +1).
        anchor_liberal: Slug of a known liberal (fixed at -1).
        n_samples: MCMC samples per chain.
        n_tune: Tuning samples (discarded).

    Returns:
        ArviZ InferenceData with full posterior.
    """
    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y = data["y"]
    n_leg = data["n_legislators"]
    n_votes = data["n_votes"]
    meta = data["legislator_meta"]

    with pm.Model() as model:
        # --- Legislator ideal points ---
        # Option 1: Fix two anchors
        if anchor_conservative and anchor_liberal:
            cons_idx = data["leg_slugs"].index(anchor_conservative)
            lib_idx = data["leg_slugs"].index(anchor_liberal)

            # Free ideal points for non-anchored legislators
            xi_free = pm.Normal("xi_free", mu=0, sigma=1, shape=n_leg - 2)

            # Build full xi vector with anchors
            xi_list = []
            free_counter = 0
            for i in range(n_leg):
                if i == cons_idx:
                    xi_list.append(pm.math.constant(1.0))
                elif i == lib_idx:
                    xi_list.append(pm.math.constant(-1.0))
                else:
                    xi_list.append(xi_free[free_counter])
                    free_counter += 1

            xi = pm.math.stack(xi_list)
            xi = pm.Deterministic("xi", xi)

        else:
            # Option 2: Soft identification via prior + sign constraint
            xi = pm.Normal("xi", mu=0, sigma=1, shape=n_leg)
            # Will need post-hoc sign correction

        # --- Roll call parameters ---
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes)  # Difficulty
        beta = pm.Normal("beta", mu=0, sigma=2.5, shape=n_votes)  # Discrimination

        # --- Likelihood ---
        eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
        obs = pm.Bernoulli("obs", logit_p=eta, observed=y)

        # --- Sample ---
        trace = pm.sample(
            n_samples,
            tune=n_tune,
            cores=2,
            target_accept=0.9,
            random_seed=42,
        )

    idata = az.from_pymc3(trace=trace, model=model)

    # Add meaningful coordinate labels
    idata.posterior = idata.posterior.assign_coords(
        xi_dim_0=data["leg_slugs"],
        alpha_dim_0=data["vote_ids"],
        beta_dim_0=data["vote_ids"],
    )

    return idata
```

### Post-Processing and Visualization

```python
def extract_ideal_points(
    idata: az.InferenceData,
    legislator_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Extract posterior summaries for ideal points."""
    summary = az.summary(idata, var_names=["xi"], hdi_prob=0.95)

    # Add legislator metadata
    summary.index = summary.index.str.replace("xi[", "").str.replace("]", "")
    # If using coordinates, index is already slug names

    meta = legislator_meta.loc[summary.index]
    summary["party"] = meta["party"]
    summary["chamber"] = meta["chamber"]
    summary["full_name"] = meta["full_name"]

    return summary.sort_values("mean")


def plot_ideal_point_forest(
    idata: az.InferenceData,
    legislator_meta: pd.DataFrame,
    chamber: str = "House",
    save_path: str | None = None,
):
    """Forest plot of ideal points with credible intervals."""
    summary = extract_ideal_points(idata, legislator_meta)
    chamber_data = summary[summary["chamber"] == chamber].sort_values("mean")

    fig, ax = plt.subplots(figsize=(10, max(12, len(chamber_data) * 0.25)))

    colors = chamber_data["party"].map({"Republican": "#E81B23", "Democrat": "#0015BC"})

    y_pos = range(len(chamber_data))
    ax.hlines(y_pos, chamber_data["hdi_2.5%"], chamber_data["hdi_97.5%"],
             colors=colors, alpha=0.4, linewidth=1.5)
    ax.scatter(chamber_data["mean"], y_pos, c=colors, s=20, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{row['full_name']} ({row['party'][0]})"
         for _, row in chamber_data.iterrows()],
        fontsize=6,
    )
    ax.set_xlabel("Ideal Point (Conservative → Liberal)")
    ax.set_title(f"{chamber} Ideal Points with 95% Credible Intervals")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_discrimination_histogram(
    idata: az.InferenceData,
    save_path: str | None = None,
):
    """Histogram of roll call discrimination parameters."""
    beta_means = idata.posterior["beta"].mean(dim=["chain", "draw"]).values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(beta_means, bins=40, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Discrimination Parameter (β)")
    ax.set_ylabel("Number of Roll Calls")
    ax.set_title("Distribution of Roll Call Discrimination")
    ax.axvline(0, color="red", linestyle="--", label="β = 0 (no discrimination)")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

### MCMC Diagnostics

```python
def check_convergence(idata: az.InferenceData):
    """Run standard MCMC convergence diagnostics."""
    # R-hat (should be < 1.01 for all parameters)
    rhat = az.rhat(idata)
    print("Max R-hat (xi):", float(rhat["xi"].max()))
    print("Max R-hat (alpha):", float(rhat["alpha"].max()))
    print("Max R-hat (beta):", float(rhat["beta"].max()))

    # Effective sample size (should be > 400 for reliable inference)
    ess = az.ess(idata)
    print("Min ESS (xi):", float(ess["xi"].min()))

    # Trace plots for a few parameters
    az.plot_trace(idata, var_names=["xi"], coords={"xi_dim_0": idata.posterior.xi_dim_0[:5]})

    # Divergences
    divergences = idata.sample_stats["diverging"].sum().values
    print(f"Divergences: {divergences}")
```

## Interpretation Guide

### Reading the Forest Plot

- **Point estimate (dot)**: Posterior mean ideal point. Farther right = more conservative (by convention).
- **Horizontal line**: 95% Highest Density Interval (HDI). Contains the true ideal point with 95% probability.
- **Overlapping intervals**: Legislators whose intervals overlap cannot be reliably distinguished ideologically.
- **Non-overlapping intervals**: Clear ideological separation.
- **Wide intervals**: Uncertain estimate — legislator either had few votes or voted inconsistently relative to the spatial model.

### Discrimination Parameters

- **High |β|** (>1.5): Strongly partisan roll calls. Ideal for distinguishing legislators.
- **β near 0**: Non-discriminating votes (near-unanimous or random). These provide little information about ideal points.
- **Negative β**: The vote reversed the usual pattern (liberal legislators voted Yea, conservative voted Nay). This happens for procedural votes or when the bill's Yea position is the liberal one.

### Model Comparison

Use WAIC or LOO-CV (via ArviZ) to compare:
- 1D vs. 2D models
- IRT vs. simpler models (e.g., party-only model)
- Different prior specifications

```python
# Compare 1D and 2D models
comparison = az.compare({"1D": idata_1d, "2D": idata_2d}, ic="loo")
print(comparison)
```

## Advantages Over NOMINATE

1. **Uncertainty quantification**: Full posterior distributions, not just point estimates
2. **Native Python**: No R dependency
3. **Natural missing data handling**: No imputation needed
4. **Model comparison**: WAIC/LOO-CV for principled model selection
5. **Extensibility**: Easy to add hierarchical structure, time dynamics, or covariates
6. **Posterior predictive checks**: Can validate the model by simulating votes

## Limitations

1. **MCMC convergence**: Requires diagnostic checking. Can be slow for very large datasets.
2. **Identification**: Must manually set anchors or constraints
3. **Computation time**: Minutes rather than seconds (vs. PCA or NOMINATE)
4. **Complexity**: More parameters to choose (priors, anchors, number of dimensions)

## Kansas-Specific Considerations

- **Choosing anchors**: Pick one legislator known to be very conservative (e.g., a Freedom Caucus member) and one known to be very liberal (e.g., a progressive Democrat from Lawrence or Kansas City). These should be legislators who voted on many roll calls.
- **Run per chamber**: Just like NOMINATE, IRT works best within a single chamber.
- **Expected running time**: ~170 legislators x ~400 votes ≈ ~68K observations. Expect 5-15 minutes of MCMC sampling.
- **The posterior will show that some moderate Republicans overlap with conservative Democrats.** This is the key finding — the parties are not perfectly separated.
- **Compare PC1 with IRT ideal points**: Correlation should be > 0.95. If it's lower, the IRT model is capturing non-linearities that PCA misses.

## Feasibility Assessment

- **Data size**: 170 x 400 = excellent for IRT (the canonical example used 102 x 645)
- **Compute time**: 5-15 minutes with PyMC/NUTS
- **Libraries**: `pymc`, `arviz`, `pytensor`
- **Difficulty**: High (model specification, identification, MCMC diagnostics)

## Key References

- Clinton, Joshua, Simon Jackman, and Douglas Rivers. "The Statistical Analysis of Roll Call Data." *American Political Science Review* 98(2), 2004. https://www.cs.princeton.edu/courses/archive/fall09/cos597A/papers/ClintonJackmanRivers2004.pdf
- Jackman, Simon. "Multidimensional Analysis of Roll Call Data via Bayesian Simulation: Identification, Estimation, Inference, and Model Checking." *Political Analysis* 9(3), 2001.
- Jeffrey Arnold's Stan implementation of Jackman's examples: https://jrnold.github.io/bugs-examples-in-stan/legislators.html
- PyMC documentation: https://docs.pymc.io/
- ArviZ documentation: https://arviz-devs.github.io/arviz/
