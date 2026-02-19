# Latent Class and Mixture Models

**Category:** Clustering & Classification
**Prerequisites:** `01_DATA_vote_matrix_construction`, ideally after `15_BAY_irt_ideal_points`
**Complexity:** High
**Related:** `19_CLU_kmeans_voting_patterns`, `23_NET_community_detection`, `16_BAY_hierarchical_legislator_model`

## What It Measures

Latent class models assume each legislator belongs to one of $K$ unobserved (latent) factions, and that within each faction, voting behavior follows a distinct probability distribution. This is a fundamentally different paradigm from ideal point models: instead of placing legislators on a continuous ideological spectrum, latent class models assign them to discrete groups.

The key insight: **continuous ideology and discrete factions answer different questions.** Ideal point models ask "where does this legislator sit on the liberal-conservative scale?" Latent class models ask "which voting bloc does this legislator belong to, and how confident are we?" In a legislature like Kansas — where we hypothesize three factions (conservative Republicans, moderate Republicans, Democrats) — latent class models directly estimate the number, composition, and behavior of those factions.

## Questions It Answers

- How many distinct voting blocs does the legislature have?
- What is the probability profile of each bloc (how does a "typical" member of each faction vote on each roll call)?
- Which legislators have ambiguous faction membership (soft assignment probabilities)?
- Which roll calls best discriminate between factions (the "faction-defining" votes)?
- Does a discrete-faction model fit the data better or worse than a continuous-ideology model?

## Mathematical Foundation

### Basic Latent Class Model (LCA)

Each legislator $i$ belongs to one of $K$ latent classes with prior probability $\pi_k$:

$$P(c_i = k) = \pi_k, \quad \sum_{k=1}^{K} \pi_k = 1$$

Within class $k$, the probability of voting Yea on roll call $j$ is:

$$P(Y_{ij} = 1 | c_i = k) = \theta_{kj}$$

The votes are conditionally independent given class membership (the "local independence" assumption):

$$P(\mathbf{Y}_i | c_i = k) = \prod_{j=1}^{m} \theta_{kj}^{Y_{ij}} (1 - \theta_{kj})^{1 - Y_{ij}}$$

### Marginal Likelihood

$$P(\mathbf{Y}_i) = \sum_{k=1}^{K} \pi_k \prod_{j=1}^{m} \theta_{kj}^{Y_{ij}} (1 - \theta_{kj})^{1 - Y_{ij}}$$

Parameters $\{\pi_k, \theta_{kj}\}$ are estimated via EM (Expectation-Maximization) or Bayesian MCMC.

### Mixture of IRT Models

A more sophisticated variant combines latent classes with ideal points. Within each class $k$, legislators have class-specific ideal point distributions:

$$\xi_i | c_i = k \sim \text{Normal}(\mu_k, \sigma_k^2)$$

$$P(Y_{ij} = 1 | \xi_i, \alpha_j, \beta_j) = \text{logit}^{-1}(\beta_j \xi_i - \alpha_j)$$

This allows for both discrete faction structure AND continuous within-faction variation. It's the most complete model of legislative behavior — and the most complex.

### Choosing $K$

| Criterion | Formula | Notes |
|-----------|---------|-------|
| BIC | $-2 \ln L + p \ln n$ | Penalizes complexity; tends to choose smaller $K$ |
| AIC | $-2 \ln L + 2p$ | Less conservative than BIC |
| ICL (Integrated Completed Likelihood) | BIC + entropy penalty | Explicitly penalizes fuzzy class assignments |
| Held-out log-likelihood | Train on 80%, evaluate on 20% | Most principled but requires careful splitting |
| Bootstrap stability | Refit on bootstrap samples | Do the same $K$ classes keep appearing? |

**Rule of thumb:** Start with $K = 2$ (parties), increase to $K = 3$ (majority-party split), and keep going until the improvement is negligible. For Kansas, expect $K = 3$ to be the sweet spot.

## Python Implementation

### Basic Latent Class Analysis with scikit-learn

The `GaussianMixture` model can approximate LCA when applied to the vote matrix (treating binary votes as approximately continuous):

```python
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def fit_latent_class_model(
    vote_matrix: pd.DataFrame,
    k_range: range = range(2, 7),
) -> dict:
    """Fit latent class models for a range of K values.

    Args:
        vote_matrix: Binary vote matrix (imputed, no NaN).
        k_range: Range of K values to try.

    Returns:
        Dict with BIC scores, best model, and class assignments.
    """
    X = vote_matrix.values

    results = {"k": [], "bic": [], "aic": [], "log_likelihood": []}
    models = {}

    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",  # Diagonal = conditional independence approximation
            n_init=10,
            random_state=42,
            max_iter=500,
        )
        gmm.fit(X)

        results["k"].append(k)
        results["bic"].append(gmm.bic(X))
        results["aic"].append(gmm.aic(X))
        results["log_likelihood"].append(gmm.score(X) * len(X))
        models[k] = gmm

    # Find best K by BIC
    results_df = pd.DataFrame(results)
    best_k = results_df.loc[results_df["bic"].idxmin(), "k"]

    return {
        "results": results_df,
        "best_k": int(best_k),
        "models": models,
        "best_model": models[int(best_k)],
    }
```

### Proper Latent Class Analysis with PyMC

For a fully Bayesian treatment:

```python
import pymc as pm
import arviz as az
import numpy as np

def fit_bayesian_lca(
    vote_matrix: np.ndarray,
    K: int = 3,
    n_samples: int = 2000,
    n_tune: int = 1000,
) -> az.InferenceData:
    """Fit Bayesian Latent Class Analysis.

    Args:
        vote_matrix: Binary vote matrix (n_legislators x n_votes), NaN-free.
        K: Number of latent classes.
        n_samples: MCMC samples per chain.
        n_tune: Tuning samples.

    Returns:
        ArviZ InferenceData with posterior.
    """
    n_legislators, n_votes = vote_matrix.shape

    with pm.Model() as model:
        # Class membership probabilities
        pi = pm.Dirichlet("pi", a=np.ones(K))

        # Class-specific vote probabilities
        # theta[k, j] = P(Yea on vote j | class k)
        theta = pm.Beta("theta", alpha=1, beta=1, shape=(K, n_votes))

        # Class assignments (marginalized out for efficiency)
        # For each legislator, compute log-likelihood under each class
        log_lik_per_class = []
        for k in range(K):
            # Log-likelihood of all votes for legislator i under class k
            log_lik_k = pm.math.sum(
                vote_matrix * pm.math.log(theta[k])
                + (1 - vote_matrix) * pm.math.log(1 - theta[k]),
                axis=1,
            )
            log_lik_per_class.append(log_lik_k + pm.math.log(pi[k]))

        # Log-sum-exp over classes (marginalize out class assignment)
        log_lik_stack = pm.math.stack(log_lik_per_class, axis=0)  # (K, n_legislators)
        log_marginal = pm.math.logsumexp(log_lik_stack, axis=0)  # (n_legislators,)

        # Total log-likelihood
        pm.Potential("likelihood", log_marginal.sum())

        # Sample
        trace = pm.sample(
            n_samples,
            tune=n_tune,
            cores=2,
            target_accept=0.9,
            random_seed=42,
        )

    return az.from_pymc3(trace=trace, model=model)


def compute_class_probabilities(
    vote_matrix: np.ndarray,
    pi: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Compute posterior class membership probabilities for each legislator.

    Args:
        vote_matrix: (n_legislators, n_votes) binary matrix.
        pi: (K,) class prior probabilities.
        theta: (K, n_votes) class-specific vote probabilities.

    Returns:
        (n_legislators, K) matrix of class membership probabilities.
    """
    K = len(pi)
    n_legislators = vote_matrix.shape[0]

    log_probs = np.zeros((n_legislators, K))
    for k in range(K):
        log_probs[:, k] = (
            np.log(pi[k])
            + np.sum(vote_matrix * np.log(theta[k] + 1e-10)
                     + (1 - vote_matrix) * np.log(1 - theta[k] + 1e-10), axis=1)
        )

    # Normalize (log-sum-exp)
    log_probs -= log_probs.max(axis=1, keepdims=True)
    probs = np.exp(log_probs)
    probs /= probs.sum(axis=1, keepdims=True)

    return probs
```

### Analysis and Visualization

```python
def analyze_latent_classes(
    vote_matrix: pd.DataFrame,
    class_probs: np.ndarray,
    legislator_meta: pd.DataFrame,
    K: int,
) -> pd.DataFrame:
    """Analyze composition and voting profiles of latent classes."""
    assignments = class_probs.argmax(axis=1)
    max_probs = class_probs.max(axis=1)

    result = pd.DataFrame({
        "legislator_slug": vote_matrix.index,
        "assigned_class": assignments,
        "assignment_confidence": max_probs,
    })
    result = result.merge(
        legislator_meta[["party", "chamber", "full_name"]].reset_index(),
        left_on="legislator_slug",
        right_on="slug",
        how="left",
    )

    # Class composition by party
    print("Class composition:")
    composition = result.groupby(["assigned_class", "party"]).size().unstack(fill_value=0)
    print(composition)
    print()

    # Legislators with uncertain assignments (low confidence)
    uncertain = result[result["assignment_confidence"] < 0.8].sort_values("assignment_confidence")
    print(f"Uncertain assignments (confidence < 0.8): {len(uncertain)}")
    if len(uncertain) > 0:
        print(uncertain[["legislator_slug", "party", "assigned_class", "assignment_confidence"]].head(10))

    return result


def find_faction_defining_votes(
    theta: np.ndarray,
    vote_ids: list[str],
    rollcalls: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Find roll calls that best discriminate between factions.

    The most "faction-defining" votes are those where theta[k,j] differs
    most across classes.
    """
    # Compute max difference across classes for each vote
    K, n_votes = theta.shape
    max_diff = np.zeros(n_votes)
    for j in range(n_votes):
        class_probs = theta[:, j]
        max_diff[j] = class_probs.max() - class_probs.min()

    # Top discriminating votes
    top_indices = np.argsort(max_diff)[::-1][:top_n]
    rc_meta = rollcalls.set_index("vote_id")

    results = []
    for idx in top_indices:
        vote_id = vote_ids[idx]
        row = {"vote_id": vote_id, "max_class_diff": max_diff[idx]}
        for k in range(K):
            row[f"class_{k}_yea_prob"] = theta[k, idx]
        if vote_id in rc_meta.index:
            rc = rc_meta.loc[vote_id]
            row["bill_number"] = rc.get("bill_number", "")
            row["bill_title"] = str(rc.get("bill_title", ""))[:80]
            row["motion"] = rc.get("motion", "")
        results.append(row)

    return pd.DataFrame(results)


def plot_class_profiles(
    theta: np.ndarray,
    K: int,
    class_labels: list[str] | None = None,
    save_path: str | None = None,
):
    """Heatmap of class-specific vote probabilities for top discriminating votes."""
    # Sort votes by discriminating power
    max_diff = theta.max(axis=0) - theta.min(axis=0)
    top_votes = np.argsort(max_diff)[::-1][:50]

    theta_subset = theta[:, top_votes]

    fig, ax = plt.subplots(figsize=(16, 4))
    import seaborn as sns

    labels = class_labels or [f"Class {k}" for k in range(K)]
    sns.heatmap(
        theta_subset,
        ax=ax,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        yticklabels=labels,
        xticklabels=False,
    )
    ax.set_xlabel("Roll Calls (sorted by discriminating power)")
    ax.set_ylabel("Latent Class")
    ax.set_title("Class-Specific Yea Probabilities for Top Discriminating Votes")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_model_selection(fit_results: pd.DataFrame, save_path: str | None = None):
    """Plot BIC/AIC vs K for model selection."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(fit_results["k"], fit_results["bic"], "bo-", label="BIC")
    ax.plot(fit_results["k"], fit_results["aic"], "rs--", label="AIC")

    best_k_bic = fit_results.loc[fit_results["bic"].idxmin(), "k"]
    ax.axvline(best_k_bic, color="blue", linestyle=":", alpha=0.5, label=f"Best K (BIC) = {best_k_bic}")

    ax.set_xlabel("Number of Classes (K)")
    ax.set_ylabel("Information Criterion")
    ax.set_title("Latent Class Model Selection")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

## Interpretation Guide

### Reading Class Assignments

- **High-confidence assignments** (>0.9): The legislator clearly belongs to one faction. Their voting pattern is typical of that class.
- **Low-confidence assignments** (0.5-0.8): The legislator sits between factions. They may be moderates, swing voters, or issue-specific crossovers.
- **Near-uniform probabilities** (~1/K for all classes): The legislator doesn't fit any faction well. Investigate their voting record.

### Expected Kansas Results

| K | Expected Classes | Interpretation |
|---|-----------------|----------------|
| 2 | Republican bloc, Democrat bloc | Party-only model. Baseline. |
| 3 | Conservative R, Moderate R, Democrat | The hypothesized factional structure. Most informative. |
| 4 | Conservative R, Moderate R, Conservative D, Progressive D | Or geographic/committee-based subgroups |
| 5+ | Diminishing returns; classes start to be small and unstable | Overfit warning |

### Faction-Defining Votes

The votes with the highest discrimination between classes are the ones where the factions diverge most. For $K = 3$:
- A vote where Class 0 (conservative R) votes Yea 95%, Class 1 (moderate R) votes Yea 40%, and Class 2 (Democrats) votes Yea 5% is a perfect faction-defining vote.
- These votes are the ones legislators and journalists would identify as "the votes that split the party."

### Latent Class vs. Other Methods

| Method | Paradigm | Output | Best When |
|--------|----------|--------|-----------|
| K-Means | Geometric clustering | Hard assignments | Quick exploration |
| Community Detection | Network topology | Hard/soft assignments | Network structure matters |
| **Latent Class** | **Probabilistic generative model** | **Soft assignments + vote profiles** | **Want faction probabilities + defining votes** |
| IRT Ideal Points | Continuous latent variable | Position + uncertainty | Want ideological spectrum |
| Mixture of IRT | Both | Factions + within-faction spectrum | Maximum complexity warranted |

The key advantage of latent class over K-Means and community detection: it's a *generative* model. It says "here is how we believe votes are generated" and provides probability-based class membership, not just distance-based assignments. This means you can compute the probability that a specific legislator belongs to each faction, and you can identify which votes define each faction.

## Kansas-Specific Considerations

- **Start with $K = 3$** as the primary hypothesis. The moderate-conservative Republican split is the defining feature of Kansas politics.
- **Validate against known caucuses.** If the Kansas House has a formal or informal moderate Republican caucus, check whether the latent class model recovers its membership.
- **Senate may need $K = 2$.** With only 42 senators (32 R, 10 D), the data may not support more than two classes. The small Democratic contingent may be too few to further subdivide.
- **Faction-defining votes are journalistically interesting.** These are the votes that define the internal fissures of the majority party — useful for reporting or advocacy.
- **Compare with community detection** from `23_NET_community_detection`. If both methods identify similar groupings, the finding is robust.

## Feasibility Assessment

- **Data size**: 170 legislators x ~400 contested votes = adequate for K ≤ 5
- **Compute time**: GMM — seconds. Bayesian LCA — 5-15 minutes.
- **Libraries**: `scikit-learn` (GaussianMixture), `pymc` (Bayesian LCA)
- **Difficulty**: Medium (GMM approximation), High (full Bayesian LCA)

## Key References

- Goodman, Leo A. "Exploratory Latent Structure Analysis Using Both Identifiable and Unidentifiable Models." *Biometrika* 61(2), 1974.
- McLachlan, Geoffrey J., and David Peel. *Finite Mixture Models*. Wiley, 2000.
- Gross, Justin H., and Kelly T. Manrique. "Revealing the Political Arena: A Semi-supervised Approach to Finding Political Actors in Legislative Text." *Political Analysis*, 2023. (Uses mixture models for legislative behavior.)
- Treier, Shawn, and Simon Jackman. "Democracy as a Latent Variable." *American Journal of Political Science* 52(1), 2008. (Latent class models in political science.)
