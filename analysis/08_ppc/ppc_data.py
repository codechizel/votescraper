"""Posterior predictive check computations — pure functions, no I/O.

All functions take numpy/xarray arrays and return numpy/xarray results.
No file reading, no prints, fully testable with synthetic data.

Log-likelihood uses manual numpy computation (not PyMC model rebuild) to avoid
model reconstruction complexity and PyTensor compilation (~30-60s per model).
"""

from typing import Any

import arviz as az
import numpy as np
import xarray as xr
from numpy.typing import NDArray

RANDOM_SEED = 42


# ── Numerically Stable Log-Sigmoid ──────────────────────────────────────────


def _log_sigmoid(eta: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute log(sigmoid(eta)) = -log(1 + exp(-eta)) with numerical stability."""
    return np.where(eta >= 0, -np.log1p(np.exp(-eta)), eta - np.log1p(np.exp(eta)))


def _log1m_sigmoid(eta: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute log(1 - sigmoid(eta)) = -log(1 + exp(eta)) with numerical stability."""
    return np.where(eta >= 0, -eta - np.log1p(np.exp(-eta)), -np.log1p(np.exp(eta)))


# ── Log-Likelihood Computation ──────────────────────────────────────────────


def compute_log_likelihood_1d(
    idata: az.InferenceData,
    data: dict[str, Any],
) -> xr.Dataset:
    """Compute Bernoulli log-likelihood from 1D IRT posterior.

    eta = beta[j] * xi[i] - alpha[j]
    log_lik = y * log(sigmoid(eta)) + (1-y) * log(1 - sigmoid(eta))

    Returns xarray Dataset with shape (chain, draw, n_obs).
    """
    xi = idata.posterior["xi"].values  # (chain, draw, n_leg)
    alpha = idata.posterior["alpha"].values  # (chain, draw, n_votes)
    beta = idata.posterior["beta"].values  # (chain, draw, n_votes)

    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y = data["y"].astype(np.float64)

    # Vectorized across all observations for each (chain, draw)
    eta = beta[:, :, vote_idx] * xi[:, :, leg_idx] - alpha[:, :, vote_idx]

    log_lik = y * _log_sigmoid(eta) + (1.0 - y) * _log1m_sigmoid(eta)

    n_chains, n_draws = xi.shape[0], xi.shape[1]
    ds = xr.Dataset(
        {"y": (["chain", "draw", "obs"], log_lik)},
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "obs": np.arange(len(y)),
        },
    )
    return ds


def compute_log_likelihood_2d(
    idata: az.InferenceData,
    data: dict[str, Any],
) -> xr.Dataset:
    """Compute Bernoulli log-likelihood from 2D IRT posterior.

    eta = sum_d(beta_matrix[j,d] * xi[i,d]) - alpha[j]
    """
    xi = idata.posterior["xi"].values  # (chain, draw, n_leg, 2)
    alpha = idata.posterior["alpha"].values  # (chain, draw, n_votes)

    # 2D discrimination — look for beta_matrix or separate beta_1/beta_2
    if "beta_matrix" in idata.posterior:
        beta_matrix = idata.posterior["beta_matrix"].values  # (chain, draw, n_votes, 2)
    else:
        b1 = idata.posterior["beta_1"].values  # (chain, draw, n_votes)
        b2 = idata.posterior["beta_2"].values
        beta_matrix = np.stack([b1, b2], axis=-1)

    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y = data["y"].astype(np.float64)

    # eta = sum_d(beta[j,d] * xi[i,d]) - alpha[j]
    xi_obs = xi[:, :, leg_idx, :]  # (chain, draw, n_obs, 2)
    beta_obs = beta_matrix[:, :, vote_idx, :]  # (chain, draw, n_obs, 2)
    eta = (beta_obs * xi_obs).sum(axis=-1) - alpha[:, :, vote_idx]

    log_lik = y * _log_sigmoid(eta) + (1.0 - y) * _log1m_sigmoid(eta)

    n_chains, n_draws = xi.shape[0], xi.shape[1]
    ds = xr.Dataset(
        {"y": (["chain", "draw", "obs"], log_lik)},
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "obs": np.arange(len(y)),
        },
    )
    return ds


def compute_log_likelihood_hierarchical(
    idata: az.InferenceData,
    data: dict[str, Any],
) -> xr.Dataset:
    """Compute Bernoulli log-likelihood from hierarchical IRT posterior.

    Uses 'xi' (the computed ideal point Deterministic), not 'xi_offset'.
    Likelihood formula is identical to 1D flat IRT.
    """
    return compute_log_likelihood_1d(idata, data)


def add_log_likelihood_to_idata(
    idata: az.InferenceData,
    log_lik_dataset: xr.Dataset,
) -> az.InferenceData:
    """Add log_likelihood group to InferenceData (returns new object)."""
    return idata.copy() + az.InferenceData(log_likelihood=log_lik_dataset)


# ── PPC Battery ─────────────────────────────────────────────────────────────


def _compute_eta_1d(
    xi: NDArray[np.floating],
    alpha: NDArray[np.floating],
    beta: NDArray[np.floating],
    leg_idx: NDArray[np.integer],
    vote_idx: NDArray[np.integer],
) -> NDArray[np.floating]:
    """Compute eta for a single posterior draw (1D model)."""
    return beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]


def _compute_eta_2d(
    xi: NDArray[np.floating],
    alpha: NDArray[np.floating],
    beta_matrix: NDArray[np.floating],
    leg_idx: NDArray[np.integer],
    vote_idx: NDArray[np.integer],
) -> NDArray[np.floating]:
    """Compute eta for a single posterior draw (2D model)."""
    return (beta_matrix[vote_idx, :] * xi[leg_idx, :]).sum(axis=-1) - alpha[vote_idx]


def run_ppc_battery(
    idata: az.InferenceData,
    data: dict[str, Any],
    *,
    n_reps: int = 500,
    model_type: str = "1d",
) -> dict[str, Any]:
    """Run comprehensive PPC battery.

    Returns dict with:
      - yea_rate: observed, replicated distribution, Bayesian p-value
      - accuracy: mean classification accuracy
      - gmp: geometric mean probability
      - apre: aggregate proportional reduction in error
    """
    xi_post = idata.posterior["xi"].values
    alpha_post = idata.posterior["alpha"].values

    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y_obs = data["y"].astype(np.float64)
    observed_yea_rate = float(y_obs.mean())
    n_obs = len(y_obs)

    n_chains, n_draws = xi_post.shape[0], xi_post.shape[1]
    n_reps = min(n_reps, n_chains * n_draws)

    rng = np.random.default_rng(RANDOM_SEED)

    # Handle 2D beta
    if model_type == "2d":
        if "beta_matrix" in idata.posterior:
            beta_post = idata.posterior["beta_matrix"].values
        else:
            b1 = idata.posterior["beta_1"].values
            b2 = idata.posterior["beta_2"].values
            beta_post = np.stack([b1, b2], axis=-1)
    else:
        beta_post = idata.posterior["beta"].values

    rep_yea_rates = np.empty(n_reps)
    rep_accuracies = np.empty(n_reps)
    rep_gmps = np.empty(n_reps)

    for i in range(n_reps):
        c = rng.integers(0, n_chains)
        d = rng.integers(0, n_draws)

        if model_type == "2d":
            eta = _compute_eta_2d(
                xi_post[c, d],
                alpha_post[c, d],
                beta_post[c, d],
                leg_idx,
                vote_idx,
            )
        else:
            eta = _compute_eta_1d(
                xi_post[c, d],
                alpha_post[c, d],
                beta_post[c, d],
                leg_idx,
                vote_idx,
            )

        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        y_rep = rng.binomial(1, p)

        rep_yea_rates[i] = float(y_rep.mean())
        rep_accuracies[i] = float((y_rep == y_obs).mean())

        # GMP: geometric mean probability — exp(mean(log(p_correct)))
        p_correct = np.where(y_obs == 1, p, 1.0 - p)
        p_correct = np.clip(p_correct, 1e-15, 1.0)
        rep_gmps[i] = float(np.exp(np.mean(np.log(p_correct))))

    # APRE: aggregate proportional reduction in error
    # Compares model predictions to modal-category baseline
    baseline_accuracy = max(observed_yea_rate, 1.0 - observed_yea_rate)
    mean_accuracy = float(rep_accuracies.mean())
    if baseline_accuracy < 1.0:
        apre = (mean_accuracy - baseline_accuracy) / (1.0 - baseline_accuracy)
    else:
        apre = 0.0

    # Bayesian p-values
    p_yea_rate = float(np.mean(rep_yea_rates >= observed_yea_rate))

    return {
        "observed_yea_rate": observed_yea_rate,
        "replicated_yea_rate_mean": float(rep_yea_rates.mean()),
        "replicated_yea_rate_sd": float(rep_yea_rates.std()),
        "bayesian_p_yea_rate": p_yea_rate,
        "mean_accuracy": mean_accuracy,
        "accuracy_sd": float(rep_accuracies.std()),
        "mean_gmp": float(rep_gmps.mean()),
        "gmp_sd": float(rep_gmps.std()),
        "apre": apre,
        "baseline_accuracy": baseline_accuracy,
        "n_reps": n_reps,
        "n_obs": n_obs,
        "replicated_yea_rates": rep_yea_rates,
        "replicated_accuracies": rep_accuracies,
    }


# ── Item-Level and Person-Level Checks ──────────────────────────────────────


def compute_item_ppc(
    idata: az.InferenceData,
    data: dict[str, Any],
    *,
    n_reps: int = 500,
    model_type: str = "1d",
) -> dict[str, Any]:
    """Per-roll-call endorsement rates: observed vs replicated.

    Returns dict with:
      - observed_rates: per-item observed endorsement rate
      - replicated_rates: (n_reps, n_votes) replicated rates
      - misfitting_items: indices where 95% interval excludes observed
    """
    xi_post = idata.posterior["xi"].values
    alpha_post = idata.posterior["alpha"].values

    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y_obs = data["y"].astype(np.float64)
    n_votes = data["n_votes"]

    n_chains, n_draws = xi_post.shape[0], xi_post.shape[1]
    n_reps = min(n_reps, n_chains * n_draws)

    if model_type == "2d":
        if "beta_matrix" in idata.posterior:
            beta_post = idata.posterior["beta_matrix"].values
        else:
            b1 = idata.posterior["beta_1"].values
            b2 = idata.posterior["beta_2"].values
            beta_post = np.stack([b1, b2], axis=-1)
    else:
        beta_post = idata.posterior["beta"].values

    rng = np.random.default_rng(RANDOM_SEED)

    # Observed per-item rates
    observed_rates = np.zeros(n_votes)
    item_counts = np.zeros(n_votes)
    for j in range(n_votes):
        mask = vote_idx == j
        if mask.any():
            observed_rates[j] = y_obs[mask].mean()
            item_counts[j] = mask.sum()

    # Replicated per-item rates
    replicated_rates = np.zeros((n_reps, n_votes))
    for i in range(n_reps):
        c = rng.integers(0, n_chains)
        d = rng.integers(0, n_draws)

        if model_type == "2d":
            eta = _compute_eta_2d(
                xi_post[c, d],
                alpha_post[c, d],
                beta_post[c, d],
                leg_idx,
                vote_idx,
            )
        else:
            eta = _compute_eta_1d(
                xi_post[c, d],
                alpha_post[c, d],
                beta_post[c, d],
                leg_idx,
                vote_idx,
            )

        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        y_rep = rng.binomial(1, p)

        for j in range(n_votes):
            mask = vote_idx == j
            if mask.any():
                replicated_rates[i, j] = y_rep[mask].mean()

    # Identify misfitting items (observed outside 95% replicated interval)
    lo = np.percentile(replicated_rates, 2.5, axis=0)
    hi = np.percentile(replicated_rates, 97.5, axis=0)
    misfitting = np.where((observed_rates < lo) | (observed_rates > hi))[0]

    return {
        "observed_rates": observed_rates,
        "replicated_rates": replicated_rates,
        "item_counts": item_counts,
        "lo_95": lo,
        "hi_95": hi,
        "misfitting_items": misfitting,
        "n_misfitting": len(misfitting),
        "n_votes": n_votes,
    }


def compute_person_ppc(
    idata: az.InferenceData,
    data: dict[str, Any],
    *,
    n_reps: int = 500,
    model_type: str = "1d",
) -> dict[str, Any]:
    """Per-legislator total scores: observed vs replicated.

    Returns dict with:
      - observed_totals: per-person observed total Yea count
      - replicated_totals: (n_reps, n_legislators) replicated totals
      - misfitting_persons: indices where 95% interval excludes observed
    """
    xi_post = idata.posterior["xi"].values
    alpha_post = idata.posterior["alpha"].values

    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y_obs = data["y"].astype(np.float64)
    n_legislators = data["n_legislators"]

    n_chains, n_draws = xi_post.shape[0], xi_post.shape[1]
    n_reps = min(n_reps, n_chains * n_draws)

    if model_type == "2d":
        if "beta_matrix" in idata.posterior:
            beta_post = idata.posterior["beta_matrix"].values
        else:
            b1 = idata.posterior["beta_1"].values
            b2 = idata.posterior["beta_2"].values
            beta_post = np.stack([b1, b2], axis=-1)
    else:
        beta_post = idata.posterior["beta"].values

    rng = np.random.default_rng(RANDOM_SEED)

    # Observed per-person totals
    observed_totals = np.zeros(n_legislators)
    person_counts = np.zeros(n_legislators)
    for i in range(n_legislators):
        mask = leg_idx == i
        if mask.any():
            observed_totals[i] = y_obs[mask].sum()
            person_counts[i] = mask.sum()

    # Replicated per-person totals
    replicated_totals = np.zeros((n_reps, n_legislators))
    for rep in range(n_reps):
        c = rng.integers(0, n_chains)
        d = rng.integers(0, n_draws)

        if model_type == "2d":
            eta = _compute_eta_2d(
                xi_post[c, d],
                alpha_post[c, d],
                beta_post[c, d],
                leg_idx,
                vote_idx,
            )
        else:
            eta = _compute_eta_1d(
                xi_post[c, d],
                alpha_post[c, d],
                beta_post[c, d],
                leg_idx,
                vote_idx,
            )

        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        y_rep = rng.binomial(1, p)

        for i in range(n_legislators):
            mask = leg_idx == i
            if mask.any():
                replicated_totals[rep, i] = y_rep[mask].sum()

    # Identify misfitting persons
    lo = np.percentile(replicated_totals, 2.5, axis=0)
    hi = np.percentile(replicated_totals, 97.5, axis=0)
    misfitting = np.where((observed_totals < lo) | (observed_totals > hi))[0]

    return {
        "observed_totals": observed_totals,
        "replicated_totals": replicated_totals,
        "person_counts": person_counts,
        "lo_95": lo,
        "hi_95": hi,
        "misfitting_persons": misfitting,
        "n_misfitting": len(misfitting),
        "n_legislators": n_legislators,
    }


def compute_vote_margin_ppc(
    idata: az.InferenceData,
    data: dict[str, Any],
    *,
    n_reps: int = 500,
    model_type: str = "1d",
) -> dict[str, Any]:
    """Vote margin distribution: observed vs replicated.

    Margin = abs(yea_fraction - 0.5) * 2 for each roll call.
    """
    xi_post = idata.posterior["xi"].values
    alpha_post = idata.posterior["alpha"].values

    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y_obs = data["y"].astype(np.float64)
    n_votes = data["n_votes"]

    n_chains, n_draws = xi_post.shape[0], xi_post.shape[1]
    n_reps = min(n_reps, n_chains * n_draws)

    if model_type == "2d":
        if "beta_matrix" in idata.posterior:
            beta_post = idata.posterior["beta_matrix"].values
        else:
            b1 = idata.posterior["beta_1"].values
            b2 = idata.posterior["beta_2"].values
            beta_post = np.stack([b1, b2], axis=-1)
    else:
        beta_post = idata.posterior["beta"].values

    rng = np.random.default_rng(RANDOM_SEED)

    # Observed margins
    observed_margins = np.zeros(n_votes)
    for j in range(n_votes):
        mask = vote_idx == j
        if mask.any():
            yea_frac = y_obs[mask].mean()
            observed_margins[j] = abs(yea_frac - 0.5) * 2.0

    # Replicated margins
    replicated_margins = np.zeros((n_reps, n_votes))
    for rep in range(n_reps):
        c = rng.integers(0, n_chains)
        d = rng.integers(0, n_draws)

        if model_type == "2d":
            eta = _compute_eta_2d(
                xi_post[c, d],
                alpha_post[c, d],
                beta_post[c, d],
                leg_idx,
                vote_idx,
            )
        else:
            eta = _compute_eta_1d(
                xi_post[c, d],
                alpha_post[c, d],
                beta_post[c, d],
                leg_idx,
                vote_idx,
            )

        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        y_rep = rng.binomial(1, p)

        for j in range(n_votes):
            mask = vote_idx == j
            if mask.any():
                yea_frac = y_rep[mask].mean()
                replicated_margins[rep, j] = abs(yea_frac - 0.5) * 2.0

    return {
        "observed_margins": observed_margins,
        "replicated_margins": replicated_margins,
        "n_votes": n_votes,
    }


# ── Yen's Q3 Local Dependence ──────────────────────────────────────────────


def compute_yens_q3(
    idata: az.InferenceData,
    data: dict[str, Any],
    *,
    n_draws_sample: int = 100,
    model_type: str = "1d",
) -> dict[str, Any]:
    """Compute Yen's Q3 residual correlation matrix.

    For each sampled posterior draw:
      1. Compute residual = y_obs - sigmoid(eta) for each observation
      2. Reshape to (n_legislators, n_votes) residual matrix
      3. Correlate columns (items) across persons
    Average across draws. Flag item pairs where |Q3| > 0.2.
    """
    xi_post = idata.posterior["xi"].values
    alpha_post = idata.posterior["alpha"].values

    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y_obs = data["y"].astype(np.float64)
    n_legislators = data["n_legislators"]
    n_votes = data["n_votes"]

    n_chains, n_draws = xi_post.shape[0], xi_post.shape[1]
    n_draws_sample = min(n_draws_sample, n_chains * n_draws)

    if model_type == "2d":
        if "beta_matrix" in idata.posterior:
            beta_post = idata.posterior["beta_matrix"].values
        else:
            b1 = idata.posterior["beta_1"].values
            b2 = idata.posterior["beta_2"].values
            beta_post = np.stack([b1, b2], axis=-1)
    else:
        beta_post = idata.posterior["beta"].values

    rng = np.random.default_rng(RANDOM_SEED)

    q3_sum = np.zeros((n_votes, n_votes))
    valid_draws = 0

    for _ in range(n_draws_sample):
        c = rng.integers(0, n_chains)
        d = rng.integers(0, n_draws)

        if model_type == "2d":
            eta = _compute_eta_2d(
                xi_post[c, d],
                alpha_post[c, d],
                beta_post[c, d],
                leg_idx,
                vote_idx,
            )
        else:
            eta = _compute_eta_1d(
                xi_post[c, d],
                alpha_post[c, d],
                beta_post[c, d],
                leg_idx,
                vote_idx,
            )

        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        residual = y_obs - p

        # Build residual matrix (nan for missing observations)
        resid_matrix = np.full((n_legislators, n_votes), np.nan)
        resid_matrix[leg_idx, vote_idx] = residual

        # Correlation across persons for each item pair (ignore nan)
        # Use np.corrcoef on columns with enough shared observations
        # Build pairwise correlations manually for nan handling
        q3_draw = np.full((n_votes, n_votes), np.nan)
        for j1 in range(n_votes):
            for j2 in range(j1, n_votes):
                col1 = resid_matrix[:, j1]
                col2 = resid_matrix[:, j2]
                valid = ~np.isnan(col1) & ~np.isnan(col2)
                if valid.sum() >= 3:  # Need at least 3 shared observations
                    r = np.corrcoef(col1[valid], col2[valid])[0, 1]
                    q3_draw[j1, j2] = r
                    q3_draw[j2, j1] = r

        # Accumulate valid entries
        valid_mask = ~np.isnan(q3_draw)
        q3_sum[valid_mask] += q3_draw[valid_mask]
        valid_draws += 1

    # Average Q3 matrix
    q3_matrix = q3_sum / max(valid_draws, 1)
    np.fill_diagonal(q3_matrix, 1.0)

    # Count violations
    upper_tri = np.triu_indices(n_votes, k=1)
    q3_upper = q3_matrix[upper_tri]
    n_violations = int(np.sum(np.abs(q3_upper) > 0.2))
    n_pairs = len(q3_upper)

    return {
        "q3_matrix": q3_matrix,
        "n_violations": n_violations,
        "n_pairs": n_pairs,
        "violation_rate": n_violations / max(n_pairs, 1),
        "max_abs_q3": float(np.nanmax(np.abs(q3_upper))) if n_pairs > 0 else 0.0,
        "mean_abs_q3": float(np.nanmean(np.abs(q3_upper))) if n_pairs > 0 else 0.0,
        "n_draws_sampled": valid_draws,
    }


# ── LOO-CV Model Comparison ────────────────────────────────────────────────


def compute_loo(idata: az.InferenceData) -> az.ELPDData:
    """Compute LOO-CV using PSIS (Pareto-smoothed importance sampling).

    Requires log_likelihood group in idata.
    """
    return az.loo(idata, pointwise=True)


def compare_models(
    model_idatas: dict[str, az.InferenceData],
) -> tuple[object, dict[str, az.ELPDData]]:
    """Compare multiple models via LOO-CV.

    Args:
        model_idatas: {model_name: idata_with_log_likelihood}

    Returns:
        (comparison_df, {model_name: loo_result})
    """
    loo_results = {}
    for name, idata in model_idatas.items():
        loo_results[name] = compute_loo(idata)

    comparison = az.compare(loo_results)
    return comparison, loo_results


def summarize_pareto_k(loo_result: az.ELPDData) -> dict[str, int]:
    """Count observations in each Pareto k diagnostic category.

    Categories (Vehtari et al. 2017):
      good:       k < 0.5  (reliable)
      ok:         0.5 <= k < 0.7  (marginal)
      bad:        0.7 <= k < 1.0  (unreliable, higher variance)
      very_bad:   k >= 1.0  (PSIS fails)
    """
    k_values = loo_result.pareto_k.values

    return {
        "good": int(np.sum(k_values < 0.5)),
        "ok": int(np.sum((k_values >= 0.5) & (k_values < 0.7))),
        "bad": int(np.sum((k_values >= 0.7) & (k_values < 1.0))),
        "very_bad": int(np.sum(k_values >= 1.0)),
        "total": len(k_values),
        "max_k": float(np.max(k_values)),
        "mean_k": float(np.mean(k_values)),
    }
