"""
Beta Prior Experiment: Testing fixes for the D-Yea blind spot.

Runs the IRT model with different beta priors on House data and compares:
- Beta distributions (especially D-Yea bill handling)
- Ideal point correlations with PCA and LogNormal baseline
- Holdout accuracy
- Convergence diagnostics

See analysis/design/beta_prior_investigation.md for the full writeup.

Usage:
    uv run python analysis/irt_beta_experiment.py [--n-samples 500] [--n-tune 300]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt
from scipy import stats
from sklearn.metrics import roc_auc_score

# ── Import shared functions from irt.py ──────────────────────────────────────

try:
    from analysis.irt import (
        load_eda_matrices,
        load_metadata,
        load_pca_scores,
        prepare_irt_data,
        select_anchors,
    )
except ModuleNotFoundError:
    from irt import (  # type: ignore[no-redef]
        load_eda_matrices,
        load_metadata,
        load_pca_scores,
        prepare_irt_data,
        select_anchors,
    )

# ── Constants ────────────────────────────────────────────────────────────────

TARGET_ACCEPT = 0.9
RANDOM_SEED = 42
HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 42


# ── Model builders ───────────────────────────────────────────────────────────


def build_and_sample_variant(
    data: dict,
    cons_idx: int,
    lib_idx: int,
    n_samples: int,
    n_tune: int,
    n_chains: int,
    beta_prior: str,
) -> tuple[az.InferenceData, float]:
    """Build 2PL IRT with a configurable beta prior.

    beta_prior: one of "lognormal_0.5_0.5", "normal_0_2.5", "normal_0_1"
    """
    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y = data["y"]
    n_leg = data["n_legislators"]
    n_votes = data["n_votes"]

    coords = {
        "legislator": data["leg_slugs"],
        "vote": data["vote_ids"],
        "obs_id": np.arange(data["n_obs"]),
    }

    with pm.Model(coords=coords):
        # Ideal points with anchors (same for all variants)
        xi_free = pm.Normal("xi_free", mu=0, sigma=1, shape=n_leg - 2)
        xi_raw = pt.zeros(n_leg)
        xi_raw = pt.set_subtensor(xi_raw[cons_idx], 1.0)
        xi_raw = pt.set_subtensor(xi_raw[lib_idx], -1.0)
        free_positions = [i for i in range(n_leg) if i != cons_idx and i != lib_idx]
        for k, pos in enumerate(free_positions):
            xi_raw = pt.set_subtensor(xi_raw[pos], xi_free[k])
        xi = pm.Deterministic("xi", xi_raw, dims="legislator")

        # Difficulty (same for all variants)
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")

        # Discrimination — the variable under test
        if beta_prior == "lognormal_0.5_0.5":
            beta = pm.LogNormal("beta", mu=0.5, sigma=0.5, shape=n_votes, dims="vote")
        elif beta_prior == "normal_0_2.5":
            beta = pm.Normal("beta", mu=0, sigma=2.5, shape=n_votes, dims="vote")
        elif beta_prior == "normal_0_1":
            beta = pm.Normal("beta", mu=0, sigma=1, shape=n_votes, dims="vote")
        else:
            msg = f"Unknown beta_prior: {beta_prior}"
            raise ValueError(msg)

        # Likelihood
        eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

        # Sample
        t0 = time.time()
        idata = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            progressbar=True,
        )
        sampling_time = time.time() - t0

    return idata, sampling_time


# ── Analysis functions ───────────────────────────────────────────────────────


def classify_bill_direction(
    matrix: pl.DataFrame,
    vote_ids: list[str],
    ideal_points: dict[str, float],
) -> dict[str, str]:
    """Classify each bill as R-Yea or D-Yea based on which party votes Yea."""
    # We need party info; use ideal points as proxy (positive = R, negative = D)
    # But better: use actual party from legislators
    directions = {}
    slug_col = "legislator_slug"

    for vid in vote_ids:
        if vid not in matrix.columns:
            directions[vid] = "unknown"
            continue

        r_yea, r_nay, d_yea, d_nay = 0, 0, 0, 0
        for row in matrix.iter_rows(named=True):
            slug = row[slug_col]
            val = row[vid]
            if val is None or slug not in ideal_points:
                continue
            xi = ideal_points[slug]
            # Use PCA/IRT sign convention: positive = Republican
            if xi > 0:
                if val == 1:
                    r_yea += 1
                else:
                    r_nay += 1
            else:
                if val == 1:
                    d_yea += 1
                else:
                    d_nay += 1

        if r_yea > r_nay:
            directions[vid] = "R-Yea"
        elif d_yea > d_nay:
            directions[vid] = "D-Yea"
        else:
            directions[vid] = "mixed"

    return directions


def extract_metrics(
    idata: az.InferenceData,
    data: dict,
    pca_scores: pl.DataFrame,
    matrix: pl.DataFrame,
    legislators: pl.DataFrame,
    bill_directions: dict[str, str],
    sampling_time: float,
) -> dict:
    """Extract all comparison metrics from a fitted model."""
    metrics: dict = {"sampling_time": sampling_time}

    # --- Convergence ---
    rhat = az.rhat(idata)
    ess = az.ess(idata)
    xi_rhat_max = float(rhat["xi"].max())
    beta_rhat_max = float(rhat["beta"].max())
    xi_ess_min = float(ess["xi"].min())

    sampler_stats = idata.sample_stats
    if hasattr(sampler_stats, "diverging"):
        divergences = int(sampler_stats["diverging"].sum())
    else:
        divergences = 0

    metrics["xi_rhat_max"] = xi_rhat_max
    metrics["beta_rhat_max"] = beta_rhat_max
    metrics["xi_ess_min"] = xi_ess_min
    metrics["divergences"] = divergences
    metrics["convergence_ok"] = xi_rhat_max < 1.05 and divergences < 50  # relaxed for short runs

    # --- Beta distribution ---
    beta_means = idata.posterior["beta"].mean(dim=["chain", "draw"]).values
    metrics["beta_mean_overall"] = float(np.mean(beta_means))
    metrics["beta_std_overall"] = float(np.std(beta_means))
    metrics["beta_min"] = float(np.min(beta_means))
    metrics["beta_max"] = float(np.max(beta_means))
    metrics["n_negative_beta"] = int(np.sum(beta_means < 0))

    # D-Yea bill beta values
    d_yea_betas = []
    r_yea_betas = []
    for i, vid in enumerate(data["vote_ids"]):
        direction = bill_directions.get(vid, "unknown")
        if direction == "D-Yea":
            d_yea_betas.append(beta_means[i])
        elif direction == "R-Yea":
            r_yea_betas.append(beta_means[i])

    metrics["d_yea_count"] = len(d_yea_betas)
    metrics["r_yea_count"] = len(r_yea_betas)
    if d_yea_betas:
        metrics["d_yea_beta_mean"] = float(np.mean(d_yea_betas))
        metrics["d_yea_beta_absmax"] = float(np.max(np.abs(d_yea_betas)))
        metrics["d_yea_beta_absmean"] = float(np.mean(np.abs(d_yea_betas)))
    if r_yea_betas:
        metrics["r_yea_beta_mean"] = float(np.mean(r_yea_betas))

    # --- Ideal points ---
    xi_means = idata.posterior["xi"].mean(dim=["chain", "draw"]).values
    slug_to_xi = dict(zip(data["leg_slugs"], xi_means))

    # PCA correlation
    pca_map = {row["legislator_slug"]: row["PC1"] for row in pca_scores.iter_rows(named=True)}
    shared_slugs = [s for s in data["leg_slugs"] if s in pca_map]
    xi_vals = [slug_to_xi[s] for s in shared_slugs]
    pca_vals = [pca_map[s] for s in shared_slugs]
    pearson_r = float(np.corrcoef(xi_vals, pca_vals)[0, 1])
    spearman_rho = float(stats.spearmanr(xi_vals, pca_vals).statistic)
    metrics["pca_pearson_r"] = pearson_r
    metrics["pca_spearman_rho"] = spearman_rho

    # --- Holdout ---
    alpha_means = idata.posterior["alpha"].mean(dim=["chain", "draw"]).values
    rng = np.random.RandomState(HOLDOUT_SEED)
    holdout_mask = rng.random(data["n_obs"]) < HOLDOUT_FRACTION
    holdout_leg = data["leg_idx"][holdout_mask]
    holdout_vote = data["vote_idx"][holdout_mask]
    holdout_y = data["y"][holdout_mask]

    if len(holdout_y) > 0:
        eta_holdout = beta_means[holdout_vote] * xi_means[holdout_leg] - alpha_means[holdout_vote]
        p_yea = 1.0 / (1.0 + np.exp(-eta_holdout))
        pred = (p_yea >= 0.5).astype(int)
        accuracy = float(np.mean(pred == holdout_y))
        base_rate = float(np.mean(holdout_y))
        try:
            auc = float(roc_auc_score(holdout_y, p_yea))
        except ValueError:
            auc = float("nan")
        metrics["holdout_accuracy"] = accuracy
        metrics["holdout_base_rate"] = base_rate
        metrics["holdout_auc"] = auc

    return metrics


def make_comparison_plots(
    all_results: dict[str, dict],
    all_idatas: dict[str, az.InferenceData],
    data: dict,
    bill_directions: dict[str, str],
    output_dir: Path,
) -> None:
    """Generate comparison plots across all variants."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Beta distributions ---
    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 4), sharey=True)
    if len(all_results) == 1:
        axes = [axes]

    for ax, (label, idata) in zip(axes, all_idatas.items()):
        beta_means = idata.posterior["beta"].mean(dim=["chain", "draw"]).values
        d_yea_mask = [bill_directions.get(vid, "") == "D-Yea" for vid in data["vote_ids"]]
        r_yea_mask = [bill_directions.get(vid, "") == "R-Yea" for vid in data["vote_ids"]]

        ax.hist(
            beta_means[r_yea_mask],
            bins=40,
            alpha=0.6,
            color="#E81B23",
            label="R-Yea bills",
        )
        ax.hist(
            beta_means[d_yea_mask],
            bins=40,
            alpha=0.6,
            color="#0015BC",
            label="D-Yea bills",
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("β (discrimination)")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count")
    fig.suptitle("Beta Distributions by Prior Choice", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "beta_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: beta_distributions.png")

    # --- Plot 2: Ideal point comparison scatter ---
    if "LogNormal(0.5,0.5)" in all_idatas and len(all_idatas) > 1:
        baseline_xi = (
            all_idatas["LogNormal(0.5,0.5)"].posterior["xi"].mean(dim=["chain", "draw"]).values
        )
        others = {k: v for k, v in all_idatas.items() if k != "LogNormal(0.5,0.5)"}

        fig, axes = plt.subplots(1, len(others), figsize=(5 * len(others), 5))
        if len(others) == 1:
            axes = [axes]

        for ax, (label, idata) in zip(axes, others.items()):
            xi_vals = idata.posterior["xi"].mean(dim=["chain", "draw"]).values
            r = float(np.corrcoef(baseline_xi, xi_vals)[0, 1])
            rho = float(stats.spearmanr(baseline_xi, xi_vals).statistic)
            ax.scatter(baseline_xi, xi_vals, s=10, alpha=0.6)
            ax.plot([-2, 3], [-2, 3], "k--", linewidth=0.5)
            ax.set_xlabel("LogNormal ξ")
            ax.set_ylabel(f"{label} ξ")
            ax.set_title(f"{label}\nr={r:.4f}, ρ={rho:.4f}", fontsize=10)

        fig.suptitle("Ideal Points: Variant vs LogNormal Baseline", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / "xi_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: xi_comparison.png")

    # --- Plot 3: Metrics comparison table as figure ---
    metric_keys = [
        ("sampling_time", "Sampling (s)", ".0f"),
        ("divergences", "Divergences", "d"),
        ("xi_rhat_max", "ξ R-hat max", ".4f"),
        ("xi_ess_min", "ξ ESS min", ".0f"),
        ("beta_rhat_max", "β R-hat max", ".4f"),
        ("pca_pearson_r", "PCA Pearson r", ".4f"),
        ("pca_spearman_rho", "PCA Spearman ρ", ".4f"),
        ("holdout_accuracy", "Holdout acc", ".3f"),
        ("holdout_auc", "Holdout AUC", ".3f"),
        ("d_yea_beta_absmean", "D-Yea |β| mean", ".3f"),
        ("r_yea_beta_mean", "R-Yea β mean", ".2f"),
        ("n_negative_beta", "Bills w/ β<0", "d"),
    ]

    labels = list(all_results.keys())
    fig, ax = plt.subplots(figsize=(3 + 2 * len(labels), 0.4 * len(metric_keys) + 1.5))
    ax.axis("off")

    cell_text = []
    row_labels = []
    for key, display_name, fmt in metric_keys:
        row = []
        for label in labels:
            val = all_results[label].get(key, "—")
            if isinstance(val, float):
                row.append(f"{val:{fmt}}")
            elif isinstance(val, int):
                row.append(f"{val:{fmt}}")
            else:
                row.append(str(val))
        cell_text.append(row)
        row_labels.append(display_name)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    fig.suptitle("Beta Prior Experiment: Metrics Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: metrics_comparison.png")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="IRT Beta Prior Experiment")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-tune", type=int, default=300)
    parser.add_argument("--n-chains", type=int, default=2)
    args = parser.parse_args()

    # Resolve paths
    from ks_vote_scraper.session import KSSession

    ks = KSSession.from_session_string(args.session)
    data_dir = Path("data") / ks.output_name
    results_root = Path("results") / ks.output_name
    eda_dir = results_root / "eda" / "latest"
    pca_dir = results_root / "pca" / "latest"
    output_dir = results_root / "irt" / "beta_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  BETA PRIOR EXPERIMENT")
    print(f"  Samples: {args.n_samples}, Tune: {args.n_tune}, Chains: {args.n_chains}")
    print(f"  Output: {output_dir}")
    print("=" * 80)

    # Load data (House only for speed)
    print("\n--- Loading data ---")
    house_matrix, _, _ = load_eda_matrices(eda_dir)
    pca_house, _ = load_pca_scores(pca_dir)
    _, legislators = load_metadata(data_dir)

    print("\n--- Preparing IRT data ---")
    data = prepare_irt_data(house_matrix, "House")
    cons_idx, cons_slug, lib_idx, lib_slug = select_anchors(pca_house, house_matrix, "House")

    # Build PCA-based xi map for bill direction classification
    pca_xi = {row["legislator_slug"]: row["PC1"] for row in pca_house.iter_rows(named=True)}
    bill_directions = classify_bill_direction(house_matrix, data["vote_ids"], pca_xi)
    n_r_yea = sum(1 for d in bill_directions.values() if d == "R-Yea")
    n_d_yea = sum(1 for d in bill_directions.values() if d == "D-Yea")
    print(f"\n  Bill directions: {n_r_yea} R-Yea, {n_d_yea} D-Yea")

    # Run variants
    variants = [
        ("LogNormal(0.5,0.5)", "lognormal_0.5_0.5"),
        ("Normal(0,2.5)", "normal_0_2.5"),
        ("Normal(0,1)", "normal_0_1"),
    ]

    all_results: dict[str, dict] = {}
    all_idatas: dict[str, az.InferenceData] = {}

    for label, prior_key in variants:
        print(f"\n{'=' * 80}")
        print(f"  VARIANT: {label}")
        print(f"{'=' * 80}")

        idata, sampling_time = build_and_sample_variant(
            data,
            cons_idx,
            lib_idx,
            args.n_samples,
            args.n_tune,
            args.n_chains,
            prior_key,
        )

        metrics = extract_metrics(
            idata,
            data,
            pca_house,
            house_matrix,
            legislators,
            bill_directions,
            sampling_time,
        )

        all_results[label] = metrics
        all_idatas[label] = idata

        # Print summary
        print(f"\n  --- {label} Summary ---")
        print(f"  Sampling time: {metrics['sampling_time']:.0f}s")
        print(f"  Divergences: {metrics['divergences']}")
        print(f"  ξ R-hat max: {metrics['xi_rhat_max']:.4f}")
        print(f"  ξ ESS min: {metrics['xi_ess_min']:.0f}")
        print(f"  β R-hat max: {metrics['beta_rhat_max']:.4f}")
        print(f"  PCA r: {metrics['pca_pearson_r']:.4f}, ρ: {metrics['pca_spearman_rho']:.4f}")
        print(
            f"  Holdout: acc={metrics.get('holdout_accuracy', 'N/A'):.3f}, "
            f"AUC={metrics.get('holdout_auc', 'N/A'):.3f}"
        )
        print(f"  D-Yea |β| mean: {metrics.get('d_yea_beta_absmean', 'N/A')}")
        print(f"  R-Yea β mean: {metrics.get('r_yea_beta_mean', 'N/A')}")
        print(f"  Bills with β<0: {metrics.get('n_negative_beta', 0)}")

    # Generate comparison plots
    print(f"\n{'=' * 80}")
    print("  GENERATING COMPARISON PLOTS")
    print(f"{'=' * 80}")
    make_comparison_plots(all_results, all_idatas, data, bill_directions, output_dir)

    # Save metrics as JSON
    # Convert numpy types for JSON serialization
    clean_results = {}
    for label, metrics in all_results.items():
        clean_results[label] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in metrics.items()
        }

    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)
    print("  Saved: experiment_results.json")

    # Print final comparison table
    print(f"\n{'=' * 80}")
    print("  FINAL COMPARISON")
    print(f"{'=' * 80}")
    header = f"  {'Metric':<25s}"
    for label in all_results:
        header += f"  {label:>18s}"
    print(header)
    print("  " + "-" * (25 + 20 * len(all_results)))

    comparison_rows = [
        ("Sampling (s)", "sampling_time", ".0f"),
        ("Divergences", "divergences", "d"),
        ("ξ R-hat max", "xi_rhat_max", ".4f"),
        ("ξ ESS min", "xi_ess_min", ".0f"),
        ("β R-hat max", "beta_rhat_max", ".4f"),
        ("PCA Pearson r", "pca_pearson_r", ".4f"),
        ("PCA Spearman ρ", "pca_spearman_rho", ".4f"),
        ("Holdout acc", "holdout_accuracy", ".3f"),
        ("Holdout AUC", "holdout_auc", ".3f"),
        ("D-Yea |β| mean", "d_yea_beta_absmean", ".3f"),
        ("R-Yea β mean", "r_yea_beta_mean", ".2f"),
        ("Bills β<0", "n_negative_beta", "d"),
    ]

    for display_name, key, fmt in comparison_rows:
        row = f"  {display_name:<25s}"
        for label in all_results:
            val = all_results[label].get(key, "—")
            if isinstance(val, (int, float)):
                row += f"  {val:>18{fmt}}"
            else:
                row += f"  {'—':>18s}"
        print(row)

    print(f"\n  All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
