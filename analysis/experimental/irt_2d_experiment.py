"""
2D Bayesian IRT Experiment: Multidimensional Ideal Points via PLT Identification.

Tests a 2D M2PL IRT model on the Senate chamber to resolve the Tyson paradox —
separating ideology (Dim 1) from contrarianism (Dim 2) in a single Bayesian model.

Identification: Positive Lower Triangular (PLT) constraint on the discrimination matrix.
- beta[0, 1] = 0 (rotation anchor: highest-discrimination bill from 1D IRT)
- beta[1, 1] > 0 (positive diagonal via HalfNormal)
- Dim 1 sign: post-hoc verification that Republican mean is positive

See analysis/design/irt_2d.md for full design and docs/2d-irt-deep-dive.md for motivation.

Usage:
    uv run python analysis/experimental/irt_2d_experiment.py [--session 2025-26]
        [--n-samples 2000] [--n-tune 2000] [--n-chains 4] [--chamber senate]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import nutpie
import polars as pl
import pymc as pm
import pytensor.tensor as pt
from scipy import stats

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

N_SAMPLES = 2000
N_TUNE = 2000
N_CHAINS = 4
TARGET_ACCEPT = 0.95
RANDOM_SEED = 42

# Relaxed convergence thresholds for experimental 2D model
RHAT_THRESHOLD = 1.05
ESS_THRESHOLD = 200
MAX_DIVERGENCES = 50

PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}
CT_TZ = ZoneInfo("America/Chicago")

# Legislators of interest for the Tyson paradox
TYSON_SLUGS = {"sen_tyson_caryn"}
THOMPSON_SLUGS = {"sen_thompson_mike"}
ANNOTATE_SLUGS = TYSON_SLUGS | THOMPSON_SLUGS


# ── Model builder ────────────────────────────────────────────────────────────


def build_2d_irt_model(
    data: dict,
    anchors_1d: list[tuple[int, float]],
    xi_initvals_2d: np.ndarray | None = None,
    n_samples: int = N_SAMPLES,
    n_tune: int = N_TUNE,
    n_chains: int = N_CHAINS,
    target_accept: float = TARGET_ACCEPT,
) -> tuple[az.InferenceData, float]:
    """Build and sample a 2D M2PL IRT model with PLT identification.

    Args:
        data: IRT data dict from prepare_irt_data().
        anchors_1d: 1D anchors [(cons_idx, +1.0), (lib_idx, -1.0)] for reference
            (used to select anchor items, not to constrain legislators).
        xi_initvals_2d: Optional (n_leg, 2) array of initial ideal points from PCA.
        n_samples: MCMC posterior draws per chain.
        n_tune: MCMC tuning steps (discarded).
        n_chains: Number of independent MCMC chains.
        target_accept: NUTS target acceptance probability.

    Returns (InferenceData, sampling_time_seconds).
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
        "dim": ["dim1", "dim2"],
    }

    with pm.Model(coords=coords) as model:
        # ── Legislator ideal points: (n_leg, 2) ──
        xi = pm.Normal("xi", mu=0, sigma=1, shape=(n_leg, 2), dims=("legislator", "dim"))

        # ── Bill difficulty (scalar, same as 1D) ──
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")

        # ── Discrimination: PLT-constrained (n_votes, 2) ──
        # Column 0: unconstrained (all bills load freely on Dim 1)
        beta_col0 = pm.Normal("beta_col0", mu=0, sigma=1, shape=n_votes, dims="vote")

        # Column 1: PLT constraint
        # Item 0: beta[0,1] = 0 (rotation anchor — fixes rotation invariance)
        # Item 1: beta[1,1] > 0 (positive diagonal — fixes Dim 2 sign)
        # Items 2+: free Normal
        beta_anchor_positive = pm.HalfNormal("beta_anchor_positive", sigma=1)
        beta_col1_rest = pm.Normal("beta_col1_rest", mu=0, sigma=1, shape=n_votes - 2)

        beta_col1 = pt.zeros(n_votes)
        beta_col1 = pt.set_subtensor(beta_col1[1], beta_anchor_positive)
        beta_col1 = pt.set_subtensor(beta_col1[2:], beta_col1_rest)

        # Stack into (n_votes, 2) discrimination matrix
        beta = pt.stack([beta_col0, beta_col1], axis=1)
        pm.Deterministic("beta_matrix", beta, dims=("vote", "dim"))

        # ── Likelihood ──
        # For each observation: sum_d(beta[vote_idx, d] * xi[leg_idx, d]) - alpha[vote_idx]
        eta = pt.sum(beta[vote_idx] * xi[leg_idx], axis=1) - alpha[vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

    # ── Compile with nutpie ──
    compile_kwargs: dict = {}
    if xi_initvals_2d is not None:
        compile_kwargs["initial_points"] = {"xi": xi_initvals_2d}
        compile_kwargs["jitter_rvs"] = {rv for rv in model.free_RVs if rv.name != "xi"}
        print(f"  PCA-informed 2D initvals: ({xi_initvals_2d.shape})")
        jittered = [rv.name for rv in compile_kwargs["jitter_rvs"]]
        print(f"  jitter_rvs: {jittered} (xi excluded)")

    print("  Compiling model with nutpie...")
    compiled = nutpie.compile_pymc_model(model, **compile_kwargs)

    # ── Sample ──
    print(f"  Sampling: {n_samples} draws, {n_tune} tune, {n_chains} chains")
    print(f"  seed={RANDOM_SEED}, sampler=nutpie (Rust NUTS)")
    print(f"  Parameters: xi ({n_leg}x2), alpha ({n_votes}), beta ({n_votes}x2 PLT)")
    print(f"  Total free params: ~{n_leg * 2 + n_votes + n_votes + (n_votes - 1)}")

    t0 = time.time()
    idata = nutpie.sample(
        compiled,
        draws=n_samples,
        tune=n_tune,
        chains=n_chains,
        seed=RANDOM_SEED,
        progress_bar=True,
        store_divergences=True,
    )
    sampling_time = time.time() - t0

    print(f"  Sampling complete in {sampling_time:.1f}s ({sampling_time / 60:.1f} min)")
    return idata, sampling_time


# ── Convergence diagnostics ──────────────────────────────────────────────────


def check_2d_convergence(idata: az.InferenceData) -> dict:
    """Run convergence diagnostics for the 2D model."""
    print("\n" + "=" * 80)
    print("  CONVERGENCE DIAGNOSTICS — 2D IRT")
    print("=" * 80)

    diag: dict = {}

    # R-hat
    rhat = az.rhat(idata)
    xi_rhat_max = float(rhat["xi"].max())
    alpha_rhat_max = float(rhat["alpha"].max())
    # beta_matrix is deterministic; check the sampled components
    beta_col0_rhat_max = float(rhat["beta_col0"].max())
    beta_ap_rhat = float(rhat["beta_anchor_positive"].max())

    if "beta_col1_rest" in rhat:
        beta_col1_rhat_max = float(rhat["beta_col1_rest"].max())
    else:
        beta_col1_rhat_max = 0.0

    beta_rhat_max = max(beta_col0_rhat_max, beta_ap_rhat, beta_col1_rhat_max)

    diag["xi_rhat_max"] = xi_rhat_max
    diag["alpha_rhat_max"] = alpha_rhat_max
    diag["beta_rhat_max"] = beta_rhat_max

    for name, val in [
        ("xi", xi_rhat_max),
        ("alpha", alpha_rhat_max),
        ("beta (all)", beta_rhat_max),
    ]:
        status = "OK" if val < RHAT_THRESHOLD else "WARNING"
        print(f"  R-hat ({name}):  max = {val:.4f}  {status}")

    # Bulk ESS
    ess = az.ess(idata)
    xi_ess_min = float(ess["xi"].min())
    alpha_ess_min = float(ess["alpha"].min())
    beta_col0_ess_min = float(ess["beta_col0"].min())

    if "beta_col1_rest" in ess:
        beta_col1_ess_min = float(ess["beta_col1_rest"].min())
    else:
        beta_col1_ess_min = float("inf")

    beta_ess_min = min(beta_col0_ess_min, beta_col1_ess_min)

    diag["xi_ess_min"] = xi_ess_min
    diag["alpha_ess_min"] = alpha_ess_min
    diag["beta_ess_min"] = beta_ess_min

    for name, val in [
        ("xi", xi_ess_min),
        ("alpha", alpha_ess_min),
        ("beta (all)", beta_ess_min),
    ]:
        status = "OK" if val > ESS_THRESHOLD else "WARNING"
        print(f"  Bulk ESS ({name}):  min = {val:.0f}  {status}")

    # Tail ESS
    ess_tail = az.ess(idata, method="tail")
    xi_tail_min = float(ess_tail["xi"].min())
    diag["xi_tail_ess_min"] = xi_tail_min
    tail_status = "OK" if xi_tail_min > ESS_THRESHOLD else "WARNING"
    print(f"  Tail ESS (xi):  min = {xi_tail_min:.0f}  {tail_status}")

    # Divergences
    divergences = int(idata.sample_stats["diverging"].sum().values)
    diag["divergences"] = divergences
    print(f"  Divergences:   {divergences}  {'OK' if divergences < MAX_DIVERGENCES else 'WARNING'}")

    # E-BFMI
    bfmi_values = az.bfmi(idata)
    diag["ebfmi"] = [float(v) for v in bfmi_values]
    for i, v in enumerate(bfmi_values):
        print(f"  E-BFMI chain {i}: {v:.3f}  {'OK' if v > 0.3 else 'WARNING'}")

    diag["all_ok"] = (
        xi_rhat_max < RHAT_THRESHOLD
        and alpha_rhat_max < RHAT_THRESHOLD
        and beta_rhat_max < RHAT_THRESHOLD
        and xi_ess_min > ESS_THRESHOLD
        and divergences < MAX_DIVERGENCES
    )

    if diag["all_ok"]:
        print("  CONVERGENCE: ALL CHECKS PASSED")
    else:
        print("  CONVERGENCE: SOME CHECKS FAILED — inspect diagnostics")

    return diag


# ── Extract ideal points ─────────────────────────────────────────────────────


def extract_2d_ideal_points(
    idata: az.InferenceData,
    data: dict,
    legislators: pl.DataFrame,
) -> pl.DataFrame:
    """Extract 2D ideal point posteriors with HDIs."""
    xi_posterior = idata.posterior["xi"]  # (chain, draw, legislator, dim)

    # Posterior means
    xi_mean = xi_posterior.mean(dim=["chain", "draw"]).values  # (n_leg, 2)

    # HDI
    xi_hdi = az.hdi(idata, var_names=["xi"], hdi_prob=0.95)["xi"].values  # (n_leg, 2, 2)

    # Build DataFrame
    slug_col = "legislator_slug"
    slugs = data["leg_slugs"]

    # Lookup party
    party_map = {}
    for row in legislators.iter_rows(named=True):
        party_map[row["legislator_slug"]] = row.get("party", "Unknown")

    # Lookup display name
    name_map = {}
    for row in legislators.iter_rows(named=True):
        name_map[row["legislator_slug"]] = row.get("full_name", row["legislator_slug"])

    records = []
    for i, slug in enumerate(slugs):
        records.append(
            {
                slug_col: slug,
                "full_name": name_map.get(slug, slug),
                "party": party_map.get(slug, "Unknown"),
                "xi_dim1_mean": float(xi_mean[i, 0]),
                "xi_dim1_hdi_3%": float(xi_hdi[i, 0, 0]),
                "xi_dim1_hdi_97%": float(xi_hdi[i, 0, 1]),
                "xi_dim2_mean": float(xi_mean[i, 1]),
                "xi_dim2_hdi_3%": float(xi_hdi[i, 1, 0]),
                "xi_dim2_hdi_97%": float(xi_hdi[i, 1, 1]),
            }
        )

    return pl.DataFrame(records)


# ── Correlation checks ───────────────────────────────────────────────────────


def correlate_with_1d(
    ideal_2d: pl.DataFrame,
    pca_scores: pl.DataFrame,
    data: dict,
    anchors_1d: list[tuple[int, float]],
    idata_2d: az.InferenceData,
) -> dict:
    """Correlate 2D Dim 1 with PCA PC1 and 2D Dim 2 with PCA PC2."""
    slug_col = "legislator_slug"
    corrs: dict = {}

    # Map from ideal_2d
    dim1_map = dict(zip(ideal_2d[slug_col].to_list(), ideal_2d["xi_dim1_mean"].to_list()))
    dim2_map = dict(zip(ideal_2d[slug_col].to_list(), ideal_2d["xi_dim2_mean"].to_list()))

    # PCA scores
    pca_pc1_map = {row["legislator_slug"]: row["PC1"] for row in pca_scores.iter_rows(named=True)}
    pca_pc2_map = {row["legislator_slug"]: row["PC2"] for row in pca_scores.iter_rows(named=True)}

    # Shared slugs
    shared = [s for s in data["leg_slugs"] if s in pca_pc1_map]

    # Dim 1 vs PCA PC1
    dim1_vals = [dim1_map[s] for s in shared]
    pc1_vals = [pca_pc1_map[s] for s in shared]
    r_dim1_pc1 = float(np.corrcoef(dim1_vals, pc1_vals)[0, 1])
    rho_dim1_pc1 = float(stats.spearmanr(dim1_vals, pc1_vals).statistic)
    corrs["dim1_vs_pc1_pearson"] = r_dim1_pc1
    corrs["dim1_vs_pc1_spearman"] = rho_dim1_pc1

    # Dim 2 vs PCA PC2
    dim2_vals = [dim2_map[s] for s in shared]
    pc2_vals = [pca_pc2_map[s] for s in shared]
    r_dim2_pc2 = float(np.corrcoef(dim2_vals, pc2_vals)[0, 1])
    rho_dim2_pc2 = float(stats.spearmanr(dim2_vals, pc2_vals).statistic)
    corrs["dim2_vs_pc2_pearson"] = r_dim2_pc2
    corrs["dim2_vs_pc2_spearman"] = rho_dim2_pc2

    print("\n" + "=" * 80)
    print("  CORRELATION CHECKS")
    print("=" * 80)
    print(f"  Dim 1 vs PCA PC1:  r = {r_dim1_pc1:.4f},  rho = {rho_dim1_pc1:.4f}")
    print(f"  Dim 2 vs PCA PC2:  r = {r_dim2_pc2:.4f},  rho = {rho_dim2_pc2:.4f}")

    return corrs


# ── Tyson/Thompson check ─────────────────────────────────────────────────────


def check_tyson_thompson(ideal_2d: pl.DataFrame) -> dict:
    """Check if Tyson and Thompson are extreme on Dim 2."""
    slug_col = "legislator_slug"
    results: dict = {}

    # Sort by Dim 2 (most negative = most contrarian, following PC2 convention)
    sorted_dim2 = ideal_2d.sort("xi_dim2_mean")

    print("\n" + "=" * 80)
    print("  TYSON/THOMPSON CHECK — Dim 2 Extremes")
    print("=" * 80)

    # Show top 5 most extreme on Dim 2 (both ends)
    print("\n  Most negative Dim 2 (contrarian direction):")
    for i, row in enumerate(sorted_dim2.head(5).iter_rows(named=True)):
        marker = ""
        if row[slug_col] in TYSON_SLUGS:
            marker = " *** TYSON ***"
        elif row[slug_col] in THOMPSON_SLUGS:
            marker = " *** THOMPSON ***"
        print(
            f"    {i + 1}. {row['full_name']:<25s} ({row['party']:>10s})  "
            f"Dim2 = {row['xi_dim2_mean']:+.3f}  "
            f"[{row['xi_dim2_hdi_3%']:+.3f}, {row['xi_dim2_hdi_97%']:+.3f}]"
            f"{marker}"
        )

    print("\n  Most positive Dim 2:")
    for i, row in enumerate(sorted_dim2.tail(5).reverse().iter_rows(named=True)):
        print(
            f"    {i + 1}. {row['full_name']:<25s} ({row['party']:>10s})  "
            f"Dim2 = {row['xi_dim2_mean']:+.3f}  "
            f"[{row['xi_dim2_hdi_3%']:+.3f}, {row['xi_dim2_hdi_97%']:+.3f}]"
        )

    # Check Tyson
    tyson_rows = ideal_2d.filter(pl.col(slug_col).is_in(list(TYSON_SLUGS)))
    if tyson_rows.height > 0:
        tyson_dim2 = tyson_rows["xi_dim2_mean"][0]
        # Rank: where does Tyson fall? (rank by absolute value)
        abs_dim2 = ideal_2d.with_columns(pl.col("xi_dim2_mean").abs().alias("abs_dim2")).sort(
            "abs_dim2", descending=True
        )
        tyson_rank = None
        for i, row in enumerate(abs_dim2.iter_rows(named=True)):
            if row[slug_col] in TYSON_SLUGS:
                tyson_rank = i + 1
                break
        results["tyson_dim2_mean"] = float(tyson_dim2)
        results["tyson_dim2_rank"] = tyson_rank
        print(f"\n  Tyson Dim 2: {tyson_dim2:+.3f} (rank {tyson_rank} by |Dim 2|)")

    # Check Thompson
    thompson_rows = ideal_2d.filter(pl.col(slug_col).is_in(list(THOMPSON_SLUGS)))
    if thompson_rows.height > 0:
        thompson_dim2 = thompson_rows["xi_dim2_mean"][0]
        thompson_rank = None
        for i, row in enumerate(abs_dim2.iter_rows(named=True)):
            if row[slug_col] in THOMPSON_SLUGS:
                thompson_rank = i + 1
                break
        results["thompson_dim2_mean"] = float(thompson_dim2)
        results["thompson_dim2_rank"] = thompson_rank
        print(f"  Thompson Dim 2: {thompson_dim2:+.3f} (rank {thompson_rank} by |Dim 2|)")

    # Show full ranking on Dim 1
    print("\n  Dim 1 (ideology) ranking — Top 5 most conservative:")
    sorted_dim1 = ideal_2d.sort("xi_dim1_mean", descending=True)
    for i, row in enumerate(sorted_dim1.head(5).iter_rows(named=True)):
        marker = ""
        if row[slug_col] in TYSON_SLUGS:
            marker = " *** TYSON ***"
        elif row[slug_col] in THOMPSON_SLUGS:
            marker = " *** THOMPSON ***"
        print(
            f"    {i + 1}. {row['full_name']:<25s} ({row['party']:>10s})  "
            f"Dim1 = {row['xi_dim1_mean']:+.3f}"
            f"{marker}"
        )

    return results


# ── Plots ────────────────────────────────────────────────────────────────────


def plot_2d_scatter(ideal_2d: pl.DataFrame, output_dir: Path) -> None:
    """2D scatter: Dim 1 vs Dim 2, party-colored, annotated."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for party, color in PARTY_COLORS.items():
        subset = ideal_2d.filter(pl.col("party") == party)
        if subset.height == 0:
            continue
        ax.scatter(
            subset["xi_dim1_mean"].to_list(),
            subset["xi_dim2_mean"].to_list(),
            c=color,
            alpha=0.7,
            s=40,
            label=party,
            edgecolors="white",
            linewidths=0.5,
        )

    # Annotate key legislators
    for row in ideal_2d.iter_rows(named=True):
        slug = row["legislator_slug"]
        if slug in ANNOTATE_SLUGS:
            ax.annotate(
                row["full_name"],
                (row["xi_dim1_mean"], row["xi_dim2_mean"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.8},
            )

    ax.set_xlabel("Dimension 1 (Ideology: Liberal ← → Conservative)", fontsize=11)
    ax.set_ylabel("Dimension 2 (Contrarian ← → Establishment)", fontsize=11)
    ax.set_title(
        "2D Bayesian IRT Ideal Points — Kansas Senate",
        fontsize=13,
        fontweight="bold",
    )
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "2d_scatter.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved: 2d_scatter.png")


def plot_dim1_vs_pc1(ideal_2d: pl.DataFrame, pca_scores: pl.DataFrame, output_dir: Path) -> None:
    """Correlation scatter: 2D Dim 1 vs PCA PC1."""
    slug_col = "legislator_slug"
    pca_map = {row["legislator_slug"]: row["PC1"] for row in pca_scores.iter_rows(named=True)}

    shared = ideal_2d.filter(pl.col(slug_col).is_in(list(pca_map.keys())))
    dim1 = shared["xi_dim1_mean"].to_list()
    pc1 = [pca_map[s] for s in shared[slug_col].to_list()]

    r = float(np.corrcoef(dim1, pc1)[0, 1])

    fig, ax = plt.subplots(figsize=(7, 7))
    for party, color in PARTY_COLORS.items():
        subset = shared.filter(pl.col("party") == party)
        if subset.height == 0:
            continue
        s_dim1 = subset["xi_dim1_mean"].to_list()
        s_pc1 = [pca_map[s] for s in subset[slug_col].to_list()]
        ax.scatter(s_pc1, s_dim1, c=color, alpha=0.7, s=30, label=party)

    ax.set_xlabel("PCA PC1", fontsize=11)
    ax.set_ylabel("2D IRT Dimension 1", fontsize=11)
    ax.set_title(f"2D IRT Dim 1 vs PCA PC1  (r = {r:.4f})", fontsize=12, fontweight="bold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "dim1_vs_pc1.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: dim1_vs_pc1.png (r = {r:.4f})")


def plot_dim2_vs_pc2(ideal_2d: pl.DataFrame, pca_scores: pl.DataFrame, output_dir: Path) -> None:
    """Correlation scatter: 2D Dim 2 vs PCA PC2."""
    slug_col = "legislator_slug"
    pca_map = {row["legislator_slug"]: row["PC2"] for row in pca_scores.iter_rows(named=True)}

    shared = ideal_2d.filter(pl.col(slug_col).is_in(list(pca_map.keys())))
    dim2 = shared["xi_dim2_mean"].to_list()
    pc2 = [pca_map[s] for s in shared[slug_col].to_list()]

    r = float(np.corrcoef(dim2, pc2)[0, 1])

    fig, ax = plt.subplots(figsize=(7, 7))
    for party, color in PARTY_COLORS.items():
        subset = shared.filter(pl.col("party") == party)
        if subset.height == 0:
            continue
        s_dim2 = subset["xi_dim2_mean"].to_list()
        s_pc2 = [pca_map[s] for s in subset[slug_col].to_list()]
        ax.scatter(s_pc2, s_dim2, c=color, alpha=0.7, s=30, label=party)

    ax.set_xlabel("PCA PC2", fontsize=11)
    ax.set_ylabel("2D IRT Dimension 2", fontsize=11)
    ax.set_title(f"2D IRT Dim 2 vs PCA PC2  (r = {r:.4f})", fontsize=12, fontweight="bold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "dim2_vs_pc2.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: dim2_vs_pc2.png (r = {r:.4f})")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="2D IRT Experiment (PLT Identification)")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n-tune", type=int, default=N_TUNE)
    parser.add_argument("--n-chains", type=int, default=N_CHAINS)
    parser.add_argument(
        "--chamber",
        default="senate",
        choices=["house", "senate"],
        help="Chamber to analyze (default: senate, where Tyson is)",
    )
    args = parser.parse_args()

    # Resolve paths
    from tallgrass.session import KSSession

    ks = KSSession.from_session_string(args.session)
    data_dir = ks.data_dir
    results_root = ks.results_dir
    eda_dir = results_root / "01_eda" / "latest"
    pca_dir = results_root / "02_pca" / "latest"

    # Output directory
    today = datetime.now(CT_TZ).strftime("%Y-%m-%d")
    output_dir = Path("results/experimental_lab") / f"{today}_irt-2d"
    # Handle same-day reruns
    if output_dir.exists():
        suffix = 1
        while (Path("results/experimental_lab") / f"{today}_irt-2d.{suffix}").exists():
            suffix += 1
        output_dir = Path("results/experimental_lab") / f"{today}_irt-2d.{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    chamber = args.chamber.capitalize()
    chamber_lower = args.chamber.lower()

    print("=" * 80)
    print("  2D BAYESIAN IRT EXPERIMENT (PLT Identification)")
    print(f"  Session: {args.session}, Chamber: {chamber}")
    print(f"  Samples: {args.n_samples}, Tune: {args.n_tune}, Chains: {args.n_chains}")
    print(f"  Output: {output_dir}")
    print("=" * 80)

    # ── Load data ──
    print("\n--- Loading data ---")
    house_matrix, senate_matrix, _ = load_eda_matrices(eda_dir)
    pca_house, pca_senate = load_pca_scores(pca_dir)
    rollcalls, legislators = load_metadata(data_dir)

    if chamber_lower == "senate":
        matrix = senate_matrix
        pca_scores = pca_senate
    else:
        matrix = house_matrix
        pca_scores = pca_house

    # ── Prepare IRT data ──
    print(f"\n--- Preparing IRT data ({chamber}) ---")
    data = prepare_irt_data(matrix, chamber)

    # ── Select 1D anchors (for reference) ──
    print(f"\n--- 1D Anchor selection ({chamber}) ---")
    cons_idx, cons_slug, lib_idx, lib_slug, _ = select_anchors(pca_scores, matrix, chamber)
    anchors_1d = [(cons_idx, 1.0), (lib_idx, -1.0)]

    # ── Build 2D PCA initialization ──
    print("\n--- Building 2D PCA initvals ---")
    slugs = data["leg_slugs"]

    pca_pc1_map = {row["legislator_slug"]: row["PC1"] for row in pca_scores.iter_rows(named=True)}
    pca_pc2_map = {row["legislator_slug"]: row["PC2"] for row in pca_scores.iter_rows(named=True)}

    pc1_vals = np.array([pca_pc1_map.get(s, 0.0) for s in slugs])
    pc2_vals = np.array([pca_pc2_map.get(s, 0.0) for s in slugs])

    # Standardize
    pc1_std = (pc1_vals - pc1_vals.mean()) / pc1_vals.std() if pc1_vals.std() > 0 else pc1_vals
    pc2_std = (pc2_vals - pc2_vals.mean()) / pc2_vals.std() if pc2_vals.std() > 0 else pc2_vals

    xi_initvals_2d = np.column_stack([pc1_std, pc2_std])
    print(f"  PCA initvals shape: {xi_initvals_2d.shape}")
    print(f"  PC1 range: [{pc1_std.min():.2f}, {pc1_std.max():.2f}]")
    print(f"  PC2 range: [{pc2_std.min():.2f}, {pc2_std.max():.2f}]")

    # ── Build and sample 2D model ──
    print(f"\n--- Building and sampling 2D IRT model ({chamber}) ---")
    t_total = time.time()
    idata, sampling_time = build_2d_irt_model(
        data=data,
        anchors_1d=anchors_1d,
        xi_initvals_2d=xi_initvals_2d,
        n_samples=args.n_samples,
        n_tune=args.n_tune,
        n_chains=args.n_chains,
        target_accept=TARGET_ACCEPT,
    )

    # ── Convergence diagnostics ──
    diag = check_2d_convergence(idata)

    # ── Extract 2D ideal points ──
    print("\n--- Extracting 2D ideal points ---")
    ideal_2d = extract_2d_ideal_points(idata, data, legislators)

    # Post-hoc Dim 1 sign check: Republican mean should be positive
    r_dim1 = ideal_2d.filter(pl.col("party") == "Republican")["xi_dim1_mean"].mean()
    if r_dim1 is not None and r_dim1 < 0:
        print("  WARNING: Republican Dim 1 mean is negative — flipping Dim 1 sign")
        ideal_2d = ideal_2d.with_columns(
            (-pl.col("xi_dim1_mean")).alias("xi_dim1_mean"),
            (-pl.col("xi_dim1_hdi_3%")).alias("xi_dim1_hdi_97%_new"),
            (-pl.col("xi_dim1_hdi_97%")).alias("xi_dim1_hdi_3%_new"),
        ).rename({"xi_dim1_hdi_97%_new": "xi_dim1_hdi_97%", "xi_dim1_hdi_3%_new": "xi_dim1_hdi_3%"})
    else:
        print(f"  Republican Dim 1 mean: {r_dim1:+.3f} (positive — no flip needed)")

    # ── Correlation checks ──
    corrs = correlate_with_1d(ideal_2d, pca_scores, data, anchors_1d, idata)

    # ── Tyson/Thompson check ──
    tyson_results = check_tyson_thompson(ideal_2d)

    # ── Save outputs ──
    print("\n--- Saving outputs ---")
    data_dir_out = output_dir / "data"
    data_dir_out.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save ideal points
    ideal_2d.write_parquet(data_dir_out / "ideal_points_2d.parquet")
    print(f"  Saved: data/ideal_points_2d.parquet ({ideal_2d.height} legislators)")

    # Save convergence summary
    summary = {
        "session": args.session,
        "chamber": chamber,
        "n_samples": args.n_samples,
        "n_tune": args.n_tune,
        "n_chains": args.n_chains,
        "target_accept": TARGET_ACCEPT,
        "sampling_time_s": sampling_time,
        "convergence": diag,
        "correlations": corrs,
        "tyson_thompson": tyson_results,
        "timestamp": datetime.now(CT_TZ).isoformat(),
    }
    with open(data_dir_out / "convergence_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  Saved: data/convergence_summary.json")

    # ── Generate plots ──
    print("\n--- Generating plots ---")
    plot_2d_scatter(ideal_2d, plots_dir)
    plot_dim1_vs_pc1(ideal_2d, pca_scores, plots_dir)
    plot_dim2_vs_pc2(ideal_2d, pca_scores, plots_dir)

    # ── Print full ideal points table ──
    print("\n" + "=" * 80)
    print("  FULL 2D IDEAL POINTS")
    print("=" * 80)
    sorted_table = ideal_2d.sort("xi_dim1_mean", descending=True)
    header = (
        f"\n  {'Name':<25s} {'Party':>10s}  {'Dim1':>7s}  {'Dim2':>7s}"
        f"  {'Dim1 HDI':>18s}  {'Dim2 HDI':>18s}"
    )
    print(header)
    print("  " + "-" * 95)
    for row in sorted_table.iter_rows(named=True):
        print(
            f"  {row['full_name']:<25s} {row['party']:>10s}  "
            f"{row['xi_dim1_mean']:+.3f}  {row['xi_dim2_mean']:+.3f}  "
            f"[{row['xi_dim1_hdi_3%']:+.3f}, {row['xi_dim1_hdi_97%']:+.3f}]  "
            f"[{row['xi_dim2_hdi_3%']:+.3f}, {row['xi_dim2_hdi_97%']:+.3f}]"
        )

    # ── Final summary ──
    total_time = time.time() - t_total
    print("\n" + "=" * 80)
    print("  EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"  Chamber: {chamber}")
    print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Sampling time: {sampling_time:.0f}s ({sampling_time / 60:.1f} min)")
    print(f"  Convergence: {'PASSED' if diag['all_ok'] else 'FAILED'}")
    print(f"  Dim 1 vs PCA PC1: r = {corrs['dim1_vs_pc1_pearson']:.4f}")
    print(f"  Dim 2 vs PCA PC2: r = {corrs['dim2_vs_pc2_pearson']:.4f}")
    if "tyson_dim2_rank" in tyson_results:
        print(f"  Tyson Dim 2 rank (by |Dim 2|): {tyson_results['tyson_dim2_rank']}")
    if "thompson_dim2_rank" in tyson_results:
        print(f"  Thompson Dim 2 rank (by |Dim 2|): {tyson_results['thompson_dim2_rank']}")
    print(f"  Output: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
