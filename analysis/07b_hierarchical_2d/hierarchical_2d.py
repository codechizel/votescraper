"""
Kansas Legislature — Hierarchical 2D IRT Ideal Point Estimation (Phase 07b)

Combines 2D structure (M2PL with PLT identification, same as Phase 06) with
party-level partial pooling (same as Phase 07). Informative priors from both
upstream phases: party means from Phase 07, Dim 2 party averages from Phase 06.

Resolves horseshoe without Dim 2 convergence issues: party pooling regularizes
the second dimension while PLT identification constrains rotation.

Usage:
  uv run python analysis/07b_hierarchical_2d/hierarchical_2d.py [--session 2025-26]
      [--run-id ...] [--n-samples 2000] [--n-tune 2000] [--n-chains 4]
      [--init-strategy {auto,pca-informed}] [--contested-only]

Outputs (in results/<session>/<run_id>/07b_hierarchical_2d/):
  - data/:   ideal_points_h2d_{chamber}.parquet, group_params_{chamber}.parquet,
             convergence_summary.json, idata_{chamber}.nc
  - plots/:  2d_scatter_{chamber}.png, party_posteriors_{chamber}.png,
             shrinkage_vs_flat2d_{chamber}.png
  - 07b_hierarchical_2d_report.html
"""

import argparse
import json
import sys
import time
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir

try:
    from analysis.phase_utils import load_metadata, print_header, save_fig
except ModuleNotFoundError:
    from phase_utils import load_metadata, print_header, save_fig

try:
    from analysis.irt import (
        PARTY_COLORS,
        RANDOM_SEED,
        load_eda_matrices,
        load_pca_scores,
    )
except ModuleNotFoundError:
    from irt import (  # type: ignore[no-redef]
        PARTY_COLORS,
        RANDOM_SEED,
        load_eda_matrices,
        load_pca_scores,
    )

try:
    from analysis.hierarchical import prepare_hierarchical_data
except ModuleNotFoundError:
    from hierarchical import prepare_hierarchical_data  # type: ignore[no-redef]

try:
    from analysis.irt_2d import compute_beta_init_from_pca
except ModuleNotFoundError:
    from irt_2d import compute_beta_init_from_pca  # type: ignore[no-redef]

try:
    from analysis.init_strategy import resolve_init_source
except ModuleNotFoundError:
    from init_strategy import resolve_init_source  # type: ignore[no-redef]

try:
    from analysis.hierarchical_2d_report import build_hierarchical_2d_report
except ModuleNotFoundError:
    from hierarchical_2d_report import build_hierarchical_2d_report  # type: ignore[no-redef]

try:
    from analysis.tuning import SUPERMAJORITY_THRESHOLD
except ModuleNotFoundError:
    from tuning import SUPERMAJORITY_THRESHOLD  # type: ignore[no-redef]

# ── Primer ──────────────────────────────────────────────────────────────────

H2D_PRIMER = """\
# Hierarchical 2D IRT Ideal Point Estimation

## Purpose

Combines the strengths of Phase 06 (2D IRT) and Phase 07 (Hierarchical 1D IRT):
- **2D structure** resolves horseshoe artifacts by separating ideology (Dim 1)
  from the establishment–contrarian axis (Dim 2)
- **Party pooling** regularizes sparse legislators and stabilizes Dim 2
  convergence through informative priors

Phase 06 (flat 2D) has Dim 2 convergence issues; Phase 07 (hierarchical 1D)
can't resolve horseshoe. This phase fills the gap.

## Method

### Hierarchical M2PL IRT with PLT Identification

Party-level (per dimension):
  mu_party_dim1_raw ~ Normal(mu_07, 1.0)       ← informative from Phase 07
  mu_party_dim1 = sort(mu_party_dim1_raw)       ← D < R identification
  mu_party_dim2 ~ Normal(dim2_party_avg, 2.0)   ← soft prior from Phase 06
  sigma_party ~ HalfNormal(sigma_scale)          ← adaptive for small groups

Legislator (non-centered, per dimension):
  xi_offset_dim1 ~ Normal(0, 1)
  xi_offset_dim2 ~ Normal(0, 1)
  xi_dim1 = mu_party_dim1[party] + sigma_party_dim1[party] * xi_offset_dim1
  xi_dim2 = mu_party_dim2[party] + sigma_party_dim2[party] * xi_offset_dim2
  xi = stack([xi_dim1, xi_dim2], axis=1)

Bill parameters (PLT, same as Phase 06):
  alpha ~ Normal(0, 5)
  beta_col0 ~ Normal(0, 1)                      ← Dim 1 discrimination
  beta_col1: PLT-constrained                     ← [0]=0, [1]=HalfNormal, rest=Normal
  beta = stack([beta_col0, beta_col1], axis=1)

Likelihood (M2PL):
  eta = sum_d(beta[j,d] * xi[i,d]) - alpha[j]
  y ~ Bernoulli(logit(eta))

## Inputs
- Phase 01 (EDA): filtered vote matrices
- Phase 02 (PCA): PC1/PC2 for initialization
- Phase 06 (2D IRT): Dim 2 party averages
- Phase 07 (Hierarchical 1D): mu_party, sigma_within

## Outputs
- `ideal_points_h2d_{chamber}.parquet` — 2D ideal points with party pooling
- `group_params_{chamber}.parquet` — Per-party per-dimension posteriors
- `convergence_summary.json` — Convergence diagnostics
- `idata_{chamber}.nc` — Full posterior (ArviZ NetCDF)

## Interpretation Guide
- **Dim 1** (x-axis): Ideology, party-pooled. Positive = conservative.
- **Dim 2** (y-axis): Secondary axis (establishment–contrarian), party-pooled.
- Party pooling shrinks sparse legislators toward their party mean per dimension.
- Convergence should improve over flat 2D (Phase 06) due to informative priors.

## Caveats
- Requires Phase 06 and Phase 07 output. Gracefully skips if missing.
- Uses same relaxed convergence thresholds as Phase 06.
- Supermajority tuning (ADR-0112): N_TUNE doubles when majority > 70%.
"""

# ── Constants ────────────────────────────────────────────────────────────────

N_SAMPLES = 2000
N_TUNE = 2000
N_TUNE_SUPERMAJORITY = 4000  # ADR-0112
N_CHAINS = 4

# Relaxed convergence thresholds (same as Phase 06)
H2D_RHAT_THRESHOLD = 1.05
H2D_ESS_THRESHOLD = 200
MAX_DIVERGENCES = 50

# Party handling (same as Phase 07)
PARTY_NAMES = ["Democrat", "Republican"]
PARTY_IDX_MAP = {"Democrat": 0, "Republican": 1}
SMALL_GROUP_THRESHOLD = 20
SMALL_GROUP_SIGMA_SCALE = 0.5

CHAMBERS = ["House", "Senate"]


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical 2D IRT — Party-Pooled Multidimensional Ideal Points (Phase 07b)"
    )
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override flat IRT results directory")
    parser.add_argument("--irt-2d-dir", default=None, help="Override 2D IRT results directory")
    parser.add_argument(
        "--hierarchical-dir", default=None, help="Override hierarchical IRT results directory"
    )
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--init-strategy",
        default="pca-informed",
        choices=["auto", "pca-informed"],
        help="Dim 1 initialization source (default: pca-informed — avoids horseshoe contamination)",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n-tune", type=int, default=N_TUNE)
    parser.add_argument("--n-chains", type=int, default=N_CHAINS)
    parser.add_argument(
        "--contested-only",
        action="store_true",
        help="Filter to contested votes before H2D IRT (ADR-0112)",
    )
    parser.add_argument("--csv", action="store_true", help="Force CSV-only mode (skip PostgreSQL)")
    return parser.parse_args()


# ── Model builder ────────────────────────────────────────────────────────────


def build_hierarchical_2d_graph(
    data: dict,
    *,
    mu_dim1_prior: np.ndarray | None = None,
    mu_dim2_prior: np.ndarray | None = None,
) -> pm.Model:
    """Build hierarchical 2D M2PL IRT model with PLT identification.

    Combines party pooling (Phase 07 structure) with 2D PLT (Phase 06 structure).

    Args:
        data: Hierarchical data dict from prepare_hierarchical_data().
        mu_dim1_prior: Per-party Dim 1 prior means from Phase 07 (shape: n_parties).
        mu_dim2_prior: Per-party Dim 2 prior means from Phase 06 (shape: n_parties).

    Returns the PyMC model for use with nutpie.
    """
    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y = data["y"]
    n_leg = data["n_legislators"]
    n_votes = data["n_votes"]
    party_idx = data["party_idx"]
    n_parties = data["n_parties"]

    # Group-size-adaptive priors (Gelman 2015)
    party_counts = np.array([int((party_idx == p).sum()) for p in range(n_parties)])
    sigma_scale = np.array(
        [
            SMALL_GROUP_SIGMA_SCALE if party_counts[p] < SMALL_GROUP_THRESHOLD else 1.0
            for p in range(n_parties)
        ]
    )
    for p in range(n_parties):
        if party_counts[p] < SMALL_GROUP_THRESHOLD:
            print(
                f"  Adaptive prior: {data['party_names'][p]} ({party_counts[p]} members) → "
                f"sigma ~ HalfNormal({SMALL_GROUP_SIGMA_SCALE})"
            )

    # Dim 1 prior: informative from Phase 07 if available, else diffuse
    dim1_mu_prior = mu_dim1_prior if mu_dim1_prior is not None else np.zeros(n_parties)
    dim1_sigma_prior = 1.0 if mu_dim1_prior is not None else 2.0

    # Dim 2 prior: soft informative from Phase 06 if available, else diffuse
    dim2_mu_prior = mu_dim2_prior if mu_dim2_prior is not None else np.zeros(n_parties)
    dim2_sigma_prior = 2.0  # wider — Dim 2 has weaker signal

    coords = {
        "legislator": data["leg_slugs"],
        "vote": data["vote_ids"],
        "party": data["party_names"],
        "obs_id": np.arange(data["n_obs"]),
        "dim": ["dim1", "dim2"],
    }

    with pm.Model(coords=coords) as model:
        # ── Party-level parameters (per dimension) ──

        # Dim 1: sort for identification (D < R)
        mu_party_dim1_raw = pm.Normal(
            "mu_party_dim1_raw", mu=dim1_mu_prior, sigma=dim1_sigma_prior, shape=n_parties
        )
        mu_party_dim1 = pm.Deterministic("mu_party_dim1", pt.sort(mu_party_dim1_raw), dims="party")

        # Soft minimum-separation guard (same as Phase 07, R4)
        pm.Potential(
            "min_party_sep_dim1",
            pt.switch(mu_party_dim1[1] - mu_party_dim1[0] > 0.5, 0.0, -100.0),
        )

        # Dim 2: no ordering constraint (secondary axis)
        mu_party_dim2 = pm.Normal(
            "mu_party_dim2", mu=dim2_mu_prior, sigma=dim2_sigma_prior, shape=n_parties, dims="party"
        )

        # Per-party within-group SD (adaptive for small groups)
        sigma_party_dim1 = pm.HalfNormal(
            "sigma_party_dim1", sigma=sigma_scale, shape=n_parties, dims="party"
        )
        sigma_party_dim2 = pm.HalfNormal(
            "sigma_party_dim2", sigma=sigma_scale, shape=n_parties, dims="party"
        )

        # ── Non-centered legislator ideal points (per dimension) ──
        xi_offset_dim1 = pm.Normal("xi_offset_dim1", mu=0, sigma=1, shape=n_leg, dims="legislator")
        xi_offset_dim2 = pm.Normal("xi_offset_dim2", mu=0, sigma=1, shape=n_leg, dims="legislator")

        xi_dim1 = pm.Deterministic(
            "xi_dim1",
            mu_party_dim1[party_idx] + sigma_party_dim1[party_idx] * xi_offset_dim1,
            dims="legislator",
        )
        xi_dim2 = pm.Deterministic(
            "xi_dim2",
            mu_party_dim2[party_idx] + sigma_party_dim2[party_idx] * xi_offset_dim2,
            dims="legislator",
        )

        # Stack into (n_leg, 2)
        xi = pm.Deterministic(
            "xi", pt.stack([xi_dim1, xi_dim2], axis=1), dims=("legislator", "dim")
        )

        # ── Bill parameters (PLT, same as Phase 06) ──
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")

        # Column 0: unconstrained (Dim 1 discrimination)
        beta_col0 = pm.Normal("beta_col0", mu=0, sigma=1, shape=n_votes, dims="vote")

        # Column 1: PLT constraint
        beta_anchor_positive = pm.HalfNormal("beta_anchor_positive", sigma=1)
        beta_col1_rest = pm.Normal("beta_col1_rest", mu=0, sigma=1, shape=n_votes - 2)

        beta_col1 = pt.zeros(n_votes)
        beta_col1 = pt.set_subtensor(beta_col1[1], beta_anchor_positive)
        beta_col1 = pt.set_subtensor(beta_col1[2:], beta_col1_rest)

        beta = pt.stack([beta_col0, beta_col1], axis=1)
        pm.Deterministic("beta_matrix", beta, dims=("vote", "dim"))

        # ── Likelihood (M2PL) ──
        eta = pt.sum(beta[vote_idx] * xi[leg_idx], axis=1) - alpha[vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

    return model


def build_and_sample_h2d(
    data: dict,
    *,
    mu_dim1_prior: np.ndarray | None = None,
    mu_dim2_prior: np.ndarray | None = None,
    xi_initvals_2d: np.ndarray | None = None,
    beta_init: tuple[np.ndarray, float] | None = None,
    n_samples: int = N_SAMPLES,
    n_tune: int = N_TUNE,
    n_chains: int = N_CHAINS,
) -> tuple[az.InferenceData, float]:
    """Build hierarchical 2D IRT, compile with nutpie, and sample.

    Args:
        data: Hierarchical data dict from prepare_hierarchical_data().
        mu_dim1_prior: Per-party Dim 1 prior means from Phase 07.
        mu_dim2_prior: Per-party Dim 2 prior means from Phase 06.
        xi_initvals_2d: Optional (n_leg, 2) PCA init for xi offsets.
        beta_init: Optional (beta_col0_init, beta_anchor_positive_init) from PCA.
        n_samples: MCMC posterior draws per chain.
        n_tune: MCMC tuning steps (discarded).
        n_chains: Number of independent MCMC chains.

    Returns (InferenceData, sampling_time_seconds).
    """
    n_leg = data["n_legislators"]
    n_votes = data["n_votes"]

    model = build_hierarchical_2d_graph(
        data, mu_dim1_prior=mu_dim1_prior, mu_dim2_prior=mu_dim2_prior
    )

    # --- Compile with nutpie ---
    compile_kwargs: dict = {}
    initial_points: dict = {}
    no_jitter_rvs: set[str] = set()

    if xi_initvals_2d is not None:
        # Split 2D init into per-dimension offsets
        initial_points["xi_offset_dim1"] = xi_initvals_2d[:, 0]
        initial_points["xi_offset_dim2"] = xi_initvals_2d[:, 1]
        no_jitter_rvs.update({"xi_offset_dim1", "xi_offset_dim2"})
        print(f"  PCA-informed 2D initvals: ({xi_initvals_2d.shape})")

    if beta_init is not None:
        beta_col0_init, beta_anchor_positive_init = beta_init
        initial_points["beta_col0"] = beta_col0_init
        initial_points["beta_anchor_positive"] = np.array(beta_anchor_positive_init)
        no_jitter_rvs.update({"beta_col0", "beta_anchor_positive"})
        lo, hi = beta_col0_init.min(), beta_col0_init.max()
        print(f"  PCA-informed beta_col0 init: range [{lo:.3f}, {hi:.3f}]")
        print(f"  PCA-informed beta_anchor_positive init: {beta_anchor_positive_init:.3f}")

    if initial_points:
        compile_kwargs["initial_points"] = initial_points
        # Jitter everything EXCEPT initialized vars
        # CRITICAL: never set jitter_rvs=set() — HalfNormal support point → log(0)=-inf
        compile_kwargs["jitter_rvs"] = {rv for rv in model.free_RVs if rv.name not in no_jitter_rvs}
        jittered = [rv.name for rv in compile_kwargs["jitter_rvs"]]
        print(f"  jitter_rvs: {jittered} ({', '.join(no_jitter_rvs)} excluded)")

    print("  Compiling model with nutpie...")
    compiled = nutpie.compile_pymc_model(model, **compile_kwargs)

    # --- Sample ---
    print(f"  Sampling: {n_samples} draws, {n_tune} tune, {n_chains} chains")
    print(f"  seed={RANDOM_SEED}, sampler=nutpie (Rust NUTS)")
    print(f"  Parameters: xi ({n_leg}x2 hierarchical), alpha ({n_votes}), beta ({n_votes}x2 PLT)")

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


def check_h2d_convergence(idata: az.InferenceData, chamber: str) -> dict:
    """Run convergence diagnostics for the hierarchical 2D model."""
    print_header(f"CONVERGENCE DIAGNOSTICS — H2D ({chamber})")

    diag: dict = {}

    # R-hat
    rhat = az.rhat(idata)
    for var in (
        "xi",
        "alpha",
        "beta_col0",
        "mu_party_dim1",
        "mu_party_dim2",
        "sigma_party_dim1",
        "sigma_party_dim2",
    ):
        if var in rhat:
            max_rhat = float(rhat[var].max())
            diag[f"{var}_rhat_max"] = max_rhat
            status = "OK" if max_rhat < H2D_RHAT_THRESHOLD else "WARNING"
            print(f"  R-hat ({var}):  max = {max_rhat:.4f}  {status}")

    if "beta_anchor_positive" in rhat:
        beta_ap_rhat = float(rhat["beta_anchor_positive"].max())
        diag["beta_anchor_positive_rhat_max"] = beta_ap_rhat
    if "beta_col1_rest" in rhat:
        beta_col1_rhat = float(rhat["beta_col1_rest"].max())
        diag["beta_col1_rest_rhat_max"] = beta_col1_rhat

    xi_rhat_max = diag.get("xi_rhat_max", 999.0)

    # ESS
    ess = az.ess(idata)
    for var in (
        "xi",
        "alpha",
        "mu_party_dim1",
        "mu_party_dim2",
        "sigma_party_dim1",
        "sigma_party_dim2",
    ):
        if var in ess:
            min_ess = float(ess[var].min())
            diag[f"{var}_ess_min"] = min_ess
            status = "OK" if min_ess > H2D_ESS_THRESHOLD else "WARNING"
            print(f"  Bulk ESS ({var}):  min = {min_ess:.0f}  {status}")

    xi_ess_min = diag.get("xi_ess_min", 0.0)

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
        xi_rhat_max < H2D_RHAT_THRESHOLD
        and xi_ess_min > H2D_ESS_THRESHOLD
        and divergences < MAX_DIVERGENCES
    )

    if diag["all_ok"]:
        print("  CONVERGENCE: ALL CHECKS PASSED (relaxed thresholds)")
    else:
        print("  CONVERGENCE: SOME CHECKS FAILED — inspect diagnostics")

    return diag


# ── Extract results ──────────────────────────────────────────────────────────


def extract_h2d_ideal_points(
    idata: az.InferenceData,
    data: dict,
    legislators: pl.DataFrame,
) -> pl.DataFrame:
    """Extract 2D ideal point posteriors with party information."""
    xi_posterior = idata.posterior["xi"]  # (chain, draw, legislator, dim)
    xi_mean = xi_posterior.mean(dim=["chain", "draw"]).values  # (n_leg, 2)
    xi_hdi = az.hdi(idata, var_names=["xi"], hdi_prob=0.95)["xi"].values  # (n_leg, 2, 2)

    slugs = data["leg_slugs"]
    party_map = {}
    name_map = {}
    for row in legislators.iter_rows(named=True):
        party_map[row["legislator_slug"]] = row.get("party", "Unknown")
        name_map[row["legislator_slug"]] = row.get("full_name", row["legislator_slug"])

    records = []
    for i, slug in enumerate(slugs):
        records.append(
            {
                "legislator_slug": slug,
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


def extract_h2d_group_params(
    idata: az.InferenceData,
    data: dict,
) -> pl.DataFrame:
    """Extract per-party per-dimension group parameters."""
    party_names = data["party_names"]
    records = []

    for dim_name, mu_var, sigma_var in [
        ("dim1", "mu_party_dim1", "sigma_party_dim1"),
        ("dim2", "mu_party_dim2", "sigma_party_dim2"),
    ]:
        mu_post = idata.posterior[mu_var]
        sigma_post = idata.posterior[sigma_var]

        mu_mean = mu_post.mean(dim=["chain", "draw"]).values
        mu_hdi = az.hdi(idata, var_names=[mu_var], hdi_prob=0.95)[mu_var].values
        sigma_mean = sigma_post.mean(dim=["chain", "draw"]).values
        sigma_hdi = az.hdi(idata, var_names=[sigma_var], hdi_prob=0.95)[sigma_var].values

        for p_idx, party in enumerate(party_names):
            records.append(
                {
                    "party": party,
                    "dimension": dim_name,
                    "mu_mean": float(mu_mean[p_idx]),
                    "mu_hdi_2.5": float(mu_hdi[p_idx, 0]),
                    "mu_hdi_97.5": float(mu_hdi[p_idx, 1]),
                    "sigma_within_mean": float(sigma_mean[p_idx]),
                    "sigma_within_hdi_2.5": float(sigma_hdi[p_idx, 0]),
                    "sigma_within_hdi_97.5": float(sigma_hdi[p_idx, 1]),
                }
            )

    return pl.DataFrame(records)


def apply_dim1_sign_check(
    idata: az.InferenceData,
    ideal_h2d: pl.DataFrame,
) -> tuple[az.InferenceData, pl.DataFrame]:
    """Flip Dim 1 sign if Republican mean is negative (same as Phase 06)."""
    r_dim1 = ideal_h2d.filter(pl.col("party") == "Republican")["xi_dim1_mean"].mean()
    if r_dim1 is not None and r_dim1 < 0:
        print("  WARNING: Republican Dim 1 mean is negative — flipping Dim 1 sign")

        # Flip in posterior
        xi_post = idata.posterior["xi"].values  # (chain, draw, leg, 2)
        xi_post[:, :, :, 0] *= -1
        idata.posterior["xi"].values = xi_post

        # Flip xi_dim1 deterministic
        if "xi_dim1" in idata.posterior:
            idata.posterior["xi_dim1"].values *= -1

        # Flip in DataFrame
        ideal_h2d = (
            ideal_h2d.with_columns(
                (-pl.col("xi_dim1_mean")).alias("xi_dim1_mean"),
                (-pl.col("xi_dim1_hdi_3%")).alias("xi_dim1_hdi_97%_new"),
                (-pl.col("xi_dim1_hdi_97%")).alias("xi_dim1_hdi_3%_new"),
            )
            .drop("xi_dim1_hdi_3%", "xi_dim1_hdi_97%")
            .rename(
                {
                    "xi_dim1_hdi_97%_new": "xi_dim1_hdi_97%",
                    "xi_dim1_hdi_3%_new": "xi_dim1_hdi_3%",
                }
            )
        )
    else:
        print(f"  Republican Dim 1 mean: {r_dim1:+.3f} (positive — no flip needed)")
    return idata, ideal_h2d


# ── Plots ────────────────────────────────────────────────────────────────────


def plot_2d_scatter(
    ideal_h2d: pl.DataFrame,
    chamber: str,
    output_dir: Path,
    init_label: str | None = None,
) -> None:
    """2D scatter: Dim 1 vs Dim 2, party-colored."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for party, color in PARTY_COLORS.items():
        subset = ideal_h2d.filter(pl.col("party") == party)
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

    ax.set_xlabel("Dimension 1 (Ideology: Liberal ← → Conservative)", fontsize=11)
    ax.set_ylabel("Dimension 2 (Secondary Axis)", fontsize=11)
    title = f"Hierarchical 2D IRT Ideal Points — Kansas {chamber}"
    if init_label:
        title += f"\nInit: {init_label}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_fig(fig, output_dir / f"2d_scatter_{chamber.lower()}.png")


def plot_party_posteriors(
    idata: az.InferenceData,
    data: dict,
    chamber: str,
    output_dir: Path,
) -> None:
    """Party mean posteriors for both dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    party_names = data["party_names"]

    for ax, (dim_name, mu_var) in zip(
        axes, [("Dim 1 (Ideology)", "mu_party_dim1"), ("Dim 2 (Secondary)", "mu_party_dim2")]
    ):
        mu_post = idata.posterior[mu_var].values  # (chain, draw, party)
        for p_idx, party in enumerate(party_names):
            color = PARTY_COLORS.get(party, "#888888")
            samples = mu_post[:, :, p_idx].flatten()
            ax.hist(samples, bins=50, alpha=0.6, color=color, label=party, density=True)
        ax.set_xlabel(f"Party Mean ({dim_name})")
        ax.set_ylabel("Density")
        ax.set_title(f"{chamber} — {dim_name}")
        ax.legend()

    fig.suptitle(f"Party Mean Posteriors — {chamber}", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir / f"party_posteriors_{chamber.lower()}.png")


def plot_shrinkage_vs_flat2d(
    ideal_h2d: pl.DataFrame,
    flat_2d: pl.DataFrame | None,
    chamber: str,
    output_dir: Path,
) -> None:
    """Shrinkage scatter: H2D vs Flat 2D ideal points."""
    if flat_2d is None:
        return

    merged = ideal_h2d.join(
        flat_2d.select(
            "legislator_slug",
            pl.col("xi_dim1_mean").alias("flat_dim1"),
            pl.col("xi_dim2_mean").alias("flat_dim2"),
        ),
        on="legislator_slug",
        how="inner",
    )

    if merged.height == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (h2d_col, flat_col, dim_label) in zip(
        axes,
        [("xi_dim1_mean", "flat_dim1", "Dim 1"), ("xi_dim2_mean", "flat_dim2", "Dim 2")],
    ):
        for party, color in PARTY_COLORS.items():
            subset = merged.filter(pl.col("party") == party)
            if subset.height == 0:
                continue
            ax.scatter(
                subset[flat_col].to_list(),
                subset[h2d_col].to_list(),
                c=color,
                alpha=0.6,
                s=25,
                label=party,
            )

        lims = [
            min(merged[flat_col].min(), merged[h2d_col].min()) - 0.5,
            max(merged[flat_col].max(), merged[h2d_col].max()) + 0.5,
        ]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel(f"Flat 2D {dim_label}")
        ax.set_ylabel(f"Hierarchical 2D {dim_label}")
        ax.set_title(f"{dim_label} Shrinkage")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Shrinkage: Hierarchical 2D vs Flat 2D — {chamber}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_fig(fig, output_dir / f"shrinkage_vs_flat2d_{chamber.lower()}.png")


# ── Upstream data loading ────────────────────────────────────────────────────


def load_phase07_priors(hier_dir: Path, chamber: str) -> tuple[np.ndarray | None, float | None]:
    """Load party means from Phase 07 group_params for Dim 1 prior.

    Returns (mu_party_array, sigma_within_mean) or (None, None) if unavailable.
    mu_party_array shape: (2,) — [Democrat, Republican] sorted.
    """
    path = hier_dir / "data" / f"group_params_{chamber.lower()}.parquet"
    if not path.exists():
        return None, None

    gp = pl.read_parquet(path)
    mu_vals = []
    sigma_vals = []
    for party in PARTY_NAMES:
        row = gp.filter(pl.col("party") == party)
        if row.height == 0:
            return None, None
        mu_vals.append(float(row["mu_mean"][0]))
        sigma_vals.append(float(row["sigma_within_mean"][0]))

    return np.array(mu_vals), float(np.mean(sigma_vals))


def load_phase06_dim2_party_averages(irt_2d_dir: Path, chamber: str) -> np.ndarray | None:
    """Load Dim 2 party averages from Phase 06 ideal points for Dim 2 prior.

    Returns array of shape (2,) — [Democrat_dim2_mean, Republican_dim2_mean].
    """
    path = irt_2d_dir / "data" / f"ideal_points_2d_{chamber.lower()}.parquet"
    if not path.exists():
        return None

    ip = pl.read_parquet(path)
    avgs = []
    for party in PARTY_NAMES:
        subset = ip.filter(pl.col("party") == party)
        if subset.height == 0:
            return None
        avgs.append(float(subset["xi_dim2_mean"].mean()))

    return np.array(avgs)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from tallgrass.session import KSSession

    ks = KSSession.from_session_string(args.session)
    results_root = ks.results_dir
    data_dir = Path(args.data_dir) if args.data_dir else ks.data_dir

    eda_dir = resolve_upstream_dir(
        "01_eda",
        results_root,
        args.run_id,
        Path(args.eda_dir) if args.eda_dir else None,
    )
    pca_dir = resolve_upstream_dir(
        "02_pca",
        results_root,
        args.run_id,
        Path(args.pca_dir) if args.pca_dir else None,
    )
    irt_2d_dir = resolve_upstream_dir(
        "06_irt_2d",
        results_root,
        args.run_id,
        Path(args.irt_2d_dir) if args.irt_2d_dir else None,
    )
    hier_dir = resolve_upstream_dir(
        "07_hierarchical",
        results_root,
        args.run_id,
        Path(args.hierarchical_dir) if args.hierarchical_dir else None,
    )

    # Graceful skip if prerequisites missing
    irt_2d_exists = (irt_2d_dir / "data").exists()
    hier_exists = (hier_dir / "data").exists()
    if not irt_2d_exists and not hier_exists:
        print(
            "Phase 07b (Hierarchical 2D): skipping — "
            "both Phase 06 (2D IRT) and Phase 07 (Hierarchical) output missing"
        )
        return

    with RunContext(
        session=args.session,
        analysis_name="07b_hierarchical_2d",
        params=vars(args),
        primer=H2D_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"KS Legislature Hierarchical 2D IRT — Session {args.session}")
        print(f"Data:           {data_dir}")
        print(f"EDA:            {eda_dir}")
        print(f"PCA:            {pca_dir}")
        print(f"2D IRT:         {irt_2d_dir}")
        print(f"Hierarchical:   {hier_dir}")
        print(f"Output:         {ctx.run_dir}")

        # ── Load data ──
        print_header("LOADING DATA")
        eda_house_path = eda_dir / "data" / "vote_matrix_house_filtered.parquet"
        eda_senate_path = eda_dir / "data" / "vote_matrix_senate_filtered.parquet"
        if not eda_house_path.exists() and not eda_senate_path.exists():
            print("Phase 07b: skipping — no EDA vote matrices available")
            return
        house_matrix, senate_matrix, full_matrix = load_eda_matrices(eda_dir)
        pca_house_path = pca_dir / "data" / "pc_scores_house.parquet"
        if pca_house_path.exists():
            house_pca, senate_pca = load_pca_scores(pca_dir)
        else:
            print("  PCA scores not available — skipping PCA-informed initialization")
            house_pca, senate_pca = None, None
        rollcalls, legislators = load_metadata(data_dir)

        # ── Process chambers ──
        convergence_summary: dict = {"chambers": {}}
        all_chamber_results: dict[str, dict] = {}

        for chamber, matrix, pca_scores in [
            ("House", house_matrix, house_pca),
            ("Senate", senate_matrix, senate_pca),
        ]:
            ch = chamber.lower()
            print_header(f"HIERARCHICAL 2D IRT — {chamber}")

            if matrix.height < 5:
                print(f"  Skipping {chamber}: too few legislators ({matrix.height})")
                continue

            # Prepare hierarchical data (filters non-major-party legislators)
            data = prepare_hierarchical_data(matrix, legislators, chamber)

            # ── Supermajority tuning (ADR-0112) ──
            party_counts = np.array(
                [int((data["party_idx"] == p).sum()) for p in range(data["n_parties"])]
            )
            majority_frac = party_counts.max() / party_counts.sum()
            n_tune = args.n_tune
            if majority_frac > SUPERMAJORITY_THRESHOLD:
                n_tune = max(n_tune, N_TUNE_SUPERMAJORITY)
                print(
                    f"  Supermajority detected ({majority_frac:.1%}) — doubling N_TUNE to {n_tune}"
                )

            # ── Load upstream priors ──
            print_header(f"LOADING UPSTREAM PRIORS — {chamber}")

            # Phase 07 → Dim 1 party means
            mu_dim1_prior, _ = load_phase07_priors(hier_dir, chamber)
            if mu_dim1_prior is not None:
                print(
                    f"  Phase 07 Dim 1 prior: D={mu_dim1_prior[0]:+.3f}, R={mu_dim1_prior[1]:+.3f}"
                )
            else:
                print("  Phase 07 priors not available — using diffuse prior for Dim 1")

            # Phase 06 → Dim 2 party averages
            mu_dim2_prior = load_phase06_dim2_party_averages(irt_2d_dir, chamber)
            if mu_dim2_prior is not None:
                print(
                    f"  Phase 06 Dim 2 prior: D={mu_dim2_prior[0]:+.3f}, R={mu_dim2_prior[1]:+.3f}"
                )
            else:
                print("  Phase 06 Dim 2 priors not available — using diffuse prior")

            # ── PCA initialization (same as Phase 06) ──
            xi_initvals_2d: np.ndarray | None = None
            if pca_scores is not None:
                dim1_init, _, init_source = resolve_init_source(
                    strategy=args.init_strategy,
                    slugs=data["leg_slugs"],
                    pca_scores=pca_scores,
                    pca_column="PC1",
                )
                dim2_init, _, _ = resolve_init_source(
                    strategy="pca-informed",
                    slugs=data["leg_slugs"],
                    pca_scores=pca_scores,
                    pca_column="PC2",
                )
                xi_initvals_2d = np.stack([dim1_init, dim2_init], axis=1).astype(np.float64)
                print(f"  Init: {init_source}")

            # ── Beta init from PCA (ADR-0112) ──
            beta_init = compute_beta_init_from_pca(matrix, data)
            if beta_init is not None:
                print(f"  Beta PCA init available ({data['n_votes']} votes)")

            # ── Load flat 2D for comparison ──
            flat_2d_path = irt_2d_dir / "data" / f"ideal_points_2d_{ch}.parquet"
            flat_2d = pl.read_parquet(flat_2d_path) if flat_2d_path.exists() else None
            if flat_2d is not None:
                print(f"  Flat 2D loaded: {flat_2d.height} legislators")

            # ── Build and sample ──
            print_header(f"SAMPLING — {chamber}")
            idata, sampling_time = build_and_sample_h2d(
                data,
                mu_dim1_prior=mu_dim1_prior,
                mu_dim2_prior=mu_dim2_prior,
                xi_initvals_2d=xi_initvals_2d,
                beta_init=beta_init,
                n_samples=args.n_samples,
                n_tune=n_tune,
                n_chains=args.n_chains,
            )

            # ── Convergence ──
            convergence = check_h2d_convergence(idata, chamber)
            convergence_summary["chambers"][chamber] = {"convergence": convergence}

            # ── Extract results ──
            print_header(f"EXTRACTING RESULTS — {chamber}")
            ideal_h2d = extract_h2d_ideal_points(idata, data, legislators)

            # Post-hoc sign check
            idata, ideal_h2d = apply_dim1_sign_check(idata, ideal_h2d)

            # Dimension swap check (same as Phase 06, R7)
            from analysis.irt_2d import check_and_fix_dimension_swap

            idata, ideal_h2d, dim_swapped = check_and_fix_dimension_swap(idata, ideal_h2d)
            if dim_swapped:
                convergence["dimension_swap_corrected"] = True
                idata, ideal_h2d = apply_dim1_sign_check(idata, ideal_h2d)

            group_params = extract_h2d_group_params(idata, data)

            # Print group params
            print("\n  Group-level parameters:")
            for row in group_params.iter_rows(named=True):
                print(
                    f"    {row['party']} {row['dimension']}: mu={row['mu_mean']:+.3f} "
                    f"[{row['mu_hdi_2.5']:+.3f}, {row['mu_hdi_97.5']:+.3f}], "
                    f"sigma={row['sigma_within_mean']:.3f}"
                )

            # Correlation with flat 2D
            if flat_2d is not None:
                merged = ideal_h2d.join(
                    flat_2d.select("legislator_slug", pl.col("xi_dim1_mean").alias("flat_dim1")),
                    on="legislator_slug",
                    how="inner",
                )
                if merged.height > 2:
                    r = float(
                        stats.pearsonr(
                            merged["xi_dim1_mean"].to_numpy(),
                            merged["flat_dim1"].to_numpy(),
                        ).statistic
                    )
                    print(f"\n  Dim 1 correlation with flat 2D: r = {r:.4f}")

            # ── Save ──
            print_header(f"SAVING — {chamber}")
            ideal_h2d.write_parquet(ctx.data_dir / f"ideal_points_h2d_{ch}.parquet")
            ctx.export_csv(
                ideal_h2d,
                f"ideal_points_h2d_{ch}.csv",
                f"Hierarchical 2D ideal points for {chamber}",
            )
            group_params.write_parquet(ctx.data_dir / f"group_params_{ch}.parquet")
            ctx.export_csv(
                group_params,
                f"group_params_{ch}.csv",
                f"H2D group parameters for {chamber}",
            )
            idata.to_netcdf(str(ctx.data_dir / f"idata_{ch}.nc"))
            print(f"  Saved: ideal_points_h2d_{ch}.parquet")
            print(f"  Saved: group_params_{ch}.parquet")
            print(f"  Saved: idata_{ch}.nc")

            # ── Plots ──
            print_header(f"PLOTS — {chamber}")
            h2d_init_label = f"Dim 1: {init_source}" if pca_scores is not None else None
            plot_2d_scatter(ideal_h2d, chamber, ctx.plots_dir, init_label=h2d_init_label)
            plot_party_posteriors(idata, data, chamber, ctx.plots_dir)
            plot_shrinkage_vs_flat2d(ideal_h2d, flat_2d, chamber, ctx.plots_dir)

            all_chamber_results[chamber] = {
                "ideal_points": ideal_h2d,
                "group_params": group_params,
                "convergence": convergence,
                "sampling_time": sampling_time,
                "flat_2d": flat_2d,
            }

        if not all_chamber_results:
            print("\n  No chambers processed — exiting")
            return

        # ── Save convergence summary ──
        with open(ctx.data_dir / "convergence_summary.json", "w") as f:
            json.dump(convergence_summary, f, indent=2, default=str)
        print("\n  Saved: convergence_summary.json")

        # ── Canonical ideal point re-routing ──
        print_header("CANONICAL IDEAL POINT ROUTING")
        try:
            from analysis.canonical_ideal_points import write_canonical_ideal_points
        except ModuleNotFoundError:
            from canonical_ideal_points import (
                write_canonical_ideal_points,  # type: ignore[no-redef]
            )

        irt_dir = resolve_upstream_dir(
            "05_irt",
            results_root,
            args.run_id,
            Path(args.irt_dir) if args.irt_dir else None,
        )

        # Resolve W-NOMINATE dir for cross-validation gate (ADR-0123)
        wnom_dir: Path | None = None
        wnom_candidate = results_root / "latest" / "16_wnominate"
        if (wnom_candidate / "data").exists():
            wnom_dir = wnom_candidate

        canonical_dir = ctx.run_dir / "canonical_irt"
        canonical_sources = write_canonical_ideal_points(
            irt_1d_dir=irt_dir,
            irt_2d_dir=irt_2d_dir,
            output_dir=canonical_dir,
            pca_dir=pca_dir,
            h2d_dir=ctx.run_dir,
            wnom_dir=wnom_dir,
        )
        print(f"  Canonical sources: {canonical_sources}")

        # ── HTML Report ──
        print_header("BUILDING REPORT")
        build_hierarchical_2d_report(
            ctx.report,
            chamber_results=all_chamber_results,
            plots_dir=ctx.plots_dir,
            session=args.session,
        )
        print(f"\n  Report: {ctx.run_dir / '07b_hierarchical_2d_report.html'}")


if __name__ == "__main__":
    main()
