"""
Kansas Legislature — 2D Bayesian IRT Ideal Point Estimation (Phase 4b, EXPERIMENTAL)

EXPERIMENTAL: This phase uses a multidimensional 2-Parameter Logistic (M2PL) IRT model
with Positive Lower Triangular (PLT) identification to estimate 2D ideal points. It
resolves the Tyson paradox by separating ideology (Dim 1) from the
establishment–contrarian axis (Dim 2).

Convergence caveats: Dim 2 has known convergence challenges (R-hat up to 1.05, ESS ~200)
due to weak signal (~11% variance). Relaxed thresholds are used. Dim 2 credible intervals
may be unreliable for most legislators.

See analysis/design/irt_2d.md for full design and docs/2d-irt-deep-dive.md for motivation.

Usage:
  uv run python analysis/06_irt_2d/irt_2d.py [--session 2025-26] [--run-id ...]
      [--eda-dir ...] [--pca-dir ...] [--n-samples 2000] [--n-tune 2000] [--n-chains 4]

Outputs (in results/<session>/06_irt_2d/<date>/):
  - data/:   Parquet files (ideal_points_2d_{chamber}.parquet) + convergence_summary.json
  - plots/:  PNG visualizations (2d_scatter, dim1_vs_pc1, dim2_vs_pc2 per chamber)
  - irt_2d_report.html
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
    from run_context import RunContext, resolve_upstream_dir  # type: ignore[no-redef]

try:
    from analysis.phase_utils import load_metadata, print_header, save_fig
except ModuleNotFoundError:
    from phase_utils import load_metadata, print_header, save_fig

try:
    from analysis.irt import (
        load_eda_matrices,
        load_pca_scores,
        prepare_irt_data,
        select_anchors,
    )
except ModuleNotFoundError:
    from irt import (  # type: ignore[no-redef]
        load_eda_matrices,
        load_pca_scores,
        prepare_irt_data,
        select_anchors,
    )

try:
    from analysis.irt_2d_report import build_irt_2d_report
except ModuleNotFoundError:
    from irt_2d_report import build_irt_2d_report  # type: ignore[no-redef]

try:
    from analysis.init_strategy import InitStrategy, load_irt_scores, resolve_init_source
except ModuleNotFoundError:
    from init_strategy import (  # type: ignore[no-redef]
        InitStrategy,
        load_irt_scores,
        resolve_init_source,
    )

try:
    from analysis.tuning import CONTESTED_THRESHOLD, PARTY_COLORS, SUPERMAJORITY_THRESHOLD
except ModuleNotFoundError:
    from tuning import (  # type: ignore[no-redef]
        CONTESTED_THRESHOLD,
        PARTY_COLORS,
        SUPERMAJORITY_THRESHOLD,
    )

# ── Primer ───────────────────────────────────────────────────────────────────

IRT_2D_PRIMER = """\
# 2D Bayesian IRT Ideal Point Estimation (EXPERIMENTAL)

## Purpose

The 2D IRT model extends the canonical 1D baseline by estimating two-dimensional
ideal points for each legislator. Dimension 1 captures ideology (liberal-conservative),
while Dimension 2 captures the establishment–contrarian axis — the degree to which
legislators align with party leadership on routine bills.

This phase was developed to resolve the "Tyson paradox": Senator Caryn Tyson appears
as the most conservative legislator by 1D IRT, yet votes Nay on routine bills that
nearly all Republicans support. The 2D model reveals this as a real multidimensional
pattern (high on Dim 1 ideology, extreme on the Dim 2 establishment–contrarian
axis), not a model artifact.

## EXPERIMENTAL STATUS

This phase uses **relaxed convergence thresholds** (R-hat < 1.05, ESS > 200,
divergences < 50) compared to the production 1D model (R-hat < 1.01, ESS > 400,
divergences < 10). Dimension 2 captures ~11% of variance and may have wide credible
intervals for most legislators. Dim 2 HDIs should be interpreted with caution.

For horseshoe-affected chambers (supermajority, where 1D IRT conflates ideology
with establishment-loyalty), 2D Dim 1 is the canonical ideology score, following
the DW-NOMINATE standard. For balanced chambers, 1D IRT remains canonical.
See docs/canonical-ideal-points.md.

## Method

### Multidimensional 2-Parameter Logistic (M2PL) IRT

```
P(Yea | xi_i, alpha_j, beta_j) = logit^-1(sum_d(beta[j,d] * xi[i,d]) - alpha_j)

xi_i    ~ Normal(0, 1)     per dimension, per legislator   shape: (n_leg, 2)
alpha_j ~ Normal(0, 5)     per bill (difficulty)            shape: (n_votes,)
beta_j  ~ PLT-constrained  per bill, per dimension          shape: (n_votes, 2)
```

### PLT Identification

The discrimination matrix is constrained to be Positive Lower Triangular:
- beta[0, 1] = 0 (rotation anchor)
- beta[1, 1] > 0 (HalfNormal — positive diagonal)
- All other entries free Normal

Post-hoc Dim 1 sign check: Republican mean must be positive.

## Inputs

- EDA filtered vote matrices (House + Senate)
- PCA scores (for anchor selection and 2D initialization)
- Legislator metadata

## Outputs

- `ideal_points_2d_{chamber}.parquet` — 2D ideal points with HDIs per chamber
- `convergence_summary.json` — Per-chamber convergence diagnostics
- `2d_scatter_{chamber}.png` — Dim 1 vs Dim 2 party-colored scatter
- `dim1_vs_pc1_{chamber}.png` — Dim 1 correlation with PCA PC1
- `dim2_vs_pc2_{chamber}.png` — Dim 2 correlation with PCA PC2

## Interpretation Guide

- **Dim 1** (x-axis): Ideology. Positive = conservative, negative = liberal.
  Should correlate strongly (r > 0.90) with 1D IRT and PCA PC1.
- **Dim 2** (y-axis): Secondary pattern. In the Senate, this captures
  the establishment–contrarian axis. Interpretation varies by chamber and session.
- **Wide Dim 2 HDIs** are expected for most legislators — the second dimension
  has weak signal. Only legislators with narrow Dim 2 HDIs have reliable
  second-dimension estimates.

## Caveats

- Relaxed convergence thresholds — not production-grade.
- Dim 2 may not converge well in all chambers/sessions.
- Computational cost is 3-6x the 1D model.
- Not fed into downstream synthesis/profiles (premature given convergence caveats).
"""

# ── Constants ────────────────────────────────────────────────────────────────

N_SAMPLES = 2000
N_TUNE = 2000
N_TUNE_SUPERMAJORITY = 4000  # ADR-0112: doubled for >70% majority


N_CHAINS = 4
RANDOM_SEED = 42

# Relaxed convergence thresholds for experimental 2D model
RHAT_THRESHOLD = 1.05
ESS_THRESHOLD = 200
MAX_DIVERGENCES = 50

# Legislators of interest for the Tyson paradox
TYSON_SLUGS = {"sen_tyson_caryn"}
THOMPSON_SLUGS = {"sen_thompson_mike"}
ANNOTATE_SLUGS = TYSON_SLUGS | THOMPSON_SLUGS


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D Bayesian IRT — Multidimensional Ideal Points (EXPERIMENTAL)"
    )
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override 1D IRT results directory")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--init-strategy",
        default="pca-informed",
        choices=["auto", "irt-informed", "pca-informed"],
        help=(
            "Dim 1 initialization source (default: pca-informed). "
            "PCA is always safe for 2D init — avoids horseshoe contamination "
            "from confounded 1D IRT scores. See docs/canonical-ideal-points.md."
        ),
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n-tune", type=int, default=N_TUNE)
    parser.add_argument("--n-chains", type=int, default=N_CHAINS)
    parser.add_argument(
        "--contested-only",
        action="store_true",
        help="Filter to contested votes (minority > 2.5%%) before 2D IRT (ADR-0112)",
    )
    parser.add_argument("--csv", action="store_true", help="Force CSV-only mode (skip PostgreSQL)")
    return parser.parse_args()


# ── Model builder ────────────────────────────────────────────────────────────


def build_2d_irt_graph(data: dict) -> pm.Model:
    """Build a 2D M2PL IRT model graph with PLT identification.

    Returns the PyMC model without sampling (for nutpie compilation).

    Args:
        data: IRT data dict from prepare_irt_data().
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
        eta = pt.sum(beta[vote_idx] * xi[leg_idx], axis=1) - alpha[vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

    return model


def compute_beta_init_from_pca(
    matrix: pl.DataFrame,
    data: dict,
) -> tuple[np.ndarray, float] | None:
    """Compute beta (discrimination) init values from PCA loadings on the vote matrix.

    PCA eigenvectors on the bill dimension approximate the discrimination pattern:
    PC1 loadings → beta_col0 (ideology), PC2 loadings → beta_col1 (secondary axis).
    PLT constraints are enforced: beta_col1[0] = 0, beta_col1[1] > 0.

    Args:
        matrix: Vote matrix DataFrame (legislators x votes, with legislator_slug column).
        data: IRT data dict from prepare_irt_data() (provides vote_ids ordering).

    Returns:
        (beta_col0_init, beta_anchor_positive_init) or None if PCA fails.
        beta_col0_init: shape (n_votes,) — PC1 loadings for Dim 1 discrimination.
        beta_anchor_positive_init: float — abs(PC2 loading) for the positive anchor.
    """
    vote_ids = data["vote_ids"]
    vote_cols = [c for c in matrix.columns if c != "legislator_slug"]

    # Build numeric matrix (legislators x votes), mean-impute NaN
    mat = matrix.select(vote_cols).to_numpy().astype(np.float64)
    col_means = np.nanmean(mat, axis=0)
    nan_mask = np.isnan(mat)
    mat[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Standardize columns
    col_std = mat.std(axis=0)
    col_std[col_std == 0] = 1.0
    mat = (mat - mat.mean(axis=0)) / col_std

    # PCA via SVD on the vote dimension
    try:
        _, s, vt = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    # Loadings: rows of Vt correspond to principal components, columns to votes
    pc1_loadings = vt[0]  # shape (n_cols,)
    pc2_loadings = vt[1] if vt.shape[0] > 1 else np.zeros_like(pc1_loadings)

    # Map loadings to the model's vote ordering
    col_to_idx = {c: i for i, c in enumerate(vote_cols)}
    beta_col0_init = np.array(
        [pc1_loadings[col_to_idx[v]] if v in col_to_idx else 0.0 for v in vote_ids]
    )
    pc2_mapped = np.array(
        [pc2_loadings[col_to_idx[v]] if v in col_to_idx else 0.0 for v in vote_ids]
    )

    # PLT constraint: beta_col1[1] must be > 0
    # Use abs(pc2_loadings[1]) as the anchor positive init
    beta_anchor_positive_init = max(abs(pc2_mapped[1]), 0.01)

    return beta_col0_init, beta_anchor_positive_init


def build_and_sample_2d(
    data: dict,
    xi_initvals_2d: np.ndarray | None = None,
    beta_init: tuple[np.ndarray, float] | None = None,
    n_samples: int = N_SAMPLES,
    n_tune: int = N_TUNE,
    n_chains: int = N_CHAINS,
) -> tuple[az.InferenceData, float]:
    """Build 2D IRT graph, compile with nutpie, and sample.

    Args:
        data: IRT data dict from prepare_irt_data().
        xi_initvals_2d: Optional (n_leg, 2) array of initial ideal points from PCA.
        beta_init: Optional (beta_col0_init, beta_anchor_positive_init) from
            compute_beta_init_from_pca(). Provides PCA-informed starting values
            for discrimination parameters (ADR-0112).
        n_samples: MCMC posterior draws per chain.
        n_tune: MCMC tuning steps (discarded).
        n_chains: Number of independent MCMC chains.

    Returns (InferenceData, sampling_time_seconds).
    """
    n_leg = data["n_legislators"]
    n_votes = data["n_votes"]

    model = build_2d_irt_graph(data)

    # --- Compile with nutpie ---
    compile_kwargs: dict = {}
    initial_points: dict = {}
    no_jitter_rvs: set[str] = set()

    if xi_initvals_2d is not None:
        initial_points["xi"] = xi_initvals_2d
        no_jitter_rvs.add("xi")
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
        compile_kwargs["jitter_rvs"] = {rv for rv in model.free_RVs if rv.name not in no_jitter_rvs}
        jittered = [rv.name for rv in compile_kwargs["jitter_rvs"]]
        print(f"  jitter_rvs: {jittered} ({', '.join(no_jitter_rvs)} excluded)")

    print("  Compiling model with nutpie...")
    compiled = nutpie.compile_pymc_model(model, **compile_kwargs)

    # --- Sample ---
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


def check_2d_convergence(idata: az.InferenceData, chamber: str) -> dict:
    """Run convergence diagnostics for the 2D model."""
    print_header(f"CONVERGENCE DIAGNOSTICS — 2D IRT ({chamber})")

    diag: dict = {}

    # R-hat
    rhat = az.rhat(idata)
    xi_rhat_max = float(rhat["xi"].max())
    alpha_rhat_max = float(rhat["alpha"].max())
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
        print("  CONVERGENCE: ALL CHECKS PASSED (relaxed thresholds)")
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

    party_map = {}
    name_map = {}
    for row in legislators.iter_rows(named=True):
        party_map[row["legislator_slug"]] = row.get("party", "Unknown")
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


# ── Post-hoc sign check ─────────────────────────────────────────────────────


def apply_dim1_sign_check(ideal_2d: pl.DataFrame) -> pl.DataFrame:
    """Flip Dim 1 sign if Republican mean is negative."""
    r_dim1 = ideal_2d.filter(pl.col("party") == "Republican")["xi_dim1_mean"].mean()
    if r_dim1 is not None and r_dim1 < 0:
        print("  WARNING: Republican Dim 1 mean is negative — flipping Dim 1 sign")
        ideal_2d = (
            ideal_2d.with_columns(
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
    return ideal_2d


# ── Dimension swap detection ──────────────────────────────────────────────────


# PLT identification constraints don't guarantee which dimension captures ideology.
# If Dim 2 correlates more strongly with PCA PC1 (ideology) than Dim 1, the dimensions
# are swapped — common when within-party variance (Dim 1) exceeds the between-party axis.
def _party_d(df: pl.DataFrame, col: str) -> float:
    """Cohen's d between R and D on a given column."""
    r = df.filter(pl.col("party") == "Republican")[col]
    d_vals = df.filter(pl.col("party") == "Democrat")[col]
    if r.len() == 0 or d_vals.len() == 0:
        return 0.0
    pooled = np.sqrt((float(r.std()) ** 2 + float(d_vals.std()) ** 2) / 2)
    return abs(float(r.mean()) - float(d_vals.mean())) / pooled if pooled > 0 else 0.0


def check_and_fix_dimension_swap(
    idata: az.InferenceData,
    ideal_2d: pl.DataFrame,
) -> tuple[az.InferenceData, pl.DataFrame, bool]:
    """Check if Dim 2 separates parties better than Dim 1 and swap if so.

    In supermajority Senate sessions (78th-83rd, 88th), PCA PC1 captures
    intra-R factionalism, causing the 2D IRT to initialize with dimensions
    swapped. This function detects and corrects the swap post-hoc.

    See docs/pca-ideology-axis-instability.md (R7).
    """
    dim1_d = _party_d(ideal_2d, "xi_dim1_mean")
    dim2_d = _party_d(ideal_2d, "xi_dim2_mean")

    if dim2_d > dim1_d and dim2_d > 2.0:
        print(
            f"  DIMENSION SWAP DETECTED: Dim 2 (d={dim2_d:.2f}) separates "
            f"parties better than Dim 1 (d={dim1_d:.2f}) — swapping"
        )

        # Swap in posterior idata
        xi_post = idata.posterior["xi"].values  # (chain, draw, leg, 2)
        xi_swapped = xi_post[:, :, :, ::-1].copy()
        idata.posterior["xi"].values = xi_swapped

        # Swap in DataFrame
        ideal_2d = ideal_2d.rename(
            {
                "xi_dim1_mean": "_d2m",
                "xi_dim1_hdi_3%": "_d2lo",
                "xi_dim1_hdi_97%": "_d2hi",
                "xi_dim2_mean": "xi_dim1_mean",
                "xi_dim2_hdi_3%": "xi_dim1_hdi_3%",
                "xi_dim2_hdi_97%": "xi_dim1_hdi_97%",
            }
        ).rename(
            {
                "_d2m": "xi_dim2_mean",
                "_d2lo": "xi_dim2_hdi_3%",
                "_d2hi": "xi_dim2_hdi_97%",
            }
        )

        return idata, ideal_2d, True
    else:
        print(f"  Dimension order OK: Dim 1 (d={dim1_d:.2f}) ≥ Dim 2 (d={dim2_d:.2f})")
        return idata, ideal_2d, False


# ── Correlation checks ───────────────────────────────────────────────────────


def correlate_with_pca(
    ideal_2d: pl.DataFrame,
    pca_scores: pl.DataFrame,
    data: dict,
) -> dict:
    """Correlate 2D Dim 1 with PCA PC1 and 2D Dim 2 with PCA PC2."""
    slug_col = "legislator_slug"
    corrs: dict = {}

    dim1_map = dict(zip(ideal_2d[slug_col].to_list(), ideal_2d["xi_dim1_mean"].to_list()))
    dim2_map = dict(zip(ideal_2d[slug_col].to_list(), ideal_2d["xi_dim2_mean"].to_list()))

    pca_pc1_map = {row["legislator_slug"]: row["PC1"] for row in pca_scores.iter_rows(named=True)}
    pca_pc2_map = {row["legislator_slug"]: row["PC2"] for row in pca_scores.iter_rows(named=True)}

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

    print(f"  Dim 1 vs PCA PC1:  r = {r_dim1_pc1:.4f},  rho = {rho_dim1_pc1:.4f}")
    print(f"  Dim 2 vs PCA PC2:  r = {r_dim2_pc2:.4f},  rho = {rho_dim2_pc2:.4f}")

    return corrs


# ── Plots ────────────────────────────────────────────────────────────────────


def plot_2d_scatter(
    ideal_2d: pl.DataFrame,
    chamber: str,
    output_dir: Path,
    init_label: str | None = None,
) -> None:
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
    title = f"2D Bayesian IRT Ideal Points — Kansas {chamber} (EXPERIMENTAL)"
    if init_label:
        title += f"\nInit: {init_label}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(loc="upper left")

    fig.tight_layout()
    save_fig(fig, output_dir / f"2d_scatter_{chamber.lower()}.png")


def plot_dim1_vs_pc1(
    ideal_2d: pl.DataFrame, pca_scores: pl.DataFrame, chamber: str, output_dir: Path
) -> None:
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
    ax.set_title(
        f"2D IRT Dim 1 vs PCA PC1 — {chamber}  (r = {r:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    fig.tight_layout()
    save_fig(fig, output_dir / f"dim1_vs_pc1_{chamber.lower()}.png")


def plot_dim2_vs_pc2(
    ideal_2d: pl.DataFrame, pca_scores: pl.DataFrame, chamber: str, output_dir: Path
) -> None:
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
    ax.set_title(
        f"2D IRT Dim 2 vs PCA PC2 — {chamber}  (r = {r:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    fig.tight_layout()
    save_fig(fig, output_dir / f"dim2_vs_pc2_{chamber.lower()}.png")


# ── Dim 1 Forest Plot ───────────────────────────────────────────────────────


def plot_dim1_forest(
    ideal_2d: pl.DataFrame,
    chamber: str,
    output_dir: Path,
    init_label: str | None = None,
) -> None:
    """Forest plot: Dim 1 ideal points with 95% HDI bars, party-colored, sorted.

    This is the canonical ideology ranking from the 2D model — Dim 1 separated
    from the establishment–contrarian axis. For horseshoe-affected chambers,
    this is the correct ordering (unlike 1D IRT which conflates dimensions).
    """
    from matplotlib.patches import Patch

    # Map 2D column names to the standard forest plot expectations
    forest_df = ideal_2d.select(
        "legislator_slug",
        "full_name",
        "party",
        pl.col("xi_dim1_mean").alias("xi_mean"),
        pl.col("xi_dim1_hdi_3%").alias("xi_hdi_low"),
        pl.col("xi_dim1_hdi_97%").alias("xi_hdi_high"),
    ).sort("xi_mean")

    n = forest_df.height
    fig, ax = plt.subplots(figsize=(10, max(14, n * 0.22)))

    y_pos = np.arange(n)
    for i, row in enumerate(forest_df.iter_rows(named=True)):
        color = PARTY_COLORS.get(row["party"], "#888888")
        slug = row["legislator_slug"]
        is_annotated = slug in ANNOTATE_SLUGS

        ax.hlines(
            i,
            row["xi_hdi_low"],
            row["xi_hdi_high"],
            colors=color,
            alpha=0.7 if is_annotated else 0.4,
            linewidth=2.5 if is_annotated else 1.5,
        )
        ax.scatter(
            row["xi_mean"],
            i,
            c=color,
            s=40 if is_annotated else 20,
            zorder=5,
            edgecolors="black",
            linewidth=0.3,
            marker="D" if is_annotated else "o",
        )

        if is_annotated:
            ax.annotate(
                row["full_name"],
                (row["xi_hdi_high"], i),
                fontsize=7,
                fontweight="bold",
                color="#333333",
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                bbox={"boxstyle": "round,pad=0.2", "fc": "lightyellow", "alpha": 0.7},
            )

    labels = []
    for row in forest_df.iter_rows(named=True):
        name = row["full_name"]
        party_initial = row["party"][0] if row["party"] else "?"
        labels.append(f"{name} ({party_initial})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Dim 1 Ideal Point (Liberal \u2190 \u2192 Conservative)")
    title = f"{chamber} \u2014 Dim 1 Ideology Ranking (from 2D IRT)"
    if init_label:
        title += f"\nInit: {init_label}"
    ax.set_title(title, fontsize=12)
    legend_handles = [
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
    ]
    if ideal_2d.filter(pl.col("party") == "Independent").height > 0:
        legend_handles.append(Patch(facecolor=PARTY_COLORS["Independent"], label="Independent"))
    ax.legend(handles=legend_handles, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, output_dir / f"dim1_forest_{chamber.lower()}.png")


# ── Interactive Plotly Plots ─────────────────────────────────────────────────


def plot_2d_scatter_interactive(ideal_2d: pl.DataFrame, chamber: str, output_dir: Path) -> None:
    """Plotly interactive 2D scatter: Dim 1 vs Dim 2 with hover details."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for party, color in PARTY_COLORS.items():
        subset = ideal_2d.filter(pl.col("party") == party)
        if subset.is_empty():
            continue

        hover_text = [
            f"<b>{row['full_name']}</b><br>"
            f"Party: {row['party']}<br>"
            f"Dim 1: {row['xi_dim1_mean']:+.3f} "
            f"[{row['xi_dim1_hdi_3%']:+.3f}, {row['xi_dim1_hdi_97%']:+.3f}]<br>"
            f"Dim 2: {row['xi_dim2_mean']:+.3f} "
            f"[{row['xi_dim2_hdi_3%']:+.3f}, {row['xi_dim2_hdi_97%']:+.3f}]"
            for row in subset.iter_rows(named=True)
        ]

        fig.add_trace(
            go.Scatter(
                x=subset["xi_dim1_mean"].to_list(),
                y=subset["xi_dim2_mean"].to_list(),
                mode="markers",
                name=party,
                marker={
                    "color": color,
                    "size": 9,
                    "opacity": 0.7,
                    "line": {"width": 0.5, "color": "white"},
                },
                text=hover_text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"2D Bayesian IRT Ideal Points — Kansas {chamber} (EXPERIMENTAL)",
        xaxis_title="Dimension 1 (Ideology: Liberal ← → Conservative)",
        yaxis_title="Dimension 2 (Contrarian ← → Establishment)",
        hovermode="closest",
        template="plotly_white",
        width=800,
        height=640,
        shapes=[
            {
                "type": "line",
                "x0": 0,
                "x1": 0,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "gray", "width": 0.5, "dash": "dash"},
            },
            {
                "type": "line",
                "y0": 0,
                "y1": 0,
                "x0": 0,
                "x1": 1,
                "xref": "paper",
                "line": {"color": "gray", "width": 0.5, "dash": "dash"},
            },
        ],
    )
    html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    (output_dir / f"2d_scatter_interactive_{chamber.lower()}.html").write_text(html)
    print(f"  Saved: 2d_scatter_interactive_{chamber.lower()}.html")


def plot_dim1_vs_pc1_interactive(
    ideal_2d: pl.DataFrame, pca_scores: pl.DataFrame, chamber: str, output_dir: Path
) -> None:
    """Plotly interactive Dim 1 vs PCA PC1 with hover details."""
    import plotly.graph_objects as go

    pca_map = {row["legislator_slug"]: row["PC1"] for row in pca_scores.iter_rows(named=True)}
    shared = ideal_2d.filter(pl.col("legislator_slug").is_in(list(pca_map.keys())))

    dim1 = shared["xi_dim1_mean"].to_list()
    pc1 = [pca_map[s] for s in shared["legislator_slug"].to_list()]
    r = float(np.corrcoef(dim1, pc1)[0, 1])

    fig = go.Figure()
    for party, color in PARTY_COLORS.items():
        subset = shared.filter(pl.col("party") == party)
        if subset.is_empty():
            continue
        s_pc1 = [pca_map[s] for s in subset["legislator_slug"].to_list()]
        hover_text = [
            f"<b>{row['full_name']}</b><br>"
            f"Party: {row['party']}<br>"
            f"Dim 1: {row['xi_dim1_mean']:+.3f}<br>"
            f"PCA PC1: {pca_map[row['legislator_slug']]:.3f}"
            for row in subset.iter_rows(named=True)
        ]
        fig.add_trace(
            go.Scatter(
                x=s_pc1,
                y=subset["xi_dim1_mean"].to_list(),
                mode="markers",
                name=party,
                marker={"color": color, "size": 8, "opacity": 0.7},
                text=hover_text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"2D IRT Dim 1 vs PCA PC1 — {chamber} (r = {r:.4f})",
        xaxis_title="PCA PC1",
        yaxis_title="2D IRT Dimension 1",
        hovermode="closest",
        template="plotly_white",
        width=640,
        height=640,
    )
    html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    (output_dir / f"dim1_vs_pc1_interactive_{chamber.lower()}.html").write_text(html)
    print(f"  Saved: dim1_vs_pc1_interactive_{chamber.lower()}.html")


def plot_dim2_vs_pc2_interactive(
    ideal_2d: pl.DataFrame, pca_scores: pl.DataFrame, chamber: str, output_dir: Path
) -> None:
    """Plotly interactive Dim 2 vs PCA PC2 with hover details."""
    import plotly.graph_objects as go

    pca_map = {row["legislator_slug"]: row["PC2"] for row in pca_scores.iter_rows(named=True)}
    shared = ideal_2d.filter(pl.col("legislator_slug").is_in(list(pca_map.keys())))

    dim2 = shared["xi_dim2_mean"].to_list()
    pc2 = [pca_map[s] for s in shared["legislator_slug"].to_list()]
    r = float(np.corrcoef(dim2, pc2)[0, 1])

    fig = go.Figure()
    for party, color in PARTY_COLORS.items():
        subset = shared.filter(pl.col("party") == party)
        if subset.is_empty():
            continue
        s_pc2 = [pca_map[s] for s in subset["legislator_slug"].to_list()]
        hover_text = [
            f"<b>{row['full_name']}</b><br>"
            f"Party: {row['party']}<br>"
            f"Dim 2: {row['xi_dim2_mean']:+.3f}<br>"
            f"PCA PC2: {pca_map[row['legislator_slug']]:.3f}"
            for row in subset.iter_rows(named=True)
        ]
        fig.add_trace(
            go.Scatter(
                x=s_pc2,
                y=subset["xi_dim2_mean"].to_list(),
                mode="markers",
                name=party,
                marker={"color": color, "size": 8, "opacity": 0.7},
                text=hover_text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"2D IRT Dim 2 vs PCA PC2 — {chamber} (r = {r:.4f})",
        xaxis_title="PCA PC2",
        yaxis_title="2D IRT Dimension 2",
        hovermode="closest",
        template="plotly_white",
        width=640,
        height=640,
    )
    html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    (output_dir / f"dim2_vs_pc2_interactive_{chamber.lower()}.html").write_text(html)
    print(f"  Saved: dim2_vs_pc2_interactive_{chamber.lower()}.html")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from tallgrass.session import KSSession

    ks = KSSession.from_session_string(args.session)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = ks.data_dir

    results_root = ks.results_dir

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
    irt_dir = resolve_upstream_dir(
        "05_irt",
        results_root,
        args.run_id,
        Path(args.irt_dir) if args.irt_dir else None,
    )

    with RunContext(
        session=args.session,
        analysis_name="06_irt_2d",
        params=vars(args),
        primer=IRT_2D_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print("=" * 80)
        print("  2D BAYESIAN IRT — MULTIDIMENSIONAL IDEAL POINTS (EXPERIMENTAL)")
        print("=" * 80)
        print(f"  Session: {args.session}")
        print(f"  Data:    {data_dir}")
        print(f"  EDA:     {eda_dir}")
        print(f"  PCA:     {pca_dir}")
        print(f"  IRT:     {irt_dir}")
        print(f"  Output:  {ctx.run_dir}")
        print(f"  Samples: {args.n_samples} draws, {args.n_tune} tune, {args.n_chains} chains")
        print()
        print("  *** EXPERIMENTAL: Relaxed convergence thresholds ***")
        print(
            f"  *** R-hat < {RHAT_THRESHOLD}, ESS > {ESS_THRESHOLD}, "
            f"divergences < {MAX_DIVERGENCES} ***"
        )
        print()

        # ── Load data ──
        print_header("LOADING DATA")
        house_matrix, senate_matrix, _ = load_eda_matrices(eda_dir)
        pca_house, pca_senate = load_pca_scores(pca_dir)
        rollcalls, legislators = load_metadata(data_dir)

        # Load 1D IRT ideal points (for init strategy)
        irt_data_dir = irt_dir / "data"
        irt_house = load_irt_scores(irt_data_dir, "house")
        irt_senate = load_irt_scores(irt_data_dir, "senate")
        for ch, irt_df in [("House", irt_house), ("Senate", irt_senate)]:
            if irt_df is not None:
                print(f"  1D IRT loaded: {ch} ({irt_df.height} legislators)")
            else:
                print(f"  1D IRT not found: {ch}")

        print(f"  House filtered: {house_matrix.height} x {len(house_matrix.columns) - 1}")
        print(f"  Senate filtered: {senate_matrix.height} x {len(senate_matrix.columns) - 1}")
        print(f"  Init strategy: {args.init_strategy}")

        chamber_configs = [
            ("House", house_matrix, pca_house, irt_house),
            ("Senate", senate_matrix, pca_senate, irt_senate),
        ]

        all_results: dict[str, dict] = {}
        t_total = time.time()

        for chamber, matrix, pca_scores, irt_1d in chamber_configs:
            if matrix.height < 5:
                print(f"\n  Skipping {chamber}: too few legislators ({matrix.height})")
                continue

            chamber_lower = chamber.lower()

            # ── Contested-only filtering (ADR-0112) ──
            if args.contested_only:
                n_votes_before = len(matrix.columns) - 1  # exclude legislator_slug
                vote_cols = [c for c in matrix.columns if c != "legislator_slug"]
                minority_fracs = []
                for vc in vote_cols:
                    col = matrix[vc].drop_nulls()
                    if col.len() > 0:
                        frac = float(col.mean())
                        minority_fracs.append((vc, min(frac, 1 - frac)))
                contested = [vc for vc, mf in minority_fracs if mf > CONTESTED_THRESHOLD]
                matrix = matrix.select(["legislator_slug"] + contested)
                print(f"  Contested-only filter: {n_votes_before} → {len(contested)} votes")

            # ── Adaptive N_TUNE for supermajority (ADR-0112) ──
            chamber_legs = legislators.filter(
                pl.col("legislator_slug").is_in(matrix["legislator_slug"])
            )
            party_counts = chamber_legs.group_by("party").agg(pl.len().alias("count"))
            max_party_frac = float(party_counts["count"].max()) / max(chamber_legs.height, 1)
            effective_n_tune = args.n_tune
            if max_party_frac > SUPERMAJORITY_THRESHOLD and args.n_tune == N_TUNE:
                effective_n_tune = N_TUNE_SUPERMAJORITY
                print(
                    f"  Supermajority detected ({max_party_frac:.0%}) — "
                    f"N_TUNE increased: {N_TUNE} → {N_TUNE_SUPERMAJORITY}"
                )

            # ── Prepare IRT data ──
            print_header(f"PREPARE IRT DATA — {chamber}")
            data = prepare_irt_data(matrix, chamber)

            # ── Select 1D anchors (for reference logging) ──
            print(f"\n  Selecting 1D anchors for {chamber} (reference only):")
            select_anchors(pca_scores, matrix, chamber)

            # ── Build initialization for 2D model ──
            print_header(f"2D INITIALIZATION — {chamber}")
            slugs = data["leg_slugs"]

            # Dim 1: configurable via --init-strategy
            dim1_std, dim1_strategy, dim1_source = resolve_init_source(
                strategy=args.init_strategy,
                slugs=slugs,
                irt_scores=irt_1d,
                pca_scores=pca_scores,
                pca_column="PC1",
            )
            print(f"  Dim 1 init: {dim1_source} (strategy: {dim1_strategy})")

            # Dim 2: always PCA PC2 (no 1D equivalent exists)
            dim2_std, _, dim2_source = resolve_init_source(
                strategy=InitStrategy.PCA_INFORMED,
                slugs=slugs,
                pca_scores=pca_scores,
                pca_column="PC2",
            )
            print(f"  Dim 2 init: {dim2_source}")

            xi_initvals_2d = np.column_stack([dim1_std, dim2_std])
            print(f"  Initvals shape: {xi_initvals_2d.shape}")
            print(f"  Dim 1 range: [{dim1_std.min():.2f}, {dim1_std.max():.2f}]")
            print(f"  Dim 2 range: [{dim2_std.min():.2f}, {dim2_std.max():.2f}]")

            # ── Beta init from PCA loadings (ADR-0112) ──
            beta_init = compute_beta_init_from_pca(matrix, data)
            if beta_init is not None:
                lo, hi = beta_init[0].min(), beta_init[0].max()
                print(f"  Beta init: PCA loadings (col0 range: [{lo:.3f}, {hi:.3f}])")
            else:
                print("  Beta init: default (PCA loadings computation failed)")

            # ── Build and sample 2D model ──
            print_header(f"MCMC SAMPLING — 2D IRT ({chamber})")
            idata, sampling_time = build_and_sample_2d(
                data=data,
                xi_initvals_2d=xi_initvals_2d,
                beta_init=beta_init,
                n_samples=args.n_samples,
                n_tune=effective_n_tune,
                n_chains=args.n_chains,
            )

            # ── Convergence diagnostics ──
            diag = check_2d_convergence(idata, chamber)

            # ── Extract 2D ideal points ──
            print_header(f"EXTRACT 2D IDEAL POINTS — {chamber}")
            ideal_2d = extract_2d_ideal_points(idata, data, legislators)
            ideal_2d = apply_dim1_sign_check(ideal_2d)

            # ── Dimension swap check ──
            idata, ideal_2d, dim_swapped = check_and_fix_dimension_swap(idata, ideal_2d)
            if dim_swapped:
                diag["dimension_swap_corrected"] = True
                # Re-run sign check after swap
                ideal_2d = apply_dim1_sign_check(ideal_2d)

            # ── Correlation checks ──
            print_header(f"CORRELATION CHECKS — {chamber}")
            corrs = correlate_with_pca(ideal_2d, pca_scores, data)

            # ── Save per-chamber outputs ──
            print_header(f"SAVING OUTPUTS — {chamber}")

            ideal_2d.write_parquet(ctx.data_dir / f"ideal_points_2d_{chamber_lower}.parquet")
            print(
                f"  Saved: ideal_points_2d_{chamber_lower}.parquet ({ideal_2d.height} legislators)"
            )

            nc_path = ctx.data_dir / f"idata_{chamber_lower}.nc"
            idata.to_netcdf(str(nc_path))
            print(f"  Saved: idata_{chamber_lower}.nc")

            # ── Generate plots ──
            init_label = f"Dim 1: {dim1_source}, Dim 2: {dim2_source}"
            plot_2d_scatter(ideal_2d, chamber, ctx.plots_dir, init_label=init_label)
            plot_dim1_forest(ideal_2d, chamber, ctx.plots_dir, init_label=dim1_source)
            plot_dim1_vs_pc1(ideal_2d, pca_scores, chamber, ctx.plots_dir)
            plot_dim2_vs_pc2(ideal_2d, pca_scores, chamber, ctx.plots_dir)

            # ── Generate interactive plots ──
            plot_2d_scatter_interactive(ideal_2d, chamber, ctx.plots_dir)
            plot_dim1_vs_pc1_interactive(ideal_2d, pca_scores, chamber, ctx.plots_dir)
            plot_dim2_vs_pc2_interactive(ideal_2d, pca_scores, chamber, ctx.plots_dir)

            # ── Print full ideal points table ──
            print_header(f"2D IDEAL POINTS — {chamber}")
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

            all_results[chamber] = {
                "ideal_points": ideal_2d,
                "diagnostics": diag,
                "correlations": corrs,
                "data": data,
                "sampling_time": sampling_time,
            }

        # ── Save convergence summary (all chambers) ──
        summary: dict = {
            "session": args.session,
            "n_samples": args.n_samples,
            "n_tune": args.n_tune,
            "n_chains": args.n_chains,
            "sampler": "nutpie (Rust NUTS)",
            "thresholds": {
                "rhat": RHAT_THRESHOLD,
                "ess": ESS_THRESHOLD,
                "max_divergences": MAX_DIVERGENCES,
            },
            "chambers": {},
        }
        for chamber, result in all_results.items():
            summary["chambers"][chamber] = {
                "sampling_time_s": result["sampling_time"],
                "convergence": result["diagnostics"],
                "correlations": result["correlations"],
            }

        with open(ctx.data_dir / "convergence_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print("\n  Saved: convergence_summary.json")

        # ── Canonical ideal point routing ──
        print_header("CANONICAL IDEAL POINT ROUTING")
        try:
            from analysis.canonical_ideal_points import write_canonical_ideal_points
        except ModuleNotFoundError:
            from canonical_ideal_points import (
                write_canonical_ideal_points,  # type: ignore[no-redef]
            )

        # Resolve W-NOMINATE dir for cross-validation gate (ADR-0123)
        wnom_dir: Path | None = None
        wnom_candidate = results_root / "latest" / "16_wnominate"
        if (wnom_candidate / "data").exists():
            wnom_dir = wnom_candidate

        canonical_dir = ctx.run_dir / "canonical_irt"
        canonical_sources = write_canonical_ideal_points(
            irt_1d_dir=irt_dir,
            irt_2d_dir=ctx.run_dir,
            output_dir=canonical_dir,
            pca_dir=pca_dir,
            wnom_dir=wnom_dir,
        )

        # ── Build HTML report ──
        print_header("BUILDING HTML REPORT")
        build_irt_2d_report(
            ctx.report,
            results=all_results,
            plots_dir=ctx.plots_dir,
            n_samples=args.n_samples,
            n_tune=args.n_tune,
            n_chains=args.n_chains,
            canonical_sources=canonical_sources,
        )

        # ── Final summary ──
        total_time = time.time() - t_total
        print_header("EXPERIMENT SUMMARY")
        print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
        for chamber, result in all_results.items():
            diag = result["diagnostics"]
            corrs = result["correlations"]
            print(f"\n  {chamber}:")
            print(f"    Sampling time: {result['sampling_time']:.0f}s")
            print(f"    Convergence: {'PASSED' if diag['all_ok'] else 'FAILED'}")
            print(f"    Dim 1 vs PCA PC1: r = {corrs['dim1_vs_pc1_pearson']:.4f}")
            print(f"    Dim 2 vs PCA PC2: r = {corrs['dim2_vs_pc2_pearson']:.4f}")
        print(f"\n  Output: {ctx.run_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()
