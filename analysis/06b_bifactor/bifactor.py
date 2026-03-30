"""
Kansas Legislature — Bifactor IRT Ideal Point Estimation (Phase 06b, EXPERIMENTAL)

EXPERIMENTAL: This phase fits a bifactor IRT model that separates a general ideology
factor (loading on all bills) from specific factors (loading on bill subsets classified
by 1D IRT discrimination). The general factor captures "pure ideology" — what is common
to all voting behavior — while specific factors capture domain-specific patterns
(partisan-only bills, bipartisan-only bills).

Bills are classified using Phase 05's discrimination parameters:
  - High discrimination (|beta| > 1.5): partisan bills → specific factor 1
  - Low discrimination (|beta| < 0.5): bipartisan bills → specific factor 2
  - Medium discrimination: general factor only (no specific loading)

The general factor should produce a cleaner ideology score than the current 2D M2PL's
Dim 1 extraction, because the specific factors are orthogonal by construction rather
than by rotation convention.

See docs/cfa-irt-dimensionality-deep-dive.md for theoretical motivation.
See analysis/design/bifactor.md for full design.

Usage:
  uv run python analysis/06b_bifactor/bifactor.py [--session 2025-26] [--run-id ...]
      [--eda-dir ...] [--pca-dir ...] [--irt-dir ...] [--n-samples 2000]

Outputs (in results/<session>/06b_bifactor/<date>/):
  - data/:   Parquet files (ideal_points_bifactor_{chamber}.parquet) + convergence_summary.json
  - plots/:  PNG visualizations per chamber
  - 06b_bifactor_report.html
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
    )
except ModuleNotFoundError:
    from irt import (  # type: ignore[no-redef]
        load_eda_matrices,
        load_pca_scores,
        prepare_irt_data,
    )

try:
    from analysis.init_strategy import InitStrategy, load_irt_scores, resolve_init_source
except ModuleNotFoundError:
    from init_strategy import (  # type: ignore[no-redef]
        InitStrategy,
        load_irt_scores,
        resolve_init_source,
    )

try:
    from analysis.tuning import (
        CONTESTED_THRESHOLD,
        HIGH_DISC_THRESHOLD,
        LOW_DISC_THRESHOLD,
        PARTY_COLORS,
        SUPERMAJORITY_THRESHOLD,
    )
except ModuleNotFoundError:
    from tuning import (  # type: ignore[no-redef]
        CONTESTED_THRESHOLD,
        HIGH_DISC_THRESHOLD,
        LOW_DISC_THRESHOLD,
        PARTY_COLORS,
        SUPERMAJORITY_THRESHOLD,
    )

try:
    from analysis.bifactor_report import build_bifactor_report
except ModuleNotFoundError:
    from bifactor_report import build_bifactor_report  # type: ignore[no-redef]


# ── Primer ───────────────────────────────────────────────────────────────────

BIFACTOR_PRIMER = """\
# Bifactor IRT Ideal Point Estimation (EXPERIMENTAL)

## Purpose

The bifactor IRT model separates a **general ideology factor** (loading on all bills)
from **specific factors** (loading on bill subsets). The general factor captures "pure
ideology" — what is common to all voting behavior — while specific factors capture
domain-specific patterns like partisan-only contrarianism or bipartisan procedural
behavior.

This addresses the Tyson paradox: by partitioning contrarian/establishment variance
into a specific factor, the general factor should produce a Tyson score that is
conservative but not extreme — unlike the 1D IRT which places her at rank #1.

## Method

### Bifactor 2-Parameter Logistic IRT

```
P(Yea | theta_G, theta_S1, theta_S2) = logit^-1(
    a_G[j] * theta_G[i]
    + a_S1[j] * theta_S1[i] * mask_high[j]
    + a_S2[j] * theta_S2[i] * mask_low[j]
    - d[j]
)

theta_G   ~ Normal(0, 1)    general ideology (all legislators)
theta_S1  ~ Normal(0, 1)    partisan-specific (all legislators)
theta_S2  ~ Normal(0, 1)    bipartisan-specific (all legislators)
a_G       ~ Normal(0, 1)    general discrimination (all bills)
a_S1_raw  ~ Normal(0, 1)    specific discrimination (masked to high-disc bills)
a_S2_raw  ~ Normal(0, 1)    specific discrimination (masked to low-disc bills)
d         ~ Normal(0, 5)    difficulty (all bills)
```

### Bill Classification

Bills are grouped using Phase 05 IRT discrimination magnitudes:
- High discrimination (|beta| > 1.5): partisan votes → specific factor 1
- Low discrimination (|beta| < 0.5): bipartisan votes → specific factor 2
- Medium discrimination: general factor only (pure ideology anchors)

### Identification

- Orthogonality by construction (independent priors on all theta factors)
- Post-hoc sign flip: Republican mean on theta_G must be positive
- No PLT constraints needed (unlike the 2D M2PL in Phase 06)

## Outputs

- `ideal_points_bifactor_{chamber}.parquet` — theta_G, theta_S1, theta_S2 with HDIs
- `bill_params_bifactor_{chamber}.parquet` — a_G, a_S1, a_S2, d per bill
- `bill_classification_{chamber}.parquet` — bill group assignments
- `convergence_summary.json` — diagnostics including ECV and omega_h
"""

# ── Constants ────────────────────────────────────────────────────────────────

N_SAMPLES = 2000
N_TUNE = 2000
N_TUNE_SUPERMAJORITY = 4000

N_CHAINS = 4
RANDOM_SEED = 42

# Relaxed convergence thresholds (same as Phase 06)
RHAT_THRESHOLD = 1.05
ESS_THRESHOLD = 200
MAX_DIVERGENCES = 50


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bifactor IRT — General + Specific Factor Ideal Points (EXPERIMENTAL)"
    )
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override 1D IRT results directory")
    parser.add_argument("--irt-2d-dir", default=None, help="Override 2D IRT results for comparison")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--init-strategy",
        default="pca-informed",
        choices=["auto", "irt-informed", "pca-informed"],
        help="Initialization strategy for general factor (default: pca-informed)",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n-tune", type=int, default=N_TUNE)
    parser.add_argument("--n-chains", type=int, default=N_CHAINS)
    parser.add_argument(
        "--contested-only",
        action="store_true",
        help="Use only contested votes (minority fraction > contested threshold)",
    )
    parser.add_argument("--csv", action="store_true", help="Force CSV-only data loading")
    return parser.parse_args()


# ── Bill classification ──────────────────────────────────────────────────────


def classify_bills_by_discrimination(
    bill_params: pl.DataFrame,
    vote_ids: list[str],
    high_threshold: float = HIGH_DISC_THRESHOLD,
    low_threshold: float = LOW_DISC_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Classify bills into high-disc, low-disc, and general-only groups.

    Args:
        bill_params: Phase 05 bill_params_{chamber}.parquet with beta_mean column.
        vote_ids: Ordered vote IDs matching the IRT data dict.
        high_threshold: |beta| above this -> specific factor 1 (partisan).
        low_threshold: |beta| below this -> specific factor 2 (bipartisan).

    Returns:
        (mask_high, mask_low, classification_df)
    """
    beta_map = {
        row["vote_id"]: abs(row["beta_mean"])
        for row in bill_params.iter_rows(named=True)
        if "beta_mean" in row and row["beta_mean"] is not None
    }

    mask_high = np.array([beta_map.get(v, 0.0) > high_threshold for v in vote_ids])
    mask_low = np.array([beta_map.get(v, 0.0) < low_threshold for v in vote_ids])
    # Medium-disc bills: neither high nor low — general only
    mask_medium = ~mask_high & ~mask_low

    records = []
    for i, v in enumerate(vote_ids):
        beta_abs = beta_map.get(v, 0.0)
        if mask_high[i]:
            group = "high_disc"
        elif mask_low[i]:
            group = "low_disc"
        else:
            group = "general_only"
        records.append({"vote_id": v, "abs_beta": beta_abs, "bill_group": group})

    classification = pl.DataFrame(records)

    n_high = int(mask_high.sum())
    n_low = int(mask_low.sum())
    n_med = int(mask_medium.sum())
    total = len(vote_ids)
    print(f"  Bill classification: {n_high} high-disc, {n_low} low-disc, {n_med} general-only")
    print(f"    ({total} total, thresholds: high > {high_threshold}, low < {low_threshold})")

    return mask_high, mask_low, classification


def classify_bills_by_party_line(
    vote_alignment: pl.DataFrame,
    vote_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Fallback: classify bills using EDA party-line classification.

    party-line -> high-disc proxy, bipartisan -> low-disc proxy, mixed -> general only.
    """
    alignment_map = {
        row["vote_id"]: row["vote_alignment"] for row in vote_alignment.iter_rows(named=True)
    }

    mask_high = np.array([alignment_map.get(v, "mixed") == "party-line" for v in vote_ids])
    mask_low = np.array([alignment_map.get(v, "mixed") == "bipartisan" for v in vote_ids])

    records = []
    for i, v in enumerate(vote_ids):
        alignment = alignment_map.get(v, "mixed")
        if mask_high[i]:
            group = "high_disc"
        elif mask_low[i]:
            group = "low_disc"
        else:
            group = "general_only"
        records.append({"vote_id": v, "vote_alignment": alignment, "bill_group": group})

    classification = pl.DataFrame(records)

    n_high = int(mask_high.sum())
    n_low = int(mask_low.sum())
    n_med = len(vote_ids) - n_high - n_low
    print(
        f"  Bill classification (fallback): {n_high} party-line, {n_low} bipartisan, {n_med} mixed"
    )

    return mask_high, mask_low, classification


# ── Model builder ────────────────────────────────────────────────────────────


def build_bifactor_graph(
    data: dict,
    mask_high: np.ndarray,
    mask_low: np.ndarray,
) -> pm.Model:
    """Build bifactor IRT model with general + 2 specific factors.

    P(Yea_ij) = logit^-1(
        a_G[j] * theta_G[i]
        + a_S1[j] * theta_S1[i] * mask_high[j]
        + a_S2[j] * theta_S2[i] * mask_low[j]
        - d[j]
    )

    Identification: orthogonal by construction (independent priors).
    Post-hoc sign flip on theta_G (Republican mean > 0).
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

    # Convert masks to float tensors
    mask_high_t = pt.as_tensor_variable(mask_high.astype(np.float64))
    mask_low_t = pt.as_tensor_variable(mask_low.astype(np.float64))

    with pm.Model(coords=coords) as model:
        # ── General factor: all legislators ──
        theta_G = pm.Normal("theta_G", mu=0, sigma=1, shape=n_leg, dims="legislator")

        # ── Specific factors: all legislators (activated by masks) ──
        theta_S1 = pm.Normal("theta_S1", mu=0, sigma=1, shape=n_leg, dims="legislator")
        theta_S2 = pm.Normal("theta_S2", mu=0, sigma=1, shape=n_leg, dims="legislator")

        # ── General discrimination: all bills ──
        a_G = pm.Normal("a_G", mu=0, sigma=1, shape=n_votes, dims="vote")

        # ── Specific discrimination: all bills (masked) ──
        a_S1_raw = pm.Normal("a_S1_raw", mu=0, sigma=1, shape=n_votes, dims="vote")
        a_S2_raw = pm.Normal("a_S2_raw", mu=0, sigma=1, shape=n_votes, dims="vote")

        # Apply masks: zero contribution for non-target bills
        a_S1 = pm.Deterministic("a_S1", a_S1_raw * mask_high_t, dims="vote")
        a_S2 = pm.Deterministic("a_S2", a_S2_raw * mask_low_t, dims="vote")

        # ── Difficulty ──
        d = pm.Normal("d", mu=0, sigma=5, shape=n_votes, dims="vote")

        # ── Likelihood ──
        eta = (
            a_G[vote_idx] * theta_G[leg_idx]
            + a_S1[vote_idx] * theta_S1[leg_idx]
            + a_S2[vote_idx] * theta_S2[leg_idx]
            - d[vote_idx]
        )
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

    return model


# ── Sampling ─────────────────────────────────────────────────────────────────


def build_and_sample_bifactor(
    data: dict,
    mask_high: np.ndarray,
    mask_low: np.ndarray,
    initvals: dict[str, np.ndarray] | None = None,
    n_samples: int = N_SAMPLES,
    n_tune: int = N_TUNE,
    n_chains: int = N_CHAINS,
) -> tuple[az.InferenceData, float]:
    """Build bifactor graph, compile with nutpie, and sample."""
    n_leg = data["n_legislators"]
    n_votes = data["n_votes"]
    n_high = int(mask_high.sum())
    n_low = int(mask_low.sum())

    model = build_bifactor_graph(data, mask_high, mask_low)

    compile_kwargs: dict = {}
    no_jitter_rvs: set[str] = set()

    if initvals:
        compile_kwargs["initial_points"] = initvals
        no_jitter_rvs = set(initvals.keys())
        compile_kwargs["jitter_rvs"] = {rv for rv in model.free_RVs if rv.name not in no_jitter_rvs}
        jittered = [rv.name for rv in compile_kwargs["jitter_rvs"]]
        print(f"  jitter_rvs: {jittered} ({', '.join(no_jitter_rvs)} excluded)")

    print("  Compiling bifactor model with nutpie...")
    compiled = nutpie.compile_pymc_model(model, **compile_kwargs)

    total_free = n_leg * 3 + n_votes * 4  # theta_G/S1/S2 + a_G + a_S1_raw + a_S2_raw + d
    print(f"  Sampling: {n_samples} draws, {n_tune} tune, {n_chains} chains")
    print(f"  seed={RANDOM_SEED}, sampler=nutpie (Rust NUTS)")
    print(
        f"  Parameters: theta ({n_leg}x3), a_G ({n_votes}), "
        f"a_S1 ({n_high} active), a_S2 ({n_low} active), d ({n_votes})"
    )
    print(f"  Total free params: ~{total_free}")

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


def check_bifactor_convergence(idata: az.InferenceData, chamber: str) -> dict:
    """Run convergence diagnostics for the bifactor model."""
    print_header(f"CONVERGENCE DIAGNOSTICS — BIFACTOR ({chamber})")

    diag: dict = {}

    rhat = az.rhat(idata)
    for name, var in [
        ("theta_G", "theta_G"),
        ("theta_S1", "theta_S1"),
        ("theta_S2", "theta_S2"),
        ("a_G", "a_G"),
        ("a_S1_raw", "a_S1_raw"),
        ("a_S2_raw", "a_S2_raw"),
        ("d", "d"),
    ]:
        if var in rhat:
            val = float(rhat[var].max())
            diag[f"{name}_rhat_max"] = val
            status = "OK" if val < RHAT_THRESHOLD else "WARNING"
            print(f"  R-hat ({name}):  max = {val:.4f}  {status}")

    ess = az.ess(idata)
    for name, var in [
        ("theta_G", "theta_G"),
        ("theta_S1", "theta_S1"),
        ("theta_S2", "theta_S2"),
        ("a_G", "a_G"),
    ]:
        if var in ess:
            val = float(ess[var].min())
            diag[f"{name}_ess_min"] = val
            status = "OK" if val > ESS_THRESHOLD else "WARNING"
            print(f"  Bulk ESS ({name}):  min = {val:.0f}  {status}")

    ess_tail = az.ess(idata, method="tail")
    if "theta_G" in ess_tail:
        val = float(ess_tail["theta_G"].min())
        diag["theta_G_tail_ess_min"] = val
        status = "OK" if val > ESS_THRESHOLD else "WARNING"
        print(f"  Tail ESS (theta_G):  min = {val:.0f}  {status}")

    divergences = int(idata.sample_stats["diverging"].sum().values)
    diag["divergences"] = divergences
    print(f"  Divergences:   {divergences}  {'OK' if divergences < MAX_DIVERGENCES else 'WARNING'}")

    bfmi_values = az.bfmi(idata)
    diag["ebfmi"] = [float(v) for v in bfmi_values]
    for i, v in enumerate(bfmi_values):
        print(f"  E-BFMI chain {i}: {v:.3f}  {'OK' if v > 0.3 else 'WARNING'}")

    theta_g_rhat = diag.get("theta_G_rhat_max", 0)
    theta_g_ess = diag.get("theta_G_ess_min", 0)
    diag["all_ok"] = (
        theta_g_rhat < RHAT_THRESHOLD
        and theta_g_ess > ESS_THRESHOLD
        and divergences < MAX_DIVERGENCES
    )

    if diag["all_ok"]:
        print("  CONVERGENCE: GENERAL FACTOR CHECKS PASSED (relaxed thresholds)")
    else:
        print("  CONVERGENCE: SOME CHECKS FAILED — inspect diagnostics")

    return diag


# ── ECV and diagnostics ──────────────────────────────────────────────────────


def compute_bifactor_diagnostics(
    idata: az.InferenceData,
    mask_high: np.ndarray,
    mask_low: np.ndarray,
) -> dict:
    """Compute ECV and omega_h from bifactor discrimination parameters.

    ECV = sum(a_G^2) / [sum(a_G^2) + sum(a_S1^2) + sum(a_S2^2)]
    omega_h = (sum(a_G))^2 / [(sum(a_G))^2 + sum(a_S1^2) + sum(a_S2^2) + sum(1 - h_j^2)]
    """
    a_G = idata.posterior["a_G"].mean(dim=["chain", "draw"]).values
    a_S1_raw = idata.posterior["a_S1_raw"].mean(dim=["chain", "draw"]).values
    a_S2_raw = idata.posterior["a_S2_raw"].mean(dim=["chain", "draw"]).values

    # Apply masks
    a_S1 = a_S1_raw * mask_high.astype(np.float64)
    a_S2 = a_S2_raw * mask_low.astype(np.float64)

    sum_aG_sq = float(np.sum(a_G**2))
    sum_aS1_sq = float(np.sum(a_S1**2))
    sum_aS2_sq = float(np.sum(a_S2**2))
    total_disc = sum_aG_sq + sum_aS1_sq + sum_aS2_sq

    ecv = sum_aG_sq / total_disc if total_disc > 0 else 1.0

    # Omega hierarchical
    sum_aG = float(np.sum(a_G))
    # Communality per item: h_j^2 = a_G[j]^2 + a_S1[j]^2 + a_S2[j]^2
    communality = a_G**2 + a_S1**2 + a_S2**2
    unique_var = np.sum(1.0 - communality)
    unique_var = max(float(unique_var), 0.0)  # floor at 0
    omega_h_denom = sum_aG**2 + sum_aS1_sq + sum_aS2_sq + unique_var
    omega_h = sum_aG**2 / omega_h_denom if omega_h_denom > 0 else 0.0

    if ecv > 0.70:
        interpretation = "high — unidimensional model likely adequate"
    elif ecv > 0.60:
        interpretation = "moderate — specific factors carry some signal"
    else:
        interpretation = "low — meaningful bifactor structure"

    result = {
        "ecv": ecv,
        "omega_h": omega_h,
        "sum_aG_sq": sum_aG_sq,
        "sum_aS1_sq": sum_aS1_sq,
        "sum_aS2_sq": sum_aS2_sq,
        "interpretation": interpretation,
    }

    print(f"  ECV = {ecv:.3f} ({interpretation})")
    print(f"  omega_h = {omega_h:.3f}")
    print(
        f"    General sum(a²) = {sum_aG_sq:.2f}, "
        f"S1 sum(a²) = {sum_aS1_sq:.2f}, S2 sum(a²) = {sum_aS2_sq:.2f}"
    )

    return result


# ── Extract ideal points ─────────────────────────────────────────────────────


def extract_bifactor_ideal_points(
    idata: az.InferenceData,
    data: dict,
    legislators: pl.DataFrame,
) -> pl.DataFrame:
    """Extract bifactor ideal point posteriors with HDIs."""
    theta_G_post = idata.posterior["theta_G"]
    theta_S1_post = idata.posterior["theta_S1"]
    theta_S2_post = idata.posterior["theta_S2"]

    theta_G_mean = theta_G_post.mean(dim=["chain", "draw"]).values
    theta_S1_mean = theta_S1_post.mean(dim=["chain", "draw"]).values
    theta_S2_mean = theta_S2_post.mean(dim=["chain", "draw"]).values

    theta_G_hdi = az.hdi(idata, var_names=["theta_G"], hdi_prob=0.95)["theta_G"].values
    theta_S1_hdi = az.hdi(idata, var_names=["theta_S1"], hdi_prob=0.95)["theta_S1"].values
    theta_S2_hdi = az.hdi(idata, var_names=["theta_S2"], hdi_prob=0.95)["theta_S2"].values

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
                "theta_G_mean": float(theta_G_mean[i]),
                "theta_G_hdi_3%": float(theta_G_hdi[i, 0]),
                "theta_G_hdi_97%": float(theta_G_hdi[i, 1]),
                "theta_S1_mean": float(theta_S1_mean[i]),
                "theta_S1_hdi_3%": float(theta_S1_hdi[i, 0]),
                "theta_S1_hdi_97%": float(theta_S1_hdi[i, 1]),
                "theta_S2_mean": float(theta_S2_mean[i]),
                "theta_S2_hdi_3%": float(theta_S2_hdi[i, 0]),
                "theta_S2_hdi_97%": float(theta_S2_hdi[i, 1]),
            }
        )

    return pl.DataFrame(records)


def extract_bifactor_bill_params(
    idata: az.InferenceData,
    data: dict,
    mask_high: np.ndarray,
    mask_low: np.ndarray,
    rollcalls: pl.DataFrame,
) -> pl.DataFrame:
    """Extract bill parameter posteriors."""
    a_G_mean = idata.posterior["a_G"].mean(dim=["chain", "draw"]).values
    a_G_sd = idata.posterior["a_G"].std(dim=["chain", "draw"]).values
    a_S1_raw_mean = idata.posterior["a_S1_raw"].mean(dim=["chain", "draw"]).values
    a_S2_raw_mean = idata.posterior["a_S2_raw"].mean(dim=["chain", "draw"]).values
    d_mean = idata.posterior["d"].mean(dim=["chain", "draw"]).values
    d_sd = idata.posterior["d"].std(dim=["chain", "draw"]).values

    vote_ids = data["vote_ids"]

    # Map vote_id to bill_number and short_title
    bill_map = {}
    for row in rollcalls.iter_rows(named=True):
        vid = row.get("vote_id", "")
        bill_map[vid] = {
            "bill_number": row.get("bill_number", ""),
            "short_title": row.get("short_title", ""),
        }

    records = []
    for j, vid in enumerate(vote_ids):
        info = bill_map.get(vid, {})
        if mask_high[j]:
            group = "high_disc"
        elif mask_low[j]:
            group = "low_disc"
        else:
            group = "general_only"

        records.append(
            {
                "vote_id": vid,
                "bill_number": info.get("bill_number", ""),
                "short_title": info.get("short_title", ""),
                "a_G_mean": float(a_G_mean[j]),
                "a_G_sd": float(a_G_sd[j]),
                "a_S1_mean": float(a_S1_raw_mean[j] * mask_high[j]),
                "a_S2_mean": float(a_S2_raw_mean[j] * mask_low[j]),
                "d_mean": float(d_mean[j]),
                "d_sd": float(d_sd[j]),
                "bill_group": group,
            }
        )

    return pl.DataFrame(records)


# ── Post-hoc sign check ─────────────────────────────────────────────────────


def apply_general_sign_check(ideal_bf: pl.DataFrame) -> pl.DataFrame:
    """Flip general factor sign if Republican mean is negative."""
    r_theta_G = ideal_bf.filter(pl.col("party") == "Republican")["theta_G_mean"].mean()
    if r_theta_G is not None and r_theta_G < 0:
        print("  WARNING: Republican theta_G mean is negative — flipping sign")
        ideal_bf = (
            ideal_bf.with_columns(
                (-pl.col("theta_G_mean")).alias("theta_G_mean"),
                (-pl.col("theta_G_hdi_3%")).alias("theta_G_hdi_97%_new"),
                (-pl.col("theta_G_hdi_97%")).alias("theta_G_hdi_3%_new"),
            )
            .drop("theta_G_hdi_3%", "theta_G_hdi_97%")
            .rename(
                {
                    "theta_G_hdi_97%_new": "theta_G_hdi_97%",
                    "theta_G_hdi_3%_new": "theta_G_hdi_3%",
                }
            )
        )
    else:
        print(f"  Republican theta_G mean: {r_theta_G:+.3f} (positive — no flip needed)")
    return ideal_bf


# ── Comparison with upstream models ──────────────────────────────────────────


def correlate_with_upstream(
    ideal_bf: pl.DataFrame,
    irt_1d: pl.DataFrame | None,
    irt_2d: pl.DataFrame | None,
    data: dict,
) -> dict:
    """Correlate bifactor general factor with 1D IRT and 2D Dim 1."""
    corrs: dict = {}
    slugs = data["leg_slugs"]
    bf_map = dict(zip(ideal_bf["legislator_slug"].to_list(), ideal_bf["theta_G_mean"].to_list()))

    if irt_1d is not None:
        irt1d_map = {row["legislator_slug"]: row["xi_mean"] for row in irt_1d.iter_rows(named=True)}
        shared = [s for s in slugs if s in irt1d_map and s in bf_map]
        if len(shared) > 3:
            bf_vals = [bf_map[s] for s in shared]
            irt_vals = [irt1d_map[s] for s in shared]
            corrs["general_vs_1d_pearson"] = float(np.corrcoef(bf_vals, irt_vals)[0, 1])
            corrs["general_vs_1d_spearman"] = float(stats.spearmanr(bf_vals, irt_vals).statistic)
            print(
                f"  theta_G vs 1D IRT: r = {corrs['general_vs_1d_pearson']:.4f}, "
                f"rho = {corrs['general_vs_1d_spearman']:.4f}"
            )

    if irt_2d is not None:
        irt2d_map = {
            row["legislator_slug"]: row["xi_dim1_mean"] for row in irt_2d.iter_rows(named=True)
        }
        shared = [s for s in slugs if s in irt2d_map and s in bf_map]
        if len(shared) > 3:
            bf_vals = [bf_map[s] for s in shared]
            dim1_vals = [irt2d_map[s] for s in shared]
            corrs["general_vs_2d_dim1_pearson"] = float(np.corrcoef(bf_vals, dim1_vals)[0, 1])
            corrs["general_vs_2d_dim1_spearman"] = float(
                stats.spearmanr(bf_vals, dim1_vals).statistic
            )
            print(
                f"  theta_G vs 2D Dim 1: r = {corrs['general_vs_2d_dim1_pearson']:.4f}, "
                f"rho = {corrs['general_vs_2d_dim1_spearman']:.4f}"
            )

    return corrs


# ── Plots ────────────────────────────────────────────────────────────────────


def plot_bifactor_scatter(
    ideal_bf: pl.DataFrame,
    chamber: str,
    output_dir: Path,
) -> None:
    """Scatter: theta_G vs theta_S1, party-colored."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for party, color in PARTY_COLORS.items():
        subset = ideal_bf.filter(pl.col("party") == party)
        if subset.height == 0:
            continue
        ax.scatter(
            subset["theta_G_mean"].to_list(),
            subset["theta_S1_mean"].to_list(),
            c=color,
            alpha=0.7,
            s=40,
            label=party,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel("General Factor (Ideology: Liberal <- -> Conservative)", fontsize=11)
    ax.set_ylabel("Specific Factor 1 (Partisan Bills)", fontsize=11)
    ax.set_title(
        f"Bifactor IRT — Kansas {chamber} (EXPERIMENTAL)",
        fontsize=13,
        fontweight="bold",
    )
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_fig(fig, output_dir / f"bifactor_scatter_{chamber.lower()}.png")


def plot_general_forest(
    ideal_bf: pl.DataFrame,
    chamber: str,
    output_dir: Path,
) -> None:
    """Forest plot of general factor (theta_G) with HDIs."""
    sorted_df = ideal_bf.sort("theta_G_mean")
    n = sorted_df.height

    fig, ax = plt.subplots(figsize=(8, max(6, n * 0.25)))

    for i, row in enumerate(sorted_df.iter_rows(named=True)):
        color = PARTY_COLORS.get(row["party"], "#999999")
        ax.errorbar(
            row["theta_G_mean"],
            i,
            xerr=[
                [row["theta_G_mean"] - row["theta_G_hdi_3%"]],
                [row["theta_G_hdi_97%"] - row["theta_G_mean"]],
            ],
            fmt="o",
            color=color,
            markersize=4,
            linewidth=1,
            capsize=2,
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_df["full_name"].to_list(), fontsize=7)
    ax.set_xlabel("General Factor (theta_G)", fontsize=11)
    ax.set_title(
        f"Bifactor General Factor — Kansas {chamber} (EXPERIMENTAL)",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    save_fig(fig, output_dir / f"general_forest_{chamber.lower()}.png")


def plot_general_vs_1d(
    ideal_bf: pl.DataFrame,
    irt_1d: pl.DataFrame,
    chamber: str,
    output_dir: Path,
) -> None:
    """Correlation scatter: bifactor theta_G vs 1D IRT xi."""
    irt1d_map = {row["legislator_slug"]: row["xi_mean"] for row in irt_1d.iter_rows(named=True)}
    shared = ideal_bf.filter(pl.col("legislator_slug").is_in(list(irt1d_map.keys())))

    bf_vals = shared["theta_G_mean"].to_list()
    irt_vals = [irt1d_map[s] for s in shared["legislator_slug"].to_list()]
    r = float(np.corrcoef(bf_vals, irt_vals)[0, 1])

    fig, ax = plt.subplots(figsize=(7, 7))
    for party, color in PARTY_COLORS.items():
        subset = shared.filter(pl.col("party") == party)
        if subset.height == 0:
            continue
        s_bf = subset["theta_G_mean"].to_list()
        s_irt = [irt1d_map[s] for s in subset["legislator_slug"].to_list()]
        ax.scatter(s_irt, s_bf, c=color, alpha=0.7, s=30, label=party)

    ax.set_xlabel("1D IRT Ideal Point (xi)", fontsize=11)
    ax.set_ylabel("Bifactor General Factor (theta_G)", fontsize=11)
    ax.set_title(
        f"Bifactor theta_G vs 1D IRT — {chamber}  (r = {r:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    fig.tight_layout()
    save_fig(fig, output_dir / f"general_vs_1d_{chamber.lower()}.png")


def plot_ecv_bar(
    bf_diag: dict,
    chamber: str,
    output_dir: Path,
) -> None:
    """Stacked bar showing variance decomposition: general vs specific factors."""
    ecv = bf_diag["ecv"]
    sum_aG = bf_diag["sum_aG_sq"]
    sum_aS1 = bf_diag["sum_aS1_sq"]
    sum_aS2 = bf_diag["sum_aS2_sq"]
    total = sum_aG + sum_aS1 + sum_aS2

    if total == 0:
        return

    fracs = [sum_aG / total, sum_aS1 / total, sum_aS2 / total]
    labels = ["General (ideology)", "Specific 1 (partisan)", "Specific 2 (bipartisan)"]
    colors = ["#4A90D9", "#E81B23", "#999999"]

    fig, ax = plt.subplots(figsize=(8, 3))
    left = 0.0
    for frac, label, color in zip(fracs, labels, colors):
        ax.barh(0, frac, left=left, color=color, label=f"{label}: {frac:.1%}", height=0.5)
        left += frac

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Proportion of Discriminating Variance", fontsize=11)
    ax.set_title(
        f"Explained Common Variance — {chamber}  (ECV = {ecv:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save_fig(fig, output_dir / f"ecv_bar_{chamber.lower()}.png")


def plot_factor_loadings_heatmap(
    bill_params: pl.DataFrame,
    chamber: str,
    output_dir: Path,
) -> None:
    """Heatmap of a_G, a_S1, a_S2 per bill, sorted by |a_G|."""
    df = bill_params.sort(pl.col("a_G_mean").abs(), descending=True).head(50)

    fig, ax = plt.subplots(figsize=(8, max(6, df.height * 0.2)))

    data_matrix = np.column_stack(
        [
            df["a_G_mean"].to_numpy(),
            df["a_S1_mean"].to_numpy(),
            df["a_S2_mean"].to_numpy(),
        ]
    )

    im = ax.imshow(data_matrix, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["General (a_G)", "Specific 1 (a_S1)", "Specific 2 (a_S2)"])

    labels = df["bill_number"].to_list()
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)

    ax.set_title(
        f"Factor Loadings (top 50 by |a_G|) — {chamber}",
        fontsize=12,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Discrimination", shrink=0.8)
    fig.tight_layout()
    save_fig(fig, output_dir / f"factor_loadings_{chamber.lower()}.png")


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
        "01_eda", results_root, args.run_id, Path(args.eda_dir) if args.eda_dir else None
    )
    pca_dir = resolve_upstream_dir(
        "02_pca", results_root, args.run_id, Path(args.pca_dir) if args.pca_dir else None
    )
    irt_dir = resolve_upstream_dir(
        "05_irt", results_root, args.run_id, Path(args.irt_dir) if args.irt_dir else None
    )

    # Optional: 2D IRT for comparison
    irt_2d_dir = None
    if args.irt_2d_dir:
        irt_2d_dir = Path(args.irt_2d_dir)
    else:
        try:
            irt_2d_dir = resolve_upstream_dir("06_irt_2d", results_root, args.run_id, None)
        except FileNotFoundError:
            pass

    with RunContext(
        session=args.session,
        analysis_name="06b_bifactor",
        params=vars(args),
        primer=BIFACTOR_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print("=" * 80)
        print("  BIFACTOR IRT — GENERAL + SPECIFIC FACTOR IDEAL POINTS (EXPERIMENTAL)")
        print("=" * 80)
        print(f"  Session: {args.session}")
        print(f"  Data:    {data_dir}")
        print(f"  EDA:     {eda_dir}")
        print(f"  PCA:     {pca_dir}")
        print(f"  IRT:     {irt_dir}")
        print(f"  2D IRT:  {irt_2d_dir or '(not found)'}")
        print(f"  Output:  {ctx.run_dir}")
        print(f"  Samples: {args.n_samples} draws, {args.n_tune} tune, {args.n_chains} chains")
        print()
        print("  *** EXPERIMENTAL: Bifactor IRT with relaxed thresholds ***")
        print()

        # ── Load data ──
        print_header("LOADING DATA")
        house_matrix, senate_matrix, _ = load_eda_matrices(eda_dir)
        pca_house, pca_senate = load_pca_scores(pca_dir)
        rollcalls, legislators = load_metadata(data_dir)

        # Load 1D IRT (for bill classification + comparison)
        irt_data_dir = irt_dir / "data"
        irt_house = load_irt_scores(irt_data_dir, "house")
        irt_senate = load_irt_scores(irt_data_dir, "senate")
        for ch, irt_df in [("House", irt_house), ("Senate", irt_senate)]:
            if irt_df is not None:
                print(f"  1D IRT loaded: {ch} ({irt_df.height} legislators)")
            else:
                print(f"  1D IRT not found: {ch}")

        # Load bill params from Phase 05 for discrimination-based classification
        bill_params_house = None
        bill_params_senate = None
        bp_house_path = irt_data_dir / "bill_params_house.parquet"
        bp_senate_path = irt_data_dir / "bill_params_senate.parquet"
        if bp_house_path.exists():
            bill_params_house = pl.read_parquet(bp_house_path)
            print(f"  Bill params loaded: House ({bill_params_house.height} bills)")
        if bp_senate_path.exists():
            bill_params_senate = pl.read_parquet(bp_senate_path)
            print(f"  Bill params loaded: Senate ({bill_params_senate.height} bills)")

        # Fallback: vote alignment from EDA
        vote_alignment = None
        va_path = eda_dir / "data" / "vote_alignment.parquet"
        if va_path.exists():
            vote_alignment = pl.read_parquet(va_path)
            print(f"  Vote alignment loaded (fallback): {vote_alignment.height} votes")

        # Load 2D IRT for comparison (optional)
        irt_2d_house = None
        irt_2d_senate = None
        if irt_2d_dir:
            irt_2d_data = irt_2d_dir / "data"
            p = irt_2d_data / "ideal_points_2d_house.parquet"
            if p.exists():
                irt_2d_house = pl.read_parquet(p)
                print(f"  2D IRT loaded: House ({irt_2d_house.height} legislators)")
            p = irt_2d_data / "ideal_points_2d_senate.parquet"
            if p.exists():
                irt_2d_senate = pl.read_parquet(p)
                print(f"  2D IRT loaded: Senate ({irt_2d_senate.height} legislators)")

        print(f"  House filtered: {house_matrix.height} x {len(house_matrix.columns) - 1}")
        print(f"  Senate filtered: {senate_matrix.height} x {len(senate_matrix.columns) - 1}")

        chamber_configs = [
            ("House", house_matrix, pca_house, irt_house, bill_params_house, irt_2d_house),
            ("Senate", senate_matrix, pca_senate, irt_senate, bill_params_senate, irt_2d_senate),
        ]

        all_results: dict[str, dict] = {}
        t_total = time.time()

        for chamber, matrix, pca_scores, irt_1d, bp, irt_2d in chamber_configs:
            if matrix.height < 5:
                print(f"\n  Skipping {chamber}: too few legislators ({matrix.height})")
                continue

            chamber_lower = chamber.lower()

            # ── Contested-only filtering ──
            if args.contested_only:
                n_before = len(matrix.columns) - 1
                vote_cols = [c for c in matrix.columns if c != "legislator_slug"]
                minority_fracs = []
                for vc in vote_cols:
                    col = matrix[vc].drop_nulls()
                    if col.len() > 0:
                        frac = float(col.mean())
                        minority_fracs.append((vc, min(frac, 1 - frac)))
                contested = [vc for vc, mf in minority_fracs if mf > CONTESTED_THRESHOLD]
                matrix = matrix.select(["legislator_slug"] + contested)
                print(f"  Contested-only filter: {n_before} -> {len(contested)} votes")

            # ── Adaptive N_TUNE for supermajority ──
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
                    f"N_TUNE increased: {N_TUNE} -> {N_TUNE_SUPERMAJORITY}"
                )

            # ── Prepare IRT data ──
            print_header(f"PREPARE IRT DATA — {chamber}")
            data = prepare_irt_data(matrix, chamber)

            # ── Classify bills ──
            print_header(f"BILL CLASSIFICATION — {chamber}")
            if bp is not None:
                mask_high, mask_low, classification = classify_bills_by_discrimination(
                    bp, data["vote_ids"]
                )
                classification_method = "discrimination"
            elif vote_alignment is not None:
                mask_high, mask_low, classification = classify_bills_by_party_line(
                    vote_alignment, data["vote_ids"]
                )
                classification_method = "party-line (fallback)"
            else:
                # No classification available — all bills general only
                print("  WARNING: No bill classification available — all bills general only")
                n_votes = data["n_votes"]
                mask_high = np.zeros(n_votes, dtype=bool)
                mask_low = np.zeros(n_votes, dtype=bool)
                classification = pl.DataFrame(
                    {"vote_id": data["vote_ids"], "bill_group": ["general_only"] * n_votes}
                )
                classification_method = "none (all general)"

            # ── Build initialization ──
            print_header(f"BIFACTOR INITIALIZATION — {chamber}")
            slugs = data["leg_slugs"]

            # General factor: ideology (PCA PC1 / LDA ideology_score / 1D IRT)
            theta_G_std, g_strategy, g_source = resolve_init_source(
                strategy=args.init_strategy,
                slugs=slugs,
                irt_scores=irt_1d,
                pca_scores=pca_scores,
                pca_column="PC1",
                session=args.session,
                chamber=chamber_lower,
            )
            print(f"  theta_G init: {g_source} (strategy: {g_strategy})")

            # Specific factor 1: PCA PC2 (establishment axis)
            theta_S1_std, _, s1_source = resolve_init_source(
                strategy=InitStrategy.PCA_INFORMED,
                slugs=slugs,
                pca_scores=pca_scores,
                pca_column="PC2",
                session=args.session,
                chamber=chamber_lower,
            )
            print(f"  theta_S1 init: {s1_source}")

            # Specific factor 2: zeros
            theta_S2_std = np.zeros(len(slugs))
            print("  theta_S2 init: zeros")

            initvals = {
                "theta_G": theta_G_std,
                "theta_S1": theta_S1_std,
                "theta_S2": theta_S2_std,
            }

            # ── Build and sample bifactor model ──
            print_header(f"MCMC SAMPLING — BIFACTOR ({chamber})")
            idata, sampling_time = build_and_sample_bifactor(
                data=data,
                mask_high=mask_high,
                mask_low=mask_low,
                initvals=initvals,
                n_samples=args.n_samples,
                n_tune=effective_n_tune,
                n_chains=args.n_chains,
            )

            # ── Convergence diagnostics ──
            diag = check_bifactor_convergence(idata, chamber)

            # ── ECV and omega_h ──
            bf_diag = compute_bifactor_diagnostics(idata, mask_high, mask_low)
            diag["ecv"] = bf_diag["ecv"]
            diag["omega_h"] = bf_diag["omega_h"]
            diag["ecv_interpretation"] = bf_diag["interpretation"]

            # ── Extract ideal points ──
            print_header(f"EXTRACT BIFACTOR IDEAL POINTS — {chamber}")
            ideal_bf = extract_bifactor_ideal_points(idata, data, legislators)
            ideal_bf = apply_general_sign_check(ideal_bf)

            # ── Extract bill parameters ──
            bill_params_bf = extract_bifactor_bill_params(
                idata, data, mask_high, mask_low, rollcalls
            )

            # ── Comparison with upstream ──
            print_header(f"CORRELATION CHECKS — {chamber}")
            corrs = correlate_with_upstream(ideal_bf, irt_1d, irt_2d, data)

            # ── Save outputs ──
            print_header(f"SAVING OUTPUTS — {chamber}")

            ideal_bf.write_parquet(ctx.data_dir / f"ideal_points_bifactor_{chamber_lower}.parquet")
            print(
                f"  Saved: ideal_points_bifactor_{chamber_lower}.parquet "
                f"({ideal_bf.height} legislators)"
            )

            bill_params_bf.write_parquet(
                ctx.data_dir / f"bill_params_bifactor_{chamber_lower}.parquet"
            )
            print(f"  Saved: bill_params_bifactor_{chamber_lower}.parquet")

            classification.write_parquet(
                ctx.data_dir / f"bill_classification_{chamber_lower}.parquet"
            )
            print(f"  Saved: bill_classification_{chamber_lower}.parquet")

            nc_path = ctx.data_dir / f"idata_{chamber_lower}.nc"
            idata.to_netcdf(str(nc_path))
            print(f"  Saved: idata_{chamber_lower}.nc")

            # ── Generate plots ──
            plot_bifactor_scatter(ideal_bf, chamber, ctx.plots_dir)
            plot_general_forest(ideal_bf, chamber, ctx.plots_dir)
            plot_ecv_bar(bf_diag, chamber, ctx.plots_dir)
            plot_factor_loadings_heatmap(bill_params_bf, chamber, ctx.plots_dir)

            if irt_1d is not None:
                plot_general_vs_1d(ideal_bf, irt_1d, chamber, ctx.plots_dir)

            # ── Print ideal points table ──
            print_header(f"BIFACTOR IDEAL POINTS — {chamber}")
            sorted_table = ideal_bf.sort("theta_G_mean", descending=True)
            header = (
                f"\n  {'Name':<25s} {'Party':>10s}  {'theta_G':>8s}  "
                f"{'theta_S1':>8s}  {'theta_S2':>8s}"
            )
            print(header)
            print("  " + "-" * 70)
            for row in sorted_table.iter_rows(named=True):
                print(
                    f"  {row['full_name']:<25s} {row['party']:>10s}  "
                    f"{row['theta_G_mean']:+.3f}  "
                    f"{row['theta_S1_mean']:+.3f}  "
                    f"{row['theta_S2_mean']:+.3f}"
                )

            all_results[chamber] = {
                "ideal_points": ideal_bf,
                "bill_params": bill_params_bf,
                "diagnostics": diag,
                "bifactor_diagnostics": bf_diag,
                "correlations": corrs,
                "data": data,
                "sampling_time": sampling_time,
                "classification_method": classification_method,
                "classification": classification,
                "mask_high": mask_high,
                "mask_low": mask_low,
            }

        # ── Save convergence summary ──
        summary: dict = {
            "session": args.session,
            "n_samples": args.n_samples,
            "n_tune": args.n_tune,
            "n_chains": args.n_chains,
            "sampler": "nutpie (Rust NUTS)",
            "model": "bifactor IRT (general + 2 specific)",
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
                "bifactor_diagnostics": result["bifactor_diagnostics"],
                "classification_method": result["classification_method"],
            }

        with open(ctx.data_dir / "convergence_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print("\n  Saved: convergence_summary.json")

        # ── Build HTML report ──
        print_header("BUILDING HTML REPORT")
        build_bifactor_report(
            ctx.report,
            results=all_results,
            plots_dir=ctx.plots_dir,
            n_samples=args.n_samples,
            n_tune=args.n_tune,
            n_chains=args.n_chains,
        )

        # ── Final summary ──
        total_time = time.time() - t_total
        print_header("EXPERIMENT SUMMARY")
        print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
        for chamber, result in all_results.items():
            diag = result["diagnostics"]
            bf_diag = result["bifactor_diagnostics"]
            print(f"\n  {chamber}:")
            print(f"    Sampling time: {result['sampling_time']:.0f}s")
            print(f"    Convergence: {'PASSED' if diag['all_ok'] else 'FAILED'}")
            print(f"    ECV = {bf_diag['ecv']:.3f}, omega_h = {bf_diag['omega_h']:.3f}")
            corrs = result["correlations"]
            if "general_vs_1d_pearson" in corrs:
                print(f"    theta_G vs 1D IRT: r = {corrs['general_vs_1d_pearson']:.4f}")
        print(f"\n  Output: {ctx.run_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()
