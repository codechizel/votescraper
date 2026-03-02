"""
Kansas Legislature — Dynamic Ideal Point Estimation (Phase 16)

Martin-Quinn style state-space IRT model that jointly estimates time-varying
ideal points across bienniums.  Tracks who moved, and when, across a
legislator's career.  Operates in cross-session flat mode, loading data from
all 8 scraped bienniums (84th–91st, 2011–2026).

Usage:
  uv run python analysis/16_dynamic_irt/dynamic_irt.py
      [--chambers house|senate|both]
      [--n-samples 1000] [--n-tune 1000] [--n-chains 2]
      [--evolution global|per_party]
      [--skip-emirt]
      [--bienniums 2023-24,2025-26]

Outputs (in results/kansas/cross-session/dynamic_irt/<date>/):
  - data/:   Parquet files (trajectories, decomposition, bridge coverage) + NetCDF
  - plots/:  PNG visualizations
  - 16_dynamic_irt_report.html
"""

import argparse
import json
import shutil
import subprocess
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
    from analysis.dynamic_irt_report import build_dynamic_irt_report
except ModuleNotFoundError:
    from dynamic_irt_report import build_dynamic_irt_report  # type: ignore[no-redef]

try:
    from analysis.dynamic_irt_data import (
        BIENNIUM_LABELS,
        BIENNIUM_SESSIONS,
        MIN_BRIDGE_OVERLAP,
        SESSION_TO_LABEL,
        build_global_roster,
        compute_adjacent_bridges,
        compute_bridge_coverage,
        load_emirt_results,
        normalize_name,
        prepare_emirt_csv,
        stack_bienniums,
    )
except ModuleNotFoundError:
    from dynamic_irt_data import (  # type: ignore[no-redef]
        BIENNIUM_LABELS,
        BIENNIUM_SESSIONS,
        MIN_BRIDGE_OVERLAP,
        SESSION_TO_LABEL,
        build_global_roster,
        compute_adjacent_bridges,
        compute_bridge_coverage,
        load_emirt_results,
        normalize_name,
        prepare_emirt_csv,
        stack_bienniums,
    )

try:
    from analysis.irt import (
        load_eda_matrices,
        load_metadata,
        load_pca_scores,
        prepare_irt_data,
    )
except ModuleNotFoundError:
    from irt import (  # type: ignore[no-redef]
        load_eda_matrices,
        load_metadata,
        load_pca_scores,
        prepare_irt_data,
    )

from tallgrass.session import KSSession

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_N_SAMPLES: int = 2000
DEFAULT_N_TUNE: int = 2000
DEFAULT_N_CHAINS: int = 4
RANDOM_SEED: int = 42
RHAT_THRESHOLD: float = 1.05
"""Relaxed from standard 1.01 — dynamic IRT with ~10K params. Document in ADR."""
ESS_THRESHOLD: int = 400
MAX_DIVERGENCES: int = 50
"""Higher tolerance for large state-space model."""
TOP_MOVERS_N: int = 20
"""Number of top movers to display."""
SMALL_CHAMBER_THRESHOLD: int = 80
"""Chambers with fewer legislators get tighter tau priors (ADR-0070)."""
SMALL_CHAMBER_TAU_SIGMA: float = 0.15
"""Tau prior sigma for small chambers — prevents mode-splitting."""
DEFAULT_TAU_SIGMA: float = 0.5
"""Tau prior sigma for large chambers."""

PARTY_COLORS_DYNAMIC: dict[str, str] = {
    "Republican": "#E81B23",
    "Democrat": "#0015BC",
    "Independent": "#999999",
}

# ── Primer ───────────────────────────────────────────────────────────────────

DYNAMIC_IRT_PRIMER = """\
# Dynamic Ideal Point Estimation

## Purpose

Estimates time-varying legislator ideal points across all scraped bienniums
(84th–91st, 2011–2026) using a Martin-Quinn style state-space IRT model.
Tracks *who moved, and when* across a legislator's career — a question static
per-biennium IRT cannot answer.

## Method

### State-Space 2PL IRT Model (Non-Centered Random Walk)

```
tau ~ HalfNormal(sigma)                      -- evolution SD (adaptive: 0.15 small, 0.5 large)
xi_init ~ Normal(mu, sigma)                  -- mu=static IRT if available, else 0; sigma=0.75/1
xi_innovations ~ Normal(0, 1)                -- non-centered innovations
xi[0] = xi_init
xi[t] = xi[t-1] + tau * xi_innovations[t-1]  -- random walk
alpha ~ Normal(0, 5)                         -- bill difficulty (per biennium)
beta ~ HalfNormal(2.5)                       -- bill discrimination (positive for sign ID)
P(Yea) = logit^-1(beta * xi - alpha)
```

**Identification:** Three-layer strategy (ADR-0070):
1. Positive beta (HalfNormal) fixes discrimination sign.
2. Informative xi_init prior from static IRT transfers the well-identified sign convention.
3. Post-hoc sign correction (ADR-0068) as diagnostic safety net.

**Small chambers:** Chambers with <80 legislators auto-switch to global tau with
tighter prior (sigma=0.15) to prevent mode-splitting.

## Inputs

- Vote matrices from 01_eda (all bienniums)
- PCA scores from 02_pca (for initialization)
- Static IRT ideal points from 04_irt (for informative prior + sign correction)
- Legislator CSVs (for party/chamber/name matching)

## Outputs

- Dynamic ideal point trajectories with 95% credible intervals
- Polarization trend (party means across time)
- Conversion vs. replacement decomposition
- Top movers by total and net ideological shift
- Bridge coverage analysis (cross-biennium legislator overlap)

## Interpretation Guide

- **Polarization trend**: increasing distance between party means → growing polarization.
- **Conversion vs. replacement**: conversion = returning members moving; replacement = new
  members are more extreme than departing ones. The balance reveals the mechanism of change.
- **Top movers**: legislators with the largest cumulative ideological shift. Net movement
  distinguishes consistent drift from oscillation.
- **tau (evolution SD)**: larger tau → more ideological movement between bienniums. Per-party
  tau reveals whether one party is more ideologically volatile.

## Caveats

- Legislators absent from a biennium have no observations → ideal point regularized by
  the random walk prior (wide posterior). These are *interpolated*, not estimated.
- 84th biennium (2011–12) has weaker data (~30% committee-of-the-whole votes).
- 84th→85th bridge is weakest (post-2012 redistricting).
"""


# ── Model ────────────────────────────────────────────────────────────────────


def build_dynamic_irt_graph(
    data: dict,
    evolution_structure: str = "per_party",
    tau_sigma: float | None = None,
    xi_init_mu: np.ndarray | None = None,
) -> pm.Model:
    """Build state-space 2PL IRT model graph (no sampling).

    Model structure:
        tau (evolution SD) → xi (random walk) → 2PL likelihood.
        Non-centered parameterization: xi[t] = xi[t-1] + tau * innovation[t-1].

    Args:
        data: Stacked data dict from ``stack_bienniums()``.
        evolution_structure: ``"per_party"`` (tau per party, 2 params) or
            ``"global"`` (single tau, 1 param).
        tau_sigma: Explicit tau prior sigma. If None, uses adaptive logic
            (0.15 for small chambers, 0.5 for large — ADR-0070).
        xi_init_mu: Optional prior mean for ``xi_init`` from static IRT.
            When provided, uses ``Normal(xi_init_mu, 0.75)`` instead of
            ``Normal(0, 1)`` for sign identification (ADR-0070).

    Returns:
        PyMC model ready for nutpie compilation.
    """
    leg_global_idx = data["leg_global_idx"]
    bill_idx = data["bill_idx"]
    time_idx = data["time_idx"]
    y = data["y"]
    n_leg = data["n_legislators"]
    n_bills = data["n_bills"]
    n_time = data["n_time"]
    party_idx = data["party_idx"]
    party_names = data["party_names"]
    n_parties = len(party_names)
    leg_periods = data["leg_periods"]

    # Build mask: which legislators served in which periods
    # This is used for the random walk — only innovate when the legislator exists
    served_mask = np.zeros((n_time, n_leg), dtype=bool)
    for gidx, periods in enumerate(leg_periods):
        for t in periods:
            if t < n_time:
                served_mask[t, gidx] = True

    coords = {
        "time": BIENNIUM_LABELS[:n_time],
        "legislator": data["leg_names"],
        "bill": data["bill_ids"],
        "party": party_names,
        "obs_id": np.arange(data["n_obs"]),
    }

    with pm.Model(coords=coords) as model:
        # --- Evolution variance ---
        # Adaptive tau sigma: small chambers get tighter prior to prevent mode-splitting
        if tau_sigma is not None:
            _tau_sigma = tau_sigma
        elif n_leg < SMALL_CHAMBER_THRESHOLD:
            _tau_sigma = SMALL_CHAMBER_TAU_SIGMA
        else:
            _tau_sigma = DEFAULT_TAU_SIGMA

        # Auto-switch to global tau for small chambers with per-party request
        _evolution = evolution_structure
        if _evolution == "per_party" and n_leg < SMALL_CHAMBER_THRESHOLD:
            print(
                f"    Small chamber ({n_leg} legislators < {SMALL_CHAMBER_THRESHOLD}): "
                f"using global tau (sigma={_tau_sigma})"
            )
            _evolution = "global"

        if _evolution == "per_party":
            tau = pm.HalfNormal("tau", sigma=_tau_sigma, shape=n_parties, dims="party")
            # Broadcast tau to each legislator's party
            tau_leg = tau[party_idx]  # shape (n_leg,)
        else:
            tau = pm.HalfNormal("tau", sigma=_tau_sigma)
            tau_leg = pt.ones(n_leg) * tau

        # --- Non-centered random walk for ideal points ---
        if xi_init_mu is not None:
            xi_init = pm.Normal(
                "xi_init", mu=xi_init_mu, sigma=0.75, shape=n_leg, dims="legislator"
            )
        else:
            xi_init = pm.Normal("xi_init", mu=0, sigma=1, shape=n_leg, dims="legislator")
        xi_innovations = pm.Normal("xi_innovations", mu=0, sigma=1, shape=(n_time - 1, n_leg))

        # Build xi trajectory: xi[0] = xi_init, xi[t] = xi[t-1] + tau * innov[t-1]
        xi_list = [xi_init]
        for t in range(1, n_time):
            xi_t = xi_list[t - 1] + tau_leg * xi_innovations[t - 1]
            xi_list.append(xi_t)

        xi = pm.Deterministic("xi", pt.stack(xi_list, axis=0), dims=("time", "legislator"))

        # --- Bill parameters (per biennium) ---
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_bills, dims="bill")
        beta = pm.HalfNormal("beta", sigma=2.5, shape=n_bills, dims="bill")

        # --- Likelihood ---
        eta = beta[bill_idx] * xi[time_idx, leg_global_idx] - alpha[bill_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

    return model


def sample_dynamic_irt(
    data: dict,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_tune: int = DEFAULT_N_TUNE,
    n_chains: int = DEFAULT_N_CHAINS,
    evolution_structure: str = "per_party",
    xi_init_values: np.ndarray | None = None,
    tau_sigma: float | None = None,
    xi_init_mu: np.ndarray | None = None,
) -> tuple[az.InferenceData, float]:
    """Build dynamic IRT model and sample with nutpie.

    Args:
        data: Stacked data dict from ``stack_bienniums()``.
        n_samples: Posterior draws per chain.
        n_tune: Tuning steps (discarded).
        n_chains: Independent MCMC chains.
        evolution_structure: ``"per_party"`` or ``"global"``.
        xi_init_values: Optional PCA-informed init for ``xi_init``.
        tau_sigma: Explicit tau prior sigma (forwarded to graph builder).
        xi_init_mu: Informative prior mean for ``xi_init`` from static IRT.

    Returns:
        (InferenceData, sampling_time_seconds).
    """
    model = build_dynamic_irt_graph(
        data, evolution_structure, tau_sigma=tau_sigma, xi_init_mu=xi_init_mu
    )

    compile_kwargs: dict = {}
    if xi_init_values is not None:
        compile_kwargs["initial_points"] = {"xi_init": xi_init_values}
        compile_kwargs["jitter_rvs"] = {rv for rv in model.free_RVs if rv.name != "xi_init"}
        print(
            f"  PCA-informed initvals: {len(xi_init_values)} params, "
            f"range [{xi_init_values.min():.2f}, {xi_init_values.max():.2f}]"
        )
        jittered = [rv.name for rv in compile_kwargs["jitter_rvs"]]
        print(f"  jitter_rvs: {jittered} (xi_init excluded)")

    print("  Compiling model with nutpie...")
    compiled = nutpie.compile_pymc_model(model, **compile_kwargs)

    print(f"  Sampling: {n_samples} draws, {n_tune} tune, {n_chains} chains")
    print(f"  seed={RANDOM_SEED}, sampler=nutpie (Rust NUTS)")

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

    print(f"  Sampling complete in {sampling_time:.1f}s")
    return idata, sampling_time


# ── Sign Correction ──────────────────────────────────────────────────────────


def fix_period_sign_flips(
    idata: az.InferenceData,
    data: dict,
    all_static_irt: dict[int, pl.DataFrame],
    roster: pl.DataFrame,
) -> tuple[az.InferenceData, list[dict]]:
    """Detect and correct per-period sign flips using static IRT as reference.

    Positive beta alone is insufficient for sign identification when the random
    walk chain is broken (e.g., missing biennium data creating 0 bridge
    legislators).  This post-hoc step compares each period's dynamic xi with
    the corresponding static IRT and negates xi if r < 0.

    Pattern follows ``fix_joint_sign_convention()`` in hierarchical IRT
    (ADR-0042).

    Returns:
        (corrected idata, list of correction records).
        Each record: ``{label, time_idx, r_before, r_after, n_matched,
        reference_legs}`` where ``reference_legs`` is a list of
        ``(name, dynamic_xi, static_xi)`` for the 3 strongest matches.
    """
    xi_post = idata.posterior["xi"].values  # (chain, draw, time, legislator)
    n_time = xi_post.shape[2]
    leg_names = data["leg_names"]
    leg_periods = data["leg_periods"]
    labels = BIENNIUM_LABELS[:n_time]
    corrections: list[dict] = []

    for t in range(n_time):
        if t not in all_static_irt:
            continue

        static = all_static_irt[t]
        if static is None or static.height == 0:
            continue

        # Build name_norm on static if needed
        xi_col = "xi_mean" if "xi_mean" in static.columns else "mean"
        if "name_norm" not in static.columns:
            if "full_name" not in static.columns:
                continue
            static = static.with_columns(
                pl.col("full_name")
                .map_elements(normalize_name, return_dtype=pl.Utf8)
                .alias("name_norm")
            )

        static_map = dict(zip(static["name_norm"].to_list(), static[xi_col].to_list()))

        # Collect matched pairs (dynamic mean vs static) for served legislators
        dyn_vals: list[float] = []
        stat_vals: list[float] = []
        match_names: list[str] = []
        for gidx, nn in enumerate(leg_names):
            if t not in leg_periods[gidx]:
                continue
            if nn not in static_map:
                continue
            dyn_mean = float(np.mean(xi_post[:, :, t, gidx]))
            dyn_vals.append(dyn_mean)
            stat_vals.append(static_map[nn])
            match_names.append(nn)

        if len(dyn_vals) < 5:
            continue

        r_before, _ = stats.pearsonr(dyn_vals, stat_vals)

        if r_before < 0:
            # Negate xi for this period across all chains and draws
            xi_post[:, :, t, :] *= -1

            r_after, _ = stats.pearsonr([-v for v in dyn_vals], stat_vals)

            # Pick 3 highest-|xi| reference legislators for transparency
            abs_xi = [abs(d) for d in dyn_vals]
            top_idxs = sorted(range(len(abs_xi)), key=lambda i: abs_xi[i], reverse=True)[:3]

            # Look up full names from roster
            ref_legs = []
            for idx in top_idxs:
                nn = match_names[idx]
                roster_row = roster.filter(pl.col("name_norm") == nn)
                full = roster_row["full_name"][0] if roster_row.height > 0 else nn
                ref_legs.append((full, dyn_vals[idx], stat_vals[idx]))

            corrections.append(
                {
                    "label": labels[t],
                    "time_idx": t,
                    "r_before": r_before,
                    "r_after": r_after,
                    "n_matched": len(dyn_vals),
                    "reference_legs": ref_legs,
                }
            )

            print(
                f"    SIGN FLIP CORRECTED: {labels[t]} "
                f"(r={r_before:.3f} -> {r_after:.3f}, n={len(dyn_vals)})"
            )
            for full, dyn, stat in ref_legs:
                print(f"      {full}: dynamic={dyn:+.2f}, static={stat:+.2f}")

    # Write corrected posterior back
    idata.posterior["xi"].values[:] = xi_post

    if not corrections:
        print("    No sign corrections needed.")

    return idata, corrections


# ── Post-Processing ──────────────────────────────────────────────────────────


def extract_dynamic_ideal_points(
    idata: az.InferenceData,
    data: dict,
    roster: pl.DataFrame,
) -> pl.DataFrame:
    """Extract posterior summaries for dynamic ideal points.

    Returns DataFrame with columns: global_idx, name_norm, full_name, party,
    time_period, biennium_label, xi_mean, xi_sd, xi_hdi_2.5, xi_hdi_97.5, served.
    """
    xi_post = idata.posterior["xi"]  # (chain, draw, time, legislator)
    n_time = xi_post.shape[2]
    n_leg = xi_post.shape[3]

    leg_names = data["leg_names"]
    party_idx = data["party_idx"]
    party_names = data["party_names"]
    leg_periods = data["leg_periods"]
    labels = BIENNIUM_LABELS[:n_time]

    rows: list[dict] = []
    for t in range(n_time):
        for gidx in range(n_leg):
            samples = xi_post[:, :, t, gidx].values.flatten()
            mean = float(np.mean(samples))
            sd = float(np.std(samples))
            hdi = az.hdi(samples, hdi_prob=0.95)

            # Look up full name from roster
            roster_row = roster.filter(pl.col("global_idx") == gidx)
            full_name = roster_row["full_name"][0] if roster_row.height > 0 else leg_names[gidx]

            rows.append(
                {
                    "global_idx": gidx,
                    "name_norm": leg_names[gidx],
                    "full_name": full_name,
                    "party": party_names[int(party_idx[gidx])],
                    "time_period": t,
                    "biennium_label": labels[t],
                    "xi_mean": mean,
                    "xi_sd": sd,
                    "xi_hdi_2.5": float(hdi[0]),
                    "xi_hdi_97.5": float(hdi[1]),
                    "served": t in leg_periods[gidx],
                }
            )

    return pl.DataFrame(rows)


def decompose_polarization(
    trajectories: pl.DataFrame,
) -> pl.DataFrame:
    """Decompose polarization change into conversion and replacement.

    For each adjacent biennium pair, per party:
      - total_shift: change in party mean ideal point
      - conversion: movement of returning members (served both periods)
      - replacement: difference due to new members replacing departing ones

    Total = conversion + replacement (approximately, since it's a mean decomposition).
    """
    labels = trajectories["biennium_label"].unique().sort().to_list()
    parties = trajectories["party"].unique().sort().to_list()

    rows: list[dict] = []
    for i in range(len(labels) - 1):
        label_a = labels[i]
        label_b = labels[i + 1]

        for party in parties:
            # Period A served members
            a_served = trajectories.filter(
                (pl.col("biennium_label") == label_a)
                & (pl.col("party") == party)
                & pl.col("served")
            )
            # Period B served members
            b_served = trajectories.filter(
                (pl.col("biennium_label") == label_b)
                & (pl.col("party") == party)
                & pl.col("served")
            )

            if a_served.height == 0 or b_served.height == 0:
                continue

            mean_a = a_served["xi_mean"].mean()
            mean_b = b_served["xi_mean"].mean()
            total_shift = mean_b - mean_a

            # Returning members (served in both periods)
            a_names = set(a_served["name_norm"].to_list())
            b_names = set(b_served["name_norm"].to_list())
            returning = a_names & b_names

            if len(returning) > 0:
                ret_a_mean = a_served.filter(pl.col("name_norm").is_in(returning))["xi_mean"].mean()
                ret_b_mean = b_served.filter(pl.col("name_norm").is_in(returning))["xi_mean"].mean()
                conversion = ret_b_mean - ret_a_mean
            else:
                conversion = 0.0

            replacement = total_shift - conversion

            rows.append(
                {
                    "pair": f"{label_a}→{label_b}",
                    "party": party,
                    "total_shift": total_shift,
                    "conversion": conversion,
                    "replacement": replacement,
                    "n_returning": len(returning),
                    "n_departing": len(a_names - b_names),
                    "n_new": len(b_names - a_names),
                }
            )

    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "pair": pl.Utf8,
                "party": pl.Utf8,
                "total_shift": pl.Float64,
                "conversion": pl.Float64,
                "replacement": pl.Float64,
                "n_returning": pl.Int64,
                "n_departing": pl.Int64,
                "n_new": pl.Int64,
            }
        )
    )


def correlate_with_static(
    trajectories: pl.DataFrame,
    all_static_irt: dict[int, pl.DataFrame],
) -> pl.DataFrame:
    """Correlate dynamic ideal points with per-biennium static IRT estimates.

    Args:
        trajectories: Dynamic ideal point trajectories.
        all_static_irt: Mapping from time_idx to static IRT DataFrames
            (must have ``legislator_slug`` and ``xi_mean``).

    Returns:
        DataFrame with per-biennium Pearson r and Spearman rho.
    """
    labels = sorted(trajectories["biennium_label"].unique().to_list())
    rows: list[dict] = []

    for t, label in enumerate(labels):
        if t not in all_static_irt:
            continue

        static = all_static_irt[t]
        if static is None or static.height == 0:
            continue

        # Get dynamic points for this period (served only)
        dyn = trajectories.filter((pl.col("biennium_label") == label) & pl.col("served")).select(
            "name_norm", "xi_mean"
        )

        # Build name_norm for static data
        xi_col = "xi_mean" if "xi_mean" in static.columns else "mean"

        # Merge on name_norm if available, otherwise skip
        if "name_norm" not in static.columns and "full_name" not in static.columns:
            continue

        if "name_norm" not in static.columns:
            static = static.with_columns(
                pl.col("full_name")
                .map_elements(normalize_name, return_dtype=pl.Utf8)
                .alias("name_norm")
            )

        merged = dyn.join(
            static.select("name_norm", pl.col(xi_col).alias("xi_static")),
            on="name_norm",
            how="inner",
        )

        if merged.height < 5:
            continue

        x = merged["xi_mean"].to_numpy()
        y = merged["xi_static"].to_numpy()
        r, _ = stats.pearsonr(x, y)
        rho, _ = stats.spearmanr(x, y)

        rows.append(
            {
                "biennium": label,
                "n_matched": merged.height,
                "pearson_r": r,
                "spearman_rho": rho,
            }
        )

    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "biennium": pl.Utf8,
                "n_matched": pl.Int64,
                "pearson_r": pl.Float64,
                "spearman_rho": pl.Float64,
            }
        )
    )


def identify_top_movers(
    trajectories: pl.DataFrame,
    n_top: int = TOP_MOVERS_N,
) -> pl.DataFrame:
    """Identify legislators with the largest ideological movement.

    Total movement = sum of |xi[t] - xi[t-1]| across served periods.
    Net movement = last served xi - first served xi.
    """
    served = trajectories.filter(pl.col("served"))
    names = served["name_norm"].unique().to_list()

    rows: list[dict] = []
    for name in names:
        leg_data = served.filter(pl.col("name_norm") == name).sort("time_period")
        if leg_data.height < 2:
            continue

        xi_vals = leg_data["xi_mean"].to_list()
        total_movement = sum(abs(xi_vals[i + 1] - xi_vals[i]) for i in range(len(xi_vals) - 1))
        net_movement = xi_vals[-1] - xi_vals[0]

        rows.append(
            {
                "name_norm": name,
                "full_name": leg_data["full_name"][0],
                "party": leg_data["party"][0],
                "n_periods": leg_data.height,
                "first_period": leg_data["biennium_label"][0],
                "last_period": leg_data["biennium_label"][-1],
                "total_movement": total_movement,
                "net_movement": net_movement,
                "direction": "rightward" if net_movement > 0 else "leftward",
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "name_norm": pl.Utf8,
                "full_name": pl.Utf8,
                "party": pl.Utf8,
                "n_periods": pl.Int64,
                "first_period": pl.Utf8,
                "last_period": pl.Utf8,
                "total_movement": pl.Float64,
                "net_movement": pl.Float64,
                "direction": pl.Utf8,
            }
        )

    result = pl.DataFrame(rows).sort("total_movement", descending=True)
    return result.head(n_top)


def extract_tau_posterior(
    idata: az.InferenceData,
    party_names: list[str],
) -> pl.DataFrame:
    """Extract posterior summary for evolution variance tau."""
    tau_post = idata.posterior["tau"]

    rows: list[dict] = []
    if len(tau_post.shape) == 2:
        # Global tau (single scalar)
        samples = tau_post.values.flatten()
        hdi = az.hdi(samples, hdi_prob=0.95)
        rows.append(
            {
                "party": "Global",
                "tau_mean": float(np.mean(samples)),
                "tau_sd": float(np.std(samples)),
                "tau_hdi_2.5": float(hdi[0]),
                "tau_hdi_97.5": float(hdi[1]),
            }
        )
    else:
        # Per-party tau
        for i, party in enumerate(party_names):
            samples = tau_post[:, :, i].values.flatten()
            hdi = az.hdi(samples, hdi_prob=0.95)
            rows.append(
                {
                    "party": party,
                    "tau_mean": float(np.mean(samples)),
                    "tau_sd": float(np.std(samples)),
                    "tau_hdi_2.5": float(hdi[0]),
                    "tau_hdi_97.5": float(hdi[1]),
                }
            )

    return pl.DataFrame(rows)


def check_convergence(
    idata: az.InferenceData,
    params: list[str] | None = None,
) -> dict:
    """Check convergence diagnostics for the dynamic IRT model.

    Returns dict with: rhat_summary, ess_summary, n_divergences, passed.
    """
    if params is None:
        params = ["xi", "tau", "alpha", "beta"]

    summary = az.summary(idata, var_names=params, kind="diagnostics")

    rhat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    n_divergences = int(idata.sample_stats.get("diverging", np.zeros(1)).sum())

    passed = (
        rhat_max <= RHAT_THRESHOLD and ess_min >= ESS_THRESHOLD and n_divergences <= MAX_DIVERGENCES
    )

    return {
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_min,
        "ess_tail_min": float(summary["ess_tail"].min()),
        "n_divergences": n_divergences,
        "passed": passed,
        "rhat_threshold": RHAT_THRESHOLD,
        "ess_threshold": ESS_THRESHOLD,
        "n_params": len(summary),
    }


# ── Plotting ─────────────────────────────────────────────────────────────────


def save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure and close to free memory."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_polarization_trend(
    trajectories: pl.DataFrame,
    out_path: Path,
) -> None:
    """Plot party mean ideal points across bienniums with 95% bands."""
    served = trajectories.filter(pl.col("served"))
    parties = sorted(served["party"].unique().to_list())
    labels = sorted(served["biennium_label"].unique().to_list())
    n_time = len(labels)

    fig, ax = plt.subplots(figsize=(12, 6))

    for party in parties:
        color = PARTY_COLORS_DYNAMIC.get(party, "#999999")
        means = []
        lowers = []
        uppers = []
        for label in labels:
            subset = served.filter((pl.col("party") == party) & (pl.col("biennium_label") == label))
            if subset.height > 0:
                means.append(subset["xi_mean"].mean())
                lowers.append(subset["xi_hdi_2.5"].mean())
                uppers.append(subset["xi_hdi_97.5"].mean())
            else:
                means.append(np.nan)
                lowers.append(np.nan)
                uppers.append(np.nan)

        x = range(n_time)
        ax.plot(x, means, color=color, linewidth=2, label=party, marker="o")
        ax.fill_between(x, lowers, uppers, color=color, alpha=0.15)

    ax.set_xticks(range(n_time))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Biennium")
    ax.set_ylabel("Mean Ideal Point (xi)")
    ax.set_title("Polarization Trend — Party Mean Ideal Points Across Bienniums")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_individual_trajectories(
    trajectories: pl.DataFrame,
    top_movers: pl.DataFrame,
    out_path: Path,
) -> None:
    """Spaghetti plot of individual trajectories, top movers highlighted."""
    served = trajectories.filter(pl.col("served"))
    labels = sorted(served["biennium_label"].unique().to_list())
    n_time = len(labels)
    all_names = served["name_norm"].unique().to_list()
    top_names = set(top_movers["name_norm"].to_list()) if top_movers.height > 0 else set()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot all trajectories faintly
    for name in all_names:
        leg_data = served.filter(pl.col("name_norm") == name).sort("time_period")
        if leg_data.height < 2:
            continue
        party = leg_data["party"][0]
        color = PARTY_COLORS_DYNAMIC.get(party, "#999999")
        x = [labels.index(lb) for lb in leg_data["biennium_label"].to_list()]
        y = leg_data["xi_mean"].to_list()

        if name in top_names:
            ax.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
            # Label at endpoint
            ax.annotate(
                leg_data["full_name"][-1],
                (x[-1], y[-1]),
                fontsize=7,
                color=color,
                ha="left",
                va="center",
                xytext=(5, 0),
                textcoords="offset points",
            )
        else:
            ax.plot(x, y, color=color, linewidth=0.3, alpha=0.15)

    ax.set_xticks(range(n_time))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Biennium")
    ax.set_ylabel("Ideal Point (xi)")
    ax.set_title("Individual Ideal Point Trajectories (top movers highlighted)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_top_movers_bar(
    top_movers: pl.DataFrame,
    out_path: Path,
) -> None:
    """Bar chart of top movers by net shift, colored by direction."""
    if top_movers.height == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by net_movement
    sorted_df = top_movers.sort("net_movement")
    names = sorted_df["full_name"].to_list()
    values = sorted_df["net_movement"].to_list()
    colors = ["#E81B23" if v > 0 else "#0015BC" for v in values]

    y_pos = range(len(names))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Net Ideological Shift")
    ax.set_title("Top Movers — Net Ideological Shift\n(red = rightward, blue = leftward)")
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_conversion_replacement(
    decomposition: pl.DataFrame,
    out_path: Path,
) -> None:
    """Stacked bar plot of conversion vs replacement per party per transition."""
    if decomposition.height == 0:
        return

    parties = sorted(decomposition["party"].unique().to_list())

    fig, axes = plt.subplots(1, len(parties), figsize=(7 * len(parties), 6), squeeze=False)

    for idx, party in enumerate(parties):
        ax = axes[0, idx]
        color = PARTY_COLORS_DYNAMIC.get(party, "#999999")
        party_data = decomposition.filter(pl.col("party") == party).sort("pair")

        pair_labels = party_data["pair"].to_list()
        conv = party_data["conversion"].to_list()
        repl = party_data["replacement"].to_list()

        x = range(len(pair_labels))
        ax.bar(x, conv, label="Conversion", color=color, alpha=0.7)
        ax.bar(x, repl, bottom=conv, label="Replacement", color=color, alpha=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Shift in Party Mean")
        ax.set_title(f"{party}")
        ax.legend()
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    fig.suptitle("Polarization Decomposition — Conversion vs. Replacement", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_tau_posterior(
    idata: az.InferenceData,
    party_names: list[str],
    out_path: Path,
) -> None:
    """KDE plot of tau (evolution SD) posterior per party."""
    tau_post = idata.posterior["tau"]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(tau_post.shape) == 2:
        # Global tau
        samples = tau_post.values.flatten()
        ax.hist(samples, bins=50, density=True, alpha=0.7, color="#666666", label="Global")
    else:
        for i, party in enumerate(party_names):
            samples = tau_post[:, :, i].values.flatten()
            color = PARTY_COLORS_DYNAMIC.get(party, "#999999")
            ax.hist(samples, bins=50, density=True, alpha=0.5, color=color, label=party)

    ax.set_xlabel("tau (evolution SD)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Distribution of Evolution Variance (tau)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_static_correlation(
    correlation_df: pl.DataFrame,
    out_path: Path,
) -> None:
    """Bar chart of per-biennium correlation between dynamic and static IRT."""
    if correlation_df.height == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = correlation_df["biennium"].to_list()
    r_vals = correlation_df["pearson_r"].to_list()
    x = range(len(labels))
    ax.bar(x, r_vals, color="#4a86c8", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("Dynamic vs. Static IRT Correlation by Biennium")
    ax.axhline(0.9, color="green", linestyle="--", linewidth=1, label="r = 0.90 target")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_bridge_coverage(
    bridge_df: pl.DataFrame,
    out_path: Path,
) -> None:
    """Heatmap of bridge coverage between all biennium pairs."""
    labels_a = bridge_df["label_a"].unique().sort().to_list()
    labels_b = bridge_df["label_b"].unique().sort().to_list()
    all_labels = sorted(set(labels_a + labels_b))
    n = len(all_labels)
    label_to_idx = {lb: i for i, lb in enumerate(all_labels)}

    matrix = np.full((n, n), np.nan)
    for row in bridge_df.iter_rows(named=True):
        i = label_to_idx[row["label_a"]]
        j = label_to_idx[row["label_b"]]
        matrix[i, j] = row["shared_count"]
        matrix[j, i] = row["shared_count"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(all_labels)
    ax.set_title("Bridge Coverage — Shared Legislators Between Bienniums")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Shared Legislators")
    fig.tight_layout()
    save_fig(fig, out_path)


# ── emIRT Interface ──────────────────────────────────────────────────────────


def run_emirt(
    data: dict,
    chamber: str,
    output_dir: Path,
) -> pl.DataFrame | None:
    """Run emIRT::dynIRT via R subprocess (optional exploration tier).

    Returns emIRT point estimates as a DataFrame, or None if R/emIRT unavailable.
    """
    r_script = Path(__file__).parent / "dynamic_irt_emirt.R"
    if not r_script.exists():
        print("  emIRT R script not found, skipping")
        return None

    if shutil.which("Rscript") is None:
        print("  Rscript not found, skipping emIRT")
        return None

    # Write input CSV
    input_csv = output_dir / "emirt_input.csv"
    prepare_emirt_csv(data, input_csv)

    # Output path
    result_csv = output_dir / f"emirt_{chamber}_results.csv"

    try:
        result = subprocess.run(
            ["Rscript", str(r_script), str(input_csv), str(result_csv), chamber],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            print(f"  emIRT failed: {result.stderr[:500]}")
            return None
        print(f"  emIRT completed for {chamber}")
        return load_emirt_results(result_csv)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  emIRT error: {e}")
        return None


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_biennium_data(
    session_str: str,
    run_id: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame | None, dict | None, pl.DataFrame]:
    """Load upstream data for a single biennium.

    Returns (vote_matrix_house, vote_matrix_senate, pca_scores, legislators).
    """
    ks = KSSession.from_session_string(session_str)
    results_root = ks.results_dir

    # Load EDA matrices
    eda_dir = resolve_upstream_dir("01_eda", results_root, run_id)
    house_matrix, senate_matrix, _ = load_eda_matrices(eda_dir)

    # Load PCA scores (optional)
    try:
        pca_dir = resolve_upstream_dir("02_pca", results_root, run_id)
        pca_house, pca_senate = load_pca_scores(pca_dir)
    except (FileNotFoundError, OSError):  # fmt: skip
        pca_house, pca_senate = None, None

    # Load legislator metadata
    _, legislators = load_metadata(ks.data_dir)

    return house_matrix, senate_matrix, {"house": pca_house, "senate": pca_senate}, legislators


# ── CLI and Orchestration ────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Dynamic ideal point estimation (Phase 16)",
    )
    parser.add_argument(
        "--chambers",
        default="both",
        choices=["house", "senate", "both"],
        help="Which chamber(s) to analyze (default: both)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"MCMC posterior draws per chain (default: {DEFAULT_N_SAMPLES})",
    )
    parser.add_argument(
        "--n-tune",
        type=int,
        default=DEFAULT_N_TUNE,
        help=f"MCMC tuning steps (default: {DEFAULT_N_TUNE})",
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=DEFAULT_N_CHAINS,
        help=f"MCMC chains (default: {DEFAULT_N_CHAINS})",
    )
    parser.add_argument(
        "--evolution",
        default="per_party",
        choices=["global", "per_party"],
        help="Evolution variance structure (default: per_party)",
    )
    parser.add_argument(
        "--skip-emirt",
        action="store_true",
        help="Skip emIRT exploration tier",
    )
    parser.add_argument(
        "--bienniums",
        default=None,
        help="Comma-separated bienniums to include (default: all 8)",
    )
    parser.add_argument(
        "--tau-sigma",
        type=float,
        default=None,
        help="Explicit tau prior sigma (overrides adaptive logic)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID for upstream data resolution",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")


def main() -> None:
    """Run dynamic ideal point estimation pipeline."""
    args = parse_args()

    # Determine which bienniums to include
    if args.bienniums:
        session_list = [s.strip() for s in args.bienniums.split(",")]
    else:
        session_list = list(BIENNIUM_SESSIONS)

    n_bienniums = len(session_list)
    label_list = [SESSION_TO_LABEL[s] for s in session_list]

    chambers = ["house", "senate"] if args.chambers == "both" else [args.chambers]

    with RunContext(
        session="cross-session",
        analysis_name="dynamic_irt",
        params=vars(args),
        primer=DYNAMIC_IRT_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print_header("Dynamic Ideal Point Estimation (Phase 16)")
        print(f"  Bienniums: {', '.join(label_list)} ({n_bienniums} sessions)")
        print(f"  Chambers:  {', '.join(chambers)}")
        print(f"  MCMC:      {args.n_samples} samples, {args.n_tune} tune, {args.n_chains} chains")
        print(f"  Evolution: {args.evolution}")

        # ── Step 1: Load all biennium data ──
        print_header("Loading Biennium Data")

        all_house_matrices: dict[int, pl.DataFrame] = {}
        all_senate_matrices: dict[int, pl.DataFrame] = {}
        all_pca: dict[int, dict] = {}
        all_legislators: dict[int, pl.DataFrame] = {}

        for t, session_str in enumerate(session_list):
            label = SESSION_TO_LABEL[session_str]
            print(f"\n  [{label}] Loading {session_str}...")
            try:
                house_m, senate_m, pca, legs = load_biennium_data(session_str, args.run_id)
                all_house_matrices[t] = house_m
                all_senate_matrices[t] = senate_m
                all_pca[t] = pca
                all_legislators[t] = legs
                print(f"    House: {house_m.height} legislators x {house_m.width - 1} votes")
                print(f"    Senate: {senate_m.height} legislators x {senate_m.width - 1} votes")
            except (FileNotFoundError, OSError) as e:
                print(f"    WARNING: Could not load {session_str}: {e}")
                continue

        if len(all_house_matrices) < 2:
            print("\n  ERROR: Need at least 2 bienniums for dynamic analysis")
            return

        # ── Per-chamber analysis ──
        all_results: dict = {"chambers": chambers}

        for chamber in chambers:
            chamber_cap = chamber.capitalize()
            print_header(f"{chamber_cap} Dynamic IRT")

            # Select matrices
            matrices = all_house_matrices if chamber == "house" else all_senate_matrices

            # ── Step 2: Prepare IRT data per biennium ──
            print("\n  Preparing IRT data per biennium...")
            all_irt_data: dict[int, dict] = {}
            for t in sorted(matrices.keys()):
                label = label_list[t] if t < len(label_list) else f"t{t}"
                print(f"    [{label}]", end=" ")
                data_t = prepare_irt_data(matrices[t], chamber_cap)
                all_irt_data[t] = data_t

            # ── Step 3: Build global roster ──
            print("\n  Building global roster...")
            roster, name_to_global = build_global_roster(all_legislators, chamber_cap)
            print(f"    {roster.height} unique legislators")
            roster.write_parquet(ctx.data_dir / f"roster_{chamber}.parquet")

            # ── Step 4: Stack bienniums ──
            print("\n  Stacking bienniums into long format...")
            stacked = stack_bienniums(chamber_cap, all_irt_data, name_to_global, all_legislators)
            print(f"    {stacked['n_legislators']} legislators, {stacked['n_bills']} bills")
            print(f"    {stacked['n_obs']:,} observations, {stacked['n_time']} time periods")
            print(f"    Parties: {stacked['party_names']}")

            # ── Step 5: Bridge coverage ──
            print("\n  Computing bridge coverage...")
            bridge_full = compute_bridge_coverage(
                stacked["leg_periods"], stacked["n_time"], label_list
            )
            bridge_adj = compute_adjacent_bridges(
                stacked["leg_periods"], stacked["n_time"], label_list
            )
            bridge_full.write_parquet(ctx.data_dir / f"bridge_coverage_{chamber}.parquet")
            bridge_adj.write_parquet(ctx.data_dir / f"bridge_adjacent_{chamber}.parquet")
            print(bridge_adj.to_pandas().to_string(index=False))

            # Check bridge sufficiency
            insufficient = bridge_adj.filter(~pl.col("sufficient"))
            if insufficient.height > 0:
                for row in insufficient.iter_rows(named=True):
                    print(
                        f"    WARNING: {row['pair']} has only {row['shared_count']} shared "
                        f"legislators (< {MIN_BRIDGE_OVERLAP})"
                    )

            # ── Step 6: PCA-informed initialization ──
            xi_init_values = None
            first_t = min(all_pca.keys())
            pca_key = "house" if chamber == "house" else "senate"
            if first_t in all_pca and all_pca[first_t].get(pca_key) is not None:
                pca_scores = all_pca[first_t][pca_key]
                if pca_scores is not None and "PC1" in pca_scores.columns:
                    print("\n  Building PCA-informed initial values...")
                    # Map PCA scores to global indices
                    init_vals = np.zeros(stacked["n_legislators"])
                    slug_col = (
                        "legislator_slug" if "legislator_slug" in pca_scores.columns else "slug"
                    )
                    for row in pca_scores.iter_rows(named=True):
                        slug = row.get(slug_col, "")
                        # Find this slug in the first biennium's legislator data
                        leg_df = all_legislators[first_t]
                        if "chamber" in leg_df.columns:
                            leg_df = leg_df.filter(pl.col("chamber") == chamber_cap)
                        for lr in leg_df.iter_rows(named=True):
                            lr_slug_col = (
                                "legislator_slug" if "legislator_slug" in leg_df.columns else "slug"
                            )
                            if lr.get(lr_slug_col, "") == slug:
                                nn = normalize_name(lr.get("full_name", ""))
                                if nn in name_to_global:
                                    gidx = name_to_global[nn]
                                    init_vals[gidx] = row["PC1"]
                                break

                    # Standardize
                    nz = init_vals[init_vals != 0]
                    if len(nz) > 0:
                        init_vals = (init_vals - nz.mean()) / max(nz.std(), 1e-6)
                    xi_init_values = init_vals
                    print(f"    Initialized {np.count_nonzero(init_vals)} of {len(init_vals)}")

            # ── Step 6b: Load static IRT for informative prior + sign correction ──
            print("\n  Loading static IRT data...")
            all_static_irt: dict[int, pl.DataFrame] = {}
            for t, session_str in enumerate(session_list):
                try:
                    ks_t = KSSession.from_session_string(session_str)
                    irt_dir = resolve_upstream_dir("04_irt", ks_t.results_dir, args.run_id)
                    irt_path = irt_dir / "data" / f"ideal_points_{chamber}.parquet"
                    if irt_path.exists():
                        static_df = pl.read_parquet(irt_path)
                        if "full_name" in static_df.columns:
                            static_df = static_df.with_columns(
                                pl.col("full_name")
                                .map_elements(normalize_name, return_dtype=pl.Utf8)
                                .alias("name_norm")
                            )
                        all_static_irt[t] = static_df
                except (FileNotFoundError, OSError):  # fmt: skip
                    pass
            print(f"    Loaded static IRT for {len(all_static_irt)} bienniums")

            # Build informative xi_init prior from static IRT (ADR-0070)
            xi_init_mu = None
            if all_static_irt:
                print("\n  Building informative xi_init prior from static IRT...")
                xi_init_mu_arr = np.zeros(stacked["n_legislators"])
                n_mapped = 0
                for t_idx, static_df in all_static_irt.items():
                    xi_col = "xi_mean" if "xi_mean" in static_df.columns else "mean"
                    if "name_norm" not in static_df.columns:
                        continue
                    for row in static_df.iter_rows(named=True):
                        nn = row.get("name_norm", "")
                        if nn in name_to_global:
                            gidx = name_to_global[nn]
                            if xi_init_mu_arr[gidx] == 0.0:
                                xi_init_mu_arr[gidx] = row[xi_col]
                                n_mapped += 1
                # Standardize to unit scale
                nz = xi_init_mu_arr[xi_init_mu_arr != 0]
                if len(nz) > 5:
                    xi_init_mu_arr = (xi_init_mu_arr - nz.mean()) / max(nz.std(), 1e-6)
                    xi_init_mu = xi_init_mu_arr
                    print(f"    Mapped {n_mapped} of {stacked['n_legislators']} legislators")
                else:
                    print("    Too few matches for informative prior, using uninformative")

            # ── Step 7: emIRT exploration (optional) ──
            emirt_results = None
            if not args.skip_emirt:
                print("\n  Running emIRT exploration tier...")
                emirt_results = run_emirt(stacked, chamber, ctx.data_dir)

            # ── Step 8: PyMC Dynamic IRT ──
            print_header(f"{chamber_cap} PyMC Sampling")
            print(
                f"  Model params: ~{stacked['n_legislators'] * stacked['n_time']} xi + "
                f"{stacked['n_bills']} alpha + {stacked['n_bills']} beta + "
                f"{len(stacked['party_names'])} tau"
            )

            idata, sampling_time = sample_dynamic_irt(
                stacked,
                n_samples=args.n_samples,
                n_tune=args.n_tune,
                n_chains=args.n_chains,
                evolution_structure=args.evolution,
                xi_init_values=xi_init_values,
                tau_sigma=args.tau_sigma,
                xi_init_mu=xi_init_mu,
            )

            # ── Step 9: Convergence ──
            print("\n  Checking convergence...")
            convergence = check_convergence(idata)
            print(f"    R-hat max: {convergence['rhat_max']:.4f} (threshold: {RHAT_THRESHOLD})")
            print(
                f"    ESS bulk min: {convergence['ess_bulk_min']:.0f} (threshold: {ESS_THRESHOLD})"
            )
            print(f"    Divergences: {convergence['n_divergences']}")
            print(f"    PASSED: {convergence['passed']}")

            # ── Step 10: Sign correction (uses static IRT from Step 6b) ──
            print("\n  Checking for per-period sign flips...")
            idata, sign_corrections = fix_period_sign_flips(idata, stacked, all_static_irt, roster)

            # Save InferenceData (after correction)
            idata.to_netcdf(ctx.data_dir / f"dynamic_irt_{chamber}.nc")

            # ── Step 12: Post-processing ──
            print("\n  Extracting dynamic ideal points...")
            trajectories = extract_dynamic_ideal_points(idata, stacked, roster)
            trajectories.write_parquet(ctx.data_dir / f"trajectories_{chamber}.parquet")
            ctx.export_csv(
                trajectories,
                f"trajectories_{chamber}.csv",
                f"Dynamic ideal point trajectories for {chamber.title()}",
            )

            print("  Decomposing polarization...")
            decomposition = decompose_polarization(trajectories)
            decomposition.write_parquet(ctx.data_dir / f"decomposition_{chamber}.parquet")
            ctx.export_csv(
                decomposition,
                f"decomposition_{chamber}.csv",
                f"Polarization decomposition for {chamber.title()}",
            )

            print("  Identifying top movers...")
            top_movers = identify_top_movers(trajectories)
            top_movers.write_parquet(ctx.data_dir / f"top_movers_{chamber}.parquet")
            ctx.export_csv(
                top_movers,
                f"top_movers_{chamber}.csv",
                f"Legislators with largest ideological movement in {chamber.title()}",
            )
            if top_movers.height > 0:
                print(
                    f"    Top mover: {top_movers['full_name'][0]} "
                    f"(total={top_movers['total_movement'][0]:.3f})"
                )

            print("  Extracting tau posterior...")
            tau_df = extract_tau_posterior(idata, stacked["party_names"])
            tau_df.write_parquet(ctx.data_dir / f"tau_posterior_{chamber}.parquet")
            for row in tau_df.iter_rows(named=True):
                print(
                    f"    {row['party']}: tau = {row['tau_mean']:.4f} "
                    f"[{row['tau_hdi_2.5']:.4f}, {row['tau_hdi_97.5']:.4f}]"
                )

            # ── Step 13: Correlation with static IRT ──
            print("\n  Correlating with static IRT...")
            correlation_df = correlate_with_static(trajectories, all_static_irt)
            if correlation_df.height > 0:
                # Add sign_corrected column
                corrected_labels = {c["label"] for c in sign_corrections}
                correlation_df = correlation_df.with_columns(
                    pl.col("biennium")
                    .map_elements(lambda b: b in corrected_labels, return_dtype=pl.Boolean)
                    .alias("sign_corrected")
                )
                correlation_df.write_parquet(ctx.data_dir / f"static_correlation_{chamber}.parquet")
                for row in correlation_df.iter_rows(named=True):
                    flag = " [corrected]" if row.get("sign_corrected") else ""
                    print(
                        f"    {row['biennium']}: r = {row['pearson_r']:.3f}, "
                        f"rho = {row['spearman_rho']:.3f}{flag}"
                    )

            # ── Step 14: Plots ──
            print("\n  Generating plots...")
            plot_polarization_trend(
                trajectories, ctx.plots_dir / f"polarization_trend_{chamber}.png"
            )
            plot_individual_trajectories(
                trajectories,
                top_movers,
                ctx.plots_dir / f"trajectories_{chamber}.png",
            )
            plot_top_movers_bar(top_movers, ctx.plots_dir / f"top_movers_bar_{chamber}.png")
            plot_conversion_replacement(
                decomposition, ctx.plots_dir / f"conversion_replacement_{chamber}.png"
            )
            plot_tau_posterior(
                idata,
                stacked["party_names"],
                ctx.plots_dir / f"tau_posterior_{chamber}.png",
            )
            if correlation_df.height > 0:
                plot_static_correlation(
                    correlation_df, ctx.plots_dir / f"static_correlation_{chamber}.png"
                )
            plot_bridge_coverage(bridge_full, ctx.plots_dir / f"bridge_coverage_{chamber}.png")

            # Collect results for report
            pca_init_range = (
                (float(np.min(xi_init_values)), float(np.max(xi_init_values)))
                if xi_init_values is not None
                else None
            )
            all_results[chamber] = {
                "stacked": stacked,
                "roster": roster,
                "trajectories": trajectories,
                "decomposition": decomposition,
                "top_movers": top_movers,
                "tau": tau_df,
                "correlation": correlation_df,
                "convergence": convergence,
                "bridge_full": bridge_full,
                "bridge_adj": bridge_adj,
                "emirt_results": emirt_results,
                "sampling_time": sampling_time,
                "idata": idata,
                "sign_corrections": sign_corrections,
                "pca_init_range": pca_init_range,
                "mcmc_params": {
                    "n_samples": args.n_samples,
                    "n_tune": args.n_tune,
                    "n_chains": args.n_chains,
                    "seed": RANDOM_SEED,
                },
            }

        # ── Step 13: Build HTML report ──
        print_header("Building Report")
        build_dynamic_irt_report(
            ctx.report,
            results=all_results,
            plots_dir=ctx.plots_dir,
            biennium_labels=label_list,
        )

        # Save summary JSON
        summary: dict = {}
        for chamber in chambers:
            if chamber in all_results:
                cr = all_results[chamber]
                summary[chamber] = {
                    "n_legislators": cr["stacked"]["n_legislators"],
                    "n_bills": cr["stacked"]["n_bills"],
                    "n_obs": cr["stacked"]["n_obs"],
                    "n_time": cr["stacked"]["n_time"],
                    "sampling_time_s": cr["sampling_time"],
                    "convergence": cr["convergence"],
                }
        with open(ctx.data_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
