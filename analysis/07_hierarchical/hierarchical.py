"""
Kansas Legislature — Hierarchical Bayesian IRT Ideal Point Estimation

Extends the flat IRT model (Phase 3) with partial pooling by party. Legislators
with sparse voting records are shrunk toward their party mean. Variance
decomposition quantifies how much party explains. Shrinkage comparison shows what
changes vs. the flat model.

Two models:
- Per-chamber (primary): 2-level model, run separately for House and Senate
- Joint cross-chamber (optional): 3-level model with both chambers, enabled with --run-joint

Uses method 16 from Analytic_Methods/16_BAY_hierarchical_legislator_model.md.

Usage:
  uv run python analysis/hierarchical.py [--session 2025-26] [--run-joint]
      [--n-samples 2000] [--n-tune 1500] [--n-chains 2]

Outputs (in results/<session>/hierarchical/<date>/):
  - data/:   Parquet files (ideal points, group params, variance decomp) + NetCDF
  - plots/:  PNG visualizations (party posteriors, ICC, shrinkage, forest, dispersion)
  - filtering_manifest.json, run_info.json, run_log.txt
  - hierarchical_report.html
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
    from analysis.hierarchical_report import build_hierarchical_report
except ModuleNotFoundError:
    from hierarchical_report import build_hierarchical_report  # type: ignore[no-redef]

try:
    from analysis.model_spec import JOINT_BETA, PRODUCTION_BETA, BetaPriorSpec
except ModuleNotFoundError:
    from model_spec import JOINT_BETA, PRODUCTION_BETA, BetaPriorSpec  # type: ignore[no-redef]

try:
    from analysis.irt_linking import compare_linking_methods
except ModuleNotFoundError:
    from irt_linking import compare_linking_methods  # type: ignore[no-redef]

try:
    from analysis.irt import (
        ESS_THRESHOLD,
        MAX_DIVERGENCES,
        PARTY_COLORS,
        RANDOM_SEED,
        RHAT_THRESHOLD,
        load_eda_matrices,
        load_metadata,
        load_pca_scores,
        plot_forest,
        prepare_irt_data,
    )
except ModuleNotFoundError:
    from irt import (  # type: ignore[no-redef]
        ESS_THRESHOLD,
        MAX_DIVERGENCES,
        PARTY_COLORS,
        RANDOM_SEED,
        RHAT_THRESHOLD,
        load_eda_matrices,
        load_metadata,
        load_pca_scores,
        plot_forest,
        prepare_irt_data,
    )

try:
    from analysis.phase_utils import print_header, save_fig
except ImportError:
    from phase_utils import print_header, save_fig  # type: ignore[no-redef]

try:
    from analysis.init_strategy import resolve_init_source
except ModuleNotFoundError:
    from init_strategy import resolve_init_source  # type: ignore[no-redef]

# ── Primer ───────────────────────────────────────────────────────────────────

HIERARCHICAL_PRIMER = """\
# Hierarchical Bayesian IRT Ideal Point Estimation

## Purpose

Extends the flat IRT model with **partial pooling by party**. Instead of
treating each legislator as an independent draw from Normal(0, 1), the
hierarchical model nests legislators within parties: each party has its own
mean ideal point, and individual legislators are drawn from their party's
distribution.

This matters because:
- Legislators with few votes get **shrunk toward their party mean**, producing
  more reliable estimates
- The model quantifies **how much party explains** (variance decomposition)
- Comparing with the flat IRT shows **who moved and by how much**

Covers analytic method 16 from `Analytic_Methods/`.

## Method

### 2-Level Hierarchical IRT (Per-Chamber)

```
mu_party_raw ~ Normal(0, 2)           -- party mean ideal points (2 parties)
mu_party     = sort(mu_party_raw)     -- ordering: D < R
sigma_within ~ HalfNormal(1)          -- within-party SD (per party)

xi_offset_i  ~ Normal(0, 1)           -- non-centered offset
xi_i         = mu_party[p_i] + sigma_within[p_i] * xi_offset_i

alpha_j ~ Normal(0, 5)                -- bill difficulty
beta_j  ~ Normal(0, 1)                -- bill discrimination

P(Yea) = logit^-1(beta_j * xi_i - alpha_j)
```

**Identification:** Ordering constraint via `sort(mu_party)` ensures
Democrat < Republican on the ideological scale.

**Non-centered parameterization** avoids the "funnel of hell" — a geometry
problem that makes hierarchical models hard to sample.

### 3-Level Joint Model (Secondary)

When both chambers are combined:
```
mu_global     ~ Normal(0, 2)
sigma_chamber ~ HalfNormal(1)
mu_chamber    = mu_global + sigma_chamber * offset_chamber

offset_party_sorted = sort_per_chamber(offset_party)  -- D < R within each chamber
mu_group      = mu_chamber[c] + sigma_party * offset_party_sorted
sigma_within  ~ HalfNormal(sigma_scale)   -- adaptive: 0.5 for small groups, 1.0 otherwise
xi            = mu_group[g_i] + sigma_within[g_i] * xi_offset_i
```

**Bill matching (ADR-0043):** Bills voted on by both chambers are matched by
`bill_number` (preferring Final Action motions) to create shared `alpha`/`beta`
parameters. These shared items bridge the chambers via concurrent calibration —
the same mechanism used by flat IRT test equating (71-174 shared bills per session).

**Sign identification:** Shared bill parameters naturally constrain cross-chamber
sign and scale. The within-chamber sort (D < R on offsets) provides additional
identification. A post-hoc safety net (`fix_joint_sign_convention`) compares
joint xi with per-chamber hierarchical xi and negates any chamber whose legislators
are flipped — but should not trigger when shared bills provide sufficient bridging.

**Adaptive priors (ADR-0043):** Groups with fewer than 20 members get
`sigma_within ~ HalfNormal(0.5)` instead of `HalfNormal(1.0)` to prevent
convergence failures in small groups (Gelman 2015).

## Inputs

Reads from upstream phases:
- `results/<session>/eda/latest/data/` — filtered vote matrices
- `results/<session>/pca/latest/data/` — PCA scores for anchor selection
- `results/<session>/irt/latest/data/` — flat IRT ideal points (for shrinkage comparison)
- `data/<session>/` — rollcalls and legislators CSVs

## Outputs

### `data/` — Parquet intermediates + NetCDF posteriors

| File | Description |
|------|-------------|
| `hierarchical_ideal_points_{chamber}.parquet` | Ideal points with shrinkage vs flat |
| `group_params_{chamber}.parquet` | Party-level mu, sigma posteriors |
| `variance_decomposition_{chamber}.parquet` | ICC with uncertainty |
| `idata_{chamber}.nc` | Full posterior (ArviZ NetCDF) |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `party_posteriors_{chamber}.png` | "Where Do the Parties Stand?" |
| `icc_{chamber}.png` | "How Much Does Party Explain?" |
| `shrinkage_scatter_{chamber}.png` | "How Does Accounting for Party Change Estimates?" |
| `forest_{chamber}.png` | Forest plot with hierarchical ideal points |
| `dispersion_{chamber}.png` | "Which Party Has More Internal Disagreement?" |

## Interpretation Guide

- **Party posterior KDEs**: Separation = polarization. Overlap = moderate overlap.
- **ICC**: 0.7 means party explains 70% of ideological variance.
- **Shrinkage scatter**: Points off the diagonal moved. Arrows show direction.
- **Forest plot**: Same as flat IRT but with hierarchical ideal points.
- **Dispersion**: Wider curve = more internal disagreement within that party.

## Caveats

- Non-centered parameterization may be slightly less efficient than centered for
  large, well-identified groups. The safety margin is worth the minor cost.
- Joint model off by default (ADR-0074) — enable with `--run-joint`.
- Small Senate-D group (~10 legislators) produces wide credible intervals on the
  Democratic party mean in the Senate.
"""

# ── Constants ────────────────────────────────────────────────────────────────

HIER_N_SAMPLES = 2000
HIER_N_TUNE = 1500
HIER_N_CHAINS = 4
HIER_TARGET_ACCEPT = 0.95

# Party index convention: 0 = Democrat, 1 = Republican (after sorting)
PARTY_NAMES = ["Democrat", "Republican"]
PARTY_IDX_MAP = {"Democrat": 0, "Republican": 1}

# Shrinkage comparison thresholds
SHRINKAGE_MIN_DISTANCE = 0.5  # Min flat-to-party-mean distance for meaningful shrinkage_pct

# Small-group warning threshold (James-Stein requires J >= 3; with J=2 and small N,
# hierarchical shrinkage may be unreliable — see docs/hierarchical-shrinkage-deep-dive.md)
MIN_GROUP_SIZE_WARN = 15

# Group-size-adaptive priors: tighter HalfNormal for small groups (Gelman 2015).
# With J=2 groups and small N (e.g. ~11 Senate Democrats), the standard HalfNormal(1)
# prior on sigma_within allows the posterior to explore pathological geometries
# (funnel/mode-splitting), producing R-hat > 1.8 and ESS < 10. A tighter prior
# keeps the posterior closer to the prior, preventing catastrophic convergence failure.
SMALL_GROUP_THRESHOLD = 20
SMALL_GROUP_SIGMA_SCALE = 0.5

# Convergence diagnostic variable lists
HIER_CONVERGENCE_VARS = ["xi", "mu_party", "sigma_within", "alpha", "beta", "log_beta"]
JOINT_EXTRA_VARS = ["mu_group", "mu_chamber", "sigma_chamber", "sigma_party"]


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Hierarchical Bayesian IRT")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override flat IRT results directory")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--n-samples", type=int, default=HIER_N_SAMPLES, help="MCMC samples per chain"
    )
    parser.add_argument(
        "--n-tune", type=int, default=HIER_N_TUNE, help="MCMC tuning samples (discarded)"
    )
    parser.add_argument("--n-chains", type=int, default=HIER_N_CHAINS, help="Number of MCMC chains")
    parser.add_argument(
        "--cores", type=int, default=None, help="CPU cores for sampling (default: n_chains)"
    )
    parser.add_argument(
        "--run-joint", action="store_true",
        help="Run joint cross-chamber model (off by default — ADR-0074)",
    )
    parser.add_argument(
        "--init-strategy",
        default="auto",
        choices=["auto", "irt-informed", "pca-informed"],
        help="xi_offset initialization source (default: auto — prefer IRT, fall back to PCA)",
    )
    return parser.parse_args()


# ── Phase 1: Prepare Hierarchical Data ──────────────────────────────────────


def prepare_hierarchical_data(
    matrix: pl.DataFrame,
    legislators: pl.DataFrame,
    chamber: str,
) -> dict:
    """Extend flat IRT data with party indices for hierarchical model.

    Independent legislators are excluded from the hierarchical model because
    partial pooling by party requires party membership. They still appear in
    the flat IRT results.

    Returns the same dict as prepare_irt_data() plus:
    - party_idx: array mapping each legislator to their party index (0=D, 1=R)
    - party_names: list of party names in index order
    - n_parties: number of parties (2)
    - n_excluded: number of legislators excluded (non-major-party)
    """
    # Filter out non-major-party legislators (e.g. Independent) before IRT prep
    major_party_slugs = set(
        legislators.filter(pl.col("party").is_in(PARTY_NAMES))["legislator_slug"].to_list()
    )
    all_slugs = set(matrix["legislator_slug"].to_list())
    non_major = all_slugs - major_party_slugs
    if non_major:
        print(f"  Excluding {len(non_major)} non-major-party legislators: {sorted(non_major)}")
        matrix = matrix.filter(~pl.col("legislator_slug").is_in(non_major))

    data = prepare_irt_data(matrix, chamber)
    data["n_excluded"] = len(non_major)

    # Map legislator slugs to parties
    meta = legislators.select("legislator_slug", "party").unique(subset=["legislator_slug"])
    slug_to_party = dict(zip(meta["legislator_slug"].to_list(), meta["party"].to_list()))

    party_idx = np.array(
        [PARTY_IDX_MAP[slug_to_party[s]] for s in data["leg_slugs"]],
        dtype=np.int64,
    )

    data["party_idx"] = party_idx
    data["party_names"] = PARTY_NAMES
    data["n_parties"] = len(PARTY_NAMES)

    # Print party composition with small-group warning
    for i, name in enumerate(PARTY_NAMES):
        count = int((party_idx == i).sum())
        print(f"  {name}: {count} legislators")
        if count < MIN_GROUP_SIZE_WARN:
            print(
                f"  WARNING: {name} has only {count} legislators. "
                "Hierarchical shrinkage may be unreliable for groups this small "
                "(J=2 groups with small N; flat IRT may be more trustworthy). "
                "See docs/hierarchical-shrinkage-deep-dive.md"
            )

    return data


# ── Phase 2: Build and Sample Models ────────────────────────────────────────


def build_per_chamber_graph(
    data: dict,
    beta_prior: BetaPriorSpec = PRODUCTION_BETA,
) -> pm.Model:
    """Build 2-level hierarchical IRT model graph (no sampling).

    Model structure:
        mu_party (sorted) → xi (non-centered) → likelihood
        sigma_within (per party) controls within-party spread

    Returns the PyMC model for use with nutpie or pm.sample().
    """
    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y = data["y"]
    n_leg = data["n_legislators"]
    n_votes = data["n_votes"]
    party_idx = data["party_idx"]
    n_parties = data["n_parties"]

    # Group-size-adaptive priors for sigma_within (Gelman 2015: informative priors for J=2)
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
                f"sigma_within ~ HalfNormal({SMALL_GROUP_SIGMA_SCALE})"
            )

    coords = {
        "legislator": data["leg_slugs"],
        "vote": data["vote_ids"],
        "party": data["party_names"],
        "obs_id": np.arange(data["n_obs"]),
    }

    with pm.Model(coords=coords) as model:
        # --- Party-level parameters ---
        # Raw party means, then sort for identification (D < R)
        mu_party_raw = pm.Normal("mu_party_raw", mu=0, sigma=2, shape=n_parties)
        mu_party = pm.Deterministic("mu_party", pt.sort(mu_party_raw), dims="party")

        # Per-party within-group standard deviation (adaptive for small groups)
        sigma_within = pm.HalfNormal(
            "sigma_within", sigma=sigma_scale, shape=n_parties, dims="party"
        )

        # --- Non-centered legislator ideal points ---
        xi_offset = pm.Normal("xi_offset", mu=0, sigma=1, shape=n_leg, dims="legislator")
        xi = pm.Deterministic(
            "xi",
            mu_party[party_idx] + sigma_within[party_idx] * xi_offset,
            dims="legislator",
        )

        # --- Bill parameters ---
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")
        beta = beta_prior.build(n_votes)
        print(f"  Beta prior: {beta_prior.describe()}")

        # --- Likelihood ---
        eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

    return model


def build_per_chamber_model(
    data: dict,
    n_samples: int,
    n_tune: int,
    n_chains: int,
    cores: int | None = None,
    target_accept: float = HIER_TARGET_ACCEPT,
    xi_offset_initvals: np.ndarray | None = None,
    beta_prior: BetaPriorSpec = PRODUCTION_BETA,
) -> tuple[az.InferenceData, float]:
    """Build 2-level hierarchical IRT and sample with nutpie's Rust NUTS.

    Builds the model graph via build_per_chamber_graph(), then compiles and
    samples with nutpie. PCA-informed initialization is passed via
    nutpie.compile_pymc_model(initial_points=...).

    Args:
        xi_offset_initvals: Optional initial xi_offset values (from PCA).
            If provided, all chains start near these values. This prevents
            reflection mode-splitting — see ADR-0044.
        beta_prior: Specification for the bill discrimination prior.
            Defaults to PRODUCTION_BETA (Normal(mu=0, sigma=1)).

    Returns (InferenceData, sampling_time_seconds).
    """
    if target_accept != HIER_TARGET_ACCEPT:
        print(
            f"  Note: target_accept={target_accept} ignored (nutpie uses adaptive dual averaging)"
        )

    model = build_per_chamber_graph(data, beta_prior)

    # --- Compile with nutpie ---
    compile_kwargs: dict = {}
    if xi_offset_initvals is not None:
        # PCA init for xi_offset; jitter all OTHER RVs.
        # Critical: jitter_rvs=set() would cause HalfNormal sigma_within to
        # initialize at its support point (~0), producing log(0)=-inf.
        compile_kwargs["initial_points"] = {"xi_offset": xi_offset_initvals}
        compile_kwargs["jitter_rvs"] = {rv for rv in model.free_RVs if rv.name != "xi_offset"}
        print(
            f"  PCA-informed initvals: {len(xi_offset_initvals)} params, "
            f"range [{xi_offset_initvals.min():.2f}, {xi_offset_initvals.max():.2f}]"
        )
        jittered = [rv.name for rv in compile_kwargs["jitter_rvs"]]
        print(f"  jitter_rvs: {jittered} (xi_offset excluded)")

    print("  Compiling model with nutpie...")
    compiled = nutpie.compile_pymc_model(model, **compile_kwargs)

    # --- Sample ---
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


def _match_bills_across_chambers(
    house_vote_ids: list[str],
    senate_vote_ids: list[str],
    rollcalls: pl.DataFrame,
) -> tuple[list[dict], list[str], list[str]]:
    """Match bills across chambers by bill_number for shared item parameters.

    For each bill_number appearing in both chambers' vote sets:
    - Prefer the vote_id with "Final Action" or "Emergency Final Action" motion
    - If multiple, pick the latest chronologically (vote_id encodes timestamp)

    Returns (matched_bills, house_only_vids, senate_only_vids) where matched_bills
    is a list of dicts with keys: bill_number, house_vote_id, senate_vote_id.
    """
    # Build vote_id → bill_number / motion mappings from rollcalls
    rc = rollcalls.select("vote_id", "bill_number", "motion").filter(
        pl.col("vote_id").is_in(house_vote_ids + senate_vote_ids)
    )
    vid_to_bill: dict[str, str] = dict(zip(rc["vote_id"].to_list(), rc["bill_number"].to_list()))
    vid_to_motion: dict[str, str] = dict(zip(rc["vote_id"].to_list(), rc["motion"].to_list()))

    # Group vote_ids by bill_number and chamber
    house_bill_vids: dict[str, list[str]] = {}
    for vid in house_vote_ids:
        bill = vid_to_bill.get(vid)
        if bill:
            house_bill_vids.setdefault(bill, []).append(vid)

    senate_bill_vids: dict[str, list[str]] = {}
    for vid in senate_vote_ids:
        bill = vid_to_bill.get(vid)
        if bill:
            senate_bill_vids.setdefault(bill, []).append(vid)

    shared_bills = set(house_bill_vids.keys()) & set(senate_bill_vids.keys())

    def _pick_best_vid(vids: list[str]) -> str:
        """Pick best vote_id: prefer Final Action, then latest chronologically."""
        final_vids = [
            v
            for v in vids
            if (vid_to_motion.get(v) or "").lower() in ("final action", "emergency final action")
        ]
        candidates = final_vids if final_vids else vids
        return sorted(candidates)[-1]  # Latest chronologically (vote_id encodes timestamp)

    matched: list[dict] = []
    house_used: set[str] = set()
    senate_used: set[str] = set()

    for bill in sorted(shared_bills):
        h_vid = _pick_best_vid(house_bill_vids[bill])
        s_vid = _pick_best_vid(senate_bill_vids[bill])
        matched.append(
            {
                "bill_number": bill,
                "house_vote_id": h_vid,
                "senate_vote_id": s_vid,
            }
        )
        house_used.add(h_vid)
        senate_used.add(s_vid)

    house_only = [v for v in house_vote_ids if v not in house_used]
    senate_only = [v for v in senate_vote_ids if v not in senate_used]

    return matched, house_only, senate_only


def build_joint_graph(
    house_data: dict,
    senate_data: dict,
    rollcalls: pl.DataFrame | None = None,
    beta_prior: BetaPriorSpec = PRODUCTION_BETA,
    alpha_sigma: float = 5.0,
) -> tuple[pm.Model, dict]:
    """Build 3-level joint cross-chamber hierarchical IRT model graph (no sampling).

    Model structure:
        mu_global → mu_chamber → mu_group (4 groups: HD, HR, SD, SR) → xi → likelihood

    When rollcalls is provided, bills are matched across chambers by bill_number
    to create shared alpha/beta parameters. These shared items bridge the chambers,
    providing natural sign and scale identification (concurrent calibration).
    Without rollcalls, falls back to vote_id deduplication (no shared items).

    Args:
        house_data: Per-chamber data dict from prepare_hierarchical_data() for House.
        senate_data: Per-chamber data dict from prepare_hierarchical_data() for Senate.
        rollcalls: Optional rollcalls DataFrame for bill matching across chambers.
        beta_prior: Specification for the bill discrimination prior.
            Defaults to PRODUCTION_BETA (Normal(mu=0, sigma=1)).
        alpha_sigma: Standard deviation for the bill difficulty prior.
            Defaults to 5.0 (legacy). Joint model uses 2.0 for tighter regularization.

    Returns (pm.Model, combined_data_dict).
    """
    # Combine legislator data
    n_house = house_data["n_legislators"]
    n_senate = senate_data["n_legislators"]
    n_leg = n_house + n_senate

    all_slugs = house_data["leg_slugs"] + senate_data["leg_slugs"]

    # --- Bill matching: shared items bridge the chambers ---
    house_vote_ids = house_data["vote_ids"]
    senate_vote_ids = senate_data["vote_ids"]
    n_shared = 0

    if rollcalls is not None:
        matched_bills, house_only_vids, senate_only_vids = _match_bills_across_chambers(
            house_vote_ids, senate_vote_ids, rollcalls
        )
        n_shared = len(matched_bills)

        # Build unified vote list: matched (shared) first, then house-only, then senate-only
        # Each matched bill gets ONE index — both chambers' observations point to the same
        # alpha/beta, which is the mathematical bridge for cross-chamber identification.
        all_vote_ids: list[str] = []
        matched_labels: list[str] = []
        for m in matched_bills:
            label = f"matched_{m['bill_number']}"
            matched_labels.append(label)
            all_vote_ids.append(label)
        all_vote_ids.extend(house_only_vids)
        all_vote_ids.extend(senate_only_vids)
        n_votes = len(all_vote_ids)
        vote_to_idx = {v: i for i, v in enumerate(all_vote_ids)}

        # Build lookup: original vote_id → unified index
        # Matched bills: both house and senate vote_ids map to the same index
        original_vid_to_idx: dict[str, int] = {}
        for i, m in enumerate(matched_bills):
            original_vid_to_idx[m["house_vote_id"]] = i  # index of matched label
            original_vid_to_idx[m["senate_vote_id"]] = i
        for vid in house_only_vids:
            original_vid_to_idx[vid] = vote_to_idx[vid]
        for vid in senate_only_vids:
            original_vid_to_idx[vid] = vote_to_idx[vid]

        print(
            f"  Joint bill matching: {n_shared} shared bills, "
            f"{len(house_only_vids)} house-only, {len(senate_only_vids)} senate-only"
        )
    else:
        # Fallback: no bill matching (legacy behavior)
        all_vote_ids = list(dict.fromkeys(house_vote_ids + senate_vote_ids))
        n_votes = len(all_vote_ids)
        original_vid_to_idx = {v: i for i, v in enumerate(all_vote_ids)}
        matched_bills = []
        print("  Joint: no rollcalls provided, falling back to vote_id dedup (no shared items)")

    # Remap indices for combined model
    senate_leg_offset = n_house

    combined_leg_idx = np.concatenate(
        [
            house_data["leg_idx"],
            senate_data["leg_idx"] + senate_leg_offset,
        ]
    )
    combined_vote_idx = np.concatenate(
        [
            np.array([original_vid_to_idx[house_vote_ids[v]] for v in house_data["vote_idx"]]),
            np.array([original_vid_to_idx[senate_vote_ids[v]] for v in senate_data["vote_idx"]]),
        ]
    )
    combined_y = np.concatenate([house_data["y"], senate_data["y"]])

    # Group indices: 0=House-D, 1=House-R, 2=Senate-D, 3=Senate-R
    group_names = ["House Democrat", "House Republican", "Senate Democrat", "Senate Republican"]
    n_groups = len(group_names)
    group_idx = np.concatenate(
        [
            house_data["party_idx"],  # 0=D, 1=R for House
            senate_data["party_idx"] + 2,  # 2=D, 3=R for Senate
        ]
    )

    # Chamber index for each group (0=House, 1=Senate)
    group_chamber = np.array([0, 0, 1, 1], dtype=np.int64)

    # Group-size-adaptive priors for sigma_within (Gelman 2015: informative priors for J=2)
    party_counts = np.array([int((group_idx == g).sum()) for g in range(n_groups)])
    sigma_scale = np.array(
        [
            SMALL_GROUP_SIGMA_SCALE if party_counts[g] < SMALL_GROUP_THRESHOLD else 1.0
            for g in range(n_groups)
        ]
    )
    for g in range(n_groups):
        if party_counts[g] < SMALL_GROUP_THRESHOLD:
            print(
                f"  Adaptive prior: {group_names[g]} ({party_counts[g]} members) → "
                f"sigma_within ~ HalfNormal({SMALL_GROUP_SIGMA_SCALE})"
            )

    n_obs = len(combined_y)

    coords = {
        "legislator": all_slugs,
        "vote": all_vote_ids,
        "group": group_names,
        "chamber": ["House", "Senate"],
        "obs_id": np.arange(n_obs),
    }

    with pm.Model(coords=coords) as model:
        # --- Chamber-level ---
        mu_global = pm.Normal("mu_global", mu=0, sigma=2)
        sigma_chamber = pm.HalfNormal("sigma_chamber", sigma=1)
        chamber_offset = pm.Normal("chamber_offset", mu=0, sigma=1, shape=2, dims="chamber")
        mu_chamber = pm.Deterministic(
            "mu_chamber", mu_global + sigma_chamber * chamber_offset, dims="chamber"
        )

        # --- Group-level (4 groups: House-D, House-R, Senate-D, Senate-R) ---
        # Use ordering constraint within each chamber (D < R) for identification,
        # mirroring the per-chamber model's pt.sort(mu_party_raw) approach.
        sigma_party = pm.HalfNormal("sigma_party", sigma=1)
        group_offset_raw = pm.Normal(
            "group_offset_raw", mu=0, sigma=1, shape=n_groups, dims="group"
        )
        # Sort each chamber's pair so D < R: indices [0,1] for House, [2,3] for Senate
        house_pair = pt.sort(group_offset_raw[:2])
        senate_pair = pt.sort(group_offset_raw[2:])
        group_offset_sorted = pt.concatenate([house_pair, senate_pair])
        mu_group = pm.Deterministic(
            "mu_group",
            mu_chamber[group_chamber] + sigma_party * group_offset_sorted,
            dims="group",
        )

        # --- Within-group (adaptive priors for small groups) ---
        sigma_within = pm.HalfNormal(
            "sigma_within", sigma=sigma_scale, shape=n_groups, dims="group"
        )

        # --- Non-centered legislator ideal points ---
        xi_offset = pm.Normal("xi_offset", mu=0, sigma=1, shape=n_leg, dims="legislator")
        xi = pm.Deterministic(
            "xi",
            mu_group[group_idx] + sigma_within[group_idx] * xi_offset,
            dims="legislator",
        )

        # --- Bill parameters ---
        alpha = pm.Normal("alpha", mu=0, sigma=alpha_sigma, shape=n_votes, dims="vote")
        beta = beta_prior.build(n_votes)
        print(f"  Alpha prior: Normal(mu=0, sigma={alpha_sigma})")
        print(f"  Beta prior: {beta_prior.describe()}")

        # --- Likelihood ---
        eta = beta[combined_vote_idx] * xi[combined_leg_idx] - alpha[combined_vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=combined_y, dims="obs_id")

    combined_data = {
        "leg_slugs": all_slugs,
        "vote_ids": all_vote_ids,
        "n_legislators": n_leg,
        "n_votes": n_votes,
        "n_shared_bills": n_shared,
        "n_obs": n_obs,
        "group_idx": group_idx,
        "group_names": group_names,
        "n_groups": n_groups,
        "group_chamber": group_chamber,
        "n_house": n_house,
        "n_senate": n_senate,
        "matched_bills": matched_bills,
    }

    return model, combined_data


def build_joint_model(
    house_data: dict,
    senate_data: dict,
    n_samples: int,
    n_tune: int,
    n_chains: int,
    rollcalls: pl.DataFrame | None = None,
    cores: int | None = None,
    target_accept: float = HIER_TARGET_ACCEPT,
    beta_prior: BetaPriorSpec = PRODUCTION_BETA,
    alpha_sigma: float = 5.0,
    xi_offset_initvals: np.ndarray | None = None,
) -> tuple[az.InferenceData, dict, float]:
    """Build 3-level joint cross-chamber hierarchical IRT and sample with nutpie.

    Builds the model graph via build_joint_graph(), then compiles and samples
    with nutpie's Rust NUTS sampler (ADR-0053).

    Args:
        beta_prior: Specification for the bill discrimination prior.
            Defaults to PRODUCTION_BETA (Normal(mu=0, sigma=1)).
        alpha_sigma: Standard deviation for the bill difficulty prior.
            Defaults to 5.0 (legacy). Joint model uses 2.0 for tighter regularization.
        xi_offset_initvals: Optional initial xi_offset values (from PCA).
            If provided, all chains start near these values.

    Returns (InferenceData, combined_data_dict, sampling_time_seconds).
    """
    if target_accept != HIER_TARGET_ACCEPT:
        print(
            f"  Note: target_accept={target_accept} ignored (nutpie uses adaptive dual averaging)"
        )
    if cores is not None:
        print(f"  Note: cores={cores} ignored (nutpie manages its own threads)")

    model, combined_data = build_joint_graph(
        house_data,
        senate_data,
        rollcalls=rollcalls,
        beta_prior=beta_prior,
        alpha_sigma=alpha_sigma,
    )

    # --- Compile with nutpie ---
    compile_kwargs: dict = {}
    if xi_offset_initvals is not None:
        # PCA init for xi_offset; jitter all OTHER RVs.
        # Critical: jitter_rvs=set() would cause HalfNormal sigma_within to
        # initialize at its support point (~0), producing log(0)=-inf.
        compile_kwargs["initial_points"] = {"xi_offset": xi_offset_initvals}
        compile_kwargs["jitter_rvs"] = {rv for rv in model.free_RVs if rv.name != "xi_offset"}
        print(
            f"  PCA-informed initvals: {len(xi_offset_initvals)} params, "
            f"range [{xi_offset_initvals.min():.2f}, {xi_offset_initvals.max():.2f}]"
        )
        jittered = [rv.name for rv in compile_kwargs["jitter_rvs"]]
        print(f"  jitter_rvs: {jittered} (xi_offset excluded)")

    print("  Compiling model with nutpie...")
    compiled = nutpie.compile_pymc_model(model, **compile_kwargs)

    # --- Sample ---
    n_shared = combined_data["n_shared_bills"]
    n_leg = combined_data["n_legislators"]
    n_votes = combined_data["n_votes"]
    n_obs = combined_data["n_obs"]
    print(f"  Joint: {n_samples} draws, {n_tune} tune, {n_chains} chains")
    print(f"  {n_leg} legislators, {n_votes} votes ({n_shared} shared), {n_obs} observations")
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

    print(f"  Joint sampling complete in {sampling_time:.1f}s")
    return idata, combined_data, sampling_time


def fix_joint_sign_convention(
    idata: az.InferenceData,
    combined_data: dict,
    per_chamber_results: dict,
) -> az.InferenceData:
    """Fix sign indeterminacy in the joint cross-chamber model.

    The joint model combines House and Senate votes with shared beta parameters.
    However, for chamber-specific votes (most bills), the model can flip the sign
    of beta_j and xi_i simultaneously without changing the likelihood. The
    within-chamber sort constraint (D < R on offsets) is necessary but not
    sufficient for global identification — it constrains the *relative* ordering
    but not the *absolute* sign.

    Fix: compare joint xi with per-chamber hierarchical xi (which have correct
    sign convention from direct sort). If a chamber's legislators are negatively
    correlated with the per-chamber model, negate xi for those legislators.

    Returns (idata, flipped_chambers) where flipped_chambers is a list of chamber
    names that were corrected (e.g., ["Senate"]). Callers should use empirical xi
    means instead of mu_group for party_mean in flipped chambers.
    """
    xi_post = idata.posterior["xi"].values  # (chain, draw, legislator)
    n_house = combined_data["n_house"]
    flipped: list[str] = []

    for chamber, start, end in [("House", 0, n_house), ("Senate", n_house, None)]:
        if chamber not in per_chamber_results:
            continue

        # Get per-chamber hierarchical xi means (known correct sign)
        chamber_ip = per_chamber_results[chamber]["ideal_points"]
        chamber_slugs = chamber_ip["legislator_slug"].to_list()
        chamber_xi = chamber_ip["xi_mean"].to_numpy()

        # Get joint xi means for this chamber's legislators
        joint_slugs = combined_data["leg_slugs"][start:end]
        joint_xi_mean = xi_post[:, :, start:end].mean(axis=(0, 1))

        # Match legislators by slug for correlation
        joint_slug_to_xi = dict(zip(joint_slugs, joint_xi_mean))
        matched_joint = []
        matched_chamber = []
        for slug, xi in zip(chamber_slugs, chamber_xi):
            if slug in joint_slug_to_xi:
                matched_joint.append(joint_slug_to_xi[slug])
                matched_chamber.append(xi)

        if len(matched_joint) < 3:
            continue

        r = float(np.corrcoef(matched_joint, matched_chamber)[0, 1])

        if r < 0:
            print(f"  Sign flip detected for {chamber} (r={r:.3f}), correcting...")
            # Negate xi for this chamber's legislators across all posterior samples
            xi_post[:, :, start:end] *= -1
            flipped.append(chamber)
        else:
            print(f"  {chamber} sign OK (r={r:.3f})")

    if flipped:
        # Write corrected xi back to the posterior
        idata.posterior["xi"].values = xi_post
        # Note: mu_group posterior is NOT corrected — it was estimated jointly with
        # the flipped xi and cannot be meaningfully recovered by simple arithmetic.
        # The extract function uses corrected xi means per group instead of mu_group
        # when flipped_chambers is set.
        print(f"  Corrected sign for: {', '.join(flipped)}")
    else:
        print("  No sign correction needed")

    return idata, flipped


# ── Phase 3: Convergence Diagnostics ────────────────────────────────────────


def check_hierarchical_convergence(idata: az.InferenceData, chamber: str) -> dict:
    """Check convergence for hierarchical model (xi + mu_party + sigma_within).

    Returns dict with all diagnostic metrics.
    """
    print_header(f"CONVERGENCE — {chamber}")

    diag: dict = {}

    # Variables to check
    available_vars = [v for v in HIER_CONVERGENCE_VARS if v in idata.posterior]

    # Check also joint-specific variables
    for extra in JOINT_EXTRA_VARS:
        if extra in idata.posterior:
            available_vars.append(extra)

    # R-hat
    rhat = az.rhat(idata)
    for var in available_vars:
        if var in rhat:
            max_rhat = float(rhat[var].max())
            diag[f"{var}_rhat_max"] = max_rhat
            status = "OK" if max_rhat < RHAT_THRESHOLD else "WARNING"
            print(f"  R-hat ({var}): max = {max_rhat:.4f}  {status}")

    # ESS
    # Note: ESS_THRESHOLD=400 comes from Vehtari et al. (2021) which recommends
    # 100 per chain (Stan default 4 chains → 400 total). We run 2 chains, so the
    # per-chain recommendation is 100/chain = 200 total. We keep 400 for consistency
    # with flat IRT but report per-chain ESS for transparency.
    n_chains = len(idata.posterior.chain)
    ess = az.ess(idata)
    for var in available_vars:
        if var in ess:
            min_ess = float(ess[var].min())
            per_chain = min_ess / n_chains
            diag[f"{var}_ess_min"] = min_ess
            diag[f"{var}_ess_per_chain"] = per_chain
            status = "OK" if min_ess > ESS_THRESHOLD else "WARNING"
            per_chain_status = "OK" if per_chain > 100 else "WARNING"
            print(
                f"  ESS ({var}): min = {min_ess:.0f}  {status}  "
                f"(per-chain: {per_chain:.0f}  {per_chain_status})"
            )

    # Divergences
    divergences = int(idata.sample_stats["diverging"].sum().values)
    diag["divergences"] = divergences
    div_ok = divergences < MAX_DIVERGENCES
    print(f"  Divergences: {divergences}  {'OK' if div_ok else 'WARNING'}")

    # E-BFMI
    bfmi_values = az.bfmi(idata)
    diag["ebfmi"] = [float(v) for v in bfmi_values]
    bfmi_ok = all(v > 0.3 for v in bfmi_values)
    for i, v in enumerate(bfmi_values):
        print(f"  E-BFMI chain {i}: {v:.3f}  {'OK' if v > 0.3 else 'WARNING'}")

    # Overall assessment
    rhat_ok = all(
        diag.get(f"{v}_rhat_max", 0) < RHAT_THRESHOLD
        for v in available_vars
        if f"{v}_rhat_max" in diag
    )
    ess_ok = all(
        diag.get(f"{v}_ess_min", float("inf")) > ESS_THRESHOLD
        for v in available_vars
        if f"{v}_ess_min" in diag
    )
    diag["all_ok"] = rhat_ok and ess_ok and div_ok and bfmi_ok
    if diag["all_ok"]:
        print("  CONVERGENCE: ALL CHECKS PASSED")
    else:
        print("  CONVERGENCE: SOME CHECKS FAILED — inspect diagnostics")

    return diag


# ── Phase 4: Extract Results ────────────────────────────────────────────────


def extract_hierarchical_ideal_points(
    idata: az.InferenceData,
    data: dict,
    legislators: pl.DataFrame,
    flat_ip: pl.DataFrame | None = None,
    flipped_chambers: list[str] | None = None,
) -> pl.DataFrame:
    """Extract posterior summaries with shrinkage comparison to flat IRT.

    Returns DataFrame with legislator_slug, xi_mean, xi_sd, xi_hdi_*,
    party_mean, shrinkage_pct, delta_from_flat, plus metadata.

    For flipped chambers (sign-corrected by fix_joint_sign_convention), party_mean
    is computed from empirical xi means per group rather than mu_group, since the
    mu_group posterior was estimated in the wrong sign convention.
    """
    xi_posterior = idata.posterior["xi"]
    xi_mean = xi_posterior.mean(dim=["chain", "draw"]).values
    xi_sd = xi_posterior.std(dim=["chain", "draw"]).values
    xi_hdi = az.hdi(idata, var_names=["xi"], hdi_prob=0.95)["xi"].values

    # Group/party means — joint model uses mu_group, per-chamber uses mu_party
    if "mu_group" in idata.posterior:
        mu_mean = idata.posterior["mu_group"].mean(dim=["chain", "draw"]).values
        group_idx = data.get("group_idx", data.get("party_idx"))
    else:
        mu_mean = idata.posterior["mu_party"].mean(dim=["chain", "draw"]).values
        group_idx = data["party_idx"]

    # For flipped chambers, compute empirical group means from corrected xi
    empirical_group_means: dict[int, float] = {}
    if flipped_chambers and "group_names" in data:
        group_names = data["group_names"]
        for chamber in flipped_chambers:
            for gi, name in enumerate(group_names):
                if chamber in name:
                    # Compute mean xi for all legislators in this group
                    members = [j for j, gj in enumerate(group_idx) if gj == gi]
                    if members:
                        empirical_group_means[gi] = float(np.mean([xi_mean[j] for j in members]))

    slugs = data["leg_slugs"]

    rows = []
    for i, slug in enumerate(slugs):
        gi = group_idx[i]
        party_mean = empirical_group_means.get(gi, float(mu_mean[gi]))
        rows.append(
            {
                "legislator_slug": slug,
                "xi_mean": float(xi_mean[i]),
                "xi_sd": float(xi_sd[i]),
                "xi_hdi_2.5": float(xi_hdi[i, 0]),
                "xi_hdi_97.5": float(xi_hdi[i, 1]),
                "party_mean": party_mean,
            }
        )

    df = pl.DataFrame(rows)

    # Join legislator metadata
    meta = legislators.select("legislator_slug", "full_name", "party", "district", "chamber")
    df = df.join(meta, on="legislator_slug", how="left")

    # Shrinkage comparison with flat IRT
    if flat_ip is not None:
        flat_cols = flat_ip.select(
            pl.col("legislator_slug"),
            pl.col("xi_mean").alias("flat_xi_mean"),
            pl.col("xi_sd").alias("flat_xi_sd"),
        )
        df = df.join(flat_cols, on="legislator_slug", how="left")

        # Rescale flat estimates to hierarchical scale via linear regression.
        # The two models produce ideal points on different scales (flat ≈ [-4,3],
        # hier ≈ [-11,9]) because their identification constraints differ.
        # A linear transform (slope, intercept) maps flat → hier for comparison.
        matched = df.filter(pl.col("flat_xi_mean").is_not_null() & pl.col("xi_mean").is_not_null())
        if len(matched) > 2:
            flat_vals = matched["flat_xi_mean"].to_numpy()
            hier_vals = matched["xi_mean"].to_numpy()
            slope, intercept = np.polyfit(flat_vals, hier_vals, 1)
            df = df.with_columns(
                (pl.col("flat_xi_mean") * slope + intercept).alias("flat_xi_rescaled"),
            )
        else:
            slope = 1.0
            print(
                f"  WARNING: Only {len(matched)} matched legislators for shrinkage "
                "rescaling (need >2). Falling back to slope=1.0 (no rescaling)."
            )
            df = df.with_columns(
                pl.col("flat_xi_mean").alias("flat_xi_rescaled"),
            )

        df = df.with_columns(
            [
                # Delta in hierarchical scale (flat rescaled to match)
                (pl.col("xi_mean") - pl.col("flat_xi_rescaled")).alias("delta_from_flat"),
            ]
        )

        # Determine if shrinkage is toward party mean (using rescaled flat)
        flat_dist = (pl.col("flat_xi_rescaled") - pl.col("party_mean")).abs()
        hier_dist = (pl.col("xi_mean") - pl.col("party_mean")).abs()
        df = df.with_columns(
            [
                (hier_dist < flat_dist).alias("toward_party_mean"),
                # Shrinkage = fraction of flat's distance to party mean absorbed by pooling.
                # 100% = fully pooled to party mean. 0% = no change. Negative = moved away.
                # Null when flat_dist < 0.5 (ratio unstable near party mean) or anchored.
                pl.when((pl.col("flat_xi_sd") > 0.01) & (flat_dist > SHRINKAGE_MIN_DISTANCE))
                .then((1 - hier_dist / flat_dist) * 100)
                .otherwise(None)
                .alias("shrinkage_pct"),
            ]
        )

        # Keep flat_xi_rescaled for downstream use (scatter plot, cross-session comparison)
    else:
        df = df.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("flat_xi_mean"),
                pl.lit(None).cast(pl.Float64).alias("flat_xi_sd"),
                pl.lit(None).cast(pl.Float64).alias("flat_xi_rescaled"),
                pl.lit(None).cast(pl.Float64).alias("delta_from_flat"),
                pl.lit(None).cast(pl.Float64).alias("shrinkage_pct"),
                pl.lit(None).cast(pl.Boolean).alias("toward_party_mean"),
            ]
        )

    return df.sort("xi_mean", descending=True)


def extract_group_params(
    idata: az.InferenceData,
    data: dict,
) -> pl.DataFrame:
    """Extract party-level mu and sigma posteriors.

    Only applicable to per-chamber models (which have mu_party). Joint models use
    mu_group with a different structure (4 groups, not 2 parties) — call with joint
    InferenceData will raise ValueError.

    Returns DataFrame with party, mu_mean, mu_sd, mu_hdi_*, sigma_within_mean, etc.
    """
    if "mu_party" not in idata.posterior:
        msg = (
            "extract_group_params only supports per-chamber models (mu_party). "
            "Joint model data has mu_group instead."
        )
        raise ValueError(msg)
    mu_post = idata.posterior["mu_party"]
    mu_mean = mu_post.mean(dim=["chain", "draw"]).values
    mu_sd = mu_post.std(dim=["chain", "draw"]).values
    mu_hdi = az.hdi(idata, var_names=["mu_party"], hdi_prob=0.95)["mu_party"].values

    sigma_post = idata.posterior["sigma_within"]
    sigma_mean = sigma_post.mean(dim=["chain", "draw"]).values
    sigma_sd = sigma_post.std(dim=["chain", "draw"]).values
    sigma_hdi = az.hdi(idata, var_names=["sigma_within"], hdi_prob=0.95)["sigma_within"].values

    party_names = data["party_names"]
    rows = []
    for i, name in enumerate(party_names):
        count = int((data["party_idx"] == i).sum())
        rows.append(
            {
                "party": name,
                "n_legislators": count,
                "mu_mean": float(mu_mean[i]),
                "mu_sd": float(mu_sd[i]),
                "mu_hdi_2.5": float(mu_hdi[i, 0]),
                "mu_hdi_97.5": float(mu_hdi[i, 1]),
                "sigma_within_mean": float(sigma_mean[i]),
                "sigma_within_sd": float(sigma_sd[i]),
                "sigma_within_hdi_2.5": float(sigma_hdi[i, 0]),
                "sigma_within_hdi_97.5": float(sigma_hdi[i, 1]),
            }
        )

    return pl.DataFrame(rows)


def compute_variance_decomposition(
    idata: az.InferenceData,
    data: dict,
) -> pl.DataFrame:
    """Compute ICC (intraclass correlation) from posterior samples.

    ICC = sigma_between² / (sigma_between² + sigma_within_pooled²)

    sigma_between is computed from the party means (var of mu_party).
    sigma_within_pooled is the group-size-weighted mean of the per-party sigma_within.

    Returns single-row DataFrame with icc_mean, icc_sd, icc_ci_*.
    """
    mu_post = idata.posterior["mu_party"].values  # (chain, draw, party)
    sigma_post = idata.posterior["sigma_within"].values  # (chain, draw, party)

    n_chain, n_draw, n_party = mu_post.shape

    # Compute ICC per posterior draw (vectorized)
    # Between-party variance: var of party means across parties per draw
    # mu_post shape: (chain, draw, party) → var across party axis
    sigma_between_sq = np.var(mu_post, axis=2)  # (chain, draw)

    # Within-party variance: pooled (weighted by group size)
    party_counts = np.array([(data["party_idx"] == p).sum() for p in range(n_party)])
    total = party_counts.sum()
    # sigma_post shape: (chain, draw, party)
    sigma_within_pooled_sq = (
        np.sum(party_counts[np.newaxis, np.newaxis, :] * sigma_post**2, axis=2) / total
    )  # (chain, draw)

    total_var = sigma_between_sq + sigma_within_pooled_sq
    icc_flat = np.where(total_var > 0, sigma_between_sq / total_var, 0.0)
    icc_samples = icc_flat.ravel()  # (chain * draw,)

    icc_mean = float(np.mean(icc_samples))
    icc_sd = float(np.std(icc_samples))
    icc_ci = np.percentile(icc_samples, [2.5, 97.5])

    print(f"  ICC: {icc_mean:.3f} ± {icc_sd:.3f} [{icc_ci[0]:.3f}, {icc_ci[1]:.3f}]")
    print(f"  → Party explains {icc_mean:.0%} of ideological variance")

    return pl.DataFrame(
        {
            "icc_mean": [icc_mean],
            "icc_sd": [icc_sd],
            "icc_ci_2.5": [float(icc_ci[0])],
            "icc_ci_97.5": [float(icc_ci[1])],
        }
    )


def compute_flat_hier_correlation(
    hier_ip: pl.DataFrame,
    flat_ip: pl.DataFrame,
    chamber: str,
) -> float:
    """Compute Pearson r between hierarchical and flat IRT ideal points."""
    merged = hier_ip.select("legislator_slug", "xi_mean").join(
        flat_ip.select("legislator_slug", pl.col("xi_mean").alias("flat_xi")),
        on="legislator_slug",
        how="inner",
    )
    if merged.height < 3:
        return float("nan")

    hier_arr = merged["xi_mean"].to_numpy()
    flat_arr = merged["flat_xi"].to_numpy()
    r = float(np.corrcoef(hier_arr, flat_arr)[0, 1])
    print(f"  {chamber}: Hierarchical vs flat Pearson r = {r:.4f}")
    return r


# ── Phase 5: Plots ──────────────────────────────────────────────────────────


def plot_party_posteriors(
    idata: az.InferenceData,
    data: dict,
    chamber: str,
    out_dir: Path,
) -> None:
    """KDE of party mean posteriors — 'Where Do the Parties Stand?'"""
    mu_post = idata.posterior["mu_party"].values  # (chain, draw, party)
    party_names = data["party_names"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, name in enumerate(party_names):
        samples = mu_post[:, :, i].flatten()
        color = PARTY_COLORS.get(name, "#888888")

        # KDE
        kde = stats.gaussian_kde(samples)
        x = np.linspace(samples.min() - 0.5, samples.max() + 0.5, 300)
        ax.plot(x, kde(x), color=color, linewidth=2.5, label=name)
        ax.fill_between(x, kde(x), alpha=0.15, color=color)

        # Posterior mean marker
        mean_val = float(np.mean(samples))
        ax.axvline(mean_val, color=color, linestyle="--", alpha=0.6, linewidth=1)

    ax.set_xlabel("Party Mean Ideal Point (Liberal ← → Conservative)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"{chamber} — Where Do the Parties Stand?\n"
        "Posterior distributions of party-level ideal points",
        fontsize=13,
        fontweight="bold",
    )

    ax.text(
        0.02,
        0.98,
        "Separation = polarization\nOverlap = ideological common ground",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.9},
    )

    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"party_posteriors_{chamber.lower()}.png")


def plot_icc(
    icc_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Bar chart of ICC — 'How Much Does Party Explain?'"""
    icc_mean = float(icc_df["icc_mean"][0])
    icc_lo = float(icc_df["icc_ci_2.5"][0])
    icc_hi = float(icc_df["icc_ci_97.5"][0])

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_colors = ["#4C72B0", "#C0C0C0"]
    ax.bar(
        ["Party", "Individual"],
        [icc_mean, 1 - icc_mean],
        color=bar_colors,
        edgecolor="white",
        linewidth=2,
        width=0.5,
    )

    # Error bar on party portion
    ax.errorbar(
        0,
        icc_mean,
        yerr=[[icc_mean - icc_lo], [icc_hi - icc_mean]],
        color="black",
        capsize=8,
        capthick=2,
        linewidth=2,
    )

    # Percentage labels
    ax.text(
        0,
        icc_mean / 2,
        f"{icc_mean:.0%}",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="white",
    )
    ax.text(
        1,
        icc_mean + (1 - icc_mean) / 2,
        f"{1 - icc_mean:.0%}",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#333333",
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of Ideological Variance", fontsize=12)
    ax.set_title(
        f"{chamber} — How Much Does Party Explain?\n"
        f"ICC = {icc_mean:.0%} [{icc_lo:.0%}, {icc_hi:.0%}]",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Party", "Individual"], fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=8)

    fig.text(
        0.5,
        0.01,
        f"Party membership explains {icc_mean:.0%} of the variation in "
        f"how {chamber} members vote. "
        f"The remaining {1 - icc_mean:.0%} reflects individual differences within parties.",
        ha="center",
        fontsize=10,
        fontstyle="italic",
        color="#555555",
        wrap=True,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, out_dir / f"icc_{chamber.lower()}.png")


def plot_shrinkage_scatter(
    hier_ip: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Flat vs hierarchical scatter — 'How Does Accounting for Party Change Estimates?'"""
    # Use rescaled flat values if available (same scale as hierarchical)
    flat_col = "flat_xi_rescaled" if "flat_xi_rescaled" in hier_ip.columns else "flat_xi_mean"
    if flat_col not in hier_ip.columns:
        return

    df = hier_ip.drop_nulls(subset=[flat_col])
    if df.height == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    for party, color in PARTY_COLORS.items():
        sub = df.filter(pl.col("party") == party)
        if sub.height == 0:
            continue
        flat = sub[flat_col].to_numpy()
        hier = sub["xi_mean"].to_numpy()
        ax.scatter(
            flat,
            hier,
            c=color,
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label=party,
        )

    # Identity line
    all_flat = df[flat_col].to_numpy()
    all_hier = df["xi_mean"].to_numpy()
    lims = [
        min(all_flat.min(), all_hier.min()) - 0.3,
        max(all_flat.max(), all_hier.max()) + 0.3,
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, label="No change")

    # Annotate biggest movers (use delta_from_flat which is scale-corrected)
    if "delta_from_flat" in df.columns:
        df_with_delta = df.with_columns(pl.col("delta_from_flat").abs().alias("abs_delta")).sort(
            "abs_delta", descending=True
        )
    else:
        df_with_delta = df.with_columns(
            (pl.col("xi_mean") - pl.col(flat_col)).abs().alias("abs_delta")
        ).sort("abs_delta", descending=True)

    for row in df_with_delta.head(5).iter_rows(named=True):
        name = row.get("full_name", row["legislator_slug"])
        last_name = name.split()[-1] if name else "?"
        ax.annotate(
            last_name,
            (row[flat_col], row["xi_mean"]),
            fontsize=8,
            fontweight="bold",
            xytext=(8, 8),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.8},
        )

    pearson_r = float(np.corrcoef(all_flat, all_hier)[0, 1])
    ax.set_xlabel("Flat IRT Ideal Point (rescaled)", fontsize=12)
    ax.set_ylabel("Hierarchical IRT Ideal Point", fontsize=12)
    ax.set_title(
        f"{chamber} — How Does Accounting for Party Change Estimates?\n"
        f"Pearson r = {pearson_r:.4f}. Points off the diagonal moved due to party pooling.",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.98,
        "Labels show the 5 legislators\nwhose estimates changed the most",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.9},
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"shrinkage_scatter_{chamber.lower()}.png")


def plot_dispersion(
    idata: az.InferenceData,
    data: dict,
    chamber: str,
    out_dir: Path,
) -> None:
    """KDE of sigma_within per party — 'Which Party Has More Internal Disagreement?'"""
    sigma_post = idata.posterior["sigma_within"].values  # (chain, draw, party)
    party_names = data["party_names"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, name in enumerate(party_names):
        samples = sigma_post[:, :, i].flatten()
        color = PARTY_COLORS.get(name, "#888888")
        kde = stats.gaussian_kde(samples)
        x = np.linspace(max(0, samples.min() - 0.1), samples.max() + 0.3, 300)
        ax.plot(x, kde(x), color=color, linewidth=2.5, label=name)
        ax.fill_between(x, kde(x), alpha=0.15, color=color)

        mean_val = float(np.mean(samples))
        ax.axvline(mean_val, color=color, linestyle="--", alpha=0.6, linewidth=1)

    ax.set_xlabel("Within-Party Standard Deviation (higher = more internal disagreement)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"{chamber} — Which Party Has More Internal Disagreement?",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.98,
        "Wider curves to the right = that party's\nmembers disagree more with each other",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.9},
    )
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"dispersion_{chamber.lower()}.png")


def plot_joint_party_spread(
    idata: az.InferenceData,
    combined_data: dict,
    out_dir: Path,
) -> None:
    """Joint model: party spread posteriors per chamber."""
    if "mu_group" not in idata.posterior:
        return

    mu_group_post = idata.posterior["mu_group"].values  # (chain, draw, group)
    group_names = combined_data["group_names"]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        "House Democrat": "#0015BC",
        "House Republican": "#E81B23",
        "Senate Democrat": "#6080D0",
        "Senate Republican": "#F07080",
    }
    styles = {
        "House Democrat": "-",
        "House Republican": "-",
        "Senate Democrat": "--",
        "Senate Republican": "--",
    }

    for i, name in enumerate(group_names):
        samples = mu_group_post[:, :, i].flatten()
        color = colors.get(name, "#888888")
        style = styles.get(name, "-")
        kde = stats.gaussian_kde(samples)
        x = np.linspace(samples.min() - 0.5, samples.max() + 0.5, 300)
        ax.plot(x, kde(x), color=color, linewidth=2, linestyle=style, label=name)
        ax.fill_between(x, kde(x), alpha=0.1, color=color)

    ax.set_xlabel("Group Mean Ideal Point", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Are Parties More Polarized in the House or Senate?\nSolid = House, Dashed = Senate",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / "joint_party_spread.png")


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
    irt_dir = resolve_upstream_dir(
        "05_irt",
        results_root,
        args.run_id,
        Path(args.irt_dir) if args.irt_dir else None,
    )

    with RunContext(
        session=args.session,
        analysis_name="07_hierarchical",
        params=vars(args),
        primer=HIERARCHICAL_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"KS Legislature Hierarchical Bayesian IRT — Session {args.session}")
        print(f"Data:     {data_dir}")
        print(f"EDA:      {eda_dir}")
        print(f"PCA:      {pca_dir}")
        print(f"Flat IRT: {irt_dir}")
        print(f"Output:   {ctx.run_dir}")

        # ── Load data ──
        print_header("LOADING DATA")
        eda_house_path = eda_dir / "data" / "vote_matrix_house_filtered.parquet"
        eda_senate_path = eda_dir / "data" / "vote_matrix_senate_filtered.parquet"
        if not eda_house_path.exists() and not eda_senate_path.exists():
            print("Phase 10 (Hierarchical): skipping — no EDA vote matrices available")
            return
        house_matrix, senate_matrix, full_matrix = load_eda_matrices(eda_dir)
        pca_house_path = pca_dir / "data" / "pc_scores_house.parquet"
        if pca_house_path.exists():
            house_pca, senate_pca = load_pca_scores(pca_dir)
        else:
            print("  PCA scores not available — skipping PCA-informed initialization")
            house_pca, senate_pca = None, None
        rollcalls, legislators = load_metadata(data_dir)

        # Load flat IRT ideal points for comparison
        flat_ip: dict[str, pl.DataFrame | None] = {}
        for ch in ("house", "senate"):
            flat_path = irt_dir / "data" / f"ideal_points_{ch}.parquet"
            if flat_path.exists():
                flat_ip[ch] = pl.read_parquet(flat_path)
                print(f"  Flat IRT ({ch}): {flat_ip[ch].height} legislators loaded")
            else:
                flat_ip[ch] = None
                print(f"  Flat IRT ({ch}): not found at {flat_path}")

        # ── Per-chamber models ──
        per_chamber_results: dict[str, dict] = {}

        for chamber, matrix, pca_scores in [
            ("House", house_matrix, house_pca),
            ("Senate", senate_matrix, senate_pca),
        ]:
            ch = chamber.lower()
            print_header(f"HIERARCHICAL IRT — {chamber}")

            if matrix.height < 5:
                print(f"  Skipping {chamber}: too few legislators ({matrix.height})")
                continue

            # Prepare data with party indices
            data = prepare_hierarchical_data(matrix, legislators, chamber)

            # xi_offset initialization: configurable via --init-strategy
            # Standardized scores approximate the N(0,1) prior on xi_offset,
            # preventing reflection mode-splitting. Same principle as flat IRT (ADR-0023).
            xi_init_vals, init_strat, init_source = resolve_init_source(
                strategy=args.init_strategy,
                slugs=data["leg_slugs"],
                irt_scores=flat_ip[ch],
                pca_scores=pca_scores,
                pca_column="PC1",
            )
            xi_init = xi_init_vals.astype(np.float64) if init_strat != "none" else None
            print(f"  Init: {init_source} (strategy: {init_strat})")

            # Build and sample
            print_header(f"SAMPLING — {chamber}")
            idata, sampling_time = build_per_chamber_model(
                data,
                n_samples=args.n_samples,
                n_tune=args.n_tune,
                n_chains=args.n_chains,
                cores=args.cores,
                xi_offset_initvals=xi_init,
            )

            # Convergence
            convergence = check_hierarchical_convergence(idata, chamber)

            # Extract results
            print_header(f"EXTRACTING RESULTS — {chamber}")
            ideal_points = extract_hierarchical_ideal_points(
                idata, data, legislators, flat_ip=flat_ip[ch]
            )
            group_params = extract_group_params(idata, data)
            icc_df = compute_variance_decomposition(idata, data)

            # Correlation with flat IRT
            flat_corr = float("nan")
            if flat_ip[ch] is not None:
                flat_corr = compute_flat_hier_correlation(ideal_points, flat_ip[ch], chamber)

            # Print group params
            print("\n  Group-level parameters:")
            for row in group_params.iter_rows(named=True):
                print(
                    f"    {row['party']}: mu={row['mu_mean']:+.3f} "
                    f"[{row['mu_hdi_2.5']:+.3f}, {row['mu_hdi_97.5']:+.3f}], "
                    f"sigma={row['sigma_within_mean']:.3f}"
                )

            # Save parquets + NetCDF
            ideal_points.write_parquet(ctx.data_dir / f"hierarchical_ideal_points_{ch}.parquet")
            ctx.export_csv(
                ideal_points,
                f"hierarchical_ideal_points_{ch}.csv",
                f"Hierarchical IRT ideal points for {ch.title()}",
            )
            group_params.write_parquet(ctx.data_dir / f"group_params_{ch}.parquet")
            icc_df.write_parquet(ctx.data_dir / f"variance_decomposition_{ch}.parquet")
            idata.to_netcdf(str(ctx.data_dir / f"idata_{ch}.nc"))

            # Plots
            print_header(f"PLOTS — {chamber}")
            plot_party_posteriors(idata, data, chamber, ctx.plots_dir)
            plot_icc(icc_df, chamber, ctx.plots_dir)
            plot_shrinkage_scatter(ideal_points, chamber, ctx.plots_dir)
            plot_forest(ideal_points, chamber, ctx.plots_dir)
            plot_dispersion(idata, data, chamber, ctx.plots_dir)

            per_chamber_results[chamber] = {
                "data": data,
                "idata": idata,
                "ideal_points": ideal_points,
                "group_params": group_params,
                "icc_df": icc_df,
                "convergence": convergence,
                "sampling_time": sampling_time,
                "flat_corr": flat_corr,
            }

        # ── Joint model (optional) ──
        joint_results: dict | None = None
        if args.run_joint:
            print_header("JOINT CROSS-CHAMBER MODEL")
            try:
                house_data = per_chamber_results["House"]["data"]
                senate_data = per_chamber_results["Senate"]["data"]

                # Init for joint model: concatenate per-chamber init values
                # in the same order as build_joint_graph() (House first, Senate).
                joint_init_parts = []
                for chamber_label, chamber_data, pca_scores, ch_key in [
                    ("House", house_data, house_pca, "house"),
                    ("Senate", senate_data, senate_pca, "senate"),
                ]:
                    vals, _, src = resolve_init_source(
                        strategy=args.init_strategy,
                        slugs=chamber_data["leg_slugs"],
                        irt_scores=flat_ip[ch_key],
                        pca_scores=pca_scores,
                        pca_column="PC1",
                    )
                    joint_init_parts.append(vals.astype(np.float64))
                    print(f"  Joint init ({chamber_label}): {src}")
                joint_xi_init = np.concatenate(joint_init_parts)
                print(
                    f"  Joint init total: {len(joint_xi_init)} params, "
                    f"range [{joint_xi_init.min():.2f}, {joint_xi_init.max():.2f}]"
                )

                joint_idata, combined_data, joint_time = build_joint_model(
                    house_data,
                    senate_data,
                    n_samples=args.n_samples,
                    n_tune=args.n_tune,
                    n_chains=args.n_chains,
                    rollcalls=rollcalls,
                    cores=args.cores,
                    beta_prior=JOINT_BETA,
                    alpha_sigma=2.0,
                    xi_offset_initvals=joint_xi_init,
                )

                joint_convergence = check_hierarchical_convergence(joint_idata, "Joint")

                # Fix sign indeterminacy using per-chamber results as reference
                joint_idata, flipped_chambers = fix_joint_sign_convention(
                    joint_idata, combined_data, per_chamber_results
                )

                # Extract joint ideal points
                joint_ip = extract_hierarchical_ideal_points(
                    joint_idata,
                    combined_data,
                    legislators,
                    flipped_chambers=flipped_chambers,
                )

                # Save
                joint_ip.write_parquet(ctx.data_dir / "hierarchical_ideal_points_joint.parquet")
                ctx.export_csv(
                    joint_ip,
                    "hierarchical_ideal_points_joint.csv",
                    "Joint hierarchical ideal points (cross-chamber)",
                )
                joint_idata.to_netcdf(str(ctx.data_dir / "idata_joint.nc"))

                # Joint plots
                print_header("JOINT PLOTS")
                plot_joint_party_spread(joint_idata, combined_data, ctx.plots_dir)
                plot_forest(joint_ip, "Joint", ctx.plots_dir)

                joint_results = {
                    "idata": joint_idata,
                    "combined_data": combined_data,
                    "ideal_points": joint_ip,
                    "convergence": joint_convergence,
                    "sampling_time": joint_time,
                }

            except Exception as e:
                print(f"\n  WARNING: Joint model failed: {e}")
                print("  Continuing with per-chamber results only.")
                joint_results = None

        # ── Stocking-Lord linking (separate-then-link alternative to joint model) ──
        linking_results: dict | None = None
        if (
            "House" in per_chamber_results
            and "Senate" in per_chamber_results
            and rollcalls is not None
        ):
            print_header("STOCKING-LORD LINKING")
            try:
                house_data_link = per_chamber_results["House"]["data"]
                senate_data_link = per_chamber_results["Senate"]["data"]

                matched_bills, _, _ = _match_bills_across_chambers(
                    house_data_link["vote_ids"],
                    senate_data_link["vote_ids"],
                    rollcalls,
                )
                print(f"  Anchor items: {len(matched_bills)} shared bills")

                house_idata_link = per_chamber_results["House"]["idata"]
                senate_idata_link = per_chamber_results["Senate"]["idata"]

                # Run all four methods as sensitivity check
                all_methods = compare_linking_methods(
                    house_idata_link,
                    senate_idata_link,
                    matched_bills,
                    house_data_link["vote_ids"],
                    senate_data_link["vote_ids"],
                    reference="House",
                )

                # Report sign-filtering diagnostics from first method
                first_result = next(iter(all_methods.values()))
                n_usable = first_result["n_usable"]
                n_sign = first_result["n_sign_disagreement"]
                print(
                    f"  Sign-filtered: {n_usable} usable, "
                    f"{n_sign} dropped (cross-chamber sign disagreement)"
                )

                # Report results
                print("\n  Linking coefficients (Senate → House scale):")
                print(f"  {'Method':<20s} {'A':>8s} {'B':>8s}")
                print(f"  {'─' * 20} {'─' * 8} {'─' * 8}")
                for name, res in all_methods.items():
                    print(f"  {name:<20s} {res['A']:>8.4f} {res['B']:>8.4f}")

                # Use Stocking-Lord as primary
                sl = all_methods["stocking_lord"]
                print("\n  Primary method: Stocking-Lord")
                print(f"  A = {sl['A']:.4f}, B = {sl['B']:.4f}")
                print(
                    f"  Interpretation: Senate xi_linked = "
                    f"{sl['A']:.4f} * xi_senate + {sl['B']:+.4f}"
                )

                # Correlation between methods (xi_senate_linked)
                sl_xi = sl["xi_senate_linked"]
                for name, res in all_methods.items():
                    if name != "stocking_lord":
                        r = float(np.corrcoef(sl_xi, res["xi_senate_linked"])[0, 1])
                        print(f"  S-L vs {name}: r = {r:.6f}")

                # Save linked ideal points
                house_slugs = house_data_link["leg_slugs"]
                senate_slugs = senate_data_link["leg_slugs"]

                linked_rows = []
                for i, slug in enumerate(house_slugs):
                    linked_rows.append(
                        {
                            "legislator_slug": slug,
                            "chamber": "House",
                            "xi_linked": float(sl["xi_house_linked"][i]),
                            "xi_sd": float(sl["xi_house_sd"][i]),
                        }
                    )
                for i, slug in enumerate(senate_slugs):
                    linked_rows.append(
                        {
                            "legislator_slug": slug,
                            "chamber": "Senate",
                            "xi_linked": float(sl["xi_senate_linked"][i]),
                            "xi_sd": float(sl["xi_senate_sd"][i]),
                        }
                    )

                linked_df = pl.DataFrame(linked_rows).sort("xi_linked", descending=True)
                linked_df.write_parquet(ctx.data_dir / "hierarchical_ideal_points_linked.parquet")
                ctx.export_csv(
                    linked_df,
                    "hierarchical_ideal_points_linked.csv",
                    "IRT-linked hierarchical ideal points (cross-chamber)",
                )
                print("\n  Saved: hierarchical_ideal_points_linked.parquet")
                print(f"  {linked_df.height} legislators on common scale")

                # Compare with joint model if available
                if joint_results is not None:
                    joint_ip = joint_results["ideal_points"]
                    for chamber, slugs, xi_linked in [
                        ("House", house_slugs, sl["xi_house_linked"]),
                        ("Senate", senate_slugs, sl["xi_senate_linked"]),
                    ]:
                        slug_to_linked = dict(zip(slugs, xi_linked))
                        joint_slugs = joint_ip.filter(pl.col("legislator_slug").is_in(slugs))[
                            "legislator_slug"
                        ].to_list()
                        joint_xi = joint_ip.filter(pl.col("legislator_slug").is_in(slugs))[
                            "xi_mean"
                        ].to_numpy()
                        linked_xi_matched = np.array([slug_to_linked[s] for s in joint_slugs])
                        r = float(np.corrcoef(joint_xi, linked_xi_matched)[0, 1])
                        print(f"  {chamber} S-L linked vs joint: r = {r:.4f}")

                linking_results = {
                    "all_methods": all_methods,
                    "primary": sl,
                    "matched_bills": matched_bills,
                    "linked_df": linked_df,
                }

            except Exception as e:
                print(f"\n  WARNING: Stocking-Lord linking failed: {e}")
                import traceback

                traceback.print_exc()
                linking_results = None  # noqa: F841

        if not per_chamber_results:
            print("Phase 10 (Hierarchical): skipping — no chambers had sufficient data")
            return

        # ── HTML Report ──
        print_header("HTML REPORT")
        ctx.report.title = f"Kansas Legislature {ctx.session} — Hierarchical Bayesian IRT"
        build_hierarchical_report(
            ctx.report,
            chamber_results=per_chamber_results,
            joint_results=joint_results,
            linking_results=linking_results,
            plots_dir=ctx.plots_dir,
        )

        # ── Filtering manifest ──
        print_header("FILTERING MANIFEST")
        manifest: dict = {
            "analysis": "hierarchical_irt",
            "session": args.session,
            "constants": {
                "HIER_N_SAMPLES": args.n_samples,
                "HIER_N_TUNE": args.n_tune,
                "HIER_N_CHAINS": args.n_chains,
                "HIER_TARGET_ACCEPT": HIER_TARGET_ACCEPT,
                "RHAT_THRESHOLD": RHAT_THRESHOLD,
                "ESS_THRESHOLD": ESS_THRESHOLD,
                "MAX_DIVERGENCES": MAX_DIVERGENCES,
            },
        }

        for chamber in ("House", "Senate"):
            if chamber not in per_chamber_results:
                continue
            ch = chamber.lower()
            res = per_chamber_results[chamber]
            manifest[f"{ch}_n_legislators"] = res["ideal_points"].height
            manifest[f"{ch}_sampling_time_s"] = round(res["sampling_time"], 1)
            manifest[f"{ch}_convergence_ok"] = res["convergence"]["all_ok"]
            manifest[f"{ch}_divergences"] = res["convergence"]["divergences"]
            manifest[f"{ch}_flat_corr"] = round(res["flat_corr"], 4)
            manifest[f"{ch}_icc_mean"] = round(float(res["icc_df"]["icc_mean"][0]), 4)

            for row in res["group_params"].iter_rows(named=True):
                party_key = row["party"].lower()
                manifest[f"{ch}_{party_key}_mu_mean"] = round(row["mu_mean"], 4)
                manifest[f"{ch}_{party_key}_sigma_within"] = round(row["sigma_within_mean"], 4)

        if joint_results is not None:
            manifest["joint_n_legislators"] = joint_results["ideal_points"].height
            manifest["joint_sampling_time_s"] = round(joint_results["sampling_time"], 1)
            manifest["joint_convergence_ok"] = joint_results["convergence"]["all_ok"]
            manifest["joint_n_shared_bills"] = joint_results["combined_data"]["n_shared_bills"]
        else:
            manifest["joint_skipped"] = True

        manifest_path = ctx.run_dir / "filtering_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  Saved: {manifest_path.name}")

        # ── Summary ──
        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")
        print(f"  NetCDF files:   {len(list(ctx.data_dir.glob('*.nc')))}")


if __name__ == "__main__":
    main()
