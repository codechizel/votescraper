"""
Kansas Legislature — Bayesian IRT Ideal Point Estimation (Phase 3)

Covers analytic method 15: 2PL Bayesian IRT on binary vote matrix.
IRT provides what PCA cannot: proper uncertainty intervals via credible intervals,
native missing data handling (absences simply absent from the likelihood), a nonlinear
logistic link function, and bill-level difficulty/discrimination parameters.

Usage:
  uv run python analysis/irt.py [--session 2025-26] [--eda-dir ...] [--pca-dir ...]
      [--n-samples 2000] [--n-tune 1000] [--n-chains 2] [--skip-sensitivity]

Outputs (in results/<session>/irt/<date>/):
  - data/:   Parquet files (ideal points, bill params) + NetCDF (full posterior)
  - plots/:  PNG visualizations (forest, discrimination, traces, PPC, sensitivity)
  - filtering_manifest.json, run_info.json, run_log.txt
  - irt_report.html
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
from matplotlib.patches import Patch
from scipy import stats

try:
    from analysis.run_context import RunContext
except ModuleNotFoundError:
    from run_context import RunContext

try:
    from analysis.irt_report import build_irt_report
except ModuleNotFoundError:
    from irt_report import build_irt_report  # type: ignore[no-redef]

# ── Primer ───────────────────────────────────────────────────────────────────
# Written to results/<session>/irt/README.md by RunContext on each run.

IRT_PRIMER = """\
# Bayesian IRT Ideal Point Estimation

## Purpose

Bayesian Item Response Theory (IRT) estimates each legislator's position on a
latent ideological spectrum, with full posterior uncertainty. Unlike PCA (which
gives point estimates), IRT produces 95% credible intervals: "we're 95% sure
this legislator is between positions 0.15 and 0.45."

IRT also estimates bill-level parameters: **difficulty** (where on the spectrum
the vote "flips") and **discrimination** (how sharply the vote separates liberals
from conservatives). This is the canonical baseline per the analytic workflow
rules: "1D Bayesian IRT on Yea/Nay only."

Covers analytic method 15 from `Analytic_Methods/`.

## Method

### 2-Parameter Logistic (2PL) IRT Model

```
P(Yea | xi_i, alpha_j, beta_j) = logit^-1(beta_j * xi_i - alpha_j)

xi_i    ~ Normal(0, 1)          -- legislator ideal point
alpha_j ~ Normal(0, 5)          -- bill difficulty (diffuse)
beta_j  ~ Normal(0, 1)           -- discrimination (unconstrained; anchors identify sign)
```

**Identification** via anchor method:
- Conservative anchor (xi = +1): highest PCA PC1 score with >= 50% participation
- Liberal anchor (xi = -1): lowest PCA PC1 score with >= 50% participation

**Missing data:** Absences are handled natively — they are simply absent from
the likelihood. No imputation needed (unlike PCA).

### Pipeline

1. Load filtered vote matrices from EDA, PCA scores for anchor selection
2. Convert to long format, drop nulls (absences)
3. Build 2PL PyMC model with anchor constraints
4. Sample with NUTS (2000 draws, 1000 tune, 2 chains)
5. Convergence diagnostics (R-hat, ESS, divergences, E-BFMI)
6. Extract posterior summaries (ideal points, bill parameters)
7. Generate plots (forest, discrimination, traces, PPC)
8. Compare with PCA PC1 scores
9. Posterior predictive checks
10. Holdout validation (in-sample prediction)
11. Sensitivity analysis (10% minority threshold re-run)

## Inputs

Reads from `results/<session>/eda/latest/data/`:
- `vote_matrix_house_filtered.parquet` — House binary vote matrix (EDA-filtered)
- `vote_matrix_senate_filtered.parquet` — Senate binary vote matrix (EDA-filtered)
- `vote_matrix_full.parquet` — Full unfiltered vote matrix (for sensitivity)

Reads from `results/<session>/pca/latest/data/`:
- `pc_scores_house.parquet` — House PCA scores (for anchor selection + comparison)
- `pc_scores_senate.parquet` — Senate PCA scores

Reads from `data/{legislature}_{start}-{end}/`:
- `{output_name}_rollcalls.csv` — Roll call metadata
- `{output_name}_legislators.csv` — Legislator metadata

## Outputs

All outputs land in `results/<session>/irt/<date>/`:

### `data/` — Parquet intermediates + NetCDF posteriors

| File | Description |
|------|-------------|
| `ideal_points_house.parquet` | House legislator ideal points + HDI + metadata |
| `ideal_points_senate.parquet` | Senate ideal points |
| `bill_params_house.parquet` | House bill difficulty/discrimination + metadata |
| `bill_params_senate.parquet` | Senate bill parameters |
| `idata_house.nc` | Full posterior (ArviZ NetCDF, ~50-100MB) |
| `idata_senate.nc` | Full Senate posterior |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `forest_house.png` | Ideal points with 95% HDI, party-colored, sorted |
| `forest_senate.png` | Senate forest plot |
| `discrimination_house.png` | Distribution of bill discrimination parameters |
| `discrimination_senate.png` | Senate discrimination histogram |
| `irt_vs_pca_house.png` | IRT xi_mean vs PCA PC1 scatter with Pearson r |
| `irt_vs_pca_senate.png` | Senate IRT vs PCA |
| `trace_house.png` | Trace plots for 5 representative ideal points |
| `trace_senate.png` | Senate traces |
| `ppc_yea_rate_house.png` | Posterior predictive Yea rate vs observed |
| `ppc_yea_rate_senate.png` | Senate PPC |
| `sensitivity_xi_house.png` | Default vs sensitivity ideal points scatter |
| `sensitivity_xi_senate.png` | Senate sensitivity scatter |

### Root files

| File | Description |
|------|-------------|
| `filtering_manifest.json` | Model params, convergence, validation results |
| `run_info.json` | Git commit, timestamp, Python version, parameters |
| `run_log.txt` | Full console output from the run |
| `irt_report.html` | Self-contained HTML report with all tables and figures |

## Interpretation Guide

- **Forest plot points**: Posterior mean ideal point. Positive = conservative.
- **Forest plot bars**: 95% HDI. Overlapping intervals = cannot distinguish.
- **Wide intervals**: Uncertain — few votes or inconsistent voting pattern.
- **Discrimination (beta)**: |beta| > 1.5 = strongly partisan vote. Near 0 = non-informative.
  Positive beta = conservatives favor Yea. Negative beta = liberals favor Yea.
- **PCA correlation**: r > 0.95 expected for a well-behaved 1D model.
- **Sensitivity**: r > 0.95 between default and 10% threshold = robust.

## Caveats

- 1D model cannot capture multi-dimensional structure (e.g., Tyson's contrarianism).
- Discrimination (beta) can be positive or negative. The sign indicates which end
  of the ideological spectrum favors Yea. Anchors provide sign identification.
- In-sample holdout validation is not a true out-of-sample test. PPC provides the
  proper Bayesian validation.
- MCMC runtime: ~5-20 min per chamber depending on hardware and sample count.
"""

# ── Constants ────────────────────────────────────────────────────────────────
# Explicit, named constants per the analytic-workflow rules.

DEFAULT_N_SAMPLES = 2000
DEFAULT_N_TUNE = 1000
DEFAULT_N_CHAINS = 2
TARGET_ACCEPT = 0.9
RANDOM_SEED = 42

MINORITY_THRESHOLD = 0.025  # Default: 2.5% (matches EDA)
SENSITIVITY_THRESHOLD = 0.10  # Sensitivity: 10% (per workflow rules)
MIN_VOTES = 20  # Minimum substantive votes per legislator
HOLDOUT_FRACTION = 0.20  # Random 20% of observed cells
HOLDOUT_SEED = 42
MIN_PARTICIPATION_FOR_ANCHOR = 0.50  # Anchors must have >= 50% participation

PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC"}

# Joint model defaults
JOINT_TARGET_ACCEPT = 0.95
JOINT_N_CHAINS = 4

# Convergence thresholds
RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400
MAX_DIVERGENCES = 10

# Plot constants
N_TRACE_LEGISLATORS = 5
TOP_DISCRIMINATING = 15


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Bayesian IRT")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="MCMC samples per chain",
    )
    parser.add_argument(
        "--n-tune",
        type=int,
        default=DEFAULT_N_TUNE,
        help="MCMC tuning samples (discarded)",
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=DEFAULT_N_CHAINS,
        help="Number of MCMC chains",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip sensitivity analysis (halves runtime)",
    )
    parser.add_argument(
        "--skip-joint",
        action="store_true",
        help="Skip joint cross-chamber model",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    width = 80
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Phase 1: Load Data ──────────────────────────────────────────────────────


def load_eda_matrices(
    eda_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load filtered vote matrices from the EDA phase output."""
    house = pl.read_parquet(eda_dir / "data" / "vote_matrix_house_filtered.parquet")
    senate = pl.read_parquet(eda_dir / "data" / "vote_matrix_senate_filtered.parquet")
    full = pl.read_parquet(eda_dir / "data" / "vote_matrix_full.parquet")
    return house, senate, full


def load_pca_scores(pca_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load PCA scores for anchor selection and comparison."""
    house = pl.read_parquet(pca_dir / "data" / "pc_scores_house.parquet")
    senate = pl.read_parquet(pca_dir / "data" / "pc_scores_senate.parquet")
    return house, senate


def load_metadata(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load rollcall and legislator CSVs for metadata enrichment."""
    prefix = data_dir.name
    rollcalls = pl.read_csv(data_dir / f"{prefix}_rollcalls.csv")
    legislators = pl.read_csv(data_dir / f"{prefix}_legislators.csv")
    return rollcalls, legislators


# ── Joint Cross-Chamber Functions ───────────────────────────────────────────


def build_joint_vote_matrix(
    house_matrix: pl.DataFrame,
    senate_matrix: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
) -> tuple[pl.DataFrame, dict]:
    """Build a joint vote matrix linking House and Senate via shared bills.

    For each bill_number appearing in both chambers' filtered matrices:
    - Prefer the vote_id with "Final Action" or "Emergency Final Action" motion
    - If multiple, pick the latest chronologically
    - Create ONE matched column: House members get their House value, Senate get theirs

    Bridging legislators (serving in both chambers) are merged into a single row
    using the House slug as canonical.

    Returns (joint_matrix, mapping_info).
    """
    slug_col = "legislator_slug"
    house_vote_ids = [c for c in house_matrix.columns if c != slug_col]
    senate_vote_ids = [c for c in senate_matrix.columns if c != slug_col]

    # Build vote_id → bill_number mapping from rollcalls
    rc = rollcalls.select("vote_id", "bill_number", "motion").filter(
        pl.col("vote_id").is_in(house_vote_ids + senate_vote_ids)
    )
    vid_to_bill = dict(zip(rc["vote_id"].to_list(), rc["bill_number"].to_list()))
    vid_to_motion = dict(zip(rc["vote_id"].to_list(), rc["motion"].to_list()))

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

    # Find shared bills
    shared_bills = set(house_bill_vids.keys()) & set(senate_bill_vids.keys())
    print(f"  Joint: {len(shared_bills)} shared bills found")

    def _pick_best_vid(vids: list[str]) -> str:
        """Pick the best vote_id: prefer Final Action, then latest chronologically."""
        final_vids = [
            v
            for v in vids
            if vid_to_motion.get(v, "").lower()
            in (
                "final action",
                "emergency final action",
            )
        ]
        candidates = final_vids if final_vids else vids
        return sorted(candidates)[-1]  # Latest chronologically (vote_id encodes timestamp)

    # Build matched columns
    matched_bills: list[dict] = []
    matched_col_names: list[str] = []
    house_used: set[str] = set()
    senate_used: set[str] = set()

    for bill in sorted(shared_bills):
        h_vid = _pick_best_vid(house_bill_vids[bill])
        s_vid = _pick_best_vid(senate_bill_vids[bill])
        col_name = f"matched_{bill}"
        matched_bills.append(
            {
                "bill_number": bill,
                "house_vote_id": h_vid,
                "senate_vote_id": s_vid,
                "matched_col": col_name,
            }
        )
        matched_col_names.append(col_name)
        house_used.add(h_vid)
        senate_used.add(s_vid)

    house_only_vids = [v for v in house_vote_ids if v not in house_used]
    senate_only_vids = [v for v in senate_vote_ids if v not in senate_used]

    # Identify bridging legislators (same person with both rep_ and sen_ slugs)
    # Match by name from legislators table
    house_legs = legislators.filter(pl.col("chamber") == "House").select("slug", "full_name")
    senate_legs = legislators.filter(pl.col("chamber") == "Senate").select("slug", "full_name")
    bridging: list[dict] = []
    for h_row in house_legs.iter_rows(named=True):
        match = senate_legs.filter(pl.col("full_name") == h_row["full_name"])
        if match.height > 0:
            bridging.append(
                {
                    "house_slug": h_row["slug"],
                    "senate_slug": match["slug"][0],
                    "full_name": h_row["full_name"],
                }
            )

    print(f"  Joint: {len(bridging)} bridging legislators found")
    for b in bridging:
        print(f"    {b['full_name']}: {b['house_slug']} + {b['senate_slug']}")

    # Build mapping from senate_slug → house_slug for bridging
    senate_to_house = {b["senate_slug"]: b["house_slug"] for b in bridging}

    # Construct the joint matrix
    # Start with all House legislators
    house_slugs = house_matrix[slug_col].to_list()
    senate_slugs = senate_matrix[slug_col].to_list()

    # All rows: House legislators + non-bridging Senate legislators
    all_slugs = list(house_slugs)
    for s_slug in senate_slugs:
        if s_slug not in senate_to_house:
            all_slugs.append(s_slug)

    # Build row data
    joint_data: dict[str, list] = {slug_col: all_slugs}

    # Matched columns
    house_slug_set = set(house_slugs)
    senate_slug_set = set(senate_slugs)

    for info in matched_bills:
        col_name = info["matched_col"]
        h_vid = info["house_vote_id"]
        s_vid = info["senate_vote_id"]

        # Build lookup dicts
        h_vals = dict(zip(house_matrix[slug_col].to_list(), house_matrix[h_vid].to_list()))
        s_vals = dict(zip(senate_matrix[slug_col].to_list(), senate_matrix[s_vid].to_list()))

        col_data = []
        for slug in all_slugs:
            if slug in house_slug_set:
                val = h_vals.get(slug)
                # If bridging, also check senate vote
                if slug in [b["house_slug"] for b in bridging]:
                    s_slug = next(b["senate_slug"] for b in bridging if b["house_slug"] == slug)
                    s_val = s_vals.get(s_slug)
                    # Prefer House vote, fall back to Senate
                    val = val if val is not None else s_val
                col_data.append(val)
            elif slug in senate_slug_set:
                col_data.append(s_vals.get(slug))
            else:
                col_data.append(None)
        joint_data[col_name] = col_data

    # House-only columns
    for vid in house_only_vids:
        h_vals = dict(zip(house_matrix[slug_col].to_list(), house_matrix[vid].to_list()))
        col_data = []
        for slug in all_slugs:
            if slug in house_slug_set:
                col_data.append(h_vals.get(slug))
            elif slug in [b["house_slug"] for b in bridging]:
                col_data.append(h_vals.get(slug))
            else:
                col_data.append(None)
        joint_data[vid] = col_data

    # Senate-only columns
    for vid in senate_only_vids:
        s_vals = dict(zip(senate_matrix[slug_col].to_list(), senate_matrix[vid].to_list()))
        col_data = []
        for slug in all_slugs:
            if slug in senate_to_house.values():
                # Bridging legislator: look up their senate slug
                s_slug = next(b["senate_slug"] for b in bridging if b["house_slug"] == slug)
                col_data.append(s_vals.get(s_slug))
            elif slug in senate_slug_set:
                col_data.append(s_vals.get(slug))
            else:
                col_data.append(None)
        joint_data[vid] = col_data

    joint_matrix = pl.DataFrame(joint_data)

    n_cols = len(joint_matrix.columns) - 1
    print(
        f"  Joint matrix: {joint_matrix.height} legislators x {n_cols} votes "
        f"({len(matched_col_names)} matched, {len(house_only_vids)} house-only, "
        f"{len(senate_only_vids)} senate-only)"
    )

    mapping_info = {
        "matched_bills": matched_bills,
        "bridging_legislators": bridging,
        "house_only_vote_ids": house_only_vids,
        "senate_only_vote_ids": senate_only_vids,
        "matched_col_names": matched_col_names,
        "senate_to_house": senate_to_house,
    }

    return joint_matrix, mapping_info


def run_joint_pca_for_anchors(
    joint_matrix: pl.DataFrame,
    legislators: pl.DataFrame,
) -> pl.DataFrame:
    """Run PCA on matched-bill columns only for anchor selection.

    Matched columns are dense (~98% fill). Row-mean impute the small amount
    of missingness, run PCA, return DataFrame compatible with select_anchors().
    """
    from sklearn.decomposition import PCA

    slug_col = "legislator_slug"
    matched_cols = [c for c in joint_matrix.columns if c.startswith("matched_")]

    if not matched_cols:
        msg = "No matched columns found in joint matrix"
        raise ValueError(msg)

    # Extract matched-column data
    slugs = joint_matrix[slug_col].to_list()
    data = joint_matrix.select(matched_cols).to_numpy().astype(float)

    # Row-mean impute
    for i in range(data.shape[0]):
        row = data[i]
        nan_mask = np.isnan(row)
        if nan_mask.any() and not nan_mask.all():
            data[i, nan_mask] = np.nanmean(row)

    # Standardize columns
    col_means = np.nanmean(data, axis=0)
    col_stds = np.nanstd(data, axis=0)
    col_stds[col_stds == 0] = 1.0
    data_std = (data - col_means) / col_stds

    # PCA
    pca = PCA(n_components=1, random_state=RANDOM_SEED)
    scores = pca.fit_transform(data_std)

    # Build output DataFrame
    result = pl.DataFrame(
        {
            "legislator_slug": slugs,
            "PC1": scores[:, 0].tolist(),
        }
    )

    # Join legislator metadata
    meta = legislators.select("slug", "full_name", "party").unique(subset=["slug"])
    result = result.join(meta, left_on="legislator_slug", right_on="slug", how="left")

    print(f"  Joint PCA: {result.height} legislators, {len(matched_cols)} matched bills")
    print(f"  PC1 range: [{result['PC1'].min():.3f}, {result['PC1'].max():.3f}]")

    return result


def unmerge_bridging_legislators(
    joint_ideal_points: pl.DataFrame,
    mapping_info: dict,
    legislators: pl.DataFrame,
) -> pl.DataFrame:
    """Expand bridging legislators back to per-chamber slugs.

    Each bridging legislator in the joint model has a single row (using their
    House slug). This duplicates that row with the original Senate slug and
    correct chamber metadata.
    """
    bridging = mapping_info["bridging_legislators"]
    if not bridging:
        return joint_ideal_points

    house_to_senate = {b["house_slug"]: b["senate_slug"] for b in bridging}
    bridging_house_slugs = set(house_to_senate.keys())

    # Separate bridging from non-bridging rows
    non_bridging = joint_ideal_points.filter(~pl.col("legislator_slug").is_in(bridging_house_slugs))

    # Build expanded rows for bridging legislators
    expanded_rows = []
    for row in joint_ideal_points.filter(
        pl.col("legislator_slug").is_in(bridging_house_slugs)
    ).iter_rows(named=True):
        # House version (keep as-is but ensure chamber is House)
        house_row = dict(row)
        house_row["chamber"] = "House"
        expanded_rows.append(house_row)

        # Senate version
        senate_slug = house_to_senate[row["legislator_slug"]]
        senate_row = dict(row)
        senate_row["legislator_slug"] = senate_slug
        senate_row["chamber"] = "Senate"
        # Update district from legislators table
        sen_meta = legislators.filter(pl.col("slug") == senate_slug)
        if sen_meta.height > 0:
            senate_row["district"] = sen_meta["district"][0]
        expanded_rows.append(senate_row)

    if expanded_rows:
        expanded_df = pl.DataFrame(expanded_rows, schema=joint_ideal_points.schema)
        result = pl.concat([non_bridging, expanded_df])
    else:
        result = non_bridging

    print(
        f"  Unmerged: {joint_ideal_points.height} → {result.height} rows "
        f"({len(bridging)} bridging legislators duplicated)"
    )
    return result.sort("xi_mean", descending=True)


def plot_joint_vs_chamber(
    joint_unmerged: pl.DataFrame,
    per_chamber_results: dict[str, dict],
    mapping_info: dict,
    out_dir: Path,
) -> dict[str, float]:
    """Scatter plot: joint ideal points (x) vs per-chamber ideal points (y).

    One subplot per chamber. Color by party, highlight bridging legislators.
    Annotate Pearson r.

    Returns {chamber: pearson_r} dict.
    """
    bridging_slugs = set()
    for b in mapping_info["bridging_legislators"]:
        bridging_slugs.add(b["house_slug"])
        bridging_slugs.add(b["senate_slug"])

    chambers = [c for c in per_chamber_results if c != "Joint"]
    n_chambers = len(chambers)
    if n_chambers == 0:
        return {}

    fig, axes = plt.subplots(1, n_chambers, figsize=(8 * n_chambers, 8))
    if n_chambers == 1:
        axes = [axes]

    correlations: dict[str, float] = {}

    for ax, chamber in zip(axes, chambers):
        chamber_ip = per_chamber_results[chamber]["ideal_points"]

        # Merge joint and per-chamber on slug
        merged = joint_unmerged.select("legislator_slug", "xi_mean", "party").join(
            chamber_ip.select("legislator_slug", pl.col("xi_mean").alias("xi_chamber")),
            on="legislator_slug",
            how="inner",
        )

        if merged.height < 3:
            correlations[chamber] = float("nan")
            continue

        joint_arr = merged["xi_mean"].to_numpy()
        chamber_arr = merged["xi_chamber"].to_numpy()
        pearson_r = float(np.corrcoef(joint_arr, chamber_arr)[0, 1])
        correlations[chamber] = pearson_r

        # Plot by party
        for party, color in PARTY_COLORS.items():
            subset = merged.filter(pl.col("party") == party)
            is_bridge = subset["legislator_slug"].is_in(bridging_slugs)
            regular = subset.filter(~is_bridge)
            bridge = subset.filter(is_bridge)

            if regular.height > 0:
                ax.scatter(
                    regular["xi_mean"].to_numpy(),
                    regular["xi_chamber"].to_numpy(),
                    c=color,
                    s=40,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                    label=party,
                )
            if bridge.height > 0:
                ax.scatter(
                    bridge["xi_mean"].to_numpy(),
                    bridge["xi_chamber"].to_numpy(),
                    c=color,
                    s=120,
                    alpha=0.9,
                    edgecolors="gold",
                    linewidth=2,
                    marker="D",
                    label=f"{party} (bridging)",
                )

        # Identity line
        lims = [
            min(joint_arr.min(), chamber_arr.min()) - 0.3,
            max(joint_arr.max(), chamber_arr.max()) + 0.3,
        ]
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.set_xlabel("Joint Model Ideal Point")
        ax.set_ylabel(f"{chamber} Model Ideal Point")
        ax.set_title(f"{chamber}: Joint vs Per-Chamber (r = {pearson_r:.4f})")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Joint vs Per-Chamber Ideal Points", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir / "joint_vs_chamber.png")

    for chamber, r in correlations.items():
        print(f"  {chamber}: Joint vs per-chamber Pearson r = {r:.4f}")

    return correlations


def equate_chambers(
    per_chamber_results: dict[str, dict],
    mapping_info: dict,
    legislators: pl.DataFrame,
    out_dir: Path,
) -> dict:
    """Place House and Senate legislators on a common scale via test equating.

    Uses a hybrid approach:
    - **A (scale)** from mean/sigma on shared bill discrimination parameters:
      A = SD(beta_senate) / SD(beta_house) over concordant shared bills.
    - **B (location)** from bridging legislators who served in both chambers:
      B = mean(xi_house) - A * mean(xi_senate) for bridging legislators.
      Falls back to median difficulty if no bridging legislators exist.

    Convention: House scale is the reference. Senate ideal points are
    transformed to the House scale via xi_equated = A * xi_senate + B.

    Returns dict with:
      - equated_ideal_points: pl.DataFrame (all legislators on common scale)
      - transformation: dict with A, B, n_usable_bills, method
      - correlations: dict {chamber: pearson_r} comparing equated vs per-chamber
    """
    house_result = per_chamber_results["House"]
    senate_result = per_chamber_results["Senate"]
    house_bp = house_result["bill_params"]
    senate_bp = senate_result["bill_params"]
    house_ip = house_result["ideal_points"]
    senate_ip = senate_result["ideal_points"]

    matched_bills = mapping_info["matched_bills"]
    bridging = mapping_info["bridging_legislators"]

    # Match shared bill parameters from each chamber
    h_beta = {}
    for row in house_bp.iter_rows(named=True):
        h_beta[row["vote_id"]] = row["beta_mean"]

    s_beta = {}
    for row in senate_bp.iter_rows(named=True):
        s_beta[row["vote_id"]] = row["beta_mean"]

    # Collect paired betas for concordant shared bills
    paired_betas_h = []
    paired_betas_s = []
    for mb in matched_bills:
        h_vid = mb["house_vote_id"]
        s_vid = mb["senate_vote_id"]
        if h_vid in h_beta and s_vid in s_beta:
            bh = h_beta[h_vid]
            bs = s_beta[s_vid]
            if bh * bs > 0:  # same sign = concordant
                paired_betas_h.append(bh)
                paired_betas_s.append(bs)

    n_usable = len(paired_betas_h)
    print(f"  Shared bills with concordant beta: {n_usable} / {len(matched_bills)}")

    if n_usable < 5:
        print("  WARNING: Too few concordant shared bills for reliable equating")

    bh_arr = np.array(paired_betas_h)
    bs_arr = np.array(paired_betas_s)

    # A (scale factor) from discrimination ratio: SD(beta_S) / SD(beta_H)
    sd_bh = float(np.std(bh_arr))
    sd_bs = float(np.std(bs_arr))

    if sd_bh < 1e-10:
        print("  WARNING: House beta SD near zero, falling back to A=1.0")
        A = 1.0
    else:
        A = sd_bs / sd_bh

    # B (location shift) from bridging legislators
    # Each bridging legislator has per-chamber ideal points from the separate
    # models. The direct relationship: xi_H = A * xi_S + B gives B directly.
    h_slugs_to_xi = dict(zip(house_ip["legislator_slug"].to_list(), house_ip["xi_mean"].to_list()))
    s_slugs_to_xi = dict(
        zip(senate_ip["legislator_slug"].to_list(), senate_ip["xi_mean"].to_list())
    )

    bridge_h_xi = []
    bridge_s_xi = []
    for b in bridging:
        h_xi = h_slugs_to_xi.get(b["house_slug"])
        s_xi = s_slugs_to_xi.get(b["senate_slug"])
        if h_xi is not None and s_xi is not None:
            bridge_h_xi.append(h_xi)
            bridge_s_xi.append(s_xi)
            print(f"  Bridging: {b['full_name']:20s}  House xi={h_xi:+.3f}, Senate xi={s_xi:+.3f}")

    if bridge_h_xi:
        # B = mean(xi_H) - A * mean(xi_S)
        B = float(np.mean(bridge_h_xi) - A * np.mean(bridge_s_xi))
        b_method = "bridging_legislators"
        print(f"  B from {len(bridge_h_xi)} bridging legislators")
    else:
        B = 0.0
        b_method = "fallback_zero"
        print("  WARNING: No bridging legislators found; B = 0.0")

    print(f"  Equating: A = {A:.4f}, B = {B:.4f}")
    print(f"  Interpretation: xi_equated = {A:.4f} * xi_senate + {B:.4f}")

    # Transform Senate ideal points to House scale
    house_ip_out = house_ip.clone()
    senate_ip_out = senate_ip.clone()

    senate_equated = senate_ip_out.with_columns(
        (pl.col("xi_mean") * A + B).alias("xi_mean"),
        (pl.col("xi_sd") * abs(A)).alias("xi_sd"),
        (pl.col("xi_hdi_2.5") * A + B).alias("xi_hdi_2.5"),
        (pl.col("xi_hdi_97.5") * A + B).alias("xi_hdi_97.5"),
    )

    # If A is negative, HDI bounds swap
    if A < 0:
        senate_equated = (
            senate_equated.with_columns(
                pl.col("xi_hdi_2.5").alias("_tmp_hi"),
                pl.col("xi_hdi_97.5").alias("_tmp_lo"),
            )
            .with_columns(
                pl.col("_tmp_lo").alias("xi_hdi_2.5"),
                pl.col("_tmp_hi").alias("xi_hdi_97.5"),
            )
            .drop("_tmp_lo", "_tmp_hi")
        )

    # Combine
    equated = pl.concat([house_ip_out, senate_equated]).sort("xi_mean", descending=True)

    # Compute correlations: equated vs original per-chamber scores
    correlations: dict[str, float] = {}
    for chamber, orig_ip in [("House", house_ip), ("Senate", senate_ip)]:
        merged = equated.select("legislator_slug", "xi_mean").join(
            orig_ip.select("legislator_slug", pl.col("xi_mean").alias("xi_orig")),
            on="legislator_slug",
            how="inner",
        )
        if merged.height >= 3:
            r = float(np.corrcoef(merged["xi_mean"].to_numpy(), merged["xi_orig"].to_numpy())[0, 1])
            correlations[chamber] = r
            print(f"  {chamber}: equated vs per-chamber r = {r:.4f}")

    # Plot
    _plot_equated_vs_chamber(equated, per_chamber_results, mapping_info, out_dir)

    # Plot equated forest
    plot_forest(equated, "Joint", out_dir)

    return {
        "equated_ideal_points": equated,
        "transformation": {
            "A": A,
            "B": B,
            "n_usable_bills": n_usable,
            "n_total_shared": len(matched_bills),
            "n_bridging": len(bridge_h_xi),
            "b_method": b_method,
            "method": "discrimination_ratio_plus_bridging",
        },
        "correlations": correlations,
    }


def _plot_equated_vs_chamber(
    equated: pl.DataFrame,
    per_chamber_results: dict[str, dict],
    mapping_info: dict,
    out_dir: Path,
) -> None:
    """Scatter plot: equated ideal points vs per-chamber ideal points."""
    bridging_slugs = set()
    for b in mapping_info["bridging_legislators"]:
        bridging_slugs.add(b["house_slug"])
        bridging_slugs.add(b["senate_slug"])

    chambers = [c for c in per_chamber_results if c != "Joint"]
    fig, axes = plt.subplots(1, len(chambers), figsize=(8 * len(chambers), 8))
    if len(chambers) == 1:
        axes = [axes]

    for ax, chamber in zip(axes, chambers):
        chamber_ip = per_chamber_results[chamber]["ideal_points"]
        merged = equated.select("legislator_slug", "xi_mean", "party").join(
            chamber_ip.select("legislator_slug", pl.col("xi_mean").alias("xi_chamber")),
            on="legislator_slug",
            how="inner",
        )

        if merged.height < 3:
            continue

        eq_arr = merged["xi_mean"].to_numpy()
        ch_arr = merged["xi_chamber"].to_numpy()
        r = float(np.corrcoef(eq_arr, ch_arr)[0, 1])

        for party, color in PARTY_COLORS.items():
            subset = merged.filter(pl.col("party") == party)
            is_bridge = subset["legislator_slug"].is_in(bridging_slugs)
            regular = subset.filter(~is_bridge)
            bridge = subset.filter(is_bridge)

            if regular.height > 0:
                ax.scatter(
                    regular["xi_mean"].to_numpy(),
                    regular["xi_chamber"].to_numpy(),
                    c=color,
                    s=40,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                    label=party,
                )
            if bridge.height > 0:
                ax.scatter(
                    bridge["xi_mean"].to_numpy(),
                    bridge["xi_chamber"].to_numpy(),
                    c=color,
                    s=120,
                    alpha=0.9,
                    edgecolors="gold",
                    linewidth=2,
                    marker="D",
                    label=f"{party} (bridging)",
                )

        # Identity line
        lims = [
            min(eq_arr.min(), ch_arr.min()) - 0.3,
            max(eq_arr.max(), ch_arr.max()) + 0.3,
        ]
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.set_xlabel("Equated Ideal Point (House Scale)")
        ax.set_ylabel(f"{chamber} Per-Chamber Ideal Point")
        ax.set_title(f"{chamber}: Equated vs Per-Chamber (r = {r:.4f})")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Test-Equated vs Per-Chamber Ideal Points (Mean/Sigma Method)",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "joint_vs_chamber.png")


# ── Phase 2: Prepare IRT Data ───────────────────────────────────────────────


def prepare_irt_data(
    matrix: pl.DataFrame,
    chamber: str,
) -> dict:
    """Convert wide vote matrix to long format for IRT model.

    Drops null rows (absences handled natively by being absent from likelihood).
    Creates integer index mappings for PyMC.

    Returns dict with leg_idx, vote_idx, y, slug/id lists, and counts.
    """
    slug_col = "legislator_slug"
    slugs = matrix[slug_col].to_list()
    vote_ids = [c for c in matrix.columns if c != slug_col]

    # Unpivot to long format
    long = matrix.unpivot(
        on=vote_ids,
        index=slug_col,
        variable_name="vote_id",
        value_name="vote",
    )

    # Drop nulls (absences — handled natively by IRT)
    long = long.drop_nulls(subset=["vote"])

    # Create integer index mappings
    slug_to_idx = {s: i for i, s in enumerate(slugs)}
    vote_to_idx = {v: i for i, v in enumerate(vote_ids)}

    long = long.with_columns(
        [
            pl.col(slug_col).replace_strict(slug_to_idx).alias("leg_idx"),
            pl.col("vote_id").replace_strict(vote_to_idx).alias("vote_idx"),
        ]
    )

    print(f"  {chamber}: {len(slugs)} legislators x {len(vote_ids)} votes")
    print(
        f"  Observed cells: {long.height:,} / {len(slugs) * len(vote_ids):,} "
        f"({100 * long.height / (len(slugs) * len(vote_ids)):.1f}%)"
    )
    print(f"  Yea rate: {long['vote'].mean():.3f}")

    return {
        "leg_idx": long["leg_idx"].to_numpy().astype(np.int64),
        "vote_idx": long["vote_idx"].to_numpy().astype(np.int64),
        "y": long["vote"].to_numpy().astype(np.int64),
        "n_legislators": len(slugs),
        "n_votes": len(vote_ids),
        "n_obs": long.height,
        "leg_slugs": slugs,
        "vote_ids": vote_ids,
    }


def select_anchors(
    pca_scores: pl.DataFrame,
    matrix: pl.DataFrame,
    chamber: str,
) -> tuple[int, str, int, str]:
    """Select conservative and liberal anchors from PCA PC1 extremes.

    Guards: anchor must have >= 50% participation in the filtered matrix.

    Returns (cons_idx, cons_slug, lib_idx, lib_slug).
    """
    slug_col = "legislator_slug"
    vote_cols = [c for c in matrix.columns if c != slug_col]
    n_votes = len(vote_cols)
    slugs = matrix[slug_col].to_list()

    # Compute participation rates
    participation = {}
    for row in matrix.iter_rows(named=True):
        slug = row[slug_col]
        n_present = sum(1 for v in vote_cols if row[v] is not None)
        participation[slug] = n_present / n_votes if n_votes > 0 else 0.0

    # Filter PCA scores to legislators in the matrix with sufficient participation
    eligible = (
        pca_scores.filter(pl.col("legislator_slug").is_in(slugs))
        .with_columns(
            pl.col("legislator_slug").replace_strict(participation).alias("participation")
        )
        .filter(pl.col("participation") >= MIN_PARTICIPATION_FOR_ANCHOR)
    )

    # Sort by PC1 to find extremes
    sorted_scores = eligible.sort("PC1", descending=True)

    cons_slug = sorted_scores["legislator_slug"][0]
    cons_name = sorted_scores["full_name"][0]
    cons_pc1 = sorted_scores["PC1"][0]

    lib_slug = sorted_scores["legislator_slug"][-1]
    lib_name = sorted_scores["full_name"][-1]
    lib_pc1 = sorted_scores["PC1"][-1]

    cons_idx = slugs.index(cons_slug)
    lib_idx = slugs.index(lib_slug)

    print(f"  Conservative anchor: {cons_name} ({cons_slug}), PC1={cons_pc1:+.3f}")
    print(f"  Liberal anchor:      {lib_name} ({lib_slug}), PC1={lib_pc1:+.3f}")

    return cons_idx, cons_slug, lib_idx, lib_slug


# ── Phase 3: Build and Sample PyMC Model ────────────────────────────────────


def build_and_sample(
    data: dict,
    anchors: list[tuple[int, float]],
    n_samples: int,
    n_tune: int,
    n_chains: int,
    target_accept: float = TARGET_ACCEPT,
) -> tuple[az.InferenceData, float]:
    """Build 2PL IRT model with anchor constraints and sample with NUTS.

    Args:
        data: IRT data dict from prepare_irt_data().
        anchors: List of (legislator_index, fixed_value) pairs. Typically 2 for
            per-chamber models [(cons_idx, +1.0), (lib_idx, -1.0)] or 4 for the
            joint model (one conservative + one liberal from each chamber).
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
    n_anchors = len(anchors)
    anchor_indices = {idx for idx, _ in anchors}

    coords = {
        "legislator": data["leg_slugs"],
        "vote": data["vote_ids"],
        "obs_id": np.arange(data["n_obs"]),
    }

    with pm.Model(coords=coords):
        # --- Legislator ideal points with anchors ---
        xi_free = pm.Normal("xi_free", mu=0, sigma=1, shape=n_leg - n_anchors)

        # Build full xi vector with anchors inserted at correct positions
        xi_raw = pt.zeros(n_leg)
        for anchor_idx, anchor_val in anchors:
            xi_raw = pt.set_subtensor(xi_raw[anchor_idx], anchor_val)

        # Fill free positions
        free_positions = [i for i in range(n_leg) if i not in anchor_indices]
        for k, pos in enumerate(free_positions):
            xi_raw = pt.set_subtensor(xi_raw[pos], xi_free[k])

        xi = pm.Deterministic("xi", xi_raw, dims="legislator")

        # --- Roll call parameters ---
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")
        # Normal(0,1): unconstrained discrimination. Anchors provide sign identification,
        # so positive constraint is unnecessary. Negative beta = liberal position is Yea.
        # See analysis/design/beta_prior_investigation.md for full rationale.
        beta = pm.Normal("beta", mu=0, sigma=1, shape=n_votes, dims="vote")

        # --- Likelihood ---
        eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

        # --- Sample ---
        print(f"  Sampling: {n_samples} draws, {n_tune} tune, {n_chains} chains")
        print(f"  target_accept={target_accept}, seed={RANDOM_SEED}")
        print(f"  Anchors: {n_anchors} fixed legislators")
        t0 = time.time()
        idata = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=RANDOM_SEED,
            progressbar=True,
        )
        sampling_time = time.time() - t0

    print(f"  Sampling complete in {sampling_time:.1f}s")
    return idata, sampling_time


# ── Phase 4: Convergence Diagnostics ────────────────────────────────────────


def check_convergence(idata: az.InferenceData, chamber: str) -> dict:
    """Run standard MCMC convergence diagnostics.

    Returns dict with all diagnostic metrics.
    """
    print_header(f"CONVERGENCE DIAGNOSTICS — {chamber}")

    diag = {}

    # R-hat
    rhat = az.rhat(idata)
    xi_rhat_max = float(rhat["xi"].max())
    alpha_rhat_max = float(rhat["alpha"].max())
    beta_rhat_max = float(rhat["beta"].max())
    diag["xi_rhat_max"] = xi_rhat_max
    diag["alpha_rhat_max"] = alpha_rhat_max
    diag["beta_rhat_max"] = beta_rhat_max

    rhat_ok = max(xi_rhat_max, alpha_rhat_max, beta_rhat_max) < RHAT_THRESHOLD
    xi_rhat_status = "OK" if xi_rhat_max < RHAT_THRESHOLD else "WARNING"
    alpha_rhat_status = "OK" if alpha_rhat_max < RHAT_THRESHOLD else "WARNING"
    beta_rhat_status = "OK" if beta_rhat_max < RHAT_THRESHOLD else "WARNING"
    print(f"  R-hat (xi):    max = {xi_rhat_max:.4f}  {xi_rhat_status}")
    print(f"  R-hat (alpha): max = {alpha_rhat_max:.4f}  {alpha_rhat_status}")
    print(f"  R-hat (beta):  max = {beta_rhat_max:.4f}  {beta_rhat_status}")

    # ESS
    ess = az.ess(idata)
    xi_ess_min = float(ess["xi"].min())
    alpha_ess_min = float(ess["alpha"].min())
    beta_ess_min = float(ess["beta"].min())
    diag["xi_ess_min"] = xi_ess_min
    diag["alpha_ess_min"] = alpha_ess_min
    diag["beta_ess_min"] = beta_ess_min

    ess_ok = min(xi_ess_min, alpha_ess_min, beta_ess_min) > ESS_THRESHOLD
    xi_ess_status = "OK" if xi_ess_min > ESS_THRESHOLD else "WARNING"
    alpha_ess_status = "OK" if alpha_ess_min > ESS_THRESHOLD else "WARNING"
    beta_ess_status = "OK" if beta_ess_min > ESS_THRESHOLD else "WARNING"
    print(f"  ESS (xi):      min = {xi_ess_min:.0f}  {xi_ess_status}")
    print(f"  ESS (alpha):   min = {alpha_ess_min:.0f}  {alpha_ess_status}")
    print(f"  ESS (beta):    min = {beta_ess_min:.0f}  {beta_ess_status}")

    # Divergences
    divergences = int(idata.sample_stats["diverging"].sum().values)
    diag["divergences"] = divergences
    div_ok = divergences < MAX_DIVERGENCES
    print(f"  Divergences:   {divergences}  {'OK' if div_ok else 'WARNING'}")

    # E-BFMI per chain
    bfmi_values = az.bfmi(idata)
    diag["ebfmi"] = [float(v) for v in bfmi_values]
    bfmi_ok = all(v > 0.3 for v in bfmi_values)
    for i, v in enumerate(bfmi_values):
        print(f"  E-BFMI chain {i}: {v:.3f}  {'OK' if v > 0.3 else 'WARNING'}")

    diag["all_ok"] = rhat_ok and ess_ok and div_ok and bfmi_ok
    if diag["all_ok"]:
        print("  CONVERGENCE: ALL CHECKS PASSED")
    else:
        print("  CONVERGENCE: SOME CHECKS FAILED — inspect diagnostics")

    return diag


# ── Phase 5: Extract Posteriors ─────────────────────────────────────────────


def extract_ideal_points(
    idata: az.InferenceData,
    data: dict,
    legislators: pl.DataFrame,
) -> pl.DataFrame:
    """Extract posterior summaries for legislator ideal points.

    Returns polars DataFrame with slug, name, party, district, xi_mean, xi_sd,
    xi_hdi_2.5, xi_hdi_97.5.
    """
    xi_posterior = idata.posterior["xi"]  # shape: (chain, draw, legislator)
    xi_mean = xi_posterior.mean(dim=["chain", "draw"]).values
    xi_sd = xi_posterior.std(dim=["chain", "draw"]).values

    # HDI via ArviZ
    xi_hdi = az.hdi(idata, var_names=["xi"], hdi_prob=0.95)["xi"].values

    slugs = data["leg_slugs"]
    rows = []
    for i, slug in enumerate(slugs):
        rows.append(
            {
                "legislator_slug": slug,
                "xi_mean": float(xi_mean[i]),
                "xi_sd": float(xi_sd[i]),
                "xi_hdi_2.5": float(xi_hdi[i, 0]),
                "xi_hdi_97.5": float(xi_hdi[i, 1]),
            }
        )

    df = pl.DataFrame(rows)

    # Join legislator metadata
    meta = legislators.select("slug", "full_name", "party", "district", "chamber")
    df = df.join(meta, left_on="legislator_slug", right_on="slug", how="left")

    return df.sort("xi_mean", descending=True)


def extract_bill_parameters(
    idata: az.InferenceData,
    data: dict,
    rollcalls: pl.DataFrame,
) -> pl.DataFrame:
    """Extract posterior summaries for bill difficulty and discrimination.

    Returns polars DataFrame with vote_id, bill_number, short_title,
    alpha_mean, alpha_sd, beta_mean, beta_sd, is_veto_override.
    """
    alpha_posterior = idata.posterior["alpha"]
    beta_posterior = idata.posterior["beta"]

    alpha_mean = alpha_posterior.mean(dim=["chain", "draw"]).values
    alpha_sd = alpha_posterior.std(dim=["chain", "draw"]).values
    beta_mean = beta_posterior.mean(dim=["chain", "draw"]).values
    beta_sd = beta_posterior.std(dim=["chain", "draw"]).values

    vote_ids = data["vote_ids"]
    rows = []
    for j, vid in enumerate(vote_ids):
        rows.append(
            {
                "vote_id": vid,
                "alpha_mean": float(alpha_mean[j]),
                "alpha_sd": float(alpha_sd[j]),
                "beta_mean": float(beta_mean[j]),
                "beta_sd": float(beta_sd[j]),
            }
        )

    df = pl.DataFrame(rows)

    # Join rollcall metadata
    meta_cols = ["vote_id", "bill_number", "short_title", "motion", "vote_type"]
    available = [c for c in meta_cols if c in rollcalls.columns]
    if available:
        meta = rollcalls.select(available)
        df = df.join(meta, on="vote_id", how="left")

    # Flag veto overrides
    if "motion" in df.columns:
        df = df.with_columns(
            pl.col("motion").str.to_lowercase().str.contains("veto").alias("is_veto_override")
        )
    else:
        df = df.with_columns(pl.lit(False).alias("is_veto_override"))

    return df.sort("beta_mean", descending=True)


# ── Phase 6: Plots ──────────────────────────────────────────────────────────


def plot_forest(
    ideal_points: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Forest plot: ideal points with 95% HDI, party-colored, sorted by xi_mean."""
    sorted_df = ideal_points.sort("xi_mean")
    n = sorted_df.height

    fig, ax = plt.subplots(figsize=(10, max(14, n * 0.22)))

    # Flagged legislators to highlight (from analytic-flags.md)
    highlight_slugs = {
        "sen_tyson_caryn_1": "Most conservative but lowest loyalty",
        "sen_thompson_mike_1": "Contrarian on routine bills",
        "rep_schreiber_mark_1": "Most bipartisan House member",
        "sen_miller_silas_1": "Mid-session replacement (30 votes)",
    }

    y_pos = np.arange(n)
    for i, row in enumerate(sorted_df.iter_rows(named=True)):
        slug = row["legislator_slug"]
        color = PARTY_COLORS.get(row["party"], "#888888")
        is_highlight = slug in highlight_slugs

        ax.hlines(
            i,
            row["xi_hdi_2.5"],
            row["xi_hdi_97.5"],
            colors=color,
            alpha=0.7 if is_highlight else 0.4,
            linewidth=2.5 if is_highlight else 1.5,
        )
        ax.scatter(
            row["xi_mean"],
            i,
            c=color,
            s=40 if is_highlight else 20,
            zorder=5,
            edgecolors="black",
            linewidth=0.3,
            marker="D" if is_highlight else "o",
        )

        # Annotate flagged legislators
        if is_highlight:
            ax.annotate(
                highlight_slugs[slug],
                (row["xi_hdi_97.5"], i),
                fontsize=6,
                fontstyle="italic",
                color="#555555",
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                bbox={"boxstyle": "round,pad=0.2", "fc": "lightyellow", "alpha": 0.7},
            )

    ax.set_yticks(y_pos)
    labels = []
    for row in sorted_df.iter_rows(named=True):
        name = row.get("full_name", row["legislator_slug"])
        party_initial = row["party"][0] if row["party"] else "?"
        labels.append(f"{name} ({party_initial})")
    ax.set_yticklabels(labels, fontsize=5.5)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Ideal Point (Liberal \u2190 \u2192 Conservative)")
    ax.set_title(
        f"{chamber} \u2014 Where Does Each Legislator Fall on the Ideological Spectrum?",
        fontsize=12,
    )
    legend_handles = [
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
        plt.Line2D(
            [],
            [],
            marker="D",
            color="gray",
            linestyle="None",
            markersize=6,
            label="Flagged legislator",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"forest_{chamber.lower()}.png")


def plot_discrimination(
    bill_params: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Histogram of bill discrimination parameters (beta_mean)."""
    beta_vals = bill_params["beta_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    # Color by sign: positive (R-Yea) red, negative (D-Yea) blue
    pos_vals = beta_vals[beta_vals >= 0]
    neg_vals = beta_vals[beta_vals < 0]
    if len(pos_vals) > 0:
        ax.hist(
            pos_vals,
            bins=30,
            alpha=0.6,
            color="#E81B23",
            label=f"Republicans favor Yea (n={len(pos_vals)})",
        )
    if len(neg_vals) > 0:
        ax.hist(
            neg_vals,
            bins=30,
            alpha=0.6,
            color="#0015BC",
            label=f"Democrats favor Yea (n={len(neg_vals)})",
        )
    ax.axvline(0, color="black", linestyle="--", alpha=0.6, label="Bipartisan (no divide)")
    median_beta = float(np.median(beta_vals))
    ax.axvline(
        median_beta, color="orange", linestyle="-", alpha=0.6, label=f"Median = {median_beta:.2f}"
    )
    ax.set_xlabel("Bill Partisanship (how sharply the bill divides the legislature)")
    ax.set_ylabel("Number of Roll Calls")
    ax.set_title(f"{chamber} \u2014 How Sharply Do Bills Divide the Legislature?")

    # Annotation explaining the scale
    ax.annotate(
        "Higher values = more partisan bills\nValues near zero = bipartisan",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        fontstyle="italic",
        color="#555555",
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "alpha": 0.8, "ec": "#cccccc"},
    )

    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"discrimination_{chamber.lower()}.png")


def plot_irt_vs_pca(
    ideal_points: pl.DataFrame,
    pca_scores: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> float:
    """Scatter plot of IRT xi_mean vs PCA PC1, with Pearson r annotation.

    Returns Pearson r.
    """
    # Merge on legislator_slug
    merged = ideal_points.select("legislator_slug", "xi_mean").join(
        pca_scores.select("legislator_slug", "PC1"),
        on="legislator_slug",
        how="inner",
    )

    xi_arr = merged["xi_mean"].to_numpy()
    pc1_arr = merged["PC1"].to_numpy()
    pearson_r = float(np.corrcoef(xi_arr, pc1_arr)[0, 1])
    spearman_r = float(stats.spearmanr(xi_arr, pc1_arr).statistic)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by party if available
    merged_with_party = merged.join(
        ideal_points.select("legislator_slug", "party"),
        on="legislator_slug",
        how="left",
    )
    for party, color in PARTY_COLORS.items():
        subset = merged_with_party.filter(pl.col("party") == party)
        if subset.height == 0:
            continue
        ax.scatter(
            subset["PC1"].to_numpy(),
            subset["xi_mean"].to_numpy(),
            c=color,
            s=40,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            label=party,
        )

    ax.set_xlabel("PCA PC1")
    ax.set_ylabel("IRT Ideal Point (xi_mean)")
    ax.set_title(
        f"{chamber} — IRT vs PCA Comparison\n"
        f"Pearson r = {pearson_r:.4f}, Spearman rho = {spearman_r:.4f}"
    )
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"irt_vs_pca_{chamber.lower()}.png")

    return pearson_r


def plot_traces(
    idata: az.InferenceData,
    data: dict,
    chamber: str,
    out_dir: Path,
) -> None:
    """Trace plots for N_TRACE_LEGISLATORS representative ideal points."""
    slugs = data["leg_slugs"]
    n = min(N_TRACE_LEGISLATORS, len(slugs))
    # Pick evenly spaced legislators
    indices = np.linspace(0, len(slugs) - 1, n, dtype=int)
    selected_slugs = [slugs[i] for i in indices]

    az.plot_trace(
        idata,
        var_names=["xi"],
        coords={"legislator": selected_slugs},
        figsize=(14, 3 * n),
    )
    fig = plt.gcf()
    fig.suptitle(f"{chamber} \u2014 Trace Plots (Selected Ideal Points)", fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir / f"trace_{chamber.lower()}.png")

    # Additionally produce a convergence summary plot for nontechnical audiences
    plot_convergence_summary(idata, data, chamber, out_dir)


def plot_convergence_summary(
    idata: az.InferenceData,
    data: dict,
    chamber: str,
    out_dir: Path,
) -> None:
    """Nontechnical convergence summary: overlapping posterior KDEs from multiple chains.

    Shows that all independent model runs agree, validating the results.
    """
    slugs = data["leg_slugs"]
    n = min(4, len(slugs))
    indices = np.linspace(0, len(slugs) - 1, n, dtype=int)
    selected_slugs = [slugs[i] for i in indices]

    n_chains = idata.posterior.dims.get("chain", 1)
    if n_chains < 2:
        return  # Need multiple chains to show agreement

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    chain_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for ax, slug in zip(axes, selected_slugs):
        xi_data = idata.posterior["xi"].sel(legislator=slug)
        for chain_idx in range(n_chains):
            chain_vals = xi_data.sel(chain=chain_idx).values
            color = chain_colors[chain_idx % len(chain_colors)]
            ax.hist(
                chain_vals,
                bins=40,
                alpha=0.4,
                color=color,
                label=f"Run {chain_idx + 1}",
                density=True,
                edgecolor="none",
            )
        # Derive last name for label
        name = slug.split("_")
        last_name = name[1].title() if len(name) > 1 else slug
        ax.set_title(last_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Ideal Point")
        ax.set_ylabel("")
        ax.set_yticks([])
        if ax == axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"{chamber} \u2014 The Model Ran {n_chains} Independent Chains and They All Agree",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Overlapping distributions confirm the model converged to a stable answer",
        ha="center",
        fontsize=10,
        fontstyle="italic",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_fig(fig, out_dir / f"convergence_summary_{chamber.lower()}.png")


def plot_ppc_yea_rate(
    observed_yea_rate: float,
    replicated_yea_rates: np.ndarray,
    chamber: str,
    out_dir: Path,
) -> float:
    """Plot posterior predictive Yea rate distribution vs observed.

    Returns Bayesian p-value.
    """
    bayesian_p = float(np.mean(replicated_yea_rates >= observed_yea_rate))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        replicated_yea_rates,
        bins=40,
        edgecolor="black",
        alpha=0.7,
        color="#4C72B0",
        label="Replicated Yea rates",
    )
    ax.axvline(
        observed_yea_rate,
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"Observed = {observed_yea_rate:.3f}",
    )
    ax.set_xlabel("Overall Yea Rate")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{chamber} — PPC: Overall Yea Rate\nBayesian p-value = {bayesian_p:.3f}")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"ppc_yea_rate_{chamber.lower()}.png")

    return bayesian_p


# ── Phase 7: PCA Comparison ─────────────────────────────────────────────────


def compare_with_pca(
    ideal_points: pl.DataFrame,
    pca_scores: pl.DataFrame,
    chamber: str,
) -> dict:
    """Compare IRT ideal points with PCA PC1 scores."""
    merged = ideal_points.select("legislator_slug", "xi_mean").join(
        pca_scores.select("legislator_slug", "PC1"),
        on="legislator_slug",
        how="inner",
    )

    xi_arr = merged["xi_mean"].to_numpy()
    pc1_arr = merged["PC1"].to_numpy()

    pearson_r = float(np.corrcoef(xi_arr, pc1_arr)[0, 1])
    spearman_result = stats.spearmanr(xi_arr, pc1_arr)
    spearman_rho = float(spearman_result.statistic)

    print(f"  {chamber}: Pearson r = {pearson_r:.4f}, Spearman rho = {spearman_rho:.4f}")
    if pearson_r > 0.95:
        print("  Result: STRONG agreement (r > 0.95)")
    elif pearson_r > 0.90:
        print("  Result: Good agreement (r > 0.90)")
    else:
        print("  Result: Moderate agreement (r <= 0.90) — IRT capturing non-linearities?")

    return {
        "chamber": chamber,
        "n_shared": merged.height,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
    }


# ── Phase 8: Posterior Predictive Checks ────────────────────────────────────


def run_ppc(
    idata: az.InferenceData,
    data: dict,
    chamber: str,
) -> dict:
    """Posterior predictive checks on overall Yea rate and classification accuracy."""
    print(f"\n  {chamber} posterior predictive checks:")

    xi_post = idata.posterior["xi"].values  # (chain, draw, leg)
    alpha_post = idata.posterior["alpha"].values  # (chain, draw, vote)
    beta_post = idata.posterior["beta"].values  # (chain, draw, vote)

    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y_obs = data["y"]
    observed_yea_rate = float(y_obs.mean())

    n_chains, n_draws = xi_post.shape[0], xi_post.shape[1]
    n_reps = min(500, n_chains * n_draws)  # Limit replications for speed

    rng = np.random.default_rng(RANDOM_SEED)
    replicated_yea_rates = []
    replicated_accuracies = []

    for _ in range(n_reps):
        c = rng.integers(0, n_chains)
        d = rng.integers(0, n_draws)

        xi_draw = xi_post[c, d]
        alpha_draw = alpha_post[c, d]
        beta_draw = beta_post[c, d]

        eta = beta_draw[vote_idx] * xi_draw[leg_idx] - alpha_draw[vote_idx]
        p = 1.0 / (1.0 + np.exp(-eta))
        y_rep = rng.binomial(1, p)

        replicated_yea_rates.append(float(y_rep.mean()))
        replicated_accuracies.append(float((y_rep == y_obs).mean()))

    rep_yea_rates = np.array(replicated_yea_rates)
    rep_accuracies = np.array(replicated_accuracies)

    # Bayesian p-values
    p_yea_rate = float(np.mean(rep_yea_rates >= observed_yea_rate))
    mean_rep_accuracy = float(rep_accuracies.mean())

    print(f"    Observed Yea rate: {observed_yea_rate:.3f}")
    print(f"    Replicated Yea rate: {rep_yea_rates.mean():.3f} +/- {rep_yea_rates.std():.3f}")
    print(f"    Bayesian p-value (Yea rate): {p_yea_rate:.3f}")
    print(f"    Mean replicated accuracy: {mean_rep_accuracy:.3f}")

    if 0.1 <= p_yea_rate <= 0.9:
        print("    Result: WELL-CALIBRATED (p in [0.1, 0.9])")
    else:
        print("    Result: POTENTIAL MISFIT (p outside [0.1, 0.9])")

    return {
        "chamber": chamber,
        "observed_yea_rate": observed_yea_rate,
        "replicated_yea_rate_mean": float(rep_yea_rates.mean()),
        "replicated_yea_rate_sd": float(rep_yea_rates.std()),
        "bayesian_p_yea_rate": p_yea_rate,
        "mean_replicated_accuracy": mean_rep_accuracy,
        "n_replications": n_reps,
        "replicated_yea_rates": rep_yea_rates,
    }


# ── Phase 9: Holdout Validation ─────────────────────────────────────────────


def run_holdout_validation(
    idata: az.InferenceData,
    data: dict,
    chamber: str,
) -> dict:
    """In-sample prediction on random 20% of observed cells.

    Uses posterior means. Documented as in-sample (model saw all data).
    PPC provides the proper Bayesian validation.
    """
    print(f"\n  {chamber} holdout validation (in-sample prediction):")

    xi_mean = idata.posterior["xi"].mean(dim=["chain", "draw"]).values
    alpha_mean = idata.posterior["alpha"].mean(dim=["chain", "draw"]).values
    beta_mean = idata.posterior["beta"].mean(dim=["chain", "draw"]).values

    leg_idx = data["leg_idx"]
    vote_idx = data["vote_idx"]
    y_obs = data["y"]

    # Select random 20% as holdout
    rng = np.random.default_rng(HOLDOUT_SEED)
    n_holdout = int(len(y_obs) * HOLDOUT_FRACTION)
    holdout_mask = np.zeros(len(y_obs), dtype=bool)
    holdout_indices = rng.choice(len(y_obs), size=n_holdout, replace=False)
    holdout_mask[holdout_indices] = True

    # Predict
    eta = beta_mean[vote_idx] * xi_mean[leg_idx] - alpha_mean[vote_idx]
    p_yea = 1.0 / (1.0 + np.exp(-eta))

    # Holdout metrics
    y_holdout = y_obs[holdout_mask]
    p_holdout = p_yea[holdout_mask]
    pred_binary = (p_holdout >= 0.5).astype(int)

    accuracy = float((pred_binary == y_holdout).mean())
    base_rate = float(y_holdout.mean())
    base_accuracy = max(base_rate, 1 - base_rate)

    from sklearn.metrics import roc_auc_score

    try:
        auc = float(roc_auc_score(y_holdout, p_holdout))
    except ValueError:
        auc = float("nan")

    print(f"    Holdout cells: {n_holdout:,}")
    print(f"    Base rate (Yea): {base_rate:.3f}")
    print(f"    Base-rate accuracy: {base_accuracy:.3f}")
    print(f"    IRT accuracy: {accuracy:.3f}")
    print(f"    AUC-ROC: {auc:.3f}")

    if accuracy > base_accuracy:
        print(f"    Result: PASS (accuracy {accuracy:.3f} > base rate {base_accuracy:.3f})")
    else:
        print(f"    Result: FAIL (accuracy {accuracy:.3f} <= base rate {base_accuracy:.3f})")

    return {
        "chamber": chamber,
        "holdout_cells": n_holdout,
        "base_rate": base_rate,
        "base_accuracy": base_accuracy,
        "accuracy": accuracy,
        "auc_roc": auc,
        "note": "In-sample prediction (model saw all data). "
        "PPC provides proper Bayesian validation.",
    }


# ── Phase 10: Sensitivity Analysis ──────────────────────────────────────────


def filter_vote_matrix_for_sensitivity(
    full_matrix: pl.DataFrame,
    rollcalls: pl.DataFrame,
    chamber: str,
    minority_threshold: float = SENSITIVITY_THRESHOLD,
    min_votes: int = MIN_VOTES,
) -> pl.DataFrame:
    """Re-filter the full vote matrix at an alternative minority threshold.

    Duplicated from PCA (keeps IRT self-contained per ADR-0005).
    """
    slug_col = "legislator_slug"
    vote_cols = [c for c in full_matrix.columns if c != slug_col]

    # Restrict to chamber
    chamber_vote_ids = set(rollcalls.filter(pl.col("chamber") == chamber)["vote_id"].to_list())
    prefix = "sen_" if chamber == "Senate" else "rep_"
    vote_cols = [c for c in vote_cols if c in chamber_vote_ids]
    matrix = full_matrix.filter(pl.col(slug_col).str.starts_with(prefix)).select(
        [slug_col, *vote_cols]
    )

    # Filter 1: Drop near-unanimous votes
    contested_cols = []
    for col in vote_cols:
        series = matrix[col].drop_nulls()
        if series.len() == 0:
            continue
        yea_frac = series.mean()
        minority_frac = min(yea_frac, 1 - yea_frac)
        if minority_frac >= minority_threshold:
            contested_cols.append(col)

    if not contested_cols:
        return matrix.select([slug_col]).head(0)

    filtered = matrix.select([slug_col, *contested_cols])

    # Filter 2: Drop low-participation legislators
    non_null_counts = filtered.select(
        slug_col,
        pl.sum_horizontal(*[pl.col(c).is_not_null().cast(pl.Int32) for c in contested_cols]).alias(
            "n_votes"
        ),
    )
    active_slugs = non_null_counts.filter(pl.col("n_votes") >= min_votes)[slug_col].to_list()
    filtered = filtered.filter(pl.col(slug_col).is_in(active_slugs))

    return filtered


def run_sensitivity(
    full_matrix: pl.DataFrame,
    default_results: dict[str, dict],
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
    pca_scores_dict: dict[str, pl.DataFrame],
    n_samples: int,
    n_tune: int,
    n_chains: int,
    plots_dir: Path,
) -> dict:
    """Run IRT with 10% minority threshold and compare ideal points.

    Full MCMC re-run at new threshold.
    """
    print_header("SENSITIVITY ANALYSIS (10% threshold)")
    findings: dict = {}

    for chamber, default in default_results.items():
        print(f"\n  {chamber}:")
        sens_matrix = filter_vote_matrix_for_sensitivity(
            full_matrix,
            rollcalls,
            chamber,
            minority_threshold=SENSITIVITY_THRESHOLD,
            min_votes=MIN_VOTES,
        )
        n_votes = len(sens_matrix.columns) - 1
        n_legs = sens_matrix.height
        print(f"    Sensitivity matrix: {n_legs} legislators x {n_votes} votes")

        if n_legs < 5 or n_votes < 5:
            print("    Skipping: too few data points")
            findings[chamber] = {"skipped": True, "reason": "insufficient data"}
            continue

        # Prepare IRT data
        sens_data = prepare_irt_data(sens_matrix, chamber)

        # Select anchors
        pca_scores = pca_scores_dict[chamber]
        cons_idx, cons_slug, lib_idx, lib_slug = select_anchors(
            pca_scores,
            sens_matrix,
            chamber,
        )

        # Sample
        sens_anchors = [(cons_idx, 1.0), (lib_idx, -1.0)]
        sens_idata, sens_time = build_and_sample(
            sens_data,
            sens_anchors,
            n_samples,
            n_tune,
            n_chains,
        )

        # Extract ideal points
        sens_ideal = extract_ideal_points(sens_idata, sens_data, legislators)

        # Compare with default
        default_ideal = default["ideal_points"]
        merged = default_ideal.select("legislator_slug", "xi_mean").join(
            sens_ideal.select("legislator_slug", pl.col("xi_mean").alias("xi_mean_sens")),
            on="legislator_slug",
            how="inner",
        )

        if merged.height < 5:
            print(f"    Skipping correlation: only {merged.height} shared legislators")
            findings[chamber] = {"skipped": True, "reason": "too few shared legislators"}
            continue

        default_arr = merged["xi_mean"].to_numpy()
        sens_arr = merged["xi_mean_sens"].to_numpy()
        correlation = float(np.corrcoef(default_arr, sens_arr)[0, 1])

        print(f"    Shared legislators: {merged.height}")
        print(f"    Pearson r: {correlation:.4f}")
        print(f"    Sampling time: {sens_time:.1f}s")

        if correlation > 0.95:
            print("    Result: ROBUST (r > 0.95)")
        else:
            print("    Result: SENSITIVE (r <= 0.95) — investigate threshold dependence")

        findings[chamber] = {
            "default_threshold": MINORITY_THRESHOLD,
            "sensitivity_threshold": SENSITIVITY_THRESHOLD,
            "default_n_legislators": default_ideal.height,
            "sensitivity_n_legislators": n_legs,
            "default_n_votes": default["data"]["n_votes"],
            "sensitivity_n_votes": n_votes,
            "shared_legislators": merged.height,
            "pearson_r": correlation,
            "sensitivity_sampling_time": sens_time,
        }

        # Plot sensitivity scatter
        _plot_sensitivity_scatter(default_arr, sens_arr, correlation, chamber, plots_dir)

    return findings


def _plot_sensitivity_scatter(
    default_xi: np.ndarray,
    sens_xi: np.ndarray,
    correlation: float,
    chamber: str,
    out_dir: Path,
) -> None:
    """Scatter plot comparing default and sensitivity ideal points."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(default_xi, sens_xi, c="#4C72B0", s=40, alpha=0.7, edgecolors="black", linewidth=0.5)

    lims = [
        min(default_xi.min(), sens_xi.min()) - 0.3,
        max(default_xi.max(), sens_xi.max()) + 0.3,
    ]
    ax.plot(lims, lims, "r--", alpha=0.5, label="Identity line")

    ax.set_xlabel(f"Ideal Point (default: {MINORITY_THRESHOLD * 100:.1f}% threshold)")
    ax.set_ylabel(f"Ideal Point (sensitivity: {SENSITIVITY_THRESHOLD * 100:.0f}% threshold)")
    ax.set_title(f"{chamber} — IRT Sensitivity (r = {correlation:.4f})")
    ax.legend()
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"sensitivity_xi_{chamber.lower()}.png")


# ── Phase 11: Filtering Manifest + Main ─────────────────────────────────────


def save_filtering_manifest(manifest: dict, out_dir: Path) -> None:
    path = out_dir / "filtering_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Saved: {path.name}")


def main() -> None:
    args = parse_args()

    from ks_vote_scraper.session import KSSession

    ks = KSSession.from_session_string(args.session)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path("data") / ks.output_name

    results_root = Path("results") / ks.output_name

    if args.eda_dir:
        eda_dir = Path(args.eda_dir)
    else:
        eda_dir = results_root / "eda" / "latest"

    if args.pca_dir:
        pca_dir = Path(args.pca_dir)
    else:
        pca_dir = results_root / "pca" / "latest"

    with RunContext(
        session=args.session,
        analysis_name="irt",
        params=vars(args),
        primer=IRT_PRIMER,
    ) as ctx:
        print(f"KS Legislature Bayesian IRT — Session {args.session}")
        print(f"Data:      {data_dir}")
        print(f"EDA:       {eda_dir}")
        print(f"PCA:       {pca_dir}")
        print(f"Output:    {ctx.run_dir}")
        print(f"Samples:   {args.n_samples} draws, {args.n_tune} tune, {args.n_chains} chains")

        # ── Phase 1: Load data ──
        print_header("PHASE 1: LOADING DATA")
        house_matrix, senate_matrix, full_matrix = load_eda_matrices(eda_dir)
        pca_house, pca_senate = load_pca_scores(pca_dir)
        rollcalls, legislators = load_metadata(data_dir)

        print(f"  House filtered: {house_matrix.height} x {len(house_matrix.columns) - 1}")
        print(f"  Senate filtered: {senate_matrix.height} x {len(senate_matrix.columns) - 1}")
        print(f"  Full matrix: {full_matrix.height} x {len(full_matrix.columns) - 1}")
        print(f"  PCA House scores: {pca_house.height}")
        print(f"  PCA Senate scores: {pca_senate.height}")
        print(f"  Rollcalls: {rollcalls.height}")
        print(f"  Legislators: {legislators.height}")

        chamber_configs = [
            ("House", house_matrix, pca_house),
            ("Senate", senate_matrix, pca_senate),
        ]

        results: dict[str, dict] = {}
        pca_comparisons: dict[str, dict] = {}
        ppc_results: dict[str, dict] = {}
        validation_results: dict[str, dict] = {}
        pca_scores_dict = {"House": pca_house, "Senate": pca_senate}

        for chamber, matrix, pca_scores in chamber_configs:
            if matrix.height < 5:
                print(f"\n  Skipping {chamber}: too few legislators ({matrix.height})")
                continue

            # ── Phase 2: Prepare IRT data ──
            print_header(f"PHASE 2: PREPARE IRT DATA — {chamber}")
            data = prepare_irt_data(matrix, chamber)

            # Select anchors from PCA extremes
            print("\n  Selecting anchors from PCA PC1 extremes:")
            cons_idx, cons_slug, lib_idx, lib_slug = select_anchors(
                pca_scores,
                matrix,
                chamber,
            )

            # ── Phase 3: Build and sample ──
            print_header(f"PHASE 3: MCMC SAMPLING — {chamber}")
            chamber_anchors = [(cons_idx, 1.0), (lib_idx, -1.0)]
            idata, sampling_time = build_and_sample(
                data,
                chamber_anchors,
                args.n_samples,
                args.n_tune,
                args.n_chains,
            )

            # ── Phase 4: Convergence diagnostics ──
            diagnostics = check_convergence(idata, chamber)

            # ── Phase 5: Extract posteriors ──
            print_header(f"PHASE 5: EXTRACT POSTERIORS — {chamber}")
            ideal_points = extract_ideal_points(idata, data, legislators)
            bill_params = extract_bill_parameters(idata, data, rollcalls)

            # Print top/bottom ideal points
            print("\n  Top 5 (most conservative):")
            for row in ideal_points.head(5).iter_rows(named=True):
                print(
                    f"    {row['full_name']:30s}  {row['party']:12s}  "
                    f"xi={row['xi_mean']:+.3f}  [{row['xi_hdi_2.5']:+.3f}, "
                    f"{row['xi_hdi_97.5']:+.3f}]"
                )
            print("  Bottom 5 (most liberal):")
            for row in ideal_points.tail(5).iter_rows(named=True):
                print(
                    f"    {row['full_name']:30s}  {row['party']:12s}  "
                    f"xi={row['xi_mean']:+.3f}  [{row['xi_hdi_2.5']:+.3f}, "
                    f"{row['xi_hdi_97.5']:+.3f}]"
                )

            # Save parquets
            ideal_points.write_parquet(ctx.data_dir / f"ideal_points_{chamber.lower()}.parquet")
            bill_params.write_parquet(ctx.data_dir / f"bill_params_{chamber.lower()}.parquet")
            print(f"  Saved: ideal_points_{chamber.lower()}.parquet")
            print(f"  Saved: bill_params_{chamber.lower()}.parquet")

            # Save InferenceData as NetCDF
            nc_path = ctx.data_dir / f"idata_{chamber.lower()}.nc"
            idata.to_netcdf(str(nc_path))
            print(f"  Saved: idata_{chamber.lower()}.nc")

            # ── Phase 6: Plots ──
            print_header(f"PHASE 6: PLOTS — {chamber}")
            plot_forest(ideal_points, chamber, ctx.plots_dir)
            plot_discrimination(bill_params, chamber, ctx.plots_dir)
            plot_irt_vs_pca(ideal_points, pca_scores, chamber, ctx.plots_dir)
            plot_traces(idata, data, chamber, ctx.plots_dir)

            # ── Phase 7: PCA comparison ──
            print_header(f"PHASE 7: PCA COMPARISON — {chamber}")
            pca_comp = compare_with_pca(ideal_points, pca_scores, chamber)
            pca_comparisons[chamber] = pca_comp

            # ── Phase 8: Posterior predictive checks ──
            print_header(f"PHASE 8: POSTERIOR PREDICTIVE CHECKS — {chamber}")
            ppc = run_ppc(idata, data, chamber)
            ppc_results[chamber] = ppc

            # Plot PPC
            plot_ppc_yea_rate(
                ppc["observed_yea_rate"],
                ppc["replicated_yea_rates"],
                chamber,
                ctx.plots_dir,
            )

            # ── Phase 9: Holdout validation ──
            print_header(f"PHASE 9: HOLDOUT VALIDATION — {chamber}")
            holdout = run_holdout_validation(idata, data, chamber)
            validation_results[chamber] = holdout

            # Store results for sensitivity comparison
            results[chamber] = {
                "ideal_points": ideal_points,
                "bill_params": bill_params,
                "idata": idata,
                "data": data,
                "diagnostics": diagnostics,
                "sampling_time": sampling_time,
                "cons_slug": cons_slug,
                "lib_slug": lib_slug,
            }

        # ── Joint Cross-Chamber Equating ──
        # A joint MCMC IRT model was attempted but does not converge: with 71
        # shared bills and 169 legislators (0.42 bills/legislator), the posterior
        # is under-identified despite 4 anchors and 4 chains (R-hat > 1.7).
        # Instead, use classical test equating (mean/sigma method) on the shared
        # bill discrimination parameters to link the per-chamber scales.
        joint_equating: dict = {}
        if not args.skip_joint and len(results) == 2:
            print_header("JOINT MODEL — PHASE J1: IDENTIFY SHARED BILLS")
            joint_matrix, mapping_info = build_joint_vote_matrix(
                house_matrix,
                senate_matrix,
                rollcalls,
                legislators,
            )

            print_header("JOINT MODEL — PHASE J2: TEST EQUATING (MEAN/SIGMA)")
            joint_equating = equate_chambers(
                results,
                mapping_info,
                legislators,
                ctx.plots_dir,
            )

            # Store correlation info in per-chamber results for report
            for chamber, r in joint_equating["correlations"].items():
                if chamber in results:
                    chamber_slugs = set(
                        results[chamber]["ideal_points"]["legislator_slug"].to_list()
                    )
                    equated_slugs = set(
                        joint_equating["equated_ideal_points"]["legislator_slug"].to_list()
                    )
                    results[chamber]["joint_correlation"] = {
                        "pearson_r": r,
                        "n_shared": len(chamber_slugs & equated_slugs),
                    }

            # Save outputs
            print_header("JOINT MODEL — PHASE J3: SAVE OUTPUTS")
            eq_ip = joint_equating["equated_ideal_points"]
            eq_ip.write_parquet(ctx.data_dir / "ideal_points_joint_equated.parquet")
            print("  Saved: ideal_points_joint_equated.parquet")

            results["Joint"] = {
                "ideal_points": eq_ip,
                "equating": joint_equating,
                "mapping_info": mapping_info,
                "joint_correlations": joint_equating["correlations"],
            }
        elif args.skip_joint:
            print_header("JOINT MODEL (SKIPPED)")
        elif len(results) < 2:
            print_header("JOINT MODEL (SKIPPED — need both chambers)")

        # ── Phase 10: Sensitivity analysis ──
        sensitivity_findings: dict = {}
        if not args.skip_sensitivity and results:
            sensitivity_findings = run_sensitivity(
                full_matrix,
                results,
                rollcalls,
                legislators,
                pca_scores_dict,
                args.n_samples,
                args.n_tune,
                args.n_chains,
                ctx.plots_dir,
            )
        elif args.skip_sensitivity:
            print_header("SENSITIVITY ANALYSIS (SKIPPED)")

        # ── Phase 11: Filtering manifest + report ──
        print_header("PHASE 11: FILTERING MANIFEST")
        manifest: dict = {
            "model": "2PL IRT",
            "priors": {
                "xi": "Normal(0, 1) with two anchors",
                "alpha": "Normal(0, 5)",
                "beta": "Normal(0, 1)",
            },
            "sampling": {
                "n_samples": args.n_samples,
                "n_tune": args.n_tune,
                "n_chains": args.n_chains,
                "target_accept": TARGET_ACCEPT,
                "seed": RANDOM_SEED,
            },
            "filters": {
                "minority_threshold_default": MINORITY_THRESHOLD,
                "minority_threshold_sensitivity": SENSITIVITY_THRESHOLD,
                "min_votes": MIN_VOTES,
                "min_participation_for_anchor": MIN_PARTICIPATION_FOR_ANCHOR,
            },
            "holdout": {
                "fraction": HOLDOUT_FRACTION,
                "seed": HOLDOUT_SEED,
                "note": "In-sample prediction (model saw all data)",
            },
        }
        for chamber, result in results.items():
            if chamber == "Joint":
                continue  # joint model has its own manifest section below
            ch = chamber.lower()
            manifest[f"{ch}_n_legislators"] = result["data"]["n_legislators"]
            manifest[f"{ch}_n_votes"] = result["data"]["n_votes"]
            manifest[f"{ch}_n_obs"] = result["data"]["n_obs"]
            manifest[f"{ch}_sampling_time_s"] = result["sampling_time"]
            manifest[f"{ch}_anchors"] = {
                "conservative": result["cons_slug"],
                "liberal": result["lib_slug"],
            }
            manifest[f"{ch}_diagnostics"] = result["diagnostics"]

        if pca_comparisons:
            manifest["pca_comparison"] = pca_comparisons
        if validation_results:
            manifest["validation"] = validation_results
        if ppc_results:
            # Strip numpy arrays for JSON serialization
            ppc_serializable = {}
            for ch, ppc in ppc_results.items():
                ppc_serializable[ch] = {k: v for k, v in ppc.items() if k != "replicated_yea_rates"}
            manifest["ppc"] = ppc_serializable
        if sensitivity_findings:
            manifest["sensitivity"] = sensitivity_findings

        if "Joint" in results:
            joint_r = results["Joint"]
            mi = joint_r.get("mapping_info", {})
            eq = joint_r.get("equating", {})
            manifest["joint_model"] = {
                "method": "mean_sigma_equating",
                "reference_scale": "House",
                "n_matched_bills": len(mi.get("matched_bills", [])),
                "n_bridging_legislators": len(mi.get("bridging_legislators", [])),
                "bridging_legislators": [
                    b["full_name"] for b in mi.get("bridging_legislators", [])
                ],
                "transformation": eq.get("transformation", {}),
                "correlations": joint_r.get("joint_correlations", {}),
            }

        save_filtering_manifest(manifest, ctx.run_dir)

        # ── HTML report ──
        print_header("HTML REPORT")
        build_irt_report(
            ctx.report,
            results=results,
            pca_comparisons=pca_comparisons,
            ppc_results=ppc_results,
            validation_results=validation_results,
            sensitivity_findings=sensitivity_findings,
            plots_dir=ctx.plots_dir,
            n_samples=args.n_samples,
            n_tune=args.n_tune,
            n_chains=args.n_chains,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  NetCDF files:   {len(list(ctx.data_dir.glob('*.nc')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")
        print(f"  JSON manifests: {len(list(ctx.run_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
