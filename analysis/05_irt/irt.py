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

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
from matplotlib.patches import Patch
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
    from analysis.irt_report import build_irt_report
except ModuleNotFoundError:
    from irt_report import build_irt_report  # type: ignore[no-redef]

try:
    from analysis.init_strategy import load_2d_scores, resolve_init_source
except ModuleNotFoundError:
    from init_strategy import load_2d_scores, resolve_init_source  # type: ignore[no-redef]

try:
    from analysis.tuning import (
        CONTESTED_THRESHOLD,
        HIGH_DISC_THRESHOLD,
        LOW_DISC_THRESHOLD,
        MIN_VOTES,
        PARTY_COLORS,
        SENSITIVITY_THRESHOLD,
        SUPERMAJORITY_THRESHOLD,
    )
except ModuleNotFoundError:
    from tuning import (  # type: ignore[no-redef]
        CONTESTED_THRESHOLD,
        HIGH_DISC_THRESHOLD,
        LOW_DISC_THRESHOLD,
        MIN_VOTES,
        PARTY_COLORS,
        SENSITIVITY_THRESHOLD,
        SUPERMAJORITY_THRESHOLD,
    )

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

**Identification** via seven-strategy system (ADR-0103):
Tallgrass implements seven identification strategies and auto-selects the best
one for each chamber's composition. See `docs/irt-identification-strategies.md`.
- Balanced chambers → anchor-pca (PCA PC1 party extremes, hard anchors at ±1)
- Supermajority (≥70%) → anchor-agreement (cross-party contested vote agreement)
- External scores available → external-prior (Shor-McCarty informative prior)
CLI override: `--identification {auto,anchor-pca,anchor-agreement,...}`

**Missing data:** Absences are handled natively — they are simply absent from
the likelihood. No imputation needed (unlike PCA).

### Pipeline

1. Load filtered vote matrices from EDA, PCA scores for anchor selection
2. Convert to long format, drop nulls (absences)
3. Select identification strategy (auto-detect or CLI override)
4. Build 2PL PyMC model with strategy-specific constraints
5. Sample with NUTS (2000 draws, 1000 tune, 2 chains)
6. Post-hoc sign validation (validate_sign)
7. Convergence diagnostics (R-hat, ESS, divergences, E-BFMI)
8. Extract posterior summaries (ideal points, bill parameters)
9. Generate plots (forest, discrimination, traces, PPC)
10. Compare with PCA PC1 scores
11. Posterior predictive checks
12. Holdout validation (in-sample prediction)
13. Sensitivity analysis (10% minority threshold re-run)

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

HOLDOUT_FRACTION = 0.20  # Random 20% of observed cells
HOLDOUT_SEED = 42
MIN_PARTICIPATION_FOR_ANCHOR = 0.50  # Anchors must have >= 50% participation
CONTESTED_VOTE_THRESHOLD = 0.10  # Both parties must have ≥10% on each side
MIN_CONTESTED_FOR_AGREEMENT = 10  # Need ≥10 contested votes for agreement-based anchors
MIN_CONTESTED_VOTES_PER_LEG = 5  # Legislator needs ≥5 contested votes for valid agreement


# Robustness flag thresholds
MIN_CONTESTED_FOR_REFIT = 50  # Need ≥50 contested votes for meaningful contested-only refit
HORSESHOE_DEM_WRONG_SIDE_FRAC = 0.20  # >20% of Democrats on conservative side → horseshoe
PROMOTE_2D_RANK_SHIFT = 10  # Flag legislators whose rank shifts >10 between 1D and 2D
MIN_PC2_VOTES_FOR_REFIT = 50  # Need ≥50 PC2-dominant votes for meaningful remediation
PC2_PRIOR_SIGMA = 1.0  # Width of the PC2 informative prior (experiment: sigma=1.0 best)
DIM1_PRIOR_SIGMA_DEFAULT = 1.0  # Width of the Dim 1 informative prior (ADR-0108)


# ── Robustness Flags ─────────────────────────────────────────────────────────
# Toggleable analysis enhancements for investigating supermajority horseshoe
# effects. All default to OFF. See ADR-0104.


@dataclass(frozen=True)
class RobustnessFlag:
    """A toggleable robustness enhancement for IRT analysis."""

    name: str
    label: str
    description: str
    enabled: bool


class RobustnessFlags:
    """Registry of available robustness flags."""

    CONTESTED_ONLY = "contested-only"
    HORSESHOE_DIAGNOSTIC = "horseshoe-diagnostic"
    HORSESHOE_REMEDIATE = "horseshoe-remediate"
    PROMOTE_2D = "promote-2d"
    DIM1_PRIOR = "dim1-prior"

    ALL_FLAGS = [
        CONTESTED_ONLY,
        HORSESHOE_DIAGNOSTIC,
        HORSESHOE_REMEDIATE,
        PROMOTE_2D,
        DIM1_PRIOR,
    ]

    LABELS: dict[str, str] = {
        CONTESTED_ONLY: "Contested Votes Only",
        HORSESHOE_DIAGNOSTIC: "Horseshoe Diagnostic",
        HORSESHOE_REMEDIATE: "Horseshoe Remediation",
        PROMOTE_2D: "2D Cross-Reference",
        DIM1_PRIOR: "Dim 1 Informative Prior",
    }

    DESCRIPTIONS: dict[str, str] = {
        CONTESTED_ONLY: (
            "Refit IRT using only cross-party contested votes, stripping "
            "intra-party establishment-vs-rebel votes that cause the horseshoe effect"
        ),
        HORSESHOE_DIAGNOSTIC: (
            "Run quantitative horseshoe detection (Democrat placement, "
            "party overlap, eigenvalue ratio) and add diagnostic section"
        ),
        HORSESHOE_REMEDIATE: (
            "When horseshoe detected, auto-refit using PC2-filtered votes and "
            "PC2 informative prior to recover the ideology dimension"
        ),
        PROMOTE_2D: (
            "Cross-reference 2D IRT results (Phase 06) and flag legislators "
            "whose rank shifts significantly between 1D and 2D models"
        ),
        DIM1_PRIOR: (
            "Use 2D IRT Dimension 1 as informative prior on ideal points "
            "(xi ~ Normal(dim1, sigma)) to recover ideology in horseshoe-affected "
            "chambers. Requires Phase 06 results. ADR-0108."
        ),
    }

    @classmethod
    def build_flags(cls, args: argparse.Namespace) -> list[RobustnessFlag]:
        """Build the list of robustness flags from CLI args."""
        flag_map = {
            cls.CONTESTED_ONLY: getattr(args, "contested_only", False),
            cls.HORSESHOE_DIAGNOSTIC: getattr(args, "horseshoe_diagnostic", False),
            cls.HORSESHOE_REMEDIATE: getattr(args, "horseshoe_remediate", False),
            cls.PROMOTE_2D: getattr(args, "promote_2d", False),
            cls.DIM1_PRIOR: getattr(args, "dim1_prior", False),
        }
        return [
            RobustnessFlag(
                name=name,
                label=cls.LABELS[name],
                description=cls.DESCRIPTIONS[name],
                enabled=flag_map.get(name, False),
            )
            for name in cls.ALL_FLAGS
        ]


# ── Identification Strategies ────────────────────────────────────────────────
# Canonical term from the IRT literature (Clinton-Jackman-Rivers 2004, Morucci
# et al. 2024). Each strategy solves the reflection invariance problem (negating
# all xi and beta yields an identical likelihood) differently.


class IdentificationStrategy:
    """Enumeration of IRT identification strategies.

    Each strategy resolves sign/location/scale invariance differently. The auto
    selector picks the best one for the data; the CLI flag overrides.
    """

    # ── Anchor-based strategies ──
    ANCHOR_PCA = "anchor-pca"
    """Hard anchors at xi=±1 selected by party-aware PCA PC1 extremes.
    Standard method (Clinton-Jackman-Rivers 2004). Works well for balanced
    chambers. In supermajority chambers, horseshoe effect distorts PC1 extremes.
    """

    ANCHOR_AGREEMENT = "anchor-agreement"
    """Hard anchors at xi=±1 selected by cross-party contested vote agreement.
    Picks the most partisan legislator from each party (lowest agreement with
    opposite party on contested votes). Robust to the horseshoe effect.
    """

    # ── Constraint-based strategies (no individual anchors) ──
    SORT_CONSTRAINT = "sort-constraint"
    """Ordering constraint: Democrat mean < Republican mean via pm.Potential.
    No individual legislators are pinned. All legislators are free parameters.
    Same approach as the hierarchical model (Phase 07). Sign identified by
    the party ordering; scale by the Normal(0,1) prior.
    """

    POSITIVE_BETA = "positive-beta"
    """Positive discrimination constraint: beta ~ HalfNormal(1). Forces all
    discrimination parameters to be positive, eliminating reflection invariance.
    Standard in educational testing. Limitation: silences D-Yea bills (where
    the liberal position is Yea → requires negative beta). ADR-0047.
    """

    HIERARCHICAL_PRIOR = "hierarchical-prior"
    """Party-informed prior: xi ~ Normal(+0.5, 1) for Republicans,
    Normal(-0.5, 1) for Democrats. No hard constraints — prior provides soft
    identification. Inspired by Bafumi-Gelman-Park-Kaplan (2005).
    """

    # ── Post-hoc only strategies ──
    UNCONSTRAINED = "unconstrained"
    """No identification constraints during MCMC. Post-hoc sign correction
    via party means + validate_sign(). Fastest to implement, relies entirely
    on post-hoc correction. Risk: chains may not mix well in the unidentified
    multimodal posterior.
    """

    # ── External data strategies ──
    EXTERNAL_PRIOR = "external-prior"
    """Informative prior from Shor-McCarty scores: xi ~ Normal(sm_score, 0.5).
    Requires Phase 17 external validation data. Solves sign, location, and
    scale simultaneously by anchoring to an external measurement system.
    Shor & McCarty (2011), Bonica & Woodruff (2015).
    """

    AUTO = "auto"
    """Auto-detect: select strategy based on chamber composition and data
    availability. Supermajority → agreement anchors. Balanced → PCA anchors.
    External scores available → external prior.
    """

    ALL_STRATEGIES = [
        ANCHOR_PCA,
        ANCHOR_AGREEMENT,
        SORT_CONSTRAINT,
        POSITIVE_BETA,
        HIERARCHICAL_PRIOR,
        UNCONSTRAINED,
        EXTERNAL_PRIOR,
    ]

    # Short human-readable descriptions for the report
    DESCRIPTIONS: dict[str, str] = {
        ANCHOR_PCA: "Hard anchors (PCA PC1 extremes, party-aware)",
        ANCHOR_AGREEMENT: "Hard anchors (cross-party contested vote agreement)",
        SORT_CONSTRAINT: "Sort constraint (D mean < R mean, no individual anchors)",
        POSITIVE_BETA: "Positive discrimination (HalfNormal beta, no anchors)",
        HIERARCHICAL_PRIOR: "Party-informed prior (soft identification, no anchors)",
        UNCONSTRAINED: "Unconstrained (post-hoc sign correction only)",
        EXTERNAL_PRIOR: "External score prior (Shor-McCarty informative prior)",
        AUTO: "Auto-detect (selects best strategy for chamber composition)",
    }

    # Literature references for each strategy
    REFERENCES: dict[str, str] = {
        ANCHOR_PCA: "Clinton, Jackman & Rivers (2004)",
        ANCHOR_AGREEMENT: "Tallgrass project (2026); extends CJR 2004",
        SORT_CONSTRAINT: "Factor analysis ordering; cf. Geweke & Zhou (1996)",
        POSITIVE_BETA: "Stan User's Guide §1.11; educational IRT convention",
        HIERARCHICAL_PRIOR: "Bafumi, Gelman, Park & Kaplan (2005)",
        UNCONSTRAINED: "pscl postProcess(); Jackman (2000)",
        EXTERNAL_PRIOR: "Shor & McCarty (2011); Bonica & Woodruff (2015)",
    }


# Joint model defaults
JOINT_TARGET_ACCEPT = 0.95
JOINT_N_CHAINS = 4

# Convergence thresholds
RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400
MAX_DIVERGENCES = 10

# Plot constants
N_TRACE_LEGISLATORS = 5
N_CONVERGENCE_SUMMARY = 4
TOP_DISCRIMINATING = 15


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Bayesian IRT")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
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
    parser.add_argument(
        "--no-pca-init",
        action="store_true",
        help="Disable PCA-informed chain initialization (on by default)",
    )
    parser.add_argument(
        "--init-strategy",
        default=None,
        choices=["auto", "irt-informed", "pca-informed", "2d-dim1"],
        help=(
            "Override chain initialization source. "
            "2d-dim1 uses 2D IRT Dim 1 for iterative refinement [research only]. "
            "Default: uses strategy-specific init. "
            "Production uses canonical routing (ADR-0109)."
        ),
    )
    parser.add_argument("--csv", action="store_true", help="Force CSV loading (skip database)")
    parser.add_argument(
        "--identification",
        default="auto",
        choices=[
            "auto",
            "anchor-pca",
            "anchor-agreement",
            "sort-constraint",
            "positive-beta",
            "hierarchical-prior",
            "unconstrained",
            "external-prior",
        ],
        help="Identification strategy (default: auto-detect from chamber composition)",
    )

    # Robustness flags (ADR-0104)
    robustness = parser.add_argument_group(
        "robustness flags",
        "Optional analysis enhancements for investigating supermajority horseshoe effects. "
        "All default to OFF. All flags appear in the HTML report with their ON/OFF status.",
    )
    robustness.add_argument(
        "--contested-only",
        action="store_true",
        help="Refit IRT using only cross-party contested votes",
    )
    robustness.add_argument(
        "--horseshoe-diagnostic",
        action="store_true",
        help="Run quantitative horseshoe detection diagnostics",
    )
    robustness.add_argument(
        "--horseshoe-remediate",
        action="store_true",
        help="[Research] Auto-refit with PC2-filtered votes when horseshoe detected "
        "(implies --horseshoe-diagnostic). Production uses canonical routing (ADR-0109).",
    )
    robustness.add_argument(
        "--promote-2d",
        action="store_true",
        help="Cross-reference 2D IRT (Phase 06) for misplaced legislators",
    )
    robustness.add_argument(
        "--irt-2d-dir",
        default=None,
        help="Override 2D IRT results directory (for --promote-2d)",
    )
    robustness.add_argument(
        "--dim1-prior",
        action="store_true",
        help="[Research] Use 2D IRT Dim 1 as informative prior for ideology recovery "
        "(requires Phase 06 results; implies --promote-2d). "
        "Production uses canonical routing (ADR-0109).",
    )
    robustness.add_argument(
        "--dim1-prior-sigma",
        type=float,
        default=DIM1_PRIOR_SIGMA_DEFAULT,
        help="Width of the Dim 1 informative prior (default: %(default)s; "
        "lower = stronger constraint)",
    )

    parsed = parser.parse_args()
    # --horseshoe-remediate implies --horseshoe-diagnostic
    if parsed.horseshoe_remediate:
        parsed.horseshoe_diagnostic = True
    # --dim1-prior implies --promote-2d (for automatic cross-referencing in report)
    if parsed.dim1_prior:
        parsed.promote_2d = True
    # Research-only flag warnings (production uses canonical routing, ADR-0109)
    if parsed.horseshoe_remediate:
        print(
            "  NOTE: --horseshoe-remediate is retained for research. "
            "Production pipelines use canonical ideal point routing (ADR-0109)."
        )
    if parsed.dim1_prior:
        print(
            "  NOTE: --dim1-prior is retained for research. "
            "Production pipelines use canonical ideal point routing (ADR-0109)."
        )
    return parsed


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


def load_pca_loadings(pca_dir: Path) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load PCA per-bill loadings for PC2-filtered vote selection.

    Returns (house_loadings, senate_loadings). Either may be None if the
    loadings parquet doesn't exist (older PCA runs).
    """
    house_path = pca_dir / "data" / "pc_loadings_house.parquet"
    senate_path = pca_dir / "data" / "pc_loadings_senate.parquet"
    house = pl.read_parquet(house_path) if house_path.exists() else None
    senate = pl.read_parquet(senate_path) if senate_path.exists() else None
    return house, senate


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
            if (vid_to_motion.get(v) or "").lower()
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
    house_legs = legislators.filter(pl.col("chamber") == "House").select(
        "legislator_slug", "full_name"
    )
    senate_legs = legislators.filter(pl.col("chamber") == "Senate").select(
        "legislator_slug", "full_name"
    )
    bridging: list[dict] = []
    for h_row in house_legs.iter_rows(named=True):
        match = senate_legs.filter(pl.col("full_name") == h_row["full_name"])
        if match.height > 0:
            bridging.append(
                {
                    "house_slug": h_row["legislator_slug"],
                    "senate_slug": match["legislator_slug"][0],
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
        sen_meta = legislators.filter(pl.col("legislator_slug") == senate_slug)
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


# ── Identification Strategy Selection ────────────────────────────────────────


def detect_supermajority(
    legislators: pl.DataFrame,
    chamber: str,
) -> tuple[bool, float]:
    """Check if chamber has a supermajority (one party ≥ SUPERMAJORITY_THRESHOLD).

    Returns (is_supermajority, majority_fraction).
    """
    chamber_legs = legislators.filter(pl.col("chamber") == chamber)
    if chamber_legs.height == 0:
        return False, 0.0
    party_counts = chamber_legs.group_by("party").len()
    total = chamber_legs.height
    max_fraction = float(party_counts["len"].max()) / total
    return max_fraction >= SUPERMAJORITY_THRESHOLD, max_fraction


def select_identification_strategy(
    requested: str,
    legislators: pl.DataFrame,
    matrix: pl.DataFrame,
    chamber: str,
    external_scores_available: bool = False,
) -> tuple[str, dict[str, str]]:
    """Select the identification strategy for this chamber.

    When requested="auto", selects based on chamber composition:
    - External scores available → external-prior
    - Supermajority chamber → anchor-agreement
    - Balanced chamber → anchor-pca

    Returns (selected_strategy, rationale_dict). The rationale_dict maps each
    strategy to a short explanation of why it was or wasn't selected, for the
    HTML report.
    """
    IS = IdentificationStrategy

    # Build rationale for every strategy
    is_super, majority_frac = detect_supermajority(legislators, chamber)

    # Count parties
    chamber_legs = legislators.filter(pl.col("chamber") == chamber)
    slug_col = "legislator_slug"
    in_matrix = set(matrix[slug_col].to_list())
    matrix_legs = chamber_legs.filter(pl.col("legislator_slug").is_in(in_matrix))
    n_r = matrix_legs.filter(pl.col("party") == "Republican").height
    n_d = matrix_legs.filter(pl.col("party") == "Democrat").height
    both_parties = n_r >= 3 and n_d >= 3

    # Check contested votes for agreement feasibility
    agree_rates, n_contested = compute_cross_party_agreement(matrix, legislators)
    agreement_feasible = n_contested >= MIN_CONTESTED_FOR_AGREEMENT and len(agree_rates) >= 6

    rationale: dict[str, str] = {}

    # Evaluate each strategy's suitability
    rationale[IS.ANCHOR_PCA] = "Standard method. " + (
        "Risk: supermajority horseshoe effect may distort PCA extremes."
        if is_super
        else "Suitable for this balanced chamber."
    )

    rationale[IS.ANCHOR_AGREEMENT] = (
        f"{n_contested} contested votes, {len(agree_rates)} legislators with agreement data. "
        + (
            "Sufficient for agreement-based anchors."
            if agreement_feasible
            else "Insufficient contested votes — not feasible."
        )
    )

    rationale[IS.SORT_CONSTRAINT] = (
        "No individual anchors — all legislators are free parameters. "
        + (
            "Both parties present."
            if both_parties
            else "Requires both parties — not feasible for single-party chamber."
        )
    )

    rationale[IS.POSITIVE_BETA] = (
        "Forces all discrimination positive. Limitation: silences D-Yea bills "
        "(~12.5% of roll calls in typical Kansas sessions). ADR-0047."
    )

    rationale[IS.HIERARCHICAL_PRIOR] = "Soft identification via party-informed priors. " + (
        "Both parties present." if both_parties else "Requires both parties — not feasible."
    )

    rationale[IS.UNCONSTRAINED] = (
        "No identification during MCMC. Relies entirely on post-hoc sign correction. "
        "Risk: poor chain mixing in multimodal posterior."
    )

    rationale[IS.EXTERNAL_PRIOR] = "Shor-McCarty scores as informative priors. " + (
        "Scores available for this biennium."
        if external_scores_available
        else "No external scores available — not feasible."
    )

    # If not auto, return the requested strategy
    if requested != IS.AUTO:
        rationale[requested] = "SELECTED (user override). " + rationale[requested]
        for s in IS.ALL_STRATEGIES:
            if s != requested:
                rationale[s] = "Not selected. " + rationale[s]
        return requested, rationale

    # Auto-detection logic
    selected = IS.ANCHOR_PCA  # default

    if external_scores_available:
        selected = IS.EXTERNAL_PRIOR
    elif is_super and agreement_feasible:
        selected = IS.ANCHOR_AGREEMENT
    elif is_super and not agreement_feasible:
        # Supermajority but not enough contested votes — sort constraint is safer
        selected = IS.SORT_CONSTRAINT if both_parties else IS.ANCHOR_PCA
    # else: balanced chamber → PCA anchors (default)

    for s in IS.ALL_STRATEGIES:
        prefix = "SELECTED (auto). " if s == selected else "Not selected. "
        rationale[s] = prefix + rationale[s]

    return selected, rationale


def compute_cross_party_agreement(
    matrix: pl.DataFrame,
    legislators: pl.DataFrame,
) -> tuple[dict[str, float], int]:
    """Compute cross-party contested vote agreement for each legislator.

    For each legislator, measures what fraction of contested votes (where both
    parties split ≥ CONTESTED_VOTE_THRESHOLD) they agree with the opposite
    party's majority position.

    Returns (slug→agreement_rate dict, n_contested_votes).
    """
    slug_col = "legislator_slug"
    vote_cols = [c for c in matrix.columns if c != slug_col]
    slugs = matrix[slug_col].to_list()

    party_map = dict(
        zip(
            legislators["legislator_slug"].to_list(),
            legislators["party"].to_list(),
        )
    )

    slug_parties = [party_map.get(s, "Unknown") for s in slugs]
    r_indices = np.array([i for i, p in enumerate(slug_parties) if p == "Republican"])
    d_indices = np.array([i for i, p in enumerate(slug_parties) if p == "Democrat"])

    if len(r_indices) < 3 or len(d_indices) < 3:
        return {}, 0

    # Build numpy vote matrix (legislators × votes), NaN for absent
    vote_array = np.full((len(slugs), len(vote_cols)), np.nan)
    for i, row in enumerate(matrix.iter_rows(named=True)):
        for j, vc in enumerate(vote_cols):
            if row[vc] is not None:
                vote_array[i, j] = float(row[vc])

    # Identify contested votes
    contested_mask = np.zeros(len(vote_cols), dtype=bool)
    for j in range(len(vote_cols)):
        r_votes = vote_array[r_indices, j]
        d_votes = vote_array[d_indices, j]
        r_valid = r_votes[~np.isnan(r_votes)]
        d_valid = d_votes[~np.isnan(d_votes)]
        if len(r_valid) < 3 or len(d_valid) < 3:
            continue
        r_yea_frac = r_valid.mean()
        d_yea_frac = d_valid.mean()
        threshold = CONTESTED_VOTE_THRESHOLD
        r_contested = threshold <= r_yea_frac <= (1 - threshold)
        d_contested = threshold <= d_yea_frac <= (1 - threshold)
        if r_contested and d_contested:
            contested_mask[j] = True

    n_contested = int(contested_mask.sum())
    if n_contested < MIN_CONTESTED_FOR_AGREEMENT:
        return {}, n_contested

    contested_votes = vote_array[:, contested_mask]
    r_majority = np.nanmean(contested_votes[r_indices], axis=0) >= 0.5
    d_majority = np.nanmean(contested_votes[d_indices], axis=0) >= 0.5

    agreement_rates: dict[str, float] = {}
    for i, slug in enumerate(slugs):
        party = party_map.get(slug, "Unknown")
        if party == "Republican":
            opposite_majority = d_majority.astype(float)
        elif party == "Democrat":
            opposite_majority = r_majority.astype(float)
        else:
            continue

        leg_votes = contested_votes[i]
        valid = ~np.isnan(leg_votes)
        if valid.sum() < MIN_CONTESTED_VOTES_PER_LEG:
            continue
        agreement_rates[slug] = float((leg_votes[valid] == opposite_majority[valid]).mean())

    return agreement_rates, n_contested


def select_anchors(
    pca_scores: pl.DataFrame,
    matrix: pl.DataFrame,
    chamber: str,
    legislators: pl.DataFrame | None = None,
) -> tuple[int, str, int, str, dict[str, float] | None]:
    """Select conservative and liberal anchors for IRT identification.

    Primary method: cross-party contested vote agreement. Picks the Republican
    with the LOWEST agreement with Democrats (most partisan R) and the Democrat
    with the LOWEST agreement with Republicans (most partisan D). This selects
    genuine ideological extremes rather than PCA artifacts from the horseshoe
    effect.

    Fallback: party-aware PC1 extremes (when insufficient contested votes or
    legislator metadata is unavailable).

    Guards: anchor must have >= 50% participation in the filtered matrix.

    Returns (cons_idx, cons_slug, lib_idx, lib_slug, agreement_rates_or_None).
    The agreement_rates dict is returned when agreement-based selection was used,
    None when PCA fallback was used. Callers can use agreement rates to build
    ideology-aware chain initialization.
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

    eligible_slugs = {s for s, p in participation.items() if p >= MIN_PARTICIPATION_FOR_ANCHOR}

    # Try agreement-based anchor selection first
    if legislators is not None:
        agreement_rates, n_contested = compute_cross_party_agreement(matrix, legislators)
        if n_contested >= MIN_CONTESTED_FOR_AGREEMENT and len(agreement_rates) >= 6:
            party_map = dict(
                zip(
                    legislators["legislator_slug"].to_list(),
                    legislators["party"].to_list(),
                )
            )

            # Filter to eligible (sufficient participation) legislators with agreement data
            r_candidates = [
                (slug, rate)
                for slug, rate in agreement_rates.items()
                if party_map.get(slug) == "Republican" and slug in eligible_slugs
            ]
            d_candidates = [
                (slug, rate)
                for slug, rate in agreement_rates.items()
                if party_map.get(slug) == "Democrat" and slug in eligible_slugs
            ]

            if len(r_candidates) >= 3 and len(d_candidates) >= 3:
                # Conservative anchor: R with LOWEST cross-party agreement (most partisan)
                r_candidates.sort(key=lambda x: x[1])
                cons_slug = r_candidates[0][0]
                cons_agree = r_candidates[0][1]

                # Liberal anchor: D with LOWEST cross-party agreement (most partisan)
                d_candidates.sort(key=lambda x: x[1])
                lib_slug = d_candidates[0][0]
                lib_agree = d_candidates[0][1]

                cons_idx = slugs.index(cons_slug)
                lib_idx = slugs.index(lib_slug)

                # Look up names from PCA scores or legislators
                name_map = dict(
                    zip(
                        pca_scores["legislator_slug"].to_list(),
                        pca_scores["full_name"].to_list(),
                    )
                )
                cons_name = name_map.get(cons_slug, cons_slug)
                lib_name = name_map.get(lib_slug, lib_slug)

                print("  Anchor method: cross-party contested vote agreement")
                print(
                    f"    {n_contested} contested votes, "
                    f"{len(r_candidates)} R / {len(d_candidates)} D candidates"
                )
                print(
                    f"  Conservative anchor: {cons_name} ({cons_slug}), "
                    f"D-agreement={cons_agree:.1%}"
                )
                print(
                    f"  Liberal anchor:      {lib_name} ({lib_slug}), R-agreement={lib_agree:.1%}"
                )

                return cons_idx, cons_slug, lib_idx, lib_slug, agreement_rates

            print(
                "  Agreement-based anchors: insufficient candidates per party, falling back to PCA"
            )
        elif n_contested < MIN_CONTESTED_FOR_AGREEMENT:
            print(
                f"  Agreement-based anchors: only {n_contested} contested votes "
                f"(need {MIN_CONTESTED_FOR_AGREEMENT}), falling back to PCA"
            )
        else:
            print(
                f"  Agreement-based anchors: only {len(agreement_rates)} legislators "
                f"with data (need 6), falling back to PCA"
            )

    # Fallback: party-aware PCA PC1 extremes
    eligible = (
        pca_scores.filter(pl.col("legislator_slug").is_in(slugs))
        .with_columns(
            pl.col("legislator_slug").replace_strict(participation).alias("participation")
        )
        .filter(pl.col("participation") >= MIN_PARTICIPATION_FOR_ANCHOR)
    )

    republicans = eligible.filter(pl.col("party") == "Republican").sort("PC1", descending=True)
    democrats = eligible.filter(pl.col("party") == "Democrat").sort("PC1", descending=False)

    if republicans.height > 0 and democrats.height > 0:
        cons_slug = republicans["legislator_slug"][0]
        cons_name = republicans["full_name"][0]
        cons_pc1 = republicans["PC1"][0]

        lib_slug = democrats["legislator_slug"][0]
        lib_name = democrats["full_name"][0]
        lib_pc1 = democrats["PC1"][0]
    else:
        sorted_scores = eligible.sort("PC1", descending=True)
        cons_slug = sorted_scores["legislator_slug"][0]
        cons_name = sorted_scores["full_name"][0]
        cons_pc1 = sorted_scores["PC1"][0]
        lib_slug = sorted_scores["legislator_slug"][-1]
        lib_name = sorted_scores["full_name"][-1]
        lib_pc1 = sorted_scores["PC1"][-1]
        print("  WARNING: Single-party chamber — using raw PC1 extremes for anchors")

    cons_idx = slugs.index(cons_slug)
    lib_idx = slugs.index(lib_slug)

    print("  Anchor method: PCA PC1 extremes (party-aware)")
    print(f"  Conservative anchor: {cons_name} ({cons_slug}), PC1={cons_pc1:+.3f}")
    print(f"  Liberal anchor:      {lib_name} ({lib_slug}), PC1={lib_pc1:+.3f}")

    return cons_idx, cons_slug, lib_idx, lib_slug, None


# ── Sign Validation ──────────────────────────────────────────────────────────


def validate_sign(
    idata: az.InferenceData,
    matrix: pl.DataFrame,
    legislators: pl.DataFrame,
    data: dict,
    chamber: str,
) -> tuple[az.InferenceData, bool]:
    """Post-hoc sign validation using cross-party contested vote agreement.

    Detects and corrects sign flips caused by the horseshoe effect in
    supermajority chambers. See docs/irt-sign-identification-deep-dive.md.

    Algorithm:
      1. Compute cross-party agreement via compute_cross_party_agreement().
      2. Correlate Republican ideal points with their D-agreement rate.
         Correct sign: negative correlation (moderates agree more with opposite party).
         Flipped sign: positive correlation (extremes agree more).
      3. If flipped (r > 0, p < 0.10), negate xi and beta posteriors.

    Returns (possibly-corrected idata, was_flipped).
    """
    slugs = data["leg_slugs"]

    agreement_rates, n_contested = compute_cross_party_agreement(matrix, legislators)
    print(f"  Sign validation: {n_contested} contested votes (of {len(matrix.columns) - 1})")

    if n_contested < MIN_CONTESTED_FOR_AGREEMENT:
        print(
            f"  Sign validation: skipped (fewer than {MIN_CONTESTED_FOR_AGREEMENT} contested votes)"
        )
        return idata, False

    if len(agreement_rates) < 10:
        print("  Sign validation: skipped (fewer than 10 legislators with agreement data)")
        return idata, False

    # Build party lookup for filtering
    party_map = dict(
        zip(
            legislators["legislator_slug"].to_list(),
            legislators["party"].to_list(),
        )
    )

    # Get ideal points
    xi_mean = idata.posterior["xi"].mean(dim=["chain", "draw"]).values

    # Correlate R xi with R D-agreement. Correct sign → negative.
    slug_to_idx = {s: i for i, s in enumerate(slugs)}
    r_xi_vals = []
    r_agree_vals = []
    for slug, rate in agreement_rates.items():
        if party_map.get(slug) == "Republican" and slug in slug_to_idx:
            r_xi_vals.append(xi_mean[slug_to_idx[slug]])
            r_agree_vals.append(rate)

    if len(r_xi_vals) < 5:
        print("  Sign validation: skipped (fewer than 5 Republicans with agreement data)")
        return idata, False

    corr, pval = stats.spearmanr(r_xi_vals, r_agree_vals)
    print(f"  Sign validation: R xi vs D-agreement Spearman r = {corr:+.3f} (p = {pval:.3f})")

    if corr > 0 and pval < 0.10:
        print("  Sign validation: SIGN FLIP DETECTED — negating xi and beta posteriors")

        idata.posterior["xi"] = -idata.posterior["xi"]
        idata.posterior["xi_free"] = -idata.posterior["xi_free"]
        idata.posterior["beta"] = -idata.posterior["beta"]

        r_mean_before = np.mean(r_xi_vals)
        print(f"    R mean before flip: {r_mean_before:+.3f}")
        print(f"    R mean after flip:  {-r_mean_before:+.3f}")

        return idata, True

    print("  Sign validation: sign is correct (no flip needed)")
    return idata, False


# ── Robustness Flag Functions ────────────────────────────────────────────────


def filter_contested_votes(
    matrix: pl.DataFrame,
    legislators: pl.DataFrame,
) -> tuple[pl.DataFrame, int, int]:
    """Filter vote matrix to keep only cross-party contested vote columns.

    A contested vote is one where both parties have between CONTESTED_VOTE_THRESHOLD
    and 1-CONTESTED_VOTE_THRESHOLD of their members voting Yea.

    Returns (filtered_matrix, n_contested, n_total).
    """
    slug_col = "legislator_slug"
    vote_cols = [c for c in matrix.columns if c != slug_col]
    n_total = len(vote_cols)
    slugs = matrix[slug_col].to_list()

    party_map = dict(
        zip(
            legislators["legislator_slug"].to_list(),
            legislators["party"].to_list(),
        )
    )

    slug_parties = [party_map.get(s, "Unknown") for s in slugs]
    r_indices = np.array([i for i, p in enumerate(slug_parties) if p == "Republican"])
    d_indices = np.array([i for i, p in enumerate(slug_parties) if p == "Democrat"])

    if len(r_indices) < 3 or len(d_indices) < 3:
        return matrix, 0, n_total

    vote_array = np.full((len(slugs), len(vote_cols)), np.nan)
    for i, row in enumerate(matrix.iter_rows(named=True)):
        for j, vc in enumerate(vote_cols):
            if row[vc] is not None:
                vote_array[i, j] = float(row[vc])

    contested_cols: list[str] = []
    threshold = CONTESTED_VOTE_THRESHOLD
    for j in range(len(vote_cols)):
        r_votes = vote_array[r_indices, j]
        d_votes = vote_array[d_indices, j]
        r_valid = r_votes[~np.isnan(r_votes)]
        d_valid = d_votes[~np.isnan(d_votes)]
        if len(r_valid) < 3 or len(d_valid) < 3:
            continue
        r_yea_frac = r_valid.mean()
        d_yea_frac = d_valid.mean()
        if threshold <= r_yea_frac <= (1 - threshold) and threshold <= d_yea_frac <= (
            1 - threshold
        ):
            contested_cols.append(vote_cols[j])

    if not contested_cols:
        return matrix.select([slug_col]), 0, n_total

    filtered = matrix.select([slug_col, *contested_cols])
    return filtered, len(contested_cols), n_total


def filter_pc2_dominant_votes(
    matrix: pl.DataFrame,
    pca_loadings: pl.DataFrame,
) -> tuple[pl.DataFrame, int, int]:
    """Filter vote matrix to keep only votes where |PC2 loading| > |PC1 loading|.

    These are votes that discriminate more on ideology than establishment-loyalty.
    Used for horseshoe remediation: removing establishment-axis votes lets the
    1D IRT model recover the ideology dimension.

    Returns (filtered_matrix, n_kept, n_total).
    """
    slug_col = "legislator_slug"
    vote_ids = [c for c in matrix.columns if c != slug_col]
    n_total = len(vote_ids)

    loadings = pca_loadings.filter(pl.col("vote_id").is_in(vote_ids))
    pc2_dominant = set(
        loadings.filter(pl.col("PC2").abs() > pl.col("PC1").abs())["vote_id"].to_list()
    )

    if not pc2_dominant:
        return matrix, n_total, n_total

    keep_cols = [slug_col] + [v for v in vote_ids if v in pc2_dominant]
    filtered = matrix.select(keep_cols)
    return filtered, len(keep_cols) - 1, n_total


def detect_horseshoe(
    ideal_points: pl.DataFrame,
    pca_scores: pl.DataFrame,
    chamber: str,
) -> dict:
    """Quantitative horseshoe detection for a chamber.

    Checks:
    1. Democrat wrong-side fraction: how many Democrats have xi > 0?
    2. Party overlap: do R and D ideal point distributions overlap?
    3. PCA eigenvalue ratio: is PC1 dominant or near-equal to PC2?
    4. Most extreme R check: is any Republican more "liberal" than D mean?

    Returns a dict with detection results.
    """
    ip = ideal_points
    r_ip = ip.filter(pl.col("party") == "Republican")
    d_ip = ip.filter(pl.col("party") == "Democrat")

    # 1. Democrat wrong-side fraction
    if d_ip.height > 0:
        d_wrong_side = float((d_ip["xi_mean"] > 0).mean())
    else:
        d_wrong_side = 0.0

    # 2. Party overlap: fraction of Republicans below D mean or Ds above R mean
    if r_ip.height > 0 and d_ip.height > 0:
        r_mean = float(r_ip["xi_mean"].mean())
        d_mean = float(d_ip["xi_mean"].mean())
        r_below_d_mean = float((r_ip["xi_mean"] < d_mean).mean())
        d_above_r_mean = float((d_ip["xi_mean"] > r_mean).mean())
        overlap_frac = (r_below_d_mean + d_above_r_mean) / 2
    else:
        r_mean = d_mean = overlap_frac = 0.0

    # 3. PCA eigenvalue ratio (approximate from score variance)
    pca_chamber = pca_scores
    if "PC1" in pca_chamber.columns and "PC2" in pca_chamber.columns:
        pc1_var = float(pca_chamber["PC1"].var())
        pc2_var = float(pca_chamber["PC2"].var())
        eigenvalue_ratio = pc1_var / pc2_var if pc2_var > 0 else float("inf")
    else:
        eigenvalue_ratio = float("inf")

    # 4. Most extreme R: any R more "liberal" than D mean?
    if r_ip.height > 0 and d_ip.height > 0:
        most_neg_r = float(r_ip["xi_mean"].min())
        r_more_liberal_than_d_mean = most_neg_r < d_mean
        most_neg_r_name = r_ip.filter(pl.col("xi_mean") == r_ip["xi_mean"].min())["full_name"][0]
    else:
        r_more_liberal_than_d_mean = False
        most_neg_r_name = "N/A"
        most_neg_r = 0.0

    # Detection verdict
    horseshoe_detected = (
        d_wrong_side > HORSESHOE_DEM_WRONG_SIDE_FRAC
        or r_more_liberal_than_d_mean
        or overlap_frac > 0.30
    )

    return {
        "detected": horseshoe_detected,
        "dem_wrong_side_frac": d_wrong_side,
        "overlap_frac": overlap_frac,
        "eigenvalue_ratio": eigenvalue_ratio,
        "r_mean": r_mean,
        "d_mean": d_mean,
        "r_more_liberal_than_d_mean": r_more_liberal_than_d_mean,
        "most_neg_r_name": most_neg_r_name,
        "most_neg_r_xi": most_neg_r,
        "n_republicans": r_ip.height,
        "n_democrats": d_ip.height,
    }


def cross_reference_2d(
    ideal_points_1d: pl.DataFrame,
    irt_2d_dir: Path,
    chamber: str,
) -> dict | None:
    """Cross-reference 1D ideal points with 2D IRT results.

    Returns None if 2D results not available.
    Returns dict with flagged legislators and rank comparisons.
    """
    fname = f"ideal_points_2d_{chamber.lower()}.parquet"
    path_2d = irt_2d_dir / "data" / fname
    if not path_2d.exists():
        print(f"  2D cross-reference: {fname} not found in {irt_2d_dir / 'data'}")
        return None

    ip_2d = pl.read_parquet(path_2d)

    # Check that the 2D data has the expected columns
    dim1_col = None
    for candidate in ["xi_dim1_mean", "dim1_mean", "xi_mean_dim1"]:
        if candidate in ip_2d.columns:
            dim1_col = candidate
            break

    if dim1_col is None:
        print("  2D cross-reference: no dimension-1 column found in 2D results")
        return None

    # Merge 1D and 2D on slug
    ip_1d = ideal_points_1d.select(
        "legislator_slug",
        "full_name",
        "party",
        pl.col("xi_mean").alias("xi_1d"),
    )

    ip_2d_slim = ip_2d.select(
        "legislator_slug",
        pl.col(dim1_col).alias("xi_2d_dim1"),
    )

    merged = ip_1d.join(ip_2d_slim, on="legislator_slug", how="inner")
    if merged.height == 0:
        print("  2D cross-reference: no matching legislators between 1D and 2D")
        return None

    # Compute ranks (descending — most conservative = rank 1)
    merged = (
        merged.with_columns(
            pl.col("xi_1d").rank(descending=True).alias("rank_1d"),
            pl.col("xi_2d_dim1").rank(descending=True).alias("rank_2d"),
        )
        .with_columns(
            (pl.col("rank_1d") - pl.col("rank_2d")).abs().alias("rank_shift"),
        )
        .sort("rank_shift", descending=True)
    )

    # Flag legislators with large rank shifts
    flagged = merged.filter(pl.col("rank_shift") > PROMOTE_2D_RANK_SHIFT)

    # Correlation between 1D and 2D Dim1
    corr_1d_2d = float(merged.select(pl.corr("xi_1d", "xi_2d_dim1")).item())

    return {
        "n_matched": merged.height,
        "n_flagged": flagged.height,
        "correlation": corr_1d_2d,
        "flagged_legislators": flagged.to_dicts(),
        "all_comparisons": merged.to_dicts(),
    }


# ── Phase 3: Build and Sample IRT Model ──────────────────────────────────────


def build_irt_graph(
    data: dict,
    anchors: list[tuple[int, float]],
    strategy: str = IdentificationStrategy.ANCHOR_PCA,
    party_indices: dict[str, list[int]] | None = None,
    external_priors: np.ndarray | None = None,
    external_prior_sigma: float = 0.5,
) -> pm.Model:
    """Build 2PL IRT model graph with configurable identification strategy.

    The strategy parameter controls how the model resolves reflection invariance.
    See IdentificationStrategy for the full catalog with literature references.

    Model structure (varies by strategy):
        xi ~ [depends on strategy] for legislator ideal points
        alpha ~ Normal(0, 5)  -- bill difficulty
        beta ~ Normal(0, 1) or HalfNormal(1)  -- bill discrimination
        P(Yea) = logit^-1(beta * xi - alpha)

    Args:
        data: IRT data dict from prepare_irt_data().
        anchors: List of (index, value) pairs for anchor-based strategies.
            Empty list for constraint-based and unconstrained strategies.
        strategy: IdentificationStrategy constant.
        party_indices: {"Republican": [idx, ...], "Democrat": [idx, ...]} for
            sort-constraint and hierarchical-prior strategies.
        external_priors: Per-legislator prior means (shape n_leg) for
            external-prior strategy. Typically from Shor-McCarty or PC2 scores.
        external_prior_sigma: Width of the external prior (default 0.5 for
            Shor-McCarty; use 1.0 for PC2 horseshoe remediation).

    Returns the PyMC model for use with nutpie or pm.sample().
    """
    IS = IdentificationStrategy
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

    use_positive_beta = strategy == IS.POSITIVE_BETA

    with pm.Model(coords=coords) as model:
        # --- Legislator ideal points (strategy-dependent) ---
        if strategy in (IS.ANCHOR_PCA, IS.ANCHOR_AGREEMENT):
            # Hard anchor strategies: fix anchor legislators at ±1
            xi_free = pm.Normal("xi_free", mu=0, sigma=1, shape=n_leg - n_anchors)
            xi_raw = pt.zeros(n_leg)
            for anchor_idx, anchor_val in anchors:
                xi_raw = pt.set_subtensor(xi_raw[anchor_idx], anchor_val)
            free_positions = [i for i in range(n_leg) if i not in anchor_indices]
            for k, pos in enumerate(free_positions):
                xi_raw = pt.set_subtensor(xi_raw[pos], xi_free[k])
            xi = pm.Deterministic("xi", xi_raw, dims="legislator")

        elif strategy == IS.HIERARCHICAL_PRIOR:
            # Party-informed soft prior: R → Normal(+0.5, 1), D → Normal(-0.5, 1)
            prior_mu = np.zeros(n_leg)
            if party_indices:
                for idx in party_indices.get("Republican", []):
                    prior_mu[idx] = 0.5
                for idx in party_indices.get("Democrat", []):
                    prior_mu[idx] = -0.5
            xi_free = pm.Normal("xi_free", mu=prior_mu, sigma=1, shape=n_leg)
            xi = pm.Deterministic("xi", xi_free, dims="legislator")

        elif strategy == IS.EXTERNAL_PRIOR:
            # Informative prior from external scores (e.g., Shor-McCarty, PC2)
            if external_priors is None:
                external_priors = np.zeros(n_leg)
            xi_free = pm.Normal(
                "xi_free", mu=external_priors, sigma=external_prior_sigma, shape=n_leg
            )
            xi = pm.Deterministic("xi", xi_free, dims="legislator")

        else:
            # Sort constraint, positive beta, or unconstrained:
            # all legislators are free parameters
            xi_free = pm.Normal("xi_free", mu=0, sigma=1, shape=n_leg)
            xi = pm.Deterministic("xi", xi_free, dims="legislator")

        # --- Sort constraint (soft penalty) ---
        if strategy == IS.SORT_CONSTRAINT and party_indices:
            r_indices = party_indices.get("Republican", [])
            d_indices = party_indices.get("Democrat", [])
            if r_indices and d_indices:
                r_mean = xi[r_indices].mean()
                d_mean = xi[d_indices].mean()
                # Large negative penalty when D mean >= R mean
                pm.Potential(
                    "party_order",
                    pt.switch(r_mean > d_mean, 0.0, -1e6),
                )

        # --- Roll call parameters ---
        alpha = pm.Normal("alpha", mu=0, sigma=5, shape=n_votes, dims="vote")

        if use_positive_beta:
            # Positive discrimination: forces sign identification via beta > 0.
            # Trade-off: silences D-Yea bills where beta should be negative.
            beta = pm.HalfNormal("beta", sigma=1, shape=n_votes, dims="vote")
        else:
            # Unconstrained: allows negative beta (liberal-Yea bills).
            # Sign from anchors/constraints/post-hoc correction.
            beta = pm.Normal("beta", mu=0, sigma=1, shape=n_votes, dims="vote")

        # --- Likelihood ---
        eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
        pm.Bernoulli("obs", logit_p=eta, observed=y, dims="obs_id")

    return model


def build_and_sample(
    data: dict,
    anchors: list[tuple[int, float]],
    n_samples: int,
    n_tune: int,
    n_chains: int,
    target_accept: float = TARGET_ACCEPT,
    xi_initvals: np.ndarray | None = None,
    strategy: str = IdentificationStrategy.ANCHOR_PCA,
    party_indices: dict[str, list[int]] | None = None,
    external_priors: np.ndarray | None = None,
    external_prior_sigma: float = 0.5,
) -> tuple[az.InferenceData, float]:
    """Build 2PL IRT model and sample with nutpie.

    Builds the model graph via build_irt_graph() with the specified
    identification strategy, then compiles and samples with nutpie's
    Rust NUTS sampler (ADR-0053).

    Args:
        data: IRT data dict from prepare_irt_data().
        anchors: List of (legislator_index, fixed_value) pairs. Empty for
            constraint-based strategies.
        n_samples: MCMC posterior draws per chain.
        n_tune: MCMC tuning steps (discarded).
        n_chains: Number of independent MCMC chains.
        target_accept: Accepted for API compatibility but ignored.
            nutpie uses adaptive dual averaging.
        xi_initvals: Optional initial xi values for free parameters.
        strategy: IdentificationStrategy constant.
        party_indices: Party membership indices for constraint strategies.
        external_priors: Per-legislator prior means for external-prior strategy.
        external_prior_sigma: Width of external prior (0.5 for Shor-McCarty,
            1.0 for PC2 horseshoe remediation).

    Returns (InferenceData, sampling_time_seconds).
    """
    if target_accept != TARGET_ACCEPT:
        print(
            f"  Note: target_accept={target_accept} ignored (nutpie uses adaptive dual averaging)"
        )

    model = build_irt_graph(
        data,
        anchors,
        strategy=strategy,
        party_indices=party_indices,
        external_priors=external_priors,
        external_prior_sigma=external_prior_sigma,
    )
    n_anchors = len(anchors)

    # --- Compile with nutpie ---
    compile_kwargs: dict = {}
    if xi_initvals is not None:
        # Strategy-informed init for xi_free; jitter all OTHER RVs.
        compile_kwargs["initial_points"] = {"xi_free": xi_initvals}
        compile_kwargs["jitter_rvs"] = {rv for rv in model.free_RVs if rv.name != "xi_free"}
        print(f"  Informed initvals: {len(xi_initvals)} free parameters")
        jittered = [rv.name for rv in compile_kwargs["jitter_rvs"]]
        print(f"  jitter_rvs: {jittered} (xi_free excluded)")

    print("  Compiling model with nutpie...")
    compiled = nutpie.compile_pymc_model(model, **compile_kwargs)

    # --- Sample ---
    print(f"  Sampling: {n_samples} draws, {n_tune} tune, {n_chains} chains")
    print(f"  seed={RANDOM_SEED}, sampler=nutpie (Rust NUTS)")
    print(f"  Anchors: {n_anchors} fixed legislators")

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

    # Bulk ESS
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
    print(f"  Bulk ESS (xi):      min = {xi_ess_min:.0f}  {xi_ess_status}")
    print(f"  Bulk ESS (alpha):   min = {alpha_ess_min:.0f}  {alpha_ess_status}")
    print(f"  Bulk ESS (beta):    min = {beta_ess_min:.0f}  {beta_ess_status}")

    # Tail ESS (Vehtari et al. 2021) — catches poor mixing in posterior tails
    ess_tail = az.ess(idata, method="tail")
    xi_tail_min = float(ess_tail["xi"].min())
    alpha_tail_min = float(ess_tail["alpha"].min())
    beta_tail_min = float(ess_tail["beta"].min())
    diag["xi_tail_ess_min"] = xi_tail_min
    diag["alpha_tail_ess_min"] = alpha_tail_min
    diag["beta_tail_ess_min"] = beta_tail_min

    tail_ok = min(xi_tail_min, alpha_tail_min, beta_tail_min) > ESS_THRESHOLD
    xi_tail_status = "OK" if xi_tail_min > ESS_THRESHOLD else "WARNING"
    alpha_tail_status = "OK" if alpha_tail_min > ESS_THRESHOLD else "WARNING"
    beta_tail_status = "OK" if beta_tail_min > ESS_THRESHOLD else "WARNING"
    print(f"  Tail ESS (xi):      min = {xi_tail_min:.0f}  {xi_tail_status}")
    print(f"  Tail ESS (alpha):   min = {alpha_tail_min:.0f}  {alpha_tail_status}")
    print(f"  Tail ESS (beta):    min = {beta_tail_min:.0f}  {beta_tail_status}")

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

    diag["all_ok"] = rhat_ok and ess_ok and tail_ok and div_ok and bfmi_ok
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
    meta = legislators.select("legislator_slug", "full_name", "party", "district", "chamber")
    df = df.join(meta, on="legislator_slug", how="left")

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


PARADOX_YEA_GAP = 0.20
PARADOX_MIN_BILLS = 3
MAX_HIGHLIGHTS = 5


def _detect_forest_highlights(
    ideal_points: pl.DataFrame,
    chamber: str,
) -> dict[str, str]:
    """Detect notable legislators for forest plot annotation (data-driven).

    Returns {slug: annotation_text} for up to MAX_HIGHLIGHTS legislators.
    Returns empty dict if fewer than 3 legislators in the DataFrame.
    """
    if ideal_points.height < 3:
        return {}

    highlights: dict[str, str] = {}

    # 1. Most extreme (top 1 by |xi_mean|)
    extreme = (
        ideal_points.with_columns(pl.col("xi_mean").abs().alias("_abs_xi"))
        .sort("_abs_xi", descending=True)
        .row(0, named=True)
    )
    direction = "conservative" if extreme["xi_mean"] > 0 else "liberal"
    highlights[extreme["legislator_slug"]] = f"Most {direction} in the {chamber}"

    # 2. Widest HDI (top 1 by xi_sd) — only if xi_sd > 2× median
    median_sd = ideal_points["xi_sd"].median()
    widest = ideal_points.sort("xi_sd", descending=True).row(0, named=True)
    if widest["xi_sd"] > 2 * median_sd:
        slug = widest["legislator_slug"]
        if slug not in highlights:
            highlights[slug] = "Highest uncertainty (wide HDI)"

    # 3. Most moderate (xi_mean closest to 0) — only if clearly between parties
    has_parties = (
        ideal_points.filter(pl.col("party") == "Republican").height > 0
        and ideal_points.filter(pl.col("party") == "Democrat").height > 0
    )
    if has_parties:
        r_min_xi = (
            ideal_points.filter(pl.col("party") == "Republican").sort("xi_mean").row(0, named=True)
        )
        d_max_xi = (
            ideal_points.filter(pl.col("party") == "Democrat")
            .sort("xi_mean", descending=True)
            .row(0, named=True)
        )
        # Pick whichever is closer to 0 and between parties
        r_candidate = r_min_xi if r_min_xi["xi_mean"] > d_max_xi["xi_mean"] else None
        d_candidate = d_max_xi if d_max_xi["xi_mean"] < r_min_xi["xi_mean"] else None

        moderate = None
        if r_candidate and d_candidate:
            if abs(r_candidate["xi_mean"]) < abs(d_candidate["xi_mean"]):
                moderate = r_candidate
            else:
                moderate = d_candidate
        elif r_candidate:
            moderate = r_candidate
        elif d_candidate:
            moderate = d_candidate

        if moderate and moderate["legislator_slug"] not in highlights:
            highlights[moderate["legislator_slug"]] = f"Most moderate {moderate['party']} member"

    # 4. Widest HDI runner-up (2nd widest) — only if also > 2× median and different from #2
    if ideal_points.height >= 2:
        second_widest = ideal_points.sort("xi_sd", descending=True).row(1, named=True)
        if (
            second_widest["xi_sd"] > 2 * median_sd
            and second_widest["legislator_slug"] not in highlights
        ):
            highlights[second_widest["legislator_slug"]] = "High uncertainty (wide HDI)"

    # Cap at MAX_HIGHLIGHTS
    if len(highlights) > MAX_HIGHLIGHTS:
        highlights = dict(list(highlights.items())[:MAX_HIGHLIGHTS])

    # If fewer than 2 highlights, return empty (not worth cluttering)
    if len(highlights) < 2:
        return {}

    return highlights


def plot_forest(
    ideal_points: pl.DataFrame,
    chamber: str,
    out_dir: Path,
    highlights: dict[str, str] | None = None,
) -> None:
    """Forest plot: ideal points with 95% HDI, party-colored, sorted by xi_mean.

    Args:
        highlights: Optional dict of {slug: annotation_text}. If None, detects
            highlights automatically via _detect_forest_highlights().
    """
    sorted_df = ideal_points.sort("xi_mean")
    n = sorted_df.height

    fig, ax = plt.subplots(figsize=(10, max(14, n * 0.22)))

    # Data-driven highlights (or caller-provided override)
    if highlights is None:
        highlights = _detect_forest_highlights(ideal_points, chamber)

    y_pos = np.arange(n)
    for i, row in enumerate(sorted_df.iter_rows(named=True)):
        slug = row["legislator_slug"]
        color = PARTY_COLORS.get(row["party"], "#888888")
        is_highlight = slug in highlights

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
                highlights[slug],
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


def find_paradox_legislator(
    ideal_points: pl.DataFrame,
    bill_params: pl.DataFrame,
    data: dict,
) -> dict | None:
    """Find a legislator who is ideologically extreme but contrarian on routine bills.

    Within the majority party, finds the legislator with the highest |xi_mean|,
    then checks whether their Yea rate differs between high-discrimination
    (partisan) and low-discrimination (routine) bills.

    Returns a metadata dict if a paradox is found (gap > PARADOX_YEA_GAP),
    or None if no paradox exists.
    """
    # Determine majority party
    party_counts = ideal_points.group_by("party").len().sort("len", descending=True)
    if party_counts.height == 0:
        return None
    majority_party = party_counts.row(0, named=True)["party"]

    # Find most extreme in majority party
    majority = ideal_points.filter(pl.col("party") == majority_party)
    if majority.height == 0:
        return None
    candidate = (
        majority.with_columns(pl.col("xi_mean").abs().alias("_abs_xi"))
        .sort("_abs_xi", descending=True)
        .row(0, named=True)
    )
    slug = candidate["legislator_slug"]

    # Split bills by discrimination
    high_disc_votes = bill_params.filter(pl.col("beta_mean").abs() > HIGH_DISC_THRESHOLD)[
        "vote_id"
    ].to_list()
    low_disc_votes = bill_params.filter(pl.col("beta_mean").abs() < LOW_DISC_THRESHOLD)[
        "vote_id"
    ].to_list()

    if len(high_disc_votes) < PARADOX_MIN_BILLS or len(low_disc_votes) < PARADOX_MIN_BILLS:
        return None

    # Compute Yea rates from raw data
    leg_slugs = data["leg_slugs"]
    vote_ids = data["vote_ids"]
    leg_idx_arr = data["leg_idx"]
    vote_idx_arr = data["vote_idx"]
    y_arr = data["y"]

    if slug not in leg_slugs:
        return None
    slug_idx = leg_slugs.index(slug)

    high_disc_idxs = {vote_ids.index(v) for v in high_disc_votes if v in vote_ids}
    low_disc_idxs = {vote_ids.index(v) for v in low_disc_votes if v in vote_ids}

    # Filter to this legislator's votes
    mask = leg_idx_arr == slug_idx
    leg_vote_idxs = vote_idx_arr[mask]
    leg_votes = y_arr[mask]

    high_mask = np.isin(leg_vote_idxs, list(high_disc_idxs))
    low_mask = np.isin(leg_vote_idxs, list(low_disc_idxs))

    n_high = high_mask.sum()
    n_low = low_mask.sum()

    if n_high < PARADOX_MIN_BILLS or n_low < PARADOX_MIN_BILLS:
        return None

    high_yea_rate = float(leg_votes[high_mask].mean())
    low_yea_rate = float(leg_votes[low_mask].mean())

    # Also compute party average for comparison
    majority_slugs = majority["legislator_slug"].to_list()
    majority_idxs = [leg_slugs.index(s) for s in majority_slugs if s in leg_slugs]
    party_mask = np.isin(leg_idx_arr, majority_idxs)

    party_high_mask = party_mask & np.isin(vote_idx_arr, list(high_disc_idxs))
    party_low_mask = party_mask & np.isin(vote_idx_arr, list(low_disc_idxs))

    party_high_yea = float(y_arr[party_high_mask].mean()) if party_high_mask.sum() > 0 else 0.0
    party_low_yea = float(y_arr[party_low_mask].mean()) if party_low_mask.sum() > 0 else 0.0

    gap = abs(high_yea_rate - low_yea_rate)
    if gap < PARADOX_YEA_GAP:
        return None

    return {
        "legislator_slug": slug,
        "full_name": candidate["full_name"],
        "party": candidate["party"],
        "xi_mean": candidate["xi_mean"],
        "high_disc_yea_rate": high_yea_rate,
        "low_disc_yea_rate": low_yea_rate,
        "party_high_disc_yea_rate": party_high_yea,
        "party_low_disc_yea_rate": party_low_yea,
        "n_high_disc": int(n_high),
        "n_low_disc": int(n_low),
        "gap": gap,
    }


def plot_paradox_spotlight(
    paradox: dict,
    ideal_points: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Two-panel figure showing a legislator's paradoxical voting behavior.

    Left panel: grouped bar chart comparing Yea rates on partisan vs routine bills.
    Right panel: simplified forest plot with the paradox legislator highlighted.
    """
    name = paradox["full_name"]
    last_name = name.split()[-1] if name else "Unknown"
    party = paradox["party"]
    party_color = PARTY_COLORS.get(party, "#888888")
    direction = "conservative" if paradox["xi_mean"] > 0 else "liberal"

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8), width_ratios=[1, 1.2])

    # ── Left panel: grouped bar chart ──
    categories = ["Highly Partisan\nBills", "Routine\nBills"]
    name_rates = [paradox["high_disc_yea_rate"], paradox["low_disc_yea_rate"]]
    party_rates = [paradox["party_high_disc_yea_rate"], paradox["party_low_disc_yea_rate"]]

    x = np.arange(len(categories))
    width = 0.3

    bars1 = ax_left.bar(
        x - width / 2,
        name_rates,
        width,
        label=last_name,
        color=party_color,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax_left.bar(
        x + width / 2,
        party_rates,
        width,
        label=f"{party} average",
        color="#CCCCCC",
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels on bars
    for bar in bars1:
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    for bar in bars2:
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#666666",
        )

    ax_left.set_ylim(0, 1.15)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(categories, fontsize=11)
    ax_left.set_ylabel("Yea Vote Rate", fontsize=11)
    ax_left.set_title(f"How {last_name} Votes Differently by Bill Type", fontsize=12)
    ax_left.legend(loc="upper right", fontsize=10)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    # Annotation callout
    low_pct = f"{paradox['low_disc_yea_rate']:.0%}"
    party_low_pct = f"{paradox['party_low_disc_yea_rate']:.0%}"
    ax_left.text(
        0.5,
        -0.12,
        f"{last_name} votes Yea on {low_pct} of routine bills, vs. the {party}\n"
        f"average of {party_low_pct} ({paradox['n_high_disc']} partisan, "
        f"{paradox['n_low_disc']} routine bills).",
        transform=ax_left.transAxes,
        fontsize=9,
        ha="center",
        va="top",
        style="italic",
        color="#555555",
    )

    # ── Right panel: simplified forest plot (majority party only) ──
    majority = ideal_points.filter(pl.col("party") == party).sort("xi_mean")
    n = majority.height

    for i, row in enumerate(majority.iter_rows(named=True)):
        slug = row["legislator_slug"]
        is_paradox = slug == paradox["legislator_slug"]

        ax_right.hlines(
            i,
            row["xi_hdi_2.5"],
            row["xi_hdi_97.5"],
            colors=party_color,
            alpha=0.9 if is_paradox else 0.3,
            linewidth=3.0 if is_paradox else 1.0,
        )
        ax_right.scatter(
            row["xi_mean"],
            i,
            c=party_color if not is_paradox else "gold",
            s=80 if is_paradox else 15,
            zorder=5,
            edgecolors="black",
            linewidth=1.0 if is_paradox else 0.2,
            marker="D" if is_paradox else "o",
        )

    # Annotate the paradox legislator
    paradox_row = majority.filter(pl.col("legislator_slug") == paradox["legislator_slug"])
    if paradox_row.height > 0:
        pr = paradox_row.row(0, named=True)
        # Find position in sorted list
        sorted_slugs = majority.sort("xi_mean")["legislator_slug"].to_list()
        y_idx = sorted_slugs.index(paradox["legislator_slug"])

        ax_right.annotate(
            f"Most {direction} by IRT,\nyet contrarian on routine bills",
            (pr["xi_hdi_97.5"], y_idx),
            fontsize=8,
            fontstyle="italic",
            color="#333333",
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
            bbox={"boxstyle": "round,pad=0.4", "fc": "lightyellow", "ec": "#CCCCCC", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": "#999999"},
        )

    # Y-axis labels (party members only)
    labels = []
    for row in majority.iter_rows(named=True):
        nm = row.get("full_name", row["legislator_slug"])
        labels.append(nm)
    ax_right.set_yticks(np.arange(n))
    ax_right.set_yticklabels(labels, fontsize=5.5 if n > 30 else 7)
    ax_right.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax_right.set_xlabel("Ideal Point (Liberal \u2190 \u2192 Conservative)")
    ax_right.set_title(
        f"Where {last_name} Sits on the {party} Spectrum",
        fontsize=12,
    )
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    fig.suptitle(
        f"The {last_name} Paradox: Extreme Ideology, Unreliable Loyalty",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.04,
        f"This legislator is the most ideologically extreme {party} in the {chamber},\n"
        f"but defects from the party line on routine, near-unanimous bills.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
    )

    fig.tight_layout()
    save_fig(fig, out_dir / f"paradox_spotlight_{chamber.lower()}.png")


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
    n = min(N_CONVERGENCE_SUMMARY, len(slugs))
    indices = np.linspace(0, len(slugs) - 1, n, dtype=int)
    selected_slugs = [slugs[i] for i in indices]

    n_chains = idata.posterior.sizes.get("chain", 1)
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


# ── Party Density + ICC Curves ───────────────────────────────────────────────


def plot_party_density(
    ideal_points: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Plot overlapping KDE density curves of ideal points per party.

    Red = Republican, Blue = Democrat, Gray = Independent.
    Vertical lines at party means, annotated overlap region.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for party in ["Republican", "Democrat", "Independent"]:
        party_data = ideal_points.filter(pl.col("party") == party)
        if party_data.height < 2:
            continue

        xi_vals = party_data["xi_mean"].to_numpy()
        color = PARTY_COLORS.get(party, "#999999")

        kde = stats.gaussian_kde(xi_vals)
        x_range = np.linspace(xi_vals.min() - 0.5, xi_vals.max() + 0.5, 300)
        density = kde(x_range)

        ax.fill_between(x_range, density, alpha=0.3, color=color)
        ax.plot(x_range, density, color=color, linewidth=2, label=party)
        ax.axvline(
            xi_vals.mean(),
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.set_xlabel("Ideal Point (xi)")
    ax.set_ylabel("Density")
    ax.set_title(f"{chamber} — Party Ideal Point Distributions")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"party_density_{chamber.lower()}.png")


def plot_icc_curves(
    bill_params: pl.DataFrame,
    chamber: str,
    out_dir: Path,
    n_curves: int = 5,
) -> None:
    """Plot Item Characteristic Curves for the most discriminating bills.

    Shows P(Yea | theta) for the top n_curves bills by |beta|, colored by
    the sign of beta (red = conservative-Yea, blue = liberal-Yea).
    """
    bp = bill_params.with_columns(pl.col("beta_mean").abs().alias("abs_beta"))
    top_bills = bp.sort("abs_beta", descending=True).head(n_curves)

    if top_bills.height == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    theta = np.linspace(-3, 3, 300)

    for row in top_bills.iter_rows(named=True):
        a = row["beta_mean"]  # discrimination
        b = row["alpha_mean"]  # difficulty
        prob = 1.0 / (1.0 + np.exp(-(a * theta - b)))
        color = PARTY_COLORS["Republican"] if a > 0 else PARTY_COLORS["Democrat"]
        label_str = row.get("bill_number", row.get("vote_id", ""))
        ax.plot(theta, prob, color=color, linewidth=2, label=label_str, alpha=0.8)

    ax.axhline(0.5, color="#999", linestyle=":", linewidth=1)
    ax.axvline(0, color="#999", linestyle=":", linewidth=1)
    ax.set_xlabel("Ideal Point (theta)")
    ax.set_ylabel("P(Yea | theta)")
    ax.set_title(f"{chamber} — Item Characteristic Curves (Top {n_curves} by |beta|)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"icc_curves_{chamber.lower()}.png")


def plot_ideal_points_interactive(
    ideal_points: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Plotly interactive scatter of ideal points with hover details."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for party, color in PARTY_COLORS.items():
        subset = ideal_points.filter(pl.col("party") == party)
        if subset.is_empty():
            continue

        hover_text = [
            f"{row['full_name']}<br>Party: {row['party']}<br>"
            f"District: {row.get('district', 'N/A')}<br>"
            f"Ideal Point: {row['xi_mean']:.3f}<br>"
            f"SD: {row['xi_sd']:.3f}<br>"
            f"HDI: [{row.get('xi_hdi_2.5', 0):.3f}, {row.get('xi_hdi_97.5', 0):.3f}]"
            for row in subset.iter_rows(named=True)
        ]

        fig.add_trace(
            go.Scatter(
                x=list(range(subset.height)),
                y=subset["xi_mean"].to_list(),
                mode="markers",
                name=party,
                marker={"color": color, "size": 8, "opacity": 0.7},
                text=hover_text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"{chamber} — Ideal Points (hover for details)",
        xaxis_title="Legislator Index (sorted by ideal point)",
        yaxis_title="Ideal Point (xi_mean)",
        hovermode="closest",
        template="plotly_white",
    )
    html = fig.to_html(full_html=False, include_plotlyjs=True)
    (out_dir / f"ideal_points_interactive_{chamber.lower()}.html").write_text(html)
    print(f"  Saved: ideal_points_interactive_{chamber.lower()}.html")


def plot_irt_vs_pca_interactive(
    ideal_points: pl.DataFrame,
    pca_scores: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Plotly interactive scatter of IRT vs PCA with hover details."""
    import plotly.graph_objects as go

    merged = ideal_points.select("legislator_slug", "xi_mean", "party", "full_name").join(
        pca_scores.select("legislator_slug", "PC1"),
        on="legislator_slug",
        how="inner",
    )
    if merged.is_empty():
        return

    xi = merged["xi_mean"].to_numpy()
    pc1 = merged["PC1"].to_numpy()
    pearson_r = float(np.corrcoef(xi, pc1)[0, 1])

    fig = go.Figure()
    for party, color in PARTY_COLORS.items():
        subset = merged.filter(pl.col("party") == party)
        if subset.is_empty():
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["PC1"].to_list(),
                y=subset["xi_mean"].to_list(),
                mode="markers",
                name=party,
                marker={"color": color, "size": 8, "opacity": 0.7},
                text=[
                    f"{row['full_name']}<br>IRT: {row['xi_mean']:.3f}<br>PCA: {row['PC1']:.3f}"
                    for row in subset.iter_rows(named=True)
                ],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"{chamber} — IRT vs PCA (r = {pearson_r:.4f})",
        xaxis_title="PCA PC1",
        yaxis_title="IRT Ideal Point (xi_mean)",
        hovermode="closest",
        template="plotly_white",
    )
    html = fig.to_html(full_html=False, include_plotlyjs=True)
    (out_dir / f"irt_vs_pca_interactive_{chamber.lower()}.html").write_text(html)
    print(f"  Saved: irt_vs_pca_interactive_{chamber.lower()}.html")


# ── Cutting Lines + Swing Votes ──────────────────────────────────────────────

MIN_BETA_FOR_CUTTING = 0.5  # Only compute cutting points for bills with |beta| > this
SWING_VOTE_DISTANCE = 0.5  # Within this distance of cutting point = swing legislator
CLOSE_VOTE_MARGIN = 5  # Margin <= this to be considered a close vote


def compute_cutting_points(
    bill_params: pl.DataFrame,
    ideal_points: pl.DataFrame,
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    chamber: str,
) -> pl.DataFrame:
    """Compute cutting points for discriminating bills.

    Cutting point = alpha / beta — the ideal point where P(Yea) = 0.5.
    For each bill with |beta| > MIN_BETA_FOR_CUTTING, finds the nearest legislator.
    """
    bp = bill_params.filter(pl.col("beta_mean").abs() > MIN_BETA_FOR_CUTTING)
    if bp.is_empty():
        return pl.DataFrame()

    ip_arr = ideal_points.sort("xi_mean")
    ip_slugs = ip_arr["legislator_slug"].to_list()
    ip_vals = ip_arr["xi_mean"].to_numpy()

    # Get margin for each vote
    prefix = "rep_" if chamber == "House" else "sen_"
    chamber_votes = votes.filter(pl.col("legislator_slug").str.starts_with(prefix))
    margins = (
        chamber_votes.filter(pl.col("vote").is_in(["Yea", "Nay"]))
        .group_by("vote_id")
        .agg(
            pl.col("vote").filter(pl.col("vote") == "Yea").len().alias("yea"),
            pl.col("vote").filter(pl.col("vote") == "Nay").len().alias("nay"),
        )
        .with_columns((pl.col("yea") - pl.col("nay")).abs().alias("margin"))
    )
    margin_map = dict(zip(margins["vote_id"].to_list(), margins["margin"].to_list()))

    rows = []
    for row in bp.iter_rows(named=True):
        beta = row["beta_mean"]
        alpha = row["alpha_mean"]
        cutting_point = alpha / beta

        # Find nearest legislator
        dists = np.abs(ip_vals - cutting_point)
        nearest_idx = int(np.argmin(dists))
        nearest_slug = ip_slugs[nearest_idx]
        nearest_dist = float(dists[nearest_idx])

        vid = row["vote_id"]
        margin = margin_map.get(vid)

        rows.append(
            {
                "vote_id": vid,
                "bill_number": row.get("bill_number", vid),
                "beta_mean": beta,
                "alpha_mean": alpha,
                "cutting_point": cutting_point,
                "nearest_legislator": nearest_slug,
                "nearest_distance": nearest_dist,
                "margin": margin,
            }
        )

    return pl.DataFrame(rows).sort("beta_mean", descending=True)


def identify_swing_votes(
    cutting_points: pl.DataFrame,
    ideal_points: pl.DataFrame,
    votes: pl.DataFrame,
) -> pl.DataFrame:
    """Identify swing legislators — those near the cutting point on close votes."""
    if cutting_points.is_empty():
        return pl.DataFrame()

    close = cutting_points.filter(
        pl.col("margin").is_not_null() & (pl.col("margin") <= CLOSE_VOTE_MARGIN)
    )
    if close.is_empty():
        return pl.DataFrame()

    ip_map = dict(
        zip(
            ideal_points["legislator_slug"].to_list(),
            ideal_points["xi_mean"].to_list(),
        )
    )
    name_map = {}
    if "full_name" in ideal_points.columns:
        name_map = dict(
            zip(
                ideal_points["legislator_slug"].to_list(),
                ideal_points["full_name"].to_list(),
            )
        )
    party_map = {}
    if "party" in ideal_points.columns:
        party_map = dict(
            zip(
                ideal_points["legislator_slug"].to_list(),
                ideal_points["party"].to_list(),
            )
        )

    swing_counts: dict[str, int] = {}
    for row in close.iter_rows(named=True):
        cp = row["cutting_point"]
        for slug, xi in ip_map.items():
            if abs(xi - cp) <= SWING_VOTE_DISTANCE:
                swing_counts[slug] = swing_counts.get(slug, 0) + 1

    if not swing_counts:
        return pl.DataFrame()

    rows = [
        {
            "legislator_slug": slug,
            "full_name": name_map.get(slug, slug),
            "party": party_map.get(slug, ""),
            "swing_count": count,
            "ideal_point": ip_map.get(slug, 0.0),
        }
        for slug, count in swing_counts.items()
    ]
    return pl.DataFrame(rows).sort("swing_count", descending=True)


def plot_cutting_lines(
    bill_params: pl.DataFrame,
    ideal_points: pl.DataFrame,
    votes: pl.DataFrame,
    chamber: str,
    out_dir: Path,
    n_bills: int = 5,
) -> None:
    """VoteView-style cutting line visualization for top discriminating bills."""
    bp = (
        bill_params.filter(pl.col("beta_mean").abs() > MIN_BETA_FOR_CUTTING)
        .with_columns(pl.col("beta_mean").abs().alias("abs_beta"))
        .sort("abs_beta", descending=True)
        .head(n_bills)
    )

    if bp.is_empty():
        return

    ip = ideal_points.sort("xi_mean")
    ip_slugs = ip["legislator_slug"].to_list()
    ip_vals = ip["xi_mean"].to_numpy()
    ip_parties = ip["party"].to_list() if "party" in ip.columns else ["Unknown"] * ip.height

    # Build vote lookup
    vote_lookup: dict[str, dict[str, str]] = {}
    for row in votes.iter_rows(named=True):
        vid = row["vote_id"]
        if vid not in vote_lookup:
            vote_lookup[vid] = {}
        vote_lookup[vid][row["legislator_slug"]] = row["vote"]

    fig, axes = plt.subplots(n_bills, 1, figsize=(12, 3 * n_bills))
    if n_bills == 1:
        axes = [axes]

    for i, (ax, row) in enumerate(zip(axes, bp.iter_rows(named=True))):
        vid = row["vote_id"]
        beta = row["beta_mean"]
        alpha = row["alpha_mean"]
        cutting_point = alpha / beta
        bill_num = row.get("bill_number", vid)

        bill_votes = vote_lookup.get(vid, {})

        for j, (slug, xi, party) in enumerate(zip(ip_slugs, ip_vals, ip_parties)):
            v = bill_votes.get(slug)
            if v == "Yea":
                color = "#2ca02c"  # green
                marker = "^"
            elif v == "Nay":
                color = "#d62728"  # red
                marker = "v"
            else:
                continue

            ax.scatter(xi, 0, c=color, marker=marker, s=40, alpha=0.7, zorder=2)

        ax.axvline(cutting_point, color="black", linewidth=2, linestyle="--", zorder=3)
        direction = "Conservative-Yea" if beta > 0 else "Liberal-Yea"
        ax.set_title(f"{bill_num} (β={beta:.2f}, {direction})", fontsize=10)
        ax.set_yticks([])
        ax.set_xlabel("Ideal Point (Liberal ← → Conservative)")
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{chamber} — Cutting Lines for Top {n_bills} Discriminating Bills",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / f"cutting_lines_{chamber.lower()}.png")


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
        cons_idx, cons_slug, lib_idx, lib_slug, _ = select_anchors(
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
        raw_correlation = float(np.corrcoef(default_arr, sens_arr)[0, 1])

        # Sign convention may flip between independent IRT runs (different anchor
        # selection on different vote subsets). Use |r| for robustness check.
        if raw_correlation < 0:
            sens_arr = -sens_arr
            print("    Sign convention flipped — aligning sensitivity scores")
        correlation = abs(raw_correlation)

        print(f"    Shared legislators: {merged.height}")
        print(f"    Pearson r: {correlation:.4f}")
        print(f"    Sampling time: {sens_time:.1f}s")

        if correlation > 0.95:
            print("    Result: ROBUST (r > 0.95)")
        else:
            print("    Result: SENSITIVE (r <= 0.95) — investigate threshold dependence")

        findings[chamber] = {
            "default_threshold": CONTESTED_THRESHOLD,
            "sensitivity_threshold": SENSITIVITY_THRESHOLD,
            "default_n_legislators": default_ideal.height,
            "sensitivity_n_legislators": n_legs,
            "default_n_votes": default["data"]["n_votes"],
            "sensitivity_n_votes": n_votes,
            "shared_legislators": merged.height,
            "pearson_r": correlation,
            "raw_pearson_r": raw_correlation,
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

    ax.set_xlabel(f"Ideal Point (default: {CONTESTED_THRESHOLD * 100:.1f}% threshold)")
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

    with RunContext(
        session=args.session,
        analysis_name="05_irt",
        params=vars(args),
        primer=IRT_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"KS Legislature Bayesian IRT — Session {args.session}")
        print(f"Data:      {data_dir}")
        print(f"EDA:       {eda_dir}")
        print(f"PCA:       {pca_dir}")
        print(f"Output:    {ctx.run_dir}")
        print(f"Samples:   {args.n_samples} draws, {args.n_tune} tune, {args.n_chains} chains")

        # ── Phase 1: Load data ──
        print_header("PHASE 1: LOADING DATA")
        eda_house_path = eda_dir / "data" / "vote_matrix_house_filtered.parquet"
        if not eda_house_path.exists():
            print("Phase 04 (IRT): skipping — no EDA vote matrices available")
            return
        house_matrix, senate_matrix, full_matrix = load_eda_matrices(eda_dir)

        # Check if any chamber has legislators after filtering
        if house_matrix.height == 0 and senate_matrix.height == 0:
            print(
                "Phase 04 (IRT): skipping — 0 legislators after filtering (too few votes for IRT)"
            )
            return

        pca_house_path = pca_dir / "data" / "pc_scores_house.parquet"
        if pca_house_path.exists():
            pca_house, pca_senate = load_pca_scores(pca_dir)
        else:
            print("  PCA scores not available — skipping PCA-informed initialization")
            pca_house, pca_senate = None, None

        # Load PCA loadings for PC2-filtered vote selection (horseshoe remediation)
        pca_loadings_house, pca_loadings_senate = None, None
        if args.horseshoe_remediate:
            pca_loadings_house, pca_loadings_senate = load_pca_loadings(pca_dir)
            if pca_loadings_house is None and pca_loadings_senate is None:
                print("  WARNING: PCA loadings not found — horseshoe remediation unavailable")
        rollcalls, legislators = load_metadata(data_dir, use_csv=args.csv)
        from analysis.db import load_votes as db_load_votes

        raw_votes = db_load_votes(data_dir, use_csv=args.csv)
        print(f"  House filtered: {house_matrix.height} x {len(house_matrix.columns) - 1}")
        print(f"  Senate filtered: {senate_matrix.height} x {len(senate_matrix.columns) - 1}")
        print(f"  Full matrix: {full_matrix.height} x {len(full_matrix.columns) - 1}")
        if pca_house is not None:
            print(f"  PCA House scores: {pca_house.height}")
        if pca_senate is not None:
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
        pca_loadings_dict = {"House": pca_loadings_house, "Senate": pca_loadings_senate}
        strategy_results: dict[str, dict] = {}

        # Build robustness flags (ADR-0104)
        robustness_flags = RobustnessFlags.build_flags(args)
        robustness_results: dict[str, dict] = {}  # per-chamber robustness results
        active_flags = [f for f in robustness_flags if f.enabled]
        if active_flags:
            print(f"\n  Robustness flags: {', '.join(f.label for f in active_flags)}")
        else:
            print("\n  Robustness flags: none enabled")

        # Resolve 2D IRT directory (for --promote-2d, --init-strategy 2d-dim1, or --dim1-prior)
        need_2d = args.promote_2d or args.init_strategy == "2d-dim1" or args.dim1_prior
        irt_2d_dir: Path | None = None
        if need_2d:
            if args.irt_2d_dir:
                irt_2d_dir = Path(args.irt_2d_dir)
            else:
                try:
                    irt_2d_dir = resolve_upstream_dir(
                        "06_irt_2d",
                        results_root,
                        args.run_id,
                        None,
                    )
                except FileNotFoundError:
                    print("  WARNING: Phase 06 (2D IRT) results not found")
                    irt_2d_dir = None

        # Load 2D scores for init strategy or dim1-prior
        irt_2d_scores: dict[str, pl.DataFrame | None] = {}
        need_2d_scores = args.init_strategy == "2d-dim1" or args.dim1_prior
        if irt_2d_dir is not None and need_2d_scores:
            for ch in ("house", "senate"):
                irt_2d_scores[ch] = load_2d_scores(irt_2d_dir / "data", ch)
                if irt_2d_scores[ch] is not None:
                    print(f"  2D IRT Dim 1 loaded: {ch} ({irt_2d_scores[ch].height} rows)")
                else:
                    print(f"  2D IRT Dim 1 not found: {ch}")

        IS = IdentificationStrategy

        for chamber, matrix, pca_scores in chamber_configs:
            if matrix.height < 5:
                print(f"\n  Skipping {chamber}: too few legislators ({matrix.height})")
                continue

            # ── Phase 2: Prepare IRT data ──
            print_header(f"PHASE 2: PREPARE IRT DATA — {chamber}")
            data = prepare_irt_data(matrix, chamber)

            # ── Select identification strategy ──
            print_header(f"IDENTIFICATION STRATEGY — {chamber}")
            strategy, rationale = select_identification_strategy(
                args.identification,
                legislators,
                matrix,
                chamber,
            )
            print(f"  Strategy: {IS.DESCRIPTIONS[strategy]}")
            print(f"  Reference: {IS.REFERENCES.get(strategy, 'N/A')}")
            print()
            for s in IS.ALL_STRATEGIES:
                marker = "→" if s == strategy else " "
                print(f"  {marker} {IS.DESCRIPTIONS[s]}")
                print(f"      {rationale[s]}")
            print()

            strategy_results[chamber] = {
                "selected": strategy,
                "description": IS.DESCRIPTIONS[strategy],
                "reference": IS.REFERENCES.get(strategy, ""),
                "rationale": rationale,
            }

            # ── Build party indices (for constraint-based strategies) ──
            party_map = dict(
                zip(
                    legislators["legislator_slug"].to_list(),
                    legislators["party"].to_list(),
                )
            )
            party_indices: dict[str, list[int]] = {"Republican": [], "Democrat": []}
            for i, slug in enumerate(data["leg_slugs"]):
                party = party_map.get(slug, "Unknown")
                if party in party_indices:
                    party_indices[party].append(i)

            # ── Select anchors (for anchor-based strategies) ──
            chamber_anchors: list[tuple[int, float]] = []
            agree_rates: dict[str, float] | None = None
            cons_slug = "N/A"
            lib_slug = "N/A"
            anchor_method = "none"

            if strategy in (IS.ANCHOR_PCA, IS.ANCHOR_AGREEMENT):
                print("  Selecting anchors:")
                cons_idx, cons_slug, lib_idx, lib_slug, agree_rates = select_anchors(
                    pca_scores,
                    matrix,
                    chamber,
                    legislators=legislators if strategy == IS.ANCHOR_AGREEMENT else None,
                )
                chamber_anchors = [(cons_idx, 1.0), (lib_idx, -1.0)]
                anchor_method = (
                    "cross-party contested vote agreement"
                    if agree_rates is not None
                    else "PCA PC1 extremes (party-aware)"
                )

            # ── Phase 3: Build and sample ──
            print_header(f"PHASE 3: MCMC SAMPLING — {chamber}")

            # Build chain initialization based on strategy
            xi_init = None
            anchor_set = {idx for idx, _ in chamber_anchors}
            free_pos = [i for i in range(data["n_legislators"]) if i not in anchor_set]

            # --init-strategy override: replaces default init with shared module
            if args.init_strategy == "2d-dim1":
                ch_lower = chamber.lower()
                init_vals, _, init_src = resolve_init_source(
                    strategy="2d-dim1",
                    slugs=data["leg_slugs"],
                    irt_2d_scores=irt_2d_scores.get(ch_lower),
                    session=args.session,
                    chamber=ch_lower,
                )
                xi_init = init_vals[free_pos].astype(np.float64)
                print(
                    f"  2D Dim 1 init: {init_src}, "
                    f"{len(xi_init)} free params, "
                    f"range [{xi_init.min():.2f}, {xi_init.max():.2f}]"
                )

            elif not args.no_pca_init:
                if strategy == IS.ANCHOR_AGREEMENT and agree_rates is not None:
                    # Agreement-based init
                    init_vals = np.zeros(data["n_legislators"])
                    for i, slug in enumerate(data["leg_slugs"]):
                        party = party_map.get(slug, "Unknown")
                        rate = agree_rates.get(slug)
                        if rate is not None:
                            if party == "Republican":
                                init_vals[i] = 1.0 - 2.0 * rate
                            elif party == "Democrat":
                                init_vals[i] = -(1.0 - 2.0 * rate)
                        else:
                            init_vals[i] = 0.5 if party == "Republican" else -0.5
                    init_std = (init_vals - init_vals.mean()) / (init_vals.std() + 1e-8)
                    xi_init = init_std[free_pos].astype(np.float64)
                    print(
                        f"  Agreement-based init: {len(xi_init)} free params, "
                        f"range [{xi_init.min():.2f}, {xi_init.max():.2f}]"
                    )

                elif strategy in (
                    IS.HIERARCHICAL_PRIOR,
                    IS.SORT_CONSTRAINT,
                    IS.POSITIVE_BETA,
                    IS.UNCONSTRAINED,
                ):
                    # Party-based init for non-anchor strategies
                    init_vals = np.zeros(data["n_legislators"])
                    for i, slug in enumerate(data["leg_slugs"]):
                        party = party_map.get(slug, "Unknown")
                        init_vals[i] = 0.5 if party == "Republican" else -0.5
                    init_std = (init_vals - init_vals.mean()) / (init_vals.std() + 1e-8)
                    xi_init = init_std.astype(np.float64)  # all free, no anchors
                    print(
                        f"  Party-based init: {len(xi_init)} free params, "
                        f"range [{xi_init.min():.2f}, {xi_init.max():.2f}]"
                    )

                elif strategy == IS.EXTERNAL_PRIOR:
                    # External scores provide both prior AND init
                    # (init is handled by the prior mu, no separate init needed)
                    pass

                else:
                    # PCA-informed init (standard for anchor-pca)
                    slug_order = {s: i for i, s in enumerate(data["leg_slugs"])}
                    pc1_vals = (
                        pca_scores.filter(pl.col("legislator_slug").is_in(data["leg_slugs"]))
                        .sort(pl.col("legislator_slug").replace_strict(slug_order))["PC1"]
                        .to_numpy()
                    )
                    pc1_std = (pc1_vals - pc1_vals.mean()) / (pc1_vals.std() + 1e-8)
                    xi_init = pc1_std[free_pos].astype(np.float64)
                    print(
                        f"  PCA init: {len(xi_init)} free params, "
                        f"range [{xi_init.min():.2f}, {xi_init.max():.2f}]"
                    )

            # ── Dim 1 prior override (ADR-0108) ──
            # When --dim1-prior is active, switch to external-prior strategy
            # using 2D IRT Dim 1 as the informative prior source.
            dim1_external_priors: np.ndarray | None = None
            dim1_prior_sigma = args.dim1_prior_sigma
            if args.dim1_prior:
                ch_lower = chamber.lower()
                dim1_scores = irt_2d_scores.get(ch_lower)
                if dim1_scores is None:
                    print(
                        f"  WARNING: 2D IRT results not found for {chamber} "
                        "— dim1-prior unavailable, using standard strategy"
                    )
                else:
                    # Build per-legislator prior means from 2D Dim 1
                    dim1_map = {
                        row["legislator_slug"]: row["xi_dim1_mean"]
                        for row in dim1_scores.iter_rows(named=True)
                    }
                    dim1_raw = np.array([dim1_map.get(s, 0.0) for s in data["leg_slugs"]])
                    dim1_std_val = dim1_raw.std()
                    if dim1_std_val > 0:
                        dim1_std = (dim1_raw - dim1_raw.mean()) / dim1_std_val
                    else:
                        dim1_std = dim1_raw

                    matched_dim1 = sum(1 for s in data["leg_slugs"] if s in dim1_map)
                    print(
                        f"  Dim 1 prior: {matched_dim1}/{data['n_legislators']} matched, "
                        f"sigma={dim1_prior_sigma}"
                    )

                    # Override: external-prior strategy with dim1 values
                    strategy = IS.EXTERNAL_PRIOR
                    chamber_anchors = []  # no hard anchors — prior identifies
                    dim1_external_priors = dim1_std.astype(np.float64)

                    # Also use as init (belt and suspenders)
                    xi_init = dim1_std.astype(np.float64)
                    print(
                        f"  Dim 1 init: {len(xi_init)} params, "
                        f"range [{xi_init.min():.2f}, {xi_init.max():.2f}]"
                    )

            idata, sampling_time = build_and_sample(
                data,
                chamber_anchors,
                args.n_samples,
                args.n_tune,
                args.n_chains,
                xi_initvals=xi_init,
                strategy=strategy,
                party_indices=party_indices,
                external_priors=dim1_external_priors,
                external_prior_sigma=dim1_prior_sigma,
            )

            # ── Phase 4: Convergence diagnostics ──
            diagnostics = check_convergence(idata, chamber)

            # ── Post-hoc sign correction ──
            # For non-anchor strategies, check party means first
            if strategy not in (IS.ANCHOR_PCA, IS.ANCHOR_AGREEMENT):
                xi_mean = idata.posterior["xi"].mean(dim=["chain", "draw"]).values
                r_xi = [xi_mean[i] for i in party_indices.get("Republican", [])]
                d_xi = [xi_mean[i] for i in party_indices.get("Democrat", [])]
                if r_xi and d_xi and np.mean(r_xi) < np.mean(d_xi):
                    print(
                        f"\n  Party-mean sign correction: R mean ({np.mean(r_xi):+.3f}) < "
                        f"D mean ({np.mean(d_xi):+.3f}) — flipping"
                    )
                    idata.posterior["xi"] = -idata.posterior["xi"]
                    idata.posterior["xi_free"] = -idata.posterior["xi_free"]
                    if strategy != IS.POSITIVE_BETA:
                        idata.posterior["beta"] = -idata.posterior["beta"]

            # ── Sign validation (safety net for all strategies) ──
            print_header(f"SIGN VALIDATION — {chamber}")
            idata, sign_flipped = validate_sign(
                idata,
                matrix,
                legislators,
                data,
                chamber,
            )
            if sign_flipped:
                diagnostics["sign_flipped"] = True

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
            ctx.export_csv(
                ideal_points,
                f"ideal_points_{chamber.lower()}.csv",
                f"IRT ideal point estimates for {chamber} members",
            )
            bill_params.write_parquet(ctx.data_dir / f"bill_params_{chamber.lower()}.parquet")
            ctx.export_csv(
                bill_params,
                f"bill_params_{chamber.lower()}.csv",
                f"IRT bill parameters (difficulty + discrimination) for {chamber}",
            )
            print(f"  Saved: ideal_points_{chamber.lower()}.parquet")
            print(f"  Saved: bill_params_{chamber.lower()}.parquet")

            # Save InferenceData as NetCDF
            nc_path = ctx.data_dir / f"idata_{chamber.lower()}.nc"
            idata.to_netcdf(str(nc_path))
            print(f"  Saved: idata_{chamber.lower()}.nc")

            # ── Phase 6: Plots ──
            print_header(f"PHASE 6: PLOTS — {chamber}")
            plot_forest(ideal_points, chamber, ctx.plots_dir)

            # Paradox spotlight
            paradox = find_paradox_legislator(ideal_points, bill_params, data)
            if paradox:
                print(f"  Paradox detected: {paradox['full_name']} (gap={paradox['gap']:.0%})")
                plot_paradox_spotlight(
                    paradox,
                    ideal_points,
                    chamber,
                    ctx.plots_dir,
                )
            else:
                print("  No paradox legislator detected")

            plot_discrimination(bill_params, chamber, ctx.plots_dir)
            plot_irt_vs_pca(ideal_points, pca_scores, chamber, ctx.plots_dir)
            plot_traces(idata, data, chamber, ctx.plots_dir)
            plot_party_density(ideal_points, chamber, ctx.plots_dir)
            plot_icc_curves(bill_params, chamber, ctx.plots_dir)
            plot_ideal_points_interactive(ideal_points, chamber, ctx.plots_dir)
            plot_irt_vs_pca_interactive(ideal_points, pca_scores, chamber, ctx.plots_dir)

            # Cutting lines + swing votes
            cutting_pts = compute_cutting_points(
                bill_params, ideal_points, raw_votes, rollcalls, chamber
            )
            cutting_pts_result = cutting_pts
            swing_result = pl.DataFrame()
            if not cutting_pts.is_empty():
                cutting_pts.write_parquet(
                    ctx.data_dir / f"cutting_points_{chamber.lower()}.parquet"
                )
                plot_cutting_lines(bill_params, ideal_points, raw_votes, chamber, ctx.plots_dir)
                swing_result = identify_swing_votes(cutting_pts, ideal_points, raw_votes)
                if not swing_result.is_empty():
                    swing_result.write_parquet(
                        ctx.data_dir / f"swing_votes_{chamber.lower()}.parquet"
                    )
                    print(f"  {swing_result.height} swing legislators identified")
                    for row in swing_result.head(5).iter_rows(named=True):
                        print(f"    {row['full_name']}: {row['swing_count']} close votes")

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

            # ── Robustness analyses ──
            chamber_robustness: dict = {}

            if args.horseshoe_diagnostic:
                print_header(f"ROBUSTNESS: HORSESHOE DIAGNOSTIC — {chamber}")
                horseshoe = detect_horseshoe(ideal_points, pca_scores, chamber)
                chamber_robustness["horseshoe"] = horseshoe
                verdict = "DETECTED" if horseshoe["detected"] else "not detected"
                print(f"  Horseshoe: {verdict}")
                print(f"    Democrat wrong-side fraction: {horseshoe['dem_wrong_side_frac']:.1%}")
                print(f"    Party overlap fraction: {horseshoe['overlap_frac']:.1%}")
                print(f"    PCA eigenvalue ratio (PC1/PC2): {horseshoe['eigenvalue_ratio']:.2f}")
                if horseshoe["r_more_liberal_than_d_mean"]:
                    print(
                        f"    Most negative R: {horseshoe['most_neg_r_name']} "
                        f"(xi={horseshoe['most_neg_r_xi']:+.3f}) < D mean "
                        f"({horseshoe['d_mean']:+.3f})"
                    )

                # ── Horseshoe remediation: PC2-filtered refit ──
                if args.horseshoe_remediate and horseshoe["detected"]:
                    pca_loadings = pca_loadings_dict.get(chamber)
                    if pca_loadings is None:
                        print("  Remediation skipped: PCA loadings not available")
                    elif pca_scores is None:
                        print("  Remediation skipped: PCA scores not available")
                    else:
                        print_header(f"HORSESHOE REMEDIATION — {chamber}")
                        # Step 1: Filter to PC2-dominant votes
                        rem_matrix, n_kept, n_total_v = filter_pc2_dominant_votes(
                            matrix,
                            pca_loadings,
                        )
                        print(
                            f"  PC2-dominant votes: {n_kept} of {n_total_v} "
                            f"({100 * n_kept / n_total_v:.0f}%)"
                        )

                        if n_kept < MIN_PC2_VOTES_FOR_REFIT:
                            print(
                                f"  Remediation skipped: only {n_kept} PC2-dominant votes "
                                f"(need {MIN_PC2_VOTES_FOR_REFIT})"
                            )
                            chamber_robustness["horseshoe_remediation"] = {
                                "n_kept": n_kept,
                                "n_total": n_total_v,
                                "skipped": True,
                            }
                        else:
                            # Step 2: Prepare data on filtered votes
                            rem_data = prepare_irt_data(rem_matrix, chamber)

                            # Step 3: Get PC2 scores as informative prior
                            slug_order = {s: i for i, s in enumerate(rem_data["leg_slugs"])}
                            pc2_vals = (
                                pca_scores.filter(
                                    pl.col("legislator_slug").is_in(rem_data["leg_slugs"])
                                )
                                .sort(pl.col("legislator_slug").replace_strict(slug_order))["PC2"]
                                .to_numpy()
                            )
                            pc2_std = (pc2_vals - pc2_vals.mean()) / (pc2_vals.std() + 1e-8)

                            # Step 4: Sample with external-prior strategy (PC2)
                            print(
                                f"  Fitting PC2-remediated model: "
                                f"{rem_data['n_legislators']} legislators x "
                                f"{rem_data['n_votes']} votes"
                            )
                            print(f"  Strategy: external-prior (PC2, sigma={PC2_PRIOR_SIGMA})")
                            rem_idata, rem_time = build_and_sample(
                                rem_data,
                                [],  # no anchors — prior identifies
                                args.n_samples,
                                args.n_tune,
                                args.n_chains,
                                xi_initvals=pc2_std.astype(np.float64),
                                strategy=IS.EXTERNAL_PRIOR,
                                external_priors=pc2_std.astype(np.float64),
                                external_prior_sigma=PC2_PRIOR_SIGMA,
                            )
                            print(f"  PC2-remediated sampling complete in {rem_time:.1f}s")

                            # Step 5: Convergence check
                            rem_convergence = check_convergence(rem_idata, chamber)

                            # Step 6: Extract and compare
                            rem_ideal = extract_ideal_points(
                                rem_idata,
                                rem_data,
                                legislators,
                            )
                            merged = ideal_points.select(
                                "legislator_slug",
                                pl.col("xi_mean").alias("xi_primary"),
                            ).join(
                                rem_ideal.select(
                                    "legislator_slug",
                                    pl.col("xi_mean").alias("xi_remediated"),
                                ),
                                on="legislator_slug",
                                how="inner",
                            )
                            if merged.height >= 3:
                                pearson_r = float(
                                    merged.select(pl.corr("xi_primary", "xi_remediated")).item()
                                )
                                print(f"  Primary vs remediated: Pearson r = {pearson_r:.4f}")
                            else:
                                pearson_r = float("nan")

                            # Step 7: Check remediation quality
                            rem_horseshoe = detect_horseshoe(
                                rem_ideal,
                                pca_scores,
                                chamber,
                            )
                            rem_verdict = (
                                "RESOLVED" if not rem_horseshoe["detected"] else "PERSISTS"
                            )
                            print(f"  Horseshoe after remediation: {rem_verdict}")
                            print(f"    D wrong side: {rem_horseshoe['dem_wrong_side_frac']:.1%}")

                            chamber_robustness["horseshoe_remediation"] = {
                                "n_kept": n_kept,
                                "n_total": n_total_v,
                                "skipped": False,
                                "ideal_points": rem_ideal,
                                "convergence": rem_convergence,
                                "sampling_time": rem_time,
                                "correlation_with_primary": pearson_r,
                                "horseshoe_resolved": not rem_horseshoe["detected"],
                                "dem_wrong_side_frac": rem_horseshoe["dem_wrong_side_frac"],
                            }

            if args.contested_only:
                print_header(f"ROBUSTNESS: CONTESTED-ONLY REFIT — {chamber}")
                contested_matrix, n_contested, n_total = filter_contested_votes(
                    matrix,
                    legislators,
                )
                print(
                    f"  Contested votes: {n_contested} of {n_total} "
                    f"({100 * n_contested / n_total:.0f}%)"
                )
                if n_contested >= MIN_CONTESTED_FOR_REFIT:
                    # Prepare and sample on contested-only matrix
                    contested_data = prepare_irt_data(contested_matrix, chamber)
                    # Use same anchors/strategy if possible
                    contested_anchors: list[tuple[int, float]] = []
                    contested_slugs = contested_data["leg_slugs"]
                    if cons_slug in contested_slugs and lib_slug in contested_slugs:
                        ci = contested_slugs.index(cons_slug)
                        li = contested_slugs.index(lib_slug)
                        contested_anchors = [(ci, 1.0), (li, -1.0)]
                    # Build init
                    c_init = None
                    if not args.no_pca_init and contested_anchors:
                        c_anchor_set = {idx for idx, _ in contested_anchors}
                        c_free = [
                            i
                            for i in range(contested_data["n_legislators"])
                            if i not in c_anchor_set
                        ]
                        c_slug_order = {s: i for i, s in enumerate(contested_data["leg_slugs"])}
                        c_pc1 = (
                            pca_scores.filter(
                                pl.col("legislator_slug").is_in(contested_data["leg_slugs"])
                            )
                            .sort(pl.col("legislator_slug").replace_strict(c_slug_order))["PC1"]
                            .to_numpy()
                        )
                        c_pc1_std = (c_pc1 - c_pc1.mean()) / (c_pc1.std() + 1e-8)
                        c_init = c_pc1_std[c_free].astype(np.float64)

                    print(
                        f"  Fitting contested-only model: "
                        f"{contested_data['n_legislators']} legislators x "
                        f"{contested_data['n_votes']} votes"
                    )
                    c_idata, c_time = build_and_sample(
                        contested_data,
                        contested_anchors,
                        args.n_samples,
                        args.n_tune,
                        args.n_chains,
                        xi_initvals=c_init,
                        strategy=strategy,
                        party_indices=party_indices,
                    )
                    print(f"  Contested-only sampling complete in {c_time:.1f}s")

                    # Extract and compare
                    c_ideal = extract_ideal_points(c_idata, contested_data, legislators)
                    # Correlate with primary
                    merged = ideal_points.select(
                        "legislator_slug",
                        pl.col("xi_mean").alias("xi_primary"),
                    ).join(
                        c_ideal.select(
                            "legislator_slug",
                            pl.col("xi_mean").alias("xi_contested"),
                        ),
                        on="legislator_slug",
                        how="inner",
                    )
                    if merged.height >= 3:
                        pearson_r = float(
                            merged.select(pl.corr("xi_primary", "xi_contested")).item()
                        )
                        # Handle sign flip: use absolute correlation
                        if pearson_r < -0.5:
                            pearson_r = -pearson_r
                            print("    Sign convention flipped — using |r|")
                        print(f"  Primary vs contested-only: Pearson r = {pearson_r:.4f}")
                        # Top movers
                        movers = (
                            merged.with_columns(
                                (pl.col("xi_primary").rank(descending=True)).alias("rank_pri"),
                                (pl.col("xi_contested").rank(descending=True)).alias("rank_con"),
                            )
                            .with_columns(
                                (pl.col("rank_pri") - pl.col("rank_con")).abs().alias("rank_shift"),
                            )
                            .sort("rank_shift", descending=True)
                            .head(10)
                        )
                    else:
                        pearson_r = float("nan")
                        movers = pl.DataFrame()
                    chamber_robustness["contested_only"] = {
                        "n_contested": n_contested,
                        "n_total": n_total,
                        "contested_ideal_points": c_ideal,
                        "pearson_r": pearson_r,
                        "top_movers": movers.to_dicts() if not movers.is_empty() else [],
                        "sampling_time": c_time,
                    }
                else:
                    print(
                        f"  Skipping: only {n_contested} contested votes "
                        f"(need {MIN_CONTESTED_FOR_REFIT})"
                    )
                    chamber_robustness["contested_only"] = {
                        "n_contested": n_contested,
                        "n_total": n_total,
                        "skipped": True,
                    }

            if args.promote_2d and irt_2d_dir is not None:
                print_header(f"ROBUSTNESS: 2D CROSS-REFERENCE — {chamber}")
                xref = cross_reference_2d(ideal_points, irt_2d_dir, chamber)
                if xref is not None:
                    chamber_robustness["promote_2d"] = xref
                    print(
                        f"  Matched: {xref['n_matched']} legislators, "
                        f"correlation: {xref['correlation']:.4f}"
                    )
                    if xref["n_flagged"] > 0:
                        print(f"  Flagged (rank shift > {PROMOTE_2D_RANK_SHIFT}):")
                        for leg in xref["flagged_legislators"][:5]:
                            print(
                                f"    {leg['full_name']}: "
                                f"1D rank {int(leg['rank_1d'])} → "
                                f"2D rank {int(leg['rank_2d'])} "
                                f"(shift {int(leg['rank_shift'])})"
                            )

            if chamber_robustness:
                robustness_results[chamber] = chamber_robustness

            # Party separation quality gate (R2)
            # Detects wrong-axis estimation where IRT captures intra-party
            # factionalism rather than ideology. See docs/pca-ideology-axis-instability.md
            r_ip = ideal_points.filter(pl.col("party") == "Republican")
            d_ip = ideal_points.filter(pl.col("party") == "Democrat")
            if r_ip.height > 0 and d_ip.height > 0:
                r_std = float(r_ip["xi_mean"].std())
                d_std = float(d_ip["xi_mean"].std())
                pooled = np.sqrt((r_std**2 + d_std**2) / 2)
                party_sep_d = (
                    abs(float(r_ip["xi_mean"].mean()) - float(d_ip["xi_mean"].mean())) / pooled
                    if pooled > 0
                    else 0.0
                )
            else:
                party_sep_d = 0.0
            axis_uncertain = party_sep_d < 1.5
            diagnostics["party_separation_d"] = float(party_sep_d)
            diagnostics["axis_uncertain"] = axis_uncertain
            print(f"\n  Party separation (Cohen's d): {party_sep_d:.2f}")
            if axis_uncertain:
                print(
                    f"  WARNING: Low party separation (d = {party_sep_d:.2f} < 1.5). "
                    "The 1D IRT may be estimating intra-party factional variation "
                    "rather than ideology. See docs/pca-ideology-axis-instability.md"
                )

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
                "anchor_method": anchor_method,
                "strategy": strategy,
                "strategy_results": strategy_results.get(chamber, {}),
                "sign_flipped": diagnostics.get("sign_flipped", False),
                "paradox": paradox,
                "cutting_points": cutting_pts_result,
                "swing_votes": swing_result,
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
            ctx.export_csv(
                eq_ip,
                "ideal_points_joint_equated.csv",
                "Joint-equated ideal points (cross-chamber)",
            )
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

        if not results:
            print("Phase 04 (IRT): skipping — no chambers with enough data for IRT")
            return

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
                "minority_threshold_default": CONTESTED_THRESHOLD,
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
            manifest[f"{ch}_identification"] = {
                "strategy": result.get("strategy", "anchor-pca"),
                "anchor_method": result.get("anchor_method", "N/A"),
                "conservative_anchor": result["cons_slug"],
                "liberal_anchor": result["lib_slug"],
                "sign_flipped": result.get("sign_flipped", False),
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

        # Add robustness flags to manifest
        manifest["robustness_flags"] = {
            f.name: {"enabled": f.enabled, "label": f.label} for f in robustness_flags
        }

        save_filtering_manifest(manifest, ctx.run_dir)

        # Write convergence_summary.json for canonical routing consumption
        conv_summary: dict = {"chambers": {}}
        for chamber, result in results.items():
            if chamber == "Joint":
                continue
            conv_summary["chambers"][chamber] = {
                "convergence": result["diagnostics"],
            }
        conv_path = ctx.data_dir / "convergence_summary.json"
        with open(conv_path, "w") as f:
            json.dump(conv_summary, f, indent=2, default=str)
        print("  Saved: convergence_summary.json")

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
            robustness_flags=robustness_flags,
            robustness_results=robustness_results,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  NetCDF files:   {len(list(ctx.data_dir.glob('*.nc')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")
        print(f"  JSON manifests: {len(list(ctx.run_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
