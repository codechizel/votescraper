"""
Joint (Pooled) 1D IRT Experiment: Cross-Chamber Ideal Points via Shared Bills.

Pools House and Senate legislators into a single 1D IRT model. The hypothesis:
the 79th Kansas Senate (30R/10D, 40 members) fails to converge in per-chamber
IRT because the supermajority structure provides insufficient identification.
By pooling with the well-identified House (128 members), the 170 shared bills
act as anchors that place senators on the same ideology scale.

IRT handles the block-sparse structure natively: Senate members have NaN on
house-only votes (and vice versa), and these are simply excluded from the
likelihood. Each legislator contributes data from the votes they actually cast.

Usage:
    uv run python analysis/experimental/joint_irt_experiment.py [--session 2001-02]
        [--n-samples 2000] [--n-tune 2000] [--n-chains 4]
        [--identification anchor-pca] [--dim1-prior] [--dim1-prior-sigma 1.0]
        [--csv]
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

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

# ── Import shared functions ──────────────────────────────────────────────────

try:
    from analysis.irt import (
        IdentificationStrategy,
        build_and_sample,
        build_joint_vote_matrix,
        check_convergence,
        extract_ideal_points,
        load_eda_matrices,
        load_pca_scores,
        prepare_irt_data,
        select_anchors,
    )
    from analysis.phase_utils import load_metadata
    from analysis.report import (
        FigureSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
    )
except ModuleNotFoundError:
    from irt import (  # type: ignore[no-redef]
        IdentificationStrategy,
        build_and_sample,
        build_joint_vote_matrix,
        check_convergence,
        extract_ideal_points,
        load_eda_matrices,
        load_pca_scores,
        prepare_irt_data,
        select_anchors,
    )
    from phase_utils import load_metadata  # type: ignore[no-redef]
    from report import (  # type: ignore[no-redef]
        FigureSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
    )

try:
    from analysis.tuning import PARTY_COLORS
except ModuleNotFoundError:
    from tuning import PARTY_COLORS  # type: ignore[no-redef]

# ── Constants ────────────────────────────────────────────────────────────────

N_SAMPLES = 2000
N_TUNE = 2000
N_CHAINS = 4
RANDOM_SEED = 42

# Relaxed convergence thresholds for experiment
RHAT_THRESHOLD = 1.05
ESS_THRESHOLD = 200
MAX_DIVERGENCES = 50

DIM1_PRIOR_SIGMA_DEFAULT = 1.0

CHAMBER_MARKERS = {"House": "o", "Senate": "s"}
CT_TZ = ZoneInfo("America/Chicago")


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_joint_forest(ideal_points: pl.DataFrame, plots_dir: Path) -> Path:
    """Forest plot of all legislators, color-coded by party, shape by chamber."""
    df = ideal_points.sort("xi_mean")
    n = df.height

    fig, ax = plt.subplots(figsize=(10, max(12, n * 0.12)))
    for i, row in enumerate(df.iter_rows(named=True)):
        color = PARTY_COLORS.get(row["party"], "#999999")
        marker = CHAMBER_MARKERS.get(row["chamber"], "o")
        ax.errorbar(
            row["xi_mean"],
            i,
            xerr=[[row["xi_mean"] - row["xi_hdi_2.5"]], [row["xi_hdi_97.5"] - row["xi_mean"]]],
            fmt=marker,
            color=color,
            markersize=3,
            linewidth=0.5,
            capsize=0,
            alpha=0.8,
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{r['full_name']} ({r['chamber'][0]})" for r in df.iter_rows(named=True)],
        fontsize=5,
    )
    ax.set_xlabel("Ideal Point (xi)")
    ax.set_title("Joint IRT Ideal Points — All Legislators")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E81B23", label="Republican"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0015BC", label="Democrat"),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="grey", label="House", markersize=6
        ),
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor="grey", label="Senate", markersize=6
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    fig.tight_layout()
    path = plots_dir / "joint_forest.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def plot_senate_forest(ideal_points: pl.DataFrame, plots_dir: Path) -> Path:
    """Forest plot of Senate legislators only from the joint model."""
    senate = ideal_points.filter(pl.col("chamber") == "Senate").sort("xi_mean")
    n = senate.height

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.25)))
    for i, row in enumerate(senate.iter_rows(named=True)):
        color = PARTY_COLORS.get(row["party"], "#999999")
        ax.errorbar(
            row["xi_mean"],
            i,
            xerr=[[row["xi_mean"] - row["xi_hdi_2.5"]], [row["xi_hdi_97.5"] - row["xi_mean"]]],
            fmt="s",
            color=color,
            markersize=5,
            linewidth=1.0,
            capsize=2,
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels([r["full_name"] for r in senate.iter_rows(named=True)], fontsize=8)
    ax.set_xlabel("Ideal Point (xi)")
    ax.set_title("Senate Ideal Points from Joint Model")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    path = plots_dir / "senate_forest.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def plot_house_comparison(
    joint_ip: pl.DataFrame,
    perchamber_ip: pl.DataFrame | None,
    plots_dir: Path,
) -> tuple[Path | None, dict]:
    """Scatter: joint vs per-chamber House ideal points."""
    if perchamber_ip is None:
        return None, {}

    house_joint = joint_ip.filter(pl.col("chamber") == "House").select(
        "legislator_slug", pl.col("xi_mean").alias("xi_joint")
    )
    house_pc = perchamber_ip.select("legislator_slug", pl.col("xi_mean").alias("xi_perchamber"))
    merged = house_joint.join(house_pc, on="legislator_slug")

    x = merged["xi_perchamber"].to_numpy()
    y = merged["xi_joint"].to_numpy()
    r_pearson, _ = stats.pearsonr(x, y)
    r_spearman, _ = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=15, alpha=0.6, color="#333333")
    ax.set_xlabel("Per-Chamber House xi")
    ax.set_ylabel("Joint Model House xi")
    ax.set_title(f"House: Joint vs Per-Chamber (r = {r_pearson:.4f})")

    # 1:1 line
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    fig.tight_layout()
    path = plots_dir / "house_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path, {"pearson_r": r_pearson, "spearman_rho": r_spearman, "n": len(x)}


def plot_party_density(ideal_points: pl.DataFrame, plots_dir: Path) -> Path:
    """Overlapping kernel densities by party × chamber."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for chamber in ("House", "Senate"):
        for party in ("Republican", "Democrat"):
            subset = ideal_points.filter(
                (pl.col("chamber") == chamber) & (pl.col("party") == party)
            )
            if subset.height < 3:
                continue
            vals = subset["xi_mean"].to_numpy()
            color = PARTY_COLORS[party]
            linestyle = "-" if chamber == "House" else "--"
            label = f"{party} ({chamber})"

            from scipy.stats import gaussian_kde

            kde = gaussian_kde(vals)
            x_grid = np.linspace(vals.min() - 1, vals.max() + 1, 200)
            ax.plot(x_grid, kde(x_grid), color=color, linestyle=linestyle, label=label)

    ax.set_xlabel("Ideal Point (xi)")
    ax.set_ylabel("Density")
    ax.set_title("Joint Model: Party × Chamber Densities")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = plots_dir / "party_density.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def plot_sparsity(joint_matrix: pl.DataFrame, plots_dir: Path) -> Path:
    """Visualize the block-sparse structure of the joint vote matrix."""
    slug_col = "legislator_slug"
    vote_cols = [c for c in joint_matrix.columns if c != slug_col]
    data = joint_matrix.select(vote_cols).to_numpy()

    # Create mask: 1 = observed, 0 = missing
    observed = ~np.isnan(data.astype(float))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(observed, aspect="auto", cmap="Blues", interpolation="none")
    ax.set_xlabel(f"Votes ({len(vote_cols)})")
    ax.set_ylabel(f"Legislators ({joint_matrix.height})")
    ax.set_title("Joint Vote Matrix Sparsity (blue = observed, white = missing)")
    fig.tight_layout()
    path = plots_dir / "sparsity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


# ── Report builder ───────────────────────────────────────────────────────────


def build_report(
    output_dir: Path,
    session: str,
    matrix_info: dict,
    diag: dict,
    ideal_points: pl.DataFrame,
    house_comparison: dict,
    plot_paths: dict,
    sampling_time: float,
) -> Path:
    """Build self-contained HTML report."""
    rb = ReportBuilder(
        title=f"Joint (Pooled) IRT Experiment — {session}",
    )

    # Key findings
    converged = diag.get("all_ok", False)
    senate_ip = ideal_points.filter(pl.col("chamber") == "Senate")
    senate_range = float(senate_ip["xi_mean"].max() - senate_ip["xi_mean"].min())
    r_senate_mean = float(senate_ip.filter(pl.col("party") == "Republican")["xi_mean"].mean())
    d_senate_mean = float(senate_ip.filter(pl.col("party") == "Democrat")["xi_mean"].mean())
    findings = [
        f"Convergence: {'PASSED' if converged else 'FAILED'}",
        f"Matrix: {matrix_info['n_legislators']} legislators × {matrix_info['n_votes']} votes "
        f"({matrix_info['n_shared']} shared, "
        f"{matrix_info['n_house_only']} house-only, "
        f"{matrix_info['n_senate_only']} senate-only)",
        f"Senate ideal point range: {senate_range:.2f} "
        f"(R mean: {r_senate_mean:+.3f}, D mean: {d_senate_mean:+.3f})",
        f"Sampling time: {sampling_time:.0f}s ({sampling_time / 60:.1f} min)",
    ]
    if house_comparison:
        findings.append(f"House joint vs per-chamber: r = {house_comparison['pearson_r']:.4f}")
    rb.add(KeyFindingsSection(findings))

    # 1. Experiment design
    design_html = (
        "<p>The 79th Kansas Senate (30R/10D) fails 1D IRT convergence in per-chamber "
        "estimation &mdash; R-hat &gt; 1.8, ESS = 3, mode-splitting across all identification "
        "strategies tested (anchor-agreement, anchor-pca, sort-constraint) and all "
        "initialization strategies (PCA, 2D Dim 1).</p>"
        "<p><strong>Hypothesis:</strong> Pooling all 168 legislators (128 House + 40 Senate) "
        "into a single IRT model lets the well-identified House majority anchor the latent "
        "scale. The 170 shared bills bridge the chambers &mdash; IRT handles the "
        "block-sparse missing data natively (Senate members have NaN on house-only "
        "votes, excluded from the likelihood).</p>"
        "<p><strong>Identification:</strong> Anchors are selected from the House (where PCA "
        "cleanly separates parties). The Senate borrows identification strength through the "
        "shared bill linkage.</p>"
    )
    rb.add(TextSection(id="design", title="Experiment Design", html=design_html))

    # 2. Sparsity pattern
    if "sparsity" in plot_paths:
        rb.add(
            FigureSection.from_file(
                id="sparsity",
                title="Joint Vote Matrix Structure",
                path=plot_paths["sparsity"],
                caption=(
                    "Block-sparse structure of the joint vote matrix. House legislators "
                    "(top) observe house-only and shared votes. Senate legislators (bottom) "
                    "observe senate-only and shared votes. The shared bills (center block) "
                    "link the two chambers."
                ),
                alt_text="Heatmap showing observed vs missing votes in the joint matrix",
            )
        )

    # 3. Convergence
    def _status(val: float, threshold: float, higher_better: bool = False) -> str:
        if higher_better:
            return "OK" if val > threshold else "WARNING"
        return "OK" if val < threshold else "WARNING"

    conv_rows = [
        {
            "Metric": "R-hat (xi) max",
            "Value": f"{diag['xi_rhat_max']:.4f}",
            "Threshold": f"< {RHAT_THRESHOLD}",
            "Status": _status(diag["xi_rhat_max"], RHAT_THRESHOLD),
        },
        {
            "Metric": "R-hat (alpha) max",
            "Value": f"{diag['alpha_rhat_max']:.4f}",
            "Threshold": f"< {RHAT_THRESHOLD}",
            "Status": _status(diag["alpha_rhat_max"], RHAT_THRESHOLD),
        },
        {
            "Metric": "R-hat (beta) max",
            "Value": f"{diag['beta_rhat_max']:.4f}",
            "Threshold": f"< {RHAT_THRESHOLD}",
            "Status": _status(diag["beta_rhat_max"], RHAT_THRESHOLD),
        },
        {
            "Metric": "Bulk ESS (xi) min",
            "Value": f"{diag['xi_ess_min']:.0f}",
            "Threshold": f"> {ESS_THRESHOLD}",
            "Status": _status(diag["xi_ess_min"], ESS_THRESHOLD, higher_better=True),
        },
        {
            "Metric": "Bulk ESS (alpha) min",
            "Value": f"{diag['alpha_ess_min']:.0f}",
            "Threshold": f"> {ESS_THRESHOLD}",
            "Status": _status(diag["alpha_ess_min"], ESS_THRESHOLD, higher_better=True),
        },
        {
            "Metric": "Bulk ESS (beta) min",
            "Value": f"{diag['beta_ess_min']:.0f}",
            "Threshold": f"> {ESS_THRESHOLD}",
            "Status": _status(diag["beta_ess_min"], ESS_THRESHOLD, higher_better=True),
        },
        {
            "Metric": "Divergences",
            "Value": str(diag["divergences"]),
            "Threshold": f"< {MAX_DIVERGENCES}",
            "Status": _status(diag["divergences"], MAX_DIVERGENCES),
        },
    ]
    conv_df = pl.DataFrame(conv_rows)
    gt_conv = make_gt(conv_df, title="MCMC Convergence Diagnostics")
    rb.add(TableSection(id="convergence", title="Convergence Diagnostics", html=gt_conv))

    # 4. Joint forest plot
    if "joint_forest" in plot_paths:
        rb.add(
            FigureSection.from_file(
                id="joint-forest",
                title="Joint Ideal Points — All Legislators",
                path=plot_paths["joint_forest"],
                caption=(
                    "Forest plot of all 168 legislators from the joint model. "
                    "Circles = House, squares = Senate. Red = Republican, blue = Democrat."
                ),
                alt_text="Forest plot of joint IRT ideal points for all legislators",
            )
        )

    # 5. Senate forest plot
    if "senate_forest" in plot_paths:
        rb.add(
            FigureSection.from_file(
                id="senate-forest",
                title="Senate Ideal Points (from Joint Model)",
                path=plot_paths["senate_forest"],
                caption=(
                    "The key result: Senate ideal points extracted from the joint model. "
                    "Compare HDI widths to the per-chamber model — narrower HDIs indicate "
                    "the joint model is borrowing strength from shared bills."
                ),
                alt_text="Forest plot of Senate ideal points from joint model",
            )
        )

    # 6. Senate ideal points table
    senate_table = (
        ideal_points.filter(pl.col("chamber") == "Senate")
        .sort("xi_mean", descending=True)
        .select("full_name", "party", "xi_mean", "xi_sd", "xi_hdi_2.5", "xi_hdi_97.5")
        .with_columns(
            pl.col("xi_mean").round(3),
            pl.col("xi_sd").round(3),
            pl.col("xi_hdi_2.5").round(3),
            pl.col("xi_hdi_97.5").round(3),
        )
    )
    gt_senate = make_gt(senate_table, title="Senate Ideal Points (Joint Model)")
    rb.add(
        TableSection(
            id="senate-table",
            title="Senate Ideal Points — Full Table",
            html=gt_senate,
        )
    )

    # 7. House comparison
    if "house_comparison" in plot_paths and house_comparison:
        rb.add(
            FigureSection.from_file(
                id="house-comparison",
                title="House: Joint vs Per-Chamber Comparison",
                path=plot_paths["house_comparison"],
                caption=(
                    f"Joint vs per-chamber House ideal points. "
                    f"Pearson r = {house_comparison['pearson_r']:.4f}, "
                    f"Spearman ρ = {house_comparison['spearman_rho']:.4f}. "
                    f"High correlation confirms the joint model doesn't distort the "
                    f"well-identified House scale."
                ),
                alt_text="Scatter comparing joint vs per-chamber House ideal points",
            )
        )

    # 8. Party density
    if "party_density" in plot_paths:
        rb.add(
            FigureSection.from_file(
                id="party-density",
                title="Party × Chamber Density",
                path=plot_paths["party_density"],
                caption=(
                    "Kernel density estimates by party and chamber. Solid = House, "
                    "dashed = Senate. Shows whether Senate members are placed on "
                    "the same scale as their House counterparts."
                ),
                alt_text="Density plot of ideal points by party and chamber",
            )
        )

    # Build and save
    report_path = output_dir / "joint_irt_report.html"
    report_path.write_text(rb.render())
    print(f"  Saved: {report_path.name}")
    return report_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint (pooled) 1D IRT experiment — cross-chamber ideal points."
    )
    parser.add_argument("--session", default="2001-02", help="Session string (default: 2001-02)")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n-tune", type=int, default=N_TUNE)
    parser.add_argument("--n-chains", type=int, default=N_CHAINS)
    parser.add_argument(
        "--identification",
        default="anchor-pca",
        choices=[
            "anchor-pca",
            "anchor-agreement",
            "sort-constraint",
            "positive-beta",
            "hierarchical-prior",
            "unconstrained",
            "external-prior",
        ],
        help="Identification strategy (default: anchor-pca)",
    )
    parser.add_argument(
        "--dim1-prior",
        action="store_true",
        help="Use 2D IRT Dim 1 as informative prior on xi (ADR-0108)",
    )
    parser.add_argument(
        "--dim1-prior-sigma",
        type=float,
        default=DIM1_PRIOR_SIGMA_DEFAULT,
        help=f"Sigma for dim1 informative prior (default: {DIM1_PRIOR_SIGMA_DEFAULT})",
    )
    parser.add_argument("--csv", action="store_true", help="Force CSV loading (skip database)")
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from analysis.experiment_monitor import ExperimentLifecycle, PlatformCheck

    # Platform safety
    platform = PlatformCheck.current()
    warnings = platform.validate(n_chains=args.n_chains)
    for w in warnings:
        print(f"  PLATFORM: {w}")

    from tallgrass.session import KSSession

    ks = KSSession.from_session_string(args.session)
    data_dir = ks.data_dir
    results_root = ks.results_dir

    # Resolve upstream dirs
    from analysis.run_context import resolve_upstream_dir

    eda_dir = resolve_upstream_dir("01_eda", results_root)
    pca_dir = resolve_upstream_dir("02_pca", results_root)

    # Output directory
    today = datetime.now(CT_TZ).strftime("%Y-%m-%d")
    output_dir = Path("results/experimental_lab") / f"{today}_joint-irt"
    if output_dir.exists():
        suffix = 1
        while (Path("results/experimental_lab") / f"{today}_joint-irt.{suffix}").exists():
            suffix += 1
        output_dir = Path("results/experimental_lab") / f"{today}_joint-irt.{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir_out = output_dir / "data"
    data_dir_out.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  JOINT (POOLED) 1D IRT EXPERIMENT")
    print(f"  Session: {args.session}")
    print(f"  Samples: {args.n_samples}, Tune: {args.n_tune}, Chains: {args.n_chains}")
    print(f"  Identification: {args.identification}")
    if args.dim1_prior:
        print(f"  Dim1 Prior: ON (sigma={args.dim1_prior_sigma})")
    print(f"  Output: {output_dir}")
    print("=" * 80)

    t_total = time.time()

    # ── Load data ──
    print("\n--- Loading data ---")
    house_matrix, senate_matrix, _ = load_eda_matrices(eda_dir)
    pca_house, pca_senate = load_pca_scores(pca_dir)
    rollcalls, legislators = load_metadata(data_dir, use_csv=args.csv)

    print(f"  House matrix: {house_matrix.height} x {len(house_matrix.columns) - 1}")
    print(f"  Senate matrix: {senate_matrix.height} x {len(senate_matrix.columns) - 1}")

    # ── Build joint matrix ──
    print("\n--- Building joint vote matrix ---")
    joint_matrix, mapping_info = build_joint_vote_matrix(
        house_matrix, senate_matrix, rollcalls, legislators
    )

    slug_col = "legislator_slug"
    n_votes = len(joint_matrix.columns) - 1
    n_shared = len(mapping_info["matched_bills"])
    n_house_only = len(mapping_info["house_only_vote_ids"])
    n_senate_only = len(mapping_info["senate_only_vote_ids"])

    matrix_info = {
        "n_legislators": joint_matrix.height,
        "n_votes": n_votes,
        "n_shared": n_shared,
        "n_house_only": n_house_only,
        "n_senate_only": n_senate_only,
        "n_bridging": len(mapping_info["bridging_legislators"]),
    }

    # ── Sparsity plot ──
    print("\n--- Plotting sparsity ---")
    sparsity_path = plot_sparsity(joint_matrix, plots_dir)

    # ── Prepare IRT data ──
    print("\n--- Preparing IRT data (Joint) ---")
    data = prepare_irt_data(joint_matrix, "Joint")

    # ── Anchor selection ──
    # Use House PCA scores for anchors — the House has clean party separation
    print("\n--- Anchor selection (from House PCA) ---")
    # select_anchors needs a matrix whose slugs match the PCA scores.
    # We use the house_matrix for anchor selection, then map to joint indices.
    # Pass legislators=None to force PCA PC1 extremes (skip agreement-based,
    # which can pick wrong anchors when the joint matrix changes dynamics)
    cons_idx_h, cons_slug, lib_idx_h, lib_slug, agree_rates = select_anchors(
        pca_house, house_matrix, "House", None
    )

    # Map house slugs to joint matrix indices
    joint_slugs = joint_matrix[slug_col].to_list()
    cons_idx = joint_slugs.index(cons_slug)
    lib_idx = joint_slugs.index(lib_slug)

    print(f"  Conservative anchor: {cons_slug} (joint idx {cons_idx})")
    print(f"  Liberal anchor: {lib_slug} (joint idx {lib_idx})")

    anchors = [(cons_idx, 1.0), (lib_idx, -1.0)]

    # ── Initialization ──
    # Merge PCA scores from both chambers, standardize jointly
    print("\n--- Building PCA-informed initvals ---")
    pca_map = {}
    for row in pca_house.iter_rows(named=True):
        pca_map[row["legislator_slug"]] = row["PC1"]
    for row in pca_senate.iter_rows(named=True):
        pca_map[row["legislator_slug"]] = row["PC1"]

    init_raw = np.array([pca_map.get(s, 0.0) for s in joint_slugs])
    std = init_raw.std()
    if std > 0:
        init_std = (init_raw - init_raw.mean()) / std
    else:
        init_std = init_raw

    # For anchor-pca: init only the free (non-anchor) parameters
    anchor_set = {idx for idx, _ in anchors}
    free_pos = [i for i in range(len(joint_slugs)) if i not in anchor_set]
    xi_init = init_std[free_pos].astype(np.float64)

    matched_count = sum(1 for s in joint_slugs if s in pca_map)
    print(f"  PCA init: {matched_count}/{len(joint_slugs)} matched")
    print(f"  Free params: {len(xi_init)}, range [{xi_init.min():.2f}, {xi_init.max():.2f}]")

    # Map the identification strategy string
    strategy_map = {
        "anchor-pca": IdentificationStrategy.ANCHOR_PCA,
        "anchor-agreement": IdentificationStrategy.ANCHOR_AGREEMENT,
        "sort-constraint": IdentificationStrategy.SORT_CONSTRAINT,
        "positive-beta": IdentificationStrategy.POSITIVE_BETA,
        "hierarchical-prior": IdentificationStrategy.HIERARCHICAL_PRIOR,
        "unconstrained": IdentificationStrategy.UNCONSTRAINED,
        "external-prior": IdentificationStrategy.EXTERNAL_PRIOR,
    }
    strategy = strategy_map[args.identification]

    # Build party indices for constraint-based strategies
    party_indices = None
    needs_party = strategy in (
        IdentificationStrategy.SORT_CONSTRAINT,
        IdentificationStrategy.HIERARCHICAL_PRIOR,
    )
    if needs_party:
        party_map_leg = dict(
            zip(
                legislators["legislator_slug"].to_list(),
                legislators["party"].to_list(),
            )
        )
        party_indices = {
            "Republican": [
                i for i, s in enumerate(joint_slugs) if party_map_leg.get(s) == "Republican"
            ],
            "Democrat": [
                i for i, s in enumerate(joint_slugs) if party_map_leg.get(s) == "Democrat"
            ],
        }

    # ── Dim1 prior override (ADR-0108) ──
    dim1_external_priors: np.ndarray | None = None
    dim1_prior_sigma = args.dim1_prior_sigma
    if args.dim1_prior:
        print("\n--- Loading 2D IRT scores for dim1 prior ---")
        irt_2d_dir = resolve_upstream_dir("06_irt_2d", results_root)
        dim1_map: dict[str, float] = {}
        for ch_label in ("house", "senate"):
            parquet_path = irt_2d_dir / "data" / f"ideal_points_2d_{ch_label}.parquet"
            if parquet_path.exists():
                ch_df = pl.read_parquet(parquet_path)
                for row in ch_df.iter_rows(named=True):
                    dim1_map[row["legislator_slug"]] = row["xi_dim1_mean"]
                print(f"  Loaded {ch_df.height} {ch_label} legislators from 2D IRT")
            else:
                print(f"  WARNING: 2D IRT not found at {parquet_path}")

        if dim1_map:
            dim1_raw = np.array([dim1_map.get(s, 0.0) for s in joint_slugs])
            dim1_std_val = dim1_raw.std()
            if dim1_std_val > 0:
                dim1_std = (dim1_raw - dim1_raw.mean()) / dim1_std_val
            else:
                dim1_std = dim1_raw
            dim1_external_priors = dim1_std.astype(np.float64)
            # Override strategy and init values
            strategy = IdentificationStrategy.EXTERNAL_PRIOR
            anchors = []
            xi_init = dim1_external_priors.copy()
            matched = sum(1 for s in joint_slugs if s in dim1_map)
            print(
                f"  Dim1 prior: {matched}/{len(joint_slugs)} matched, "
                f"sigma={dim1_prior_sigma}, range [{dim1_external_priors.min():.2f}, "
                f"{dim1_external_priors.max():.2f}]"
            )
            print("  Strategy overridden to: external-prior")
        else:
            print("  WARNING: No 2D IRT scores found — dim1-prior unavailable")

    # ── Sample ──
    print("\n--- MCMC sampling (Joint) ---")
    with ExperimentLifecycle("joint-irt"):
        idata, sampling_time = build_and_sample(
            data=data,
            anchors=anchors,
            n_samples=args.n_samples,
            n_tune=args.n_tune,
            n_chains=args.n_chains,
            xi_initvals=xi_init,
            strategy=strategy,
            party_indices=party_indices,
            external_priors=dim1_external_priors,
            external_prior_sigma=dim1_prior_sigma,
        )

    # ── Convergence ──
    diag = check_convergence(idata, "Joint")

    # ── Sign validation ──
    # Hard anchors (Weber R at +1, Spangler D at -1) set the sign convention.
    # validate_sign's cross-party agreement heuristic can misfire on the joint
    # matrix (different contested vote dynamics), so use a simple party-mean check.
    print("\n--- Sign validation ---")
    xi_mean = idata.posterior["xi"].mean(dim=["chain", "draw"]).values
    party_map_check = dict(
        zip(legislators["legislator_slug"].to_list(), legislators["party"].to_list())
    )
    slugs_list = data["leg_slugs"]
    r_vals = [
        xi_mean[i] for i, s in enumerate(slugs_list) if party_map_check.get(s) == "Republican"
    ]
    d_vals = [xi_mean[i] for i, s in enumerate(slugs_list) if party_map_check.get(s) == "Democrat"]
    r_mean_check = np.mean(r_vals)
    d_mean_check = np.mean(d_vals)
    print(f"  R mean: {r_mean_check:+.3f}, D mean: {d_mean_check:+.3f}")

    was_flipped = False
    if r_mean_check < d_mean_check:
        print("  Sign convention inverted — flipping xi and beta posteriors")
        idata.posterior["xi"] = -idata.posterior["xi"]
        idata.posterior["xi_free"] = -idata.posterior["xi_free"]
        idata.posterior["beta"] = -idata.posterior["beta"]
        was_flipped = True
    else:
        print("  Sign is correct (R mean > D mean)")

    # ── Extract ideal points ──
    print("\n--- Extracting ideal points ---")
    ideal_points = extract_ideal_points(idata, data, legislators)

    # ── Print results ──
    print("\n" + "=" * 80)
    print("  SENATE IDEAL POINTS (from joint model)")
    print("=" * 80)
    senate_ip = ideal_points.filter(pl.col("chamber") == "Senate").sort("xi_mean", descending=True)
    header = f"\n  {'Name':<30s} {'Party':>10s}  {'xi':>7s}  {'HDI':>18s}"
    print(header)
    print("  " + "-" * 70)
    for row in senate_ip.iter_rows(named=True):
        print(
            f"  {row['full_name']:<30s} {row['party']:>10s}  "
            f"{row['xi_mean']:+.3f}  "
            f"[{row['xi_hdi_2.5']:+.3f}, {row['xi_hdi_97.5']:+.3f}]"
        )

    print("\n" + "=" * 80)
    print("  HOUSE IDEAL POINTS (top/bottom 5)")
    print("=" * 80)
    house_ip = ideal_points.filter(pl.col("chamber") == "House").sort("xi_mean", descending=True)
    print(header)
    print("  " + "-" * 70)
    for row in house_ip.head(5).iter_rows(named=True):
        print(
            f"  {row['full_name']:<30s} {row['party']:>10s}  "
            f"{row['xi_mean']:+.3f}  "
            f"[{row['xi_hdi_2.5']:+.3f}, {row['xi_hdi_97.5']:+.3f}]"
        )
    print("  ...")
    for row in house_ip.tail(5).iter_rows(named=True):
        print(
            f"  {row['full_name']:<30s} {row['party']:>10s}  "
            f"{row['xi_mean']:+.3f}  "
            f"[{row['xi_hdi_2.5']:+.3f}, {row['xi_hdi_97.5']:+.3f}]"
        )

    # ── Plots ──
    print("\n--- Generating plots ---")
    plot_paths: dict[str, Path] = {"sparsity": sparsity_path}
    plot_paths["joint_forest"] = plot_joint_forest(ideal_points, plots_dir)
    plot_paths["senate_forest"] = plot_senate_forest(ideal_points, plots_dir)
    plot_paths["party_density"] = plot_party_density(ideal_points, plots_dir)

    # Compare with per-chamber House results
    house_comparison: dict = {}
    irt_dir = resolve_upstream_dir("05_irt", results_root)
    house_pc_path = irt_dir / "data" / "ideal_points_house.parquet"
    if house_pc_path.exists():
        print(f"\n--- Comparing with per-chamber House ({house_pc_path}) ---")
        house_perchamber = pl.read_parquet(house_pc_path)
        comp_path, house_comparison = plot_house_comparison(
            ideal_points, house_perchamber, plots_dir
        )
        if comp_path:
            plot_paths["house_comparison"] = comp_path
            print(
                f"  House joint vs per-chamber: "
                f"r = {house_comparison['pearson_r']:.4f}, "
                f"ρ = {house_comparison['spearman_rho']:.4f}"
            )
    else:
        print(f"  Per-chamber House results not found at {house_pc_path}")

    # ── Save data ──
    print("\n--- Saving data ---")
    ideal_points.write_parquet(data_dir_out / "ideal_points_joint.parquet")
    print(f"  Saved: data/ideal_points_joint.parquet ({ideal_points.height} legislators)")

    senate_ip.write_parquet(data_dir_out / "ideal_points_joint_senate.parquet")
    print(f"  Saved: data/ideal_points_joint_senate.parquet ({senate_ip.height} senators)")

    house_ip.write_parquet(data_dir_out / "ideal_points_joint_house.parquet")
    print(f"  Saved: data/ideal_points_joint_house.parquet ({house_ip.height} reps)")

    summary = {
        "session": args.session,
        "identification": args.identification,
        "dim1_prior": args.dim1_prior,
        "dim1_prior_sigma": args.dim1_prior_sigma if args.dim1_prior else None,
        "n_samples": args.n_samples,
        "n_tune": args.n_tune,
        "n_chains": args.n_chains,
        "sampling_time_s": sampling_time,
        "matrix_info": matrix_info,
        "convergence": diag,
        "house_comparison": house_comparison,
        "sign_flipped": was_flipped,
        "timestamp": datetime.now(CT_TZ).isoformat(),
    }
    with open(data_dir_out / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  Saved: data/experiment_summary.json")

    # ── HTML Report ──
    print("\n--- Building HTML report ---")
    build_report(
        output_dir=output_dir,
        session=args.session,
        matrix_info=matrix_info,
        diag=diag,
        ideal_points=ideal_points,
        house_comparison=house_comparison,
        plot_paths=plot_paths,
        sampling_time=sampling_time,
    )

    # ── Summary ──
    total_time = time.time() - t_total
    print("\n" + "=" * 80)
    print("  EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"  Session: {args.session}")
    print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Sampling time: {sampling_time:.0f}s ({sampling_time / 60:.1f} min)")
    print(f"  Convergence: {'PASSED' if diag['all_ok'] else 'FAILED'}")
    print(f"  Matrix: {matrix_info['n_legislators']} legs × {matrix_info['n_votes']} votes")
    print(f"    Shared: {n_shared}, House-only: {n_house_only}, Senate-only: {n_senate_only}")

    r_mean = float(
        ideal_points.filter((pl.col("chamber") == "Senate") & (pl.col("party") == "Republican"))[
            "xi_mean"
        ].mean()
    )
    d_mean = float(
        ideal_points.filter((pl.col("chamber") == "Senate") & (pl.col("party") == "Democrat"))[
            "xi_mean"
        ].mean()
    )
    print(f"  Senate R mean: {r_mean:+.3f}, D mean: {d_mean:+.3f}")
    if house_comparison:
        print(f"  House joint vs per-chamber: r = {house_comparison['pearson_r']:.4f}")
    print(f"  Output: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
