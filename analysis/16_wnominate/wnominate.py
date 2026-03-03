"""
Kansas Legislature — W-NOMINATE + Optimal Classification Validation

Standalone validation phase comparing our Bayesian IRT ideal points to the
field-standard W-NOMINATE (Poole & Rosenthal) and nonparametric Optimal
Classification (Poole 2000). Runs R via subprocess — matching Phase 16's
emIRT pattern.

Does NOT feed into synthesis or profiles — purely a validation exercise.

Usage:
  uv run python analysis/17_wnominate/wnominate.py --session 2025-26
  uv run python analysis/17_wnominate/wnominate.py --session 2025-26 --skip-oc

Outputs (in results/<session>/<run_id>/17_wnominate/):
  - data/:   Parquet files (WNOM coords, OC coords, correlations, comparison table)
  - plots/:  PNG plots (scatter, 2D plot, scree)
  - filtering_manifest.json, run_info.json, run_log.txt
  - wnominate_report.html
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir

try:
    from analysis.phase_utils import load_legislators, print_header, save_fig
except ModuleNotFoundError:
    from phase_utils import load_legislators, print_header, save_fig  # type: ignore[no-redef]

try:
    from analysis.wnominate_report import build_wnominate_report
except ModuleNotFoundError:
    from wnominate_report import build_wnominate_report  # type: ignore[no-redef]

try:
    from analysis.wnominate_data import (
        LOP_THRESHOLD,
        MIN_LEGISLATORS,
        MIN_VOTES,
        PARTY_COLORS,
        WNOMINATE_DIMS,
        build_comparison_table,
        compute_three_way_correlations,
        compute_within_party_correlations,
        convert_vote_matrix_to_rollcall_csv,
        parse_eigenvalues,
        parse_fit_statistics,
        parse_oc_results,
        parse_wnominate_results,
        select_polarity_legislator,
        sign_align_scores,
    )
except ModuleNotFoundError:
    from wnominate_data import (  # type: ignore[no-redef]
        LOP_THRESHOLD,
        MIN_LEGISLATORS,
        MIN_VOTES,
        PARTY_COLORS,
        WNOMINATE_DIMS,
        build_comparison_table,
        compute_three_way_correlations,
        compute_within_party_correlations,
        convert_vote_matrix_to_rollcall_csv,
        parse_eigenvalues,
        parse_fit_statistics,
        parse_oc_results,
        parse_wnominate_results,
        select_polarity_legislator,
        sign_align_scores,
    )

# ── Primer ───────────────────────────────────────────────────────────────────

WNOMINATE_PRIMER = """\
# W-NOMINATE + Optimal Classification Validation

## Purpose

This phase compares our Bayesian IRT ideal points against two field-standard
legislative scaling methods: W-NOMINATE (Poole & Rosenthal) and Optimal
Classification (Poole 2000). The goal is external validation — confirming that
our IRT scores align with the methods used in virtually every published paper
on congressional and state legislative voting.

## Method

1. **Convert** our EDA vote matrix to pscl rollcall format (1=Yea, 6=Nay, 9=Missing).
2. **Select polarity** legislator: highest PCA PC1 score with ≥50% participation.
3. **Run W-NOMINATE** (R `wnominate` package) with 2 dimensions, 3 trials.
4. **Run Optimal Classification** (R `oc` package) with 2 dimensions.
5. **Sign-align** WNOM and OC scores against IRT (flip if Pearson r < 0).
6. **Compute** 3×3 correlation matrix (IRT/WNOM/OC): Pearson r + Spearman ρ.
7. **Generate** comparison table, scatter plots, scree plot, fit statistics.

## Inputs

- EDA vote matrices: `results/<session>/<run_id>/01_eda/data/vote_matrix_{chamber}_filtered.parquet`
- PCA scores: `results/<session>/<run_id>/02_pca/data/pc_scores_{chamber}.parquet`
- IRT ideal points: `results/<session>/<run_id>/04_irt/data/ideal_points_{chamber}.parquet`

## Outputs

### `data/` — Parquet intermediates

| File | Description |
|------|-------------|
| `wnominate_coords_{chamber}.parquet` | W-NOMINATE dim1/dim2 + SEs |
| `oc_coords_{chamber}.parquet` | OC dim1/dim2 + correct classification |
| `correlations_{chamber}.json` | 3×3 correlation matrix |
| `comparison_{chamber}.parquet` | Full legislator table (IRT/WNOM/OC + ranks) |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `scatter_irt_vs_wnom_{chamber}.png` | IRT vs W-NOMINATE (party-colored) |
| `scatter_irt_vs_oc_{chamber}.png` | IRT vs OC (party-colored) |
| `wnom_2d_{chamber}.png` | W-NOMINATE dim1 vs dim2 with unit circle |
| `scree_{chamber}.png` | W-NOMINATE eigenvalue scree plot |

## Interpretation Guide

- **IRT vs W-NOMINATE r > 0.95**: Standard result in the literature. Our IRT
  and the field-standard method essentially agree.
- **IRT vs W-NOMINATE r = 0.90-0.95**: Good agreement. Minor differences
  reflect model specification (Bayesian vs MLE, 1D vs 2D).
- **IRT vs OC r > 0.90**: Expected. OC is nonparametric, so slightly lower
  correlation with parametric methods is normal.
- **W-NOMINATE vs OC r > 0.95**: Standard result — both capture the same
  dominant dimension.

## Caveats

- W-NOMINATE requires R + `wnominate`, `oc`, `pscl`, `jsonlite` packages.
- Polarity is set by PCA PC1, not by external knowledge. This matches the
  data-driven approach used throughout our pipeline.
- W-NOMINATE uses MLE (not Bayesian), so no posterior uncertainty. The SEs
  are parametric bootstrap estimates.
- OC failure is non-fatal — the report will show W-NOMINATE only.
"""


# ── R Package Check ──────────────────────────────────────────────────────────


def check_r_packages() -> bool:
    """Verify Rscript and required R packages are available."""
    if shutil.which("Rscript") is None:
        print("ERROR: Rscript not found on PATH.")
        print("Install R from https://cran.r-project.org/")
        return False

    pkgs = '"wnominate","oc","pscl","jsonlite"'
    check_script = f"cat(all(sapply(c({pkgs}), requireNamespace, quietly=TRUE)))"
    try:
        result = subprocess.run(
            ["Rscript", "-e", check_script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if "TRUE" not in result.stdout:
            print("ERROR: Missing R packages.")
            print("Install with:")
            print('  install.packages(c("wnominate", "oc", "pscl", "jsonlite"))')
            return False
    except subprocess.TimeoutExpired:
        print("ERROR: Could not verify R packages.")
        return False
    except FileNotFoundError:
        print("ERROR: Rscript not found.")
        return False

    return True


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KS Legislature W-NOMINATE + OC Validation (Phase 17)"
    )
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override flat IRT results directory")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument("--dims", type=int, default=WNOMINATE_DIMS, help="Number of dimensions")
    parser.add_argument("--skip-oc", action="store_true", help="Skip Optimal Classification")
    return parser.parse_args()


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_eda_matrices(
    eda_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load filtered vote matrices from the EDA phase output."""
    house = pl.read_parquet(eda_dir / "data" / "vote_matrix_house_filtered.parquet")
    senate = pl.read_parquet(eda_dir / "data" / "vote_matrix_senate_filtered.parquet")
    return house, senate


def load_pca_scores(pca_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load PCA scores for polarity selection."""
    house = pl.read_parquet(pca_dir / "data" / "pc_scores_house.parquet")
    senate = pl.read_parquet(pca_dir / "data" / "pc_scores_senate.parquet")
    return house, senate


def load_irt_ideal_points(
    irt_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load flat IRT ideal points."""
    house_path = irt_dir / "data" / "ideal_points_house.parquet"
    senate_path = irt_dir / "data" / "ideal_points_senate.parquet"

    house = pl.read_parquet(house_path) if house_path.exists() else None
    senate = pl.read_parquet(senate_path) if senate_path.exists() else None

    return house, senate


# ── R Subprocess ─────────────────────────────────────────────────────────────


def run_r_wnominate(
    vote_matrix: pl.DataFrame,
    polarity_idx: int,
    chamber: str,
    output_dir: Path,
    dims: int = 2,
) -> bool:
    """Run wnominate.R via subprocess. Returns True on success."""
    r_script = Path(__file__).parent / "wnominate.R"
    if not r_script.exists():
        print(f"  ERROR: R script not found at {r_script}")
        return False

    # Write input CSV
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="wnom_input_"
    ) as f:
        input_csv = Path(f.name)
        # Write CSV with legislator_slug as row names
        coded, slugs = convert_vote_matrix_to_rollcall_csv(vote_matrix)
        # Set legislator_slug as first column (R will use it as rownames)
        coded.write_csv(f.name)

    try:
        print(f"  Input CSV: {input_csv} ({vote_matrix.height} x {vote_matrix.width - 1})")
        result = subprocess.run(
            [
                "Rscript",
                str(r_script),
                str(input_csv),
                str(output_dir),
                chamber,
                str(polarity_idx),
                str(dims),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"  [R] {line}")

        if result.returncode != 0:
            print(f"  W-NOMINATE/OC failed (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[:20]:
                    print(f"  [R stderr] {line}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print("  ERROR: R subprocess timed out (600s)")
        return False
    except FileNotFoundError:
        print("  ERROR: Rscript not found")
        return False
    finally:
        input_csv.unlink(missing_ok=True)


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_scatter(
    comparison: pl.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    corr_result: dict,
    out_path: Path,
) -> None:
    """Party-colored scatter plot with regression line."""
    if comparison.height == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    for party in ["Republican", "Democrat", "Independent"]:
        if "party" not in comparison.columns:
            continue
        sub = comparison.filter(pl.col("party") == party)
        if sub.height == 0:
            continue

        color = PARTY_COLORS.get(party, "#888888")
        ax.scatter(
            sub[x_col].to_numpy(),
            sub[y_col].to_numpy(),
            c=color,
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label=f"{party} (n={sub.height})",
        )

    # Regression line
    valid = comparison.filter(pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null())
    if valid.height >= 2:
        x = valid[x_col].to_numpy().astype(float)
        y = valid[y_col].to_numpy().astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "k--", alpha=0.3, linewidth=1)

    r = corr_result.get("pearson_r", float("nan"))
    rho = corr_result.get("spearman_rho", float("nan"))
    n = corr_result.get("n", 0)
    quality = corr_result.get("quality", "")

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(
        f"{title}\nPearson r = {r:.3f}, Spearman rho = {rho:.3f} (n = {n}, {quality})",
        fontsize=13,
        fontweight="bold",
    )

    ax.text(
        0.02,
        0.98,
        "Red = Republican, Blue = Democrat\n"
        "Dashed line = linear fit\n"
        "Strong agreement: points cluster tightly along line",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.9},
    )

    ax.legend(fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_wnom_2d(
    wnom_df: pl.DataFrame,
    irt_df: pl.DataFrame,
    chamber: str,
    session: str,
    out_path: Path,
) -> None:
    """W-NOMINATE 2D plot (dim1 vs dim2) with unit circle."""
    merged = wnom_df.join(
        irt_df.select("legislator_slug", "party"),
        on="legislator_slug",
        how="inner",
    )

    valid = merged.filter(pl.col("wnom_dim1").is_not_null() & pl.col("wnom_dim2").is_not_null())
    if valid.height == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", alpha=0.2, linewidth=1)

    for party in ["Republican", "Democrat", "Independent"]:
        sub = valid.filter(pl.col("party") == party) if "party" in valid.columns else valid
        if sub.height == 0:
            continue

        color = PARTY_COLORS.get(party, "#888888")
        ax.scatter(
            sub["wnom_dim1"].to_numpy(),
            sub["wnom_dim2"].to_numpy(),
            c=color,
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label=f"{party} (n={sub.height})",
        )

    ax.set_xlabel("W-NOMINATE Dimension 1 (ideology)", fontsize=12)
    ax.set_ylabel("W-NOMINATE Dimension 2", fontsize=12)
    ax.set_title(
        f"{chamber} — W-NOMINATE 2D Space ({session})\n"
        "Dim 1 = left-right ideology, Dim 2 captures residual structure",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_scree(
    eigen_df: pl.DataFrame,
    chamber: str,
    session: str,
    out_path: Path,
) -> None:
    """Scree plot of W-NOMINATE eigenvalues."""
    if eigen_df.height == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    dims = eigen_df["dimension"].to_numpy()
    vals = eigen_df["eigenvalue"].to_numpy()

    ax.bar(dims, vals, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.plot(dims, vals, "ko-", markersize=6)

    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("Eigenvalue", fontsize=12)
    ax.set_title(
        f"{chamber} — W-NOMINATE Eigenvalues ({session})\n"
        "First dimension should dominate; sharp drop after dim 1 expected",
        fontsize=13,
        fontweight="bold",
    )

    if "pct_variance" in eigen_df.columns:
        for i, row in enumerate(eigen_df.iter_rows(named=True)):
            ax.text(
                row["dimension"],
                row["eigenvalue"],
                f"{row['pct_variance']:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_path)


# ── Per-Chamber Processing ───────────────────────────────────────────────────


def process_chamber(
    chamber: str,
    vote_matrix: pl.DataFrame,
    pca_scores: pl.DataFrame,
    irt_df: pl.DataFrame,
    legislators: pl.DataFrame,
    ctx: "RunContext",
    dims: int,
    skip_oc: bool,
    session: str,
) -> dict | None:
    """Run W-NOMINATE + OC for one chamber. Returns results dict or None."""
    ch = chamber.lower()
    n_legislators = vote_matrix.height

    print(f"\n  {chamber}: {n_legislators} legislators, {vote_matrix.width - 1} votes")

    if n_legislators < MIN_LEGISLATORS:
        print(f"  {chamber}: Only {n_legislators} legislators (<{MIN_LEGISLATORS}) — skipping")
        return None

    # Select polarity legislator
    polarity_idx = select_polarity_legislator(pca_scores, vote_matrix)
    slug_col = vote_matrix.columns[0]
    polarity_slug = vote_matrix[slug_col][polarity_idx - 1]
    print(f"  Polarity legislator: {polarity_slug} (index {polarity_idx})")

    # Get legislator slugs in matrix order
    _, slugs = convert_vote_matrix_to_rollcall_csv(vote_matrix)

    # Run R subprocess
    r_output_dir = ctx.data_dir
    success = run_r_wnominate(vote_matrix, polarity_idx, ch, r_output_dir, dims)
    if not success:
        print(f"  {chamber}: W-NOMINATE failed — skipping")
        return None

    # Parse W-NOMINATE results
    wnom_path = r_output_dir / f"wnominate_coords_{ch}.csv"
    if not wnom_path.exists():
        print(f"  {chamber}: W-NOMINATE output not found — skipping")
        return None

    wnom_raw = pl.read_csv(wnom_path, null_values="NA")
    # Drop row-name column if present (R writes it as first unnamed column)
    if wnom_raw.columns[0] in ("", "rownames", "X"):
        wnom_raw = wnom_raw.drop(wnom_raw.columns[0])
    wnom_df = parse_wnominate_results(wnom_raw, slugs)

    # Enrich with metadata
    wnom_df = wnom_df.join(
        legislators.select("legislator_slug", "full_name", "party"),
        on="legislator_slug",
        how="left",
    )

    # Also enrich IRT
    if "full_name" not in irt_df.columns:
        irt_df = irt_df.join(
            legislators.select("legislator_slug", "full_name", "party"),
            on="legislator_slug",
            how="left",
        )

    # Sign-align WNOM against IRT
    wnom_df = sign_align_scores(wnom_df, "wnom_dim1", irt_df, "xi_mean")
    print(f"  {chamber}: W-NOMINATE sign-aligned against IRT")

    # Parse OC results
    oc_df = None
    oc_path = r_output_dir / f"oc_coords_{ch}.csv"
    if oc_path.exists() and not skip_oc:
        oc_raw = pl.read_csv(oc_path, null_values="NA")
        if oc_raw.columns[0] in ("", "rownames", "X"):
            oc_raw = oc_raw.drop(oc_raw.columns[0])
        oc_df = parse_oc_results(oc_raw, slugs)
        oc_df = sign_align_scores(oc_df, "oc_dim1", irt_df, "xi_mean")
        print(f"  {chamber}: OC sign-aligned against IRT")

    # Parse fit statistics
    fit_path = r_output_dir / f"fit_statistics_{ch}.json"
    fit_stats = {}
    if fit_path.exists():
        with open(fit_path) as f:
            fit_json = json.load(f)
        fit_stats = parse_fit_statistics(fit_json)

    # Parse eigenvalues
    eigen_path = r_output_dir / f"eigenvalues_{ch}.csv"
    eigen_df = None
    if eigen_path.exists():
        eigen_df = parse_eigenvalues(pl.read_csv(eigen_path, null_values="NA"))

    # Correlations
    corr = compute_three_way_correlations(irt_df, wnom_df, oc_df)
    within_party = compute_within_party_correlations(irt_df, wnom_df, oc_df)

    # Print headline correlations
    iw = corr["irt_wnom"]
    print(
        f"  {chamber} IRT vs WNOM: r={iw['pearson_r']:.3f}, "
        f"rho={iw['spearman_rho']:.3f} ({iw['quality']})"
    )
    if corr["irt_oc"]["n"] > 0:
        io = corr["irt_oc"]
        print(
            f"  {chamber} IRT vs OC:   r={io['pearson_r']:.3f}, "
            f"rho={io['spearman_rho']:.3f} ({io['quality']})"
        )

    # Comparison table
    comparison = build_comparison_table(irt_df, wnom_df, oc_df)

    # Save parquets
    wnom_df.write_parquet(ctx.data_dir / f"wnominate_coords_{ch}.parquet")
    if oc_df is not None:
        oc_df.write_parquet(ctx.data_dir / f"oc_coords_{ch}.parquet")
    comparison.write_parquet(ctx.data_dir / f"comparison_{ch}.parquet")

    # Save correlations JSON
    with open(ctx.data_dir / f"correlations_{ch}.json", "w") as f:
        json.dump(
            {"overall": corr, "within_party": within_party},
            f,
            indent=2,
            default=str,
        )

    # Plots
    # IRT vs W-NOMINATE scatter
    plot_scatter(
        comparison,
        "irt_score",
        "wnom_score",
        "IRT Ideal Point (xi_mean)",
        "W-NOMINATE Dimension 1",
        f"{chamber} — IRT vs W-NOMINATE ({session})",
        corr["irt_wnom"],
        ctx.plots_dir / f"scatter_irt_vs_wnom_{ch}.png",
    )

    # IRT vs OC scatter
    if oc_df is not None and corr["irt_oc"]["n"] > 0 and "oc_score" in comparison.columns:
        plot_scatter(
            comparison,
            "irt_score",
            "oc_score",
            "IRT Ideal Point (xi_mean)",
            "Optimal Classification Dimension 1",
            f"{chamber} — IRT vs Optimal Classification ({session})",
            corr["irt_oc"],
            ctx.plots_dir / f"scatter_irt_vs_oc_{ch}.png",
        )

    # W-NOMINATE 2D plot
    if dims >= 2:
        plot_wnom_2d(
            wnom_df,
            irt_df,
            chamber,
            session,
            ctx.plots_dir / f"wnom_2d_{ch}.png",
        )

    # Scree plot
    if eigen_df is not None:
        plot_scree(eigen_df, chamber, session, ctx.plots_dir / f"scree_{ch}.png")

    return {
        "chamber": chamber,
        "n_legislators": n_legislators,
        "polarity_slug": polarity_slug,
        "polarity_idx": polarity_idx,
        "wnom_df": wnom_df,
        "oc_df": oc_df,
        "correlations": corr,
        "within_party": within_party,
        "fit_stats": fit_stats,
        "eigen_df": eigen_df,
        "comparison": comparison,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # Check R prerequisites — skip gracefully in pipeline mode
    if not check_r_packages():
        print("[Phase 16] Skipping: R not available (install R + packages for W-NOMINATE)")
        return

    from tallgrass.session import KSSession

    ks = KSSession.from_session_string(args.session)

    with RunContext(
        session=args.session,
        analysis_name="16_wnominate",
        params=vars(args),
        primer=WNOMINATE_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"W-NOMINATE + OC Validation — Session {args.session}")
        print(f"Dimensions: {args.dims}")
        print(f"Skip OC:    {args.skip_oc}")
        print(f"Output:     {ctx.run_dir}")

        results_root = ks.results_dir

        # ── Load upstream data ──
        print_header("LOADING UPSTREAM DATA")

        eda_dir = resolve_upstream_dir(
            "01_eda",
            results_root,
            args.run_id,
            override=Path(args.eda_dir) if args.eda_dir else None,
        )
        pca_dir = resolve_upstream_dir(
            "02_pca",
            results_root,
            args.run_id,
            override=Path(args.pca_dir) if args.pca_dir else None,
        )
        irt_dir = resolve_upstream_dir(
            "05_irt",
            results_root,
            args.run_id,
            override=Path(args.irt_dir) if args.irt_dir else None,
        )

        print(f"  EDA dir: {eda_dir}")
        print(f"  PCA dir: {pca_dir}")
        print(f"  IRT dir: {irt_dir}")

        house_matrix, senate_matrix = load_eda_matrices(eda_dir)
        n_h, n_hv = house_matrix.height, house_matrix.width - 1
        n_s, n_sv = senate_matrix.height, senate_matrix.width - 1
        print(f"  House matrix: {n_h} legislators x {n_hv} votes")
        print(f"  Senate matrix: {n_s} legislators x {n_sv} votes")

        pca_house, pca_senate = load_pca_scores(pca_dir)
        irt_house, irt_senate = load_irt_ideal_points(irt_dir)

        legislators = load_legislators(ks.data_dir)

        # ── Process chambers ──
        all_results: dict[str, dict] = {}

        for chamber, vote_matrix, pca_scores, irt_df in [
            ("House", house_matrix, pca_house, irt_house),
            ("Senate", senate_matrix, pca_senate, irt_senate),
        ]:
            if irt_df is None:
                print(f"\n  {chamber}: IRT ideal points not found — skipping")
                continue

            print_header(f"{chamber.upper()} CHAMBER")
            result = process_chamber(
                chamber=chamber,
                vote_matrix=vote_matrix,
                pca_scores=pca_scores,
                irt_df=irt_df,
                legislators=legislators,
                ctx=ctx,
                dims=args.dims,
                skip_oc=args.skip_oc,
                session=args.session,
            )
            if result is not None:
                all_results[chamber.lower()] = result

        if not all_results:
            print("\n  No chambers processed successfully.")
            return

        # ── Filtering manifest ──
        print_header("FILTERING MANIFEST")
        manifest: dict = {
            "analysis": "16_wnominate",
            "session": args.session,
            "dims": args.dims,
            "skip_oc": args.skip_oc,
            "constants": {
                "MIN_LEGISLATORS": MIN_LEGISLATORS,
                "MIN_VOTES": MIN_VOTES,
                "LOP_THRESHOLD": LOP_THRESHOLD,
                "WNOMINATE_TRIALS": 3,
            },
        }
        for ch, data in all_results.items():
            manifest[ch] = {
                "n_legislators": data["n_legislators"],
                "polarity_slug": data["polarity_slug"],
                "polarity_idx": data["polarity_idx"],
                "irt_wnom_r": data["correlations"]["irt_wnom"]["pearson_r"],
                "irt_oc_r": data["correlations"]["irt_oc"].get("pearson_r", float("nan")),
                "fit_stats": data["fit_stats"],
            }

        manifest_path = ctx.run_dir / "filtering_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  Saved: {manifest_path.name}")

        # ── HTML report ──
        print_header("HTML REPORT")
        build_wnominate_report(
            ctx.report,
            all_results=all_results,
            session=args.session,
            dims=args.dims,
            plots_dir=ctx.plots_dir,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
