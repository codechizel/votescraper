"""
Kansas Legislature — Exploratory Graph Analysis (Phase 02b)

Network psychometrics dimensionality assessment using Hudson Golino's EGA
framework. Estimates the number of latent dimensions in voting data via
GLASSO partial correlation networks and community detection — a principled
pre-IRT dimensionality oracle.

Usage:
  uv run python analysis/02b_ega/ega_phase.py [--session 2025-26]
      [--run-id ...] [--csv] [--eda-dir ...] [--pca-dir ...]
      [--method {glasso}] [--algorithm {walktrap,leiden}]
      [--n-boot 500] [--skip-boot] [--skip-uva]

Outputs (in results/<session>/<run_id>/02b_ega/):
  - data/:   ega_result_{chamber}.parquet, ega_summary_{chamber}.json,
             tefi_comparison_{chamber}.json, uva_redundant_{chamber}.parquet,
             tetrachoric_corr_{chamber}.parquet
  - plots/:  ega_network_{chamber}.png, tefi_curve_{chamber}.png,
             boot_k_histogram_{chamber}.png, item_stability_{chamber}.png,
             wto_heatmap_{chamber}.png
  - 02b_ega_report.html
"""

import argparse
import json
import sys
import warnings
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
    from analysis.phase_utils import print_header, save_fig
except ModuleNotFoundError:
    from phase_utils import print_header, save_fig

try:
    from analysis.ega_phase_report import build_ega_report
except ModuleNotFoundError:
    from ega_phase_report import build_ega_report  # type: ignore[no-redef]

try:
    from analysis.tuning import (
        EGA_BOOT_N,
        EGA_GLASSO_GAMMA,
        EGA_STABILITY_THRESHOLD,
        UVA_WTO_THRESHOLD,
    )
except ModuleNotFoundError:
    from tuning import (  # type: ignore[no-redef]
        EGA_BOOT_N,
        EGA_GLASSO_GAMMA,
        EGA_STABILITY_THRESHOLD,
        UVA_WTO_THRESHOLD,
    )

from analysis.ega.boot_ega import run_boot_ega
from analysis.ega.ega import run_ega
from analysis.ega.tefi import compute_tefi
from analysis.ega.uva import run_uva

# ── Primer ───────────────────────────────────────────────────────────────────

EGA_PRIMER = """\
# Exploratory Graph Analysis (EGA)

## Purpose

EGA estimates the number of latent dimensions in voting data using network
psychometrics rather than traditional factor-analytic methods. It provides
a principled pre-IRT dimensionality estimate: if EGA finds K=1, the chamber
is unidimensional; if K=2+, there is evidence for multidimensionality.

This complements PCA (Phase 02) by using conditional dependencies (partial
correlations) rather than marginal correlations, and provides bootstrap
stability assessment for dimensional structures.

## Method

1. **Tetrachoric correlations** — correct correlation type for binary data
2. **GLASSO network** — L1-penalized precision matrix (conditional dependencies)
3. **Community detection** — Walktrap/Leiden identifies dimension clusters
4. **TEFI** — Von Neumann entropy fit index compares K=1 through K=5
5. **bootEGA** — bootstrap stability of dimensionality estimate
6. **UVA** — weighted topological overlap for redundant bill detection

## Inputs

- Filtered vote matrix from EDA (Phase 01)
- PCA loadings from Phase 02 (for comparison)

## Outputs

- Dimensionality estimate (K) per chamber with bootstrap confidence
- Per-bill item stability and community assignments
- TEFI comparison across K=1..5
- Redundant bill pairs (UVA)
- GLASSO partial correlation network for visualization

## Interpretation Guide

- **K=1**: Voting is unidimensional (ideology only). Skip 2D IRT.
- **K=2**: Two dimensions present (likely ideology + establishment/contrarian).
- **Modal K from bootEGA**: If different from empirical K, structure is unstable.
- **Item stability < 0.70**: Bill is dimensionally ambiguous.
- **TEFI minimum**: Optimal K is the one with lowest TEFI.

## Caveats

- EGA is advisory — canonical routing (Phase 06) makes the final 1D/2D decision.
- Senate chambers (N~40) may produce sparse GLASSO networks.
- High-base-rate bills (>90% Yea) should be filtered before EGA.
"""

CHAMBERS = ["house", "senate"]


# ── Data Loading ─────────────────────────────────────────────────────────────


def _load_vote_matrix(eda_dir: Path, chamber: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Load filtered vote matrix from EDA output.

    Returns (matrix, legislator_ids, bill_ids) where matrix is n×p binary.
    """
    vm_path = eda_dir / "data" / f"vote_matrix_{chamber}_filtered.parquet"
    if not vm_path.exists():
        vm_path = eda_dir / "data" / f"vote_matrix_{chamber}.parquet"
    if not vm_path.exists():
        msg = f"Vote matrix not found at {vm_path}"
        raise FileNotFoundError(msg)

    df = pl.read_parquet(vm_path)
    # First column is legislator ID
    id_col = df.columns[0]
    legislator_ids = df[id_col].to_list()
    bill_cols = [c for c in df.columns if c != id_col]
    matrix = df.select(bill_cols).to_numpy().astype(np.float64)
    return matrix, legislator_ids, bill_cols


def _load_pca_loadings(pca_dir: Path, chamber: str) -> pl.DataFrame | None:
    """Load PCA loadings from Phase 02 for comparison."""
    loadings_path = pca_dir / "data" / f"loadings_{chamber}.parquet"
    if loadings_path.exists():
        return pl.read_parquet(loadings_path)
    return None


# ── Plotting ─────────────────────────────────────────────────────────────────


def _plot_ega_network(result, bill_ids: list[str], chamber: str, plots_dir: Path) -> Path:
    """Plot the GLASSO partial correlation network colored by community."""
    fig, ax = plt.subplots(figsize=(10, 10))
    g = result.network

    if g.vcount() == 0 or g.ecount() == 0:
        ax.text(0.5, 0.5, "No edges in GLASSO network", ha="center", va="center", fontsize=14)
        ax.set_title(f"EGA Network — {chamber.title()}")
    else:
        # Color by community
        n_comm = result.n_communities
        cmap = plt.cm.Set2
        colors = [cmap(result.community_assignments[i] % 8) for i in range(g.vcount())]

        layout = g.layout_fruchterman_reingold(weights="weight")
        coords = np.array(layout.coords)

        # Draw edges
        for edge in g.es:
            src, tgt = edge.tuple
            w = edge["weight"] if "weight" in edge.attributes() else 0.1
            ax.plot(
                [coords[src, 0], coords[tgt, 0]],
                [coords[src, 1], coords[tgt, 1]],
                color="gray",
                alpha=min(w * 2, 0.8),
                linewidth=max(w * 3, 0.3),
            )

        # Draw nodes
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=60, zorder=5, edgecolors="white")

        ax.set_title(
            f"EGA Network — {chamber.title()} (K={n_comm})",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    path = plots_dir / f"ega_network_{chamber}.png"
    save_fig(fig, path)
    return path


def _plot_tefi_curve(tefi_scores: dict[int, float], chamber: str, plots_dir: Path) -> Path:
    """Plot TEFI across K values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted(tefi_scores.keys())
    vals = [tefi_scores[k] for k in ks]

    ax.plot(ks, vals, "o-", color="#2C3E50", linewidth=2, markersize=8)
    best_k = min(tefi_scores, key=tefi_scores.get)
    ax.axvline(best_k, color="#E74C3C", linestyle="--", alpha=0.7, label=f"Best K={best_k}")

    ax.set_xlabel("Number of Dimensions (K)", fontsize=12)
    ax.set_ylabel("TEFI (lower = better)", fontsize=12)
    ax.set_title(f"TEFI Dimensionality Comparison — {chamber.title()}", fontsize=14)
    ax.legend()
    ax.set_xticks(ks)

    path = plots_dir / f"tefi_curve_{chamber}.png"
    save_fig(fig, path)
    return path


def _plot_boot_histogram(boot_result, chamber: str, plots_dir: Path) -> Path:
    """Plot bootstrap K frequency histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    freq = boot_result.dimension_frequency
    ks = sorted(freq.keys())
    counts = [freq[k] for k in ks]

    ax.bar(ks, counts, color="#3498DB", edgecolor="white")
    ax.axvline(
        boot_result.modal_k, color="#E74C3C", linestyle="--", label=f"Mode={boot_result.modal_k}"
    )

    ax.set_xlabel("Number of Dimensions (K)", fontsize=12)
    ax.set_ylabel("Bootstrap Replicates", fontsize=12)
    ax.set_title(f"bootEGA Dimension Frequency — {chamber.title()}", fontsize=14)
    ax.legend()
    ax.set_xticks(ks)

    path = plots_dir / f"boot_k_histogram_{chamber}.png"
    save_fig(fig, path)
    return path


def _plot_item_stability(
    stability: np.ndarray, bill_ids: list[str], chamber: str, plots_dir: Path
) -> Path:
    """Plot per-item stability from bootEGA."""
    fig, ax = plt.subplots(figsize=(12, max(6, len(bill_ids) * 0.15)))

    # Sort by stability
    order = np.argsort(stability)
    sorted_stability = stability[order]
    sorted_labels = [bill_ids[i][:30] for i in order]

    colors = ["#E74C3C" if s < EGA_STABILITY_THRESHOLD else "#2ECC71" for s in sorted_stability]
    ax.barh(range(len(sorted_stability)), sorted_stability, color=colors)
    ax.axvline(
        EGA_STABILITY_THRESHOLD,
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"Threshold={EGA_STABILITY_THRESHOLD}",
    )

    if len(sorted_labels) <= 50:
        ax.set_yticks(range(len(sorted_labels)))
        ax.set_yticklabels(sorted_labels, fontsize=6)
    else:
        ax.set_yticks([])

    ax.set_xlabel("Item Stability", fontsize=12)
    ax.set_title(f"bootEGA Item Stability — {chamber.title()}", fontsize=14)
    ax.legend()

    path = plots_dir / f"item_stability_{chamber}.png"
    save_fig(fig, path)
    return path


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="EGA Dimensionality Analysis (Phase 02b)")
    parser.add_argument("--session", default="2025-26", help="Session (e.g., 2025-26)")
    parser.add_argument("--run-id", default=None, help="Explicit run ID for pipeline")
    parser.add_argument("--csv", action="store_true", help="Force CSV-only mode")
    parser.add_argument("--eda-dir", type=Path, default=None, help="Override EDA directory")
    parser.add_argument("--pca-dir", type=Path, default=None, help="Override PCA directory")
    parser.add_argument(
        "--method", default="glasso", choices=["glasso"], help="Network estimation method"
    )
    parser.add_argument(
        "--algorithm",
        default="walktrap",
        choices=["walktrap", "leiden"],
        help="Community detection algorithm",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=EGA_BOOT_N,
        help=f"Bootstrap replicates (default {EGA_BOOT_N})",
    )
    parser.add_argument(
        "--skip-boot", action="store_true", help="Skip bootstrap stability assessment"
    )
    parser.add_argument("--skip-uva", action="store_true", help="Skip Unique Variable Analysis")
    args = parser.parse_args()

    with RunContext(
        session=args.session,
        analysis_name="02b_ega",
        params=vars(args),
        run_id=args.run_id,
        primer=EGA_PRIMER,
    ) as ctx:
        eda_dir = resolve_upstream_dir(
            "01_eda", ctx.session_root, run_id=args.run_id, override=args.eda_dir
        )
        pca_dir = resolve_upstream_dir(
            "02_pca", ctx.session_root, run_id=args.run_id, override=args.pca_dir
        )

        all_results: dict[str, dict] = {}

        for chamber in CHAMBERS:
            print_header(f"EGA — {chamber.title()}")

            # Load data
            try:
                matrix, leg_ids, bill_ids = _load_vote_matrix(eda_dir, chamber)
            except FileNotFoundError as e:
                print(f"  Skipping {chamber}: {e}")
                continue

            n_obs, n_items = matrix.shape
            print(f"  Vote matrix: {n_obs} legislators × {n_items} bills")

            # Run EGA
            print(f"  Running EGA (method={args.method}, algorithm={args.algorithm})...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ega_result = run_ega(
                    matrix,
                    method=args.method,
                    algorithm=args.algorithm,
                    gamma=EGA_GLASSO_GAMMA,
                )
            print(f"  EGA result: K={ega_result.n_communities}, edges={ega_result.glasso.n_edges}")
            if ega_result.community.fragmented:
                print("  ** Fragmentation guard: network too sparse for community detection")
                print("     Retried on largest connected component")
            if ega_result.unidimensional:
                print("  ** Unidimensional check: TRUE — data appears 1-dimensional")

            # TEFI comparison
            print("  Computing TEFI for K=1..5...")
            corr = ega_result.tetrachoric.corr_matrix
            # Generate K=1 assignment (all in one community)
            p = corr.shape[0]
            tefi_assignments = [np.zeros(p, dtype=np.int64)]
            # K=2..5: use naive equal splits for comparison
            for k in range(2, 6):
                tefi_assignments.append(np.array([i % k for i in range(p)], dtype=np.int64))
            # Replace with EGA's actual assignment for its K
            if ega_result.n_communities <= 5:
                tefi_assignments[ega_result.n_communities - 1] = ega_result.community_assignments

            tefi_scores: dict[int, float] = {}
            for k_idx, assigns in enumerate(tefi_assignments):
                k = k_idx + 1
                tefi_scores[k] = compute_tefi(corr, assigns)
            best_k_tefi = min(tefi_scores, key=tefi_scores.get)
            print(f"  TEFI best K={best_k_tefi}")

            # Bootstrap
            boot_result = None
            if not args.skip_boot:
                print(f"  Running bootEGA ({args.n_boot} replicates)...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    boot_result = run_boot_ega(
                        matrix,
                        n_boot=args.n_boot,
                        algorithm=args.algorithm,
                        gamma=EGA_GLASSO_GAMMA,
                    )
                n_unstable = int(np.sum(boot_result.item_stability < EGA_STABILITY_THRESHOLD))
                mk = boot_result.modal_k
                nb = boot_result.n_boot
                print(f"  bootEGA: modal K={mk}, {nb} replicates completed")
                print(f"  {n_unstable}/{p} items below stability threshold")

            # UVA
            uva_result = None
            if not args.skip_uva:
                print("  Running UVA...")
                uva_result = run_uva(ega_result.glasso.partial_corr, threshold=UVA_WTO_THRESHOLD)
                n_red = len(uva_result.redundant_pairs)
                n_rm = len(uva_result.suggested_removals)
                print(f"  UVA: {n_red} redundant pairs, {n_rm} removals")

            # PCA comparison
            pca_loadings = _load_pca_loadings(pca_dir, chamber)

            # Save results
            # EGA result parquet
            ega_df = pl.DataFrame(
                {
                    "bill_id": bill_ids,
                    "community": ega_result.community_assignments.tolist(),
                    "item_stability": boot_result.item_stability.tolist()
                    if boot_result
                    else [None] * len(bill_ids),
                }
            )
            for k in range(ega_result.n_communities):
                ega_df = ega_df.with_columns(
                    pl.Series(f"network_loading_{k}", ega_result.network_loadings[:, k].tolist())
                )
            ctx.export_csv(ega_df, f"ega_result_{chamber}.csv", f"EGA results for {chamber}")
            ega_df.write_parquet(ctx.data_dir / f"ega_result_{chamber}.parquet")

            # Summary JSON
            summary = {
                "chamber": chamber,
                "n_legislators": n_obs,
                "n_bills": n_items,
                "n_communities": ega_result.n_communities,
                "unidimensional": ega_result.unidimensional,
                "fragmented": ega_result.community.fragmented,
                "glasso_lambda": ega_result.glasso.selected_lambda,
                "glasso_n_edges": ega_result.glasso.n_edges,
                "algorithm": args.algorithm,
                "tetrachoric_n_fallback": ega_result.tetrachoric.n_fallback,
                "tetrachoric_n_pairs": ega_result.tetrachoric.n_pairs,
                "tefi_scores": {str(k): v for k, v in tefi_scores.items()},
                "tefi_best_k": best_k_tefi,
            }
            if boot_result:
                summary["boot_n"] = boot_result.n_boot
                summary["boot_modal_k"] = boot_result.modal_k
                summary["boot_median_k"] = boot_result.median_k
                summary["boot_dimension_frequency"] = {
                    str(k): v for k, v in boot_result.dimension_frequency.items()
                }
                summary["n_unstable_items"] = int(
                    np.sum(boot_result.item_stability < EGA_STABILITY_THRESHOLD)
                )
            if uva_result:
                summary["uva_n_redundant_pairs"] = len(uva_result.redundant_pairs)
                summary["uva_n_suggested_removals"] = len(uva_result.suggested_removals)

            with open(ctx.data_dir / f"ega_summary_{chamber}.json", "w") as f:
                json.dump(summary, f, indent=2)

            # TEFI JSON
            with open(ctx.data_dir / f"tefi_comparison_{chamber}.json", "w") as f:
                json.dump({str(k): v for k, v in tefi_scores.items()}, f, indent=2)

            # Tetrachoric correlation matrix
            corr_df = pl.DataFrame({bill_ids[i]: corr[:, i].tolist() for i in range(len(bill_ids))})
            corr_df.write_parquet(ctx.data_dir / f"tetrachoric_corr_{chamber}.parquet")

            # UVA parquet
            if uva_result and uva_result.redundant_pairs:
                uva_df = pl.DataFrame(
                    {
                        "bill_i": [bill_ids[p[0]] for p in uva_result.redundant_pairs],
                        "bill_j": [bill_ids[p[1]] for p in uva_result.redundant_pairs],
                        "wto": [p[2] for p in uva_result.redundant_pairs],
                    }
                )
                uva_df.write_parquet(ctx.data_dir / f"uva_redundant_{chamber}.parquet")
                ctx.export_csv(
                    uva_df,
                    f"uva_redundant_{chamber}.csv",
                    f"UVA redundant bill pairs for {chamber}",
                )

            # Plots
            _plot_ega_network(ega_result, bill_ids, chamber, ctx.plots_dir)
            _plot_tefi_curve(tefi_scores, chamber, ctx.plots_dir)
            if boot_result:
                _plot_boot_histogram(boot_result, chamber, ctx.plots_dir)
                _plot_item_stability(boot_result.item_stability, bill_ids, chamber, ctx.plots_dir)

            all_results[chamber] = {
                "ega": ega_result,
                "boot": boot_result,
                "uva": uva_result,
                "tefi_scores": tefi_scores,
                "summary": summary,
                "pca_loadings": pca_loadings,
            }

        # Build report
        if all_results:
            report_path = build_ega_report(all_results, ctx.plots_dir, ctx.data_dir, ctx.run_dir)
            print(f"\n  Report: {report_path}")

            # Create session-root symlink for the report
            if ctx.run_id is not None:
                report_link = ctx.session_root / f"{ctx.analysis_name}_report.html"
                if report_link.is_symlink() or report_link.exists():
                    report_link.unlink()
                report_link.symlink_to(
                    Path(ctx.run_id) / ctx.analysis_name / report_path.name
                )


if __name__ == "__main__":
    main()
