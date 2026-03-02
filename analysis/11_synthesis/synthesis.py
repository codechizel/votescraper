"""
Kansas Legislature — Synthesis Report

Combines results from all 10 upstream analysis phases (EDA, PCA, IRT, Clustering,
Network, Prediction, Indices, UMAP, Beta-Binomial, Hierarchical) into a single
narrative-driven HTML report for nontechnical audiences: journalists, policymakers,
citizens.

Reads from upstream parquets and manifests; does not recompute anything from raw CSVs.

Usage:
  uv run python analysis/synthesis.py [--session 2025-26]

Outputs (in results/<session>/synthesis/<date>/):
  - data/:   Unified legislator DataFrames (house, senate) as parquet
  - plots/:  4-8 new PNGs (dashboards, profiles, paradox, pipeline)
  - filtering_manifest.json, run_info.json, run_log.txt
  - synthesis_report.html
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext
except ModuleNotFoundError:
    from run_context import RunContext

try:
    from analysis.synthesis_detect import ParadoxCase, detect_all
except ModuleNotFoundError:
    from synthesis_detect import ParadoxCase, detect_all  # type: ignore[no-redef]

try:
    from analysis.synthesis_data import UPSTREAM_PHASES, build_legislator_df, load_all_upstream
except ModuleNotFoundError:
    from synthesis_data import (  # type: ignore[no-redef]
        UPSTREAM_PHASES,
        build_legislator_df,
        load_all_upstream,
    )

try:
    from analysis.synthesis_report import build_scrolly_synthesis_report, build_synthesis_report
except ModuleNotFoundError:
    from synthesis_report import (  # type: ignore[no-redef]
        build_scrolly_synthesis_report,
        build_synthesis_report,
    )

# ── Primer ───────────────────────────────────────────────────────────────────

SYNTHESIS_PRIMER = """\
# Synthesis Report

## Purpose

Narrative summary of Kansas Legislature voting patterns, combining findings
from ten upstream analysis phases into a single deliverable for nontechnical audiences.

## Method

No new computation. Joins upstream parquet outputs on `legislator_slug`, adds
percentile ranks, and produces narrative-driven visualizations that tell the
story of Kansas politics. Notable legislators (mavericks, bridge-builders,
metric paradoxes) are detected from data, not hardcoded.

## Inputs

Parquet files from: IRT, Indices, Network, Clustering, Prediction, PCA, EDA, UMAP,
Beta-Binomial, Hierarchical. Filtering manifests from each phase for headline statistics.

## Outputs

- `synthesis_report.html` — Self-contained narrative report (29-32 sections)
- `plots/` — New PNGs: 2 dashboard scatters, 2-3 profile cards (data-driven),
  0-1 paradox visualizations, 1 pipeline summary
- `data/` — Unified legislator DataFrames per chamber (parquet)

## Interpretation Guide

The report is organized by story, not by method. It starts with the headline
finding (party dominates everything), then profiles notable legislators, and
ends with a full scorecard. Every plot is annotated for a reader with no
statistical training.

## Caveats

- All statistics inherit the upstream filtering: near-unanimous votes removed
  (minority < 2.5%), legislators with < 20 votes excluded.
- Profile narratives are editorial interpretations of quantitative patterns.
- "Maverick" is a descriptive label based on party-vote defections, not a
  normative judgment.
- Notable legislators are detected dynamically; different sessions will
  highlight different legislators.
"""

# ── Constants ────────────────────────────────────────────────────────────────

PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}
PARTY_COLORS_LIGHT = {"Republican": "#F5A0A5", "Democrat": "#8090E0", "Independent": "#CCCCCC"}


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_dashboard_scatter(
    leg_df: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
    *,
    annotate_slugs: list[str] | None = None,
) -> Path:
    """Legislature dashboard scatter: IRT x Unity, colored by party, sized by maverick rate."""
    fig, ax = plt.subplots(figsize=(14, 9))

    chamber_title = chamber.title()

    # Extract numpy arrays from polars
    has_unity = "unity_score" in leg_df.columns
    has_maverick = "weighted_maverick" in leg_df.columns

    for party, color in PARTY_COLORS.items():
        sub = leg_df.filter(pl.col("party") == party)
        x = sub["xi_mean"].to_numpy()
        y = sub["unity_score"].to_numpy() if has_unity else np.full(sub.height, 0.5)

        if has_maverick:
            s = 30 + sub["weighted_maverick"].fill_null(0).to_numpy() * 800
        else:
            s = np.full(sub.height, 50)

        ax.scatter(
            x,
            y,
            c=color,
            s=s,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label=party,
            zorder=3,
        )

    # Annotate key legislators (caller passes slugs from detect_annotation_slugs)
    slugs = list(annotate_slugs or [])

    for slug in slugs:
        row = leg_df.filter(pl.col("legislator_slug") == slug)
        if row.height == 0:
            continue
        r = row.to_dicts()[0]
        x = r["xi_mean"]
        y = (r.get("unity_score") or 0.5) if has_unity else 0.5
        if x is None:
            continue
        name = r["full_name"]
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            fontweight="bold",
            color="#333333",
            arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.8},
            zorder=5,
        )

    ax.set_xlabel("Ideology (Liberal ←→ Conservative)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Party Unity (higher = more loyal)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Every {chamber_title} Member at a Glance",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )

    # Callout box
    ax.text(
        0.02,
        0.02,
        "Big circles = frequent rebels on close votes",
        transform=ax.transAxes,
        fontsize=9,
        fontstyle="italic",
        color="#555555",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f0f0f0", "edgecolor": "#cccccc"},
    )

    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    out = plots_dir / f"dashboard_scatter_{chamber}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


def plot_profile_card(
    leg_df: pl.DataFrame,
    slug: str,
    meta: dict,
    plots_dir: Path,
) -> Path | None:
    """Horizontal bar chart showing 6 normalized metrics for one legislator."""
    row = leg_df.filter(pl.col("legislator_slug") == slug)
    if row.height == 0:
        print(f"  WARNING: {slug} not found in legislator_df")
        return None

    r = row.to_dicts()[0]
    party = r["party"]
    color = PARTY_COLORS.get(party, "#666666")
    light = PARTY_COLORS_LIGHT.get(party, "#cccccc")

    # Build 6 metric bars (all normalized 0-1)
    metrics = []

    # 1. IRT Ideological Rank (percentile)
    val = r.get("xi_mean_percentile")
    if val is not None:
        metrics.append(("Ideological Rank", val, f"{val:.0%}"))

    # 2. CQ Party Unity (raw, already 0-1)
    val = r.get("unity_score")
    if val is not None:
        metrics.append(("Party Unity (CQ)", val, f"{val:.0%}"))

    # 3. Clustering Loyalty (raw, already 0-1)
    val = r.get("loyalty_rate")
    if val is not None:
        metrics.append(("Clustering Loyalty", val, f"{val:.0%}"))

    # 4. Maverick Rate (raw, 0-1)
    val = r.get("maverick_rate")
    if val is not None:
        metrics.append(("Maverick Rate", val, f"{val:.0%}"))

    # 5. Network Betweenness (percentile)
    val = r.get("betweenness_percentile")
    if val is not None:
        metrics.append(("Network Influence", val, f"{val:.0%}"))

    # 6. Prediction Accuracy (raw)
    val = r.get("accuracy")
    if val is not None:
        metrics.append(("Prediction Accuracy", val, f"{val:.0%}"))

    if not metrics:
        print(f"  WARNING: no metrics available for {slug}")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    texts = [m[2] for m in metrics]

    y_pos = np.arange(len(metrics))
    bars = ax.barh(y_pos, values, color=color, alpha=0.8, height=0.6, edgecolor="white")

    # Background bars (full 0-1 range)
    ax.barh(y_pos, [1.0] * len(metrics), color=light, alpha=0.2, height=0.6, zorder=0)

    # Value labels
    for i, (bar, txt) in enumerate(zip(bars, texts)):
        x_pos = bar.get_width() + 0.02
        if bar.get_width() > 0.85:
            x_pos = bar.get_width() - 0.02
            ax.text(
                x_pos,
                i,
                txt,
                va="center",
                ha="right",
                fontsize=11,
                fontweight="bold",
                color="white",
            )
        else:
            ax.text(
                x_pos,
                i,
                txt,
                va="center",
                ha="left",
                fontsize=11,
                fontweight="bold",
                color="#333333",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score (0 to 1)", fontsize=10)
    ax.invert_yaxis()

    title = f"{meta['title']} — {meta['role']}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Subtitle
    fig.text(
        0.5,
        0.01,
        meta["subtitle"],
        ha="center",
        fontsize=9,
        fontstyle="italic",
        color="#555555",
        wrap=True,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    slug_short = slug.split("_", 1)[1]  # e.g., "schreiber" or "van_dyk"
    out = plots_dir / f"profile_{slug_short}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


def plot_metric_paradox(
    chamber_df: pl.DataFrame,
    paradox: ParadoxCase,
    plots_dir: Path,
) -> Path | None:
    """Three horizontal bars showing a legislator's contradictory metrics."""
    slug = paradox.slug
    row = chamber_df.filter(pl.col("legislator_slug") == slug)
    if row.height == 0:
        print(f"  WARNING: {slug} not found in {paradox.chamber}_df")
        return None

    rv = paradox.raw_values

    # Three contrasting metrics
    bars = [
        {
            "label": (
                f"Most {paradox.direction.title()} "
                f"(IRT Rank #{paradox.rank_high} of {paradox.n_in_party} "
                f"{paradox.party[0]}s)"
            ),
            "value": 1.0,
            "color": "#B71C1C",
            "text": f"IRT = {rv['xi_mean']:.1f}",
        },
        {
            "label": f"Lowest Clustering Loyalty ({rv['loyalty_rate']:.0%})",
            "value": rv.get("loyalty_rate", 0),
            "color": "#FF8F00",
            "text": f"{rv.get('loyalty_rate', 0):.0%}",
        },
    ]

    unity = rv.get("unity_score")
    if unity is not None:
        bars.append(
            {
                "label": f"High Party Unity ({unity:.0%})",
                "value": unity,
                "color": "#2E7D32",
                "text": f"{unity:.0%}",
            }
        )

    fig, ax = plt.subplots(figsize=(12, 5))
    y_pos = np.arange(len(bars))

    for i, b in enumerate(bars):
        ax.barh(i, b["value"], color=b["color"], alpha=0.85, height=0.55, edgecolor="white")
        # Value label
        x = b["value"] + 0.02
        if b["value"] > 0.85:
            x = b["value"] - 0.02
            ax.text(
                x,
                i,
                b["text"],
                va="center",
                ha="right",
                fontsize=12,
                fontweight="bold",
                color="white",
            )
        else:
            ax.text(
                x,
                i,
                b["text"],
                va="center",
                ha="left",
                fontsize=12,
                fontweight="bold",
                color="#333333",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([b["label"] for b in bars], fontsize=11)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()

    ax.set_title(
        f"{paradox.full_name} ({paradox.party[0]}-{paradox.district}): "
        "Three Measures, Three Answers",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    # Annotation explaining the paradox
    explanation = (
        "Why do these disagree? IRT measures overall ideology across all votes.\n"
        f"Clustering loyalty measures agreement with fellow {paradox.party}s on contested votes.\n"
        "Party Unity (CQ) counts only votes where parties formally oppose each other.\n"
        f"{paradox.full_name} defects {paradox.direction} — away from their party's mainstream."
    )
    fig.text(
        0.5,
        -0.02,
        explanation,
        ha="center",
        fontsize=9,
        fontstyle="italic",
        color="#555555",
        wrap=True,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    out = plots_dir / f"metric_paradox_{paradox.chamber}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


def _extract_best_auc(upstream: dict) -> float | None:
    """Extract best XGBoost AUC from holdout_results parquets."""
    best = None
    for chamber in ("house", "senate"):
        hr = upstream.get(chamber, {}).get("holdout_results")
        if hr is None:
            continue
        xgb = hr.filter(pl.col("model") == "XGBoost")
        if xgb.height > 0 and "auc" in xgb.columns:
            auc = xgb["auc"].item()
            if best is None or auc > best:
                best = auc
    return best


def plot_pipeline_summary(
    manifests: dict,
    plots_dir: Path,
    session: str = "",
    *,
    upstream: dict | None = None,
) -> Path:
    """Pipeline summary infographic: five boxes connected by arrows."""
    eda = manifests.get("eda", {})
    indices = manifests.get("indices", {})
    clustering = manifests.get("clustering", {})

    total_votes = eda.get("All", {}).get("votes_before", "?")
    contested = eda.get("All", {}).get("votes_after", "?")
    h_party = indices.get("house_n_party_votes", 0)
    s_party = indices.get("senate_n_party_votes", 0)
    party_votes = h_party + s_party if (h_party or s_party) else "?"

    # Best AUC across chambers — extract from holdout_results parquets if available
    best_auc = _extract_best_auc(upstream) if upstream else None
    auc_label = f"{best_auc:.2f}" if best_auc is not None else "N/A"

    # Optimal k from clustering manifest
    k_optimal = clustering.get("house_optimal_k", 2)

    boxes = [
        {
            "label": f"{total_votes}\nRoll Calls",
            "sub": f"All recorded votes\nin {session}" if session else "All recorded votes",
        },
        {"label": f"{contested}\nContested", "sub": "After removing\nnear-unanimous votes"},
        {"label": f"{party_votes}\nParty Votes", "sub": "Where parties\nformally disagree"},
        {"label": f"k = {k_optimal}\nClusters", "sub": "Party is the only\nstable grouping"},
        {"label": f"AUC {auc_label}", "sub": "Model predicts votes\nwith high confidence"},
    ]

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(-0.5, len(boxes) * 2.5 - 1)
    ax.set_ylim(-1.5, 2.5)
    ax.axis("off")

    box_w, box_h = 2.0, 1.8
    colors = ["#1565C0", "#1976D2", "#1E88E5", "#42A5F5", "#E53935"]

    for i, b in enumerate(boxes):
        x = i * 2.5
        y = 0.0
        fancy = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.15",
            facecolor=colors[i],
            edgecolor="white",
            linewidth=2,
        )
        ax.add_patch(fancy)

        # Main label
        ax.text(
            x,
            y + 0.15,
            b["label"],
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
        )
        # Subtitle
        ax.text(
            x, y - 0.65, b["sub"], ha="center", va="center", fontsize=8, color="white", alpha=0.85
        )

        # Arrow to next box
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=((i + 1) * 2.5 - box_w / 2, y),
                xytext=(x + box_w / 2, y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#666666",
                    "lw": 2,
                    "mutation_scale": 20,
                },
            )

    ax.set_title(
        f"From {total_votes} Votes to One Number: AUC {auc_label}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    out = plots_dir / "pipeline_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ── Argument Parsing ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session",
        default="2025-26",
        help="Session identifier (default: %(default)s)",
    )
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--scrolly",
        action="store_true",
        help="Use scrollytelling layout (progressive narrative reveal)",
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    with RunContext(
        session=args.session,
        analysis_name="11_synthesis",
        params=vars(args),
        primer=SYNTHESIS_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        # Resolve results base path
        from tallgrass.session import STATE_DIR

        results_base = Path("results") / STATE_DIR / ctx.session
        print(f"Loading upstream data from {results_base}")

        # ── Load ─────────────────────────────────────────────────────────
        upstream = load_all_upstream(results_base, run_id=args.run_id)
        manifests = upstream["manifests"]

        # ── Join ─────────────────────────────────────────────────────────
        leg_dfs: dict[str, pl.DataFrame] = {}
        for chamber in ("house", "senate"):
            print(f"\nBuilding unified DataFrame: {chamber}")
            df = build_legislator_df(upstream, chamber)
            leg_dfs[chamber] = df
            print(f"  {chamber}: {df.height} legislators, {df.width} columns")
            df.write_parquet(ctx.data_dir / f"legislator_df_{chamber}.parquet")
            ctx.export_csv(
                df,
                f"full_scorecard_{chamber}.csv",
                f"Full legislator scorecard for {chamber.title()} (all metrics)",
            )

        # ── Detect Notable Legislators ────────────────────────────────
        print("\nDetecting notable legislators...")
        notables = detect_all(leg_dfs, network_manifest=manifests.get("06_network"))

        for chamber, mav in notables["mavericks"].items():
            print(f"  {chamber} maverick: {mav.full_name} ({mav.party})")
        for chamber, mav in notables.get("minority_mavericks", {}).items():
            print(f"  {chamber} minority maverick: {mav.full_name} ({mav.party})")
        for chamber, bridge in notables["bridges"].items():
            print(f"  {chamber} bridge: {bridge.full_name} ({bridge.party})")
        for slug, paradox in notables["paradoxes"].items():
            print(f"  paradox: {paradox.full_name} ({paradox.direction})")

        # ── New Plots ────────────────────────────────────────────────────
        print("\nGenerating new plots...")

        # Pipeline summary
        plot_pipeline_summary(manifests, ctx.plots_dir, session=ctx.session, upstream=upstream)

        # Dashboard scatters — pass dynamic annotation slugs
        for chamber in ("house", "senate"):
            plot_dashboard_scatter(
                leg_dfs[chamber],
                chamber,
                ctx.plots_dir,
                annotate_slugs=notables["annotations"].get(chamber, []),
            )

        # Profile cards — dynamic selection
        for slug, notable in notables["profiles"].items():
            plot_profile_card(
                leg_dfs[notable.chamber],
                slug,
                {
                    "title": notable.title,
                    "role": notable.role,
                    "subtitle": notable.subtitle,
                    "chamber": notable.chamber,
                },
                ctx.plots_dir,
            )

        # Metric paradox — dynamic (may be 0 or more)
        for slug, paradox in notables["paradoxes"].items():
            plot_metric_paradox(leg_dfs[paradox.chamber], paradox, ctx.plots_dir)

        # ── Resolve upstream plot paths ──────────────────────────────────
        upstream_plots: dict[str, Path] = {}
        plot_map = {
            "community_network_house": "06_network/community_network_house.png",
            "community_network_senate": "06_network/community_network_senate.png",
            "forest_house": "04_irt/forest_house.png",
            "forest_senate": "04_irt/forest_senate.png",
            "maverick_landscape_house": "07_indices/maverick_landscape_house.png",
            "per_legislator_accuracy_house": "08_prediction/per_legislator_accuracy_house.png",
            "shap_bar_house": "08_prediction/shap_bar_house.png",
            "irt_clusters_house": "05_clustering/irt_clusters_house.png",
            "convergence_summary_house": "04_irt/convergence_summary_house.png",
            "calibration_house": "08_prediction/calibration_house.png",
            "agreement_heatmap_house": "01_eda/agreement_heatmap_house.png",
            "discrimination_house": "04_irt/discrimination_house.png",
            "maverick_landscape_senate": "07_indices/maverick_landscape_senate.png",
            "per_legislator_accuracy_senate": "08_prediction/per_legislator_accuracy_senate.png",
            "umap_landscape_house": "03_umap/umap_landscape_house.png",
            "umap_landscape_senate": "03_umap/umap_landscape_senate.png",
        }
        for key, rel_path in plot_map.items():
            phase = rel_path.split("/")[0]
            filename = rel_path.split("/")[1]
            full_path = upstream["plots"].get(phase, Path()) / filename
            if full_path.exists():
                upstream_plots[key] = full_path
            else:
                print(f"  WARNING: upstream plot not found: {full_path}")

        # ── Build Report ─────────────────────────────────────────────────
        report_style = "scrolly" if args.scrolly else "linear"
        print(f"\nBuilding synthesis report ({report_style})...")
        ctx.report.title = f"Kansas Legislature {ctx.session} — Synthesis Report"

        report_kwargs = dict(
            leg_dfs=leg_dfs,
            manifests=manifests,
            upstream=upstream,
            plots_dir=ctx.plots_dir,
            upstream_plots=upstream_plots,
            notables=notables,
            session=ctx.session,
        )
        if args.scrolly:
            build_scrolly_synthesis_report(ctx.report, **report_kwargs)
        else:
            build_synthesis_report(ctx.report, **report_kwargs)

        # ── Manifest ─────────────────────────────────────────────────────
        manifest = {
            "analysis": "synthesis",
            "upstream_phases": UPSTREAM_PHASES,
            "house_n_legislators": leg_dfs["house"].height,
            "house_n_columns": leg_dfs["house"].width,
            "senate_n_legislators": leg_dfs["senate"].height,
            "senate_n_columns": leg_dfs["senate"].width,
            "new_plots": sorted(str(p.name) for p in ctx.plots_dir.iterdir()),
            "reused_plots": sorted(upstream_plots.keys()),
            "report_sections": len(ctx.report._sections),
        }
        manifest_path = ctx.run_dir / "filtering_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        print(f"\nSynthesis complete. Results: {ctx.run_dir}")


if __name__ == "__main__":
    main()
