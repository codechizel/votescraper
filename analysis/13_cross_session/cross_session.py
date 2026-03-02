"""
Kansas Legislature — Cross-Session Validation

Compares two bienniums (default: 2023-24 and 2025-26) across four dimensions:
1. Ideology stability: IRT ideal point shifts for returning legislators
2. Metric consistency: Party loyalty, network influence, maverick rates
3. Prediction transfer: Train on one session, test on the other (AUC comparison)
4. Detection validation: Do synthesis thresholds generalize across sessions?

Usage:
  uv run python analysis/cross_session.py
  uv run python analysis/cross_session.py --session-a 2023-24 --session-b 2025-26
  uv run python analysis/cross_session.py --chambers house
  uv run python analysis/cross_session.py --skip-prediction

Outputs (in results/kansas/cross-session/<pair>/<YYMMDD>.n/):
  - data/:   Parquet files (ideology_shift, metric_stability, prediction_transfer)
  - plots/:  PNG visualizations (shift scatter, movers, turnover, prediction AUC)
  - filtering_manifest.json, run_info.json, run_log.txt
  - validation_report.html
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.cross_session_data import (
        CORRELATION_WARN,
        FEATURE_IMPORTANCE_TOP_K,
        SHIFT_THRESHOLD_SD,
        align_feature_columns,
        align_irt_scales,
        analyze_freshmen_cohort,
        classify_turnover,
        compare_feature_importance,
        compute_bloc_stability,
        compute_ideology_shift,
        compute_metric_stability,
        compute_turnover_impact,
        match_legislators,
        standardize_features,
    )
except ModuleNotFoundError:
    from cross_session_data import (  # type: ignore[no-redef]
        CORRELATION_WARN,
        FEATURE_IMPORTANCE_TOP_K,
        SHIFT_THRESHOLD_SD,
        align_feature_columns,
        align_irt_scales,
        analyze_freshmen_cohort,
        classify_turnover,
        compare_feature_importance,
        compute_bloc_stability,
        compute_ideology_shift,
        compute_metric_stability,
        compute_turnover_impact,
        match_legislators,
        standardize_features,
    )

try:
    from analysis.cross_session_report import build_cross_session_report
except ModuleNotFoundError:
    from cross_session_report import build_cross_session_report  # type: ignore[no-redef]

try:
    from analysis.run_context import RunContext, resolve_upstream_dir, strip_leadership_suffix
except ModuleNotFoundError:
    from run_context import (  # type: ignore[no-redef]
        RunContext,
        resolve_upstream_dir,
        strip_leadership_suffix,
    )

try:
    from analysis.synthesis_data import (
        _read_parquet_safe,
        build_legislator_df,
        load_all_upstream,
    )
except ModuleNotFoundError:
    from synthesis_data import (  # type: ignore[no-redef]
        _read_parquet_safe,
        build_legislator_df,
        load_all_upstream,
    )

try:
    from analysis.synthesis_detect import (
        detect_bridge_builder,
        detect_chamber_maverick,
        detect_metric_paradox,
    )
except ModuleNotFoundError:
    from synthesis_detect import (  # type: ignore[no-redef]
        detect_bridge_builder,
        detect_chamber_maverick,
        detect_metric_paradox,
    )

# ── Constants ────────────────────────────────────────────────────────────────

PARTY_COLORS: dict[str, str] = {
    "Republican": "#E81B23",
    "Democrat": "#0015BC",
    "Independent": "#999999",
}
TOP_MOVERS_N: int = 15
ANNOTATE_N: int = 5
XGBOOST_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
    "eval_metric": "logloss",
    "verbosity": 0,
    "n_jobs": -1,
}

# ── Primer ───────────────────────────────────────────────────────────────────

CROSS_SESSION_PRIMER = """\
# Cross-Session Validation

## Purpose

Compares two Kansas Legislature bienniums to answer four questions:

1. **Who moved ideologically?** IRT ideal points for returning legislators are
   placed on a common scale via affine transformation, then compared.
2. **Are our metrics stable?** Party loyalty, maverick rates, and network
   influence are correlated across sessions for returning legislators.
3. **Do vote prediction models generalize?** Train XGBoost on one session's
   vote features, test on the other. Compare cross-session AUC to within-session.
4. **Do our detection methods generalize?** Synthesis detection thresholds
   (maverick, bridge-builder, metric paradox) are run on both sessions.

## Method

IRT ideal points are fitted independently per session. To compare them, we use
a robust affine transformation fitted on the overlapping legislators, trimming
the most extreme residuals (genuine movers) before the final fit.

For prediction transfer, features are z-score standardized within each session
before cross-session application (since IRT scales differ). Feature importance
rankings are compared via Kendall's tau on SHAP values.

## Inputs

Reads from both sessions' `results/<session>/` directories:
- IRT ideal points (per chamber)
- Synthesis legislator DataFrames (all upstream phases joined)
- Vote features parquets (from prediction phase, if available)
- Raw legislator CSVs (for matching)

## Outputs

- `ideology_shift_{chamber}.parquet` — per-legislator shift metrics
- `metric_stability_{chamber}.parquet` — cross-session correlations
- `prediction_transfer_{chamber}.parquet` — cross-session AUC/accuracy
- `feature_importance_{chamber}.parquet` — SHAP comparison
- `turnover_impact_{chamber}.json` — cohort distribution comparison
- `detection_validation.json` — detection threshold comparison
- `validation_report.html` — narrative HTML report

## Interpretation Guide

- **Ideology shift scatter:** Dots on the diagonal = no change. Dots above =
  moved rightward (more conservative). Dots below = moved leftward.
- **Significant movers:** Flagged when |shift| > 1 SD of all shifts.
- **Metric stability:** Pearson r > 0.7 = good stability. < 0.5 = weak.
- **Prediction transfer:** Cross-session AUC < within-session is expected.
  A large drop (>0.1) indicates session-specific overfitting.
- **Feature importance:** Kendall's tau > 0.7 = stable model structure.
- **Detection validation:** Same role flagged in both sessions = threshold
  generalizes. Different people in the same role = expected turnover.
"""


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Cross-Session Validation")
    parser.add_argument("--session-a", default="2023-24", help="Earlier session (default: 2023-24)")
    parser.add_argument("--session-b", default="2025-26", help="Later session (default: 2025-26)")
    parser.add_argument(
        "--chambers",
        default="both",
        choices=["house", "senate", "both"],
        help="Which chambers to analyze (default: both)",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Skip cross-session prediction transfer (faster)",
    )
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────


def print_header(title: str) -> None:
    width = 80
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _majority_party(leg_df: pl.DataFrame) -> str | None:
    """Return the party with the most legislators."""
    counts = leg_df.group_by("party").len().sort("len", descending=True)
    if counts.height == 0:
        return None
    return counts["party"][0]


def _extract_name(full_name: str) -> str:
    """Extract last name for plot annotation, stripping leadership suffixes."""
    name = full_name.split(" - ")[0].strip()
    parts = name.split()
    return parts[-1] if parts else full_name


# ── Plot Functions ───────────────────────────────────────────────────────────


def plot_ideology_shift_scatter(
    shifted: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
    session_a_label: str,
    session_b_label: str,
) -> None:
    """Scatter: previous ideology vs current ideology, colored by party."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Diagonal (no change)
    all_xi = shifted.select("xi_a_aligned", "xi_b").to_numpy().flatten()
    lo, hi = float(np.min(all_xi)) - 0.3, float(np.max(all_xi)) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color="#999999", linewidth=1, zorder=1)

    # Points by party
    for party, color in PARTY_COLORS.items():
        subset = shifted.filter(pl.col("party") == party)
        if subset.height == 0:
            continue
        ax.scatter(
            subset["xi_a_aligned"].to_numpy(),
            subset["xi_b"].to_numpy(),
            c=color,
            label=party,
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidth=0.5,
            zorder=2,
        )

    # Annotate top movers
    top = shifted.sort("abs_delta_xi", descending=True).head(ANNOTATE_N)
    for row in top.iter_rows(named=True):
        name = _extract_name(row["full_name"])
        ax.annotate(
            name,
            (row["xi_a_aligned"], row["xi_b"]),
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
            xytext=(5, 5),
            textcoords="offset points",
            zorder=3,
        )

    ax.set_xlabel(f"Ideology — {session_a_label} (aligned)", fontsize=11)
    ax.set_ylabel(f"Ideology — {session_b_label}", fontsize=11)
    ax.set_title(f"{chamber}: Who Moved Between Sessions?", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    fig.tight_layout()
    save_fig(fig, plots_dir / f"ideology_shift_scatter_{chamber.lower()}.png")


def plot_biggest_movers(
    shifted: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Horizontal bar chart of top N biggest movers by |delta_xi|."""
    top = shifted.sort("abs_delta_xi", descending=True).head(TOP_MOVERS_N)
    if top.height == 0:
        return

    names = [_extract_name(n) for n in top["full_name"].to_list()]
    deltas = top["delta_xi"].to_numpy()
    colors = ["#E81B23" if d > 0 else "#0015BC" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, deltas, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Ideology Shift (rightward →, ← leftward)", fontsize=10)
    ax.set_title(
        f"{chamber}: Top {top.height} Biggest Ideological Movers",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()
    save_fig(fig, plots_dir / f"biggest_movers_{chamber.lower()}.png")


def plot_shift_distribution(
    shifted: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Histogram of ideology shifts with threshold lines."""
    deltas = shifted["delta_xi"].to_numpy()
    # ddof=1 matches Polars std() used in compute_ideology_shift
    std = float(np.std(deltas, ddof=1))
    threshold = SHIFT_THRESHOLD_SD * std

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(deltas, bins=25, color="#666666", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="#333333", linewidth=1, linestyle="-")
    ax.axvline(threshold, color="#E81B23", linewidth=1.5, linestyle="--", label="Significant mover")
    ax.axvline(-threshold, color="#0015BC", linewidth=1.5, linestyle="--")

    n_movers = int(np.sum(np.abs(deltas) > threshold))
    ax.set_xlabel("Ideology Shift", fontsize=11)
    ax.set_ylabel("Number of Legislators", fontsize=11)
    ax.set_title(
        f"{chamber}: Distribution of Ideology Shifts ({n_movers} significant movers)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    save_fig(fig, plots_dir / f"shift_distribution_{chamber.lower()}.png")


def plot_turnover_impact(
    xi_returning: np.ndarray,
    xi_departing: np.ndarray,
    xi_new: np.ndarray,
    chamber: str,
    plots_dir: Path,
    session_a_label: str,
    session_b_label: str,
) -> None:
    """Strip plot comparing ideology distributions by turnover cohort."""
    fig, ax = plt.subplots(figsize=(8, 4))

    cohorts = [
        (f"Departing\n(left after {session_a_label})", xi_departing, "#999999"),
        ("Returning", xi_returning, "#555555"),
        (f"New\n(joined in {session_b_label})", xi_new, "#333333"),
    ]

    rng = np.random.default_rng(42)
    for i, (label, data, color) in enumerate(cohorts):
        if len(data) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(data))
        ax.scatter(
            data,
            np.full_like(data, i) + jitter,
            c=color,
            alpha=0.6,
            s=20,
            edgecolors="white",
            linewidth=0.3,
        )
        ax.plot(
            [np.mean(data)],
            [i],
            marker="D",
            color=color,
            markersize=10,
            zorder=5,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

    ax.set_yticks(range(len(cohorts)))
    ax.set_yticklabels([c[0] for c in cohorts], fontsize=10)
    ax.set_xlabel("IRT Ideology (liberal ← → conservative)", fontsize=11)
    ax.set_title(
        f"{chamber}: Who Left and Who Replaced Them?",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()
    save_fig(fig, plots_dir / f"turnover_impact_{chamber.lower()}.png")


# ── Prediction Plots ────────────────────────────────────────────────────────


def plot_prediction_comparison(
    within_auc_a: float,
    within_auc_b: float,
    cross_auc_ab: float,
    cross_auc_ba: float,
    chamber: str,
    plots_dir: Path,
    session_a_label: str,
    session_b_label: str,
) -> None:
    """Grouped bar chart: within-session vs cross-session AUC."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [
        f"Within {session_a_label}",
        f"Within {session_b_label}",
        f"Train {session_a_label}\nTest {session_b_label}",
        f"Train {session_b_label}\nTest {session_a_label}",
    ]
    values = [within_auc_a, within_auc_b, cross_auc_ab, cross_auc_ba]
    colors = ["#4CAF50", "#4CAF50", "#FF9800", "#FF9800"]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title(
        f"{chamber}: Within-Session vs Cross-Session Prediction",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.8, label="Random")
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_fig(fig, plots_dir / f"prediction_comparison_{chamber.lower()}.png")


def plot_feature_importance_comparison(
    comparison_df: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
    session_a_label: str,
    session_b_label: str,
    top_k: int | None = None,
) -> None:
    """Side-by-side horizontal bar chart of top-K SHAP features."""
    if top_k is None:
        top_k = FEATURE_IMPORTANCE_TOP_K

    try:
        from analysis.prediction import FEATURE_DISPLAY_NAMES
    except ModuleNotFoundError:
        from prediction import FEATURE_DISPLAY_NAMES  # type: ignore[no-redef]

    top = comparison_df.head(min(top_k, comparison_df.height))
    if top.height == 0:
        return

    names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in top["feature"].to_list()]
    imp_a = top["importance_a"].to_numpy()
    imp_b = top["importance_b"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, len(names) * 0.4)))

    y_pos = np.arange(len(names))
    ax1.barh(y_pos, imp_a, color="#4CAF50", edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean |SHAP|", fontsize=10)
    ax1.set_title(session_a_label, fontsize=11, fontweight="bold")

    ax2.barh(y_pos, imp_b, color="#FF9800", edgecolor="white")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean |SHAP|", fontsize=10)
    ax2.set_title(session_b_label, fontsize=11, fontweight="bold")

    fig.suptitle(
        f"{chamber}: Feature Importance Comparison (Top {top.height})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_fig(
        fig,
        plots_dir / f"feature_importance_comparison_{chamber.lower()}.png",
    )


# ── Freshmen & Bloc Plots ──────────────────────────────────────────────────


def _plot_freshmen_density(
    irt_b: pl.DataFrame,
    turnover: dict[str, pl.DataFrame],
    chamber: str,
    plots_dir: Path,
) -> None:
    """Density overlay of IRT ideal points: new vs returning legislators."""
    new_slugs = set(turnover["new"]["legislator_slug"].to_list())
    ret_slugs = set(turnover["returning"]["slug_b"].to_list())

    new_xi = irt_b.filter(pl.col("legislator_slug").is_in(new_slugs))["xi_mean"].to_numpy()
    ret_xi = irt_b.filter(pl.col("legislator_slug").is_in(ret_slugs))["xi_mean"].to_numpy()

    if len(new_xi) < 2 or len(ret_xi) < 2:
        return

    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(10, 5))

    xi_all = np.concatenate([new_xi, ret_xi])
    x_range = np.linspace(xi_all.min() - 0.5, xi_all.max() + 0.5, 200)

    kde_ret = gaussian_kde(ret_xi)
    kde_new = gaussian_kde(new_xi)

    ax.fill_between(x_range, kde_ret(x_range), alpha=0.4, color="#4a90d9", label="Returning")
    ax.fill_between(x_range, kde_new(x_range), alpha=0.4, color="#e74c3c", label="New Members")
    ax.plot(x_range, kde_ret(x_range), color="#4a90d9", linewidth=1.5)
    ax.plot(x_range, kde_new(x_range), color="#e74c3c", linewidth=1.5)

    ax.set_xlabel("IRT Ideology (Liberal ← → Conservative)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"{chamber}: Freshmen vs Returning — Ideology Distribution",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig(fig, plots_dir / f"freshmen_ideology_{chamber.lower()}.png")


def _plot_bloc_sankey(
    bloc: dict,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Generate a Plotly Sankey diagram of cluster transitions."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    transition_df = bloc.get("transition_df")
    if transition_df is None or transition_df.height == 0:
        return

    clusters_a = sorted(set(transition_df["cluster_a"].to_list()))
    clusters_b = sorted(set(transition_df["cluster_b"].to_list()))

    # Node labels
    labels = [f"Session A: Cluster {c}" for c in clusters_a] + [
        f"Session B: Cluster {c}" for c in clusters_b
    ]
    a_idx = {c: i for i, c in enumerate(clusters_a)}
    b_idx = {c: i + len(clusters_a) for i, c in enumerate(clusters_b)}

    sources, targets, values = [], [], []
    for row in transition_df.iter_rows(named=True):
        sources.append(a_idx[row["cluster_a"]])
        targets.append(b_idx[row["cluster_b"]])
        values.append(row["count"])

    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 15,
                    "thickness": 20,
                    "label": labels,
                    "color": ["#4a90d9"] * len(clusters_a) + ["#e74c3c"] * len(clusters_b),
                },
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "color": "rgba(200,200,200,0.4)",
                },
            )
        ]
    )
    fig.update_layout(
        title=f"{chamber}: Voting Bloc Transitions Between Sessions",
        font={"size": 12},
        width=700,
        height=400,
    )
    out = plots_dir / f"bloc_sankey_{chamber.lower()}.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"    Saved {out.name}")


# ── Detection Validation ────────────────────────────────────────────────────


def validate_detection(
    leg_df_a: pl.DataFrame,
    leg_df_b: pl.DataFrame,
    chamber: str,
) -> dict:
    """Run synthesis detection on both sessions, compare results."""
    result: dict = {}

    majority_a = _majority_party(leg_df_a)
    majority_b = _majority_party(leg_df_b)

    # Maverick
    mav_a = detect_chamber_maverick(leg_df_a, majority_a, chamber) if majority_a else None
    mav_b = detect_chamber_maverick(leg_df_b, majority_b, chamber) if majority_b else None
    result["maverick_a"] = mav_a.full_name if mav_a else None
    result["maverick_b"] = mav_b.full_name if mav_b else None

    # Bridge-builder
    bridge_a = detect_bridge_builder(leg_df_a, chamber)
    bridge_b = detect_bridge_builder(leg_df_b, chamber)
    result["bridge_a"] = bridge_a.full_name if bridge_a else None
    result["bridge_b"] = bridge_b.full_name if bridge_b else None

    # Paradox
    paradox_a = detect_metric_paradox(leg_df_a, chamber)
    paradox_b = detect_metric_paradox(leg_df_b, chamber)
    result["paradox_a"] = paradox_a.full_name if paradox_a else None
    result["paradox_b"] = paradox_b.full_name if paradox_b else None

    return result


# ── Cross-Session Prediction ────────────────────────────────────────────────


def _load_vote_features(
    ks: object,
    chamber: str,
) -> pl.DataFrame | None:
    """Load vote_features parquet from a session's prediction results."""
    results_dir = ks.results_dir  # type: ignore[attr-defined]
    path = results_dir / "08_prediction" / "latest" / "data" / f"vote_features_{chamber}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def _load_within_session_auc(ks: object, chamber: str) -> float | None:
    """Load the within-session XGBoost AUC from holdout results."""
    results_dir = ks.results_dir  # type: ignore[attr-defined]
    path = results_dir / "08_prediction" / "latest" / "data" / f"holdout_results_{chamber}.parquet"
    if not path.exists():
        return None
    holdout = pl.read_parquet(path)
    xgb_row = holdout.filter(pl.col("model") == "XGBoost")
    if xgb_row.height == 0:
        return None
    return float(xgb_row["auc"][0])


def _run_cross_prediction(
    ks_a: object,
    ks_b: object,
    chamber: str,
    chamber_cap: str,
    ctx: RunContext,
    session_a_label: str,
    session_b_label: str,
    matched: pl.DataFrame | None = None,
) -> dict | None:
    """Run cross-session prediction transfer for one chamber.

    Trains XGBoost on session A features, tests on session B
    (and vice versa). Compares SHAP feature importance.

    When *matched* is provided, vote features are filtered to returning
    legislators only — ensuring the cross-session test measures
    generalization on the *same* people in a new context.

    Returns dict with prediction metrics, or None if data is missing.
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from xgboost import XGBClassifier

    print("\n  Cross-session prediction transfer...")

    # Load vote features
    vf_a = _load_vote_features(ks_a, chamber)
    vf_b = _load_vote_features(ks_b, chamber)
    if vf_a is None or vf_b is None:
        missing = []
        if vf_a is None:
            missing.append(session_a_label)
        if vf_b is None:
            missing.append(session_b_label)
        print(f"    SKIP: vote features not found for {', '.join(missing)}")
        return None

    # Filter to returning legislators only (design doc §5, Experiment A)
    if matched is not None:
        ret_slugs_a = set(matched["slug_a"].to_list())
        ret_slugs_b = set(matched["slug_b"].to_list())
        vf_a = vf_a.filter(pl.col("legislator_slug").is_in(ret_slugs_a))
        vf_b = vf_b.filter(pl.col("legislator_slug").is_in(ret_slugs_b))

    # Align columns (intersection of features)
    vf_a, vf_b, feature_cols = align_feature_columns(vf_a, vf_b)
    print(f"    Shared features: {len(feature_cols)}")
    print(f"    Session A votes: {vf_a.height:,}")
    print(f"    Session B votes: {vf_b.height:,}")

    # Identify continuous columns for standardization
    numeric_cols = [c for c in feature_cols if vf_a[c].dtype in (pl.Float64, pl.Float32)]

    # Standardize within each session
    vf_a_std = standardize_features(vf_a, numeric_cols)
    vf_b_std = standardize_features(vf_b, numeric_cols)

    # Extract arrays
    X_a = vf_a_std.select(feature_cols).to_numpy().astype(np.float64)
    y_a = vf_a_std["vote_binary"].to_numpy()
    X_b = vf_b_std.select(feature_cols).to_numpy().astype(np.float64)
    y_b = vf_b_std["vote_binary"].to_numpy()

    # Train A → predict B
    print(f"    Training on {session_a_label}, testing on {session_b_label}...")
    model_ab = XGBClassifier(**XGBOOST_PARAMS)
    model_ab.fit(X_a, y_a)
    y_pred_ab = model_ab.predict(X_b)
    y_prob_ab = model_ab.predict_proba(X_b)[:, 1]
    auc_ab = float(roc_auc_score(y_b, y_prob_ab))
    acc_ab = float(accuracy_score(y_b, y_pred_ab))
    f1_ab = float(f1_score(y_b, y_pred_ab, zero_division=0))
    print(f"      AUC={auc_ab:.3f}, Accuracy={acc_ab:.3f}, F1={f1_ab:.3f}")

    # Train B → predict A
    print(f"    Training on {session_b_label}, testing on {session_a_label}...")
    model_ba = XGBClassifier(**XGBOOST_PARAMS)
    model_ba.fit(X_b, y_b)
    y_pred_ba = model_ba.predict(X_a)
    y_prob_ba = model_ba.predict_proba(X_a)[:, 1]
    auc_ba = float(roc_auc_score(y_a, y_prob_ba))
    acc_ba = float(accuracy_score(y_a, y_pred_ba))
    f1_ba = float(f1_score(y_a, y_pred_ba, zero_division=0))
    print(f"      AUC={auc_ba:.3f}, Accuracy={acc_ba:.3f}, F1={f1_ba:.3f}")

    # Within-session AUC for comparison
    within_auc_a = _load_within_session_auc(ks_a, chamber)
    within_auc_b = _load_within_session_auc(ks_b, chamber)
    if within_auc_a is not None:
        print(f"    Within-session AUC ({session_a_label}): {within_auc_a:.3f}")
    if within_auc_b is not None:
        print(f"    Within-session AUC ({session_b_label}): {within_auc_b:.3f}")

    # SHAP comparison
    print("    Computing SHAP feature importance...")
    import shap

    shap_sample_n = min(5000, X_a.shape[0], X_b.shape[0])
    rng = np.random.default_rng(42)
    idx_a = rng.choice(X_a.shape[0], shap_sample_n, replace=False)
    idx_b = rng.choice(X_b.shape[0], shap_sample_n, replace=False)

    explainer_ab = shap.TreeExplainer(model_ab)
    shap_ab = explainer_ab(X_b[idx_b]).values

    explainer_ba = shap.TreeExplainer(model_ba)
    shap_ba = explainer_ba(X_a[idx_a]).values

    comp_df, kendall_tau = compare_feature_importance(
        shap_ab,
        shap_ba,
        feature_cols,
    )
    print(f"    Kendall's tau (top-{FEATURE_IMPORTANCE_TOP_K} features): {kendall_tau:.3f}")

    # Plots
    if within_auc_a is not None and within_auc_b is not None:
        plot_prediction_comparison(
            within_auc_a,
            within_auc_b,
            auc_ab,
            auc_ba,
            chamber_cap,
            ctx.plots_dir,
            session_a_label,
            session_b_label,
        )

    plot_feature_importance_comparison(
        comp_df,
        chamber_cap,
        ctx.plots_dir,
        session_a_label,
        session_b_label,
    )

    # Save prediction parquets
    pred_summary = pl.DataFrame(
        {
            "direction": [
                f"{session_a_label}→{session_b_label}",
                f"{session_b_label}→{session_a_label}",
            ],
            "auc": [auc_ab, auc_ba],
            "accuracy": [acc_ab, acc_ba],
            "f1": [f1_ab, f1_ba],
        }
    )
    pred_summary.write_parquet(
        ctx.data_dir / f"prediction_transfer_{chamber}.parquet",
    )
    comp_df.write_parquet(
        ctx.data_dir / f"feature_importance_{chamber}.parquet",
    )

    return {
        "auc_ab": auc_ab,
        "auc_ba": auc_ba,
        "acc_ab": acc_ab,
        "acc_ba": acc_ba,
        "f1_ab": f1_ab,
        "f1_ba": f1_ba,
        "within_auc_a": within_auc_a,
        "within_auc_b": within_auc_b,
        "kendall_tau": kendall_tau,
        "feature_comparison": comp_df,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from tallgrass.session import KSSession

    ks_a = KSSession.from_session_string(args.session_a)
    ks_b = KSSession.from_session_string(args.session_b)
    session_a_label = ks_a.output_name
    session_b_label = ks_b.output_name
    comparison_label = f"{ks_a.legislature_number}-vs-{ks_b.legislature_number}"

    chambers = ["house", "senate"] if args.chambers == "both" else [args.chambers]

    with RunContext(
        session="cross-session",
        analysis_name=comparison_label,
        params=vars(args),
        primer=CROSS_SESSION_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print_header("Cross-Session Validation")
        print(f"  Session A: {session_a_label}")
        print(f"  Session B: {session_b_label}")

        # ── Load raw legislator CSVs ──
        print("\n── Loading legislator data ──")
        _suffix_strip = pl.col("full_name").map_elements(
            strip_leadership_suffix, return_dtype=pl.Utf8
        )
        _party_fill = (
            pl.col("party").fill_null("Independent").replace("", "Independent").alias("party")
        )
        leg_a = pl.read_csv(ks_a.data_dir / f"{ks_a.output_name}_legislators.csv").with_columns(
            _suffix_strip, _party_fill
        )
        leg_b = pl.read_csv(ks_b.data_dir / f"{ks_b.output_name}_legislators.csv").with_columns(
            _suffix_strip, _party_fill
        )
        print(f"  Session A: {leg_a.height} legislators")
        print(f"  Session B: {leg_b.height} legislators")

        # ── Match legislators ──
        print("\n── Matching legislators ──")
        matched = match_legislators(leg_a, leg_b)
        turnover = classify_turnover(leg_a, leg_b, matched)
        n_departing = turnover["departing"].height
        n_new = turnover["new"].height
        n_chamber_switch = int(matched["is_chamber_switch"].sum())
        n_party_switch = int(matched["is_party_switch"].sum())
        print(f"  Matched: {matched.height}")
        print(f"  Departing: {n_departing}")
        print(f"  New: {n_new}")
        if n_chamber_switch:
            print(f"  Chamber switches: {n_chamber_switch}")
        if n_party_switch:
            print(f"  Party switches: {n_party_switch}")

        # ── Load upstream results ──
        print("\n── Loading upstream analysis results ──")
        upstream_a = load_all_upstream(ks_a.results_dir, run_id=args.run_id)
        upstream_b = load_all_upstream(ks_b.results_dir, run_id=args.run_id)

        # ── Per-chamber analysis ──
        all_results: dict = {
            "matched": matched,
            "n_departing": n_departing,
            "n_new": n_new,
            "n_matched": matched.height,
            "chambers": chambers,
            "alignment_coefficients": {},
        }

        for chamber in chambers:
            chamber_cap = chamber.capitalize()
            print_header(f"{chamber_cap} Analysis")

            # Build legislator DataFrames
            leg_df_a = build_legislator_df(upstream_a, chamber)
            leg_df_b = build_legislator_df(upstream_b, chamber)
            print(f"  Legislator DF A: {leg_df_a.height} rows, {leg_df_a.width} cols")
            print(f"  Legislator DF B: {leg_df_b.height} rows, {leg_df_b.width} cols")

            # IRT ideal points
            irt_a = upstream_a[chamber].get("irt")
            irt_b = upstream_b[chamber].get("irt")
            if irt_a is None or irt_b is None:
                print(f"  WARNING: Missing IRT data for {chamber}, skipping")
                continue

            # ── Align IRT scales ──
            print("\n  Aligning IRT scales...")
            a_coef, b_coef, aligned = align_irt_scales(irt_a, irt_b, matched)
            print(f"    A = {a_coef:.4f}, B = {b_coef:.4f}")
            all_results["alignment_coefficients"][chamber_cap] = {"A": a_coef, "B": b_coef}

            # ── Ideology shift ──
            print("  Computing ideology shift...")
            shifted = compute_ideology_shift(aligned)
            n_movers = int(shifted["is_significant_mover"].sum())
            print(f"    {n_movers} significant movers out of {shifted.height}")

            # Correlation check
            xi_a_arr = shifted["xi_a_aligned"].to_numpy()
            xi_b_arr = shifted["xi_b"].to_numpy()
            r_val, _ = sp_stats.pearsonr(xi_a_arr, xi_b_arr)
            print(f"    Cross-session ideology correlation: r = {r_val:.3f}")
            if r_val < CORRELATION_WARN:
                print(f"    WARNING: r < {CORRELATION_WARN} — alignment may be unreliable")

            # ── Metric stability ──
            print("  Computing metric stability...")
            stability = compute_metric_stability(leg_df_a, leg_df_b, matched)
            for row in stability.iter_rows(named=True):
                flag = " ⚠" if row["pearson_r"] < CORRELATION_WARN else ""
                m, pr, sr = row["metric"], row["pearson_r"], row["spearman_rho"]
                print(f"    {m:20s}  r={pr:.3f}  ρ={sr:.3f}{flag}")

            # ── Turnover impact ──
            print("  Computing turnover impact...")
            chamber_matched = matched.filter(pl.col("chamber_b").str.to_lowercase() == chamber)
            dep_slugs = set(
                turnover["departing"]
                .filter(pl.col("chamber").str.to_lowercase() == chamber)["legislator_slug"]
                .to_list()
            )
            new_slugs = set(
                turnover["new"]
                .filter(pl.col("chamber").str.to_lowercase() == chamber)["legislator_slug"]
                .to_list()
            )

            # Get xi values for each cohort — all on Session B's scale.
            # Returning and new legislators come from irt_b directly.
            # Departing legislators come from irt_a and must be affine-transformed
            # onto Session B's scale using the alignment coefficients.
            ret_slugs = set(chamber_matched["slug_b"].to_list())
            xi_ret = irt_b.filter(pl.col("legislator_slug").is_in(ret_slugs))["xi_mean"].to_numpy()
            xi_dep_raw = irt_a.filter(pl.col("legislator_slug").is_in(dep_slugs))[
                "xi_mean"
            ].to_numpy()
            xi_dep = xi_dep_raw * a_coef + b_coef  # Transform to Session B scale
            xi_new = irt_b.filter(pl.col("legislator_slug").is_in(new_slugs))["xi_mean"].to_numpy()

            turnover_impact = compute_turnover_impact(xi_ret, xi_dep, xi_new)
            print(
                f"    Returning: n={turnover_impact['returning_n']}, "
                f"mean={turnover_impact['returning_mean']:.2f}"
            )
            if turnover_impact["departing_n"] > 0 and turnover_impact["departing_mean"] is not None:
                print(
                    f"    Departing: n={turnover_impact['departing_n']}, "
                    f"mean={turnover_impact['departing_mean']:.2f}"
                )
            if turnover_impact["new_n"] > 0 and turnover_impact["new_mean"] is not None:
                print(
                    f"    New:       n={turnover_impact['new_n']}, "
                    f"mean={turnover_impact['new_mean']:.2f}"
                )

            # ── Detection validation ──
            print("  Validating detection thresholds...")
            detection = validate_detection(leg_df_a, leg_df_b, chamber)
            for role in ["maverick", "bridge", "paradox"]:
                na = detection.get(f"{role}_a", "—")
                nb = detection.get(f"{role}_b", "—")
                same = "✓" if (na and nb and na == nb) else ""
                print(f"    {role:8s}  A: {na or '—':25s}  B: {nb or '—':25s}  {same}")

            # ── Plots ──
            print("  Generating plots...")
            plot_ideology_shift_scatter(
                shifted,
                chamber_cap,
                ctx.plots_dir,
                session_a_label,
                session_b_label,
            )
            plot_biggest_movers(shifted, chamber_cap, ctx.plots_dir)
            plot_shift_distribution(shifted, chamber_cap, ctx.plots_dir)
            plot_turnover_impact(
                xi_ret,
                xi_dep,
                xi_new,
                chamber_cap,
                ctx.plots_dir,
                session_a_label,
                session_b_label,
            )

            # ── Freshmen cohort analysis ──
            print("  Analyzing freshmen cohort...")
            freshmen_result = analyze_freshmen_cohort(
                turnover,
                leg_df_b,
                irt_b,
            )
            if freshmen_result is not None:
                print(f"    {freshmen_result.n_new} new, {freshmen_result.n_returning} returning")
                if freshmen_result.ideology_ks_p is not None:
                    print(f"    Ideology KS p = {freshmen_result.ideology_ks_p:.3f}")
                if freshmen_result.unity_t_p is not None:
                    print(f"    Unity t-test p = {freshmen_result.unity_t_p:.3f}")

                # Freshmen density overlay plot
                _plot_freshmen_density(
                    irt_b,
                    turnover,
                    chamber_cap,
                    ctx.plots_dir,
                )
            else:
                print("    Skipped: insufficient data")

            # ── Bloc stability ──
            print("  Analyzing voting bloc stability...")
            bloc_result: dict | None = None
            clust_a = resolve_upstream_dir("05_clustering", ks_a.results_dir, args.run_id)
            clust_b = resolve_upstream_dir("05_clustering", ks_b.results_dir, args.run_id)
            km_a = _read_parquet_safe(clust_a / "data" / f"kmeans_assignments_{chamber}.parquet")
            km_b = _read_parquet_safe(clust_b / "data" / f"kmeans_assignments_{chamber}.parquet")
            if km_a is not None and km_b is not None:
                bloc_result = compute_bloc_stability(
                    km_a,
                    km_b,
                    matched,
                    leg_df_a,
                    leg_df_b,
                )
                if bloc_result is not None:
                    print(f"    ARI = {bloc_result['ari']:.3f}")
                    print(f"    {bloc_result['switchers'].height} switchers")
                    _plot_bloc_sankey(bloc_result, chamber_cap, ctx.plots_dir)
                else:
                    print("    Skipped: insufficient paired data")
            else:
                print("    Skipped: missing k-means data")

            # ── Cross-session prediction ──
            prediction_result: dict | None = None
            if not args.skip_prediction:
                prediction_result = _run_cross_prediction(
                    ks_a,
                    ks_b,
                    chamber,
                    chamber_cap,
                    ctx,
                    session_a_label,
                    session_b_label,
                    matched=matched,
                )

            # ── Save data ──
            shifted.write_parquet(
                ctx.data_dir / f"ideology_shift_{chamber}.parquet",
            )
            ctx.export_csv(
                shifted,
                f"ideology_shift_{chamber}.csv",
                f"Ideology shift between sessions for {chamber.title()}",
            )
            stability.write_parquet(
                ctx.data_dir / f"metric_stability_{chamber}.parquet",
            )
            ctx.export_csv(
                stability,
                f"metric_stability_{chamber}.csv",
                f"Metric stability between sessions for {chamber.title()}",
            )
            with open(ctx.data_dir / f"turnover_impact_{chamber}.json", "w") as f:
                json.dump(turnover_impact, f, indent=2)

            all_results[chamber] = {
                "shifted": shifted,
                "stability": stability,
                "turnover": turnover_impact,
                "detection": detection,
                "r_value": r_val,
                "prediction": prediction_result,
                "freshmen": freshmen_result,
                "bloc_stability": bloc_result,
            }

        # ── Save detection results ──
        detection_summary = {
            ch: all_results[ch]["detection"]
            for ch in chambers
            if ch in all_results and "detection" in all_results[ch]
        }
        with open(ctx.data_dir / "detection_validation.json", "w") as f:
            json.dump(detection_summary, f, indent=2)

        # ── Build report ──
        print_header("Building Report")
        build_cross_session_report(
            ctx.report,
            results=all_results,
            plots_dir=ctx.plots_dir,
            session_a_label=session_a_label,
            session_b_label=session_b_label,
        )

        print("\n  Done.")


if __name__ == "__main__":
    main()
