"""
Kansas Legislature — Issue-Specific Ideal Points (Phase 19, BT4)

Estimates per-topic Bayesian IRT ideal points by running the Phase 04 flat IRT
model on topic-stratified vote subsets.  Answers: "How conservative is each
legislator on education vs healthcare vs taxes?"

Uses Phase 18 BERTopic topics (and/or optional CAP classifications) to assign
bills to policy areas, then runs `build_irt_graph()` + `build_and_sample()` from
Phase 04 on each eligible topic.

Usage:
  uv run python analysis/22_issue_irt/issue_irt.py
  uv run python analysis/22_issue_irt/issue_irt.py --session 2025-26 --taxonomy bertopic
  uv run python analysis/22_issue_irt/issue_irt.py --taxonomy both --min-bills 15

Outputs (in results/<session>/<run_id>/22_issue_irt/):
  - data/:   Parquet files (per-topic ideal points, cross-topic matrix, correlations)
  - plots/:  PNG scatter plots, heatmaps
  - filtering_manifest.json, run_info.json, run_log.txt
  - 22_issue_irt_report.html
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir

try:
    from analysis.phase_utils import load_metadata, print_header, save_fig
except ModuleNotFoundError:
    from phase_utils import load_metadata, print_header, save_fig  # type: ignore[no-redef]

try:
    from analysis.issue_irt_report import build_issue_irt_report
except ModuleNotFoundError:
    from issue_irt_report import build_issue_irt_report  # type: ignore[no-redef]

try:
    from analysis.issue_irt_data import (
        ESS_THRESHOLD,
        MIN_BILLS_PER_TOPIC,
        MIN_LEGISLATORS_PER_TOPIC,
        MIN_VOTES_IN_TOPIC,
        OUTLIER_TOP_N,
        PARTY_COLORS,
        RHAT_THRESHOLD,
        align_topic_ideal_points,
        build_cross_topic_matrix,
        check_anchor_stability,
        compute_cross_topic_correlations,
        compute_topic_irt_correlation,
        compute_topic_pca_scores,
        filter_legislators_in_topic,
        get_eligible_topics,
        identify_topic_outliers,
        load_topic_assignments,
        subset_vote_matrix_for_topic,
    )
except ModuleNotFoundError:
    from issue_irt_data import (  # type: ignore[no-redef]
        ESS_THRESHOLD,
        MIN_BILLS_PER_TOPIC,
        MIN_LEGISLATORS_PER_TOPIC,
        MIN_VOTES_IN_TOPIC,
        OUTLIER_TOP_N,
        PARTY_COLORS,
        RHAT_THRESHOLD,
        align_topic_ideal_points,
        build_cross_topic_matrix,
        check_anchor_stability,
        compute_cross_topic_correlations,
        compute_topic_irt_correlation,
        compute_topic_pca_scores,
        filter_legislators_in_topic,
        get_eligible_topics,
        identify_topic_outliers,
        load_topic_assignments,
        subset_vote_matrix_for_topic,
    )

# ── Primer ───────────────────────────────────────────────────────────────────

ISSUE_IRT_PRIMER = """\
# Issue-Specific Ideal Points (BT4)

## Purpose

Estimate per-topic Bayesian IRT ideal points to answer: "How conservative is
each legislator on education vs healthcare vs taxes?"  While Phase 04 gives a
single overall position, real legislators may be conservative on fiscal issues
but moderate on social policy.

## Method

1. **Load Phase 18 topic assignments** (BERTopic and/or CAP classifications).
2. **For each eligible topic** (≥ {min_bills} bills with roll calls):
   a. Subset the EDA vote matrix to vote_ids in this topic.
   b. Filter legislators with ≥ {min_votes} non-null votes.
   c. Run standard Phase 04 flat IRT (2PL model with anchor constraints).
   d. Sign-align against the full-model IRT ideal points.
3. **Assemble cross-topic matrix** (legislator × topic ideal points).
4. **Analyze**: pairwise topic correlations, topic outliers, anchor stability.

## Inputs

- EDA vote matrices: `results/{{session}}/{{run_id}}/01_eda/data/`
- PCA scores: `results/{{session}}/{{run_id}}/02_pca/data/`
- Full IRT ideal points: `results/{{session}}/{{run_id}}/05_irt/data/`
- Phase 18 topics: `results/{{session}}/{{run_id}}/20_bill_text/data/`

## Outputs

All in `results/{{session}}/{{run_id}}/22_issue_irt/`:

| File | Description |
|------|-------------|
| `data/ideal_points_{{topic}}_{{chamber}}.parquet` | Per-topic ideal points |
| `data/cross_topic_matrix_{{chamber}}.parquet` | Legislator × topic matrix |
| `data/correlations.json` | Per-topic and cross-topic correlations |
| `plots/scatter_{{topic}}_{{chamber}}.png` | Topic xi vs full IRT xi |
| `plots/heatmap_{{chamber}}.png` | Cross-topic correlation heatmap |
| `plots/profile_{{chamber}}.png` | Legislator × topic ideological profile |

## Interpretation Guide

- **Per-topic r ≥ 0.80**: Strong — topic ideal points track overall ideology.
- **0.60 ≤ r < 0.80**: Good — some topic-specific divergence.
- **0.40 ≤ r < 0.60**: Moderate — meaningful policy-area differences.
- **r < 0.40**: Weak — topic may have distinct ideological alignment.
- **Outliers**: Legislators far from their overall position on a topic =
  cross-pressured on that policy area.

## Caveats

- Per-topic vote subsets are small (often 10-30 bills) → wider credible intervals.
- MCMC convergence thresholds relaxed (R-hat < 1.05, ESS > 200).
- BERTopic topics are data-driven and may not map cleanly to policy areas.
- CAP categories are standardized but require Claude API classification.
"""


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Issue-Specific Ideal Points (BT4)")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument(
        "--taxonomy",
        choices=["bertopic", "cap", "both"],
        default="bertopic",
        help="Topic taxonomy to use (default: bertopic)",
    )
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override flat IRT results directory")
    parser.add_argument("--bill-text-dir", default=None, help="Override Phase 18 results directory")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    parser.add_argument(
        "--min-bills",
        type=int,
        default=MIN_BILLS_PER_TOPIC,
        help=f"Min roll-call bills per topic (default: {MIN_BILLS_PER_TOPIC})",
    )
    parser.add_argument(
        "--min-legislators",
        type=int,
        default=MIN_LEGISLATORS_PER_TOPIC,
        help=f"Min legislators per topic (default: {MIN_LEGISLATORS_PER_TOPIC})",
    )
    parser.add_argument(
        "--min-votes-in-topic",
        type=int,
        default=MIN_VOTES_IN_TOPIC,
        help=f"Min votes per legislator in topic (default: {MIN_VOTES_IN_TOPIC})",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="MCMC draws per chain (default: 1000)"
    )
    parser.add_argument(
        "--n-tune", type=int, default=1000, help="MCMC tuning steps (default: 1000)"
    )
    parser.add_argument(
        "--n-chains", type=int, default=2, help="Number of MCMC chains (default: 2)"
    )
    return parser.parse_args()


# ── IRT Loading ──────────────────────────────────────────────────────────────


def _load_full_irt(
    results_root: Path,
    chamber: str,
    irt_dir_override: Path | None = None,
    run_id: str | None = None,
) -> pl.DataFrame | None:
    """Load full-model flat IRT ideal points for a chamber."""
    base = irt_dir_override or resolve_upstream_dir("05_irt", results_root, run_id)
    path = base / "data" / f"ideal_points_{chamber}.parquet"

    if path.exists():
        df = pl.read_parquet(path)
        print(f"  Full IRT ({chamber}): {df.height} legislators loaded")
        return df
    else:
        print(f"  Full IRT ({chamber}): not found at {path}")
        return None


# ── Per-Topic IRT ────────────────────────────────────────────────────────────


def _run_topic_irt(
    topic_matrix: pl.DataFrame,
    pca_scores: pl.DataFrame | None,
    full_irt: pl.DataFrame,
    chamber: str,
    topic_label: str,
    legislators: pl.DataFrame,
    n_samples: int,
    n_tune: int,
    n_chains: int,
) -> dict | None:
    """Run flat IRT on a per-topic vote matrix.

    Reuses Phase 04's prepare_irt_data, select_anchors, build_and_sample,
    check_convergence, and extract_ideal_points.

    Returns dict with ideal_points, convergence, sampling_time, or None on failure.
    """

    try:
        from analysis.irt import (
            build_and_sample,
            check_convergence,
            extract_ideal_points,
            prepare_irt_data,
            select_anchors,
        )
    except ModuleNotFoundError:
        from irt import (  # type: ignore[no-redef]
            build_and_sample,
            check_convergence,
            extract_ideal_points,
            prepare_irt_data,
            select_anchors,
        )

    slug_col = "legislator_slug"
    vote_cols = [c for c in topic_matrix.columns if c != slug_col]

    if len(vote_cols) < 2 or topic_matrix.height < 3:
        print(f"    Skipping '{topic_label}': too few votes or legislators")
        return None

    # Prepare IRT data
    try:
        irt_data = prepare_irt_data(topic_matrix, chamber)
    except Exception as e:
        print(f"    Skipping '{topic_label}': prepare_irt_data failed: {e}")
        return None

    # Anchor selection — try full model's PCA first, fallback to per-topic PCA
    try:
        if pca_scores is not None:
            cons_idx, cons_slug, lib_idx, lib_slug = select_anchors(
                pca_scores, topic_matrix, chamber
            )
        else:
            # Fallback: compute per-topic PCA
            topic_pca = compute_topic_pca_scores(topic_matrix)
            if topic_pca is None:
                print(f"    Skipping '{topic_label}': cannot compute PCA for anchor selection")
                return None
            # Add full_name column for select_anchors
            topic_pca_with_names = topic_pca.join(
                full_irt.select(["legislator_slug", "full_name"]),
                on="legislator_slug",
                how="left",
            ).with_columns(
                pl.when(pl.col("full_name").is_null())
                .then(pl.col("legislator_slug"))
                .otherwise(pl.col("full_name"))
                .alias("full_name")
            )
            cons_idx, cons_slug, lib_idx, lib_slug = select_anchors(
                topic_pca_with_names, topic_matrix, chamber
            )
    except Exception as e:
        print(f"    Skipping '{topic_label}': anchor selection failed: {e}")
        return None

    anchors = [(cons_idx, 1.0), (lib_idx, -1.0)]

    # PCA-informed init for xi_free
    xi_initvals = None
    if pca_scores is not None:
        try:
            pca_init = pca_scores.select("legislator_slug", "PC1")
            merged = pl.DataFrame({"legislator_slug": irt_data["leg_slugs"]}).join(
                pca_init, on="legislator_slug", how="left"
            )
            pc1 = merged["PC1"].to_numpy().astype(float)
            pc1 = np.nan_to_num(pc1, nan=0.0)

            # Normalize to ~N(0,1)
            std = np.std(pc1)
            if std > 0:
                pc1 = pc1 / std

            # Extract free (non-anchor) values
            anchor_indices = {cons_idx, lib_idx}
            free_vals = [pc1[i] for i in range(len(pc1)) if i not in anchor_indices]
            xi_initvals = np.array(free_vals)
        except Exception:
            xi_initvals = None

    # Sample
    try:
        idata, sampling_time = build_and_sample(
            irt_data,
            anchors,
            n_samples=n_samples,
            n_tune=n_tune,
            n_chains=n_chains,
            xi_initvals=xi_initvals,
        )
    except Exception as e:
        print(f"    Skipping '{topic_label}': sampling failed: {e}")
        return None

    # Convergence
    try:
        diag = check_convergence(idata, f"{chamber}/{topic_label}")
    except Exception as e:
        print(f"    Warning: convergence check failed for '{topic_label}': {e}")
        diag = {"all_ok": False, "error": str(e)}

    # Check convergence thresholds (relaxed for per-topic)
    converged = True
    if isinstance(diag.get("xi_rhat_max"), float) and diag["xi_rhat_max"] >= RHAT_THRESHOLD:
        converged = False
    if isinstance(diag.get("xi_ess_min"), float) and diag["xi_ess_min"] < ESS_THRESHOLD:
        converged = False

    if not converged:
        print(f"    WARNING: '{topic_label}' did not converge (R-hat or ESS out of range)")

    # Extract ideal points
    try:
        ideal_points = extract_ideal_points(idata, irt_data, legislators)
    except Exception as e:
        print(f"    Skipping '{topic_label}': ideal point extraction failed: {e}")
        return None

    return {
        "ideal_points": ideal_points,
        "convergence": diag,
        "sampling_time": sampling_time,
        "converged": converged,
        "anchor_slugs": (cons_slug, lib_slug),
        "n_legislators": irt_data["n_legislators"],
        "n_votes": irt_data["n_votes"],
    }


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_topic_scatter(
    topic_xi: pl.DataFrame,
    full_irt: pl.DataFrame,
    topic_label: str,
    chamber: str,
    corr: dict,
    out_dir: Path,
    topic_id: int,
) -> None:
    """Scatter plot: per-topic ideal point (y) vs full IRT (x), party-colored."""
    matched = topic_xi.select(["legislator_slug", pl.col("xi_mean").alias("topic_xi")]).join(
        full_irt.select(["legislator_slug", pl.col("xi_mean").alias("full_xi"), "party"]),
        on="legislator_slug",
        how="inner",
    )

    if matched.height < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    for party in ["Republican", "Democrat", "Independent"]:
        if "party" not in matched.columns:
            continue
        sub = matched.filter(pl.col("party") == party)
        if sub.height == 0:
            continue
        color = PARTY_COLORS.get(party, "#888888")
        ax.scatter(
            sub["full_xi"].to_numpy(),
            sub["topic_xi"].to_numpy(),
            c=color,
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label=f"{party} (n={sub.height})",
        )

    # Regression line
    xi_full = matched["full_xi"].to_numpy().astype(float)
    xi_topic = matched["topic_xi"].to_numpy().astype(float)
    if len(xi_full) >= 2:
        z = np.polyfit(xi_full, xi_topic, 1)
        p = np.poly1d(z)
        x_line = np.linspace(xi_full.min(), xi_full.max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.3, linewidth=1)

    r = corr.get("pearson_r", float("nan"))
    n = corr.get("n", 0)

    short_label = topic_label[:60]
    ax.set_xlabel("Full IRT Ideal Point", fontsize=11)
    ax.set_ylabel("Topic Ideal Point", fontsize=11)
    ax.set_title(
        f"{chamber} — {short_label}\nPearson r = {r:.3f} (n = {n})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    save_fig(fig, out_dir / f"scatter_t{topic_id}_{chamber.lower()}.png")


def plot_cross_topic_heatmap(
    cross_corr: pl.DataFrame,
    topic_labels: dict[str, str],
    chamber: str,
    out_dir: Path,
) -> None:
    """Heatmap of pairwise correlations between topic ideal points."""
    if cross_corr.height == 0:
        return

    # Build symmetric matrix
    all_topics = sorted(set(cross_corr["topic_a"].to_list() + cross_corr["topic_b"].to_list()))
    n = len(all_topics)
    if n < 2:
        return

    topic_idx = {t: i for i, t in enumerate(all_topics)}
    mat = np.eye(n)
    for row in cross_corr.iter_rows(named=True):
        i = topic_idx[row["topic_a"]]
        j = topic_idx[row["topic_b"]]
        r = row["pearson_r"]
        if not np.isnan(r):
            mat[i, j] = r
            mat[j, i] = r

    # Short labels
    short_labels = []
    for t in all_topics:
        label = topic_labels.get(t, t)
        # Trim prefix like "t0_" and truncate
        if label.startswith("t") and "_" in label:
            label = label[label.index("_") + 1 :]
        short_labels.append(label[:25])

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.6)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i != j:
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if abs(val) > 0.5 else "black",
                    )

    ax.set_title(
        f"{chamber} — Cross-Topic Ideal Point Correlations",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_fig(fig, out_dir / f"heatmap_{chamber.lower()}.png")


def plot_ideological_profile(
    cross_matrix: pl.DataFrame,
    full_irt: pl.DataFrame,
    topic_labels: dict[str, str],
    chamber: str,
    out_dir: Path,
) -> None:
    """Heatmap: legislator × topic, z-standardized, sorted by full IRT."""
    slug_col = "legislator_slug"
    topic_cols = [c for c in cross_matrix.columns if c != slug_col]

    if not topic_cols or cross_matrix.height < 3:
        return

    # Join full IRT for sort order and party
    joined = cross_matrix.join(
        full_irt.select([slug_col, "xi_mean", "party", "full_name"]),
        on=slug_col,
        how="inner",
    ).sort("xi_mean")

    if joined.height < 3:
        return

    # Build matrix and z-standardize per topic
    data = joined.select(topic_cols).to_numpy().astype(float)
    for j in range(data.shape[1]):
        col = data[:, j]
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            mean = np.nanmean(col)
            std = np.nanstd(col)
            if std > 0:
                data[valid, j] = (col[valid] - mean) / std

    # Short labels
    short_labels = []
    for t in topic_cols:
        label = topic_labels.get(t, t)
        if label.startswith("t") and "_" in label:
            label = label[label.index("_") + 1 :]
        short_labels.append(label[:25])

    legislator_labels = []
    parties = joined["party"].to_list()
    names = (
        joined["full_name"].to_list()
        if "full_name" in joined.columns
        else joined[slug_col].to_list()
    )
    for name, party in zip(names, parties, strict=True):
        p_code = "R" if party == "Republican" else "D" if party == "Democrat" else "I"
        short_name = str(name).split()[-1] if name else "?"
        legislator_labels.append(f"{short_name} ({p_code})")

    fig, ax = plt.subplots(figsize=(max(10, len(topic_cols) * 1.2), max(8, joined.height * 0.2)))

    im = ax.imshow(data, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    fig.colorbar(im, ax=ax, label="Z-score (within topic)", shrink=0.8)

    ax.set_xticks(range(len(topic_cols)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)

    # Only show every Nth legislator label to avoid clutter
    n_labels = min(40, joined.height)
    step = max(1, joined.height // n_labels)
    ytick_positions = list(range(0, joined.height, step))
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([legislator_labels[i] for i in ytick_positions], fontsize=7)

    ax.set_title(
        f"{chamber} — Ideological Profiles by Topic (sorted by overall IRT)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_fig(fig, out_dir / f"profile_{chamber.lower()}.png")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from tallgrass.session import KSSession

    session = KSSession.from_session_string(args.session)
    data_dir = Path(args.data_dir) if args.data_dir else session.data_dir
    results_root = session.results_dir

    # Check Phase 20 (bill text) output exists
    bt_dir = (
        Path(args.bill_text_dir)
        if args.bill_text_dir
        else resolve_upstream_dir("20_bill_text", results_root, args.run_id)
    )
    if not (bt_dir / "data").exists():
        print("[Phase 22] Skipping: bill text analysis not yet run (run `just text-analysis` first)")
        return

    # Resolve upstream dirs
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
    irt_dir = Path(args.irt_dir) if args.irt_dir else None
    bill_text_dir = (
        Path(args.bill_text_dir)
        if args.bill_text_dir
        else resolve_upstream_dir("20_bill_text", results_root, args.run_id)
    )

    # Determine taxonomies
    taxonomies: list[str] = []
    if args.taxonomy in ("bertopic", "both"):
        taxonomies.append("bertopic")
    if args.taxonomy in ("cap", "both"):
        taxonomies.append("cap")

    primer = ISSUE_IRT_PRIMER.format(min_bills=args.min_bills, min_votes=args.min_votes_in_topic)

    with RunContext(
        session=args.session,
        analysis_name="22_issue_irt",
        params=vars(args),
        results_root=results_root,
        primer=primer,
        run_id=args.run_id,
    ) as ctx:
        print(f"KS Legislature Issue-Specific Ideal Points — Session {args.session}")
        print(f"Taxonomies: {', '.join(taxonomies)}")
        print(f"MCMC:       {args.n_samples} draws, {args.n_tune} tune, {args.n_chains} chains")
        print(f"Thresholds: min_bills={args.min_bills}, min_legislators={args.min_legislators}")
        print(f"            min_votes_in_topic={args.min_votes_in_topic}")
        print(f"Output:     {ctx.run_dir}")

        # ── Load upstream data ──
        print_header("DATA LOADING")

        # EDA vote matrices
        eda_house_path = eda_dir / "data" / "vote_matrix_house_filtered.parquet"
        if not eda_house_path.exists():
            print("Phase 22: skipping — no EDA vote matrices available")
            return

        house_matrix = pl.read_parquet(eda_dir / "data" / "vote_matrix_house_filtered.parquet")
        senate_matrix = pl.read_parquet(eda_dir / "data" / "vote_matrix_senate_filtered.parquet")
        print(f"  House matrix: {house_matrix.height} x {len(house_matrix.columns) - 1}")
        print(f"  Senate matrix: {senate_matrix.height} x {len(senate_matrix.columns) - 1}")

        # PCA scores (optional, for anchor selection)
        pca_house_path = pca_dir / "data" / "pc_scores_house.parquet"
        if pca_house_path.exists():
            pca_house = pl.read_parquet(pca_dir / "data" / "pc_scores_house.parquet")
            pca_senate = pl.read_parquet(pca_dir / "data" / "pc_scores_senate.parquet")
            print(f"  PCA scores loaded: House={pca_house.height}, Senate={pca_senate.height}")
        else:
            pca_house, pca_senate = None, None
            print("  PCA scores not available — will use per-topic PCA for anchors")

        # Metadata
        rollcalls, legislators = load_metadata(data_dir)
        print(f"  Rollcalls: {rollcalls.height}, Legislators: {legislators.height}")

        # Full IRT ideal points
        full_irt: dict[str, pl.DataFrame | None] = {}
        for ch in ["house", "senate"]:
            full_irt[ch] = _load_full_irt(results_root, ch, irt_dir, args.run_id)

        # ── Per-taxonomy loop ──
        all_taxonomy_results: dict[str, dict] = {}

        for taxonomy in taxonomies:
            print_header(f"TAXONOMY: {taxonomy.upper()}")

            try:
                topics = load_topic_assignments(bill_text_dir, taxonomy)
                print(f"  Topics loaded: {topics.height} bill-topic assignments")
                print(
                    f"  Unique topics: {topics['topic_id'].n_unique()}, "
                    f"Unique bills: {topics['bill_number'].n_unique()}"
                )
            except FileNotFoundError as e:
                print(f"  {e}")
                if taxonomy == "cap":
                    print("  Falling back to BERTopic only")
                continue

            # ── Per-chamber loop ──
            taxonomy_results: dict[str, dict] = {}

            chamber_configs = [
                ("House", house_matrix, pca_house, "house"),
                ("Senate", senate_matrix, pca_senate, "senate"),
            ]

            for chamber, matrix, pca_scores, ch_key in chamber_configs:
                print_header(f"{taxonomy.upper()} — {chamber}")

                if full_irt[ch_key] is None:
                    print(f"  Skipping {chamber}: full IRT not available")
                    continue

                full_irt_chamber = full_irt[ch_key]

                # Filter rollcalls to chamber
                chamber_rollcalls = rollcalls
                if "chamber" in rollcalls.columns:
                    chamber_rollcalls = rollcalls.filter(pl.col("chamber") == chamber)

                # Get eligible topics
                eligible, report = get_eligible_topics(
                    topics, chamber_rollcalls, min_bills=args.min_bills
                )

                n_eligible = eligible.height
                n_skipped = report.filter(~pl.col("eligible")).height
                print(f"  Eligible topics: {n_eligible}, Skipped: {n_skipped}")

                if n_eligible == 0:
                    print(f"  No eligible topics for {chamber}")
                    continue

                # Save eligibility report
                report.write_parquet(
                    ctx.data_dir / f"topic_eligibility_{taxonomy}_{ch_key}.parquet"
                )

                # ── Per-topic IRT loop ──
                topic_results: dict[int, dict] = {}
                total_time = 0.0
                n_converged = 0
                n_failed = 0

                for row in eligible.iter_rows(named=True):
                    tid = row["topic_id"]
                    tlabel = row["topic_label"]
                    n_rc_bills = row["n_rollcall_bills"]

                    print(f"\n  --- Topic {tid}: {tlabel[:50]} ({n_rc_bills} bills) ---")

                    # Subset vote matrix
                    topic_matrix = subset_vote_matrix_for_topic(
                        matrix, topics, chamber_rollcalls, tid
                    )
                    vote_cols = [c for c in topic_matrix.columns if c != "legislator_slug"]
                    print(f"    Vote columns: {len(vote_cols)}")

                    if len(vote_cols) < 2:
                        print(f"    Skipping: too few vote columns ({len(vote_cols)})")
                        n_failed += 1
                        continue

                    # Filter legislators
                    topic_matrix = filter_legislators_in_topic(
                        topic_matrix, min_votes=args.min_votes_in_topic
                    )
                    print(f"    Legislators after filter: {topic_matrix.height}")

                    if topic_matrix.height < args.min_legislators:
                        print(
                            f"    Skipping: too few legislators "
                            f"({topic_matrix.height} < {args.min_legislators})"
                        )
                        n_failed += 1
                        continue

                    # Run IRT
                    result = _run_topic_irt(
                        topic_matrix,
                        pca_scores,
                        full_irt_chamber,
                        chamber,
                        tlabel,
                        legislators,
                        n_samples=args.n_samples,
                        n_tune=args.n_tune,
                        n_chains=args.n_chains,
                    )

                    if result is None:
                        n_failed += 1
                        continue

                    # Sign-align
                    result["ideal_points"] = align_topic_ideal_points(
                        result["ideal_points"], full_irt_chamber
                    )

                    # Correlation with full IRT
                    corr = compute_topic_irt_correlation(result["ideal_points"], full_irt_chamber)
                    result["correlation"] = corr

                    r_val = corr.get("pearson_r", float("nan"))
                    q_val = corr.get("quality", "?")
                    print(f"    r = {r_val:.3f} ({q_val}), time = {result['sampling_time']:.1f}s")

                    # Outliers
                    result["outliers"] = identify_topic_outliers(
                        result["ideal_points"], full_irt_chamber, top_n=OUTLIER_TOP_N
                    )

                    # Save per-topic parquet
                    result["ideal_points"].write_parquet(
                        ctx.data_dir / f"ideal_points_t{tid}_{ch_key}.parquet"
                    )

                    # Plot scatter
                    plot_topic_scatter(
                        result["ideal_points"],
                        full_irt_chamber,
                        tlabel,
                        chamber,
                        corr,
                        ctx.plots_dir,
                        tid,
                    )

                    topic_results[tid] = {
                        "label": tlabel,
                        **result,
                    }

                    total_time += result["sampling_time"]
                    if result["converged"]:
                        n_converged += 1

                print(
                    f"\n  {chamber} summary: {len(topic_results)} modeled, "
                    f"{n_converged} converged, {n_failed} skipped"
                )
                print(f"  Total MCMC time: {total_time:.1f}s")

                if not topic_results:
                    continue

                # ── Cross-topic analysis ──
                print_header(f"CROSS-TOPIC ANALYSIS — {chamber}")

                # Build cross-topic matrix
                cross_matrix = build_cross_topic_matrix(topic_results)
                cross_matrix.write_parquet(ctx.data_dir / f"cross_topic_matrix_{ch_key}.parquet")

                # Column name → label mapping
                topic_col_labels = {}
                for col in cross_matrix.columns:
                    if col != "legislator_slug":
                        topic_col_labels[col] = col

                # Cross-topic correlations
                cross_corr = compute_cross_topic_correlations(cross_matrix)
                print(f"  Cross-topic pairs: {cross_corr.height}")

                if cross_corr.height > 0:
                    cross_corr.write_parquet(
                        ctx.data_dir / f"cross_topic_correlations_{ch_key}.parquet"
                    )
                    plot_cross_topic_heatmap(cross_corr, topic_col_labels, chamber, ctx.plots_dir)

                # Ideological profile heatmap
                plot_ideological_profile(
                    cross_matrix, full_irt_chamber, topic_col_labels, chamber, ctx.plots_dir
                )

                # Anchor stability
                anchor_slugs_set = set()
                for tr in topic_results.values():
                    if "anchor_slugs" in tr:
                        anchor_slugs_set.update(tr["anchor_slugs"])

                if len(anchor_slugs_set) >= 2:
                    # Use the first topic's anchors (consistent across topics in a chamber)
                    first_anchors = list(topic_results.values())[0].get("anchor_slugs", ("", ""))
                    anchor_stability = check_anchor_stability(
                        topic_results, full_irt_chamber, first_anchors
                    )
                    if anchor_stability.height > 0:
                        anchor_stability.write_parquet(
                            ctx.data_dir / f"anchor_stability_{ch_key}.parquet"
                        )

                taxonomy_results[ch_key] = {
                    "chamber": chamber,
                    "topic_results": topic_results,
                    "cross_matrix": cross_matrix,
                    "cross_correlations": cross_corr,
                    "eligibility_report": report,
                    "n_converged": n_converged,
                    "n_failed": n_failed,
                    "total_time": total_time,
                }

            all_taxonomy_results[taxonomy] = taxonomy_results

        # ── Save correlations JSON ──
        print_header("CORRELATIONS SUMMARY")
        corr_summary: dict = {}
        for taxonomy, tax_results in all_taxonomy_results.items():
            corr_summary[taxonomy] = {}
            for ch_key, ch_data in tax_results.items():
                chamber_corrs: dict = {}
                for tid, tr in ch_data["topic_results"].items():
                    chamber_corrs[f"t{tid}_{tr['label'][:30]}"] = tr.get("correlation", {})
                corr_summary[taxonomy][ch_key] = chamber_corrs

        corr_path = ctx.data_dir / "correlations.json"
        with open(corr_path, "w") as f:
            json.dump(corr_summary, f, indent=2, default=str)
        print(f"  Saved: {corr_path.name}")

        # ── Filtering manifest ──
        print_header("FILTERING MANIFEST")
        manifest: dict = {
            "analysis": "issue_irt",
            "session": args.session,
            "taxonomies": taxonomies,
            "mcmc": {
                "n_samples": args.n_samples,
                "n_tune": args.n_tune,
                "n_chains": args.n_chains,
            },
            "thresholds": {
                "min_bills_per_topic": args.min_bills,
                "min_legislators_per_topic": args.min_legislators,
                "min_votes_in_topic": args.min_votes_in_topic,
                "rhat_threshold": RHAT_THRESHOLD,
                "ess_threshold": ESS_THRESHOLD,
            },
        }

        for taxonomy, tax_results in all_taxonomy_results.items():
            manifest[taxonomy] = {}
            for ch_key, ch_data in tax_results.items():
                manifest[taxonomy][ch_key] = {
                    "n_topics_modeled": len(ch_data["topic_results"]),
                    "n_converged": ch_data["n_converged"],
                    "n_failed": ch_data["n_failed"],
                    "total_mcmc_time": ch_data["total_time"],
                }

        manifest_path = ctx.run_dir / "filtering_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  Saved: {manifest_path.name}")

        # ── HTML report ──
        print_header("HTML REPORT")
        build_issue_irt_report(
            ctx.report,
            all_taxonomy_results=all_taxonomy_results,
            plots_dir=ctx.plots_dir,
            session=args.session,
            taxonomies=taxonomies,
            args=vars(args),
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
