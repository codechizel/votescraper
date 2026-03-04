"""
Kansas Legislature — Text-Based Ideal Points (Embedding-Vote Approach)

Derives text-informed ideology scores from Phase 18 bill embeddings weighted
by voting behavior, then compares with IRT ideal points.

Not a true TBIP (Vafa et al. 2020) — Kansas bills are ~92% committee-sponsored,
making authorship-based models inapplicable. Instead uses vote-weighted embeddings
projected via PCA to extract a text-informed ideological dimension.

Usage:
  uv run python analysis/21_tbip/tbip.py
  uv run python analysis/21_tbip/tbip.py --session 2025-26 --irt-model both

Outputs (in results/<session>/<run_id>/21_tbip/):
  - data/:   Parquet files (matched legislators, outliers, correlations.json)
  - plots/:  PNG scatter plots, party distributions, PCA scree
  - filtering_manifest.json, run_info.json, run_log.txt
  - 21_tbip_report.html
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
    from analysis.phase_utils import print_header, save_fig
except ModuleNotFoundError:
    from phase_utils import print_header, save_fig  # type: ignore[no-redef]

try:
    from analysis.tbip_report import build_tbip_report
except ModuleNotFoundError:
    from tbip_report import build_tbip_report  # type: ignore[no-redef]

try:
    from analysis.tbip_data import (
        MIN_BILLS,
        MIN_MATCHED,
        OUTLIER_TOP_N,
        PARTY_COLORS,
        align_sign_convention,
        build_matched_df,
        build_vote_embedding_profiles,
        compute_correlations,
        compute_intra_party_correlations,
        compute_text_ideal_points,
        identify_outliers,
    )
except ModuleNotFoundError:
    from tbip_data import (  # type: ignore[no-redef]
        MIN_BILLS,
        MIN_MATCHED,
        OUTLIER_TOP_N,
        PARTY_COLORS,
        align_sign_convention,
        build_matched_df,
        build_vote_embedding_profiles,
        compute_correlations,
        compute_intra_party_correlations,
        compute_text_ideal_points,
        identify_outliers,
    )

try:
    from analysis.bill_text_data import (
        DEFAULT_EMBEDDING_MODEL,
        get_or_compute_embeddings,
        load_bill_texts,
        load_rollcalls,
        load_votes,
        preprocess_for_embedding,
    )
except ModuleNotFoundError:
    from bill_text_data import (  # type: ignore[no-redef]
        DEFAULT_EMBEDDING_MODEL,
        get_or_compute_embeddings,
        load_bill_texts,
        load_rollcalls,
        load_votes,
        preprocess_for_embedding,
    )

# ── Primer ───────────────────────────────────────────────────────────────────

TBIP_PRIMER = """\
# Text-Based Ideal Points (Embedding-Vote Approach)

## Purpose

Derive text-informed ideology scores by combining bill text embeddings with
voting data. This provides a complementary perspective to IRT ideal points —
while IRT uses only vote direction, text ideal points also consider what
legislation is about.

## Method

1. **Load Phase 18 bill embeddings** (384-dim bge-small-en-v1.5).
2. **Build vote matrix**: legislator × bill (+1 Yea, -1 Nay, 0 absent).
3. **Multiply**: text_profiles = vote_matrix @ embeddings (legislator × 384-dim).
4. **Normalize** each row by count of non-zero (non-absent) votes.
5. **PCA** on text profiles → PC1 score = text-derived ideal point.
6. **Align sign convention** with IRT (Republicans positive) via correlation check.
7. **Compare** with IRT xi_mean (Pearson r, Spearman ρ, Fisher z CI).

## Inputs

- Bill texts: `data/kansas/{session}/{prefix}_bill_texts.csv` (from `just text`)
- Roll calls: `data/kansas/{session}/{prefix}_rollcalls.csv`
- Votes: `data/kansas/{session}/{prefix}_votes.csv`
- IRT ideal points: `results/{session}/{run_id}/05_irt/data/ideal_points_{chamber}.parquet`
- Hierarchical ideal points: `results/{session}/{run_id}/07_hierarchical/data/...`

## Outputs

All outputs land in `results/{session}/{run_id}/21_tbip/`:

| File | Description |
|------|-------------|
| `data/matched_{model}_{chamber}.parquet` | Matched legislators with both scores |
| `data/outliers_{model}_{chamber}.parquet` | Top outliers by z-score discrepancy |
| `data/correlations.json` | All correlation results |
| `plots/scatter_{model}_{chamber}.png` | IRT vs text ideal points, party-colored |
| `plots/party_dist_{model}_{chamber}.png` | Party separation in text scores |
| `plots/pca_scree_{chamber}.png` | PCA explained variance |

## Interpretation Guide

- **Pearson r ≥ 0.80**: Strong — text captures ideology well.
- **0.65 ≤ r < 0.80**: Good — expected range for embedding-vote approach.
- **0.50 ≤ r < 0.65**: Moderate — captures partisan direction, less within-party.
- **r < 0.50**: Weak — text and votes may capture different dimensions.

## Caveats

- Not a true TBIP model — Kansas bills are ~92% committee-sponsored.
- Embedding quality depends on bill text extraction (PDF → text is noisy).
- PCA assumes linear structure — the primary ideological dimension is well-captured
  but secondary dimensions may be missed.
- Text profiles inherit all biases from the embedding model (bge-small-en-v1.5).
"""


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KS Legislature Text-Based Ideal Points (Embedding-Vote Approach)"
    )
    parser.add_argument("--session", default="2025-26")
    parser.add_argument(
        "--irt-model",
        choices=["flat", "hierarchical", "both"],
        default="both",
        help="Which IRT model to validate against",
    )
    parser.add_argument("--irt-dir", default=None, help="Override flat IRT results directory")
    parser.add_argument(
        "--hierarchical-dir", default=None, help="Override hierarchical IRT results directory"
    )
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    parser.add_argument(
        "--min-votes",
        type=int,
        default=20,
        help="Minimum non-absent votes to include a legislator (default: 20)",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"FastEmbed model name (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    return parser.parse_args()


# ── IRT Loading ──────────────────────────────────────────────────────────────


def _load_irt(
    results_root: Path,
    model: str,
    chamber: str,
    irt_dir_override: Path | None = None,
    hierarchical_dir_override: Path | None = None,
    run_id: str | None = None,
) -> pl.DataFrame | None:
    """Load IRT ideal points for a model/chamber combination."""
    if model == "flat":
        base = irt_dir_override or resolve_upstream_dir("05_irt", results_root, run_id)
        path = base / "data" / f"ideal_points_{chamber}.parquet"
    else:
        base = hierarchical_dir_override or resolve_upstream_dir(
            "07_hierarchical", results_root, run_id
        )
        path = base / "data" / f"hierarchical_ideal_points_{chamber}.parquet"

    if path.exists():
        df = pl.read_parquet(path)
        print(f"  {model.capitalize()} IRT ({chamber}): {df.height} legislators loaded")
        return df
    else:
        print(f"  {model.capitalize()} IRT ({chamber}): not found at {path}")
        return None


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_scatter(
    matched: pl.DataFrame,
    model: str,
    chamber: str,
    corr_result: dict,
    out_dir: Path,
) -> None:
    """Scatter plot: IRT xi_mean (x) vs text ideal point (y), party-colored."""
    if matched.height == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    for party in ["Republican", "Democrat", "Independent"]:
        if "party" not in matched.columns:
            continue
        sub = matched.filter(pl.col("party") == party)
        if sub.height == 0:
            continue

        color = PARTY_COLORS.get(party, "#888888")
        ax.scatter(
            sub["xi_mean"].to_numpy(),
            sub["text_score"].to_numpy(),
            c=color,
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label=f"{party} (n={sub.height})",
        )

    # Regression line
    xi = matched["xi_mean"].to_numpy().astype(float)
    text = matched["text_score"].to_numpy().astype(float)
    if len(xi) >= 2:
        z = np.polyfit(xi, text, 1)
        p = np.poly1d(z)
        x_line = np.linspace(xi.min(), xi.max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.3, linewidth=1)

    r = corr_result.get("pearson_r", float("nan"))
    rho = corr_result.get("spearman_rho", float("nan"))
    n = corr_result.get("n", 0)
    quality = corr_result.get("quality", "")

    ax.set_xlabel("IRT Ideal Point (xi_mean)", fontsize=12)
    ax.set_ylabel("Text-Derived Ideal Point (PC1)", fontsize=12)

    model_label = "Flat IRT" if model == "flat" else "Hierarchical IRT"
    ax.set_title(
        f"{chamber} — {model_label} vs Text Ideal Point\n"
        f"Pearson r = {r:.3f}, Spearman \u03c1 = {rho:.3f} (n = {n}, {quality})",
        fontsize=13,
        fontweight="bold",
    )

    ax.text(
        0.02,
        0.98,
        "Red = Republican, Blue = Democrat\n"
        "Dashed line = linear fit\n"
        "Text scores from vote-weighted bill embeddings",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.9},
    )

    ax.legend(fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    ch = chamber.lower()
    save_fig(fig, out_dir / f"scatter_{model}_{ch}.png")


def plot_party_distributions(
    matched: pl.DataFrame,
    model: str,
    chamber: str,
    out_dir: Path,
) -> None:
    """Strip plot showing party separation in text ideal points."""
    if matched.height == 0 or "party" not in matched.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    party_list = matched["party"].to_list()
    parties = [p for p in ["Republican", "Democrat", "Independent"] if p in party_list]

    for i, party in enumerate(parties):
        sub = matched.filter(pl.col("party") == party)
        if sub.height == 0:
            continue
        color = PARTY_COLORS.get(party, "#888888")
        scores = sub["text_score"].to_numpy()
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(scores))
        ax.scatter(
            scores,
            np.full_like(scores, i) + jitter,
            c=color,
            s=30,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label=f"{party} (n={sub.height})",
        )

    ax.set_yticks(range(len(parties)))
    ax.set_yticklabels(parties, fontsize=11)
    ax.set_xlabel("Text-Derived Ideal Point (PC1)", fontsize=12)

    model_label = "Flat IRT" if model == "flat" else "Hierarchical IRT"
    ax.set_title(
        f"{chamber} — Party Separation in Text Ideal Points ({model_label})",
        fontsize=13,
        fontweight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()

    ch = chamber.lower()
    save_fig(fig, out_dir / f"party_dist_{model}_{ch}.png")


def plot_pca_scree(
    var_ratios: np.ndarray,
    chamber: str,
    out_dir: Path,
) -> None:
    """Scree plot of PCA explained variance."""
    fig, ax = plt.subplots(figsize=(8, 5))

    n_components = len(var_ratios)
    x = range(1, n_components + 1)

    ax.bar(x, var_ratios * 100, color="#4A90D9", edgecolor="white", linewidth=0.5)
    ax.plot(x, np.cumsum(var_ratios) * 100, "ro-", markersize=5, linewidth=1.5, label="Cumulative")

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Explained Variance (%)", fontsize=12)
    ax.set_title(
        f"{chamber} — PCA on Legislator Text Profiles",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xticks(list(x))
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    ch = chamber.lower()
    save_fig(fig, out_dir / f"pca_scree_{ch}.png")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from tallgrass.session import KSSession

    session = KSSession.from_session_string(args.session)
    data_dir = Path(args.data_dir) if args.data_dir else session.data_dir
    results_root = session.results_dir

    # Check Phase 20 (bill text) output exists
    bt_dir = resolve_upstream_dir("20_bill_text", results_root, args.run_id)
    if not (bt_dir / "data").exists():
        print("[Phase 21] Skipping: bill text analysis not yet run (run `just text-analysis` first)")
        return

    # Determine which models to validate
    models = []
    if args.irt_model in ("flat", "both"):
        models.append("flat")
    if args.irt_model in ("hierarchical", "both"):
        models.append("hierarchical")

    with RunContext(
        session=args.session,
        analysis_name="21_tbip",
        params=vars(args),
        results_root=results_root,
        primer=TBIP_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"KS Legislature Text-Based Ideal Points — Session {args.session}")
        print(f"Models:    {', '.join(models)}")
        print(f"Min votes: {args.min_votes}")
        print(f"Output:    {ctx.run_dir}")

        # ── Load data ──
        print_header("DATA LOADING")

        bill_texts = load_bill_texts(data_dir)
        print(f"  Bill texts: {bill_texts.height} bills")

        rollcalls = load_rollcalls(data_dir)
        print(f"  Roll calls: {rollcalls.height}")

        votes = load_votes(data_dir)
        print(f"  Votes: {votes.height}")

        # ── Compute embeddings ──
        print_header("EMBEDDINGS")

        texts = [preprocess_for_embedding(t) for t in bill_texts["text"].to_list()]
        bill_nums = bill_texts["bill_number"].to_list()

        cache_dir = data_dir / ".cache" / "text"
        embeddings = get_or_compute_embeddings(
            texts, bill_nums, cache_dir, model_name=args.embedding_model
        )
        print(f"  Embeddings shape: {embeddings.shape}")

        # ── Per-chamber analysis ──
        all_results: dict[str, dict] = {}
        irt_dir = Path(args.irt_dir) if args.irt_dir else None
        hier_dir = Path(args.hierarchical_dir) if args.hierarchical_dir else None

        for chamber in ["House", "Senate"]:
            print_header(f"CHAMBER: {chamber}")
            ch = chamber.lower()

            try:
                profiles, slugs, n_bills_matched = build_vote_embedding_profiles(
                    votes,
                    rollcalls,
                    embeddings,
                    bill_nums,
                    chamber=chamber,
                    min_votes=args.min_votes,
                )
            except ValueError as e:
                print(f"  Skipping {chamber}: {e}")
                continue

            print(f"  Bills matched: {n_bills_matched}")
            print(f"  Legislators: {len(slugs)}")
            print(f"  Profile shape: {profiles.shape}")

            # PCA
            pc1_scores, pc1_var, all_var_ratios = compute_text_ideal_points(profiles)
            print(f"  PC1 explains: {pc1_var:.1%} of variance")

            # Scree plot (once per chamber)
            plot_pca_scree(all_var_ratios, chamber, ctx.plots_dir)

            for model in models:
                print(f"\n  --- {model.capitalize()} IRT ---")

                irt_df = _load_irt(results_root, model, ch, irt_dir, hier_dir, run_id=args.run_id)
                if irt_df is None:
                    continue

                # Align sign convention
                aligned_scores = align_sign_convention(pc1_scores, slugs, irt_df)

                # Match
                matched = build_matched_df(aligned_scores, slugs, irt_df)
                print(f"  Matched: {matched.height} legislators")

                if matched.height < MIN_MATCHED:
                    print(f"  Only {matched.height} matches — skipping")
                    continue

                # Correlations
                corr = compute_correlations(matched)
                intra = compute_intra_party_correlations(matched)
                outliers = identify_outliers(matched)

                print(
                    f"  r = {corr['pearson_r']:.3f}, "
                    f"\u03c1 = {corr['spearman_rho']:.3f}, "
                    f"n = {corr['n']}, "
                    f"quality = {corr['quality']}"
                )

                key = f"{model}_{ch}"
                all_results[key] = {
                    "model": model,
                    "chamber": chamber,
                    "matched": matched,
                    "correlations": corr,
                    "intra_party": intra,
                    "outliers": outliers,
                    "n_bills_matched": n_bills_matched,
                    "n_legislators": len(slugs),
                    "pc1_variance_ratio": pc1_var,
                }

                # Save parquets
                matched.write_parquet(ctx.data_dir / f"matched_{model}_{ch}.parquet")
                if outliers.height > 0:
                    outliers.write_parquet(ctx.data_dir / f"outliers_{model}_{ch}.parquet")

                # Plots
                plot_scatter(matched, model, chamber, corr, ctx.plots_dir)
                plot_party_distributions(matched, model, chamber, ctx.plots_dir)

        # ── Save correlations JSON ──
        print_header("CORRELATIONS SUMMARY")
        corr_summary: dict = {}
        for key, data in all_results.items():
            corr_summary[key] = data["correlations"]

        corr_path = ctx.data_dir / "correlations.json"
        with open(corr_path, "w") as f:
            json.dump(corr_summary, f, indent=2, default=str)
        print(f"  Saved: {corr_path.name}")

        # ── Filtering manifest ──
        print_header("FILTERING MANIFEST")
        manifest: dict = {
            "analysis": "tbip",
            "session": args.session,
            "models": models,
            "embedding_model": args.embedding_model,
            "min_votes": args.min_votes,
            "constants": {
                "MIN_MATCHED": MIN_MATCHED,
                "MIN_BILLS": MIN_BILLS,
                "OUTLIER_TOP_N": OUTLIER_TOP_N,
            },
        }

        for key, data in all_results.items():
            manifest[key] = {
                "n_bills_matched": data["n_bills_matched"],
                "n_legislators": data["n_legislators"],
                "n_matched": data["matched"].height,
                "pc1_variance_ratio": data["pc1_variance_ratio"],
                "pearson_r": data["correlations"]["pearson_r"],
                "spearman_rho": data["correlations"]["spearman_rho"],
                "quality": data["correlations"]["quality"],
            }

        manifest_path = ctx.run_dir / "filtering_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  Saved: {manifest_path.name}")

        # ── HTML report ──
        print_header("HTML REPORT")
        build_tbip_report(
            ctx.report,
            all_results=all_results,
            plots_dir=ctx.plots_dir,
            session=args.session,
            models=models,
            embedding_model=args.embedding_model,
            min_votes=args.min_votes,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
