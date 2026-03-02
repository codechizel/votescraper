"""
Kansas Legislature — External Validation Against DIME/CFscores

Compares our IRT ideal points to DIME campaign-finance ideology scores
(Stanford, ODC-BY). Provides a second, independent external validation —
CFscores measure who *funds* a legislator, not how they *vote*.

Overlap: 84th-89th bienniums (2011-2022), extending one biennium beyond
Shor-McCarty coverage.

Usage:
  uv run python analysis/14b_external_validation_dime/external_validation_dime.py
  uv run python analysis/14b_external_validation_dime/external_validation_dime.py \
    --all-sessions --irt-model both

Outputs (in results/<session>/<run_id>/14b_external_validation_dime/):
  - data/:   Parquet files (matched legislators, correlations, outliers)
  - plots/:  PNG scatter plots (our xi_mean vs DIME CFscore)
  - filtering_manifest.json, run_info.json, run_log.txt
  - external_validation_dime_report.html
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
    from analysis.external_validation_dime_report import build_dime_report
except ModuleNotFoundError:
    from external_validation_dime_report import (
        build_dime_report,  # type: ignore[no-redef]
    )

try:
    from analysis.external_validation_dime_data import (
        CONCERN_CORRELATION,
        DIME_CACHE_PATH,
        DIME_OVERLAPPING_BIENNIUMS,
        GOOD_CORRELATION,
        MIN_GIVERS,
        MIN_MATCHED,
        OUTLIER_TOP_N,
        STRONG_CORRELATION,
        compute_correlations,
        compute_intra_party_correlations,
        filter_dime_to_biennium,
        has_dime_overlap,
        identify_outliers,
        match_dime_legislators,
        parse_dime_kansas,
    )
except ModuleNotFoundError:
    from external_validation_dime_data import (  # type: ignore[no-redef]
        CONCERN_CORRELATION,
        DIME_CACHE_PATH,
        DIME_OVERLAPPING_BIENNIUMS,
        GOOD_CORRELATION,
        MIN_GIVERS,
        MIN_MATCHED,
        OUTLIER_TOP_N,
        STRONG_CORRELATION,
        compute_correlations,
        compute_intra_party_correlations,
        filter_dime_to_biennium,
        has_dime_overlap,
        identify_outliers,
        match_dime_legislators,
        parse_dime_kansas,
    )

# ── Primer ───────────────────────────────────────────────────────────────────

DIME_PRIMER = """\
# External Validation Against DIME/CFscores

## Purpose

This is the pipeline's second external validation source. Phase 14 validated
against Shor-McCarty (roll-call-based ideology). This phase validates against
DIME/CFscores (campaign-finance-based ideology) — a completely independent
construct. High correlation between our IRT scores and CFscores means the money
agrees with the votes.

## Method

1. **Load** the pre-downloaded DIME CSV from `data/external/`.
2. **Filter** to Kansas state legislators, incumbents, with sufficient donors.
3. **Match** our legislators to DIME candidates by normalized name (two-phase:
   exact match, then last-name fallback).
4. **Correlate** our IRT ideal points (xi_mean) with DIME's static CFscores
   (recipient.cfscore) and dynamic CFscores (recipient.cfscore.dyn).
5. **Identify outliers** — legislators whose scores disagree between datasets.

## Inputs

- DIME CSV: `data/external/dime_recipients_1979_2024.csv`
- IRT ideal points: `results/<session>/<run_id>/04_irt/data/ideal_points_{chamber}.parquet`
- Hierarchical ideal points:
  `results/<session>/<run_id>/10_hierarchical/data/hierarchical_ideal_points_{chamber}.parquet`

## Outputs

All outputs land in `results/<session>/<run_id>/14b_external_validation_dime/`:

### `data/` — Parquet intermediates

| File | Description |
|------|-------------|
| `matched_{model}_{chamber}.parquet` | Matched legislators with both scores |
| `outliers_{model}_{chamber}.parquet` | Top outliers by z-score discrepancy |
| `correlations.json` | All correlation results |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `scatter_dime_{model}_{chamber}_{session}.png` | Our xi_mean vs DIME CFscore, party-colored |

## Interpretation Guide

- **Pearson r ≈ 0.75-0.90**: Expected for state-level CFscore comparisons.
- **Pearson r > 0.90**: Exceptional — donor ideology closely tracks voting.
- **Pearson r < 0.70**: Investigate — may reflect data quality issues.
- **Intra-party r ≈ 0.50-0.70**: Expected — CFscores discriminate poorly within parties.

## Caveats

- CFscores measure donor ideology, not voting ideology. These are correlated
  but distinct constructs.
- Within-party discrimination is limited due to access-motivated donations.
- DIME coverage ends at 2022 cycle. 90th-91st bienniums cannot be validated.
- Legislators with fewer than 5 donors are excluded (unreliable CFscores).
"""

# ── Constants ────────────────────────────────────────────────────────────────

PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KS Legislature External Validation (DIME/CFscores)"
    )
    parser.add_argument("--session", default="2019-20")
    parser.add_argument(
        "--all-sessions", action="store_true", help="Run all 6 overlapping bienniums + pooled"
    )
    parser.add_argument(
        "--irt-model",
        choices=["flat", "hierarchical", "both"],
        default="both",
        help="Which IRT model to validate",
    )
    parser.add_argument("--irt-dir", default=None, help="Override flat IRT results directory")
    parser.add_argument(
        "--hierarchical-dir", default=None, help="Override hierarchical IRT results directory"
    )
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--min-givers",
        type=int,
        default=MIN_GIVERS,
        help=f"Minimum unique donors for reliable CFscore (default: {MIN_GIVERS})",
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
        base = irt_dir_override or resolve_upstream_dir("04_irt", results_root, run_id)
        path = base / "data" / f"ideal_points_{chamber}.parquet"
    else:
        base = hierarchical_dir_override or resolve_upstream_dir(
            "10_hierarchical", results_root, run_id
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
    session: str,
    corr_result: dict,
    out_dir: Path,
) -> None:
    """Scatter plot: our xi_mean (x) vs DIME CFscore (y), party-colored."""
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
            sub["recipient_cfscore"].to_numpy(),
            c=color,
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label=f"{party} (n={sub.height})",
        )

    # Regression line
    xi = matched["xi_mean"].to_numpy().astype(float)
    cf_scores = matched["recipient_cfscore"].to_numpy().astype(float)
    if len(xi) >= 2:
        z = np.polyfit(xi, cf_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(xi.min(), xi.max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.3, linewidth=1)

    r = corr_result.get("pearson_r", float("nan"))
    rho = corr_result.get("spearman_rho", float("nan"))
    n = corr_result.get("n", 0)
    quality = corr_result.get("quality", "")

    ax.set_xlabel("Our IRT Ideal Point (xi_mean)", fontsize=12)
    ax.set_ylabel("DIME CFscore (recipient.cfscore)", fontsize=12)

    model_label = "Flat IRT" if model == "flat" else "Hierarchical IRT"
    ax.set_title(
        f"{chamber} — {model_label} vs DIME CFscore ({session})\n"
        f"Pearson r = {r:.3f}, Spearman \u03c1 = {rho:.3f} (n = {n}, {quality})",
        fontsize=13,
        fontweight="bold",
    )

    ax.text(
        0.02,
        0.98,
        "Red = Republican, Blue = Democrat\n"
        "Dashed line = linear fit\n"
        "CFscores measure donor ideology, not voting",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.9},
    )

    ax.legend(fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"scatter_dime_{model}_{chamber.lower()}_{session}.png")


# ── SM Comparison Loading ────────────────────────────────────────────────────


def _load_sm_correlations(results_root: Path, run_id: str | None) -> dict | None:
    """Try to load Shor-McCarty correlations from Phase 14 for side-by-side comparison."""
    try:
        sm_dir = resolve_upstream_dir("14_external_validation", results_root, run_id)
        corr_path = sm_dir / "data" / "correlations.json"
        if corr_path.exists():
            with open(corr_path) as f:
                return json.load(f)
    except Exception:
        pass
    return None


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from tallgrass.session import KSSession

    # Determine which sessions to run
    if args.all_sessions:
        sessions = list(DIME_OVERLAPPING_BIENNIUMS.keys())
        primary_session = "2019-20"  # for RunContext
    else:
        ks = KSSession.from_session_string(args.session)
        if not has_dime_overlap(ks.output_name):
            print(f"Session {ks.output_name} does not overlap with DIME coverage.")
            print(f"Overlapping sessions: {', '.join(DIME_OVERLAPPING_BIENNIUMS.keys())}")
            return
        sessions = [ks.output_name]
        primary_session = args.session

    # Determine which models to validate
    models = []
    if args.irt_model in ("flat", "both"):
        models.append("flat")
    if args.irt_model in ("hierarchical", "both"):
        models.append("hierarchical")

    with RunContext(
        session=primary_session,
        analysis_name="14b_external_validation_dime",
        params=vars(args),
        primer=DIME_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"KS Legislature External Validation (DIME/CFscores) — Session {primary_session}")
        print(f"Sessions:  {', '.join(sessions)}")
        print(f"Models:    {', '.join(models)}")
        print(f"Min donors: {args.min_givers}")
        print(f"Output:    {ctx.run_dir}")

        # ── Load DIME data ──
        print_header("DIME DATA")
        cache_path = Path(DIME_CACHE_PATH)
        if not cache_path.exists():
            print(f"  DIME CSV not found at: {cache_path}")
            print("  Download from: https://data.stanford.edu/dime")
            print(f"  Save to: {cache_path}")
            return

        dime_df = parse_dime_kansas(str(cache_path))
        print(f"  Kansas state legislators in DIME: {dime_df.height}")

        # ── Process each session ──
        all_results: dict[str, dict] = {}
        all_matched: list[pl.DataFrame] = []
        all_unmatched: list[pl.DataFrame] = []

        for session_name in sessions:
            print_header(f"SESSION: {session_name}")

            cycles = DIME_OVERLAPPING_BIENNIUMS[session_name]
            house_dime, senate_dime = filter_dime_to_biennium(
                dime_df, cycles, min_givers=args.min_givers
            )
            print(f"  DIME House: {house_dime.height}, DIME Senate: {senate_dime.height}")

            # Derive session string from session name
            # "84th_2011-2012" -> "2011-12"
            parts = session_name.split("_")[1] if "_" in session_name else session_name
            start_year = int(parts.split("-")[0])
            end_year_str = parts.split("-")[1]
            ks = KSSession.from_session_string(f"{start_year}-{end_year_str[-2:]}")
            results_root = ks.results_dir

            irt_dir = Path(args.irt_dir) if args.irt_dir else None
            hier_dir = Path(args.hierarchical_dir) if args.hierarchical_dir else None

            session_results: dict[str, dict] = {}

            for model in models:
                for chamber, dime_chamber_df in [("House", house_dime), ("Senate", senate_dime)]:
                    if dime_chamber_df.height == 0:
                        print(f"  {model}/{chamber}: No DIME data — skipping")
                        continue

                    ch = chamber.lower()
                    irt_df = _load_irt(
                        results_root, model, ch, irt_dir, hier_dir, run_id=args.run_id
                    )
                    if irt_df is None:
                        continue

                    # Match
                    matched, unmatched = match_dime_legislators(irt_df, dime_chamber_df, chamber)
                    n_unmatched_ours = unmatched.filter(pl.col("source") == "our_data").height
                    n_unmatched_dime = unmatched.filter(pl.col("source") == "dime").height
                    print(
                        f"  {model}/{chamber}: "
                        f"{matched.height} matched, "
                        f"{n_unmatched_ours} unmatched (ours), "
                        f"{n_unmatched_dime} unmatched (DIME)"
                    )

                    if matched.height < MIN_MATCHED:
                        print(f"  {model}/{chamber}: Only {matched.height} matches — skipping")
                        continue

                    # Static CFscore correlations
                    corr = compute_correlations(
                        matched, xi_col="xi_mean", np_col="recipient_cfscore"
                    )
                    intra = compute_intra_party_correlations(
                        matched, xi_col="xi_mean", np_col="recipient_cfscore"
                    )
                    outliers = identify_outliers(
                        matched, xi_col="xi_mean", np_col="recipient_cfscore"
                    )

                    # Dynamic CFscore correlations (if available)
                    dyn_corr = None
                    if "recipient_cfscore_dyn" in matched.columns:
                        valid_dyn = matched.filter(pl.col("recipient_cfscore_dyn").is_not_null())
                        if valid_dyn.height >= MIN_MATCHED:
                            dyn_corr = compute_correlations(
                                valid_dyn,
                                xi_col="xi_mean",
                                np_col="recipient_cfscore_dyn",
                            )

                    key = f"{session_name}_{model}_{ch}"
                    session_results[key] = {
                        "session": session_name,
                        "model": model,
                        "chamber": chamber,
                        "matched": matched,
                        "unmatched": unmatched,
                        "correlations": corr,
                        "dynamic_correlations": dyn_corr,
                        "intra_party": intra,
                        "outliers": outliers,
                        "match_rate_ours": (
                            matched.height / irt_df.height if irt_df.height > 0 else 0
                        ),
                        "n_ours": irt_df.height,
                        "n_dime": dime_chamber_df.height,
                    }

                    # Print summary
                    print(
                        f"    Static:  r = {corr['pearson_r']:.3f}, "
                        f"\u03c1 = {corr['spearman_rho']:.3f}, "
                        f"n = {corr['n']}, "
                        f"quality = {corr['quality']}"
                    )
                    if dyn_corr and dyn_corr.get("quality") != "insufficient_data":
                        print(
                            f"    Dynamic: r = {dyn_corr['pearson_r']:.3f}, "
                            f"\u03c1 = {dyn_corr['spearman_rho']:.3f}"
                        )

                    # Save parquets
                    parq_name = f"matched_{model}_{ch}_{session_name}.parquet"
                    matched.write_parquet(ctx.data_dir / parq_name)
                    if outliers.height > 0:
                        outliers.write_parquet(
                            ctx.data_dir / f"outliers_{model}_{ch}_{session_name}.parquet"
                        )

                    # Plot
                    plot_scatter(matched, model, chamber, session_name, corr, ctx.plots_dir)

                    all_matched.append(
                        matched.with_columns(
                            pl.lit(session_name).alias("session"),
                            pl.lit(model).alias("model"),
                            pl.lit(chamber).alias("chamber"),
                        )
                    )
                    all_unmatched.append(unmatched)

            all_results.update(session_results)

        # ── Pooled analysis (if --all-sessions) ──
        pooled_results: dict[str, dict] = {}
        if args.all_sessions and all_matched:
            print_header("POOLED ANALYSIS")
            pooled_df = pl.concat(all_matched, how="diagonal")

            for model in models:
                for chamber in ["House", "Senate"]:
                    ch = chamber.lower()
                    pool = (
                        pooled_df.filter(
                            (pl.col("model") == model) & (pl.col("chamber") == chamber)
                        )
                        if "chamber" in pooled_df.columns
                        else pooled_df.filter(pl.col("model") == model)
                    )

                    if pool.height < MIN_MATCHED:
                        continue

                    # Deduplicate by legislator (use most recent session's scores)
                    pool_deduped = pool.sort("session", descending=True).unique(
                        subset=["normalized_name"], keep="first"
                    )

                    corr = compute_correlations(
                        pool_deduped, xi_col="xi_mean", np_col="recipient_cfscore"
                    )
                    key = f"pooled_{model}_{ch}"
                    pooled_results[key] = {
                        "session": "pooled",
                        "model": model,
                        "chamber": chamber,
                        "matched": pool_deduped,
                        "correlations": corr,
                        "n": pool_deduped.height,
                    }

                    print(
                        f"  Pooled {model}/{chamber}: "
                        f"r = {corr['pearson_r']:.3f}, "
                        f"\u03c1 = {corr['spearman_rho']:.3f}, "
                        f"n = {corr['n']}"
                    )

                    # Pooled scatter
                    plot_scatter(pool_deduped, model, chamber, "pooled", corr, ctx.plots_dir)

        # ── SM comparison (if available) ──
        sm_comparison: dict[str, dict] | None = None
        if all_results:
            # Try to load SM correlations for overlapping bienniums (84th-88th)
            first_session = sessions[0]
            parts = first_session.split("_")[1] if "_" in first_session else first_session
            start_year = int(parts.split("-")[0])
            end_year_str = parts.split("-")[1]
            ks = KSSession.from_session_string(f"{start_year}-{end_year_str[-2:]}")
            sm_corrs = _load_sm_correlations(ks.results_dir, args.run_id)

            if sm_corrs:
                sm_comparison = {}
                for key, data in all_results.items():
                    session = data["session"]
                    model = data["model"]
                    chamber = data["chamber"]
                    ch = chamber.lower()
                    sm_key = f"{session}_{model}_{ch}"
                    if sm_key in sm_corrs:
                        sm_corr = sm_corrs[sm_key]
                        sm_comparison[key] = {
                            "session": session,
                            "model": model,
                            "chamber": chamber,
                            "sm_pearson_r": sm_corr.get("pearson_r", float("nan")),
                            "sm_spearman_rho": sm_corr.get("spearman_rho", float("nan")),
                            "sm_n": sm_corr.get("n", 0),
                            "dime_pearson_r": data["correlations"]["pearson_r"],
                            "dime_spearman_rho": data["correlations"]["spearman_rho"],
                            "dime_n": data["correlations"]["n"],
                        }

        # ── Save correlations JSON ──
        print_header("CORRELATIONS SUMMARY")
        corr_summary: dict = {}
        for key, data in all_results.items():
            corr_summary[key] = data["correlations"]
            if data.get("dynamic_correlations"):
                corr_summary[f"{key}_dynamic"] = data["dynamic_correlations"]
        for key, data in pooled_results.items():
            corr_summary[key] = data["correlations"]

        corr_path = ctx.data_dir / "correlations.json"
        with open(corr_path, "w") as f:
            json.dump(corr_summary, f, indent=2, default=str)
        print(f"  Saved: {corr_path.name}")

        # ── Filtering manifest ──
        print_header("FILTERING MANIFEST")
        manifest: dict = {
            "analysis": "external_validation_dime",
            "sessions": sessions,
            "models": models,
            "constants": {
                "MIN_MATCHED": MIN_MATCHED,
                "MIN_GIVERS": args.min_givers,
                "STRONG_CORRELATION": STRONG_CORRELATION,
                "GOOD_CORRELATION": GOOD_CORRELATION,
                "CONCERN_CORRELATION": CONCERN_CORRELATION,
                "OUTLIER_TOP_N": OUTLIER_TOP_N,
            },
            "dime_kansas_records": dime_df.height,
        }

        for key, data in all_results.items():
            manifest[key] = {
                "n_matched": data["matched"].height,
                "n_ours": data["n_ours"],
                "n_dime": data["n_dime"],
                "match_rate": data["match_rate_ours"],
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
        build_dime_report(
            ctx.report,
            all_results=all_results,
            pooled_results=pooled_results,
            sm_comparison=sm_comparison,
            dime_total=dime_df.height,
            sessions=sessions,
            models=models,
            plots_dir=ctx.plots_dir,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
