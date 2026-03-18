"""
Kansas Legislature — External Validation Against Shor-McCarty Ideology Scores

Compares our IRT ideal points to the field-standard Shor-McCarty dataset
(Harvard Dataverse, CC0). Provides the first external validation of our
pipeline — every prior validation has been internal.

Overlap: 84th-88th bienniums (2011-2020), ~610 Kansas legislators.

Usage:
  uv run python analysis/external_validation.py [--session 2019-20]
  uv run python analysis/external_validation.py --all-sessions --irt-model both

Outputs (in results/<session>/external_validation/<date>/):
  - data/:   Parquet files (matched legislators, correlations, outliers)
  - plots/:  PNG scatter plots (our xi_mean vs SM np_score)
  - filtering_manifest.json, run_info.json, run_log.txt
  - external_validation_report.html
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
    from analysis.tuning import PARTY_COLORS
except ModuleNotFoundError:
    from tuning import PARTY_COLORS  # type: ignore[no-redef]

try:
    from analysis.external_validation_report import build_external_validation_report
except ModuleNotFoundError:
    from external_validation_report import (
        build_external_validation_report,  # type: ignore[no-redef]
    )

try:
    from analysis.external_validation_data import (
        CONCERN_CORRELATION,
        GOOD_CORRELATION,
        MIN_MATCHED,
        OUTLIER_TOP_N,
        OVERLAPPING_BIENNIUMS,
        SHOR_MCCARTY_CACHE_PATH,
        SHOR_MCCARTY_URL,
        STRONG_CORRELATION,
        compute_correlations,
        compute_intra_party_correlations,
        filter_to_biennium,
        has_shor_mccarty_overlap,
        identify_outliers,
        match_legislators,
        parse_shor_mccarty,
    )
except ModuleNotFoundError:
    from external_validation_data import (  # type: ignore[no-redef]
        CONCERN_CORRELATION,
        GOOD_CORRELATION,
        MIN_MATCHED,
        OUTLIER_TOP_N,
        OVERLAPPING_BIENNIUMS,
        SHOR_MCCARTY_CACHE_PATH,
        SHOR_MCCARTY_URL,
        STRONG_CORRELATION,
        compute_correlations,
        compute_intra_party_correlations,
        filter_to_biennium,
        has_shor_mccarty_overlap,
        identify_outliers,
        match_legislators,
        parse_shor_mccarty,
    )

# ── Primer ───────────────────────────────────────────────────────────────────

EXTERNAL_VALIDATION_PRIMER = """\
# External Validation Against Shor-McCarty Ideology Scores

## Purpose

This is the pipeline's first external validation. Every prior validation has been
internal — IRT correlates with PCA, holdout accuracy is high, cross-session scores
are stable. But none of those prove the scores measure what political scientists
mean by "ideology." Comparing our IRT ideal points to the Shor-McCarty dataset —
the field standard for state legislator ideology — closes this gap.

## Method

1. **Download** the Shor-McCarty dataset from Harvard Dataverse (CC0 license).
2. **Filter** to Kansas legislators active during each biennium.
3. **Match** our legislators to SM legislators by normalized name (two-phase:
   exact match, then last-name + district tiebreaker).
4. **Correlate** our session-specific IRT ideal points (xi_mean) with SM's
   career-level ideology scores (np_score) using Pearson r and Spearman rho.
5. **Identify outliers** — legislators whose scores disagree between datasets.

## Inputs

- Shor-McCarty tab file: `data/external/shor_mccarty.tab`
- IRT ideal points: `results/<session>/irt/latest/data/ideal_points_{chamber}.parquet`
- Hierarchical ideal points:
  `results/<session>/hierarchical/latest/data/hierarchical_ideal_points_{chamber}.parquet`

## Outputs

All outputs land in `results/<session>/external_validation/<date>/`:

### `data/` — Parquet intermediates

| File | Description |
|------|-------------|
| `matched_{model}_{chamber}.parquet` | Matched legislators with both scores |
| `outliers_{model}_{chamber}.parquet` | Top outliers by z-score discrepancy |
| `correlations.json` | All correlation results |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `scatter_{model}_{chamber}.png` | Our xi_mean vs SM np_score, party-colored |

## Interpretation Guide

- **Pearson r > 0.90**: Strong external validation. Our scores agree with the field standard.
- **Pearson r = 0.85-0.90**: Good agreement. Differences likely reflect session-specific dynamics.
- **Pearson r < 0.85**: Investigate — could be data quality, convergence failures, or methodology.
- **Spearman rho**: Tests rank-order agreement (robust to scale differences).
- **Intra-party correlations** are lower because within-party variation is smaller.

## Caveats

- SM `np_score` is career-fixed; our `xi_mean` varies by biennium. The comparison
  tests rank ordering, not scale equivalence.
- Name matching is imperfect. Unmatched legislators are reported.
- 84th biennium has known convergence issues (ODT data, 30% missing).
"""

# ── Constants ────────────────────────────────────────────────────────────────


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KS Legislature External Validation (Shor-McCarty)"
    )
    parser.add_argument("--session", default="2019-20")
    parser.add_argument(
        "--all-sessions", action="store_true", help="Run all 5 overlapping bienniums + pooled"
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
    return parser.parse_args()


# ── Download / Cache ─────────────────────────────────────────────────────────


def download_shor_mccarty(cache_path: Path) -> str | None:
    """Download SM data from Harvard Dataverse, caching to disk.

    Returns the raw text content, or None on failure.
    """
    if cache_path.exists():
        print(f"  Using cached: {cache_path}")
        return cache_path.read_text(encoding="utf-8")

    print("  Downloading from Harvard Dataverse...")
    print(f"  URL: {SHOR_MCCARTY_URL}")

    try:
        import requests

        resp = requests.get(SHOR_MCCARTY_URL, timeout=60, allow_redirects=True)
        resp.raise_for_status()

        # Check for HTML error page (same pattern as scraper)
        if resp.content[:5].lower().startswith(b"<html"):
            print("  ERROR: Server returned HTML error page instead of data file.")
            print("  Manual download instructions:")
            print(
                "    1. Visit: https://dataverse.harvard.edu/dataset.xhtml?"
                "persistentId=doi:10.7910/DVN/NWSYOS"
            )
            print("    2. Download the .tab file")
            print(f"    3. Save to: {cache_path}")
            return None

        text = resp.text

        # Cache to disk
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        print(f"  Cached to: {cache_path}")

        return text

    except Exception as e:
        print(f"  ERROR downloading: {e}")
        print("  Manual download instructions:")
        print(
            "    1. Visit: https://dataverse.harvard.edu/dataset.xhtml?"
            "persistentId=doi:10.7910/DVN/NWSYOS"
        )
        print("    2. Download the .tab file")
        print(f"    3. Save to: {cache_path}")
        return None


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
    session: str,
    corr_result: dict,
    out_dir: Path,
) -> None:
    """Scatter plot: our xi_mean (x) vs SM np_score (y), party-colored."""
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
            sub["np_score"].to_numpy(),
            c=color,
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label=f"{party} (n={sub.height})",
        )

    # Regression line
    xi = matched["xi_mean"].to_numpy().astype(float)
    np_scores = matched["np_score"].to_numpy().astype(float)
    if len(xi) >= 2:
        z = np.polyfit(xi, np_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(xi.min(), xi.max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.3, linewidth=1)

    r = corr_result.get("pearson_r", float("nan"))
    rho = corr_result.get("spearman_rho", float("nan"))
    n = corr_result.get("n", 0)
    quality = corr_result.get("quality", "")

    ax.set_xlabel("Our IRT Ideal Point (xi_mean)", fontsize=12)
    ax.set_ylabel("Shor-McCarty Score (np_score)", fontsize=12)

    model_label = "Flat IRT" if model == "flat" else "Hierarchical IRT"
    ax.set_title(
        f"{chamber} — {model_label} vs Shor-McCarty ({session})\n"
        f"Pearson r = {r:.3f}, Spearman ρ = {rho:.3f} (n = {n}, {quality})",
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
    save_fig(fig, out_dir / f"scatter_{model}_{chamber.lower()}.png")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from tallgrass.session import KSSession

    # Determine which sessions to run
    if args.all_sessions:
        sessions = list(OVERLAPPING_BIENNIUMS.keys())
        primary_session = "2019-20"  # for RunContext
    else:
        ks = KSSession.from_session_string(args.session)
        if not has_shor_mccarty_overlap(ks.output_name):
            print(f"Session {ks.output_name} does not overlap with Shor-McCarty coverage.")
            print(f"Overlapping sessions: {', '.join(OVERLAPPING_BIENNIUMS.keys())}")
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
        analysis_name="17_external_validation",
        params=vars(args),
        primer=EXTERNAL_VALIDATION_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"KS Legislature External Validation (Shor-McCarty) — Session {primary_session}")
        print(f"Sessions:  {', '.join(sessions)}")
        print(f"Models:    {', '.join(models)}")
        print(f"Output:    {ctx.run_dir}")

        # ── Download / load SM data ──
        print_header("SHOR-MCCARTY DATA")
        cache_path = Path(SHOR_MCCARTY_CACHE_PATH)
        raw_text = download_shor_mccarty(cache_path)
        if raw_text is None:
            print("  Cannot proceed without Shor-McCarty data. Exiting.")
            return

        sm_df = parse_shor_mccarty(raw_text)
        print(f"  Kansas legislators in SM data: {sm_df.height}")

        # ── Process each session ──
        all_results: dict[str, dict] = {}
        all_matched: list[pl.DataFrame] = []
        all_unmatched: list[pl.DataFrame] = []

        for session_name in sessions:
            print_header(f"SESSION: {session_name}")

            start_year, end_year = OVERLAPPING_BIENNIUMS[session_name]
            house_sm, senate_sm = filter_to_biennium(sm_df, start_year, end_year)
            print(f"  SM House: {house_sm.height}, SM Senate: {senate_sm.height}")

            ks = KSSession.from_session_string(f"{start_year}-{str(end_year)[-2:]}")
            results_root = ks.results_dir

            irt_dir = Path(args.irt_dir) if args.irt_dir else None
            hier_dir = Path(args.hierarchical_dir) if args.hierarchical_dir else None

            session_results: dict[str, dict] = {}

            for model in models:
                for chamber, sm_chamber_df in [("House", house_sm), ("Senate", senate_sm)]:
                    if sm_chamber_df.height == 0:
                        print(f"  {model}/{chamber}: No SM data — skipping")
                        continue

                    ch = chamber.lower()
                    irt_df = _load_irt(
                        results_root, model, ch, irt_dir, hier_dir, run_id=args.run_id
                    )
                    if irt_df is None:
                        continue

                    # Match
                    matched, unmatched = match_legislators(
                        irt_df, sm_chamber_df, chamber, start_year=start_year
                    )
                    n_unmatched_ours = unmatched.filter(pl.col("source") == "our_data").height
                    n_unmatched_sm = unmatched.filter(pl.col("source") == "shor_mccarty").height
                    print(
                        f"  {model}/{chamber}: "
                        f"{matched.height} matched, "
                        f"{n_unmatched_ours} unmatched (ours), "
                        f"{n_unmatched_sm} unmatched (SM)"
                    )

                    if matched.height < MIN_MATCHED:
                        print(f"  {model}/{chamber}: Only {matched.height} matches — skipping")
                        continue

                    # Correlations
                    corr = compute_correlations(matched)
                    intra = compute_intra_party_correlations(matched)
                    outliers = identify_outliers(matched)

                    key = f"{session_name}_{model}_{ch}"
                    session_results[key] = {
                        "session": session_name,
                        "model": model,
                        "chamber": chamber,
                        "matched": matched,
                        "unmatched": unmatched,
                        "correlations": corr,
                        "intra_party": intra,
                        "outliers": outliers,
                        "match_rate_ours": (
                            matched.height / irt_df.height if irt_df.height > 0 else 0
                        ),
                        "n_ours": irt_df.height,
                        "n_sm": sm_chamber_df.height,
                    }

                    # Print summary
                    print(
                        f"    r = {corr['pearson_r']:.3f}, "
                        f"ρ = {corr['spearman_rho']:.3f}, "
                        f"n = {corr['n']}, "
                        f"quality = {corr['quality']}"
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

                    corr = compute_correlations(pool_deduped)
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
                        f"ρ = {corr['spearman_rho']:.3f}, "
                        f"n = {corr['n']}"
                    )

                    # Pooled scatter
                    plot_scatter(pool_deduped, model, chamber, "pooled", corr, ctx.plots_dir)

        # ── Save correlations JSON ──
        print_header("CORRELATIONS SUMMARY")
        corr_summary: dict = {}
        for key, data in all_results.items():
            corr_summary[key] = data["correlations"]
        for key, data in pooled_results.items():
            corr_summary[key] = data["correlations"]

        corr_path = ctx.data_dir / "correlations.json"
        with open(corr_path, "w") as f:
            json.dump(corr_summary, f, indent=2, default=str)
        print(f"  Saved: {corr_path.name}")

        # ── Filtering manifest ──
        print_header("FILTERING MANIFEST")
        manifest: dict = {
            "analysis": "external_validation",
            "sessions": sessions,
            "models": models,
            "constants": {
                "MIN_MATCHED": MIN_MATCHED,
                "STRONG_CORRELATION": STRONG_CORRELATION,
                "GOOD_CORRELATION": GOOD_CORRELATION,
                "CONCERN_CORRELATION": CONCERN_CORRELATION,
                "OUTLIER_TOP_N": OUTLIER_TOP_N,
            },
            "sm_kansas_legislators": sm_df.height,
        }

        for key, data in all_results.items():
            manifest[key] = {
                "n_matched": data["matched"].height,
                "n_ours": data["n_ours"],
                "n_sm": data["n_sm"],
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
        build_external_validation_report(
            ctx.report,
            all_results=all_results,
            pooled_results=pooled_results,
            sm_total=sm_df.height,
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
