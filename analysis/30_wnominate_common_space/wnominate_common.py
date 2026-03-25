"""
Kansas Legislature — W-NOMINATE Common Space Ideal Points (Phase 30)

Applies the same pairwise chain affine alignment as Phase 28 to W-NOMINATE
Dim 1 scores from Phase 16, producing field-standard W-NOMINATE-scaled
common-space ideal points and career scores.  Cross-method validation
compares W-NOMINATE career scores against Phase 28 IRT career scores.

Usage:
  uv run python analysis/30_wnominate_common_space/wnominate_common.py
  uv run python analysis/30_wnominate_common_space/wnominate_common.py --chambers house
  uv run python analysis/30_wnominate_common_space/wnominate_common.py --no-bootstrap

Outputs (in results/kansas/cross-session/wnominate-common-space/<YYMMDD>.n/):
  - data/:   wnom_common_space_{house,senate}.parquet, career scores, validation
  - plots/:  bridge_heatmap.png, linking coefficients, career vs recent
  - wnominate_common_report.html
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scipy import stats as sp_stats

try:
    from analysis.common_space_data import (
        CORRELATION_WARN,
        N_BOOTSTRAP,
        PARTY_D_MIN,
        REFERENCE_SESSION,
        TRIM_PCT,
        bootstrap_alignment_direct,
        build_global_roster,
        compute_bridge_matrix,
        compute_career_scores,
        compute_polarization_trajectory,
        compute_quality_gates,
        compute_unified_career_scores,
        link_chambers,
        solve_simultaneous_alignment,
        transform_scores,
    )
except ModuleNotFoundError:
    from common_space_data import (  # type: ignore[no-redef]
        CORRELATION_WARN,
        N_BOOTSTRAP,
        PARTY_D_MIN,
        REFERENCE_SESSION,
        TRIM_PCT,
        bootstrap_alignment_direct,
        build_global_roster,
        compute_bridge_matrix,
        compute_career_scores,
        compute_polarization_trajectory,
        compute_quality_gates,
        compute_unified_career_scores,
        link_chambers,
        solve_simultaneous_alignment,
        transform_scores,
    )

try:
    from analysis.common_space_report import build_common_space_reports
except ModuleNotFoundError:
    from common_space_report import build_common_space_reports  # type: ignore[no-redef]

try:
    from analysis.phase_utils import normalize_name, print_header
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from phase_utils import normalize_name, print_header  # type: ignore[no-redef]
    from run_context import RunContext, resolve_upstream_dir  # type: ignore[no-redef]

import numpy as np
import polars as pl

from tallgrass.session import KSSession

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SE: float = 0.05
"""Floor for W-NOMINATE SEs.  Phase 16 bootstrap SEs are all zero; this
floor ensures alignment uncertainty dominates (not zero-variance artifacts).
0.05 is ~5% of the [-1, +1] W-NOMINATE scale."""

WNOM_PRIMER = """\
# W-NOMINATE Common Space Ideal Points

## Purpose
Produce a field-standard W-NOMINATE-scaled common space across all bienniums.
Validates Phase 28 IRT common space via cross-method comparison.

## Method
Same pairwise chain affine alignment as Phase 28, applied to Phase 16
W-NOMINATE Dim 1 scores instead of canonical IRT ideal points.  Bootstrap
resampling for uncertainty; delta-method propagation through the chain.

## Inputs
- W-NOMINATE Dim 1 scores from Phase 16 (per biennium × chamber)
- Phase 28 IRT career scores (for cross-method validation)

## Outputs
- `wnom_common_space_{chamber}.csv` — W-NOMINATE scores on common scale
- `wnom_career_scores_unified.csv` — one career score per legislator
- `wnom_vs_irt_comparison.csv` — cross-method validation

## Interpretation Guide
W-NOMINATE scores are bounded to [-1, +1] per session but may exceed
these bounds after affine transformation to the common scale.  Higher
scores = more conservative.  Compare with Phase 28 IRT career scores
to assess robustness of ideological rankings.

## Caveats
- W-NOMINATE bootstrap SEs are zero in our data; all uncertainty comes
  from the alignment bootstrap, not per-legislator estimation.
- The bounded [-1, +1] scale creates compression at extremes that IRT
  avoids.
"""


# ---------------------------------------------------------------------------
# W-NOMINATE score loading
# ---------------------------------------------------------------------------


def _load_wnominate_for_session(
    results_root: Path,
    session_name: str,
    chamber: str,
    run_id: str | None = None,
) -> pl.DataFrame | None:
    """Load W-NOMINATE Dim 1 scores for one chamber-session from Phase 16.

    Applies sign correction so that Republicans are positive (convention).
    W-NOMINATE polarity varies across sessions; without correction, the
    pairwise chain alignment breaks when adjacent sessions have opposite signs.
    """
    try:
        phase_dir = resolve_upstream_dir("16_wnominate", results_root, run_id=run_id)
        parquet = phase_dir / "data" / f"wnominate_coords_{chamber.lower()}.parquet"
        if not parquet.exists():
            return None
        df = pl.read_parquet(parquet)
        if "wnom_dim1" not in df.columns or df.height == 0:
            return None

        # Sign correction: ensure Republicans are positive on Dim 1
        r_mean = df.filter(pl.col("party") == "Republican")["wnom_dim1"].mean()
        d_mean = df.filter(pl.col("party") == "Democrat")["wnom_dim1"].mean()
        sign_flip = -1.0 if (r_mean is not None and d_mean is not None and r_mean < d_mean) else 1.0

        return df.select(
            "legislator_slug",
            "full_name",
            "party",
            (pl.col("wnom_dim1") * sign_flip).alias("xi_mean"),
            pl.col("wnom_se1").cast(pl.Float64).clip(lower_bound=MIN_SE).alias("xi_sd"),
        )
    except FileNotFoundError, ValueError:
        return None


def load_all_wnominate_scores(
    sessions: list[str],
    chambers: list[str],
    run_id: str | None = None,
) -> dict[str, dict[str, pl.DataFrame]]:
    """Load W-NOMINATE Dim 1 scores for all available sessions and chambers."""
    all_scores: dict[str, dict[str, pl.DataFrame]] = {}

    for session_name in sessions:
        try:
            parts = session_name.split("_")
            if len(parts) < 2:
                continue
            year_range = parts[1]
            start_year = int(year_range.split("-")[0])
            ks = KSSession(start_year)
            results_root = ks.results_dir
        except ValueError, IndexError:
            continue

        chamber_scores: dict[str, pl.DataFrame] = {}
        for chamber in chambers:
            chamber_cap = chamber.capitalize()
            df = _load_wnominate_for_session(results_root, session_name, chamber_cap, run_id)
            if df is not None:
                chamber_scores[chamber_cap] = df

        if chamber_scores:
            all_scores[session_name] = chamber_scores

    return all_scores


# ---------------------------------------------------------------------------
# Cross-method validation
# ---------------------------------------------------------------------------


def compute_cross_method_validation(
    wnom_career: pl.DataFrame,
    irt_career_path: Path,
) -> tuple[dict, pl.DataFrame | None]:
    """Compare W-NOMINATE career scores against Phase 28 IRT career scores.

    Returns (summary_dict, comparison_df).  Returns ({}, None) if IRT
    career scores are unavailable.
    """
    if not irt_career_path.exists():
        return {}, None

    irt_career = pl.read_parquet(irt_career_path)

    # Join on person_key
    joined = wnom_career.select(
        "person_key",
        "full_name",
        "party",
        pl.col("career_score").alias("wnom_score"),
    ).join(
        irt_career.select(
            "person_key",
            pl.col("career_score").alias("irt_score"),
        ),
        on="person_key",
        how="inner",
    )

    if joined.height < 10:
        return {"n_matched": joined.height, "note": "too few matches"}, None

    wnom = joined["wnom_score"].to_numpy()
    irt = joined["irt_score"].to_numpy()

    # Overall correlations
    pearson_r, _ = sp_stats.pearsonr(wnom, irt)
    spearman_rho, _ = sp_stats.spearmanr(wnom, irt)

    # Within-party
    result: dict = {
        "n_matched": joined.height,
        "pearson_r": round(float(pearson_r), 4),
        "spearman_rho": round(float(spearman_rho), 4),
    }

    for party in ("Republican", "Democrat"):
        party_df = joined.filter(pl.col("party") == party)
        if party_df.height >= 5:
            w = party_df["wnom_score"].to_numpy()
            i = party_df["irt_score"].to_numpy()
            r_p, _ = sp_stats.pearsonr(w, i)
            result[f"{party.lower()}_r"] = round(float(r_p), 4)
            result[f"{party.lower()}_n"] = party_df.height

    # Rank comparison
    joined = joined.with_columns(
        pl.col("wnom_score").rank(descending=True).alias("wnom_rank"),
        pl.col("irt_score").rank(descending=True).alias("irt_rank"),
    ).with_columns(
        (pl.col("wnom_rank") - pl.col("irt_rank")).abs().alias("rank_diff"),
    ).sort("rank_diff", descending=True)

    return result, joined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="W-NOMINATE Common Space — cross-temporal alignment of W-NOMINATE scores"
    )
    parser.add_argument(
        "--chambers",
        default="both",
        choices=["house", "senate", "both"],
        help="Which chamber(s) to align (default: both)",
    )
    parser.add_argument(
        "--reference-session",
        default=REFERENCE_SESSION,
        help=f"Reference biennium (default: {REFERENCE_SESSION})",
    )
    parser.add_argument(
        "--sessions",
        default=None,
        help="Comma-separated list of sessions (default: all available)",
    )
    parser.add_argument(
        "--bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run bootstrap for uncertainty (default: yes)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help=f"Bootstrap iterations (default: {N_BOOTSTRAP})",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID for loading upstream pipeline results",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Force CSV-only mode",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    chambers = ["house", "senate"] if args.chambers == "both" else [args.chambers]

    # Discover sessions
    if args.sessions:
        sessions = [s.strip() for s in args.sessions.split(",")]
    else:
        data_root = Path("data") / "kansas"
        sessions = sorted(
            [
                d.name
                for d in data_root.iterdir()
                if d.is_dir()
                and "_" in d.name
                and any(
                    d.name.startswith(p)
                    for p in [
                        "78th", "79th", "80th", "81st", "82nd", "83rd", "84th",
                        "85th", "86th", "87th", "88th", "89th", "90th", "91st",
                    ]
                )
            ],
            key=lambda s: int(s.split("_")[0].rstrip("thsndr")),
        )

    print_header("Phase 30: W-NOMINATE Common Space")
    print(f"Sessions: {len(sessions)}")
    print(f"Chambers: {', '.join(c.capitalize() for c in chambers)}")
    print(f"Reference: {args.reference_session}")
    print(f"Bootstrap: {args.bootstrap} ({args.n_bootstrap} iterations)")
    print()

    results_base = Path("results") / "kansas" / "cross-session"
    results_base.mkdir(parents=True, exist_ok=True)

    with RunContext(
        session="wnominate-common-space",
        analysis_name="wnominate_common",
        params=vars(args),
        primer=WNOM_PRIMER,
        run_id=args.run_id,
        results_root=results_base,
    ) as ctx:
        # Step 1: Load W-NOMINATE scores
        print("Loading W-NOMINATE scores...")
        all_scores = load_all_wnominate_scores(
            sessions=sessions,
            chambers=chambers,
            run_id=args.run_id,
        )
        loaded_sessions = sorted(
            all_scores.keys(), key=lambda s: int(s.split("_")[0].rstrip("thsndr"))
        )
        print(
            f"  Loaded {len(loaded_sessions)} sessions: "
            f"{loaded_sessions[0]} through {loaded_sessions[-1]}"
        )

        if len(loaded_sessions) < 2:
            print("ERROR: Need at least 2 sessions. Exiting.")
            return

        reference = args.reference_session
        if reference not in loaded_sessions:
            reference = loaded_sessions[-1]
            print(f"  Reference not available; using {reference}")

        # Step 2: Global roster
        print("\nBuilding global roster...")
        roster = build_global_roster(all_scores, normalize_name)
        n_unique = roster["person_key"].n_unique()
        print(f"  {roster.height} legislator-session records, {n_unique} unique legislators")

        # Step 3: Bridge matrix
        print("\nComputing bridge coverage...")
        bridge_matrix = compute_bridge_matrix(roster, loaded_sessions)
        ctx.export_csv(bridge_matrix, "wnom_bridge_coverage.csv", "Pairwise bridge counts")
        bridge_matrix.write_parquet(ctx.data_dir / "wnom_bridge_coverage.parquet")

        for row in bridge_matrix.iter_rows(named=True):
            sa, sb = row["session_a"], row["session_b"]
            sa_idx = loaded_sessions.index(sa) if sa in loaded_sessions else -1
            sb_idx = loaded_sessions.index(sb) if sb in loaded_sessions else -1
            if sb_idx == sa_idx + 1:
                print(f"  {sa} -> {sb}: {row['n_bridges']} bridges")

        # Step 4-7: Per-chamber alignment (identical to Phase 28)
        all_results: dict[str, dict] = {}

        for chamber in chambers:
            chamber_cap = chamber.capitalize()
            print(f"\n{'=' * 60}")
            print(f"Aligning {chamber_cap} (W-NOMINATE)...")
            print(f"{'=' * 60}")

            chamber_sessions = [
                s for s in loaded_sessions if chamber_cap in all_scores.get(s, {})
            ]
            if len(chamber_sessions) < 2:
                print(f"  Skipping {chamber_cap}: fewer than 2 sessions")
                continue

            print(f"  Solving pairwise chain alignment ({len(chamber_sessions)} sessions)...")
            coefficients = solve_simultaneous_alignment(
                roster=roster,
                sessions=chamber_sessions,
                chamber=chamber_cap,
                reference=reference,
                trim_pct=TRIM_PCT,
            )

            for s, (A, B) in sorted(coefficients.items()):
                if s != reference:
                    print(f"    {s}: A={A:.4f}, B={B:+.4f}")

            bootstrap_stats = None
            if args.bootstrap:
                print(f"  Bootstrap ({args.n_bootstrap} iterations)...")
                bootstrap_stats = bootstrap_alignment_direct(
                    roster=roster,
                    sessions=chamber_sessions,
                    chamber=chamber_cap,
                    reference=reference,
                    n_bootstrap=args.n_bootstrap,
                    trim_pct=TRIM_PCT,
                )

            print("  Transforming scores to common scale...")
            chamber_roster = roster.filter(pl.col("chamber") == chamber_cap)
            transformed = transform_scores(chamber_roster, coefficients, bootstrap_stats)

            print("  Running quality gates...")
            gates = compute_quality_gates(
                transformed, chamber_sessions, chamber_cap,
                party_d_min=PARTY_D_MIN, correlation_warn=CORRELATION_WARN,
            )
            for g in gates:
                status = "PASS" if g.passed else "FAIL"
                sign_str = "R>D" if g.sign_ok else "R<D"
                print(f"    {g.session}: d={g.party_d:.2f}, {sign_str} → {status}")

            trajectory = compute_polarization_trajectory(
                transformed, chamber_sessions, chamber_cap,
            )

            # Save outputs
            out_name = f"wnom_common_space_{chamber}"
            transformed.write_parquet(ctx.data_dir / f"{out_name}.parquet")
            ctx.export_csv(
                transformed, f"{out_name}.csv",
                f"W-NOMINATE common space — {chamber_cap}",
            )

            coef_rows = []
            for s in chamber_sessions:
                A, B = coefficients[s]
                if bootstrap_stats and s in bootstrap_stats:
                    bs = bootstrap_stats[s]
                    coef_rows.append({
                        "session": s, "chamber": chamber_cap, "A": A, "B": B,
                        "A_lo": bs.A_lo, "A_hi": bs.A_hi,
                        "B_lo": bs.B_lo, "B_hi": bs.B_hi,
                    })
                else:
                    coef_rows.append({
                        "session": s, "chamber": chamber_cap, "A": A, "B": B,
                        "A_lo": A, "A_hi": A, "B_lo": B, "B_hi": B,
                    })
            coef_df = pl.DataFrame(coef_rows)

            all_results[chamber_cap] = {
                "transformed": transformed,
                "coefficients": coefficients,
                "bootstrap_stats": bootstrap_stats,
                "coef_df": coef_df,
                "gates": gates,
                "trajectory": trajectory,
                "sessions": chamber_sessions,
            }

        # Career scores
        for chamber_cap in [k for k in all_results if k in ("House", "Senate")]:
            r = all_results[chamber_cap]
            print(f"\n  Computing career scores for {chamber_cap}...")
            career = compute_career_scores(r["transformed"], chamber_cap)
            if career.height > 0:
                career.write_parquet(
                    ctx.data_dir / f"wnom_career_scores_{chamber_cap.lower()}.parquet"
                )
                ctx.export_csv(
                    career,
                    f"wnom_career_scores_{chamber_cap.lower()}.csv",
                    f"W-NOMINATE career scores — {chamber_cap}",
                )
                n_multi = career.filter(pl.col("n_sessions") >= 2).height
                n_stable = career.filter(pl.col("movement_flag") == "stable").height
                n_mover = career.filter(pl.col("movement_flag") == "mover").height
                print(
                    f"    {career.height} legislators, "
                    f"{n_multi} multi-session, {n_stable} stable, {n_mover} movers"
                )
                r["career"] = career

        # Cross-chamber unification
        if "House" in all_results and "Senate" in all_results:
            house_t = all_results["House"]["transformed"]
            senate_t = all_results["Senate"]["transformed"]
            print("\n  Linking House and Senate scales...")
            unified, A_cs, B_cs = link_chambers(house_t, senate_t, trim_pct=TRIM_PCT)
            n_bridges = unified.filter(
                pl.col("person_key").is_in(
                    set(house_t["person_key"].to_list())
                    & set(senate_t["person_key"].to_list())
                )
            )["person_key"].n_unique()
            print(f"    {n_bridges} chamber-switcher bridges, A={A_cs:.4f}, B={B_cs:+.4f}")

            unified.write_parquet(ctx.data_dir / "wnom_common_space_unified.parquet")
            ctx.export_csv(
                unified,
                "wnom_common_space_unified.csv",
                "Unified W-NOMINATE common space (House + Senate)",
            )

            print("\n  Computing unified career scores...")
            unified_career = compute_unified_career_scores(unified)
            if unified_career.height > 0:
                unified_career.write_parquet(
                    ctx.data_dir / "wnom_career_scores_unified.parquet"
                )
                ctx.export_csv(
                    unified_career,
                    "wnom_career_scores_unified.csv",
                    "Unified W-NOMINATE career scores",
                )
                n_cross = unified_career.filter(
                    pl.col("chambers").str.contains("&")
                ).height
                n_multi = unified_career.filter(pl.col("n_sessions") >= 2).height
                n_stable = unified_career.filter(
                    pl.col("movement_flag") == "stable"
                ).height
                n_mover = unified_career.filter(
                    pl.col("movement_flag") == "mover"
                ).height
                print(
                    f"    {unified_career.height} legislators, "
                    f"{n_cross} cross-chamber, "
                    f"{n_multi} multi-session, "
                    f"{n_stable} stable, {n_mover} movers"
                )
                all_results["unified_career"] = unified_career
                all_results["chamber_link"] = {
                    "A": A_cs, "B": B_cs, "n_bridges": n_bridges,
                }

        # Cross-method validation (compare with Phase 28 IRT career scores)
        irt_career_path = (
            Path("results")
            / "kansas"
            / "cross-session"
            / "common-space"
            / "common_space"
            / "latest"
            / "data"
            / "career_scores_unified.parquet"
        )
        unified_career = all_results.get("unified_career")
        if unified_career is not None:
            print("\n  Cross-method validation (W-NOMINATE vs IRT)...")
            validation_summary, comparison_df = compute_cross_method_validation(
                unified_career, irt_career_path,
            )
            if comparison_df is not None:
                comparison_df.write_parquet(
                    ctx.data_dir / "wnom_vs_irt_comparison.parquet"
                )
                ctx.export_csv(
                    comparison_df,
                    "wnom_vs_irt_comparison.csv",
                    "W-NOMINATE vs IRT career score comparison",
                )
                print(f"    Matched: {validation_summary['n_matched']} legislators")
                print(f"    Pearson r: {validation_summary['pearson_r']:.4f}")
                print(f"    Spearman ρ: {validation_summary['spearman_rho']:.4f}")
                for party in ("republican", "democrat"):
                    if f"{party}_r" in validation_summary:
                        print(
                            f"    {party.capitalize()} r: "
                            f"{validation_summary[f'{party}_r']:.4f} "
                            f"(n={validation_summary[f'{party}_n']})"
                        )
                all_results["validation"] = validation_summary
                all_results["comparison"] = comparison_df
            else:
                print("    SKIP: Phase 28 IRT career scores not found")

        # Save validation summary
        val_out = {
            "n_sessions": len(loaded_sessions),
            "sessions": loaded_sessions,
            "reference": reference,
            "method": "W-NOMINATE Dim 1",
            "se_floor": MIN_SE,
            "chambers": {
                chamber: {
                    "n_sessions": len(r["sessions"]),
                    "quality_gates": [
                        {"session": g.session, "party_d": g.party_d,
                         "sign_ok": g.sign_ok, "passed": g.passed}
                        for g in r["gates"]
                    ],
                }
                for chamber, r in all_results.items()
                if chamber in ("House", "Senate")
            },
        }
        if "validation" in all_results:
            val_out["cross_method"] = all_results["validation"]
        with open(ctx.data_dir / "validation.json", "w") as f:
            json.dump(val_out, f, indent=2)

        # Save linking coefficients
        chamber_results = {k: v for k, v in all_results.items() if k in ("House", "Senate")}
        if chamber_results:
            all_coefs = pl.concat([r["coef_df"] for r in chamber_results.values()])
            all_coefs.write_parquet(ctx.data_dir / "wnom_linking_coefficients.parquet")
            ctx.export_csv(
                all_coefs,
                "wnom_linking_coefficients.csv",
                "W-NOMINATE linking coefficients with bootstrap CIs",
            )

        # Build reports
        print("\nBuilding reports...")
        sub_reports = build_common_space_reports(
            all_results=all_results,
            bridge_matrix=bridge_matrix,
            sessions=loaded_sessions,
            reference=reference,
            plots_dir=ctx.plots_dir,
        )

        session_root = results_base / "wnominate-common-space"
        for name, sub_report in sub_reports.items():
            sub_report.title = sub_report.title.replace("Common Space", "W-NOMINATE Common Space")
            sub_path = ctx.run_dir / f"wnominate_common_{name}_report.html"
            sub_report.session = ctx.session
            sub_report.write(sub_path)
            print(f"  Saved: wnominate_common_{name}_report.html")

            link_path = session_root / f"wnominate_common_{name}_report.html"
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(
                Path("wnominate_common")
                / "latest"
                / f"wnominate_common_{name}_report.html"
            )

        # Validation report (cross-method comparison)
        comparison_df = all_results.get("comparison")
        validation_summary = all_results.get("validation")
        if comparison_df is not None and validation_summary is not None:
            from analysis.report import (
                InteractiveTableSection,
                KeyFindingsSection,
                ReportBuilder,
                TableSection,
                TextSection,
                make_gt,
                make_interactive_table,
            )

            val_report = ReportBuilder(title="W-NOMINATE vs IRT — Cross-Method Validation")
            val_report.add(
                KeyFindingsSection(
                    findings=[
                        f"Matched {validation_summary['n_matched']} legislators across both methods",
                        f"Overall: Pearson r = {validation_summary['pearson_r']:.4f}, "
                        f"Spearman ρ = {validation_summary['spearman_rho']:.4f}",
                    ]
                    + [
                        f"{party.capitalize()}: r = {validation_summary[f'{party}_r']:.4f} "
                        f"(n={validation_summary[f'{party}_n']})"
                        for party in ("republican", "democrat")
                        if f"{party}_r" in validation_summary
                    ]
                )
            )

            # Top 25 divergent legislators
            top_25 = comparison_df.head(25).select(
                "full_name", "party", "wnom_score", "irt_score",
                "wnom_rank", "irt_rank", "rank_diff",
            )
            for col in ("wnom_score", "irt_score"):
                top_25 = top_25.with_columns(pl.col(col).round(3))
            top_25 = top_25.with_columns(
                pl.col("wnom_rank").cast(pl.Int32),
                pl.col("irt_rank").cast(pl.Int32),
                pl.col("rank_diff").cast(pl.Int32),
            )

            html = make_interactive_table(
                top_25,
                title="Top 25 Legislators with Largest Rank Differences",
                column_labels={
                    "full_name": "Legislator",
                    "party": "Party",
                    "wnom_score": "W-NOM Score",
                    "irt_score": "IRT Score",
                    "wnom_rank": "W-NOM Rank",
                    "irt_rank": "IRT Rank",
                    "rank_diff": "Rank Diff",
                },
                number_formats={
                    "wnom_score": ".3f",
                    "irt_score": ".3f",
                },
                caption=(
                    "Legislators where W-NOMINATE and IRT career scores disagree "
                    "most on rank ordering. Large rank differences may indicate "
                    "sessions where W-NOMINATE's bounded [-1,+1] scale compresses "
                    "scores differently than IRT's unbounded scale, or where the "
                    "two methods weight different voting patterns."
                ),
            )
            val_report.add(
                InteractiveTableSection(
                    id="top-divergent",
                    title="Top 25 Divergent Legislators",
                    html=html,
                )
            )

            # Full comparison table
            full_display = comparison_df.select(
                "full_name", "party", "wnom_score", "irt_score",
                "wnom_rank", "irt_rank", "rank_diff",
            )
            for col in ("wnom_score", "irt_score"):
                full_display = full_display.with_columns(pl.col(col).round(3))
            full_display = full_display.with_columns(
                pl.col("wnom_rank").cast(pl.Int32),
                pl.col("irt_rank").cast(pl.Int32),
                pl.col("rank_diff").cast(pl.Int32),
            )
            full_html = make_interactive_table(
                full_display,
                title=f"All {full_display.height} Legislators — W-NOMINATE vs IRT Rank Comparison",
                column_labels={
                    "full_name": "Legislator",
                    "party": "Party",
                    "wnom_score": "W-NOM Score",
                    "irt_score": "IRT Score",
                    "wnom_rank": "W-NOM Rank",
                    "irt_rank": "IRT Rank",
                    "rank_diff": "Rank Diff",
                },
                number_formats={
                    "wnom_score": ".3f",
                    "irt_score": ".3f",
                },
                caption="Sorted by rank difference (descending). Searchable and sortable.",
            )
            val_report.add(
                InteractiveTableSection(
                    id="full-comparison",
                    title="Full Comparison Table",
                    html=full_html,
                )
            )

            # Write validation report
            val_path = ctx.run_dir / "wnominate_common_validation_report.html"
            val_report.session = ctx.session
            val_report.write(val_path)
            print(f"  Saved: wnominate_common_validation_report.html")

            link_path = session_root / "wnominate_common_validation_report.html"
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(
                Path("wnominate_common")
                / "latest"
                / "wnominate_common_validation_report.html"
            )

        # Combined report (auto-written by RunContext)
        from analysis.common_space_report import build_common_space_report

        build_common_space_report(
            report=ctx.report,
            all_results=all_results,
            bridge_matrix=bridge_matrix,
            sessions=loaded_sessions,
            reference=reference,
            plots_dir=ctx.plots_dir,
        )

        # Add validation sections to combined report
        if comparison_df is not None and validation_summary is not None:
            for _, section in val_report._sections:
                ctx.report.add(section)

        print(f"\n{'=' * 60}")
        print("Phase 30 Complete")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
