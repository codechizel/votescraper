"""
Kansas Legislature — Common Space Ideal Points

Produces a single ideological scale spanning all available bienniums (78th-91st,
1999-2026) using simultaneous affine alignment of canonical ideal points via
bridge legislators.

Usage:
  uv run python analysis/28_common_space/common_space.py
  uv run python analysis/28_common_space/common_space.py --chambers house
  uv run python analysis/28_common_space/common_space.py --no-bootstrap
  uv run python analysis/28_common_space/common_space.py --sessions 90th_2023-2024,91st_2025-2026

Outputs (in results/kansas/cross-session/common-space/<YYMMDD>.n/):
  - data/:   common_space_house.parquet, common_space_senate.parquet,
             linking_coefficients.parquet, bridge_coverage.parquet
  - plots/:  bridge_heatmap.png, polarization_trajectory.png, etc.
  - common_space_report.html
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
        solve_simultaneous_alignment,
        transform_scores,
    )

try:
    from analysis.common_space_report import build_common_space_report
except ModuleNotFoundError:
    from common_space_report import build_common_space_report  # type: ignore[no-redef]

try:
    from analysis.phase_utils import normalize_name, print_header
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from phase_utils import normalize_name, print_header  # type: ignore[no-redef]
    from run_context import RunContext, resolve_upstream_dir  # type: ignore[no-redef]

import polars as pl

from tallgrass.session import KSSession

# ---------------------------------------------------------------------------
# Primer (auto-rendered to README.md by RunContext)
# ---------------------------------------------------------------------------

COMMON_SPACE_PRIMER = """\
# Common Space Ideal Points

## Purpose
Produce a single ideological scale spanning all available bienniums so that
legislators who never served together can be directly compared. Answers
questions like "Was Tim Huelskamp in 2001 more conservative than any current
legislator?" and "Has the Kansas Senate polarized over the last 25 years?"

## Method
Simultaneous affine alignment: fix the most recent biennium as the reference
scale, then estimate a scale (A) and shift (B) parameter for every other
biennium that minimizes discrepancies between bridge legislators' scores
across all session pairs. Bootstrap resampling provides 95% confidence
intervals. Input: canonical ideal points (horseshoe-corrected via the routing
system). External validation against Shor-McCarty, DIME CFscores, and
Dynamic IRT.

## Inputs
- Canonical ideal points from each biennium (Phase 05/06/07b routing)
- Legislator identity via normalized names and OCD person IDs

## Outputs
- `common_space_{chamber}.csv` — every legislator × biennium on a common scale
- `linking_coefficients.csv` — (A, B) per session with bootstrap CIs
- `bridge_coverage.csv` — pairwise bridge counts
- HTML report with bridge heatmap, polarization trajectory, career trajectories

## Interpretation Guide
Higher scores indicate more conservative ideology (Republicans positive,
Democrats negative). Scores are directly comparable across bienniums: a
score of +1.5 in the 79th Legislature means the same thing as +1.5 in the
91st. Confidence intervals widen for bienniums further from the reference.

## Caveats
- Absolute ideological drift (the entire legislature shifting right or left)
  is undetectable from roll call data alone — only relative positioning is
  calibrated.
- Bridge legislator selection bias: returning legislators may be systematically
  different from departing ones.
- The 84th-85th bridge (2012 redistricting) is the weakest link at 62.7%
  overlap.
"""


# ---------------------------------------------------------------------------
# Canonical score loading
# ---------------------------------------------------------------------------


def _load_canonical_for_session(
    results_root: Path,
    session_name: str,
    chamber: str,
    run_id: str | None = None,
) -> pl.DataFrame | None:
    """Attempt to load canonical ideal points for one chamber-session.

    Tries Phase 07b canonical routing, then Phase 06, then Phase 05 flat IRT.
    Returns None if nothing is available.
    """
    # Try H2D canonical routing first
    for phase in ["07b_hierarchical_2d", "06_irt_2d"]:
        try:
            phase_dir = resolve_upstream_dir(phase, results_root, run_id=run_id)
            canonical_dir = phase_dir / "canonical_irt"
            parquet = canonical_dir / f"canonical_ideal_points_{chamber.lower()}.parquet"
            if parquet.exists():
                df = pl.read_parquet(parquet)
                if "xi_mean" in df.columns and df.height > 0:
                    return df
        except FileNotFoundError, ValueError:
            continue

    # Fall back to flat IRT
    try:
        irt_dir = resolve_upstream_dir("05_irt", results_root, run_id=run_id)
        csv_path = irt_dir / "data" / f"ideal_points_{chamber.lower()}.csv"
        if csv_path.exists():
            return pl.read_csv(csv_path)
        parquet_path = irt_dir / "data" / f"ideal_points_{chamber.lower()}.parquet"
        if parquet_path.exists():
            return pl.read_parquet(parquet_path)
    except FileNotFoundError, ValueError:
        pass

    return None


def load_all_canonical_scores(
    sessions: list[str],
    chambers: list[str],
    run_id: str | None = None,
) -> dict[str, dict[str, pl.DataFrame]]:
    """Load canonical ideal points for all available sessions and chambers.

    Returns nested dict: session_name -> chamber -> DataFrame.
    Sessions with no available data are silently skipped.
    """
    all_scores: dict[str, dict[str, pl.DataFrame]] = {}

    for session_name in sessions:
        # Parse session to get results directory
        try:
            # Extract year range from session name (e.g., "91st_2025-2026" -> "2025-26")
            parts = session_name.split("_")
            if len(parts) < 2:
                continue
            year_range = parts[1]  # "2025-2026"
            start_year = int(year_range.split("-")[0])
            ks = KSSession(start_year)
            results_root = ks.results_dir
        except ValueError, IndexError:
            continue

        chamber_scores: dict[str, pl.DataFrame] = {}
        for chamber in chambers:
            chamber_cap = chamber.capitalize()
            df = _load_canonical_for_session(results_root, session_name, chamber_cap, run_id)
            if df is not None:
                # Ensure required columns
                required = {"legislator_slug", "full_name", "party", "xi_mean"}
                if required.issubset(set(df.columns)):
                    chamber_scores[chamber_cap] = df

        if chamber_scores:
            all_scores[session_name] = chamber_scores

    return all_scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Common Space Ideal Points — cross-temporal alignment"
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
        help="Comma-separated list of sessions to include (default: all available)",
    )
    parser.add_argument(
        "--bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run bootstrap for uncertainty quantification (default: yes)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help=f"Number of bootstrap iterations (default: {N_BOOTSTRAP})",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID for loading upstream pipeline results",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Force CSV-only mode (no database)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    chambers = ["house", "senate"] if args.chambers == "both" else [args.chambers]

    # Discover available sessions
    if args.sessions:
        sessions = [s.strip() for s in args.sessions.split(",")]
    else:
        # Auto-discover from data directory
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
                        "78th",
                        "79th",
                        "80th",
                        "81st",
                        "82nd",
                        "83rd",
                        "84th",
                        "85th",
                        "86th",
                        "87th",
                        "88th",
                        "89th",
                        "90th",
                        "91st",
                    ]
                )
            ],
            key=lambda s: int(s.split("_")[0].rstrip("thsndr")),
        )

    print_header("Phase 28: Common Space Ideal Points")
    print(f"Sessions: {len(sessions)}")
    print(f"Chambers: {', '.join(c.capitalize() for c in chambers)}")
    print(f"Reference: {args.reference_session}")
    print(f"Bootstrap: {args.bootstrap} ({args.n_bootstrap} iterations)")
    print()

    # Use cross-session results directory
    results_base = Path("results") / "kansas" / "cross-session"
    results_base.mkdir(parents=True, exist_ok=True)

    with RunContext(
        session="common-space",
        analysis_name="common_space",
        params=vars(args),
        primer=COMMON_SPACE_PRIMER,
        run_id=args.run_id,
        results_root=results_base,
    ) as ctx:
        # Step 1: Load canonical scores
        print("Loading canonical ideal points...")
        all_scores = load_all_canonical_scores(
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
            print("ERROR: Need at least 2 sessions for alignment. Exiting.")
            return

        # Ensure reference session is in the loaded set
        reference = args.reference_session
        if reference not in loaded_sessions:
            reference = loaded_sessions[-1]
            print(f"  Reference session not available; using {reference}")

        # Step 2: Build global roster
        print("\nBuilding global roster...")
        roster = build_global_roster(all_scores, normalize_name)
        n_unique = roster["name_norm"].n_unique()
        print(f"  {roster.height} legislator-session records, {n_unique} unique legislators")

        # Step 3: Bridge matrix
        print("\nComputing bridge coverage...")
        bridge_matrix = compute_bridge_matrix(roster, loaded_sessions)
        ctx.export_csv(bridge_matrix, "bridge_coverage.csv", "Pairwise bridge counts")
        bridge_matrix.write_parquet(ctx.data_dir / "bridge_coverage.parquet")

        # Print adjacent bridges
        for row in bridge_matrix.iter_rows(named=True):
            sa, sb = row["session_a"], row["session_b"]
            sa_idx = loaded_sessions.index(sa) if sa in loaded_sessions else -1
            sb_idx = loaded_sessions.index(sb) if sb in loaded_sessions else -1
            if sb_idx == sa_idx + 1:
                print(f"  {sa} -> {sb}: {row['n_bridges']} bridges")

        # Step 4-7: Per-chamber alignment
        all_results: dict[str, dict] = {}

        for chamber in chambers:
            chamber_cap = chamber.capitalize()
            print(f"\n{'=' * 60}")
            print(f"Aligning {chamber_cap}...")
            print(f"{'=' * 60}")

            # Check which sessions have this chamber
            chamber_sessions = [s for s in loaded_sessions if chamber_cap in all_scores.get(s, {})]
            if len(chamber_sessions) < 2:
                print(f"  Skipping {chamber_cap}: fewer than 2 sessions available")
                continue

            # Solve alignment
            print(f"  Solving simultaneous alignment ({len(chamber_sessions)} sessions)...")
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

            # Bootstrap
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

            # Transform scores
            print("  Transforming scores to common scale...")
            chamber_roster = roster.filter(pl.col("chamber") == chamber_cap)
            transformed = transform_scores(chamber_roster, coefficients, bootstrap_stats)

            # Quality gates
            print("  Running quality gates...")
            gates = compute_quality_gates(
                transformed,
                chamber_sessions,
                chamber_cap,
                party_d_min=PARTY_D_MIN,
                correlation_warn=CORRELATION_WARN,
            )
            for g in gates:
                status = "PASS" if g.passed else "FAIL"
                sign_str = "R>D" if g.sign_ok else "R<D"
                print(f"    {g.session}: d={g.party_d:.2f}, {sign_str} → {status}")

            # Polarization trajectory
            trajectory = compute_polarization_trajectory(
                transformed,
                chamber_sessions,
                chamber_cap,
            )

            # Save outputs
            out_name = f"common_space_{chamber}"
            transformed.write_parquet(ctx.data_dir / f"{out_name}.parquet")
            ctx.export_csv(
                transformed, f"{out_name}.csv", f"Common space ideal points — {chamber_cap}"
            )

            # Save linking coefficients
            coef_rows = []
            for s in chamber_sessions:
                A, B = coefficients[s]
                if bootstrap_stats and s in bootstrap_stats:
                    bs = bootstrap_stats[s]
                    coef_rows.append(
                        {
                            "session": s,
                            "chamber": chamber_cap,
                            "A": A,
                            "B": B,
                            "A_lo": bs.A_lo,
                            "A_hi": bs.A_hi,
                            "B_lo": bs.B_lo,
                            "B_hi": bs.B_hi,
                        }
                    )
                else:
                    coef_rows.append(
                        {
                            "session": s,
                            "chamber": chamber_cap,
                            "A": A,
                            "B": B,
                            "A_lo": A,
                            "A_hi": A,
                            "B_lo": B,
                            "B_hi": B,
                        }
                    )
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

        # Career scores (random-effects meta-analysis)
        for chamber_cap, r in all_results.items():
            print(f"\n  Computing career scores for {chamber_cap}...")
            career = compute_career_scores(r["transformed"], chamber_cap)
            if career.height > 0:
                career.write_parquet(ctx.data_dir / f"career_scores_{chamber_cap.lower()}.parquet")
                ctx.export_csv(
                    career,
                    f"career_scores_{chamber_cap.lower()}.csv",
                    f"Career-fixed ideal points — {chamber_cap} (RE meta-analysis)",
                )
                n_multi = career.filter(pl.col("n_sessions") >= 2).height
                n_stable = career.filter(pl.col("movement_flag") == "stable").height
                n_mover = career.filter(pl.col("movement_flag") == "mover").height
                print(
                    f"    {career.height} legislators, "
                    f"{n_multi} multi-session, "
                    f"{n_stable} stable, {n_mover} movers"
                )
                r["career"] = career

        # Save combined linking coefficients
        if all_results:
            all_coefs = pl.concat([r["coef_df"] for r in all_results.values()])
            all_coefs.write_parquet(ctx.data_dir / "linking_coefficients.parquet")
            ctx.export_csv(
                all_coefs,
                "linking_coefficients.csv",
                "Affine linking coefficients with bootstrap CIs",
            )

        # Save validation summary
        validation = {
            "n_sessions": len(loaded_sessions),
            "sessions": loaded_sessions,
            "reference": reference,
            "chambers": {
                chamber: {
                    "n_sessions": len(r["sessions"]),
                    "quality_gates": [
                        {
                            "session": g.session,
                            "party_d": g.party_d,
                            "sign_ok": g.sign_ok,
                            "passed": g.passed,
                        }
                        for g in r["gates"]
                    ],
                }
                for chamber, r in all_results.items()
            },
        }
        with open(ctx.data_dir / "validation.json", "w") as f:
            json.dump(validation, f, indent=2)

        # Build report
        print("\nBuilding report...")
        build_common_space_report(
            report=ctx.report,
            all_results=all_results,
            bridge_matrix=bridge_matrix,
            sessions=loaded_sessions,
            reference=reference,
            plots_dir=ctx.plots_dir,
        )

        print(f"\n{'=' * 60}")
        print("Phase 28 Complete")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
