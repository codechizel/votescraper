"""Canonical ideal point routing — auto-select 1D IRT or 2D Dim 1 per chamber.

For horseshoe-affected chambers (supermajority, where 1D IRT conflates ideology
with establishment-loyalty), the 2D IRT Dim 1 is the canonical ideology score,
following the DW-NOMINATE standard. For balanced chambers, 1D IRT remains
canonical.

This module runs after Phase 06 (2D IRT) and writes a canonical output that
downstream phases (synthesis, profiles, cross-session) consume instead of
reading Phase 05 directly.

See docs/canonical-ideal-points.md for the full rationale and ADR-0109.

Usage (called from irt_2d.py at end of Phase 06):
    from analysis.canonical_ideal_points import write_canonical_ideal_points
    write_canonical_ideal_points(chamber, irt_1d_dir, irt_2d_dir, output_dir)
"""

from pathlib import Path

import polars as pl

# Horseshoe detection threshold (must match Phase 05)
HORSESHOE_DEM_WRONG_SIDE_FRAC = 0.20
HORSESHOE_OVERLAP_FRAC = 0.30

# 2D convergence quality gates — if 2D doesn't meet these, fall back to 1D
DIM1_RHAT_THRESHOLD = 1.05
DIM1_ESS_THRESHOLD = 200


def detect_horseshoe_from_ideal_points(ideal_points: pl.DataFrame) -> dict:
    """Lightweight horseshoe detection using 1D ideal points.

    Checks:
    1. Democrat wrong-side fraction (>20% on conservative side)
    2. Party overlap fraction (>30%)
    3. Any Republican more liberal than Democrat mean

    Returns dict with 'detected' bool and diagnostic metrics.
    """
    r_ip = ideal_points.filter(pl.col("party") == "Republican")
    d_ip = ideal_points.filter(pl.col("party") == "Democrat")

    if d_ip.height == 0 or r_ip.height == 0:
        return {"detected": False, "reason": "single_party", "metrics": {}}

    # 1. Democrat wrong-side fraction
    d_wrong_side = float((d_ip["xi_mean"] > 0).mean())

    # 2. Party overlap
    r_mean = float(r_ip["xi_mean"].mean())
    d_mean = float(d_ip["xi_mean"].mean())
    r_below_d_mean = float((r_ip["xi_mean"] < d_mean).mean())
    d_above_r_mean = float((d_ip["xi_mean"] > r_mean).mean())
    overlap_frac = (r_below_d_mean + d_above_r_mean) / 2

    # 3. Most extreme R check
    most_neg_r = float(r_ip["xi_mean"].min())
    r_more_liberal_than_d_mean = most_neg_r < d_mean

    detected = (
        d_wrong_side > HORSESHOE_DEM_WRONG_SIDE_FRAC
        or r_more_liberal_than_d_mean
        or overlap_frac > HORSESHOE_OVERLAP_FRAC
    )

    reasons = []
    if d_wrong_side > HORSESHOE_DEM_WRONG_SIDE_FRAC:
        reasons.append(f"dem_wrong_side={d_wrong_side:.2f}")
    if r_more_liberal_than_d_mean:
        reasons.append(f"r_more_liberal_than_d_mean (min_r={most_neg_r:.3f}, d_mean={d_mean:.3f})")
    if overlap_frac > HORSESHOE_OVERLAP_FRAC:
        reasons.append(f"overlap={overlap_frac:.2f}")

    return {
        "detected": detected,
        "reason": "; ".join(reasons) if reasons else "no_horseshoe",
        "metrics": {
            "dem_wrong_side_frac": d_wrong_side,
            "overlap_frac": overlap_frac,
            "r_mean": r_mean,
            "d_mean": d_mean,
            "r_more_liberal_than_d_mean": r_more_liberal_than_d_mean,
        },
    }


def load_1d_ideal_points(irt_1d_dir: Path, chamber: str) -> pl.DataFrame | None:
    """Load 1D IRT ideal points from Phase 05 data directory."""
    path = irt_1d_dir / "data" / f"ideal_points_{chamber.lower()}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    return None


def load_2d_dim1_ideal_points(irt_2d_dir: Path, chamber: str) -> pl.DataFrame | None:
    """Load 2D IRT Dim 1 ideal points from Phase 06 data directory.

    Maps 2D column names to the standard schema:
    xi_dim1_mean → xi_mean, xi_dim1_hdi_3% → xi_hdi_2.5, xi_dim1_hdi_97% → xi_hdi_97.5
    """
    path = irt_2d_dir / "data" / f"ideal_points_2d_{chamber.lower()}.parquet"
    if not path.exists():
        return None

    df = pl.read_parquet(path)

    # Compute xi_sd from HDI (approximate: HDI width / 3.92 for 95% interval)
    df = df.with_columns(
        ((pl.col("xi_dim1_hdi_97%") - pl.col("xi_dim1_hdi_3%")) / 3.92).alias("xi_sd")
    )

    # Map to standard column names
    result = df.select(
        "legislator_slug",
        "full_name",
        "party",
        pl.col("xi_dim1_mean").alias("xi_mean"),
        "xi_sd",
        pl.col("xi_dim1_hdi_3%").alias("xi_hdi_2.5"),
        pl.col("xi_dim1_hdi_97%").alias("xi_hdi_97.5"),
    )

    return result


def check_2d_convergence_quality(irt_2d_dir: Path, chamber: str) -> bool:
    """Check if 2D model converged well enough for Dim 1 to be trusted.

    Reads convergence_summary.json and checks R-hat and ESS for xi.
    """
    import json

    summary_path = irt_2d_dir / "data" / "convergence_summary.json"
    if not summary_path.exists():
        print(f"  WARNING: No convergence summary found at {summary_path}")
        return False

    summary = json.loads(summary_path.read_text())
    chamber_data = summary.get("chambers", {}).get(chamber, {})
    convergence = chamber_data.get("convergence", {})

    xi_rhat = convergence.get("xi_rhat_max", 999.0)
    xi_ess = convergence.get("xi_ess_min", 0.0)

    ok = xi_rhat < DIM1_RHAT_THRESHOLD and xi_ess > DIM1_ESS_THRESHOLD
    if not ok:
        print(
            f"  WARNING: 2D {chamber} convergence insufficient "
            f"(R-hat={xi_rhat:.4f}, ESS={xi_ess:.0f})"
        )
    return ok


def route_canonical_ideal_points(
    irt_1d_dir: Path,
    irt_2d_dir: Path,
    chamber: str,
) -> tuple[pl.DataFrame, str, dict]:
    """Determine the canonical ideal points for a chamber.

    Returns (ideal_points_df, source_label, routing_metadata).

    source_label is "2d_dim1" or "1d_irt".
    """
    ip_1d = load_1d_ideal_points(irt_1d_dir, chamber)
    ip_2d = load_2d_dim1_ideal_points(irt_2d_dir, chamber)

    metadata: dict = {"chamber": chamber}

    # If 1D not available, can't detect horseshoe — use 2D if available
    if ip_1d is None:
        if ip_2d is not None:
            metadata["reason"] = "1d_unavailable"
            return ip_2d.with_columns(pl.lit("2d_dim1").alias("source")), "2d_dim1", metadata
        msg = f"No ideal points available for {chamber}"
        raise FileNotFoundError(msg)

    # Detect horseshoe
    horseshoe = detect_horseshoe_from_ideal_points(ip_1d)
    metadata["horseshoe"] = horseshoe

    if horseshoe["detected"]:
        print(f"  Horseshoe DETECTED in {chamber}: {horseshoe['reason']}")

        # Check 2D availability and convergence
        if ip_2d is not None and check_2d_convergence_quality(irt_2d_dir, chamber):
            print("  → Canonical source: 2D Dim 1 (horseshoe detected, 2D convergence OK)")
            metadata["reason"] = f"horseshoe_detected: {horseshoe['reason']}"

            # Add district and chamber columns from 1D if available
            extra_cols = []
            for col in ("district", "chamber"):
                if col in ip_1d.columns and col not in ip_2d.columns:
                    extra_cols.append(col)
            if extra_cols:
                ip_2d = ip_2d.join(
                    ip_1d.select("legislator_slug", *extra_cols),
                    on="legislator_slug",
                    how="left",
                )

            return ip_2d.with_columns(pl.lit("2d_dim1").alias("source")), "2d_dim1", metadata
        else:
            reason = "2d_unavailable" if ip_2d is None else "2d_convergence_failed"
            print(f"  → Falling back to 1D IRT ({reason})")
            metadata["reason"] = f"horseshoe_detected_but_{reason}"
            return (
                ip_1d.with_columns(pl.lit("1d_irt").alias("source")),
                "1d_irt",
                metadata,
            )
    else:
        print(f"  No horseshoe in {chamber} — using 1D IRT")
        metadata["reason"] = "no_horseshoe"
        return ip_1d.with_columns(pl.lit("1d_irt").alias("source")), "1d_irt", metadata


def write_canonical_ideal_points(
    irt_1d_dir: Path,
    irt_2d_dir: Path,
    output_dir: Path,
    chambers: list[str] | None = None,
) -> dict[str, str]:
    """Write canonical ideal points for all chambers.

    Args:
        irt_1d_dir: Phase 05 run directory (contains data/ideal_points_{chamber}.parquet)
        irt_2d_dir: Phase 06 run directory (contains data/ideal_points_2d_{chamber}.parquet)
        output_dir: Where to write canonical output (typically {run_dir}/canonical_irt/)
        chambers: List of chambers to process. Defaults to ["House", "Senate"].

    Returns dict mapping chamber → source_label ("1d_irt" or "2d_dim1").
    """
    import json

    if chambers is None:
        chambers = ["House", "Senate"]

    output_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, str] = {}
    all_metadata: dict[str, dict] = {}

    for chamber in chambers:
        try:
            ip, source, metadata = route_canonical_ideal_points(irt_1d_dir, irt_2d_dir, chamber)
            ip.write_parquet(output_dir / f"canonical_ideal_points_{chamber.lower()}.parquet")
            sources[chamber] = source
            all_metadata[chamber] = metadata
            print(f"  Saved: canonical_ideal_points_{chamber.lower()}.parquet (source: {source})")
        except FileNotFoundError as e:
            print(f"  Skipping {chamber}: {e}")

    # Write routing manifest
    manifest = {
        "sources": sources,
        "metadata": all_metadata,
        "thresholds": {
            "horseshoe_dem_wrong_side_frac": HORSESHOE_DEM_WRONG_SIDE_FRAC,
            "horseshoe_overlap_frac": HORSESHOE_OVERLAP_FRAC,
            "dim1_rhat_threshold": DIM1_RHAT_THRESHOLD,
            "dim1_ess_threshold": DIM1_ESS_THRESHOLD,
        },
    }
    manifest_path = output_dir / "routing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print("  Saved: routing_manifest.json")

    return sources
