"""Canonical ideal point routing — auto-select best IRT source per chamber.

Three-layer routing architecture:
1. **Manual PCA overrides** (pca_overrides.yaml): human-vetted dimension assignments
   for 8 problematic supermajority Senate sessions where unsupervised methods
   capture intra-party factionalism rather than the party divide.
2. **Horseshoe detection**: identifies chambers where 1D IRT conflates ideology
   with establishment-loyalty. Prefers Hierarchical 2D Dim 1 (Phase 07b),
   falling back to flat 2D Dim 1 (Phase 06), then 1D IRT (Phase 05).
3. **Tiered convergence**: validates 2D model quality before using it.

For balanced chambers without overrides, 1D IRT remains canonical.

This module runs after Phase 06 (2D IRT) or Phase 07b (Hierarchical 2D IRT)
and writes a canonical output that downstream phases (synthesis, profiles,
cross-session) consume instead of reading Phase 05 directly.

See docs/canonical-ideal-points.md, ADR-0109, ADR-0110, ADR-0117.
W-NOMINATE gate (ADR-0123) removed — see ADR-0126.

Usage (called from irt_2d.py or hierarchical_2d.py):
    from analysis.canonical_ideal_points import write_canonical_ideal_points
    write_canonical_ideal_points(irt_1d_dir, irt_2d_dir, output_dir, h2d_dir=h2d_dir)
"""

from pathlib import Path

import polars as pl

# Horseshoe detection threshold (must match Phase 05)
HORSESHOE_DEM_WRONG_SIDE_FRAC = 0.20
HORSESHOE_OVERLAP_FRAC = 0.30

# ── Tiered convergence quality gate (ADR-0110) ──────────────────────────────
# Tier 1: full convergence — use 2D Dim 1 with trusted HDIs
TIER1_RHAT_THRESHOLD = 1.10
TIER1_ESS_THRESHOLD = 100

# Tier 2: point estimates credible — use 2D Dim 1 but flag wide HDIs
TIER2_RHAT_THRESHOLD = 2.50
TIER2_RANK_CORR_THRESHOLD = 0.70  # legacy: kept for backward compat, no longer gates
TIER2_PARTY_D_THRESHOLD = 1.5  # party separation (Cohen's d) for credible ideology estimate

# Tier 3: failed — fall back to 1D (implicit: anything beyond Tier 2)

# Legacy aliases for backward compatibility in tests
DIM1_RHAT_THRESHOLD = TIER1_RHAT_THRESHOLD
DIM1_ESS_THRESHOLD = TIER1_ESS_THRESHOLD


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


def _compute_party_d(ip_df: pl.DataFrame) -> float:
    """Compute Cohen's d for party separation on ideal points."""
    import numpy as np

    r_ip = ip_df.filter(pl.col("party") == "Republican")["xi_mean"].to_numpy()
    d_ip = ip_df.filter(pl.col("party") == "Democrat")["xi_mean"].to_numpy()
    if len(r_ip) == 0 or len(d_ip) == 0:
        return 0.0
    pooled_sd = np.sqrt((r_ip.std() ** 2 + d_ip.std() ** 2) / 2)
    if pooled_sd == 0:
        return 0.0
    return float(abs(r_ip.mean() - d_ip.mean()) / pooled_sd)


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


def load_pca_scores(pca_dir: Path, chamber: str) -> pl.DataFrame | None:
    """Load PCA scores from Phase 02 data directory.

    Returns DataFrame with legislator_slug and PC1 columns, or None if unavailable.
    """
    path = pca_dir / "data" / f"pca_scores_{chamber.lower()}.parquet"
    if not path.exists():
        # Try alternative naming
        path = pca_dir / "data" / f"scores_{chamber.lower()}.parquet"
    if not path.exists():
        return None
    df = pl.read_parquet(path)
    # Select only what we need — PC1 column name varies
    pc1_col = None
    for candidate in ("PC1", "pc1", "PC_1"):
        if candidate in df.columns:
            pc1_col = candidate
            break
    if pc1_col is None:
        return None
    return df.select("legislator_slug", pl.col(pc1_col).alias("PC1"))


def _compute_party_separation(ip: pl.DataFrame) -> float:
    """Compute Cohen's d between Republican and Democrat mean ideal points.

    Uses the 'xi_mean' column and 'party' column. Returns 0.0 if either party
    is missing. This breaks the circular PCA dependency in Tier 2 — validates
    that the axis actually separates parties, regardless of which PCA component
    it correlates with. See docs/pca-ideology-axis-instability.md (R3).
    """
    r = ip.filter(pl.col("party") == "Republican")
    d = ip.filter(pl.col("party") == "Democrat")
    if r.height == 0 or d.height == 0:
        return 0.0

    import numpy as np

    r_vals = r["xi_mean"].to_numpy().astype(np.float64)
    d_vals = d["xi_mean"].to_numpy().astype(np.float64)
    pooled_sd = np.sqrt((r_vals.std() ** 2 + d_vals.std() ** 2) / 2)
    if pooled_sd == 0:
        return 0.0
    return float(abs(r_vals.mean() - d_vals.mean()) / pooled_sd)


def _compute_rank_correlation(ip_2d: pl.DataFrame, pca_scores: pl.DataFrame) -> float | None:
    """Compute Spearman rank correlation between 2D Dim 1 and PCA PC1.

    Returns |ρ| (absolute value) or None if insufficient overlap.
    """
    from scipy.stats import spearmanr

    merged = ip_2d.join(pca_scores, on="legislator_slug", how="inner")
    if merged.height < 5:
        return None

    rho, _ = spearmanr(merged["xi_mean"].to_numpy(), merged["PC1"].to_numpy())
    return abs(float(rho))


def assess_2d_convergence_tier(
    irt_2d_dir: Path,
    chamber: str,
    ip_2d: pl.DataFrame | None = None,
    pca_dir: Path | None = None,
) -> dict:
    """Assess 2D model convergence using a three-tier quality gate (ADR-0110).

    Returns dict with:
        tier: 1, 2, or 3
        usable: True if tier 1 or 2 (2D Dim 1 can be used as canonical)
        xi_rhat: R-hat value
        xi_ess: ESS value
        rank_corr: Spearman |ρ| with PCA PC1 (Tier 2 only, None otherwise)
        reason: human-readable explanation
    """
    import json

    result: dict = {
        "tier": 3,
        "usable": False,
        "xi_rhat": 999.0,
        "xi_ess": 0.0,
        "rank_corr": None,
        "reason": "",
    }

    summary_path = irt_2d_dir / "data" / "convergence_summary.json"
    if not summary_path.exists():
        result["reason"] = "no convergence summary found"
        print(f"  WARNING: No convergence summary found at {summary_path}")
        return result

    summary = json.loads(summary_path.read_text())
    chamber_data = summary.get("chambers", {}).get(chamber, {})
    convergence = chamber_data.get("convergence", {})

    xi_rhat = convergence.get("xi_rhat_max", 999.0)
    xi_ess = convergence.get("xi_ess_min", 0.0)
    result["xi_rhat"] = xi_rhat
    result["xi_ess"] = xi_ess

    # Tier 1: full convergence
    if xi_rhat < TIER1_RHAT_THRESHOLD and xi_ess > TIER1_ESS_THRESHOLD:
        result["tier"] = 1
        result["usable"] = True
        result["reason"] = f"converged (R-hat={xi_rhat:.4f}, ESS={xi_ess:.0f})"
        print(f"  Tier 1 (converged): {chamber} R-hat={xi_rhat:.4f}, ESS={xi_ess:.0f}")
        return result

    # Tier 2: point estimates credible (R-hat below catastrophic + party separation OK)
    # Uses party separation (Cohen's d) instead of PCA rank correlation to avoid
    # circular dependency when PCA PC1 ≠ ideology. See docs/pca-ideology-axis-instability.md.
    if xi_rhat < TIER2_RHAT_THRESHOLD:
        party_d = None
        if ip_2d is not None:
            party_d = _compute_party_separation(ip_2d)
            result["party_separation_d"] = party_d

        # Also compute PCA rank correlation as secondary diagnostic (logged, not gating)
        rank_corr = None
        if ip_2d is not None and pca_dir is not None:
            pca_scores = load_pca_scores(pca_dir, chamber)
            if pca_scores is not None:
                rank_corr = _compute_rank_correlation(ip_2d, pca_scores)
                result["rank_corr"] = rank_corr

        if party_d is not None and party_d > TIER2_PARTY_D_THRESHOLD:
            result["tier"] = 2
            result["usable"] = True
            corr_note = f", ρ(Dim1,PC1)={rank_corr:.3f}" if rank_corr is not None else ""
            result["reason"] = (
                f"point estimates credible (R-hat={xi_rhat:.4f}, ESS={xi_ess:.0f}, "
                f"party d={party_d:.2f}{corr_note})"
            )
            print(
                f"  Tier 2 (point estimates credible): {chamber} "
                f"R-hat={xi_rhat:.4f}, party d={party_d:.2f}"
            )
            return result
        elif party_d is not None:
            result["reason"] = (
                f"party separation too low (R-hat={xi_rhat:.4f}, "
                f"d={party_d:.2f} < {TIER2_PARTY_D_THRESHOLD})"
            )
        else:
            result["reason"] = (
                f"no ideal points available for party separation check "
                f"(R-hat={xi_rhat:.4f}, ESS={xi_ess:.0f})"
            )

    else:
        result["reason"] = f"R-hat too high ({xi_rhat:.4f} ≥ {TIER2_RHAT_THRESHOLD})"

    # Tier 3: fall back
    print(f"  Tier 3 (convergence failed): {chamber} — {result['reason']}")
    return result


# Keep old name for backward compatibility (tests may reference it)
def check_2d_convergence_quality(irt_2d_dir: Path, chamber: str) -> bool:
    """Legacy wrapper — returns True if tier 1 or 2.

    Prefer assess_2d_convergence_tier() for new code.
    """
    tier_result = assess_2d_convergence_tier(irt_2d_dir, chamber)
    return tier_result["usable"]


def load_h2d_dim1_ideal_points(h2d_dir: Path, chamber: str) -> pl.DataFrame | None:
    """Load Hierarchical 2D IRT Dim 1 ideal points from Phase 07b data directory.

    Maps H2D column names to the standard schema:
    xi_dim1_mean → xi_mean, xi_dim1_hdi_3% → xi_hdi_2.5, xi_dim1_hdi_97% → xi_hdi_97.5
    """
    path = h2d_dir / "data" / f"ideal_points_h2d_{chamber.lower()}.parquet"
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


def load_dim2_ideal_points(
    data_dir: Path,
    chamber: str,
    filename_prefix: str,
) -> pl.DataFrame | None:
    """Load Dim 2 ideal points from a 2D IRT phase, mapped to standard schema.

    Works for both flat 2D (``ideal_points_2d_``) and hierarchical 2D
    (``ideal_points_h2d_``).  Maps ``xi_dim2_mean → xi_mean``, etc.
    """
    path = data_dir / "data" / f"{filename_prefix}{chamber.lower()}.parquet"
    if not path.exists():
        return None

    df = pl.read_parquet(path)
    if "xi_dim2_mean" not in df.columns:
        return None

    df = df.with_columns(
        ((pl.col("xi_dim2_hdi_97%") - pl.col("xi_dim2_hdi_3%")) / 3.92).alias("xi_sd")
    )

    return df.select(
        "legislator_slug",
        "full_name",
        "party",
        pl.col("xi_dim2_mean").alias("xi_mean"),
        "xi_sd",
        pl.col("xi_dim2_hdi_3%").alias("xi_hdi_2.5"),
        pl.col("xi_dim2_hdi_97%").alias("xi_hdi_97.5"),
    )


def route_canonical_ideal_points(
    irt_1d_dir: Path,
    irt_2d_dir: Path,
    chamber: str,
    pca_dir: Path | None = None,
    h2d_dir: Path | None = None,
    session: str | None = None,
) -> tuple[pl.DataFrame, str, dict]:
    """Determine the canonical ideal points for a chamber.

    Args:
        irt_1d_dir: Phase 05 run directory.
        irt_2d_dir: Phase 06 run directory.
        chamber: "House" or "Senate".
        pca_dir: Phase 02 run directory (for Tier 2 rank correlation check).
        h2d_dir: Phase 07b run directory (optional, preferred over flat 2D when converged).
        session: Canonical session string (for PCA override lookup).

    Returns (ideal_points_df, source_label, routing_metadata).

    source_label is "hierarchical_2d_dim1", "hierarchical_2d_dim2", "2d_dim1",
    "2d_dim2", or "1d_irt".
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

    # ── Select initial canonical source via existing routing logic ──────────
    ip: pl.DataFrame
    source: str

    if horseshoe["detected"]:
        print(f"  Horseshoe DETECTED in {chamber}: {horseshoe['reason']}")

        # Try Hierarchical 2D first (preferred: combines party pooling + 2D structure)
        ip_h2d: pl.DataFrame | None = None
        if h2d_dir is not None:
            ip_h2d = load_h2d_dim1_ideal_points(h2d_dir, chamber)
            if ip_h2d is not None:
                h2d_tier = assess_2d_convergence_tier(
                    h2d_dir,
                    chamber,
                    ip_2d=ip_h2d,
                    pca_dir=pca_dir,
                )
                metadata["h2d_convergence_tier"] = h2d_tier

                if h2d_tier["usable"]:
                    tier_label = (
                        "converged" if h2d_tier["tier"] == 1 else "point estimates credible"
                    )
                    print(f"  → Initial selection: Hierarchical 2D Dim 1 ({tier_label})")
                    metadata["reason"] = f"horseshoe_detected: {horseshoe['reason']}"
                    ip, source = ip_h2d, "hierarchical_2d_dim1"
                else:
                    print("  H2D convergence insufficient — trying flat 2D")
                    ip_h2d = None  # mark unusable

        # Fall back to flat 2D if H2D not selected
        selected_2d = horseshoe["detected"] and ip_h2d is not None
        if horseshoe["detected"] and not selected_2d and ip_2d is not None:
            tier_result = assess_2d_convergence_tier(
                irt_2d_dir,
                chamber,
                ip_2d=ip_2d,
                pca_dir=pca_dir,
            )
            metadata["convergence_tier"] = tier_result

            if tier_result["usable"]:
                tier_label = "converged" if tier_result["tier"] == 1 else "point estimates credible"
                print(f"  → Initial selection: 2D Dim 1 ({tier_label})")
                metadata["reason"] = f"horseshoe_detected: {horseshoe['reason']}"
                ip, source = ip_2d, "2d_dim1"
                selected_2d = True

        # Final fallback: 1D
        if horseshoe["detected"] and not selected_2d:
            reason = "2d_unavailable" if ip_2d is None else "2d_convergence_failed"
            print(f"  → Falling back to 1D IRT ({reason})")
            metadata["reason"] = f"horseshoe_detected_but_{reason}"
            ip, source = ip_1d, "1d_irt"
    else:
        print(f"  No horseshoe in {chamber} — using 1D IRT")
        metadata["reason"] = "no_horseshoe"
        ip, source = ip_1d, "1d_irt"

    # ── PCA override: prefer Dim 2 when PC2 is the ideology axis ──────────
    from analysis.init_strategy import load_pca_override

    override_pc = load_pca_override(session, chamber)
    if override_pc == "PC2" and source.endswith("_dim1"):
        dim2_label = source.replace("dim1", "dim2")
        dim2_ip: pl.DataFrame | None = None
        if source == "hierarchical_2d_dim1" and h2d_dir is not None:
            dim2_ip = load_dim2_ideal_points(h2d_dir, chamber, "ideal_points_h2d_")
        elif source == "2d_dim1":
            dim2_ip = load_dim2_ideal_points(irt_2d_dir, chamber, "ideal_points_2d_")
        if dim2_ip is not None:
            dim2_d = _compute_party_d(dim2_ip)
            dim1_d = _compute_party_d(ip)
            if dim2_d > dim1_d:
                print(
                    f"  → PCA override: {dim2_label} (party d={dim2_d:.2f} > "
                    f"dim1 d={dim1_d:.2f})"
                )
                ip, source = dim2_ip, dim2_label
                metadata["pca_override"] = {
                    "override_pc": override_pc,
                    "dim1_party_d": dim1_d,
                    "dim2_party_d": dim2_d,
                    "swapped_to": dim2_label,
                }
            else:
                metadata["pca_override"] = {
                    "override_pc": override_pc,
                    "dim1_party_d": dim1_d,
                    "dim2_party_d": dim2_d,
                    "kept": source,
                    "reason": "dim1 has better party separation despite PC2 override",
                }

    # ── Add extra columns from 1D and finalize ────────────────────────────
    extra_cols = []
    for col in ("district", "chamber"):
        if col in ip_1d.columns and col not in ip.columns:
            extra_cols.append(col)
    if extra_cols:
        ip = ip.join(
            ip_1d.select("legislator_slug", *extra_cols),
            on="legislator_slug",
            how="left",
        )

    return ip.with_columns(pl.lit(source).alias("source")), source, metadata


def write_canonical_ideal_points(
    irt_1d_dir: Path,
    irt_2d_dir: Path,
    output_dir: Path,
    chambers: list[str] | None = None,
    pca_dir: Path | None = None,
    h2d_dir: Path | None = None,
    session: str | None = None,
) -> dict[str, str]:
    """Write canonical ideal points for all chambers.

    Args:
        irt_1d_dir: Phase 05 run directory (contains data/ideal_points_{chamber}.parquet)
        irt_2d_dir: Phase 06 run directory (contains data/ideal_points_2d_{chamber}.parquet)
        output_dir: Where to write canonical output (typically {run_dir}/canonical_irt/)
        chambers: List of chambers to process. Defaults to ["House", "Senate"].
        pca_dir: Phase 02 run directory (for Tier 2 rank correlation check). Optional.
        h2d_dir: Phase 07b run directory (contains data/ideal_points_h2d_{chamber}.parquet).
            Optional. When available and converged, preferred over flat 2D for horseshoe
            chambers.
        session: Canonical session string for PCA override lookup. Optional.

    Returns dict mapping chamber → source_label ("1d_irt", "2d_dim1",
    "2d_dim2", "hierarchical_2d_dim1", or "hierarchical_2d_dim2").
    """
    import json

    if chambers is None:
        chambers = ["House", "Senate"]

    output_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, str] = {}
    all_metadata: dict[str, dict] = {}

    for chamber in chambers:
        try:
            ip, source, metadata = route_canonical_ideal_points(
                irt_1d_dir,
                irt_2d_dir,
                chamber,
                pca_dir=pca_dir,
                h2d_dir=h2d_dir,
                session=session,
            )
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
            "tier1_rhat_threshold": TIER1_RHAT_THRESHOLD,
            "tier1_ess_threshold": TIER1_ESS_THRESHOLD,
            "tier2_rhat_threshold": TIER2_RHAT_THRESHOLD,
            "tier2_rank_corr_threshold": TIER2_RANK_CORR_THRESHOLD,
        },
    }
    manifest_path = output_dir / "routing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print("  Saved: routing_manifest.json")

    return sources
