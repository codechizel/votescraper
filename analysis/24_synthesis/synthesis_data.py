"""
Synthesis data loading and joining — pure I/O and DataFrame operations.

Loads parquets and manifests from 10 upstream phases, joins them into unified
per-chamber legislator DataFrames. Reused by profiles and cross-session phases.
"""

import json
from pathlib import Path

import polars as pl

from analysis.run_context import resolve_upstream_dir

UPSTREAM_PHASES = [
    "01_eda",
    "02_pca",
    "05_irt",
    "09_clustering",
    "11_network",
    "15_prediction",
    "13_indices",
    "04_umap",
    "14_beta_binomial",
    "07_hierarchical",
]


def _read_parquet_safe(path: Path) -> pl.DataFrame | None:
    """Read a parquet file, returning None if it doesn't exist."""
    if path.exists():
        return pl.read_parquet(path)
    print(f"  WARNING: missing {path}")
    return None


def _read_manifest(path: Path) -> dict:
    """Read a JSON manifest, returning empty dict if missing."""
    if path.exists():
        return json.loads(path.read_text())
    print(f"  WARNING: missing manifest {path}")
    return {}


def _load_canonical_irt(
    results_base: Path, run_id: str | None, chamber: str
) -> pl.DataFrame | None:
    """Load canonical ideal points from Phase 06's routing output.

    Canonical routing selects 1D IRT or 2D Dim 1 per chamber based on horseshoe
    detection. When available, this supersedes raw Phase 05 output.
    See docs/canonical-ideal-points.md.
    """
    irt_2d_dir = resolve_upstream_dir("06_irt_2d", results_base, run_id)
    canonical_path = irt_2d_dir / "canonical_irt" / f"canonical_ideal_points_{chamber}.parquet"
    if canonical_path.exists():
        df = pl.read_parquet(canonical_path)
        source = df["source"][0] if "source" in df.columns else "unknown"
        print(f"  Canonical IRT ({chamber}): loaded from {source}")
        return df
    return None


def load_all_upstream(results_base: Path, run_id: str | None = None) -> dict:
    """Read all parquets and manifests from the 10 upstream phases.

    When run_id is set, reads from results_base/{run_id}/{phase}/ instead of
    results_base/{phase}/latest/. Falls back to results_base/latest/{phase}
    for new-layout sessions.

    Returns a dict with keys: manifests, and per-chamber parquet DataFrames.
    """
    upstream: dict = {"manifests": {}, "house": {}, "senate": {}, "plots": {}}

    for phase in UPSTREAM_PHASES:
        phase_dir = resolve_upstream_dir(phase, results_base, run_id)
        data_dir = phase_dir / "data"
        plots_dir = phase_dir / "plots"

        # Manifests
        manifest_path = phase_dir / "filtering_manifest.json"
        upstream["manifests"][phase] = _read_manifest(manifest_path)

        # Per-chamber parquets
        for chamber in ("house", "senate"):
            if phase == "05_irt":
                # Prefer canonical routing (from Phase 06) over raw 1D IRT
                canonical_df = _load_canonical_irt(results_base, run_id, chamber)
                if canonical_df is not None:
                    upstream[chamber]["irt"] = canonical_df
                else:
                    df = _read_parquet_safe(data_dir / f"ideal_points_{chamber}.parquet")
                    if df is not None:
                        upstream[chamber]["irt"] = df
            elif phase == "13_indices":
                df = _read_parquet_safe(data_dir / f"maverick_scores_{chamber}.parquet")
                if df is not None:
                    upstream[chamber]["maverick"] = df
            elif phase == "11_network":
                df = _read_parquet_safe(data_dir / f"centrality_{chamber}.parquet")
                if df is not None:
                    upstream[chamber]["centrality"] = df
            elif phase == "09_clustering":
                df = _read_parquet_safe(data_dir / f"kmeans_assignments_{chamber}.parquet")
                if df is not None:
                    upstream[chamber]["kmeans"] = df
                df2 = _read_parquet_safe(data_dir / f"party_loyalty_{chamber}.parquet")
                if df2 is not None:
                    upstream[chamber]["loyalty"] = df2
            elif phase == "15_prediction":
                df = _read_parquet_safe(data_dir / f"per_legislator_accuracy_{chamber}.parquet")
                if df is not None:
                    upstream[chamber]["accuracy"] = df
                sv = _read_parquet_safe(data_dir / f"surprising_votes_{chamber}.parquet")
                if sv is not None:
                    upstream[chamber]["surprising_votes"] = sv
                hr = _read_parquet_safe(data_dir / f"holdout_results_{chamber}.parquet")
                if hr is not None:
                    upstream[chamber]["holdout_results"] = hr
            elif phase == "02_pca":
                df = _read_parquet_safe(data_dir / f"pc_scores_{chamber}.parquet")
                if df is not None:
                    upstream[chamber]["pca"] = df
            elif phase == "04_umap":
                df = _read_parquet_safe(data_dir / f"umap_embedding_{chamber}.parquet")
                if df is not None:
                    upstream[chamber]["umap"] = df
            elif phase == "14_beta_binomial":
                df = _read_parquet_safe(data_dir / f"posterior_loyalty_{chamber}.parquet")
                if df is not None:
                    upstream[chamber]["beta_posterior"] = df
            elif phase == "07_hierarchical":
                df = _read_parquet_safe(data_dir / f"hierarchical_ideal_points_{chamber}.parquet")
                if df is not None:
                    upstream[chamber]["hierarchical"] = df

        # Track upstream plot paths
        upstream["plots"][phase] = plots_dir

    return upstream


def build_legislator_df(upstream: dict, chamber: str) -> pl.DataFrame | None:
    """Join upstream parquets into a unified legislator DataFrame for one chamber.

    Base table: IRT ideal points. All other tables LEFT JOIN on legislator_slug.
    Returns None if IRT ideal points are not available for this chamber.
    """
    base = upstream[chamber].get("irt")
    if base is None:
        return None

    df = base.select(
        "legislator_slug", "xi_mean", "xi_sd", "full_name", "party", "district", "chamber"
    )

    # Maverick scores (indices)
    mav = upstream[chamber].get("maverick")
    if mav is not None:
        df = df.join(
            mav.select(
                "legislator_slug",
                "unity_score",
                "maverick_rate",
                "weighted_maverick",
                "n_defections",
                "loyalty_zscore",
            ),
            on="legislator_slug",
            how="left",
        )

    # Network centrality
    cent = upstream[chamber].get("centrality")
    if cent is not None:
        df = df.join(
            cent.select(
                [
                    c
                    for c in [
                        "legislator_slug",
                        "betweenness",
                        "eigenvector",
                        "pagerank",
                        "harmonic",
                        "cross_party_fraction",
                    ]
                    if c in cent.columns
                ]
            ),
            on="legislator_slug",
            how="left",
        )

    # Clustering assignments (optimal k varies by session)
    km = upstream[chamber].get("kmeans")
    if km is not None:
        cluster_cols = [
            c for c in km.columns if c.startswith("cluster_k") and not c.startswith("cluster_2d")
        ]
        select_cols = ["legislator_slug", "distance_to_centroid"] + cluster_cols
        select_cols = [c for c in select_cols if c in km.columns]
        if select_cols:
            df = df.join(km.select(select_cols), on="legislator_slug", how="left")

    # Per-legislator prediction accuracy
    acc = upstream[chamber].get("accuracy")
    if acc is not None:
        df = df.join(
            acc.select("legislator_slug", "accuracy", "n_votes", "n_correct"),
            on="legislator_slug",
            how="left",
        )

    # PCA scores
    pca = upstream[chamber].get("pca")
    if pca is not None:
        df = df.join(
            pca.select("legislator_slug", "PC1", "PC2"),
            on="legislator_slug",
            how="left",
        )

    # UMAP coordinates
    umap = upstream[chamber].get("umap")
    if umap is not None:
        df = df.join(
            umap.select("legislator_slug", "UMAP1", "UMAP2"),
            on="legislator_slug",
            how="left",
        )

    # Clustering party loyalty
    loy = upstream[chamber].get("loyalty")
    if loy is not None:
        df = df.join(
            loy.select("legislator_slug", "loyalty_rate"),
            on="legislator_slug",
            how="left",
        )

    # Beta-Binomial posterior loyalty
    bp = upstream[chamber].get("beta_posterior")
    if bp is not None:
        df = df.join(
            bp.select("legislator_slug", "posterior_mean", "ci_width", "shrinkage"),
            on="legislator_slug",
            how="left",
        )

    # Hierarchical IRT ideal points
    hier = upstream[chamber].get("hierarchical")
    if hier is not None:
        hier_cols = ["legislator_slug"]
        rename_map: dict[str, str] = {}
        for col in ["xi_mean", "xi_sd", "shrinkage_pct"]:
            if col in hier.columns:
                hier_cols.append(col)
                rename_map[col] = f"hier_{col}"
        df = df.join(
            hier.select(hier_cols).rename(rename_map),
            on="legislator_slug",
            how="left",
        )

    # Add percentile ranks (within chamber, 0-1 scale)
    n = df.height
    for col, ascending in [
        ("xi_mean", True),
        ("betweenness", True),
        ("accuracy", True),
    ]:
        if col in df.columns:
            df = df.with_columns((pl.col(col).rank("ordinal") / n).alias(f"{col}_percentile"))

    return df.sort("xi_mean")
