"""Pure data logic for cross-session validation.

Matching, IRT scale alignment, ideology shift metrics, and metric stability
comparison. No I/O, no plotting — all functions take DataFrames in, return
DataFrames or dicts out.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np
import polars as pl
from scipy import stats

try:
    from analysis.phase_utils import normalize_name
except ModuleNotFoundError:
    from phase_utils import normalize_name  # type: ignore[no-redef]

# ── Constants ────────────────────────────────────────────────────────────────

MIN_OVERLAP: int = 20
"""Minimum returning legislators for meaningful comparison."""

SHIFT_THRESHOLD_SD: float = 1.0
"""Flag legislators who moved > this many SDs as significant movers."""

ALIGNMENT_TRIM_PCT: int = 10
"""Trim this % of extreme residuals from affine fit for robustness."""

CORRELATION_WARN: float = 0.70
"""Warn if cross-session Pearson r falls below this value."""

FEATURE_IMPORTANCE_TOP_K: int = 10
"""Compare top K SHAP features across sessions."""

PREDICTION_META_COLS: list[str] = ["legislator_slug", "vote_id", "vote_binary"]
"""Columns to exclude from feature sets during prediction."""

STABILITY_METRICS: list[str] = [
    "unity_score",
    "maverick_rate",
    "weighted_maverick",
    "betweenness",
    "eigenvector",
    "pagerank",
    "loyalty_rate",
    "PC1",
]
"""Metrics to compare across sessions in the stability analysis."""

SIGN_ARBITRARY_METRICS: set[str] = {"PC1"}
"""Metrics whose sign is conventional (orient_pc1 normalizes, but edge cases can flip).
Correlations use abs() so a sign flip doesn't masquerade as instability."""

# ── Legislator Matching ──────────────────────────────────────────────────────


def _normalize_slug_col(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure the slug column is named ``legislator_slug``."""
    if "slug" in df.columns and "legislator_slug" not in df.columns:
        return df.rename({"slug": "legislator_slug"})
    return df


def _add_name_norm(df: pl.DataFrame) -> pl.DataFrame:
    """Add a ``name_norm`` column by applying :func:`normalize_name`."""
    return df.with_columns(
        pl.col("full_name").map_elements(normalize_name, return_dtype=pl.Utf8).alias("name_norm")
    )


def _has_ocd_ids(df: pl.DataFrame) -> bool:
    """Check if a DataFrame has non-empty OCD IDs for matching."""
    if "ocd_id" not in df.columns:
        return False
    non_empty = df.filter(pl.col("ocd_id") != "").height
    return non_empty > 0


def match_legislators(
    leg_a: pl.DataFrame,
    leg_b: pl.DataFrame,
    *,
    fuzzy_threshold: float | None = None,
) -> pl.DataFrame:
    """Match legislators across sessions, preferring OCD ID over name.

    Three-phase matching:
      0. OCD ID join (when both sessions have ocd_id columns with data)
      1. Name-norm join on unmatched remainder
      2. Optional fuzzy matching on still-unmatched names

    Args:
        leg_a: Legislators from session A (needs ``full_name``,
            ``slug`` or ``legislator_slug``, ``party``, ``chamber``,
            ``district``).  Optional: ``ocd_id``.
        leg_b: Legislators from session B (same columns).
        fuzzy_threshold: If set, unmatched names go through a second pass
            with :func:`fuzzy_match_legislators` at this similarity
            threshold.  Matched fuzzy pairs are appended to the result.

    Returns:
        DataFrame with columns: ``name_norm``, ``full_name_a``,
        ``full_name_b``, ``slug_a``, ``slug_b``, ``party_a``, ``party_b``,
        ``chamber_a``, ``chamber_b``, ``district_a``, ``district_b``,
        ``is_chamber_switch``, ``is_party_switch``.

    Raises:
        ValueError: If fewer than :data:`MIN_OVERLAP` legislators match.
    """
    a = _add_name_norm(_normalize_slug_col(leg_a))
    b = _add_name_norm(_normalize_slug_col(leg_b))

    a_sel = a.select(
        "name_norm",
        pl.col("full_name").alias("full_name_a"),
        pl.col("legislator_slug").alias("slug_a"),
        pl.col("party").alias("party_a"),
        pl.col("chamber").alias("chamber_a"),
        pl.col("district").alias("district_a"),
        *([pl.col("ocd_id").alias("ocd_id_a")] if "ocd_id" in a.columns else []),
    )
    b_sel = b.select(
        "name_norm",
        pl.col("full_name").alias("full_name_b"),
        pl.col("legislator_slug").alias("slug_b"),
        pl.col("party").alias("party_b"),
        pl.col("chamber").alias("chamber_b"),
        pl.col("district").alias("district_b"),
        *([pl.col("ocd_id").alias("ocd_id_b")] if "ocd_id" in b.columns else []),
    )

    matched: pl.DataFrame | None = None

    # Phase 0: OCD ID join (stable cross-biennium identity)
    ocd_matched_slugs_a: set[str] = set()
    ocd_matched_slugs_b: set[str] = set()
    if _has_ocd_ids(a) and _has_ocd_ids(b):
        ocd_pairs = a_sel.filter(pl.col("ocd_id_a") != "").join(
            b_sel.filter(pl.col("ocd_id_b") != ""),
            left_on="ocd_id_a",
            right_on="ocd_id_b",
            how="inner",
            suffix="_r",
        )
        if ocd_pairs.height > 0:
            # Use session A's name_norm as canonical key
            ocd_matched = ocd_pairs.select(
                pl.col("name_norm"),
                "full_name_a",
                "slug_a",
                "party_a",
                "chamber_a",
                "district_a",
                pl.col("full_name_b"),
                pl.col("slug_b"),
                pl.col("party_b"),
                pl.col("chamber_b"),
                pl.col("district_b"),
            ).with_columns(
                (pl.col("chamber_a") != pl.col("chamber_b")).alias("is_chamber_switch"),
                (pl.col("party_a") != pl.col("party_b")).alias("is_party_switch"),
            )
            matched = ocd_matched
            ocd_matched_slugs_a = set(ocd_matched["slug_a"].to_list())
            ocd_matched_slugs_b = set(ocd_matched["slug_b"].to_list())

    # Phase 1: Name-norm join on remaining unmatched legislators
    a_remaining = a_sel.filter(~pl.col("slug_a").is_in(ocd_matched_slugs_a))
    b_remaining = b_sel.filter(~pl.col("slug_b").is_in(ocd_matched_slugs_b))

    # Drop OCD ID columns before name join
    drop_a = [c for c in a_remaining.columns if c.startswith("ocd_id")]
    drop_b = [c for c in b_remaining.columns if c.startswith("ocd_id")]
    if drop_a:
        a_remaining = a_remaining.drop(drop_a)
    if drop_b:
        b_remaining = b_remaining.drop(drop_b)

    name_matched = a_remaining.join(b_remaining, on="name_norm", how="inner").with_columns(
        (pl.col("chamber_a") != pl.col("chamber_b")).alias("is_chamber_switch"),
        (pl.col("party_a") != pl.col("party_b")).alias("is_party_switch"),
    )
    if name_matched.height > 0:
        if matched is None:
            matched = name_matched
        else:
            matched = pl.concat([matched, name_matched])

    if matched is None:
        matched = name_matched.head(0)  # empty with correct schema

    matched = matched.sort("name_norm")

    # Phase 2: Optional fuzzy second pass
    if fuzzy_threshold is not None:
        all_matched_slugs_a = set(matched["slug_a"].to_list()) if matched.height > 0 else set()
        all_matched_slugs_b = set(matched["slug_b"].to_list()) if matched.height > 0 else set()

        unmatched_a_df = a_remaining.filter(~pl.col("slug_a").is_in(all_matched_slugs_a))
        unmatched_b_df = b_remaining.filter(~pl.col("slug_b").is_in(all_matched_slugs_b))

        if unmatched_a_df.height > 0 and unmatched_b_df.height > 0:
            fuzzy_matches = fuzzy_match_legislators(
                unmatched_a_df["name_norm"].to_list(),
                unmatched_b_df["name_norm"].to_list(),
                threshold=fuzzy_threshold,
            )

            for row in fuzzy_matches.iter_rows(named=True):
                a_row = unmatched_a_df.filter(pl.col("name_norm") == row["name_a"])
                b_row = unmatched_b_df.filter(pl.col("name_norm") == row["name_b"])
                if a_row.height == 1 and b_row.height == 1:
                    # Use session B's name_norm as the canonical key
                    fuzzy_row = pl.DataFrame(
                        {
                            "name_norm": [row["name_b"]],
                            "full_name_a": [a_row["full_name_a"][0]],
                            "slug_a": [a_row["slug_a"][0]],
                            "party_a": [a_row["party_a"][0]],
                            "chamber_a": [a_row["chamber_a"][0]],
                            "district_a": [a_row["district_a"][0]],
                            "full_name_b": [b_row["full_name_b"][0]],
                            "slug_b": [b_row["slug_b"][0]],
                            "party_b": [b_row["party_b"][0]],
                            "chamber_b": [b_row["chamber_b"][0]],
                            "district_b": [b_row["district_b"][0]],
                            "is_chamber_switch": [a_row["chamber_a"][0] != b_row["chamber_b"][0]],
                            "is_party_switch": [a_row["party_a"][0] != b_row["party_b"][0]],
                        }
                    )
                    matched = pl.concat([matched, fuzzy_row]).sort("name_norm")

    if matched.height < MIN_OVERLAP:
        msg = (
            f"Only {matched.height} legislators matched across sessions "
            f"(minimum {MIN_OVERLAP}). Check data quality."
        )
        raise ValueError(msg)

    return matched


def fuzzy_match_legislators(
    unmatched_a: list[str],
    unmatched_b: list[str],
    threshold: float = 0.85,
) -> pl.DataFrame:
    """Find close name matches between two lists using SequenceMatcher.

    Returns a DataFrame of suggested fuzzy matches with similarity scores.
    Only pairs exceeding *threshold* are included. Each name from *a* is
    matched to at most one name from *b* (the best match).

    Args:
        unmatched_a: Normalized names from session A with no exact match.
        unmatched_b: Normalized names from session B with no exact match.
        threshold: Minimum similarity ratio (0–1) to include a match.

    Returns:
        DataFrame with columns: ``name_a``, ``name_b``, ``similarity``.
    """
    if not unmatched_a or not unmatched_b:
        return pl.DataFrame(schema={"name_a": pl.Utf8, "name_b": pl.Utf8, "similarity": pl.Float64})

    rows: list[dict] = []
    used_b: set[str] = set()

    for name_a in unmatched_a:
        best_score = 0.0
        best_name: str | None = None

        for name_b in unmatched_b:
            if name_b in used_b:
                continue
            score = SequenceMatcher(None, name_a, name_b).ratio()
            if score > best_score:
                best_score = score
                best_name = name_b

        if best_name is not None and best_score >= threshold:
            rows.append({"name_a": name_a, "name_b": best_name, "similarity": round(best_score, 4)})
            used_b.add(best_name)

    if not rows:
        return pl.DataFrame(schema={"name_a": pl.Utf8, "name_b": pl.Utf8, "similarity": pl.Float64})

    return pl.DataFrame(rows).sort("similarity", descending=True)


def classify_turnover(
    leg_a: pl.DataFrame,
    leg_b: pl.DataFrame,
    matched: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Classify legislators into returning, departing, and new cohorts.

    Args:
        leg_a: All legislators from session A.
        leg_b: All legislators from session B.
        matched: Output of :func:`match_legislators`.

    Returns:
        ``{"returning": matched, "departing": in A not B, "new": in B not A}``
    """
    a = _normalize_slug_col(leg_a)
    b = _normalize_slug_col(leg_b)

    matched_slugs_a = set(matched["slug_a"].to_list())
    matched_slugs_b = set(matched["slug_b"].to_list())

    departing = a.filter(~pl.col("legislator_slug").is_in(matched_slugs_a))
    new = b.filter(~pl.col("legislator_slug").is_in(matched_slugs_b))

    return {"returning": matched, "departing": departing, "new": new}


# ── IRT Scale Alignment ─────────────────────────────────────────────────────


def align_irt_scales(
    xi_a: pl.DataFrame,
    xi_b: pl.DataFrame,
    matched: pl.DataFrame,
) -> tuple[float, float, pl.DataFrame]:
    """Robust affine alignment of IRT ideal points across sessions.

    Transforms session A onto session B's scale:
    ``xi_a_aligned = A * xi_a + B``.

    Uses overlapping legislators as anchors.  Trims the
    ``ALIGNMENT_TRIM_PCT`` most extreme residuals (genuine movers) before
    the final fit for robustness.

    Args:
        xi_a: IRT ideal points from session A (needs ``legislator_slug``,
            ``xi_mean``, ``full_name``).
        xi_b: IRT ideal points from session B (same columns).
        matched: Output of :func:`match_legislators`.

    Returns:
        ``(A, B, aligned_df)`` where *aligned_df* has columns:
        ``name_norm``, ``slug_a``, ``slug_b``, ``full_name``, ``party``,
        ``chamber``, ``xi_a``, ``xi_b``, ``xi_a_aligned``, ``delta_xi``,
        ``abs_delta_xi``.

    Raises:
        ValueError: If fewer than :data:`MIN_OVERLAP` legislators have IRT
            scores in both sessions.
    """
    pairs = (
        matched.select("name_norm", "slug_a", "slug_b", "party_b", "chamber_b")
        .join(
            xi_a.select(
                pl.col("legislator_slug").alias("slug_a"),
                pl.col("xi_mean").alias("xi_a"),
            ),
            on="slug_a",
            how="inner",
        )
        .join(
            xi_b.select(
                pl.col("legislator_slug").alias("slug_b"),
                pl.col("xi_mean").alias("xi_b"),
                pl.col("full_name"),
            ),
            on="slug_b",
            how="inner",
        )
    )

    if pairs.height < MIN_OVERLAP:
        msg = f"Only {pairs.height} legislators have IRT scores in both sessions"
        raise ValueError(msg)

    x = pairs["xi_a"].to_numpy().astype(np.float64)
    y = pairs["xi_b"].to_numpy().astype(np.float64)

    # Initial OLS fit
    result = stats.linregress(x, y)
    a_init, b_init = float(result.slope), float(result.intercept)

    # Trim extreme residuals (genuine movers distort alignment)
    residuals = y - (a_init * x + b_init)
    abs_residuals = np.abs(residuals)
    cutoff = np.percentile(abs_residuals, 100 - ALIGNMENT_TRIM_PCT)
    keep_mask = abs_residuals <= cutoff

    if np.sum(keep_mask) >= MIN_OVERLAP:
        result_trimmed = stats.linregress(x[keep_mask], y[keep_mask])
        a_final = float(result_trimmed.slope)
        b_final = float(result_trimmed.intercept)
    else:
        a_final, b_final = a_init, b_init

    aligned = (
        pairs.with_columns(
            (pl.col("xi_a") * a_final + b_final).alias("xi_a_aligned"),
            # Strip leadership suffixes (" - President of the Senate" etc.)
            pl.col("full_name").str.replace(r"\s*-\s+.*$", "").alias("full_name"),
        )
        .with_columns(
            (pl.col("xi_b") - pl.col("xi_a_aligned")).alias("delta_xi"),
        )
        .with_columns(
            pl.col("delta_xi").abs().alias("abs_delta_xi"),
        )
        .rename({"party_b": "party", "chamber_b": "chamber"})
    )

    return a_final, b_final, aligned


# ── Shift Analysis ───────────────────────────────────────────────────────────


def compute_ideology_shift(aligned: pl.DataFrame) -> pl.DataFrame:
    """Add shift classification to an aligned DataFrame.

    New columns:
        ``rank_a`` / ``rank_b``: within-group ordinal rank by ideology.
        ``rank_shift``: ``rank_b - rank_a`` (positive = moved rightward in ranking).
        ``is_significant_mover``: ``|delta_xi| > SHIFT_THRESHOLD_SD * std(delta_xi)``.
        ``shift_direction``: ``"leftward"`` / ``"rightward"`` / ``"stable"``.
    """
    delta_std = aligned["delta_xi"].std()
    threshold = (
        SHIFT_THRESHOLD_SD * delta_std if delta_std is not None and delta_std > 0 else float("inf")
    )

    return aligned.with_columns(
        pl.col("xi_a_aligned").rank("ordinal").cast(pl.Int32).alias("rank_a"),
        pl.col("xi_b").rank("ordinal").cast(pl.Int32).alias("rank_b"),
    ).with_columns(
        (pl.col("rank_b") - pl.col("rank_a")).alias("rank_shift"),
        (pl.col("abs_delta_xi") > threshold).alias("is_significant_mover"),
        pl.when(pl.col("delta_xi") > threshold)
        .then(pl.lit("rightward"))
        .when(pl.col("delta_xi") < -threshold)
        .then(pl.lit("leftward"))
        .otherwise(pl.lit("stable"))
        .alias("shift_direction"),
    )


# ── PSI / ICC Helpers ──────────────────────────────────────────────────────


def compute_psi(a: np.ndarray, b: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two distributions.

    Bins *a* and *b* into *n_bins* quantile buckets (derived from *a*),
    then computes:  ``PSI = Σ (p_b[i] - p_a[i]) * ln(p_b[i] / p_a[i])``.

    Interpretation (standard thresholds):
        - < 0.10  →  stable
        - 0.10–0.25  →  investigate
        - > 0.25  →  significant drift

    Returns ``NaN`` if either array has fewer than 2 elements.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if len(a) < 2 or len(b) < 2:
        return float("nan")

    # Quantile-based bin edges from distribution a
    edges = np.quantile(a, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Deduplicate edges (can happen with low-variance data)
    edges = np.unique(edges)

    counts_a = np.histogram(a, bins=edges)[0].astype(np.float64)
    counts_b = np.histogram(b, bins=edges)[0].astype(np.float64)

    # Convert to proportions with floor to avoid log(0)
    eps = 1e-4
    p_a = np.maximum(counts_a / counts_a.sum(), eps)
    p_b = np.maximum(counts_b / counts_b.sum(), eps)

    return float(np.sum((p_b - p_a) * np.log(p_b / p_a)))


def interpret_psi(psi: float) -> str:
    """Return a human-readable PSI interpretation."""
    if np.isnan(psi):
        return "insufficient data"
    if psi < 0.10:
        return "stable"
    if psi <= 0.25:
        return "investigate"
    return "significant drift"


def compute_icc(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(3,1) — two-way mixed, single measures, consistency.

    Formula: ``(MS_row - MS_error) / (MS_row + (k-1) * MS_error)``
    where *k* = 2 (two sessions).

    Interpretation (Koo & Li 2016):
        - < 0.50  →  poor
        - 0.50–0.75  →  moderate
        - 0.75–0.90  →  good
        - > 0.90  →  excellent

    Returns ``NaN`` if fewer than 3 subjects.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    n = len(a)
    if n < 3 or len(b) != n:
        return float("nan")

    k = 2  # two sessions (raters)
    data = np.column_stack([a, b])  # n x k

    grand_mean = data.mean()
    row_means = data.mean(axis=1)

    ss_row = k * np.sum((row_means - grand_mean) ** 2)
    ss_error = np.sum((data - row_means[:, np.newaxis]) ** 2)

    ms_row = ss_row / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_row + (k - 1) * ms_error
    if denom == 0:
        return float("nan")

    return float((ms_row - ms_error) / denom)


def interpret_icc(icc: float) -> str:
    """Return a human-readable ICC interpretation per Koo & Li 2016."""
    if np.isnan(icc):
        return "insufficient data"
    if icc < 0.50:
        return "poor"
    if icc <= 0.75:
        return "moderate"
    if icc <= 0.90:
        return "good"
    return "excellent"


def interpret_stability(rho: float) -> str:
    """Interpret Spearman rho as test-retest reliability (Koo & Li 2016 thresholds)."""
    if np.isnan(rho):
        return "insufficient data"
    rho_abs = abs(rho)
    if rho_abs < 0.50:
        return "poor"
    if rho_abs <= 0.75:
        return "moderate"
    if rho_abs <= 0.90:
        return "good"
    return "excellent"


# ── Metric Stability ────────────────────────────────────────────────────────


def _empty_stability_df() -> pl.DataFrame:
    """Return an empty DataFrame with the metric stability schema."""
    return pl.DataFrame(
        schema={
            "metric": pl.Utf8,
            "pearson_r": pl.Float64,
            "spearman_rho": pl.Float64,
            "n_legislators": pl.Int64,
            "psi": pl.Float64,
            "psi_interpretation": pl.Utf8,
            "icc": pl.Float64,
            "icc_interpretation": pl.Utf8,
            "stability_interpretation": pl.Utf8,
        }
    )


def compute_metric_stability(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    matched: pl.DataFrame,
    metrics: list[str] | None = None,
) -> pl.DataFrame:
    """Compute correlation of metrics across sessions for returning legislators.

    Args:
        df_a: Legislator DataFrame from session A (from ``build_legislator_df``).
        df_b: Legislator DataFrame from session B.
        matched: Output of :func:`match_legislators`.
        metrics: Column names to compare.  Defaults to :data:`STABILITY_METRICS`.

    Returns:
        DataFrame with columns: ``metric``, ``pearson_r``, ``spearman_rho``,
        ``n_legislators``, ``psi``, ``psi_interpretation``, ``icc``,
        ``icc_interpretation``, ``stability_interpretation``.
        Metrics missing from either session are skipped.
    """
    if metrics is None:
        metrics = STABILITY_METRICS

    rows: list[dict] = []

    for metric in metrics:
        if metric not in df_a.columns or metric not in df_b.columns:
            continue

        pairs = (
            matched.select("slug_a", "slug_b")
            .join(
                df_a.select(
                    pl.col("legislator_slug").alias("slug_a"),
                    pl.col(metric).alias("val_a"),
                ),
                on="slug_a",
                how="inner",
            )
            .join(
                df_b.select(
                    pl.col("legislator_slug").alias("slug_b"),
                    pl.col(metric).alias("val_b"),
                ),
                on="slug_b",
                how="inner",
            )
            .drop_nulls(subset=["val_a", "val_b"])
        )

        if pairs.height < 3:
            continue

        va = pairs["val_a"].to_numpy()
        vb = pairs["val_b"].to_numpy()

        pearson_r, _ = stats.pearsonr(va, vb)
        spearman_rho, _ = stats.spearmanr(va, vb)

        # Sign-arbitrary metrics: use abs() so a convention flip isn't misread
        if metric in SIGN_ARBITRARY_METRICS:
            pearson_r = abs(pearson_r)
            spearman_rho = abs(spearman_rho)

        psi_val = compute_psi(va, vb)
        icc_val = compute_icc(va, vb)

        rows.append(
            {
                "metric": metric,
                "pearson_r": round(float(pearson_r), 4),
                "spearman_rho": round(float(spearman_rho), 4),
                "n_legislators": pairs.height,
                "psi": round(psi_val, 4) if not np.isnan(psi_val) else psi_val,
                "psi_interpretation": interpret_psi(psi_val),
                "icc": round(icc_val, 4) if not np.isnan(icc_val) else icc_val,
                "icc_interpretation": interpret_icc(icc_val),
                "stability_interpretation": interpret_stability(float(spearman_rho)),
            }
        )

    if not rows:
        return _empty_stability_df()

    return pl.DataFrame(rows)


@dataclass(frozen=True)
class FreshmenAnalysis:
    """Results of comparing new vs returning legislators in session B."""

    n_new: int
    n_returning: int
    ideology_new_mean: float | None
    ideology_returning_mean: float | None
    ideology_ks_stat: float | None
    ideology_ks_p: float | None
    unity_new_mean: float | None
    unity_returning_mean: float | None
    unity_t_stat: float | None
    unity_t_p: float | None
    maverick_new_mean: float | None
    maverick_returning_mean: float | None
    cohort_df: pl.DataFrame  # per-legislator with is_new flag


def analyze_freshmen_cohort(
    turnover: dict[str, pl.DataFrame],
    leg_df_b: pl.DataFrame,
    irt_b: pl.DataFrame,
) -> FreshmenAnalysis | None:
    """Compare new (freshmen) vs returning legislators in session B.

    Computes ideology distribution comparison (KS test), party unity comparison
    (t-test), and maverick rate comparison for the two cohorts.

    Args:
        turnover: Output of classify_turnover() with "returning" and "new" keys.
        leg_df_b: Session B legislator DataFrame (from build_legislator_df).
        irt_b: IRT ideal points for session B.

    Returns FreshmenAnalysis or None if insufficient data.
    """
    new_slugs = set(turnover["new"]["legislator_slug"].to_list())
    ret_slugs_b = set(turnover["returning"]["slug_b"].to_list())

    if len(new_slugs) < 3 or len(ret_slugs_b) < 3:
        return None

    # Tag each session-B legislator
    cohort_df = leg_df_b.with_columns(
        pl.col("legislator_slug").is_in(new_slugs).alias("is_new"),
    )

    new_df = cohort_df.filter(pl.col("is_new"))
    ret_df = cohort_df.filter(~pl.col("is_new") & pl.col("legislator_slug").is_in(ret_slugs_b))

    # Ideology comparison
    ideology_ks_stat = ideology_ks_p = None
    ideology_new_mean = ideology_ret_mean = None

    new_xi = irt_b.filter(pl.col("legislator_slug").is_in(new_slugs))
    ret_xi = irt_b.filter(pl.col("legislator_slug").is_in(ret_slugs_b))

    if new_xi.height >= 2 and ret_xi.height >= 2:
        xi_new_arr = new_xi["xi_mean"].to_numpy()
        xi_ret_arr = ret_xi["xi_mean"].to_numpy()
        ideology_new_mean = float(np.mean(xi_new_arr))
        ideology_ret_mean = float(np.mean(xi_ret_arr))
        ks_result = stats.ks_2samp(xi_new_arr, xi_ret_arr)
        ideology_ks_stat = float(ks_result.statistic)
        ideology_ks_p = float(ks_result.pvalue)

    # Party unity comparison
    unity_t_stat = unity_t_p = None
    unity_new_mean = unity_ret_mean = None
    if "unity_score" in new_df.columns and "unity_score" in ret_df.columns:
        u_new = new_df["unity_score"].drop_nulls().to_numpy()
        u_ret = ret_df["unity_score"].drop_nulls().to_numpy()
        if len(u_new) >= 2 and len(u_ret) >= 2:
            unity_new_mean = float(np.mean(u_new))
            unity_ret_mean = float(np.mean(u_ret))
            t_result = stats.ttest_ind(u_new, u_ret, equal_var=False)
            unity_t_stat = float(t_result.statistic)
            unity_t_p = float(t_result.pvalue)

    # Maverick rate comparison
    maverick_new_mean = maverick_ret_mean = None
    if "maverick_rate" in new_df.columns and "maverick_rate" in ret_df.columns:
        m_new = new_df["maverick_rate"].drop_nulls().to_numpy()
        m_ret = ret_df["maverick_rate"].drop_nulls().to_numpy()
        if len(m_new) >= 1:
            maverick_new_mean = float(np.mean(m_new))
        if len(m_ret) >= 1:
            maverick_ret_mean = float(np.mean(m_ret))

    return FreshmenAnalysis(
        n_new=len(new_slugs),
        n_returning=len(ret_slugs_b),
        ideology_new_mean=ideology_new_mean,
        ideology_returning_mean=ideology_ret_mean,
        ideology_ks_stat=ideology_ks_stat,
        ideology_ks_p=ideology_ks_p,
        unity_new_mean=unity_new_mean,
        unity_returning_mean=unity_ret_mean,
        unity_t_stat=unity_t_stat,
        unity_t_p=unity_t_p,
        maverick_new_mean=maverick_new_mean,
        maverick_returning_mean=maverick_ret_mean,
        cohort_df=cohort_df,
    )


def compute_bloc_stability(
    km_a: pl.DataFrame,
    km_b: pl.DataFrame,
    matched: pl.DataFrame,
    leg_df_a: pl.DataFrame | None = None,
    leg_df_b: pl.DataFrame | None = None,
) -> dict | None:
    """Track voting bloc (cluster) stability between sessions.

    Computes ARI between cluster assignments, builds a transition matrix,
    and identifies legislators who switched clusters.

    Args:
        km_a: K-means assignments from session A (legislator_slug, cluster).
        km_b: K-means assignments from session B (legislator_slug, cluster).
        matched: Output of match_legislators().
        leg_df_a: Optional session A legislator DataFrame for names/party.
        leg_df_b: Optional session B legislator DataFrame for names/party.

    Returns dict with ARI, transition_matrix, switchers DataFrame, or None.
    """
    from sklearn.metrics import adjusted_rand_score

    # Normalize column names
    km_a = _normalize_slug_col(km_a)
    km_b = _normalize_slug_col(km_b)

    # Find the cluster column dynamically (cluster_k2, cluster_k6, etc.)
    def _find_cluster_col(df: pl.DataFrame) -> str:
        for c in df.columns:
            if c.startswith("cluster_k") and not c.startswith("cluster_2d_"):
                return c
        if "cluster" in df.columns:
            return "cluster"
        msg = f"No cluster column found in {df.columns}"
        raise ValueError(msg)

    cluster_col_a = _find_cluster_col(km_a)
    cluster_col_b = _find_cluster_col(km_b)

    # Join through matched to get cluster labels for same legislators
    pairs = (
        matched.select("slug_a", "slug_b")
        .join(
            km_a.select(
                pl.col("legislator_slug").alias("slug_a"),
                pl.col(cluster_col_a).alias("cluster_a"),
            ),
            on="slug_a",
            how="inner",
        )
        .join(
            km_b.select(
                pl.col("legislator_slug").alias("slug_b"),
                pl.col(cluster_col_b).alias("cluster_b"),
            ),
            on="slug_b",
            how="inner",
        )
        .drop_nulls(subset=["cluster_a", "cluster_b"])
    )

    if pairs.height < 5:
        return None

    labels_a = pairs["cluster_a"].to_numpy()
    labels_b = pairs["cluster_b"].to_numpy()
    ari = float(adjusted_rand_score(labels_a, labels_b))

    # Transition matrix
    transitions: dict[tuple[int, int], int] = {}
    for row in pairs.iter_rows(named=True):
        key = (int(row["cluster_a"]), int(row["cluster_b"]))
        transitions[key] = transitions.get(key, 0) + 1

    # Build transition DataFrame
    trans_rows = [
        {"cluster_a": ca, "cluster_b": cb, "count": cnt}
        for (ca, cb), cnt in sorted(transitions.items())
    ]
    transition_df = pl.DataFrame(trans_rows)

    # Identify switchers (legislators who changed cluster)
    switchers = pairs.filter(pl.col("cluster_a") != pl.col("cluster_b"))
    if leg_df_b is not None:
        name_lookup = leg_df_b.select("legislator_slug", "full_name", "party")
        switchers = switchers.join(
            name_lookup.rename({"legislator_slug": "slug_b"}),
            on="slug_b",
            how="left",
        )

    return {
        "ari": ari,
        "n_paired": pairs.height,
        "transition_df": transition_df,
        "switchers": switchers,
        "pairs": pairs,
    }


def compute_turnover_impact(
    xi_returning: np.ndarray,
    xi_departing: np.ndarray,
    xi_new: np.ndarray,
) -> dict:
    """Compare ideology distributions across turnover cohorts.

    Args:
        xi_returning: IRT ideal points for returning legislators.
        xi_departing: IRT ideal points for departing legislators.
        xi_new: IRT ideal points for new legislators.

    Returns:
        Dict with per-cohort stats (``{cohort}_mean``, ``{cohort}_std``,
        ``{cohort}_n``) and KS test results for departing-vs-returning and
        new-vs-returning comparisons.
    """
    result: dict = {}

    for label, arr in [
        ("returning", xi_returning),
        ("departing", xi_departing),
        ("new", xi_new),
    ]:
        result[f"{label}_mean"] = float(np.mean(arr)) if len(arr) > 0 else None
        result[f"{label}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else None
        result[f"{label}_n"] = len(arr)

    if len(xi_departing) >= 2 and len(xi_returning) >= 2:
        ks_dep, p_dep = stats.ks_2samp(xi_departing, xi_returning)
        result["ks_departing_vs_returning"] = float(ks_dep)
        result["p_departing_vs_returning"] = float(p_dep)

    if len(xi_new) >= 2 and len(xi_returning) >= 2:
        ks_new, p_new = stats.ks_2samp(xi_new, xi_returning)
        result["ks_new_vs_returning"] = float(ks_new)
        result["p_new_vs_returning"] = float(p_new)

    return result


# ── Cross-Session Prediction Helpers ────────────────────────────────────────


def align_feature_columns(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Align feature columns across two vote-feature DataFrames.

    Takes the intersection of feature columns (excluding metadata).
    One-hot vote_type columns may differ between sessions — only shared
    columns are kept.

    Args:
        df_a: Vote features from session A.
        df_b: Vote features from session B.

    Returns:
        ``(df_a_aligned, df_b_aligned, feature_cols)`` where both
        DataFrames have exactly the same columns in the same order.
    """
    meta = set(PREDICTION_META_COLS)
    cols_a = set(df_a.columns) - meta
    cols_b = set(df_b.columns) - meta
    shared = sorted(cols_a & cols_b)

    select_cols = list(PREDICTION_META_COLS) + shared
    # Only select columns that actually exist (some meta cols may be absent)
    select_a = [c for c in select_cols if c in df_a.columns]
    select_b = [c for c in select_cols if c in df_b.columns]

    return df_a.select(select_a), df_b.select(select_b), shared


def standardize_features(
    df: pl.DataFrame,
    numeric_cols: list[str],
) -> pl.DataFrame:
    """Z-score standardize numeric feature columns in-place.

    Binary and one-hot columns (detected by having only 0/1 values)
    are left untouched. Only truly continuous columns are standardized.

    Args:
        df: Vote features DataFrame.
        numeric_cols: Columns to consider for standardization.

    Returns:
        DataFrame with continuous numeric columns z-scored.
    """
    binary_cols = set()
    for col in numeric_cols:
        vals = df[col].drop_nulls()
        if vals.n_unique() <= 2:
            binary_cols.add(col)

    z_exprs = []
    for col in numeric_cols:
        if col in binary_cols:
            continue
        mean = df[col].mean()
        std = df[col].std()
        if std is not None and std > 0:
            z_exprs.append(((pl.col(col) - mean) / std).alias(col))

    if z_exprs:
        return df.with_columns(z_exprs)
    return df


def compare_feature_importance(
    shap_a: np.ndarray,
    shap_b: np.ndarray,
    feature_names: list[str],
    top_k: int | None = None,
) -> tuple[pl.DataFrame, float]:
    """Compare SHAP importance rankings across sessions.

    .. note::

       Kendall's tau is computed **asymmetrically**: the top-K features
       are selected by session A's importance ranking, then tau measures
       how well session B preserves that ranking.  Swapping sessions
       may produce a different tau.  This is intentional — session A is
       the "training" session in the A→B prediction direction.

    Args:
        shap_a: SHAP values from session A model (n_samples x n_features).
        shap_b: SHAP values from session B model.
        feature_names: Feature names matching the columns.
        top_k: Compare top K features. Defaults to
            :data:`FEATURE_IMPORTANCE_TOP_K`.

    Returns:
        ``(comparison_df, kendall_tau)`` where *comparison_df* has columns
        ``feature``, ``importance_a``, ``importance_b``, ``rank_a``,
        ``rank_b``; and *kendall_tau* is Kendall's tau on the top-K
        rankings.
    """
    if top_k is None:
        top_k = FEATURE_IMPORTANCE_TOP_K

    imp_a = np.abs(shap_a).mean(axis=0)
    imp_b = np.abs(shap_b).mean(axis=0)

    # argsort gives indices that sort the array; we need ranks (position of each element)
    rank_a = np.empty_like(np.argsort(-imp_a))
    rank_a[np.argsort(-imp_a)] = np.arange(1, len(imp_a) + 1)
    rank_b = np.empty_like(np.argsort(-imp_b))
    rank_b[np.argsort(-imp_b)] = np.arange(1, len(imp_b) + 1)

    df = pl.DataFrame(
        {
            "feature": feature_names,
            "importance_a": imp_a.tolist(),
            "importance_b": imp_b.tolist(),
            "rank_a": rank_a.tolist(),
            "rank_b": rank_b.tolist(),
        }
    ).sort("rank_a")

    # Kendall's tau on top-K features (by session A ranking)
    top_features = df.head(min(top_k, len(feature_names)))
    if top_features.height >= 2:
        tau, _ = stats.kendalltau(
            top_features["rank_a"].to_numpy(),
            top_features["rank_b"].to_numpy(),
        )
    else:
        tau = float("nan")

    return df, float(tau)
