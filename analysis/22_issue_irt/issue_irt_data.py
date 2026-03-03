"""Pure data logic for issue-specific ideal point estimation (Phase 19).

Topic-stratified flat IRT: run the battle-tested Phase 04 IRT model on
per-topic vote subsets to answer "how conservative is each legislator
on education vs healthcare vs taxes?"

All functions are pure (no I/O, no plotting) — fully testable with synthetic data.

References:
  Shin, M. (2024). "issueirt: Issue-Specific Ideal Point Estimation."
  Clinton, J. & Lapinski, J. (2006). "Measuring Legislative Accomplishment."
  Lauderdale, B. & Clark, T. (2014). "Scaling Politically Meaningful Dimensions."
"""

import math

import numpy as np
import polars as pl
from scipy import stats as sp_stats

# ── Constants ────────────────────────────────────────────────────────────────

MIN_BILLS_PER_TOPIC = 10
"""Minimum roll calls in a topic to run IRT. Below this, posterior is too diffuse."""

MIN_LEGISLATORS_PER_TOPIC = 10
"""Minimum legislators with enough votes in a topic. Small N → unreliable ideal points."""

MIN_VOTES_IN_TOPIC = 5
"""Per-legislator minimum non-null votes within a topic to be included."""

RHAT_THRESHOLD = 1.05
"""Per-topic R-hat threshold (relaxed from Phase 04's 1.01 — smaller models are noisier)."""

ESS_THRESHOLD = 200
"""Per-topic ESS threshold (relaxed from Phase 04's 400 — smaller models)."""

OUTLIER_TOP_N = 10
"""Number of top deviators to report per topic."""

# Quality thresholds for per-topic correlations with full IRT
STRONG_CORRELATION = 0.80
GOOD_CORRELATION = 0.60
MODERATE_CORRELATION = 0.40

PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}


# ── Topic Loading ────────────────────────────────────────────────────────────


def load_topic_assignments(
    bill_text_dir: str | object,
    taxonomy: str,
) -> pl.DataFrame:
    """Load topic assignments from Phase 18 output.

    Args:
        bill_text_dir: Path to Phase 18 results directory (contains data/).
        taxonomy: "bertopic" or "cap".

    Returns:
        DataFrame with columns: bill_number, topic_id, topic_label.
        BERTopic topic_id == -1 (noise) is excluded.
    """
    from pathlib import Path

    data_dir = Path(str(bill_text_dir)) / "data"

    if taxonomy == "bertopic":
        path = data_dir / "bill_topics_all.parquet"
        if not path.exists():
            msg = f"BERTopic topics not found: {path}. Run `just text-analysis` first."
            raise FileNotFoundError(msg)
        df = pl.read_parquet(path)
        # Exclude noise cluster (topic_id == -1)
        df = df.filter(pl.col("topic_id") != -1)
        return df.select(["bill_number", "topic_id", "topic_label"])

    elif taxonomy == "cap":
        path = data_dir / "cap_classifications.parquet"
        if not path.exists():
            msg = (
                f"CAP classifications not found: {path}. Run `just text-analysis --classify` first."
            )
            raise FileNotFoundError(msg)
        df = pl.read_parquet(path)
        # CAP uses cap_code as topic_id and cap_label as topic_label
        if "cap_code" in df.columns:
            df = df.rename({"cap_code": "topic_id", "cap_label": "topic_label"})
        return df.select(["bill_number", "topic_id", "topic_label"])

    else:
        msg = f"Unknown taxonomy: {taxonomy}. Expected 'bertopic' or 'cap'."
        raise ValueError(msg)


# ── Eligibility Filtering ────────────────────────────────────────────────────


def get_eligible_topics(
    topics: pl.DataFrame,
    rollcalls: pl.DataFrame,
    min_bills: int = MIN_BILLS_PER_TOPIC,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Filter topics with enough roll-call bills to run IRT.

    Args:
        topics: DataFrame with bill_number, topic_id, topic_label.
        rollcalls: Rollcalls DataFrame with vote_id, bill_number.
        min_bills: Minimum bills with roll calls in a topic.

    Returns:
        (eligible, report) where:
        - eligible: topics with >= min_bills roll-call bills
        - report: per-topic DataFrame with topic_id, topic_label, n_bills,
          n_rollcall_bills, eligible
    """
    # Get unique bill_numbers that have roll calls
    rc_bills = rollcalls["bill_number"].unique().to_list()

    # Count per topic: total bills and bills with roll calls
    topic_stats = (
        topics.group_by(["topic_id", "topic_label"])
        .agg(
            pl.col("bill_number").n_unique().alias("n_bills"),
            pl.col("bill_number")
            .filter(pl.col("bill_number").is_in(rc_bills))
            .n_unique()
            .alias("n_rollcall_bills"),
        )
        .sort("n_rollcall_bills", descending=True)
    )

    report = topic_stats.with_columns((pl.col("n_rollcall_bills") >= min_bills).alias("eligible"))

    eligible = report.filter(pl.col("eligible"))

    return eligible, report


# ── Vote Matrix Subsetting ───────────────────────────────────────────────────


def subset_vote_matrix_for_topic(
    matrix: pl.DataFrame,
    topics: pl.DataFrame,
    rollcalls: pl.DataFrame,
    topic_id: int,
) -> pl.DataFrame:
    """Subset the wide vote matrix to vote_ids belonging to a specific topic.

    Args:
        matrix: Wide vote matrix (legislator_slug × vote_ids, values 0/1/null).
        topics: Topic assignments (bill_number, topic_id, topic_label).
        rollcalls: Rollcalls (vote_id, bill_number).
        topic_id: The topic to subset.

    Returns:
        Subsetted wide vote matrix with only vote_ids for bills in this topic.
    """
    # Get bill_numbers for this topic
    topic_bills = topics.filter(pl.col("topic_id") == topic_id)["bill_number"].unique().to_list()

    # Get vote_ids for these bills
    topic_vote_ids = (
        rollcalls.filter(pl.col("bill_number").is_in(topic_bills))["vote_id"].unique().to_list()
    )

    # Subset matrix columns
    slug_col = "legislator_slug"
    available_vote_ids = [c for c in matrix.columns if c != slug_col and c in topic_vote_ids]

    if not available_vote_ids:
        return pl.DataFrame({slug_col: matrix[slug_col]})

    return matrix.select([slug_col, *available_vote_ids])


def filter_legislators_in_topic(
    matrix: pl.DataFrame,
    min_votes: int = MIN_VOTES_IN_TOPIC,
) -> pl.DataFrame:
    """Drop legislators with fewer than min_votes non-null votes in topic.

    Args:
        matrix: Wide vote matrix (legislator_slug × vote_ids).
        min_votes: Minimum non-null votes required.

    Returns:
        Filtered matrix (same columns, fewer rows).
    """
    slug_col = "legislator_slug"
    vote_cols = [c for c in matrix.columns if c != slug_col]

    if not vote_cols:
        return matrix.head(0)

    # Count non-null votes per legislator
    matrix = matrix.with_columns(
        pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in vote_cols]).alias(
            "_n_votes"
        )
    )

    filtered = matrix.filter(pl.col("_n_votes") >= min_votes).drop("_n_votes")
    return filtered


# ── PCA Scores for Fallback Anchors ──────────────────────────────────────────


def compute_topic_pca_scores(
    matrix: pl.DataFrame,
) -> pl.DataFrame | None:
    """Quick PCA on a per-topic vote matrix for fallback anchor selection.

    Returns DataFrame with legislator_slug and PC1, or None if too few legislators.
    """
    from sklearn.decomposition import PCA

    slug_col = "legislator_slug"
    vote_cols = [c for c in matrix.columns if c != slug_col]

    if len(vote_cols) < 2 or matrix.height < 3:
        return None

    slugs = matrix[slug_col].to_list()
    X = matrix.select(vote_cols).to_numpy().astype(float)

    # Fill NaN with column mean for PCA
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Check for zero-variance columns
    col_std = np.std(X, axis=0)
    valid_cols = col_std > 0
    if valid_cols.sum() < 2:
        return None

    X = X[:, valid_cols]

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X)[:, 0]

    return pl.DataFrame({"legislator_slug": slugs, "PC1": pc1.tolist()})


# ── Sign Alignment ───────────────────────────────────────────────────────────


def align_topic_ideal_points(
    topic_xi: pl.DataFrame,
    full_irt: pl.DataFrame,
) -> pl.DataFrame:
    """Correlation-based sign flip of per-topic ideal points against full IRT.

    If Pearson r between matched legislators' topic xi_mean and full xi_mean
    is negative, negate the topic ideal points.

    Args:
        topic_xi: Per-topic ideal points (legislator_slug, xi_mean, ...).
        full_irt: Full-model IRT ideal points (legislator_slug, xi_mean, ...).

    Returns:
        topic_xi with xi_mean (and xi_hdi columns) potentially negated.
    """
    slug_col = "legislator_slug"

    # Match on slug
    matched = topic_xi.join(
        full_irt.select([slug_col, pl.col("xi_mean").alias("full_xi_mean")]),
        on=slug_col,
        how="inner",
    )

    if matched.height < 3:
        return topic_xi

    topic_vals = matched["xi_mean"].to_numpy().astype(float)
    full_vals = matched["full_xi_mean"].to_numpy().astype(float)

    if np.std(topic_vals) == 0 or np.std(full_vals) == 0:
        return topic_xi

    r, _ = sp_stats.pearsonr(topic_vals, full_vals)

    if r < 0:
        # Negate xi columns
        negate_cols = [c for c in topic_xi.columns if c.startswith("xi_")]
        exprs = []
        for c in topic_xi.columns:
            if c in negate_cols:
                exprs.append((-pl.col(c)).alias(c))
            else:
                exprs.append(pl.col(c))
        return topic_xi.select(exprs)

    return topic_xi


# ── Cross-Topic Matrix ───────────────────────────────────────────────────────


def build_cross_topic_matrix(
    results: dict[int, dict],
) -> pl.DataFrame:
    """Assemble a legislator × topic wide matrix of ideal points.

    Args:
        results: {topic_id: {"label": str, "ideal_points": DataFrame}} where
                 each DataFrame has legislator_slug and xi_mean.

    Returns:
        Wide DataFrame: legislator_slug, topic_0, topic_1, ... (raw xi_mean values).
    """
    if not results:
        return pl.DataFrame({"legislator_slug": []})

    # Collect all unique slugs
    all_slugs: set[str] = set()
    for data in results.values():
        ip = data["ideal_points"]
        all_slugs.update(ip["legislator_slug"].to_list())

    base = pl.DataFrame({"legislator_slug": sorted(all_slugs)})

    for topic_id, data in sorted(results.items()):
        ip = data["ideal_points"]
        label = data["label"]
        col_name = f"t{topic_id}_{_sanitize_label(label)}"

        topic_col = ip.select(["legislator_slug", pl.col("xi_mean").alias(col_name)])
        base = base.join(topic_col, on="legislator_slug", how="left")

    return base


def _sanitize_label(label: str) -> str:
    """Convert a topic label to a safe column name fragment."""
    import re

    # Take first 30 chars, lowercase, replace non-alphanum with underscore
    clean = re.sub(r"[^a-z0-9]+", "_", label[:30].lower()).strip("_")
    return clean or "unknown"


# ── Cross-Topic Correlations ─────────────────────────────────────────────────


def compute_cross_topic_correlations(
    matrix: pl.DataFrame,
) -> pl.DataFrame:
    """Pairwise Pearson r between all topic ideal point columns.

    Args:
        matrix: Wide DataFrame from build_cross_topic_matrix().

    Returns:
        DataFrame with columns: topic_a, topic_b, pearson_r, n.
    """
    topic_cols = [c for c in matrix.columns if c != "legislator_slug"]

    if len(topic_cols) < 2:
        return pl.DataFrame(
            {"topic_a": [], "topic_b": [], "pearson_r": [], "n": []},
            schema={"topic_a": pl.Utf8, "topic_b": pl.Utf8, "pearson_r": pl.Float64, "n": pl.Int64},
        )

    rows: list[dict] = []
    for i, col_a in enumerate(topic_cols):
        for j, col_b in enumerate(topic_cols):
            if j <= i:
                continue

            vals_a = matrix[col_a].to_numpy().astype(float)
            vals_b = matrix[col_b].to_numpy().astype(float)

            # Drop pairs where either is NaN
            valid = ~(np.isnan(vals_a) | np.isnan(vals_b))
            n = int(valid.sum())

            if n < 3:
                rows.append({"topic_a": col_a, "topic_b": col_b, "pearson_r": float("nan"), "n": n})
                continue

            r, _ = sp_stats.pearsonr(vals_a[valid], vals_b[valid])
            rows.append({"topic_a": col_a, "topic_b": col_b, "pearson_r": float(r), "n": n})

    return pl.DataFrame(rows)


# ── Outlier Detection ────────────────────────────────────────────────────────


def identify_topic_outliers(
    topic_xi: pl.DataFrame,
    full_irt: pl.DataFrame,
    top_n: int = OUTLIER_TOP_N,
) -> pl.DataFrame:
    """Biggest deviators: legislators furthest from their overall IRT position in this topic.

    Both topic and full xi_mean are z-standardized before computing discrepancy.

    Args:
        topic_xi: Per-topic ideal points (legislator_slug, xi_mean, ...).
        full_irt: Full-model IRT ideal points (legislator_slug, xi_mean, ...).
        top_n: Number of top outliers to return.

    Returns:
        DataFrame with legislator_slug, full_name, party, xi_topic, xi_full,
        discrepancy_z, sorted by largest discrepancy.
    """
    slug_col = "legislator_slug"

    # Build full IRT lookup
    full_cols = [slug_col, "xi_mean"]
    for optional in ["full_name", "party"]:
        if optional in full_irt.columns:
            full_cols.append(optional)

    matched = topic_xi.select([slug_col, pl.col("xi_mean").alias("xi_topic")]).join(
        full_irt.select(full_cols).rename({"xi_mean": "xi_full"}),
        on=slug_col,
        how="inner",
    )

    if matched.height < 3:
        return pl.DataFrame()

    xi_topic = matched["xi_topic"].to_numpy().astype(float)
    xi_full = matched["xi_full"].to_numpy().astype(float)

    # Z-standardize
    topic_std = np.std(xi_topic)
    full_std = np.std(xi_full)
    topic_z = (xi_topic - np.mean(xi_topic)) / topic_std if topic_std > 0 else xi_topic * 0
    full_z = (xi_full - np.mean(xi_full)) / full_std if full_std > 0 else xi_full * 0

    discrepancy = np.abs(topic_z - full_z)

    result = matched.with_columns(pl.Series("discrepancy_z", discrepancy))
    return result.sort("discrepancy_z", descending=True).head(top_n)


# ── Anchor Stability ─────────────────────────────────────────────────────────


def check_anchor_stability(
    results: dict[int, dict],
    full_irt: pl.DataFrame,
    anchor_slugs: tuple[str, str],
) -> pl.DataFrame:
    """Check whether the full-model anchors stay extreme per topic.

    For each modeled topic, check where the conservative and liberal anchors
    rank in the per-topic ideal point distribution. Stable anchors should
    remain near the extremes.

    Args:
        results: {topic_id: {"label": str, "ideal_points": DataFrame}}.
        full_irt: Full-model IRT ideal points.
        anchor_slugs: (conservative_slug, liberal_slug).

    Returns:
        DataFrame with topic_id, topic_label, cons_rank, cons_pctile,
        lib_rank, lib_pctile, n_legislators.
    """
    cons_slug, lib_slug = anchor_slugs

    rows: list[dict] = []
    for topic_id, data in sorted(results.items()):
        ip = data["ideal_points"]
        label = data["label"]
        n = ip.height

        # Sort by xi_mean descending
        sorted_ip = ip.sort("xi_mean", descending=True)
        slugs = sorted_ip["legislator_slug"].to_list()

        cons_rank = slugs.index(cons_slug) + 1 if cons_slug in slugs else None
        lib_rank = slugs.index(lib_slug) + 1 if lib_slug in slugs else None

        rows.append(
            {
                "topic_id": topic_id,
                "topic_label": label,
                "cons_rank": cons_rank,
                "cons_pctile": (cons_rank / n * 100) if cons_rank is not None and n > 0 else None,
                "lib_rank": lib_rank,
                "lib_pctile": (lib_rank / n * 100) if lib_rank is not None and n > 0 else None,
                "n_legislators": n,
            }
        )

    return pl.DataFrame(rows)


# ── Per-Topic Correlation ────────────────────────────────────────────────────


def compute_topic_irt_correlation(
    topic_xi: pl.DataFrame,
    full_irt: pl.DataFrame,
) -> dict:
    """Pearson r, Spearman rho, Fisher z CI, and quality label for one topic.

    Args:
        topic_xi: Per-topic ideal points (legislator_slug, xi_mean).
        full_irt: Full-model IRT ideal points (legislator_slug, xi_mean).

    Returns:
        Dict with pearson_r, spearman_rho, pearson_p, spearman_p, n,
        ci_lower, ci_upper, quality.
    """
    slug_col = "legislator_slug"

    matched = topic_xi.select([slug_col, pl.col("xi_mean").alias("topic_xi")]).join(
        full_irt.select([slug_col, pl.col("xi_mean").alias("full_xi")]),
        on=slug_col,
        how="inner",
    )

    n = matched.height

    if n < 5:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n": n,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "quality": "insufficient_data",
        }

    topic_vals = matched["topic_xi"].to_numpy().astype(float)
    full_vals = matched["full_xi"].to_numpy().astype(float)

    # Drop NaN pairs
    valid = ~(np.isnan(topic_vals) | np.isnan(full_vals))
    topic_vals = topic_vals[valid]
    full_vals = full_vals[valid]
    n = int(len(topic_vals))

    if n < 5 or np.std(topic_vals) == 0 or np.std(full_vals) == 0:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n": n,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "quality": "insufficient_data",
        }

    r, p_r = sp_stats.pearsonr(topic_vals, full_vals)
    rho, p_rho = sp_stats.spearmanr(topic_vals, full_vals)

    ci_lower, ci_upper = _fisher_z_ci(r, n)
    quality = _quality_label(abs(r))

    return {
        "pearson_r": float(r),
        "pearson_p": float(p_r),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "n": n,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "quality": quality,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute Fisher z-transform confidence interval for Pearson r."""
    if n < 4 or abs(r) >= 1.0:
        return (float("nan"), float("nan"))

    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)

    z_lower = z - z_crit * se
    z_upper = z + z_crit * se

    return (float(np.tanh(z_lower)), float(np.tanh(z_upper)))


def _quality_label(abs_r: float) -> str:
    """Assign quality label based on absolute Pearson r.

    Lower thresholds than Phase 04 full model — per-topic subsets are smaller
    and noisier.
    """
    if abs_r >= STRONG_CORRELATION:
        return "strong"
    elif abs_r >= GOOD_CORRELATION:
        return "good"
    elif abs_r >= MODERATE_CORRELATION:
        return "moderate"
    else:
        return "weak"
