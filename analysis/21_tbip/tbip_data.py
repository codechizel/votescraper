"""Pure data logic for text-based ideal point estimation (Phase 21).

Embedding-vote approach: multiply vote matrix by bill embeddings to get
legislator-level text profiles, then extract PC1 as a text-derived ideal point.

All functions are pure (no I/O, no plotting) — fully testable with synthetic data.
Correlation and outlier functions are adapted from Phase 14 external_validation_data.py
with lower quality thresholds (text is further removed from ideology than SM/DIME).

References:
  Vafa, K., Naidu, S., & Blei, D.M. (2020). "Text-Based Ideal Points." ACL.
  Lauderdale, B.E. & Herzog, A. (2016). "Measuring Political Positions from
    Legislative Speech." Political Analysis 24(3): 374-394.
"""

import math

import numpy as np
import polars as pl
from scipy import stats as sp_stats
from sklearn.decomposition import PCA

# ── Constants ────────────────────────────────────────────────────────────────

MIN_MATCHED = 10
"""Minimum legislators to compute correlations."""

MIN_BILLS = 5
"""Minimum bills with both embeddings and roll calls to proceed."""

OUTLIER_TOP_N = 5
"""Number of outliers to report per chamber/model."""

# Quality thresholds — lower than Phase 14 (text is further removed from ideology)
STRONG_CORRELATION = 0.80
GOOD_CORRELATION = 0.65
MODERATE_CORRELATION = 0.50

PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}

# ── Vote-Embedding Profiles ─────────────────────────────────────────────────


def build_vote_embedding_profiles(
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    embeddings: np.ndarray,
    bill_numbers: list[str],
    chamber: str | None = None,
    min_votes: int = 20,
) -> tuple[np.ndarray, list[str], int]:
    """Build legislator × embedding-dim profiles from vote-weighted embeddings.

    For each legislator, computes: sum(vote_i * embedding_i) / n_votes_cast,
    where vote_i is +1 (Yea), -1 (Nay), or 0 (absent/not voting).

    Args:
        votes: Vote DataFrame (must have legislator_slug, vote_id, vote columns).
        rollcalls: Rollcall DataFrame (must have vote_id, bill_number, optionally chamber).
        embeddings: (n_bills, embedding_dim) array aligned with bill_numbers.
        bill_numbers: Bill numbers corresponding to rows of embeddings.
        chamber: If set, filter rollcalls to this chamber ("House" or "Senate").
        min_votes: Minimum non-zero votes for a legislator to be included.

    Returns:
        (profiles, slugs, n_bills_matched) where profiles is (n_legislators, embedding_dim).

    Raises:
        ValueError: If fewer than MIN_BILLS bills have both embeddings and roll calls.
    """
    # Build bill_number → embedding index mapping
    bill_to_idx = {bn: i for i, bn in enumerate(bill_numbers)}

    # Filter rollcalls to chamber if specified
    rc = rollcalls
    if chamber and "chamber" in rc.columns:
        rc = rc.filter(pl.col("chamber") == chamber)

    # Map vote_id → bill_number for matching
    vote_to_bill = dict(
        zip(
            rc["vote_id"].to_list(),
            rc["bill_number"].to_list(),
            strict=True,
        )
    )

    # Find bills with both embeddings and roll calls
    matched_bills = {bn for bn in vote_to_bill.values() if bn in bill_to_idx}
    n_bills_matched = len(matched_bills)

    if n_bills_matched < MIN_BILLS:
        msg = (
            f"Only {n_bills_matched} bills have both embeddings and roll calls "
            f"(minimum: {MIN_BILLS}). Run `just text` first or check chamber filter."
        )
        raise ValueError(msg)

    # Keep only vote_ids that map to bills with embeddings
    valid_vote_ids = {vid for vid, bn in vote_to_bill.items() if bn in matched_bills}

    # Filter votes to valid vote_ids
    filtered = votes.filter(pl.col("vote_id").is_in(list(valid_vote_ids)))

    # Encode votes: Yea → +1, Nay → -1, else → 0
    filtered = filtered.with_columns(
        pl.when(pl.col("vote") == "Yea")
        .then(pl.lit(1))
        .when(pl.col("vote") == "Nay")
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("vote_numeric")
    )

    # Build per-legislator profiles
    embedding_dim = embeddings.shape[1]
    legislator_profiles: dict[str, np.ndarray] = {}
    legislator_counts: dict[str, int] = {}

    for row in filtered.iter_rows(named=True):
        slug = row["legislator_slug"]
        vote_id = row["vote_id"]
        v = row["vote_numeric"]

        if v == 0:
            continue

        bn = vote_to_bill.get(vote_id)
        if bn is None or bn not in bill_to_idx:
            continue

        emb = embeddings[bill_to_idx[bn]]

        if slug not in legislator_profiles:
            legislator_profiles[slug] = np.zeros(embedding_dim, dtype=np.float64)
            legislator_counts[slug] = 0

        legislator_profiles[slug] += v * emb
        legislator_counts[slug] += 1

    # Filter to legislators with enough votes and normalize
    slugs = []
    profiles = []
    for slug in sorted(legislator_profiles.keys()):
        count = legislator_counts[slug]
        if count >= min_votes:
            slugs.append(slug)
            profiles.append(legislator_profiles[slug] / count)

    if not profiles:
        msg = f"No legislators have >= {min_votes} non-absent votes on bills with embeddings."
        raise ValueError(msg)

    return np.array(profiles), slugs, n_bills_matched


# ── PCA ──────────────────────────────────────────────────────────────────────


def compute_text_ideal_points(
    profiles: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Extract PC1 from legislator text profiles as a text-derived ideal point.

    Args:
        profiles: (n_legislators, embedding_dim) array.

    Returns:
        (pc1_scores, explained_var_ratio_pc1, all_explained_var_ratios)
    """
    n_components = min(profiles.shape[0], profiles.shape[1], 10)
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(profiles)

    pc1 = transformed[:, 0]
    return pc1, float(pca.explained_variance_ratio_[0]), pca.explained_variance_ratio_


# ── Sign Alignment ───────────────────────────────────────────────────────────


def align_sign_convention(
    text_scores: np.ndarray,
    slugs: list[str],
    irt_df: pl.DataFrame,
) -> np.ndarray:
    """Flip text scores if negatively correlated with IRT xi_mean.

    Convention: Republicans should be positive (matching IRT sign convention).
    Uses Pearson r between matched legislators to decide.

    Returns a copy of text_scores (potentially negated).
    """
    # Build a quick lookup
    irt_lookup = {}
    for row in irt_df.iter_rows(named=True):
        irt_lookup[row["legislator_slug"]] = row.get("xi_mean", 0.0)

    matched_text = []
    matched_irt = []
    for i, slug in enumerate(slugs):
        if slug in irt_lookup:
            matched_text.append(text_scores[i])
            matched_irt.append(irt_lookup[slug])

    if len(matched_text) < 3:
        # Not enough matches to determine sign — return as-is
        return text_scores.copy()

    r, _ = sp_stats.pearsonr(matched_text, matched_irt)

    if r < 0:
        return -text_scores
    return text_scores.copy()


# ── Matching ─────────────────────────────────────────────────────────────────


def build_matched_df(
    text_scores: np.ndarray,
    slugs: list[str],
    irt_df: pl.DataFrame,
) -> pl.DataFrame:
    """Inner join text scores to IRT ideal points by legislator_slug.

    Returns a DataFrame with columns: legislator_slug, text_score, xi_mean,
    full_name, party, district, chamber (where available from IRT).
    """
    text_df = pl.DataFrame({"legislator_slug": slugs, "text_score": text_scores.tolist()})

    # Normalize IRT slug column name
    irt = irt_df
    # Select columns available in IRT
    keep_cols = ["legislator_slug", "xi_mean"]
    for optional in ["full_name", "party", "district", "chamber"]:
        if optional in irt.columns:
            keep_cols.append(optional)

    irt_subset = irt.select(keep_cols)

    matched = text_df.join(irt_subset, on="legislator_slug", how="inner")
    return matched


# ── Correlations ─────────────────────────────────────────────────────────────


def compute_correlations(
    matched: pl.DataFrame,
    xi_col: str = "xi_mean",
    text_col: str = "text_score",
) -> dict:
    """Compute Pearson r, Spearman rho, Fisher z CIs, and quality label.

    Adapted from Phase 14 external_validation_data.py with lower thresholds
    for text-derived scores.

    Returns dict with keys: pearson_r, pearson_p, spearman_rho, spearman_p,
    n, ci_lower, ci_upper, quality.
    """
    if matched.height < MIN_MATCHED:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n": matched.height,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "quality": "insufficient_data",
        }

    xi = matched[xi_col].to_numpy().astype(float)
    text = matched[text_col].to_numpy().astype(float)

    # Drop NaN pairs
    valid = ~(np.isnan(xi) | np.isnan(text))
    xi = xi[valid]
    text = text[valid]

    if len(xi) < MIN_MATCHED:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n": int(len(xi)),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "quality": "insufficient_data",
        }

    # Check for zero variance
    if np.std(xi) == 0 or np.std(text) == 0:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n": int(len(xi)),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "quality": "insufficient_data",
        }

    r, p_r = sp_stats.pearsonr(xi, text)
    rho, p_rho = sp_stats.spearmanr(xi, text)

    ci_lower, ci_upper = _fisher_z_ci(r, len(xi))
    quality = _quality_label(abs(r))

    return {
        "pearson_r": float(r),
        "pearson_p": float(p_r),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "n": int(len(xi)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "quality": quality,
    }


def compute_intra_party_correlations(
    matched: pl.DataFrame,
    xi_col: str = "xi_mean",
    text_col: str = "text_score",
) -> dict[str, dict]:
    """Compute within-party correlations (R and D separately).

    Returns dict: {"Republican": {...}, "Democrat": {...}}.
    """
    results: dict[str, dict] = {}

    for party in ["Republican", "Democrat"]:
        if "party" not in matched.columns:
            continue
        party_df = matched.filter(pl.col("party") == party)
        results[party] = compute_correlations(party_df, xi_col, text_col)

    return results


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

    Lower thresholds than Phase 14 — text scores are further removed from
    direct ideology measurement than SM/DIME scores.
    """
    if abs_r >= STRONG_CORRELATION:
        return "strong"
    elif abs_r >= GOOD_CORRELATION:
        return "good"
    elif abs_r >= MODERATE_CORRELATION:
        return "moderate"
    else:
        return "weak"


# ── Outlier Detection ────────────────────────────────────────────────────────


def identify_outliers(
    matched: pl.DataFrame,
    xi_col: str = "xi_mean",
    text_col: str = "text_score",
    top_n: int = OUTLIER_TOP_N,
) -> pl.DataFrame:
    """Identify top-N outliers by z-score of (xi_mean - text_score) discrepancy.

    Both columns are z-standardized before computing the discrepancy,
    so the comparison is scale-invariant.
    """
    if matched.height < 3:
        return pl.DataFrame()

    xi = matched[xi_col].to_numpy().astype(float)
    text = matched[text_col].to_numpy().astype(float)

    # Z-standardize both
    xi_z = (xi - np.mean(xi)) / np.std(xi) if np.std(xi) > 0 else xi * 0
    text_z = (text - np.mean(text)) / np.std(text) if np.std(text) > 0 else text * 0

    discrepancy = np.abs(xi_z - text_z)

    result = matched.with_columns(
        pl.Series("xi_z", xi_z),
        pl.Series("text_z", text_z),
        pl.Series("discrepancy_z", discrepancy),
    )

    return result.sort("discrepancy_z", descending=True).head(top_n)
