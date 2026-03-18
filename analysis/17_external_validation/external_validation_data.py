"""Pure data logic for external validation against Shor-McCarty scores.

All functions are pure (no I/O, no network). Parsing, name normalization,
matching, and correlation computation live here so they can be tested with
synthetic data.

Shor-McCarty dataset: career-level ideal points for state legislators,
published at Harvard Dataverse (CC0 license). Overlaps our 84th-88th
bienniums (2011-2020), providing the first external validation of our
IRT ideal points.
"""

import math
import re

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from analysis.run_context import strip_leadership_suffix
from analysis.tuning import CONCERN_CORRELATION, GOOD_CORRELATION, STRONG_CORRELATION

# ── Constants ────────────────────────────────────────────────────────────────

SHOR_MCCARTY_URL = "https://dataverse.harvard.edu/api/access/datafile/7067107"
SHOR_MCCARTY_CACHE_PATH = "data/external/shor_mccarty.tab"

OVERLAPPING_BIENNIUMS: dict[str, tuple[int, int]] = {
    "84th_2011-2012": (2011, 2012),
    "85th_2013-2014": (2013, 2014),
    "86th_2015-2016": (2015, 2016),
    "87th_2017-2018": (2017, 2018),
    "88th_2019-2020": (2019, 2020),
}

MIN_MATCHED = 10
OUTLIER_TOP_N = 5

# ── Name Normalization ───────────────────────────────────────────────────────

_SUFFIX_RE = re.compile(r"\s+(?:Sr\.?|Jr\.?|III?|IV)$", re.IGNORECASE)
_MIDDLE_INITIAL_RE = re.compile(r"\s+[A-Z]\.?\s+")


def normalize_sm_name(name: str) -> str:
    """Normalize a Shor-McCarty name: "Alcala, John" -> "john alcala".

    Strips suffixes (Sr., Jr., III), middle initials, and extra whitespace.
    """
    if not name or not name.strip():
        return ""

    name = name.strip()

    # Handle "Last, First [Middle]" format
    if "," in name:
        parts = name.split(",", 1)
        last = parts[0].strip()
        first_parts = parts[1].strip()
    else:
        # Fallback: just lowercase
        return _clean_name_parts(name)

    # Extract first name only (drop middle names/initials)
    first_tokens = first_parts.split()
    first = first_tokens[0] if first_tokens else ""

    canonical = f"{first} {last}"
    return _clean_name_parts(canonical)


def normalize_our_name(name: str) -> str:
    """Normalize our legislator name: "John Alcala" -> "john alcala".

    Strips leadership suffixes (" - President"), generational suffixes,
    and middle names beyond the first.
    """
    if not name or not name.strip():
        return ""

    name = name.strip()

    # Strip leadership suffix: "Ty Masterson - President of the Senate"
    name = strip_leadership_suffix(name)

    # Strip generational suffix: "John Barker Sr."
    name = _SUFFIX_RE.sub("", name).strip()

    # Split into tokens: first [middle...] last
    tokens = name.split()
    if len(tokens) <= 2:
        return _clean_name_parts(name)

    # Keep first and last only
    canonical = f"{tokens[0]} {tokens[-1]}"
    return _clean_name_parts(canonical)


def _clean_name_parts(name: str) -> str:
    """Lowercase, strip extra whitespace, remove periods."""
    name = name.replace(".", "").lower().strip()
    return re.sub(r"\s+", " ", name)


# ── Shor-McCarty Parsing ────────────────────────────────────────────────────


def parse_shor_mccarty(raw_text: str) -> pl.DataFrame:
    """Parse tab-separated Shor-McCarty data, filter to Kansas.

    Returns a DataFrame with columns: name, st, np_score, and all year
    indicator columns (house{YYYY}, senate{YYYY}), plus normalized_name.
    """
    lines = raw_text.strip().split("\n")
    if len(lines) < 2:
        return pl.DataFrame()

    header = [h.strip('"') for h in lines[0].split("\t")]
    rows = []
    for line in lines[1:]:
        fields = [f.strip('"') for f in line.split("\t")]
        if len(fields) >= len(header):
            rows.append(dict(zip(header, fields)))

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)

    # Filter to Kansas
    if "st" in df.columns:
        df = df.filter(pl.col("st") == "KS")
    else:
        return pl.DataFrame()

    if df.height == 0:
        return pl.DataFrame()

    # Cast np_score to float
    if "np_score" in df.columns:
        df = df.with_columns(
            pl.col("np_score")
            .str.strip_chars()
            .replace("", None)
            .cast(pl.Float64, strict=False)
            .alias("np_score")
        )

    # Cast year indicator columns to int
    year_cols = [c for c in df.columns if re.match(r"^(house|senate)\d{4}$", c)]
    for col in year_cols:
        df = df.with_columns(
            pl.col(col).str.strip_chars().replace("", "0").cast(pl.Int64, strict=False).alias(col)
        )

    # Add normalized name
    if "name" in df.columns:
        df = df.with_columns(
            pl.col("name")
            .map_elements(normalize_sm_name, return_dtype=pl.Utf8)
            .alias("normalized_name")
        )

    # Drop rows with null np_score
    df = df.filter(pl.col("np_score").is_not_null())

    return df


# ── Biennium Filtering ───────────────────────────────────────────────────────


def filter_to_biennium(
    sm_df: pl.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Filter SM data to legislators active in a given biennium.

    Returns (house_df, senate_df) based on house{YYYY}/senate{YYYY}
    indicator columns. A legislator is included if they have a 1 in
    any year of the biennium for that chamber.
    """
    years = list(range(start_year, end_year + 1))

    house_df = _filter_chamber(sm_df, "house", years)
    senate_df = _filter_chamber(sm_df, "senate", years)

    return house_df, senate_df


def _filter_chamber(
    sm_df: pl.DataFrame,
    chamber: str,
    years: list[int],
) -> pl.DataFrame:
    """Filter to legislators active in the given chamber during any of the years."""
    cols = [f"{chamber}{y}" for y in years if f"{chamber}{y}" in sm_df.columns]
    if not cols:
        return pl.DataFrame()

    # A legislator is active if any year column == 1
    mask = pl.lit(False)
    for col in cols:
        mask = mask | (pl.col(col) == 1)

    filtered = sm_df.filter(mask)

    # Deduplicate by name (same legislator may span multiple sessions)
    if "name" in filtered.columns:
        filtered = filtered.unique(subset=["name"])

    return filtered


# ── Legislator Matching ──────────────────────────────────────────────────────


def match_legislators(
    our_df: pl.DataFrame,
    sm_df: pl.DataFrame,
    chamber: str,
    start_year: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Match our legislators to SM legislators by normalized name.

    Phase 1: Exact normalized name match.
    Phase 2: Last-name-only match with district tiebreaker.

    Args:
        our_df: Our IRT ideal points (must have full_name, xi_mean, district, party).
        sm_df: SM data filtered to the correct chamber/biennium.
        chamber: "House" or "Senate" for the unmatched report.
        start_year: Biennium start year (e.g. 2011). Used to resolve SM district columns.

    Returns:
        (matched_df, unmatched_report_df)
        matched_df has columns from both datasets.
        unmatched_report_df lists unmatched legislators from both sides.
    """
    if our_df.height == 0 or sm_df.height == 0:
        return pl.DataFrame(), _build_unmatched_report(our_df, sm_df, chamber)

    # Ensure our_df has normalized names
    if "normalized_name" not in our_df.columns:
        our_df = our_df.with_columns(
            pl.col("full_name")
            .map_elements(normalize_our_name, return_dtype=pl.Utf8)
            .alias("normalized_name")
        )

    # Phase 1: Exact normalized name match
    matched = our_df.join(
        sm_df.select("normalized_name", "np_score", "name").rename({"name": "sm_name"}),
        on="normalized_name",
        how="inner",
    )

    matched_our_names = set(matched["normalized_name"].to_list())

    # Phase 2: Last-name-only match for unmatched legislators
    unmatched_ours = our_df.filter(~pl.col("normalized_name").is_in(matched_our_names))
    unmatched_sm_names = set(sm_df["normalized_name"].to_list()) - matched_our_names

    if unmatched_ours.height > 0 and unmatched_sm_names:
        phase2_matches = _phase2_last_name_match(
            unmatched_ours,
            sm_df.filter(pl.col("normalized_name").is_in(list(unmatched_sm_names))),
            start_year=start_year,
            chamber=chamber,
        )
        if phase2_matches.height > 0:
            matched = pl.concat([matched, phase2_matches])
            matched_our_names = set(matched["normalized_name"].to_list())

    # Build unmatched report
    unmatched_ours_final = our_df.filter(~pl.col("normalized_name").is_in(matched_our_names))
    matched_sm_names = set(matched["sm_name"].to_list()) if "sm_name" in matched.columns else set()
    unmatched_sm_final = sm_df.filter(~pl.col("name").is_in(matched_sm_names))

    unmatched_report = _build_unmatched_report(unmatched_ours_final, unmatched_sm_final, chamber)

    return matched, unmatched_report


def _extract_sm_district(
    sm_df: pl.DataFrame,
    start_year: int,
    chamber: str,
) -> pl.DataFrame:
    """Add an ``_sm_district`` Int64 column from year-specific SM district columns.

    SM stores districts as ``hdistrict{YYYY}`` (House) / ``sdistrict{YYYY}`` (Senate).
    Coalesces across biennium years, returning the first non-null value per legislator.
    """
    prefix = "h" if chamber == "House" else "s"
    end_year = start_year + 1
    dist_cols = [
        f"{prefix}district{y}"
        for y in (start_year, end_year)
        if f"{prefix}district{y}" in sm_df.columns
    ]

    if not dist_cols:
        return sm_df.with_columns(pl.lit(None).cast(pl.Int64).alias("_sm_district"))

    # Coalesce across years, strip leading zeros, cast to Int64
    expr = pl.coalesce([pl.col(c) for c in dist_cols])
    return sm_df.with_columns(
        expr.str.strip_chars()
        .str.replace(r"^0+", "")
        .cast(pl.Int64, strict=False)
        .alias("_sm_district")
    )


def _phase2_last_name_match(
    our_df: pl.DataFrame,
    sm_df: pl.DataFrame,
    start_year: int,
    chamber: str,
) -> pl.DataFrame:
    """Phase 2: Match by last name with district tiebreaker for ambiguous cases.

    When multiple SM candidates share a last name with one of our legislators:
    - If one candidate's district matches → use that match.
    - If no candidate's district matches → reject (no match is better than wrong match).
    Single-candidate matches are kept as-is regardless of district.
    """
    if our_df.height == 0 or sm_df.height == 0:
        return pl.DataFrame()

    # Add SM district column for tiebreaking
    sm_df = _extract_sm_district(sm_df, start_year, chamber)

    # Extract last names
    our_with_last = our_df.with_columns(
        pl.col("normalized_name")
        .map_elements(lambda n: n.split()[-1] if n.split() else "", return_dtype=pl.Utf8)
        .alias("_last_name")
    )
    sm_with_last = sm_df.with_columns(
        pl.col("normalized_name")
        .map_elements(lambda n: n.split()[-1] if n.split() else "", return_dtype=pl.Utf8)
        .alias("_last_name")
    )

    # Join on last name
    candidates = our_with_last.join(
        sm_with_last.select(
            "_last_name", "normalized_name", "np_score", "name", "_sm_district"
        ).rename({"normalized_name": "sm_normalized_name", "name": "sm_name"}),
        on="_last_name",
        how="inner",
    )

    if candidates.height == 0:
        return pl.DataFrame()

    # District tiebreaker for ambiguous last-name matches
    deduped = _deduplicate_with_district(candidates, "_sm_district")

    # Drop helper columns
    helper = ("_last_name", "sm_normalized_name", "_sm_district")
    drop_cols = [c for c in helper if c in deduped.columns]
    return deduped.drop(drop_cols)


def _deduplicate_with_district(
    candidates: pl.DataFrame,
    ext_district_col: str,
) -> pl.DataFrame:
    """Deduplicate last-name matches using district tiebreaker.

    - Single candidate per legislator → kept as-is.
    - Multiple candidates → prefer the one whose district matches; if none match, reject.
    """
    # Count candidates per our legislator
    counts = candidates.group_by("normalized_name").agg(pl.len().alias("_n"))

    # Single-candidate matches: keep directly
    single_names = counts.filter(pl.col("_n") == 1)["normalized_name"].to_list()
    singles = candidates.filter(pl.col("normalized_name").is_in(single_names))

    # Multi-candidate matches: use district to disambiguate
    multi_names = counts.filter(pl.col("_n") > 1)["normalized_name"].to_list()
    if not multi_names:
        return singles

    multi = candidates.filter(pl.col("normalized_name").is_in(multi_names))

    # Keep rows where our district matches the external dataset's district
    has_district = "district" in multi.columns and ext_district_col in multi.columns
    if has_district:
        district_matched = multi.filter(pl.col("district") == pl.col(ext_district_col))
        # Take first per legislator in case of remaining duplicates
        district_matched = district_matched.unique(subset=["normalized_name"])
    else:
        # No district data — reject all ambiguous matches
        district_matched = pl.DataFrame()

    if district_matched.height > 0:
        return pl.concat([singles, district_matched], how="diagonal")
    return singles


def _build_unmatched_report(
    our_unmatched: pl.DataFrame,
    sm_unmatched: pl.DataFrame,
    chamber: str,
) -> pl.DataFrame:
    """Build a report of unmatched legislators from both sides."""
    rows = []

    if our_unmatched.height > 0 and "full_name" in our_unmatched.columns:
        for row in our_unmatched.iter_rows(named=True):
            rows.append(
                {
                    "source": "our_data",
                    "name": row.get("full_name", ""),
                    "normalized": row.get("normalized_name", ""),
                    "chamber": chamber,
                    "party": row.get("party", ""),
                }
            )

    if sm_unmatched.height > 0 and "name" in sm_unmatched.columns:
        for row in sm_unmatched.iter_rows(named=True):
            rows.append(
                {
                    "source": "shor_mccarty",
                    "name": row.get("name", ""),
                    "normalized": row.get("normalized_name", ""),
                    "chamber": chamber,
                    "party": "",
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "source": pl.Utf8,
                "name": pl.Utf8,
                "normalized": pl.Utf8,
                "chamber": pl.Utf8,
                "party": pl.Utf8,
            }
        )

    return pl.DataFrame(rows)


# ── Correlations ─────────────────────────────────────────────────────────────


def compute_correlations(
    matched: pl.DataFrame,
    xi_col: str = "xi_mean",
    np_col: str = "np_score",
) -> dict:
    """Compute Pearson r, Spearman rho, Fisher z CIs, and quality label.

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
    np_scores = matched[np_col].to_numpy().astype(float)

    # Drop NaN pairs
    valid = ~(np.isnan(xi) | np.isnan(np_scores))
    xi = xi[valid]
    np_scores = np_scores[valid]

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

    r, p_r = sp_stats.pearsonr(xi, np_scores)
    rho, p_rho = sp_stats.spearmanr(xi, np_scores)

    # Fisher z confidence interval for Pearson r
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
    np_col: str = "np_score",
) -> dict[str, dict]:
    """Compute within-party correlations (R and D separately).

    Returns dict: {"Republican": {...}, "Democrat": {...}}.
    """
    results: dict[str, dict] = {}

    for party in ["Republican", "Democrat"]:
        if "party" not in matched.columns:
            continue
        party_df = matched.filter(pl.col("party") == party)
        results[party] = compute_correlations(party_df, xi_col, np_col)

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
    """Assign a quality label based on absolute Pearson r."""
    if abs_r >= STRONG_CORRELATION:
        return "strong"
    elif abs_r >= GOOD_CORRELATION:
        return "good"
    elif abs_r >= CONCERN_CORRELATION:
        return "moderate"
    else:
        return "concern"


# ── Outlier Detection ────────────────────────────────────────────────────────


def identify_outliers(
    matched: pl.DataFrame,
    xi_col: str = "xi_mean",
    np_col: str = "np_score",
    top_n: int = OUTLIER_TOP_N,
) -> pl.DataFrame:
    """Identify top-N outliers by z-score of (xi_mean - np_score) discrepancy.

    Both columns are z-standardized before computing the discrepancy,
    so the comparison is scale-invariant.
    """
    if matched.height < 3:
        return pl.DataFrame()

    xi = matched[xi_col].to_numpy().astype(float)
    np_scores = matched[np_col].to_numpy().astype(float)

    # Z-standardize both
    xi_z = (xi - np.mean(xi)) / np.std(xi) if np.std(xi) > 0 else xi * 0
    if np.std(np_scores) > 0:
        np_z = (np_scores - np.mean(np_scores)) / np.std(np_scores)
    else:
        np_z = np_scores * 0

    discrepancy = np.abs(xi_z - np_z)

    result = matched.with_columns(
        pl.Series("xi_z", xi_z),
        pl.Series("np_z", np_z),
        pl.Series("discrepancy_z", discrepancy),
    )

    return result.sort("discrepancy_z", descending=True).head(top_n)


# ── Session Overlap Detection ────────────────────────────────────────────────


def has_shor_mccarty_overlap(session_name: str) -> bool:
    """Check if a session name overlaps with Shor-McCarty coverage (84th-88th)."""
    return session_name in OVERLAPPING_BIENNIUMS
