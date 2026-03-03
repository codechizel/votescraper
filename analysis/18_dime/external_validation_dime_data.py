"""Pure data logic for external validation against DIME/CFscores.

All functions are pure (no I/O, no network). Parsing, name normalization,
matching, and correlation computation live here so they can be tested with
synthetic data.

DIME dataset: campaign-finance ideology scores for candidates, published by
Stanford University Libraries (ODC-BY license). Overlaps our 84th-89th
bienniums (2011-2022), providing a second external validation source
independent of Shor-McCarty's roll-call-based scores.
"""

import polars as pl

# Reuse shared infrastructure from Phase 14
try:
    from analysis.external_validation_data import (
        CONCERN_CORRELATION,
        GOOD_CORRELATION,
        MIN_MATCHED,
        OUTLIER_TOP_N,
        STRONG_CORRELATION,
        compute_correlations,
        compute_intra_party_correlations,
        identify_outliers,
        normalize_our_name,
    )
except ModuleNotFoundError:
    from external_validation_data import (  # type: ignore[no-redef]
        CONCERN_CORRELATION,
        GOOD_CORRELATION,
        MIN_MATCHED,
        OUTLIER_TOP_N,
        STRONG_CORRELATION,
        compute_correlations,
        compute_intra_party_correlations,
        identify_outliers,
        normalize_our_name,
    )

# Re-export for convenient imports by downstream modules
__all__ = [
    "CONCERN_CORRELATION",
    "GOOD_CORRELATION",
    "MIN_MATCHED",
    "OUTLIER_TOP_N",
    "STRONG_CORRELATION",
    "compute_correlations",
    "compute_intra_party_correlations",
    "identify_outliers",
    "normalize_our_name",
    "DIME_CACHE_PATH",
    "DIME_PARTY_MAP",
    "DIME_OVERLAPPING_BIENNIUMS",
    "MIN_GIVERS",
    "normalize_dime_name",
    "parse_dime_kansas",
    "filter_dime_to_biennium",
    "match_dime_legislators",
    "has_dime_overlap",
]

# ── Constants ────────────────────────────────────────────────────────────────

DIME_CACHE_PATH = "data/external/dime_recipients_1979_2024.csv"

DIME_PARTY_MAP: dict[str, str] = {
    "100": "Democrat",
    "200": "Republican",
    "328": "Libertarian",
    "500": "Independent",
}

# Biennium -> list of election cycles that feed into that biennium.
# Kansas House: 2-year terms, elected every even year.
# Kansas Senate: 4-year terms, staggered — half elected each even year.
# A legislator elected in cycle 2020 serves in the 89th (2021-2022).
DIME_OVERLAPPING_BIENNIUMS: dict[str, list[int]] = {
    "84th_2011-2012": [2010, 2012],
    "85th_2013-2014": [2012, 2014],
    "86th_2015-2016": [2014, 2016],
    "87th_2017-2018": [2016, 2018],
    "88th_2019-2020": [2018, 2020],
    "89th_2021-2022": [2020, 2022],
}

MIN_GIVERS = 5  # Minimum unique donors for reliable CFscore

# Populated as edge cases are found during real matching runs.
NICKNAME_MAP: dict[str, str] = {}


# ── Name Normalization ───────────────────────────────────────────────────────


def normalize_dime_name(lname: str, fname: str) -> str:
    """Normalize a DIME name from separate lname/fname fields.

    DIME provides lowercase separate fields: lname="hodge", fname="timothy".
    Output: "timothy hodge" (lowercase, stripped).

    Handles edge cases: empty fields, extra whitespace, periods.
    """
    first = (fname or "").strip().replace(".", "").lower()
    last = (lname or "").strip().replace(".", "").lower()

    if not first and not last:
        return ""
    if not first:
        return last
    if not last:
        return first

    # Take only the first token of fname (drop middle names)
    first_tokens = first.split()
    first = first_tokens[0] if first_tokens else first

    return f"{first} {last}".strip()


# ── DIME Parsing ─────────────────────────────────────────────────────────────


def parse_dime_kansas(path: str) -> pl.DataFrame:
    """Read the DIME CSV and filter to Kansas state legislators.

    Filters: state == "KS", seat in ("state:lower", "state:upper").
    Casts types, maps party codes, adds normalized_name column.

    Returns DataFrame with columns: name, lname, fname, party, party_code,
    state, seat, district, cycle, ico_status, recipient_cfscore,
    recipient_cfscore_dyn, num_givers, bonica_rid, normalized_name.
    """
    df = pl.read_csv(
        path,
        infer_schema_length=10000,
        null_values=["", "NA"],
        truncate_ragged_lines=True,
        schema_overrides={"party": pl.Utf8, "party.orig": pl.Utf8},
    )

    # Normalize column names: dots to underscores for Polars ergonomics
    rename_map = {
        "recipient.cfscore": "recipient_cfscore",
        "recipient.cfscore.dyn": "recipient_cfscore_dyn",
        "num.givers": "num_givers",
        "ico.status": "ico_status",
        "bonica.rid": "bonica_rid",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename({old: new})

    # Filter to Kansas state legislators
    df = df.filter(
        (pl.col("state") == "KS") & (pl.col("seat").is_in(["state:lower", "state:upper"]))
    )

    if df.height == 0:
        return df

    # Cast key columns
    df = df.with_columns(
        pl.col("cycle").cast(pl.Int64, strict=False),
        pl.col("num_givers").cast(pl.Int64, strict=False),
        pl.col("recipient_cfscore").cast(pl.Float64, strict=False),
        pl.col("recipient_cfscore_dyn").cast(pl.Float64, strict=False),
        pl.col("party").cast(pl.Utf8),
    )

    # Map party codes to names
    df = df.with_columns(
        pl.col("party").replace(DIME_PARTY_MAP, default="Other").alias("party_name"),
    )

    # Add normalized name
    df = df.with_columns(
        pl.struct(["lname", "fname"])
        .map_elements(
            lambda s: normalize_dime_name(s["lname"] or "", s["fname"] or ""),
            return_dtype=pl.Utf8,
        )
        .alias("normalized_name")
    )

    # Select and order columns
    keep_cols = [
        "name",
        "lname",
        "fname",
        "party",
        "party_name",
        "state",
        "seat",
        "district",
        "cycle",
        "ico_status",
        "recipient_cfscore",
        "recipient_cfscore_dyn",
        "num_givers",
        "bonica_rid",
        "normalized_name",
    ]
    available = [c for c in keep_cols if c in df.columns]
    return df.select(available)


# ── Biennium Filtering ───────────────────────────────────────────────────────


def filter_dime_to_biennium(
    dime_df: pl.DataFrame,
    cycles: list[int],
    min_givers: int = MIN_GIVERS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Filter DIME data to a biennium's election cycles.

    Filters:
    1. cycle in the given list
    2. Incumbents only (ico_status == "I")
    3. num_givers >= min_givers (reliable CFscores)
    4. Valid recipient_cfscore (not null)
    5. Deduplicate by normalized_name (keep most recent cycle)

    Returns (house_df, senate_df) split by seat.
    """
    if dime_df.height == 0:
        return dime_df, dime_df

    filtered = dime_df.filter(pl.col("cycle").is_in(cycles))

    # Incumbents only — we're comparing to legislators who actually voted
    if "ico_status" in filtered.columns:
        filtered = filtered.filter(pl.col("ico_status") == "I")

    # Minimum donor threshold
    if "num_givers" in filtered.columns:
        filtered = filtered.filter(pl.col("num_givers") >= min_givers)

    # Valid CFscore
    filtered = filtered.filter(pl.col("recipient_cfscore").is_not_null())

    # Deduplicate by name (keep most recent cycle for returning legislators)
    filtered = filtered.sort("cycle", descending=True).unique(
        subset=["normalized_name"], keep="first"
    )

    # Split by chamber
    house_df = filtered.filter(pl.col("seat") == "state:lower")
    senate_df = filtered.filter(pl.col("seat") == "state:upper")

    return house_df, senate_df


# ── Legislator Matching ──────────────────────────────────────────────────────


def match_dime_legislators(
    our_df: pl.DataFrame,
    dime_df: pl.DataFrame,
    chamber: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Match our legislators to DIME legislators by normalized name.

    Phase 1: Exact normalized name match.
    Phase 2: Last-name-only match with deduplication.

    Args:
        our_df: Our IRT ideal points (must have full_name, xi_mean, district, party).
        dime_df: DIME data filtered to the correct chamber/biennium.
        chamber: "House" or "Senate" for the unmatched report.

    Returns:
        (matched_df, unmatched_report_df)
        matched_df has columns from both datasets.
        unmatched_report_df lists unmatched legislators from both sides.
    """
    if our_df.height == 0 or dime_df.height == 0:
        return pl.DataFrame(), _build_unmatched_report(our_df, dime_df, chamber)

    # Ensure our_df has normalized names
    if "normalized_name" not in our_df.columns:
        our_df = our_df.with_columns(
            pl.col("full_name")
            .map_elements(normalize_our_name, return_dtype=pl.Utf8)
            .alias("normalized_name")
        )

    # Apply nickname map
    if NICKNAME_MAP:
        our_df = our_df.with_columns(
            pl.col("normalized_name")
            .map_elements(lambda n: NICKNAME_MAP.get(n, n), return_dtype=pl.Utf8)
            .alias("normalized_name")
        )

    # Phase 1: Exact normalized name match
    dime_join_cols = ["normalized_name", "recipient_cfscore", "name"]
    if "recipient_cfscore_dyn" in dime_df.columns:
        dime_join_cols.append("recipient_cfscore_dyn")
    if "num_givers" in dime_df.columns:
        dime_join_cols.append("num_givers")

    dime_select = dime_df.select([c for c in dime_join_cols if c in dime_df.columns]).rename(
        {"name": "dime_name"}
    )

    matched = our_df.join(dime_select, on="normalized_name", how="inner")

    matched_our_names = set(matched["normalized_name"].to_list())

    # Phase 2: Last-name-only match for unmatched legislators
    unmatched_ours = our_df.filter(~pl.col("normalized_name").is_in(matched_our_names))
    unmatched_dime_names = set(dime_df["normalized_name"].to_list()) - matched_our_names

    if unmatched_ours.height > 0 and unmatched_dime_names:
        phase2_matches = _phase2_last_name_match(
            unmatched_ours,
            dime_df.filter(pl.col("normalized_name").is_in(list(unmatched_dime_names))),
            dime_join_cols,
        )
        if phase2_matches.height > 0:
            matched = pl.concat([matched, phase2_matches], how="diagonal")
            matched_our_names = set(matched["normalized_name"].to_list())

    # Build unmatched report
    unmatched_ours_final = our_df.filter(~pl.col("normalized_name").is_in(matched_our_names))
    matched_dime_names = (
        set(matched["dime_name"].to_list()) if "dime_name" in matched.columns else set()
    )
    unmatched_dime_final = dime_df.filter(~pl.col("name").is_in(matched_dime_names))

    unmatched_report = _build_unmatched_report(unmatched_ours_final, unmatched_dime_final, chamber)

    return matched, unmatched_report


def _parse_dime_district(dime_df: pl.DataFrame) -> pl.DataFrame:
    """Add ``_dime_district`` Int64 column parsed from DIME's district string.

    DIME district formats vary: ``"KS-113"``, ``"KS01"``, ``"KS-7"``, ``"27"``.
    Extracts the trailing numeric portion and casts to Int64.
    """
    if "district" not in dime_df.columns:
        return dime_df.with_columns(pl.lit(None).cast(pl.Int64).alias("_dime_district"))

    return dime_df.with_columns(
        pl.col("district")
        .cast(pl.Utf8)
        .str.replace(r"^[A-Za-z]*-?0*", "")
        .cast(pl.Int64, strict=False)
        .alias("_dime_district")
    )


def _deduplicate_with_district(
    candidates: pl.DataFrame,
    ext_district_col: str,
) -> pl.DataFrame:
    """Deduplicate last-name matches using district tiebreaker.

    - Single candidate per legislator → kept as-is.
    - Multiple candidates → prefer the one whose district matches; if none match, reject.
    """
    counts = candidates.group_by("normalized_name").agg(pl.len().alias("_n"))

    single_names = counts.filter(pl.col("_n") == 1)["normalized_name"].to_list()
    singles = candidates.filter(pl.col("normalized_name").is_in(single_names))

    multi_names = counts.filter(pl.col("_n") > 1)["normalized_name"].to_list()
    if not multi_names:
        return singles

    multi = candidates.filter(pl.col("normalized_name").is_in(multi_names))

    has_district = "district" in multi.columns and ext_district_col in multi.columns
    if has_district:
        district_matched = multi.filter(pl.col("district") == pl.col(ext_district_col))
        district_matched = district_matched.unique(subset=["normalized_name"])
    else:
        district_matched = pl.DataFrame()

    if district_matched.height > 0:
        return pl.concat([singles, district_matched], how="diagonal")
    return singles


def _phase2_last_name_match(
    our_df: pl.DataFrame,
    dime_df: pl.DataFrame,
    dime_join_cols: list[str],
) -> pl.DataFrame:
    """Phase 2: Match by last name with district tiebreaker for ambiguous cases.

    When multiple DIME candidates share a last name with one of our legislators:
    - If one candidate's district matches → use that match.
    - If no candidate's district matches → reject (no match is better than wrong match).
    Single-candidate matches are kept as-is regardless of district.
    """
    if our_df.height == 0 or dime_df.height == 0:
        return pl.DataFrame()

    # Add DIME district column for tiebreaking
    dime_df = _parse_dime_district(dime_df)

    # Extract last names
    our_with_last = our_df.with_columns(
        pl.col("normalized_name")
        .map_elements(lambda n: n.split()[-1] if n.split() else "", return_dtype=pl.Utf8)
        .alias("_last_name")
    )
    dime_with_last = dime_df.with_columns(
        pl.col("normalized_name")
        .map_elements(lambda n: n.split()[-1] if n.split() else "", return_dtype=pl.Utf8)
        .alias("_last_name")
    )

    # Build DIME select for join
    dime_select_cols = ["_last_name", "normalized_name", "name", "_dime_district"]
    for col in dime_join_cols:
        if col not in ("normalized_name", "name") and col in dime_with_last.columns:
            dime_select_cols.append(col)
    dime_select_cols = [c for c in dime_select_cols if c in dime_with_last.columns]

    # Join on last name
    candidates = our_with_last.join(
        dime_with_last.select(dime_select_cols).rename(
            {"normalized_name": "dime_normalized_name", "name": "dime_name"}
        ),
        on="_last_name",
        how="inner",
    )

    if candidates.height == 0:
        return pl.DataFrame()

    # District tiebreaker for ambiguous last-name matches
    deduped = _deduplicate_with_district(candidates, "_dime_district")

    # Drop helper columns
    drop_cols = ["_last_name", "dime_normalized_name", "_dime_district"]
    drop_cols = [c for c in drop_cols if c in deduped.columns]
    return deduped.drop(drop_cols)


def _build_unmatched_report(
    our_unmatched: pl.DataFrame,
    dime_unmatched: pl.DataFrame,
    chamber: str,
) -> pl.DataFrame:
    """Build a report of unmatched legislators from both sides."""
    rows: list[dict] = []

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

    if dime_unmatched.height > 0 and "name" in dime_unmatched.columns:
        for row in dime_unmatched.iter_rows(named=True):
            rows.append(
                {
                    "source": "dime",
                    "name": row.get("name", ""),
                    "normalized": row.get("normalized_name", ""),
                    "chamber": chamber,
                    "party": row.get("party_name", ""),
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


# ── Session Overlap Detection ────────────────────────────────────────────────


def has_dime_overlap(session_name: str) -> bool:
    """Check if a session name overlaps with DIME coverage (84th-89th)."""
    return session_name in DIME_OVERLAPPING_BIENNIUMS
