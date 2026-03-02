"""Pure data logic for legislator profile deep-dives.

No plotting, no I/O, no report building. Computes scorecards, bill-type breakdowns,
defection analysis, voting neighbors, and surprising votes for individually notable
legislators detected by synthesis_detect.

Consumed by profiles.py (orchestration) and profiles_report.py (HTML sections).
"""

from dataclasses import dataclass

import polars as pl

try:
    from analysis.cross_session_data import normalize_name
except ModuleNotFoundError:
    from cross_session_data import normalize_name  # type: ignore[no-redef]

try:
    from analysis.synthesis_detect import detect_all
except ModuleNotFoundError:
    from synthesis_detect import detect_all  # type: ignore[no-redef]

# ── Constants ────────────────────────────────────────────────────────────────

HIGH_DISC_THRESHOLD = 1.5  # |beta_mean| > this → highly discriminating bill
LOW_DISC_THRESHOLD = 0.5  # |beta_mean| < this → low-discrimination bill
MIN_BILLS_PER_TIER = 3  # need at least this many bills per tier for breakdown
MAX_PROFILE_TARGETS = 8  # cap on number of legislators to profile

# Metrics to extract for scorecard, in display order.
# All metrics must be on a 0-1 scale (raw or percentile) so the bar chart
# is visually comparable. Raw dimensional values (IRT, PCA, UMAP) belong
# in the position plot, not here.
SCORECARD_METRICS = [
    ("xi_mean_percentile", "Ideological Rank", ".0%"),
    ("unity_score", "Party Unity (CQ)", ".0%"),
    ("loyalty_rate", "Clustering Loyalty", ".0%"),
    ("maverick_rate", "Maverick Rate", ".0%"),
    ("betweenness_percentile", "Network Influence", ".0%"),
    ("accuracy", "Prediction Accuracy", ".0%"),
]


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProfileTarget:
    """A legislator selected for deep-dive profiling."""

    slug: str
    full_name: str
    party: str
    district: str
    chamber: str
    role: str  # "Maverick", "Bridge-Builder", "Paradox", "Requested"
    title: str  # "Mark Schreiber (R-60)"
    subtitle: str  # one-sentence data-driven explanation


@dataclass(frozen=True)
class NameMatch:
    """Result of resolving a user-supplied name query to legislator slug(s)."""

    query: str  # original input, e.g., "Masterson"
    status: str  # "ok" | "ambiguous" | "no_match"
    matches: list[dict]  # [{slug, full_name, party, district, chamber}, ...]


@dataclass(frozen=True)
class BillTypeBreakdown:
    """Yea rates on high- vs low-discrimination bills for one legislator."""

    high_disc_yea_rate: float
    high_disc_n: int
    low_disc_yea_rate: float
    low_disc_n: int
    party_high_disc_yea_rate: float
    party_low_disc_yea_rate: float


# ── Target Selection ─────────────────────────────────────────────────────────


def gather_profile_targets(
    leg_dfs: dict[str, pl.DataFrame],
    extra_slugs: list[str] | None = None,
) -> list[ProfileTarget]:
    """Select legislators for profiling via synthesis detection + optional extras.

    Calls detect_all() to find mavericks, bridge-builders, and paradoxes, then
    converts each to a ProfileTarget. Extra slugs are added with role="Requested".
    Deduplicates by slug, caps at MAX_PROFILE_TARGETS.

    Returns list sorted by chamber then |xi_mean| descending.
    """
    notables = detect_all(leg_dfs)
    targets: dict[str, ProfileTarget] = {}

    # Add detected profiles (mavericks, bridges, paradoxes)
    for slug, notable in notables["profiles"].items():
        targets[slug] = ProfileTarget(
            slug=notable.slug,
            full_name=notable.full_name,
            party=notable.party,
            district=notable.district,
            chamber=notable.chamber,
            role=notable.role,
            title=notable.title,
            subtitle=notable.subtitle,
        )

    # Add paradoxes that aren't already in profiles
    for slug, paradox in notables["paradoxes"].items():
        if slug not in targets:
            targets[slug] = ProfileTarget(
                slug=paradox.slug,
                full_name=paradox.full_name,
                party=paradox.party,
                district=paradox.district,
                chamber=paradox.chamber,
                role=f"The {paradox.full_name.split()[-1]} Paradox",
                title=f"{paradox.full_name} ({paradox.party[0]}-{paradox.district})",
                subtitle=(
                    f"Ranked #{paradox.rank_high} on {paradox.metric_high_name} but "
                    f"lowest on {paradox.metric_low_name} — defects {paradox.direction}."
                ),
            )

    # Add extra user-requested slugs
    if extra_slugs:
        all_legs = _build_slug_lookup(leg_dfs)
        for slug in extra_slugs:
            if slug in targets:
                continue
            info = all_legs.get(slug)
            if info is None:
                continue
            targets[slug] = ProfileTarget(
                slug=slug,
                full_name=info["full_name"],
                party=info["party"],
                district=info["district"],
                chamber=info["chamber"],
                role="Requested",
                title=f"{info['full_name']} ({info['party'][0]}-{info['district']})",
                subtitle="User-requested profile.",
            )

    # Sort by chamber then |xi_mean| descending
    result = sorted(
        targets.values(),
        key=lambda t: (t.chamber, -_abs_xi(leg_dfs, t)),
    )
    return result[:MAX_PROFILE_TARGETS]


def _build_slug_lookup(leg_dfs: dict[str, pl.DataFrame]) -> dict[str, dict]:
    """Build a slug → {full_name, party, district, chamber} lookup."""
    lookup: dict[str, dict] = {}
    for chamber, df in leg_dfs.items():
        for row in df.select("legislator_slug", "full_name", "party", "district").to_dicts():
            lookup[row["legislator_slug"]] = {
                "full_name": row["full_name"],
                "party": row["party"],
                "district": row["district"],
                "chamber": chamber,
            }
    return lookup


def _build_name_lookup(
    leg_dfs: dict[str, pl.DataFrame],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Build full-name and last-name lookup tables from leg_dfs.

    Returns (full_name_lookup, last_name_lookup) where both map normalized names
    to lists of info dicts (to handle cross-chamber duplicates).
    """
    full_name_lookup: dict[str, list[dict]] = {}
    last_name_lookup: dict[str, list[dict]] = {}
    for chamber, df in leg_dfs.items():
        for row in df.select("legislator_slug", "full_name", "party", "district").to_dicts():
            info = {
                "slug": row["legislator_slug"],
                "full_name": row["full_name"],
                "party": row["party"],
                "district": row["district"],
                "chamber": chamber,
            }
            norm_full = normalize_name(row["full_name"])
            full_name_lookup.setdefault(norm_full, []).append(info)

            # Last name = last token of normalized full name
            last = norm_full.split()[-1] if norm_full else ""
            if last:
                last_name_lookup.setdefault(last, []).append(info)
    return full_name_lookup, last_name_lookup


def resolve_names(
    names: list[str],
    leg_dfs: dict[str, pl.DataFrame],
) -> list[NameMatch]:
    """Resolve user-supplied name queries to legislator slugs.

    Multi-stage matching:
    1. Exact case-insensitive match on full name
    2. Last-name-only match
    3. First-name disambiguation within last-name matches

    Returns one NameMatch per query.
    """
    if not names:
        return []

    full_lookup, last_lookup = _build_name_lookup(leg_dfs)
    results: list[NameMatch] = []

    for raw_query in names:
        query_norm = normalize_name(raw_query)
        if not query_norm:
            results.append(NameMatch(query=raw_query, status="no_match", matches=[]))
            continue

        # Stage 1: exact full-name match
        if query_norm in full_lookup:
            matches = full_lookup[query_norm]
            status = "ok" if len(matches) == 1 else "ambiguous"
            results.append(NameMatch(query=raw_query, status=status, matches=matches))
            continue

        # Stage 2: last-name-only match
        query_parts = query_norm.split()
        last_name = query_parts[-1]
        candidates = last_lookup.get(last_name, [])

        if not candidates:
            results.append(NameMatch(query=raw_query, status="no_match", matches=[]))
            continue

        if len(candidates) == 1:
            results.append(NameMatch(query=raw_query, status="ok", matches=candidates))
            continue

        # Stage 3: first-name disambiguation (if query has 2+ tokens)
        if len(query_parts) >= 2:
            first_name = query_parts[0]
            narrowed = [
                c for c in candidates if normalize_name(c["full_name"]).split()[0] == first_name
            ]
            if len(narrowed) == 1:
                results.append(NameMatch(query=raw_query, status="ok", matches=narrowed))
                continue
            if narrowed:
                # Still ambiguous but narrowed
                results.append(NameMatch(query=raw_query, status="ambiguous", matches=narrowed))
                continue

        # Multiple last-name matches, no disambiguation possible
        results.append(NameMatch(query=raw_query, status="ambiguous", matches=candidates))

    return results


def _abs_xi(leg_dfs: dict[str, pl.DataFrame], target: ProfileTarget) -> float:
    """Get |xi_mean| for sorting. Returns 0.0 if not found."""
    df = leg_dfs.get(target.chamber)
    if df is None or "xi_mean" not in df.columns:
        return 0.0
    row = df.filter(pl.col("legislator_slug") == target.slug)
    if row.height == 0:
        return 0.0
    val = row["xi_mean"][0]
    return abs(val) if val is not None else 0.0


# ── Scorecard ────────────────────────────────────────────────────────────────


def build_scorecard(leg_df: pl.DataFrame, slug: str) -> dict | None:
    """Extract all available metrics for one legislator plus party averages.

    Returns a flat dict with keys like "xi_mean", "xi_mean_party_avg",
    "xi_mean_label", "xi_mean_fmt" for each metric present. Returns None
    if the slug is not found.
    """
    row = leg_df.filter(pl.col("legislator_slug") == slug)
    if row.height == 0:
        return None

    r = row.to_dicts()[0]
    party = r["party"]
    party_df = leg_df.filter(pl.col("party") == party)

    result: dict = {}
    for col, label, fmt in SCORECARD_METRICS:
        if col not in leg_df.columns:
            continue
        val = r.get(col)
        if val is None:
            continue
        result[col] = val
        result[f"{col}_label"] = label
        result[f"{col}_fmt"] = fmt
        # Party average
        party_avg = party_df[col].mean()
        result[f"{col}_party_avg"] = party_avg

    return result if result else None


# ── Bill Type Breakdown ──────────────────────────────────────────────────────


def compute_bill_type_breakdown(
    slug: str,
    bill_params: pl.DataFrame,
    votes_long: pl.DataFrame,
    party: str,
    party_slugs: list[str],
) -> BillTypeBreakdown | None:
    """Compute Yea rates on high- vs low-discrimination bills.

    Uses IRT bill_params (beta_mean) to classify bills into high/low tiers.
    Joins with votes_long to compute Yea rates per tier for the target and party.

    Args:
        slug: Legislator slug.
        bill_params: DataFrame with vote_id, beta_mean columns.
        votes_long: DataFrame with legislator_slug, vote_id, vote_binary columns.
        party: Party name of the target legislator.
        party_slugs: All slugs in the same party (same chamber).

    Returns None if fewer than MIN_BILLS_PER_TIER bills in either tier.
    """
    if "beta_mean" not in bill_params.columns or "vote_id" not in bill_params.columns:
        return None

    # Classify bills by discrimination
    high_disc = bill_params.filter(pl.col("beta_mean").abs() > HIGH_DISC_THRESHOLD)
    low_disc = bill_params.filter(pl.col("beta_mean").abs() < LOW_DISC_THRESHOLD)

    high_ids = set(high_disc["vote_id"].to_list())
    low_ids = set(low_disc["vote_id"].to_list())

    # Compute yea rate for the target legislator
    target_votes = votes_long.filter(pl.col("legislator_slug") == slug)

    target_high = target_votes.filter(pl.col("vote_id").is_in(high_ids))
    target_low = target_votes.filter(pl.col("vote_id").is_in(low_ids))

    if target_high.height < MIN_BILLS_PER_TIER or target_low.height < MIN_BILLS_PER_TIER:
        return None

    high_yea_rate = target_high["vote_binary"].mean()
    low_yea_rate = target_low["vote_binary"].mean()

    # Compute party averages
    party_votes = votes_long.filter(pl.col("legislator_slug").is_in(party_slugs))
    party_high = party_votes.filter(pl.col("vote_id").is_in(high_ids))
    party_low = party_votes.filter(pl.col("vote_id").is_in(low_ids))

    party_high_rate = party_high["vote_binary"].mean() if party_high.height > 0 else 0.0
    party_low_rate = party_low["vote_binary"].mean() if party_low.height > 0 else 0.0

    return BillTypeBreakdown(
        high_disc_yea_rate=high_yea_rate,
        high_disc_n=target_high.height,
        low_disc_yea_rate=low_yea_rate,
        low_disc_n=target_low.height,
        party_high_disc_yea_rate=party_high_rate,
        party_low_disc_yea_rate=party_low_rate,
    )


# ── Defection Analysis ───────────────────────────────────────────────────────


def find_defection_bills(
    slug: str,
    votes_long: pl.DataFrame,
    rollcalls: pl.DataFrame,
    party: str,
    party_slugs: list[str],
    n: int = 15,
) -> pl.DataFrame:
    """Find bills where this legislator voted against their party majority.

    For each vote_id, determines party majority direction (Yea if >50% voted Yea).
    Returns bills where the target disagreed, sorted by closeness of party margin.

    Returns a DataFrame with columns: bill_number, short_title, motion,
    legislator_vote, party_majority_vote, party_yea_pct. Empty if no defections.
    """
    # Get target's votes
    target_votes = votes_long.filter(pl.col("legislator_slug") == slug).select(
        "vote_id", "vote_binary"
    )

    if target_votes.height == 0:
        return _empty_defection_df()

    # Compute party majority per vote_id
    party_votes = votes_long.filter(pl.col("legislator_slug").is_in(party_slugs))
    party_agg = party_votes.group_by("vote_id").agg(
        pl.col("vote_binary").mean().alias("party_yea_pct"),
    )
    party_agg = party_agg.with_columns(
        pl.when(pl.col("party_yea_pct") > 0.5).then(1).otherwise(0).alias("party_majority_vote"),
    )

    # Join target votes with party majority
    joined = target_votes.join(party_agg, on="vote_id", how="inner")

    # Filter to defections
    defections = joined.filter(pl.col("vote_binary") != pl.col("party_majority_vote"))

    if defections.height == 0:
        return _empty_defection_df()

    # Sort by closeness (party_yea_pct closest to 0.5)
    defections = defections.with_columns(
        (pl.col("party_yea_pct") - 0.5).abs().alias("_margin")
    ).sort("_margin")

    # Join with rollcalls for bill metadata
    rc_cols = ["vote_id"]
    for col in ("bill_number", "short_title", "motion"):
        if col in rollcalls.columns:
            rc_cols.append(col)

    result = defections.head(n).join(
        rollcalls.select(rc_cols).unique(subset=["vote_id"]),
        on="vote_id",
        how="left",
    )

    # Ensure expected metadata columns exist (rollcalls may lack them)
    for col_name in ("bill_number", "short_title", "motion"):
        if col_name not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Utf8).alias(col_name))

    return result.select(
        pl.col("bill_number").fill_null("Unknown"),
        pl.col("short_title").fill_null(""),
        pl.col("motion").fill_null(""),
        pl.when(pl.col("vote_binary") == 1)
        .then(pl.lit("Yea"))
        .otherwise(pl.lit("Nay"))
        .alias("legislator_vote"),
        pl.when(pl.col("party_majority_vote") == 1)
        .then(pl.lit("Yea"))
        .otherwise(pl.lit("Nay"))
        .alias("party_majority_vote"),
        (pl.col("party_yea_pct") * 100).round(1).alias("party_yea_pct"),
    )


def _empty_defection_df() -> pl.DataFrame:
    """Return an empty DataFrame with the defection schema."""
    return pl.DataFrame(
        schema={
            "bill_number": pl.Utf8,
            "short_title": pl.Utf8,
            "motion": pl.Utf8,
            "legislator_vote": pl.Utf8,
            "party_majority_vote": pl.Utf8,
            "party_yea_pct": pl.Float64,
        }
    )


# ── Voting Neighbors ─────────────────────────────────────────────────────────


def find_voting_neighbors(
    slug: str,
    votes_long: pl.DataFrame,
    leg_df: pl.DataFrame,
    n: int = 5,
) -> dict | None:
    """Find the most similar and most different legislators by agreement rate.

    Builds pairwise simple agreement from the vote matrix (same chamber only).
    Returns {"closest": [...], "most_different": [...]} with n entries each,
    or None if slug not found.
    """
    # Build a vote matrix: rows=vote_id, cols=legislator_slug, vals=vote_binary
    chamber_slugs = leg_df["legislator_slug"].to_list()
    if slug not in chamber_slugs:
        return None

    # Filter to chamber slugs only
    chamber_votes = votes_long.filter(pl.col("legislator_slug").is_in(chamber_slugs))

    # Pivot to wide
    matrix = chamber_votes.pivot(on="legislator_slug", index="vote_id", values="vote_binary")

    if slug not in matrix.columns:
        return None

    target_col = matrix[slug]

    # Compute pairwise agreement
    agreements = []
    name_lookup = dict(
        zip(
            leg_df["legislator_slug"].to_list(),
            leg_df.select("full_name", "party").to_dicts(),
        )
    )

    for other_slug in chamber_slugs:
        if other_slug == slug or other_slug not in matrix.columns:
            continue

        other_col = matrix[other_slug]
        # Only compare where both voted
        mask = target_col.is_not_null() & other_col.is_not_null()
        if mask.sum() < 5:
            continue

        t = target_col.filter(mask)
        o = other_col.filter(mask)
        agreement = (t == o).mean()

        info = name_lookup.get(other_slug, {})
        agreements.append(
            {
                "slug": other_slug,
                "full_name": info.get("full_name", other_slug),
                "party": info.get("party", "Unknown"),
                "agreement": float(agreement),
            }
        )

    if not agreements:
        return None

    # Sort for closest and most_different
    by_agreement = sorted(agreements, key=lambda x: x["agreement"], reverse=True)
    closest = by_agreement[:n]
    most_different = sorted(agreements, key=lambda x: x["agreement"])[:n]

    return {"closest": closest, "most_different": most_different}


# ── Surprising Votes ─────────────────────────────────────────────────────────


def find_legislator_surprising_votes(
    slug: str,
    surprising_votes_df: pl.DataFrame | None,
    n: int = 10,
) -> pl.DataFrame | None:
    """Filter the prediction phase's surprising votes to this legislator.

    Returns top N by confidence_error, or None if no data.
    """
    if surprising_votes_df is None or surprising_votes_df.height == 0:
        return None

    filtered = surprising_votes_df.filter(pl.col("legislator_slug") == slug)
    if filtered.height == 0:
        return None

    return filtered.sort("confidence_error", descending=True).head(n)


# ── Full Voting Record ──────────────────────────────────────────────────────


def build_full_voting_record(
    slug: str,
    votes_long: pl.DataFrame,
    rollcalls: pl.DataFrame,
    party: str,
    party_slugs: list[str],
) -> pl.DataFrame:
    """Build a complete voting record for one legislator.

    Returns every Yea/Nay vote cast, joined with bill metadata and party context.
    Columns: date, bill_number, short_title, motion, vote, party_majority,
    with_party, passed.

    Args:
        slug: Legislator slug.
        votes_long: Long-form vote DataFrame (legislator_slug, vote_id, vote_binary).
        rollcalls: Roll calls DataFrame with bill metadata.
        party: Party name of the target legislator.
        party_slugs: All slugs in the same party (same chamber).

    Returns empty DataFrame if no votes found.
    """
    target_votes = votes_long.filter(pl.col("legislator_slug") == slug)
    if target_votes.height == 0:
        return _empty_voting_record()

    # Compute party majority per vote_id
    party_votes = votes_long.filter(pl.col("legislator_slug").is_in(party_slugs))
    party_agg = party_votes.group_by("vote_id").agg(
        pl.col("vote_binary").mean().alias("party_yea_pct"),
    )
    party_agg = party_agg.with_columns(
        pl.when(pl.col("party_yea_pct") > 0.5)
        .then(pl.lit("Yea"))
        .otherwise(pl.lit("Nay"))
        .alias("party_majority"),
    )

    # Join target votes with party majority
    joined = target_votes.join(party_agg, on="vote_id", how="left")

    # Add vote label and with_party flag
    joined = joined.with_columns(
        pl.when(pl.col("vote_binary") == 1)
        .then(pl.lit("Yea"))
        .otherwise(pl.lit("Nay"))
        .alias("vote"),
    )
    joined = joined.with_columns(
        (pl.col("vote") == pl.col("party_majority")).alias("with_party"),
    )

    # Extract date from vote_id (format: je_YYYYMMDDHHmmss)
    joined = joined.with_columns(
        pl.col("vote_id").str.extract(r"_(\d{4})(\d{2})(\d{2})", 1).alias("_year"),
        pl.col("vote_id").str.extract(r"_(\d{4})(\d{2})(\d{2})", 2).alias("_month"),
        pl.col("vote_id").str.extract(r"_(\d{4})(\d{2})(\d{2})", 3).alias("_day"),
    )
    joined = joined.with_columns(
        (pl.col("_year") + "-" + pl.col("_month") + "-" + pl.col("_day")).alias("date"),
    )

    # Join with rollcalls for bill metadata
    rc_cols = ["vote_id"]
    for col in ("bill_number", "short_title", "motion", "passed"):
        if col in rollcalls.columns:
            rc_cols.append(col)

    result = joined.join(
        rollcalls.select(rc_cols).unique(subset=["vote_id"]),
        on="vote_id",
        how="left",
    )

    # Ensure expected columns exist
    for col_name in ("bill_number", "short_title", "motion"):
        if col_name not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Utf8).alias(col_name))
    if "passed" not in result.columns:
        result = result.with_columns(pl.lit(None).cast(pl.Boolean).alias("passed"))

    return result.select(
        pl.col("date"),
        pl.col("bill_number").fill_null("Unknown"),
        pl.col("short_title").fill_null(""),
        pl.col("motion").fill_null(""),
        "vote",
        "party_majority",
        "with_party",
        "passed",
    ).sort("date", descending=True)


def _empty_voting_record() -> pl.DataFrame:
    """Return an empty DataFrame with the voting record schema."""
    return pl.DataFrame(
        schema={
            "date": pl.Utf8,
            "bill_number": pl.Utf8,
            "short_title": pl.Utf8,
            "motion": pl.Utf8,
            "vote": pl.Utf8,
            "party_majority": pl.Utf8,
            "with_party": pl.Boolean,
            "passed": pl.Boolean,
        }
    )
