"""
Data-driven detection of notable legislators for the synthesis report.

Pure data logic — no plotting, no report building. Detects mavericks,
bridge-builders, and metric paradoxes from upstream analysis DataFrames.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

# ── Detection Threshold Constants ──────────────────────────────────────────

UNITY_SKIP_THRESHOLD: float = 0.95
"""Maverick: skip detection if all party unity scores exceed this value."""

BRIDGE_SD_TOLERANCE: float = 1.0
"""Bridge-builder: candidate must be within this many SDs of the cross-party midpoint."""

PARADOX_RANK_GAP: float = 0.5
"""Paradox: minimum percentile rank gap between IRT and loyalty to flag."""

PARADOX_MIN_PARTY_SIZE: int = 5
"""Paradox: minimum legislators in the majority party for detection."""


@dataclass(frozen=True)
class NotableLegislator:
    slug: str
    full_name: str
    party: str
    district: str
    chamber: str
    role: str  # "House Maverick", "Senate Bridge-Builder"
    title: str  # "Mark Schreiber (R-60)"
    subtitle: str  # one-sentence data-driven explanation
    reason: str  # machine-readable: "maverick" | "bridge" | "paradox"


@dataclass(frozen=True)
class ParadoxCase:
    slug: str
    full_name: str
    party: str
    district: str
    chamber: str
    metric_high_name: str  # "IRT ideology"
    metric_low_name: str  # "clustering loyalty"
    rank_high: int  # 1 = most extreme
    n_in_party: int
    raw_values: dict  # {"xi_mean": 4.17, "loyalty_rate": 0.42, "unity_score": 0.92}
    direction: str  # "rightward" or "leftward"


def ideology_label(party: str, xi_mean: float) -> str:
    """Human-readable label for ideological direction."""
    if party == "Republican" and xi_mean > 0:
        return "conservative"
    if party == "Democrat" and xi_mean < 0:
        return "liberal"
    return "moderate"


def _majority_party(leg_df: pl.DataFrame) -> str | None:
    """Return the party with the most legislators in the chamber."""
    counts = leg_df.group_by("party").len().sort("len", descending=True)
    if counts.height == 0:
        return None
    return counts["party"][0]


def detect_chamber_maverick(
    leg_df: pl.DataFrame,
    party: str,
    chamber: str,
    *,
    percentile: float | None = None,
) -> NotableLegislator | None:
    """Detect the maverick in a given party: lowest unity_score.

    Ties broken by highest weighted_maverick. Returns None if all unity
    exceed :data:`UNITY_SKIP_THRESHOLD` or if the required columns are missing.

    Args:
        chamber: lowercase key matching leg_dfs dict ("house" or "senate").
        percentile: If set (e.g. 0.10), select the bottom Nth percentile
            by unity score rather than a single lowest + threshold check.
            Default ``None`` preserves the original behavior.
    """
    if "unity_score" not in leg_df.columns:
        return None

    party_df = leg_df.filter((pl.col("party") == party) & pl.col("unity_score").is_not_null())
    if party_df.height == 0:
        return None

    if percentile is not None:
        # Percentile-based: select bottom N% by unity
        cutoff = float(np.percentile(party_df["unity_score"].to_numpy(), percentile * 100))
        candidates = party_df.filter(pl.col("unity_score") <= cutoff)
        if candidates.height == 0:
            return None
        # Among candidates, pick the lowest unity (ties: highest weighted_maverick)
        sort_cols = ["unity_score"]
        sort_desc = [False]
        if "weighted_maverick" in leg_df.columns:
            sort_cols.append("weighted_maverick")
            sort_desc.append(True)
        candidate = candidates.sort(sort_cols, descending=sort_desc).head(1).to_dicts()[0]
    else:
        # Original behavior: skip if everyone is highly disciplined
        min_unity = party_df["unity_score"].min()
        if min_unity is not None and min_unity > UNITY_SKIP_THRESHOLD:
            return None

        # Sort: lowest unity first, then highest weighted_maverick as tiebreaker
        sort_cols = ["unity_score"]
        sort_desc = [False]
        if "weighted_maverick" in leg_df.columns:
            sort_cols.append("weighted_maverick")
            sort_desc.append(True)

        candidate = party_df.sort(sort_cols, descending=sort_desc).head(1).to_dicts()[0]

    # Compute party average unity for subtitle
    avg_unity = party_df["unity_score"].mean()
    unity = candidate["unity_score"]
    chamber_title = chamber.title()

    return NotableLegislator(
        slug=candidate["legislator_slug"],
        full_name=candidate["full_name"],
        party=party,
        district=candidate["district"],
        chamber=chamber,
        role=f"{chamber_title} Maverick",
        title=f"{candidate['full_name']} ({party[0]}-{candidate['district']})",
        subtitle=(
            f"The {party} most likely to break ranks — "
            f"party unity {unity:.0%} vs. party average {avg_unity:.0%}."
        ),
        reason="maverick",
    )


def detect_bridge_builder(
    leg_df: pl.DataFrame,
    chamber: str,
    *,
    network_manifest: dict | None = None,
) -> NotableLegislator | None:
    """Detect the bridge-builder: highest centrality near the cross-party midpoint.

    When the co-voting graph is connected (n_components == 1), uses betweenness
    centrality (the standard measure). When disconnected (n_components >= 2),
    betweenness is mostly zeros, so falls back to harmonic centrality + cross-party
    edge fraction. Returns None if required columns are missing.

    Args:
        leg_df: Unified legislator DataFrame with centrality columns.
        chamber: Lowercase key matching leg_dfs dict ("house" or "senate").
        network_manifest: Optional network phase manifest with {chamber}_n_components.
    """
    if "xi_mean" not in leg_df.columns:
        return None

    # Determine graph connectivity from manifest
    n_components = 1  # assume connected unless manifest says otherwise
    if network_manifest is not None:
        n_components = network_manifest.get(f"{chamber}_n_components", 1) or 1

    # Choose ranking metric based on connectivity
    disconnected = n_components >= 2
    if disconnected:
        # Harmonic centrality is finite for disconnected graphs; cross-party fraction
        # measures actual cross-party bridging
        if "harmonic" not in leg_df.columns:
            # Fall back to betweenness if harmonic not available (old data)
            if "betweenness" not in leg_df.columns:
                return None
            rank_col = "betweenness"
        else:
            rank_col = "harmonic"
    else:
        if "betweenness" not in leg_df.columns:
            return None
        rank_col = "betweenness"

    parties = leg_df["party"].unique().to_list()
    if len(parties) < 2:
        return None

    # Cross-party midpoint: midpoint between median of each party's xi_mean
    medians = {}
    for p in parties:
        med = leg_df.filter(pl.col("party") == p)["xi_mean"].median()
        if med is not None:
            medians[p] = med

    if len(medians) < 2:
        return None

    midpoint = sum(medians.values()) / len(medians)
    overall_sd = leg_df["xi_mean"].std()
    if overall_sd is None or overall_sd == 0:
        overall_sd = 1.0

    # Candidates within BRIDGE_SD_TOLERANCE SDs of the midpoint
    near_center = leg_df.filter(
        (pl.col("xi_mean") - midpoint).abs() <= BRIDGE_SD_TOLERANCE * overall_sd
    )

    if near_center.height > 0:
        candidate_df = near_center.sort(rank_col, descending=True).head(1)
    else:
        # Fallback: highest centrality regardless
        candidate_df = leg_df.sort(rank_col, descending=True).head(1)

    if candidate_df.height == 0:
        return None

    c = candidate_df.to_dicts()[0]
    party = c["party"]
    xi = c["xi_mean"]

    # Determine the "other" party for subtitle
    other_parties = [p for p in parties if p != party]
    other_party = other_parties[0] if other_parties else "the other party"

    # Determine party median for context
    party_median = medians.get(party, 0)

    # Role label depends on connectivity
    if disconnected:
        role = f"{chamber.title()} Within-Party Connector"
        subtitle = (
            f"A {party} whose IRT score ({xi:.2f}) places them closer to "
            f"{other_party}s than to their own party's median ({party_median:.2f}). "
            f"Graph is disconnected ({n_components} components) — ranked by harmonic "
            f"centrality, not betweenness."
        )
    else:
        role = f"{chamber.title()} Bridge-Builder"
        subtitle = (
            f"A {party} whose IRT score ({xi:.2f}) places them closer to "
            f"{other_party}s than to their own party's median ({party_median:.2f})."
        )

    return NotableLegislator(
        slug=c["legislator_slug"],
        full_name=c["full_name"],
        party=party,
        district=c["district"],
        chamber=chamber,
        role=role,
        title=f"{c['full_name']} ({party[0]}-{c['district']})",
        subtitle=subtitle,
        reason="bridge",
    )


def detect_metric_paradox(
    leg_df: pl.DataFrame,
    chamber: str,
    *,
    rank_gap_percentile: float | None = None,
) -> ParadoxCase | None:
    """Detect the metric paradox: largest gap between IRT rank and loyalty rank.

    Within the majority party, finds the legislator with the largest gap between
    xi_mean percentile rank and loyalty_rate percentile rank. Must exceed
    :data:`PARADOX_RANK_GAP` (top half on one, bottom half on other).

    Args:
        chamber: lowercase key matching leg_dfs dict ("house" or "senate").
        rank_gap_percentile: If set, compute the rank gap threshold as this
            percentile of all rank gaps rather than the fixed
            :data:`PARADOX_RANK_GAP`.  Default ``None`` preserves the
            original behavior.
    """
    if "xi_mean" not in leg_df.columns or "loyalty_rate" not in leg_df.columns:
        return None

    majority = _majority_party(leg_df)
    if majority is None:
        return None

    party_df = leg_df.filter(pl.col("party") == majority)
    n = party_df.height
    if n < PARADOX_MIN_PARTY_SIZE:
        return None

    # Compute percentile ranks within majority party
    ranked = party_df.with_columns(
        (pl.col("xi_mean").rank("ordinal") / n).alias("xi_pct"),
        (pl.col("loyalty_rate").rank("ordinal") / n).alias("loyalty_pct"),
    ).with_columns((pl.col("xi_pct") - pl.col("loyalty_pct")).abs().alias("rank_gap"))

    best = ranked.sort("rank_gap", descending=True).head(1)
    if best.height == 0:
        return None

    b = best.to_dicts()[0]

    # Determine threshold
    if rank_gap_percentile is not None:
        gap_threshold = float(
            np.percentile(ranked["rank_gap"].to_numpy(), rank_gap_percentile * 100)
        )
    else:
        gap_threshold = PARADOX_RANK_GAP

    if b["rank_gap"] < gap_threshold:
        return None

    # Determine direction: if xi_pct > loyalty_pct, they're extreme on ideology
    # but low on loyalty → they defect in the direction of their ideology
    xi_pct = b["xi_pct"]
    loyalty_pct = b["loyalty_pct"]

    if xi_pct > loyalty_pct:
        # High IRT rank, low loyalty → extreme ideologue who defects from party
        metric_high_name = "IRT ideology"
        metric_low_name = "clustering loyalty"
        if majority == "Republican":
            direction = "rightward" if b["xi_mean"] > 0 else "leftward"
        else:
            direction = "leftward" if b["xi_mean"] < 0 else "rightward"
    else:
        # High loyalty, low IRT rank → loyal but moderate
        metric_high_name = "clustering loyalty"
        metric_low_name = "IRT ideology"
        direction = "toward the center"

    # Compute IRT rank among party (descending xi_mean for R, ascending for D)
    if majority == "Republican":
        irt_sorted = party_df.sort("xi_mean", descending=True)
    else:
        irt_sorted = party_df.sort("xi_mean", descending=False)

    irt_rank = (
        irt_sorted.with_row_index("_rank")
        .filter(pl.col("legislator_slug") == b["legislator_slug"])["_rank"]
        .item()
        + 1
    )

    raw_values = {
        "xi_mean": b["xi_mean"],
        "loyalty_rate": b["loyalty_rate"],
    }
    if "unity_score" in leg_df.columns:
        raw_values["unity_score"] = b.get("unity_score")

    return ParadoxCase(
        slug=b["legislator_slug"],
        full_name=b["full_name"],
        party=majority,
        district=b["district"],
        chamber=chamber,
        metric_high_name=metric_high_name,
        metric_low_name=metric_low_name,
        rank_high=irt_rank,
        n_in_party=n,
        raw_values=raw_values,
        direction=direction,
    )


def detect_annotation_slugs(
    leg_df: pl.DataFrame,
    notables: list[NotableLegislator | ParadoxCase],
    max_n: int = 3,
) -> list[str]:
    """Collect slugs worth annotating on dashboard scatters.

    Includes detected notable slugs plus most extreme per party. Deduplicates
    and caps at max_n.
    """
    slugs: list[str] = []

    # Add detected notable slugs
    for n in notables:
        if n.slug not in slugs:
            slugs.append(n.slug)

    # Add most extreme per party
    for party in leg_df["party"].unique().to_list():
        party_df = leg_df.filter(pl.col("party") == party)
        if party_df.height > 0:
            most_extreme = party_df.sort("xi_mean", descending=(party == "Republican")).head(1)
            s = most_extreme["legislator_slug"][0]
            if s not in slugs:
                slugs.append(s)

    return slugs[:max_n]


def _minority_parties(leg_df: pl.DataFrame) -> list[str]:
    """Return all parties except the majority party and Independents.

    Independents are excluded because party-unity metrics (unity_score, maverick_rate)
    are undefined for legislators with no party caucus.
    """
    majority = _majority_party(leg_df)
    if majority is None:
        return []
    return [p for p in leg_df["party"].unique().to_list() if p != majority and p != "Independent"]


def detect_all(
    leg_dfs: dict[str, pl.DataFrame],
    network_manifest: dict | None = None,
) -> dict:
    """Run all detection on both chambers. Single entry point.

    Args:
        leg_dfs: Chamber → unified legislator DataFrame.
        network_manifest: Optional network phase manifest (for bridge-builder
            connectivity awareness).

    Returns a dict with keys:
        profiles: dict[str, NotableLegislator] — slug → notable (for profile cards)
        paradoxes: dict[str, ParadoxCase] — slug → paradox
        annotations: dict[str, list[str]] — chamber → slugs to annotate
        mavericks: dict[str, NotableLegislator] — chamber → majority-party maverick
        minority_mavericks: dict[str, NotableLegislator] — chamber → minority-party maverick
        bridges: dict[str, NotableLegislator] — chamber → bridge-builder
    """
    profiles: dict[str, NotableLegislator] = {}
    paradoxes: dict[str, ParadoxCase] = {}
    annotations: dict[str, list[str]] = {}
    mavericks: dict[str, NotableLegislator] = {}
    minority_mavericks: dict[str, NotableLegislator] = {}
    bridges: dict[str, NotableLegislator] = {}

    for chamber, leg_df in leg_dfs.items():
        majority = _majority_party(leg_df)
        chamber_notables: list[NotableLegislator | ParadoxCase] = []

        # Detect maverick in majority party
        if majority:
            mav = detect_chamber_maverick(leg_df, majority, chamber)
            if mav is not None:
                mavericks[chamber] = mav
                profiles[mav.slug] = mav
                chamber_notables.append(mav)

        # Detect maverick in minority party (most likely to cross the aisle)
        for minority in _minority_parties(leg_df):
            min_mav = detect_chamber_maverick(leg_df, minority, chamber)
            if min_mav is not None:
                minority_mavericks[chamber] = min_mav
                if min_mav.slug not in profiles:
                    profiles[min_mav.slug] = min_mav
                    chamber_notables.append(min_mav)
                break  # one minority maverick per chamber

        # Detect bridge-builder
        bridge = detect_bridge_builder(leg_df, chamber, network_manifest=network_manifest)
        if bridge is not None:
            bridges[chamber] = bridge
            # Only add as profile if not already present
            if bridge.slug not in profiles:
                profiles[bridge.slug] = bridge
                chamber_notables.append(bridge)

        # Detect metric paradox
        paradox = detect_metric_paradox(leg_df, chamber)
        if paradox is not None:
            paradoxes[paradox.slug] = paradox
            # Add as profile if not already present
            if paradox.slug not in profiles:
                label = ideology_label(paradox.party, paradox.raw_values["xi_mean"])
                profiles[paradox.slug] = NotableLegislator(
                    slug=paradox.slug,
                    full_name=paradox.full_name,
                    party=paradox.party,
                    district=paradox.district,
                    chamber=paradox.chamber,
                    role=f"The {paradox.full_name.split()[-1]} Paradox",
                    title=f"{paradox.full_name} ({paradox.party[0]}-{paradox.district})",
                    subtitle=(
                        f"The most {label} by IRT — yet the least loyal by clustering, "
                        f"because they defect {paradox.direction} on close votes."
                    ),
                    reason="paradox",
                )
            chamber_notables.append(paradox)

        # Annotation slugs for dashboard scatters
        annotations[chamber] = detect_annotation_slugs(leg_df, chamber_notables)

    return {
        "profiles": profiles,
        "paradoxes": paradoxes,
        "annotations": annotations,
        "mavericks": mavericks,
        "minority_mavericks": minority_mavericks,
        "bridges": bridges,
    }
