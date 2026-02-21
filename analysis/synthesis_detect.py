"""
Data-driven detection of notable legislators for the synthesis report.

Pure data logic — no plotting, no report building. Detects mavericks,
bridge-builders, and metric paradoxes from upstream analysis DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


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
    leg_df: pl.DataFrame, party: str, chamber: str
) -> NotableLegislator | None:
    """Detect the maverick in a given party: lowest unity_score.

    Ties broken by highest weighted_maverick. Returns None if all unity > 0.95
    or if the required columns are missing.

    Args:
        chamber: lowercase key matching leg_dfs dict ("house" or "senate").
    """
    if "unity_score" not in leg_df.columns:
        return None

    party_df = leg_df.filter(pl.col("party") == party)
    if party_df.height == 0:
        return None

    # Skip if everyone is highly disciplined
    min_unity = party_df["unity_score"].min()
    if min_unity is not None and min_unity > 0.95:
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


def detect_bridge_builder(leg_df: pl.DataFrame, chamber: str) -> NotableLegislator | None:
    """Detect the bridge-builder: highest betweenness near the cross-party midpoint.

    Finds the legislator with the highest betweenness whose xi_mean is within 1 SD
    of the cross-party midpoint. Falls back to highest betweenness regardless.
    Returns None if betweenness column is missing.

    Args:
        chamber: lowercase key matching leg_dfs dict ("house" or "senate").
    """
    if "betweenness" not in leg_df.columns or "xi_mean" not in leg_df.columns:
        return None

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

    # Candidates within 1 SD of the midpoint
    near_center = leg_df.filter((pl.col("xi_mean") - midpoint).abs() <= overall_sd)

    if near_center.height > 0:
        candidate_df = near_center.sort("betweenness", descending=True).head(1)
    else:
        # Fallback: highest betweenness regardless
        candidate_df = leg_df.sort("betweenness", descending=True).head(1)

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

    return NotableLegislator(
        slug=c["legislator_slug"],
        full_name=c["full_name"],
        party=party,
        district=c["district"],
        chamber=chamber,
        role=f"{chamber.title()} Bridge-Builder",
        title=f"{c['full_name']} ({party[0]}-{c['district']})",
        subtitle=(
            f"A {party} whose IRT score ({xi:.2f}) places them closer to "
            f"{other_party}s than to their own party's median ({party_median:.2f})."
        ),
        reason="bridge",
    )


def detect_metric_paradox(leg_df: pl.DataFrame, chamber: str) -> ParadoxCase | None:
    """Detect the metric paradox: largest gap between IRT rank and loyalty rank.

    Within the majority party, finds the legislator with the largest gap between
    xi_mean percentile rank and loyalty_rate percentile rank. Must exceed 0.5 gap
    (top half on one, bottom half on other).
    """
    if "xi_mean" not in leg_df.columns or "loyalty_rate" not in leg_df.columns:
        return None

    majority = _majority_party(leg_df)
    if majority is None:
        return None

    party_df = leg_df.filter(pl.col("party") == majority)
    n = party_df.height
    if n < 5:
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
    if b["rank_gap"] < 0.5:
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


def detect_all(leg_dfs: dict[str, pl.DataFrame]) -> dict:
    """Run all detection on both chambers. Single entry point.

    Returns a dict with keys:
        profiles: dict[str, NotableLegislator] — slug → notable (for profile cards)
        paradoxes: dict[str, ParadoxCase] — slug → paradox
        annotations: dict[str, list[str]] — chamber → slugs to annotate
        mavericks: dict[str, NotableLegislator] — chamber → maverick
        bridges: dict[str, NotableLegislator] — chamber → bridge-builder
    """
    profiles: dict[str, NotableLegislator] = {}
    paradoxes: dict[str, ParadoxCase] = {}
    annotations: dict[str, list[str]] = {}
    mavericks: dict[str, NotableLegislator] = {}
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

        # Detect bridge-builder
        bridge = detect_bridge_builder(leg_df, chamber)
        if bridge is not None:
            bridges[chamber] = bridge
            # Only add as profile if not already the maverick
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
        "bridges": bridges,
    }
