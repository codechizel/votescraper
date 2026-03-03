"""Coalition labeling for clustering and LCA results.

Auto-names clusters/classes based on party composition and IRT ideal points.
Produces human-readable labels like "Moderate Republicans" or "Bipartisan Coalition"
for use in reports and narrative descriptions.
"""

from dataclasses import dataclass

import polars as pl

# ── Thresholds ────────────────────────────────────────────────────────────────

DOMINANT_PARTY_THRESHOLD: float = 0.80
"""If a single party exceeds this fraction of the cluster, use party-based naming."""

MODERATE_DISTANCE: float = 0.3
"""If cluster median IRT is within this distance of party mean (toward center), label 'Moderate'."""


@dataclass(frozen=True)
class Coalition:
    """A labeled cluster or latent class with descriptive metadata."""

    name: str  # "Moderate Republicans"
    description: str  # "12 members, median IRT -0.3"
    members: tuple[str, ...]  # legislator_slugs
    chamber: str
    party_composition: dict[str, int]  # {"Republican": 10, "Democrat": 2}
    median_irt: float
    size: int


def _compute_party_means(ideal_points: pl.DataFrame) -> dict[str, float]:
    """Compute mean IRT ideal point per party from the full ideal_points DataFrame.

    Args:
        ideal_points: DataFrame with 'party' and 'xi_mean' columns.

    Returns:
        dict mapping party name to mean xi_mean.
    """
    means: dict[str, float] = {}
    for row in (
        ideal_points.filter(pl.col("xi_mean").is_not_null())
        .group_by("party")
        .agg(pl.col("xi_mean").mean().alias("mean_xi"))
        .iter_rows(named=True)
    ):
        means[row["party"]] = row["mean_xi"]
    return means


def _modifier_for_party(
    party: str,
    cluster_median: float,
    party_mean: float,
) -> str:
    """Choose a modifier based on how the cluster median compares to the party mean.

    Rules:
        - Within MODERATE_DISTANCE of party mean: "Mainstream"
        - More extreme than party mean: "Conservative" (R) or "Progressive" (D)
        - Less extreme than party mean: "Moderate"
    """
    distance = cluster_median - party_mean

    # "More extreme" means further from zero on the same side as the party mean.
    # For Republicans (positive mean): more extreme = cluster median > party mean.
    # For Democrats (negative mean): more extreme = cluster median < party mean.
    if abs(distance) <= MODERATE_DISTANCE:
        return "Mainstream"

    if party == "Republican":
        if distance > 0:
            return "Conservative"
        return "Moderate"

    if party == "Democrat":
        if distance < 0:
            return "Progressive"
        return "Moderate"

    # Fallback for Independent or other parties
    if abs(cluster_median) < abs(party_mean):
        return "Moderate"
    return "Mainstream"


def label_single_coalition(
    cluster_members: pl.DataFrame,
    party_means: dict[str, float],
    chamber: str,
    cluster_id: int,
) -> Coalition:
    """Label a single cluster with a descriptive name.

    Args:
        cluster_members: DataFrame with 'legislator_slug', 'party', 'xi_mean' for members
            of one cluster.
        party_means: dict from party name to full-chamber mean IRT ideal point.
        chamber: "House" or "Senate".
        cluster_id: integer cluster identifier (used in fallback naming).

    Returns:
        A Coalition with a human-readable name and full metadata.
    """
    size = cluster_members.height

    # ── Edge case: empty cluster ──────────────────────────────────────────
    if size == 0:
        return Coalition(
            name=f"Empty Cluster {cluster_id}",
            description="No members",
            members=(),
            chamber=chamber,
            party_composition={},
            median_irt=0.0,
            size=0,
        )

    # ── Gather members and party composition ──────────────────────────────
    members = tuple(cluster_members["legislator_slug"].to_list())

    party_counts: dict[str, int] = {}
    for row in cluster_members.group_by("party").len().iter_rows(named=True):
        party_counts[row["party"]] = row["len"]

    # ── Median IRT (handle missing xi_mean) ───────────────────────────────
    valid_irt = cluster_members.filter(pl.col("xi_mean").is_not_null())
    if valid_irt.height > 0:
        median_irt = float(valid_irt["xi_mean"].median())  # type: ignore[arg-type]
    else:
        median_irt = 0.0

    # ── Single-member cluster ─────────────────────────────────────────────
    if size == 1:
        row = cluster_members.to_dicts()[0]
        party = row["party"]
        name = f"Solo {party}" if party else f"Solo Legislator {cluster_id}"
        return Coalition(
            name=name,
            description=f"1 member, median IRT {median_irt:.2f}",
            members=members,
            chamber=chamber,
            party_composition=party_counts,
            median_irt=median_irt,
            size=1,
        )

    # ── Determine dominant party (if any) ─────────────────────────────────
    dominant_party: str | None = None
    dominant_frac: float = 0.0
    for party, count in party_counts.items():
        frac = count / size
        if frac > dominant_frac:
            dominant_frac = frac
            dominant_party = party

    # ── Name the coalition ────────────────────────────────────────────────
    if dominant_party is not None and dominant_frac > DOMINANT_PARTY_THRESHOLD:
        # Party-dominated cluster
        party_mean = party_means.get(dominant_party, 0.0)
        modifier = _modifier_for_party(dominant_party, median_irt, party_mean)

        # Pluralize the party name
        if dominant_party.endswith("s"):
            party_plural = dominant_party
        else:
            party_plural = dominant_party + "s"

        name = f"{modifier} {party_plural}"
    else:
        # Mixed-party cluster
        if size <= 5:
            name = "Cross-Party Bloc"
        else:
            name = "Bipartisan Coalition"

    description = f"{size} members, median IRT {median_irt:.2f}"

    return Coalition(
        name=name,
        description=description,
        members=members,
        chamber=chamber,
        party_composition=party_counts,
        median_irt=median_irt,
        size=size,
    )


def label_coalitions(
    clusters: pl.DataFrame,
    ideal_points: pl.DataFrame,
    chamber: str,
) -> list[Coalition]:
    """Auto-label clusters with descriptive names.

    Args:
        clusters: DataFrame with 'legislator_slug', 'cluster' (int), 'party' columns.
        ideal_points: DataFrame with 'legislator_slug', 'xi_mean' columns.
        chamber: "House" or "Senate".

    Returns:
        List of Coalition objects, one per cluster, sorted by cluster ID.

    Naming rules:
        - If >80% one party: "{modifier} {party}"
          where modifier depends on median IRT vs party mean:
          - Within 0.3 of party mean: "Mainstream"
          - More extreme than party mean: "Conservative" (R) or "Progressive" (D)
          - Less extreme than party mean: "Moderate"
        - If mixed (no party >80%): "Bipartisan Coalition" or "Cross-Party Bloc"
    """
    if clusters.height == 0:
        return []

    # Merge clusters with ideal points to get xi_mean per member
    merged = clusters.join(
        ideal_points.select("legislator_slug", "xi_mean"),
        on="legislator_slug",
        how="left",
    )

    # Compute party mean ideal points from the full ideal_points DataFrame
    party_means = _compute_party_means(ideal_points)

    # Label each cluster
    cluster_ids = sorted(merged["cluster"].unique().to_list())
    coalitions: list[Coalition] = []

    for cid in cluster_ids:
        cluster_df = merged.filter(pl.col("cluster") == cid)
        coalition = label_single_coalition(cluster_df, party_means, chamber, cid)
        coalitions.append(coalition)

    return coalitions


def describe_coalition(coalition: Coalition) -> str:
    """Produce a one-sentence narrative description of a coalition.

    Examples:
        "The Moderate Republicans are a 12-member House bloc (10 R, 2 D)
         with a median ideal point of -0.30."
        "The Bipartisan Coalition is a 7-member Senate bloc (4 R, 3 D)
         with a median ideal point of 0.05."
    """
    if coalition.size == 0:
        return f"{coalition.name}: no members."

    # Build party breakdown string
    party_parts: list[str] = []
    for party, count in sorted(coalition.party_composition.items(), key=lambda x: -x[1]):
        abbreviation = party[0] if party else "?"
        party_parts.append(f"{count} {abbreviation}")
    party_str = ", ".join(party_parts)

    member_word = "member" if coalition.size == 1 else "members"

    return (
        f"The {coalition.name} {'is' if coalition.size == 1 else 'are'} a "
        f"{coalition.size}-{member_word} {coalition.chamber} bloc "
        f"({party_str}) with a median ideal point of {coalition.median_irt:.2f}."
    )
