"""Shared data factory functions for Tallgrass tests.

Provides reusable builders for legislators, votes, and rollcalls DataFrames
used across test modules. Factory functions use a ``slug_column`` parameter to
handle the scraper/analysis schema split (ADR-0066): scraper CSVs use "slug",
analysis phases use "legislator_slug".
"""

import numpy as np
import polars as pl


def make_legislators(
    names: list[str] | None = None,
    n: int = 10,
    *,
    prefix: str = "rep",
    party: str = "Republican",
    chamber: str = "House",
    start_district: int = 1,
    slug_column: str = "legislator_slug",
    ocd_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Build a legislators DataFrame.

    Args:
        names: Explicit legislator names. If None, generates *n* legislators
            named "Member 0", "Member 1", etc.
        n: Number of legislators when *names* is not provided.
        prefix: Slug prefix (e.g. "rep", "sen").
        party: Party affiliation for all legislators.
        chamber: Chamber for all legislators.
        start_district: Starting district number.
        slug_column: Column name for the slug field.
        ocd_ids: Explicit OCD person IDs.  If None, generates empty strings.
    """
    if names is None:
        names = [f"Member {i}" for i in range(n)]
    slugs = [f"{prefix}_{name.split()[-1].lower()}" for name in names]
    if ocd_ids is None:
        ocd_ids = [""] * len(names)
    return pl.DataFrame(
        {
            slug_column: slugs,
            "full_name": names,
            "party": [party] * len(names),
            "chamber": [chamber] * len(names),
            "district": list(range(start_district, start_district + len(names))),
            "ocd_id": ocd_ids,
        }
    )


def make_votes(
    legislators: pl.DataFrame | None = None,
    n_votes: int = 30,
    *,
    n_legislators: int = 20,
    slug_column: str = "legislator_slug",
    include_absence: bool = False,
    seed: int = 42,
) -> pl.DataFrame:
    """Build a long-form votes DataFrame with party-structured voting.

    Republicans vote Yea ~80%, Democrats vote Nay ~80%.

    Args:
        legislators: DataFrame with *slug_column* and ``party`` columns.
            If None, generates a balanced R/D set of *n_legislators*.
        n_votes: Number of roll call votes to generate.
        n_legislators: Number of legislators when *legislators* is not provided.
        slug_column: Name of the slug column in the output.
        include_absence: If True, ~5% of votes become "Absent and Not Voting".
        seed: Random seed for reproducibility.
    """
    if legislators is None:
        half = n_legislators // 2
        legislators = pl.concat(
            [
                make_legislators(
                    n=half,
                    prefix="rep_r",
                    party="Republican",
                    slug_column=slug_column,
                ),
                make_legislators(
                    n=n_legislators - half,
                    prefix="rep_d",
                    party="Democrat",
                    start_district=half + 1,
                    slug_column=slug_column,
                ),
            ]
        )

    rng = np.random.default_rng(seed)
    slugs = legislators[slug_column].to_list()
    parties = dict(zip(legislators[slug_column].to_list(), legislators["party"].to_list()))

    rows: list[dict[str, str]] = []
    for j in range(n_votes):
        vid = f"vote_{j}"
        for slug in slugs:
            party = parties[slug]
            if party == "Republican":
                vote = "Yea" if rng.random() > 0.2 else "Nay"
            else:
                vote = "Nay" if rng.random() > 0.2 else "Yea"

            if include_absence and rng.random() < 0.05:
                vote = "Absent and Not Voting"

            rows.append({slug_column: slug, "vote_id": vid, "vote": vote})

    return pl.DataFrame(rows)


def make_rollcalls(
    n_votes: int = 30,
    *,
    chamber: str = "House",
) -> pl.DataFrame:
    """Build a rollcalls DataFrame with vote_id, chamber, and bill_number."""
    return pl.DataFrame(
        {
            "vote_id": [f"vote_{j}" for j in range(n_votes)],
            "chamber": [chamber] * n_votes,
            "bill_number": [f"HB {j}" for j in range(n_votes)],
        }
    )
