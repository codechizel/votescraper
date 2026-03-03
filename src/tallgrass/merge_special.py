"""Merge special session data into parent biennium CSVs.

Special sessions are short (1-2 days, 3-8 roll calls) on 1-2 issues.
Rather than running a standalone pipeline that can't do IRT/PCA/clustering
(MIN_VOTES=20 filters out all legislators), we merge special session votes
into their parent biennium so `just pipeline 2019-20` includes them.

The merge is idempotent: it filters out any previously merged special rows
(by the ``session`` column) before concatenating fresh data.
"""

from pathlib import Path

import polars as pl

from tallgrass.session import SPECIAL_SESSION_YEARS, KSSession


def merge_special_into_parent(special_year: int) -> dict[str, int]:
    """Merge one special session's CSVs into its parent biennium.

    Args:
        special_year: The special session year (e.g. 2020).

    Returns:
        Dict with keys ``votes_added``, ``rollcalls_added``, ``legislators_added``
        showing how many rows were merged.
    """
    special = KSSession(start_year=special_year, special=True)
    parent = special.parent_session

    stats: dict[str, int] = {}

    # -- votes --
    stats["votes_added"] = _merge_csv(
        parent_dir=parent.data_dir,
        special_dir=special.data_dir,
        filename_stem="votes",
        parent_name=parent.output_name,
        special_name=special.output_name,
        special_label=special.label,
        dedup_subset=None,
    )

    # -- rollcalls --
    stats["rollcalls_added"] = _merge_csv(
        parent_dir=parent.data_dir,
        special_dir=special.data_dir,
        filename_stem="rollcalls",
        parent_name=parent.output_name,
        special_name=special.output_name,
        special_label=special.label,
        dedup_subset=None,
    )

    # -- legislators (dedup by slug, parent wins) --
    stats["legislators_added"] = _merge_csv(
        parent_dir=parent.data_dir,
        special_dir=special.data_dir,
        filename_stem="legislators",
        parent_name=parent.output_name,
        special_name=special.output_name,
        special_label=special.label,
        dedup_subset=["slug"],
    )

    # -- bill_actions: no-op (specials don't produce them) --

    return stats


def _merge_csv(
    *,
    parent_dir: Path,
    special_dir: Path,
    filename_stem: str,
    parent_name: str,
    special_name: str,
    special_label: str,
    dedup_subset: list[str] | None,
) -> int:
    """Read parent + special CSVs, filter old special rows, concat, write back.

    Returns the number of rows added from the special session.
    """
    parent_path = parent_dir / f"{parent_name}_{filename_stem}.csv"
    special_path = special_dir / f"{special_name}_{filename_stem}.csv"

    if not special_path.exists():
        return 0

    special_df = pl.read_csv(special_path, infer_schema_length=0)
    if special_df.is_empty():
        return 0

    if not parent_path.exists():
        return 0

    parent_df = pl.read_csv(parent_path, infer_schema_length=0)

    # Filter out previously merged special rows (idempotent)
    if "session" in parent_df.columns:
        parent_df = parent_df.filter(pl.col("session") != special_label)

    # Align columns: add missing columns with empty strings
    for col in special_df.columns:
        if col not in parent_df.columns:
            parent_df = parent_df.with_columns(pl.lit("").alias(col))
    for col in parent_df.columns:
        if col not in special_df.columns:
            special_df = special_df.with_columns(pl.lit("").alias(col))

    # Reorder special columns to match parent
    special_df = special_df.select(parent_df.columns)

    merged = pl.concat([parent_df, special_df])

    # Dedup if requested (legislators: keep first = parent row)
    if dedup_subset:
        merged = merged.unique(subset=dedup_subset, keep="first")
    rows_added = merged.height - parent_df.height

    merged.write_csv(parent_path)

    return rows_added


def merge_all_specials() -> dict[int, dict[str, int]]:
    """Merge all known special sessions into their parent bienniums.

    Returns:
        Dict mapping special year to merge stats.
    """
    results: dict[int, dict[str, int]] = {}
    for year in sorted(SPECIAL_SESSION_YEARS):
        special = KSSession(start_year=year, special=True)
        if not special.data_dir.exists():
            continue
        parent = special.parent_session
        if not parent.data_dir.exists():
            continue
        results[year] = merge_special_into_parent(year)
    return results
