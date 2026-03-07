"""Shared utilities for analysis phases.

Extracted from per-phase duplicates (R1-R3 in code audit).
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from analysis.run_context import strip_leadership_suffix

# ── Console Output ─────────────────────────────────────────────────────────


def print_header(title: str) -> None:
    """Print a section header to stdout."""
    width = 80
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ── Figure Saving ──────────────────────────────────────────────────────────


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure, close it, and print the filename."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Data Loading ───────────────────────────────────────────────────────────


def load_metadata(
    data_dir: Path, *, use_csv: bool = False
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load rollcalls and legislators for metadata enrichment.

    Returns (rollcalls, legislators) with leadership suffixes stripped
    and empty/null party values filled to "Independent".

    Loads from PostgreSQL by default; falls back to CSV when the DB is
    unavailable or when *use_csv* is True.
    """
    from analysis.db import load_legislators as db_load_legislators
    from analysis.db import load_rollcalls as db_load_rollcalls

    rollcalls = db_load_rollcalls(data_dir, use_csv=use_csv)
    legislators = db_load_legislators(data_dir, use_csv=use_csv)
    return rollcalls, legislators


def load_legislators(data_dir: Path, *, use_csv: bool = False) -> pl.DataFrame:
    """Load legislators with standard cleaning.

    Same as ``load_metadata`` but returns only the legislators DataFrame.
    Loads from PostgreSQL by default; CSV fallback.
    """
    from analysis.db import load_legislators as db_load_legislators

    return db_load_legislators(data_dir, use_csv=use_csv)


# ── Name Normalization ─────────────────────────────────────────────────────


def normalize_name(name: str) -> str:
    """Normalize a legislator name for cross-session/biennium matching.

    Lowercases, strips whitespace, and removes leadership suffixes.
    """
    name = name.strip().lower()
    name = strip_leadership_suffix(name)
    return name


# ── Sponsor Matching ──────────────────────────────────────────────────────

_TITLE_RE = re.compile(r"^(Senator|Representative)\s+", re.IGNORECASE)
"""Matches chamber title prefix in sponsor text."""


def parse_sponsor_name(raw: str) -> tuple[str | None, str | None]:
    """Parse a single sponsor entry like 'Senator Tyson' into (name, chamber).

    Returns (None, None) for committee sponsors or empty/malformed input.
    """
    if not raw or not raw.strip():
        return None, None
    raw = raw.strip()

    # Detect committee sponsors
    if "committee" in raw.lower():
        return None, None

    match = _TITLE_RE.match(raw)
    if not match:
        return None, None

    title = match.group(1).lower()
    name = raw[match.end() :].strip()
    if not name:
        return None, None

    chamber = "Senate" if title == "senator" else "House"
    return name, chamber


def match_sponsor_to_slug(sponsor_text: str, legislators: pl.DataFrame) -> str | None:
    """Match the first person sponsor to a legislator slug via name + chamber.

    Uses text-based matching as a fallback when slug-based matching is unavailable
    (e.g., CSVs scraped before sponsor_slugs was added).

    The slug column is auto-detected: ``legislator_slug`` (analysis convention) or
    ``slug`` (scraper convention).

    Returns the legislator slug string or None.
    """
    if not sponsor_text:
        return None

    # Take the first sponsor entry
    first = sponsor_text.split(";")[0].strip()
    name, chamber = parse_sponsor_name(first)
    if name is None or chamber is None:
        return None

    # Normalize for matching
    name_lower = name.strip().lower()

    # Filter to matching chamber
    chamber_legs = legislators.filter(pl.col("chamber") == chamber)
    if chamber_legs.height == 0:
        return None

    # Try last-name match against full_name column
    for row in chamber_legs.iter_rows(named=True):
        full_name = (row.get("full_name") or "").lower()
        # full_name is "Last" or "Last, First" — check if our name ends with the last name
        last_name = full_name.split(",")[0].strip() if full_name else ""
        if last_name and last_name == name_lower.split()[-1].lower():
            return row.get("legislator_slug")

    return None


def match_sponsor_to_party(sponsor_text: str, legislators: pl.DataFrame) -> str | None:
    """Match the first person sponsor to a party via name + chamber.

    Thin wrapper around :func:`match_sponsor_to_slug` — resolves identity first,
    then looks up party.

    Returns party string ("Republican", "Democrat", "Independent") or None.
    """
    slug = match_sponsor_to_slug(sponsor_text, legislators)
    if slug is None:
        return None

    matched = legislators.filter(pl.col("legislator_slug") == slug)
    if matched.height == 0:
        return None
    return matched[0, "party"]
