"""Shared utilities for analysis phases.

Extracted from per-phase duplicates (R1-R3 in code audit).
"""

from __future__ import annotations

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


def load_metadata(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load rollcall and legislator CSVs for metadata enrichment.

    Returns (rollcalls, legislators) with leadership suffixes stripped
    and empty/null party values filled to "Independent".
    """
    prefix = data_dir.name
    rollcalls = pl.read_csv(data_dir / f"{prefix}_rollcalls.csv")
    legislators = _clean_legislators(data_dir)
    return rollcalls, legislators


def load_legislators(data_dir: Path) -> pl.DataFrame:
    """Load legislator CSV with standard cleaning.

    Same as ``load_metadata`` but returns only the legislators DataFrame.
    """
    return _clean_legislators(data_dir)


def _clean_legislators(data_dir: Path) -> pl.DataFrame:
    """Load and clean legislator CSV (shared implementation)."""
    prefix = data_dir.name
    legislators = pl.read_csv(data_dir / f"{prefix}_legislators.csv")
    # Rename slug → legislator_slug to match analysis convention (ADR-0066)
    if "slug" in legislators.columns and "legislator_slug" not in legislators.columns:
        legislators = legislators.rename({"slug": "legislator_slug"})
    return legislators.with_columns(
        pl.col("full_name")
        .map_elements(strip_leadership_suffix, return_dtype=pl.Utf8)
        .alias("full_name"),
        pl.col("party").fill_null("Independent").replace("", "Independent").alias("party"),
    )


# ── Name Normalization ─────────────────────────────────────────────────────

_LEADERSHIP_SUFFIX_RE = re.compile(r"\s*-\s+.*$")
"""Matches leadership suffixes like ' - House Minority Caucus Chair'."""


def normalize_name(name: str) -> str:
    """Normalize a legislator name for cross-session/biennium matching.

    Lowercases, strips whitespace, and removes leadership suffixes.
    """
    name = name.strip().lower()
    name = _LEADERSHIP_SUFFIX_RE.sub("", name)
    return name
