"""Shared utilities for analysis phases.

Extracted from per-phase duplicates (R1-R3 in code audit).
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from analysis.run_context import resolve_upstream_dir, strip_leadership_suffix

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


def load_metadata(data_dir: Path, *, use_csv: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
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


# ── Horseshoe Status ─────────────────────────────────────────────────────


def load_horseshoe_status(
    results_root: Path,
    run_id: str | None = None,
) -> dict[str, dict]:
    """Load per-chamber horseshoe detection status from the canonical routing manifest.

    Returns a dict mapping chamber name to its routing metadata, e.g.:
        {"Senate": {"detected": True, "source": "2d_dim1", "reason": "..."}, ...}

    Returns empty dict if no manifest exists (pre-canonical-routing pipelines).
    """
    canonical_dir = resolve_upstream_dir("canonical_irt", results_root, run_id=run_id)
    manifest_path = canonical_dir / "routing_manifest.json"
    if not manifest_path.exists():
        return {}

    manifest = json.loads(manifest_path.read_text())
    sources = manifest.get("sources", {})
    metadata = manifest.get("metadata", {})

    result: dict[str, dict] = {}
    for chamber in sources:
        meta = metadata.get(chamber, {})
        horseshoe = meta.get("horseshoe", {})
        result[chamber] = {
            "detected": horseshoe.get("detected", False),
            "source": sources.get(chamber, "unknown"),
            "reason": meta.get("reason", ""),
            "dem_wrong_side_frac": horseshoe.get("dem_wrong_side_frac", 0.0),
            "overlap_frac": horseshoe.get("overlap_frac", 0.0),
        }
    return result


def horseshoe_warning_html(chamber: str, status: dict) -> str:
    """Generate an HTML warning banner for a horseshoe-affected chamber.

    Args:
        chamber: Chamber name (e.g., "Senate").
        status: Per-chamber dict from ``load_horseshoe_status()``.

    Returns HTML string. Empty string if horseshoe not detected.
    """
    if not status.get("detected", False):
        return ""

    source = status.get("source", "unknown")
    if source == "2d_dim1":
        routing_note = (
            "Canonical ideal points use <strong>2D IRT Dim 1</strong> to correct "
            "for the horseshoe distortion."
        )
    else:
        routing_note = (
            "2D IRT was unavailable or did not converge; downstream phases use the "
            "distorted <strong>1D IRT</strong> scores. Interpret with caution."
        )

    return (
        '<div style="background:#fff3cd; border:1px solid #ffc107; border-radius:6px; '
        'padding:12px 16px; margin:16px 0;">'
        f"<strong>Horseshoe Effect Detected ({chamber})</strong><br>"
        f"The {chamber} has a large party-size imbalance that causes the standard 1D IRT "
        "model to fold minority-party members back toward the majority, distorting ideal "
        "point estimates. This is visible as a U-shaped (horseshoe) pattern in PCA and MCA. "
        f"{routing_note}"
        "</div>"
    )


def drop_empty_optional_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Drop columns that are all-null or all-empty-string.

    Useful for KanFocus sessions where optional fields like ``short_title`` are
    unavailable.
    """
    to_drop = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        if series.is_null().all():
            to_drop.append(col)
        elif series.dtype == pl.String and series.fill_null("").str.strip_chars().eq("").all():
            to_drop.append(col)
    return df.drop(to_drop) if to_drop else df
