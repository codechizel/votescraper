"""Legislator slug generation and cross-reference matching for KanFocus data.

KanFocus names follow the format ``Name, Party-District`` (e.g.,
``Steve Abrams, R-32nd``). This module generates tallgrass-compatible slugs
(``sen_abrams_steve_1``) and cross-references against existing CSVs for
overlapping sessions (84th+).

Name normalization handles:
- Suffixes: ``Jr.``, ``Sr.``, ``II``, ``III``, ``IV``
- Nicknames in parens: ``Thomas C. (Tim) Owens`` → use formal name
- Middle initials: ``Stephen R. Morris`` → ``stephen_morris``
- Multi-word last names: ``Mary Pilcher Cook`` → ``pilcher_cook_mary``
- Missing apostrophes: ``OBrien`` → ``obrien`` (KanFocus strips apostrophes)
"""

import csv
import re
from pathlib import Path

# Suffixes to strip from names
_SUFFIXES = re.compile(r"\s+(Jr\.?|Sr\.?|II|III|IV)\s*$", re.IGNORECASE)

# Nickname in parentheses: "Thomas C. (Tim) Owens"
_NICKNAME = re.compile(r"\s*\([^)]+\)\s*")

# Middle initials: single letter followed by period
_MIDDLE_INITIAL = re.compile(r"\s+[A-Z]\.\s*")

# Chamber prefix mapping
_CHAMBER_PREFIX = {"S": "sen", "H": "rep"}


def normalize_name(name: str) -> str:
    """Normalize a KanFocus legislator name for slug generation.

    Strips suffixes, nicknames, and middle initials.

    >>> normalize_name("Ramon Gonzalez Jr.")
    'Ramon Gonzalez'
    >>> normalize_name("Thomas C. (Tim) Owens")
    'Thomas Owens'
    >>> normalize_name("Stephen R. Morris")
    'Stephen Morris'
    """
    result = _SUFFIXES.sub("", name).strip()
    result = _NICKNAME.sub(" ", result).strip()
    result = _MIDDLE_INITIAL.sub(" ", result).strip()
    # Collapse multiple spaces
    result = re.sub(r"\s+", " ", result)
    return result


def generate_slug(name: str, chamber: str) -> str:
    """Generate a tallgrass slug from a KanFocus name and chamber.

    Follows the existing slug convention: ``{prefix}_{lastname}_{firstname}_{1}``
    where multi-word last names use underscores.

    >>> generate_slug("Steve Abrams", "S")
    'sen_abrams_steve_1'
    >>> generate_slug("Mary Pilcher Cook", "H")
    'rep_pilcher_cook_mary_1'
    >>> generate_slug("Ramon Gonzalez Jr.", "H")
    'rep_gonzalez_ramon_1'
    >>> generate_slug("Thomas C. (Tim) Owens", "S")
    'sen_owens_thomas_1'
    """
    prefix = _CHAMBER_PREFIX.get(chamber, chamber.lower())
    normalized = normalize_name(name)
    parts = normalized.split()

    if len(parts) < 2:
        # Fallback for single-name edge cases
        return f"{prefix}_{parts[0].lower()}_1"

    first = parts[0].lower()
    last = "_".join(p.lower() for p in parts[1:])

    return f"{prefix}_{last}_{first}_1"


def load_existing_slugs(data_dir: Path, output_name: str) -> dict[str, str]:
    """Load name→slug mapping from an existing legislators CSV.

    Returns a dict mapping lowercase normalized names to existing slugs.
    Empty dict if the CSV doesn't exist.
    """
    csv_path = data_dir / f"{output_name}_legislators.csv"
    if not csv_path.exists():
        return {}

    mapping: dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row.get("slug", "")
            name = row.get("name", "")
            full_name = row.get("full_name", "")
            if slug and name:
                mapping[name.lower().strip()] = slug
            if slug and full_name:
                mapping[full_name.lower().strip()] = slug
    return mapping


def match_to_existing(
    kf_name: str,
    kf_chamber: str,
    kf_district: str,
    existing: dict[str, str],
) -> str | None:
    """Try to match a KanFocus legislator to an existing slug.

    Checks normalized name against the existing mapping. Returns the
    existing slug if found, None otherwise.
    """
    if not existing:
        return None

    normalized = normalize_name(kf_name).lower().strip()

    # Direct match on normalized name
    if normalized in existing:
        return existing[normalized]

    # Try original name (KanFocus may include middle initials the scraper stripped)
    if kf_name.lower().strip() in existing:
        return existing[kf_name.lower().strip()]

    # Try last name + first name only (drop middle parts)
    parts = normalized.split()
    if len(parts) >= 2:
        # Try "firstname lastname" and "lastname, firstname" patterns
        first_last = f"{parts[0]} {parts[-1]}"
        if first_last in existing:
            return existing[first_last]

    return None


def build_slug_lookup(
    kf_name: str,
    kf_chamber: str,
    kf_district: str,
    existing: dict[str, str],
) -> str:
    """Get the slug for a KanFocus legislator, preferring existing matches.

    First attempts to match against existing slugs (for overlapping sessions).
    Falls back to generating a fresh slug.
    """
    matched = match_to_existing(kf_name, kf_chamber, kf_district, existing)
    if matched:
        return matched
    return generate_slug(kf_name, kf_chamber)
