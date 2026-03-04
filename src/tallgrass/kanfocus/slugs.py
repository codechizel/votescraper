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
- Hyphens: ``Faust-Goudeau`` → ``faust_goudeau`` (normalized to spaces)
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

# Slug prefix → expected chamber
_SLUG_PREFIX_CHAMBER = {"sen_": "S", "rep_": "H"}

# Common nickname → formal name(s) mapping
_NICKNAMES: dict[str, list[str]] = {
    "bill": ["william", "lewis"],
    "brad": ["bradley"],
    "nate": ["nathan"],
    "dan": ["daniel"],
    "rick": ["richard"],
    "dick": ["richard"],
    "doug": ["douglas"],
    "steve": ["steven", "stephen"],
    "bob": ["robert"],
    "jim": ["james"],
    "mike": ["michael"],
    "tom": ["thomas"],
    "joe": ["joseph"],
    "ed": ["edward", "edwin"],
    "ted": ["theodore", "edward"],
    "rob": ["robert"],
    "marty": ["martin"],
    "chip": ["charles"],
    "angel": ["angelina", "angela"],
    "ricky": ["richard"],
}

# Reverse map: formal name → list of nicknames
_FORMAL_TO_NICK: dict[str, list[str]] = {}
for _nick, _formals in _NICKNAMES.items():
    for _formal in _formals:
        _FORMAL_TO_NICK.setdefault(_formal, []).append(_nick)


def normalize_name(name: str) -> str:
    """Normalize a KanFocus legislator name for slug generation.

    Strips suffixes, nicknames, middle initials, and replaces hyphens
    with spaces so ``Faust-Goudeau`` → ``Faust Goudeau``.

    >>> normalize_name("Ramon Gonzalez Jr.")
    'Ramon Gonzalez'
    >>> normalize_name("Thomas C. (Tim) Owens")
    'Thomas Owens'
    >>> normalize_name("Stephen R. Morris")
    'Stephen Morris'
    >>> normalize_name("Oletha Faust-Goudeau")
    'Oletha Faust Goudeau'
    """
    result = _SUFFIXES.sub("", name).strip()
    result = _NICKNAME.sub(" ", result).strip()
    result = _MIDDLE_INITIAL.sub(" ", result).strip()
    # Replace hyphens with spaces for slug normalization
    result = result.replace("-", " ")
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


def _chamber_compatible(slug: str, kf_chamber: str) -> bool:
    """Check that a slug's prefix matches the expected chamber."""
    for prefix, chamber in _SLUG_PREFIX_CHAMBER.items():
        if slug.startswith(prefix):
            return chamber == kf_chamber
    return True  # unknown prefix → allow


def _try_aliases(first: str, last_parts: list[str], existing: dict[str, str]) -> str | None:
    """Try nickname/formal name aliases for the first name."""
    first_lower = first.lower()
    aliases: list[str] = []
    # nickname → formal
    if first_lower in _NICKNAMES:
        aliases.extend(_NICKNAMES[first_lower])
    # formal → nicknames
    if first_lower in _FORMAL_TO_NICK:
        aliases.extend(_FORMAL_TO_NICK[first_lower])

    last_str = " ".join(last_parts)
    for alias in aliases:
        candidate = f"{alias} {last_str}"
        if candidate in existing:
            return existing[candidate]
    return None


def match_to_existing(
    kf_name: str,
    kf_chamber: str,
    kf_district: str,
    existing: dict[str, str],
) -> str | None:
    """Try to match a KanFocus legislator to an existing slug.

    Checks normalized name against the existing mapping with four
    strategies: direct match, original name, first+last only, and
    nickname/alias expansion. Chamber prefix validation rejects
    mismatches (e.g., Senate slug for a House vote).
    """
    if not existing:
        return None

    normalized = normalize_name(kf_name).lower().strip()

    # Strategy 1: direct match on normalized name
    slug = existing.get(normalized)
    if slug and _chamber_compatible(slug, kf_chamber):
        return slug

    # Strategy 2: original name (KanFocus may include middle initials the scraper stripped)
    slug = existing.get(kf_name.lower().strip())
    if slug and _chamber_compatible(slug, kf_chamber):
        return slug

    # Strategy 3: first name + last name only (drop middle parts)
    parts = normalized.split()
    if len(parts) >= 2:
        first_last = f"{parts[0]} {parts[-1]}"
        slug = existing.get(first_last)
        if slug and _chamber_compatible(slug, kf_chamber):
            return slug

    # Strategy 4: nickname/alias expansion
    if len(parts) >= 2:
        slug = _try_aliases(parts[0], parts[1:], existing)
        if slug and _chamber_compatible(slug, kf_chamber):
            return slug

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
