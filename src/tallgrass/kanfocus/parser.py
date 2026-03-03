"""Pure parsing functions for KanFocus vote tally pages.

KanFocus tally pages have a consistent structure across all sessions (1999-2026):

1. **Metadata** — ``Vote #: N  Date: MM/DD/YYYY  Bill Number: ...  Question: ...  Result: ...``
2. **Counts table** — For/Against/Present/Not Voting × All Members/Republicans/Democrats
3. **Legislator sections** — grouped by category (``Yea (N)``, ``Nay (N)``, etc.)
   with ``document.write()`` JS for column layout interspersed between names.

Legislator entries follow ``Name, Party-District`` format, e.g.:
- ``Steve Abrams, R-32nd``
- ``Thomas C. (Tim) Owens, R-8th``
- ``Ramon Gonzalez Jr., R-47th``
- ``Melody McCray Miller, D-89th``

Empty/nonexistent vote pages return all-zero counts and blank metadata.
"""

import re

from tallgrass.kanfocus.models import KanFocusLegislator, KanFocusVoteRecord

# Vote category headers as they appear on KanFocus pages
_CATEGORY_PATTERN = re.compile(r"(Yea|Nay|Present|Not Voting)\s*\((\d+)\)")

# Legislator entry: "Name, Party-District" where Party is R/D/I and District is ordinal
_LEGISLATOR_PATTERN = re.compile(r"([A-Z][^,]+),\s*([RDI])-(\d+(?:st|nd|rd|th))")

# KanFocus → tallgrass category mapping
CATEGORY_MAP: dict[str, str] = {
    "Yea": "Yea",
    "Nay": "Nay",
    "Present": "Present and Passing",
    "Not Voting": "Not Voting",
}


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text using BeautifulSoup.

    KanFocus pages use ``<table>`` layout and ``document.write()`` JS for
    column formatting. BeautifulSoup's ``get_text()`` extracts readable text
    with newline separators that the regex-based parser can process.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


def parse_vote_page(
    html_or_text: str,
    vote_num: int,
    year: int,
    chamber: str,
    source_url: str,
) -> KanFocusVoteRecord | None:
    """Parse a KanFocus vote tally page into a ``KanFocusVoteRecord``.

    Accepts either raw HTML or pre-extracted text. If the input looks like HTML
    (starts with ``<`` or contains ``<html``), it is automatically converted to
    text via BeautifulSoup. Returns ``None`` for empty/nonexistent vote pages.

    Args:
        html_or_text: Raw HTML or extracted text from a tally page.
        vote_num: Vote number from URL.
        year: Year from URL.
        chamber: Chamber from URL ("H" or "S").
        source_url: Original URL for provenance.
    """
    text = html_or_text
    stripped = text.strip()
    if stripped.startswith("<!") or stripped.startswith("<html") or "<html" in stripped[:500]:
        text = _html_to_text(html_or_text)

    if is_empty_page(text):
        return None

    metadata = _parse_metadata(text)
    if not metadata:
        return None

    counts = _parse_counts(text)
    legislators = _parse_legislators_by_category(text)

    return KanFocusVoteRecord(
        vote_num=vote_num,
        year=year,
        chamber=chamber,
        date=metadata["date"],
        bill_number=metadata["bill_number"],
        question=metadata["question"],
        result=metadata["result"],
        yea_count=counts.get("for", 0),
        nay_count=counts.get("against", 0),
        present_count=counts.get("present", 0),
        not_voting_count=counts.get("not_voting", 0),
        legislators=tuple(legislators),
        source_url=source_url,
    )


def is_empty_page(text: str) -> bool:
    """Detect an empty/nonexistent vote page.

    KanFocus returns a page with blank metadata and all-zero counts for
    vote IDs that don't exist.
    """
    if not text or not text.strip():
        return True
    # Empty pages have "Vote #:" with no number following
    if re.search(r"Vote\s*#:\s*\d", text):
        return False
    # If there's no vote number, it's empty
    if "Vote #:" in text or "Vote #" in text:
        # Has the field but no value
        return not re.search(r"Vote\s*#\s*:?\s*\d", text)
    return True


def _parse_metadata(text: str) -> dict[str, str] | None:
    """Extract vote metadata from page text.

    Returns dict with keys: vote_num_str, date, bill_number, question, result.
    Returns None if metadata cannot be parsed.
    """
    # Vote #: 33  Date: 02/03/2011  Bill Number: SB 13  Question: ...  Result: ...
    vote_match = re.search(r"Vote\s*#:\s*(\d+)", text)
    date_match = re.search(r"Date:\s*(\d{2}/\d{2}/\d{4})", text)
    bill_match = re.search(r"Bill\s*Number:\s*(.+?)(?:\s*Question:)", text, re.DOTALL)
    question_match = re.search(r"Question:\s*(.+?)(?:\s*Result:)", text, re.DOTALL)
    result_match = re.search(r"Result:\s*(\S+(?:\s+\S+)*?)(?:\s+All\s+Members|\s*\n)", text)

    if not vote_match or not date_match:
        return None

    return {
        "vote_num_str": vote_match.group(1),
        "date": date_match.group(1).strip(),
        "bill_number": bill_match.group(1).strip() if bill_match else "",
        "question": question_match.group(1).strip() if question_match else "",
        "result": result_match.group(1).strip() if result_match else "",
    }


def _parse_counts(text: str) -> dict[str, int]:
    """Extract vote count totals from the counts table.

    Looks for the "All Members" column values for For/Against/Present/Not Voting.
    """
    counts: dict[str, int] = {"for": 0, "against": 0, "present": 0, "not_voting": 0}

    # The counts table has rows like "For\n\n\n38\n\n\n31%\n..." when extracted
    # from HTML. We want the first number after each label (the "All Members" count).
    for label, key in [
        ("For", "for"),
        ("Against", "against"),
        ("Present", "present"),
        ("Not Voting", "not_voting"),
    ]:
        # Handle both inline ("For 38 100%") and newline-separated formats
        pattern = rf"(?:^|\n|\s){re.escape(label)}\s+(\d+)"
        match = re.search(pattern, text)
        if match:
            counts[key] = int(match.group(1))

    return counts


def _parse_legislators_by_category(text: str) -> list[KanFocusLegislator]:
    """Parse legislator entries grouped by vote category.

    KanFocus pages list legislators under category headers like ``Yea (38)``.
    Between entries, ``document.write()`` JS code handles column layout.
    We strip the JS and parse ``Name, Party-District`` entries.
    """
    legislators: list[KanFocusLegislator] = []

    # Find all category sections
    category_positions: list[tuple[int, str, int]] = []
    for match in _CATEGORY_PATTERN.finditer(text):
        category_positions.append((match.end(), match.group(1), int(match.group(2))))

    if not category_positions:
        return legislators

    # For each category, extract legislators between this header and the next
    for i, (start_pos, category, expected_count) in enumerate(category_positions):
        if i + 1 < len(category_positions):
            # Section ends at the "var acell" before the next category
            end_pos = text.rfind("var acell", start_pos, category_positions[i + 1][0])
            if end_pos == -1:
                end_pos = category_positions[i + 1][0]
        else:
            end_pos = len(text)

        section = text[start_pos:end_pos]
        tallgrass_category = CATEGORY_MAP.get(category, category)

        for leg in _parse_legislator_entries(section, tallgrass_category):
            legislators.append(leg)

    return legislators


def _parse_legislator_entries(section_text: str, vote_category: str) -> list[KanFocusLegislator]:
    """Parse individual legislator entries from a category section.

    Strips JS code (``document.write()``, ``if (acell == ...)``) and extracts
    ``Name, Party-District`` entries.
    """
    # Remove JS code blocks
    cleaned = re.sub(
        r"if\s*\(acell\s*==\s*\d+\)\s*\{[^}]*\}\s*else\s*\{[^}]*\}\s*;?",
        "\n",
        section_text,
    )
    cleaned = re.sub(r"var\s+acell\s*=\s*\d+\s*,\s*x\s*;\s*x\s*=\s*acell\s*;?", "\n", cleaned)
    cleaned = re.sub(r"document\.write\([^)]*\)\s*;?", "\n", cleaned)

    results: list[KanFocusLegislator] = []
    for match in _LEGISLATOR_PATTERN.finditer(cleaned):
        name = match.group(1).strip()
        party = match.group(2)
        district = match.group(3)
        results.append(
            KanFocusLegislator(
                name=name,
                party=party,
                district=district,
                vote_category=vote_category,
            )
        )

    return results


def parse_legislator_entry(text: str) -> tuple[str, str, str] | None:
    """Parse a single ``Name, Party-District`` entry.

    Returns (name, party, district) or None if the text doesn't match.

    >>> parse_legislator_entry("Steve Abrams, R-32nd")
    ('Steve Abrams', 'R', '32nd')
    >>> parse_legislator_entry("Thomas C. (Tim) Owens, R-8th")
    ('Thomas C. (Tim) Owens', 'R', '8th')
    """
    match = _LEGISLATOR_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).strip(), match.group(2), match.group(3)
