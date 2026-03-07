"""Parse ODT (OpenDocument Text) vote files from Kansas Legislature (2011-2014).

Sessions before 2015 use downloadable .odt files instead of HTML vote_view pages.
These are ZIP archives containing content.xml with structured metadata (user-field
declarations) and paragraph text listing vote categories and legislators.

This module is pure logic — no I/O, no HTTP.  It takes ODT bytes and context,
returning IndividualVote/RollCall instances compatible with the HTML parser output.
"""

import json
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from io import BytesIO

from tallgrass.config import BILL_TITLE_MAX_LENGTH
from tallgrass.models import IndividualVote, RollCall

# Namespace map for ODF content.xml
_NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}

# Map ODT vote category names (both House and Senate variants) to our canonical names.
# House uses "Present but not voting" / "Absent or not voting";
# Senate uses "Present and Passing" / "Absent or Not Voting".
_ODT_CATEGORY_MAP: dict[str, str] = {
    "yeas": "Yea",
    "nays": "Nay",
    "present and passing": "Present and Passing",
    "present but not voting": "Present and Passing",
    "absent or not voting": "Absent and Not Voting",
}


@dataclass(frozen=True)
class OdtMetadata:
    """Structured metadata extracted from ODT user-field declarations."""

    chamber: str  # "house" or "senate"
    bill_numbers: list[str]  # e.g., ["HB2101"]
    occurred: str  # "2011/02/25 11:43:27"
    vote_tally: str  # "66:53"
    action_code: str  # "fa_fabc_341"


def parse_odt_votes(
    odt_bytes: bytes,
    bill_number: str,
    bill_path: str,
    vote_url: str,
    session_label: str,
    member_directory: dict[tuple[str, str], dict] | None = None,
    bill_metadata: dict[str, dict] | None = None,
) -> tuple[list[RollCall], list[IndividualVote], list[dict]]:
    """Parse an ODT vote file into RollCall and IndividualVote records.

    Args:
        odt_bytes: Raw bytes of the .odt file.
        bill_number: Bill identifier (e.g., "SB 1") from the bill page.
        bill_path: URL path to the bill page.
        vote_url: Full URL of the ODT file.
        session_label: Session label (e.g., "85th (2013-2014)").
        member_directory: Optional mapping of ``(chamber_lower, last_name_lower)``
            to ``{"slug": ..., "name": ..., "chamber": ...}``.
        bill_metadata: Optional mapping of normalized bill codes to
            ``{"short_title": ..., "sponsor": ...}``.

    Returns:
        (rollcalls, individual_votes, new_legislators) where new_legislators
        is a list of dicts suitable for merging into the scraper's legislators dict.
    """
    xml_str = _extract_content_xml(odt_bytes)
    meta = _parse_odt_metadata(xml_str)
    body_text = _extract_body_text(xml_str)

    # Derive chamber label
    chamber = meta.chamber.capitalize() if meta.chamber else ""

    # Parse vote datetime from the occurred timestamp
    vote_datetime = _parse_occurred_datetime(meta.occurred)
    vote_date = _format_vote_date(meta.occurred)

    # Generate vote_id from URL (same je_* pattern as HTML pages)
    vote_id_match = re.search(r"odt_view/([^/.]+)", vote_url)
    vote_id = vote_id_match.group(1) if vote_id_match else vote_url

    # Parse categories from body text
    categories, result_text = _parse_odt_body_votes(body_text, chamber, member_directory)

    # Look up metadata
    from tallgrass.scraper import _normalize_bill_code

    bill_code = _normalize_bill_code(bill_number)
    meta_info = (bill_metadata or {}).get(bill_code, {})
    short_title = meta_info.get("short_title", "")
    sponsor = meta_info.get("sponsor", "")

    # Derive vote_type and passed from result_text
    vote_type, result = _classify_motion(result_text, meta.action_code)
    passed = _derive_passed(result_text)

    # Extract bill title from body text (first line often has it)
    bill_title = _extract_bill_title(body_text)
    if len(bill_title) > BILL_TITLE_MAX_LENGTH:
        warnings.warn(
            f"{bill_number} title truncated ({len(bill_title)} -> {BILL_TITLE_MAX_LENGTH} chars)",
            stacklevel=2,
        )
        bill_title = bill_title[:BILL_TITLE_MAX_LENGTH]

    total_votes = sum(len(members) for members in categories.values())

    rollcalls: list[RollCall] = []
    individual_votes: list[IndividualVote] = []
    new_legislators: list[dict] = []

    if total_votes == 0:
        return rollcalls, individual_votes, new_legislators

    rollcall = RollCall(
        session=session_label,
        bill_number=bill_number,
        bill_title=bill_title,
        vote_id=vote_id,
        vote_url=vote_url,
        vote_datetime=vote_datetime,
        vote_date=vote_date,
        chamber=chamber,
        motion=result_text,
        vote_type=vote_type,
        result=result,
        short_title=short_title,
        sponsor=sponsor,
        yea_count=len(categories.get("Yea", [])),
        nay_count=len(categories.get("Nay", [])),
        present_passing_count=len(categories.get("Present and Passing", [])),
        absent_not_voting_count=len(categories.get("Absent and Not Voting", [])),
        not_voting_count=len(categories.get("Not Voting", [])),
        total_votes=total_votes,
        passed=passed,
    )
    rollcalls.append(rollcall)

    for category, members in categories.items():
        for member in members:
            iv = IndividualVote(
                session=session_label,
                bill_number=bill_number,
                bill_title=bill_title,
                vote_id=vote_id,
                vote_datetime=vote_datetime,
                vote_date=vote_date,
                chamber=chamber,
                motion=result_text,
                legislator_name=member["name"],
                legislator_slug=member["slug"],
                vote=category,
            )
            individual_votes.append(iv)

            if member["slug"]:
                new_legislators.append(
                    {
                        "legislator_slug": member["slug"],
                        "name": member["name"],
                        "chamber": chamber,
                    }
                )

    return rollcalls, individual_votes, new_legislators


def _extract_content_xml(odt_bytes: bytes) -> str:
    """Extract content.xml from an ODT (ZIP) archive."""
    try:
        with zipfile.ZipFile(BytesIO(odt_bytes)) as zf:
            return zf.read("content.xml").decode("utf-8")
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError) as e:
        warnings.warn(f"Failed to extract content.xml from ODT: {e}", stacklevel=2)
        return ""


def _parse_odt_metadata(xml_str: str) -> OdtMetadata:
    """Extract structured metadata from user-field-decl elements in content.xml."""
    if not xml_str:
        return OdtMetadata(chamber="", bill_numbers=[], occurred="", vote_tally="", action_code="")

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return OdtMetadata(chamber="", bill_numbers=[], occurred="", vote_tally="", action_code="")

    fields: dict[str, str] = {}
    for decl in root.iter(f"{{{_NS['text']}}}user-field-decl"):
        name = decl.get(f"{{{_NS['text']}}}name", "")
        # Value can be in office:string-value or office:value
        value = decl.get(f"{{{_NS['office']}}}string-value", "")
        if not value:
            value = decl.get(f"{{{_NS['office']}}}value", "")
        if name:
            fields[name] = value

    chamber = fields.get("T_JE_S_CHAMBER", "")
    bill_numbers_raw = fields.get("T_JE_T_BILLNUMBER", "[]")
    occurred = fields.get("T_JE_DT_OCCURRED", "")
    vote_tally = fields.get("T_JE_T_VOTE", "")
    action_code = fields.get("T_JE_S_ACTIONCODE", "")

    try:
        bill_numbers = json.loads(bill_numbers_raw)
        if not isinstance(bill_numbers, list):
            bill_numbers = [str(bill_numbers)]
    except ValueError:
        bill_numbers = []

    return OdtMetadata(
        chamber=chamber.lower().strip(),
        bill_numbers=bill_numbers,
        occurred=occurred,
        vote_tally=vote_tally,
        action_code=action_code,
    )


def _extract_body_text(xml_str: str) -> str:
    """Extract all paragraph text from content.xml body."""
    if not xml_str:
        return ""

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return ""

    paragraphs: list[str] = []
    body = root.find(f"{{{_NS['office']}}}body")
    if body is None:
        return ""

    text_elem = body.find(f"{{{_NS['office']}}}text")
    if text_elem is None:
        return ""

    for p in text_elem.findall(f"{{{_NS['text']}}}p"):
        # Collect all text content from spans and direct text
        parts: list[str] = []
        if p.text:
            parts.append(p.text)
        for child in p:
            if child.text:
                parts.append(child.text)
            if child.tail:
                parts.append(child.tail)
        line = " ".join("".join(parts).split())
        if line:
            paragraphs.append(line)

    return "\n".join(paragraphs)


def _parse_odt_body_votes(
    body_text: str,
    chamber: str,
    member_directory: dict[tuple[str, str], dict] | None = None,
) -> tuple[dict[str, list[dict]], str]:
    """Parse vote categories and result from ODT body text.

    Returns:
        (categories, result_text) where categories maps canonical category names
        to lists of ``{"name": ..., "slug": ...}`` dicts.
    """
    from tallgrass.scraper import VOTE_CATEGORIES

    categories: dict[str, list[dict]] = {cat: [] for cat in VOTE_CATEGORIES}
    result_text = ""

    if not body_text:
        return categories, result_text

    lines = body_text.split("\n")

    # Find result text — look for "passed", "failed", "concurred", etc.
    for line in lines:
        lower = line.lower().strip()
        if any(
            kw in lower
            for kw in ("passed", "failed", "adopted", "rejected", "concurred", "sustained")
        ):
            # Prefer lines that look like result sentences
            if lower.startswith("the ") or lower.startswith("motion"):
                result_text = line.strip()
                break

    # If no explicit result line, try the action from motion text
    if not result_text:
        for line in lines:
            lower = line.lower()
            if "final action" in lower or "considered on final" in lower:
                result_text = line.strip()
                break

    # Parse vote categories from "Yeas: name1, name2, ..." lines
    current_category: str | None = None
    chamber_lower = chamber.lower() if chamber else ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this line starts a vote category
        category_match = re.match(
            r"(Yeas|Nays|Present and Passing|Present but not voting"
            r"|Absent or [Nn]ot [Vv]oting)\s*[:\-]\s*(.*)",
            stripped,
            re.I,
        )
        if category_match:
            cat_label = category_match.group(1).lower()
            canonical = _ODT_CATEGORY_MAP.get(cat_label)
            if canonical:
                current_category = canonical
                remainder = category_match.group(2).strip()
                if remainder and remainder.lower() != "none.":
                    _add_names_to_category(
                        categories, current_category, remainder, chamber_lower, member_directory
                    )
            continue

        # If we're in a category and this line has names (comma-separated)
        if current_category and not stripped.lower().startswith(("on roll call", "the ")):
            # Check if this looks like a continuation of names
            if re.match(r"^[A-Z]", stripped) and ":" not in stripped[:20]:
                _add_names_to_category(
                    categories, current_category, stripped, chamber_lower, member_directory
                )
            else:
                current_category = None

    return categories, result_text


def _add_names_to_category(
    categories: dict[str, list[dict]],
    category: str,
    names_text: str,
    chamber_lower: str,
    member_directory: dict[tuple[str, str], dict] | None,
) -> None:
    """Parse comma-separated legislator names and add to the category."""
    # Clean trailing period
    names_text = names_text.rstrip(".")

    # Split on comma
    for raw_name in names_text.split(","):
        name = raw_name.strip()
        if not name or name.lower() == "none":
            continue

        slug, full_name = _resolve_last_name(name, chamber_lower, member_directory)
        categories[category].append({"name": full_name or name, "slug": slug})


def _resolve_last_name(
    name: str,
    chamber_lower: str,
    member_directory: dict[tuple[str, str], dict] | None,
) -> tuple[str, str]:
    """Resolve a last name (possibly with initial) to a slug and full name.

    Args:
        name: Legislator name as it appears in the ODT (e.g., "Smith", "C. Holmes").
        chamber_lower: Chamber in lowercase ("house" or "senate").
        member_directory: Mapping of ``(chamber_lower, last_name_lower)`` to member info.

    Returns:
        (slug, full_name) — empty strings if no match found.
    """
    if not member_directory or not chamber_lower:
        return "", name

    # Handle initials: "C. Holmes" → initial="C", last="Holmes"
    initial = ""
    last_name = name
    init_match = re.match(r"^([A-Z])\.\s+(.+)$", name)
    if init_match:
        initial = init_match.group(1)
        last_name = init_match.group(2)

    key = (chamber_lower, last_name.lower())
    entry = member_directory.get(key)

    if entry is None:
        return "", name

    # If the entry is marked ambiguous and we have an initial, try to disambiguate
    if entry.get("ambiguous") and initial:
        # Look for the specific initial-qualified key
        init_key = (chamber_lower, f"{initial.lower()}. {last_name.lower()}")
        init_entry = member_directory.get(init_key)
        if init_entry:
            return init_entry.get("slug", ""), init_entry.get("name", name)
        # Can't disambiguate — return empty slug
        return "", name

    if entry.get("ambiguous"):
        return "", name

    return entry.get("slug", ""), entry.get("name", name)


def _parse_occurred_datetime(occurred: str) -> str:
    """Convert ODT timestamp 'YYYY/MM/DD HH:MM:SS' to ISO 8601."""
    if not occurred:
        return ""
    match = re.match(r"(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", occurred)
    if match:
        y, mo, d, h, mi, s = match.groups()
        return f"{y}-{mo}-{d}T{h}:{mi}:{s}"
    return ""


def _format_vote_date(occurred: str) -> str:
    """Convert ODT timestamp to MM/DD/YYYY date string (matching HTML format)."""
    if not occurred:
        return ""
    match = re.match(r"(\d{4})/(\d{2})/(\d{2})", occurred)
    if match:
        y, mo, d = match.groups()
        return f"{mo}/{d}/{y}"
    return ""


def _extract_bill_title(body_text: str) -> str:
    """Extract bill title from ODT body text.

    The first paragraph typically contains the bill description, e.g.:
    "HB 2101, AN ACT concerning taxation; ..."
    """
    if not body_text:
        return ""

    for line in body_text.split("\n"):
        stripped = line.strip()
        # Look for "AN ACT" or similar
        act_match = re.search(r"(AN ACT|A CONCURRENT|A RESOLUTION|A JOINT)", stripped, re.I)
        if act_match:
            # Return from the match to end of line
            return stripped[act_match.start() :].strip()

    return ""


def _classify_motion(result_text: str, action_code: str) -> tuple[str, str]:
    """Derive vote_type and result from ODT result text and action code."""
    if not result_text:
        return "", ""

    result_lower = result_text.lower()

    # Use action code for classification when available
    if action_code.startswith("fa_"):
        if "emergency" in result_lower:
            return "Emergency Final Action", result_text
        return "Final Action", result_text

    if "concur" in result_lower:
        return "Concurrence", result_text
    if "override" in result_lower and "veto" in result_lower:
        return "Veto Override", result_text
    if "committee of the whole" in result_lower:
        return "Committee of the Whole", result_text
    if action_code.startswith("misc_"):
        return "Procedural Motion", result_text

    return "", result_text


def _derive_passed(result_text: str) -> bool | None:
    """Derive passed boolean from ODT result text."""
    if not result_text:
        return None
    lower = result_text.lower()
    if re.search(r"\b(not\s+passed|failed|rejected)\b", lower):
        return False
    if "sustained" in lower:
        return False
    if re.search(r"\b(passed|adopted|prevailed|concurred)\b", lower):
        return True
    return None
