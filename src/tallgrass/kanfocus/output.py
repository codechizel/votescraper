"""Convert KanFocus intermediate models to standard tallgrass format and save CSVs.

Produces the same 3 CSV files as the main scraper (votes, rollcalls, legislators)
so the analysis pipeline works unchanged. Reuses ``save_csvs()`` from
``tallgrass.output`` for the actual file writing.
"""

import csv
import re
import warnings
from datetime import datetime
from pathlib import Path

from tallgrass.kanfocus.models import KanFocusVoteRecord
from tallgrass.kanfocus.session import generate_vote_id
from tallgrass.kanfocus.slugs import build_slug_lookup
from tallgrass.models import IndividualVote, RollCall
from tallgrass.output import save_csvs


def _convert_date(kf_date: str) -> tuple[str, str]:
    """Convert KanFocus date format to tallgrass format.

    KanFocus uses ``MM/DD/YYYY``; tallgrass uses ``YYYY-MM-DDT00:00:00``
    for vote_datetime and ``MM/DD/YYYY`` for vote_date.

    >>> _convert_date("02/03/2011")
    ('2011-02-03T00:00:00', '02/03/2011')
    """
    try:
        dt = datetime.strptime(kf_date, "%m/%d/%Y")
        return dt.strftime("%Y-%m-%dT00:00:00"), kf_date
    except ValueError:
        return "", kf_date


def _derive_passed(result: str) -> bool | None:
    """Derive passed boolean from KanFocus result text.

    Mirrors ``KSVoteScraper._derive_passed()`` logic.
    """
    if not result:
        return None
    result_lower = result.lower()
    if re.search(r"\b(not\s+passed|failed|rejected)\b", result_lower):
        return False
    if "sustained" in result_lower:
        return False
    if re.search(r"\b(passed|adopted|prevailed|concurred)\b", result_lower):
        return True
    return None


def _classify_vote_type(question: str) -> tuple[str, str]:
    """Classify question text into vote_type and result.

    Mirrors ``KSVoteScraper._parse_vote_type_and_result()`` logic.
    """
    if not question:
        return "", ""

    q_lower = question.lower()

    type_prefixes = [
        ("Emergency Final Action", "emergency final action"),
        ("Final Action", "final action"),
        ("Committee of the Whole", "committee of the whole"),
        ("Consent Calendar", "consent calendar"),
    ]
    for vote_type, prefix in type_prefixes:
        if q_lower.startswith(prefix):
            remainder = question[len(prefix) :].strip(" -;")
            return vote_type, remainder if remainder else question

    if "override" in q_lower and "veto" in q_lower:
        return "Veto Override", question
    if "conference committee" in q_lower:
        return "Conference Committee", question
    if "concur" in q_lower:
        return "Concurrence", question
    if q_lower.startswith(("motion", "citing rule")):
        return "Procedural Motion", question

    return "", question


def _chamber_name(chamber: str) -> str:
    """Convert single-letter chamber to full name."""
    return "Senate" if chamber == "S" else "House"


def _safe_int(value: str | int) -> int:
    """Convert a CSV field to int, defaulting to 0 on empty or malformed values."""
    try:
        return int(value) if value else 0
    except ValueError, TypeError:
        return 0


def convert_to_standard(
    records: list[KanFocusVoteRecord],
    session_label: str,
    existing_slugs: dict[str, str],
) -> tuple[list[IndividualVote], list[RollCall], dict[str, dict]]:
    """Convert KanFocus records to standard tallgrass format.

    Returns (individual_votes, rollcalls, legislators_dict).
    """
    all_votes: list[IndividualVote] = []
    all_rollcalls: list[RollCall] = []
    legislators: dict[str, dict] = {}

    seen_vote_ids: set[str] = set()

    for record in records:
        vote_id = generate_vote_id(record.vote_num, record.year, record.chamber)

        # Deduplicate by vote_id (safety net — should never trigger with correct upstream)
        if vote_id in seen_vote_ids:
            continue
        seen_vote_ids.add(vote_id)

        vote_datetime, vote_date = _convert_date(record.date)
        chamber_name = _chamber_name(record.chamber)
        vote_type, motion_result = _classify_vote_type(record.question)
        passed = _derive_passed(record.result)

        # Skip voice votes / unrecorded votes (no individual legislator data)
        if not record.legislators:
            continue

        # Warn if tally counts don't match parsed legislator list
        n_legislators = len(record.legislators)
        expected_total = (
            record.yea_count + record.nay_count + record.present_count + record.not_voting_count
        )
        if n_legislators > 0 and n_legislators != expected_total:
            warnings.warn(
                f"Tally mismatch for {vote_id}: "
                f"expected {expected_total} legislators, parsed {n_legislators}",
                stacklevel=2,
            )

        rollcall = RollCall(
            session=session_label,
            bill_number=record.bill_number,
            bill_title="",  # KanFocus doesn't provide bill titles
            vote_id=vote_id,
            vote_url=record.source_url,
            vote_datetime=vote_datetime,
            vote_date=vote_date,
            chamber=chamber_name,
            motion=record.question,
            vote_type=vote_type,
            result=record.result,
            short_title="",
            sponsor="",
            sponsor_slugs="",
            yea_count=record.yea_count,
            nay_count=record.nay_count,
            present_passing_count=record.present_count,
            absent_not_voting_count=0,
            not_voting_count=record.not_voting_count,
            total_votes=record.yea_count + record.nay_count,
            passed=passed,
        )
        all_rollcalls.append(rollcall)

        for kf_leg in record.legislators:
            slug = build_slug_lookup(kf_leg.name, record.chamber, kf_leg.district, existing_slugs)

            vote = IndividualVote(
                session=session_label,
                bill_number=record.bill_number,
                bill_title="",
                vote_id=vote_id,
                vote_datetime=vote_datetime,
                vote_date=vote_date,
                chamber=chamber_name,
                motion=record.question,
                legislator_name=kf_leg.name,
                legislator_slug=slug,
                vote=kf_leg.vote_category,
            )
            all_votes.append(vote)

            # Build legislator dict (dedup by slug)
            if slug not in legislators:
                party_full = {"R": "Republican", "D": "Democrat", "I": "Independent"}.get(
                    kf_leg.party, ""
                )
                legislators[slug] = {
                    "name": kf_leg.name,
                    "full_name": kf_leg.name,
                    "legislator_slug": slug,
                    "chamber": chamber_name,
                    "party": party_full,
                    "district": kf_leg.district,
                    "member_url": "",
                    "ocd_id": "",
                }

    return all_votes, all_rollcalls, legislators


def save_full(
    output_dir: Path,
    output_name: str,
    votes: list[IndividualVote],
    rollcalls: list[RollCall],
    legislators: dict[str, dict],
) -> None:
    """Save full KanFocus scrape results using the standard CSV writer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csvs(output_dir, output_name, votes, rollcalls, legislators)


def merge_gap_fill(
    data_dir: Path,
    output_name: str,
    new_votes: list[IndividualVote],
    new_rollcalls: list[RollCall],
    new_legislators: dict[str, dict],
) -> None:
    """Merge new KanFocus votes into existing CSVs (gap-fill mode).

    Loads existing CSVs, appends new records that aren't already present
    (filtering by vote_id prefix ``kf_`` to avoid duplicates on re-run),
    and writes merged output.
    """
    from dataclasses import fields

    votes_path = data_dir / f"{output_name}_votes.csv"
    rollcalls_path = data_dir / f"{output_name}_rollcalls.csv"
    legislators_path = data_dir / f"{output_name}_legislators.csv"

    # Load existing votes (filter out previous kf_ entries for idempotent re-run)
    existing_votes: list[IndividualVote] = []
    if votes_path.exists():
        with open(votes_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fld_names = [fld.name for fld in fields(IndividualVote)]
            for row in reader:
                if not row.get("vote_id", "").startswith("kf_"):
                    existing_votes.append(IndividualVote(**{k: row.get(k, "") for k in fld_names}))

    # Load existing rollcalls (filter out previous kf_ entries for idempotent re-run)
    existing_rollcalls: list[RollCall] = []
    if rollcalls_path.exists():
        with open(rollcalls_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("vote_id", "").startswith("kf_"):
                    # Reconstruct RollCall with proper types
                    existing_rollcalls.append(
                        RollCall(
                            session=row.get("session", ""),
                            bill_number=row.get("bill_number", ""),
                            bill_title=row.get("bill_title", ""),
                            vote_id=row.get("vote_id", ""),
                            vote_url=row.get("vote_url", ""),
                            vote_datetime=row.get("vote_datetime", ""),
                            vote_date=row.get("vote_date", ""),
                            chamber=row.get("chamber", ""),
                            motion=row.get("motion", ""),
                            vote_type=row.get("vote_type", ""),
                            result=row.get("result", ""),
                            short_title=row.get("short_title", ""),
                            sponsor=row.get("sponsor", ""),
                            sponsor_slugs=row.get("sponsor_slugs", ""),
                            yea_count=_safe_int(row.get("yea_count", 0)),
                            nay_count=_safe_int(row.get("nay_count", 0)),
                            present_passing_count=_safe_int(row.get("present_passing_count", 0)),
                            absent_not_voting_count=_safe_int(
                                row.get("absent_not_voting_count", 0)
                            ),
                            not_voting_count=_safe_int(row.get("not_voting_count", 0)),
                            total_votes=_safe_int(row.get("total_votes", 0)),
                            passed=_parse_bool(row.get("passed", "")),
                        )
                    )

    # Load existing legislators
    existing_legislators: dict[str, dict] = {}
    if legislators_path.exists():
        with open(legislators_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                slug = row.get("legislator_slug", "")
                if slug:
                    existing_legislators[slug] = dict(row)

    # Deduplicate: only keep kf_ rollcalls where no je_ rollcall exists
    # for the same bill + chamber + date. This is the core gap-fill logic —
    # we only want votes that kslegislature.gov didn't have.
    existing_keys = {(rc.bill_number, rc.chamber, rc.vote_date) for rc in existing_rollcalls}
    gap_rollcalls = [
        rc
        for rc in new_rollcalls
        if (rc.bill_number, rc.chamber, rc.vote_date) not in existing_keys
    ]
    gap_vote_ids = {rc.vote_id for rc in gap_rollcalls}
    gap_votes = [v for v in new_votes if v.vote_id in gap_vote_ids]

    skipped = len(new_rollcalls) - len(gap_rollcalls)
    print(
        f"  Gap-fill merge: {len(gap_rollcalls)} new rollcalls "
        f"(skipped {skipped} already covered by kslegislature.gov)"
    )

    # Merge
    merged_votes = existing_votes + gap_votes
    merged_rollcalls = existing_rollcalls + gap_rollcalls
    merged_legislators = {**existing_legislators, **new_legislators}

    save_csvs(data_dir, output_name, merged_votes, merged_rollcalls, merged_legislators)


def _parse_bool(value: str) -> bool | None:
    """Parse a boolean string from CSV."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return None
