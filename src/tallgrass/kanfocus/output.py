"""Convert KanFocus intermediate models to standard tallgrass format and save CSVs.

Produces the same 3 CSV files as the main scraper (votes, rollcalls, legislators)
so the analysis pipeline works unchanged. Reuses ``save_csvs()`` from
``tallgrass.output`` for the actual file writing.
"""

import csv
import re
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

    for record in records:
        vote_id = generate_vote_id(record.vote_num, record.year, record.chamber)
        vote_datetime, vote_date = _convert_date(record.date)
        chamber_name = _chamber_name(record.chamber)
        vote_type, motion_result = _classify_vote_type(record.question)
        passed = _derive_passed(record.result)

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
                    "slug": slug,
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
                            yea_count=int(row.get("yea_count", 0) or 0),
                            nay_count=int(row.get("nay_count", 0) or 0),
                            present_passing_count=int(row.get("present_passing_count", 0) or 0),
                            absent_not_voting_count=int(row.get("absent_not_voting_count", 0) or 0),
                            not_voting_count=int(row.get("not_voting_count", 0) or 0),
                            total_votes=int(row.get("total_votes", 0) or 0),
                            passed=_parse_bool(row.get("passed", "")),
                        )
                    )

    # Load existing legislators
    existing_legislators: dict[str, dict] = {}
    if legislators_path.exists():
        with open(legislators_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                slug = row.get("slug", "")
                if slug:
                    existing_legislators[slug] = dict(row)

    # Merge
    merged_votes = existing_votes + new_votes
    merged_rollcalls = existing_rollcalls + new_rollcalls
    merged_legislators = {**existing_legislators, **new_legislators}

    print(
        f"  Gap-fill merge: +{len(new_votes)} votes, +{len(new_rollcalls)} rollcalls, "
        f"+{len(new_legislators)} legislators"
    )

    save_csvs(data_dir, output_name, merged_votes, merged_rollcalls, merged_legislators)


def _parse_bool(value: str) -> bool | None:
    """Parse a boolean string from CSV."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return None
