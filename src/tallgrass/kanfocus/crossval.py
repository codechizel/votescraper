"""Cross-validation between KanFocus and kslegislature.gov vote data.

Re-parses KanFocus cache and compares overlapping rollcalls against the
authoritative kslegislature.gov (``je_`` prefix) data. Pure read-only
diagnostic — no data mutation.

Vote category comparison rules:
    KF "Yea"        ↔ JE "Yea"                       → Match
    KF "Nay"        ↔ JE "Nay"                       → Match
    KF "Present"    ↔ JE "Present and Passing"        → Match (mapped)
    KF "Not Voting" ↔ JE "Not Voting"                → Match
    KF "Not Voting" ↔ JE "Absent and Not Voting"     → Compatible (ANV/NV ambiguity)
    Anything else                                      → Genuine mismatch
"""

import csv
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tallgrass.kanfocus.output import convert_to_standard
from tallgrass.kanfocus.parser import parse_vote_page
from tallgrass.kanfocus.session import (
    biennium_streams,
    session_id_for_biennium,
    vote_tally_url,
)
from tallgrass.kanfocus.slugs import normalize_name
from tallgrass.models import IndividualVote, RollCall

# KF→JE category equivalence for individual vote comparison
_KF_TO_JE_CATEGORY: dict[str, str] = {
    "Yea": "Yea",
    "Nay": "Nay",
    "Present and Passing": "Present and Passing",
    "Not Voting": "Not Voting",
}

# "Sub for" prefix pattern — strips nested substitution prefixes
_SUB_FOR_RE = re.compile(r"^(?:(?:[HS]\s+)?Sub\s+for\s+)+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoteMismatch:
    """One legislator whose vote differs between KF and JE."""

    slug: str
    name: str
    kf_vote: str
    je_vote: str
    compatible: bool  # True if ANV/NV ambiguity (not a genuine error)


@dataclass(frozen=True)
class RollCallComparison:
    """Comparison of one overlapping rollcall between KF and JE."""

    bill_number: str
    chamber: str
    vote_date: str
    kf_vote_id: str
    je_vote_id: str
    # Tally comparison
    yea_match: bool
    nay_match: bool
    present_match: bool
    nv_match: bool  # exact match on Not Voting
    nv_compatible: bool  # kf_nv == je_nv + je_anv
    passed_match: bool
    # Individual vote mismatches
    mismatches: tuple[VoteMismatch, ...]
    kf_only_slugs: tuple[str, ...]  # legislators in KF but not JE
    je_only_slugs: tuple[str, ...]  # legislators in JE but not KF


@dataclass
class CrossValReport:
    """Full biennium cross-validation report."""

    session_label: str
    total_kf_rollcalls: int = 0
    total_je_rollcalls: int = 0
    matched_rollcalls: int = 0
    unmatched_kf_rollcalls: int = 0
    # Tally agreement
    tally_perfect: int = 0  # all tallies match exactly
    tally_compatible: int = 0  # NV differs but kf_nv == je_nv + je_anv
    tally_mismatch: int = 0  # genuine tally mismatch
    passed_agree: int = 0
    passed_disagree: int = 0
    # Individual votes
    individual_perfect: int = 0  # all individual votes match
    individual_compatible: int = 0  # only ANV/NV differences
    individual_mismatch: int = 0  # genuine individual vote mismatches
    total_genuine_mismatches: int = 0
    total_compatible_mismatches: int = 0
    comparisons: list[RollCallComparison] = field(default_factory=list)
    normalizations: list[tuple[str, str]] = field(default_factory=list)  # (original, normalized)
    unmatched_kf_bills: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bill number normalization
# ---------------------------------------------------------------------------


def normalize_bill_number(bn: str) -> str:
    """Strip 'Sub for' prefixes and normalize whitespace.

    Handles nested substitution prefixes like "S Sub for Sub for HB 2007".

    >>> normalize_bill_number("Sub for HB 2007")
    'HB 2007'
    >>> normalize_bill_number("S Sub for Sub for HB 2007")
    'HB 2007'
    >>> normalize_bill_number("H Sub for SB 123")
    'SB 123'
    >>> normalize_bill_number("HB 2001")
    'HB 2001'
    >>> normalize_bill_number("  SB   55  ")
    'SB 55'
    """
    result = bn.strip()
    result = _SUB_FOR_RE.sub("", result).strip()
    # Collapse internal whitespace
    result = re.sub(r"\s+", " ", result)
    return result


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, url: str) -> Path:
    """Reproduce KanFocusFetcher's hash-keyed cache path."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return cache_dir / f"{url_hash}.html"


def load_kf_from_cache(
    cache_dir: Path,
    start_year: int,
    existing_slugs: dict[str, str],
    session_label: str,
    max_empty: int = 20,
) -> tuple[list[RollCall], list[IndividualVote]]:
    """Re-parse KanFocus data from cached HTML files.

    Iterates the same (vote_num, year, chamber) streams as the fetcher,
    reconstructing URLs to find cache files. No network access.

    Returns (rollcalls, individual_votes) in standard tallgrass format.
    """
    session_id = session_id_for_biennium(start_year)
    all_records = []

    for year, chamber in biennium_streams(start_year):
        vote_num = 1
        consecutive_empty = 0

        while consecutive_empty < max_empty:
            url = vote_tally_url(session_id, vote_num, year, chamber)
            cache_file = _cache_path(cache_dir, url)

            if not cache_file.exists():
                consecutive_empty += 1
                vote_num += 1
                continue

            html = cache_file.read_text(encoding="utf-8")
            record = parse_vote_page(html, vote_num, year, chamber, url)

            if record is None:
                consecutive_empty += 1
            else:
                consecutive_empty = 0
                all_records.append(record)

            vote_num += 1

    if not all_records:
        return [], []

    votes, rollcalls, _legislators = convert_to_standard(all_records, session_label, existing_slugs)
    return rollcalls, votes


# ---------------------------------------------------------------------------
# JE data loading
# ---------------------------------------------------------------------------


def load_je_data(data_dir: Path, output_name: str) -> tuple[list[RollCall], list[IndividualVote]]:
    """Load kslegislature.gov rollcalls and votes from CSVs.

    Only loads ``je_`` prefixed records (skips any ``kf_`` gap-fill data).
    """
    rollcalls_path = data_dir / f"{output_name}_rollcalls.csv"
    votes_path = data_dir / f"{output_name}_votes.csv"

    rollcalls: list[RollCall] = []
    if rollcalls_path.exists():
        with open(rollcalls_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("vote_id", "").startswith("je_"):
                    continue
                rollcalls.append(
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

    votes: list[IndividualVote] = []
    if votes_path.exists():
        with open(votes_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("vote_id", "").startswith("je_"):
                    continue
                votes.append(
                    IndividualVote(
                        session=row.get("session", ""),
                        bill_number=row.get("bill_number", ""),
                        bill_title=row.get("bill_title", ""),
                        vote_id=row.get("vote_id", ""),
                        vote_datetime=row.get("vote_datetime", ""),
                        vote_date=row.get("vote_date", ""),
                        chamber=row.get("chamber", ""),
                        motion=row.get("motion", ""),
                        legislator_name=row.get("legislator_name", ""),
                        legislator_slug=row.get("legislator_slug", ""),
                        vote=row.get("vote", ""),
                    )
                )

    return rollcalls, votes


def _parse_bool(value: str) -> bool | None:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return None


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _match_key(rc: RollCall) -> tuple[str, str, str]:
    """Build a match key from (normalized_bill_number, chamber, vote_date)."""
    return (normalize_bill_number(rc.bill_number), rc.chamber, rc.vote_date)


def _tally_key(rc: RollCall) -> tuple[int, int, int]:
    """Build a tally vector for sub-matching: (yea, nay, nv_total).

    For JE rollcalls, nv_total = not_voting + absent_not_voting (KanFocus
    merges these into a single not_voting count).
    """
    nv_total = rc.not_voting_count + rc.absent_not_voting_count
    return (rc.yea_count, rc.nay_count, nv_total)


def _match_by_tally(
    kf_list: list[RollCall],
    je_list: list[RollCall],
    matched: list[tuple[RollCall, RollCall]],
    unmatched: list[RollCall],
) -> None:
    """Sub-match rollcalls within a (bill, chamber, date) group by tally vector.

    Groups each side by tally key. For each tally that appears on both sides,
    pairs rollcalls positionally. Leftover KF rollcalls go to unmatched.
    """
    kf_by_tally: dict[tuple[int, int, int], list[RollCall]] = defaultdict(list)
    je_by_tally: dict[tuple[int, int, int], list[RollCall]] = defaultdict(list)

    for rc in kf_list:
        kf_by_tally[_tally_key(rc)].append(rc)
    for rc in je_list:
        je_by_tally[_tally_key(rc)].append(rc)

    used_je: set[str] = set()  # track consumed JE vote_ids

    for tally, kf_group in kf_by_tally.items():
        je_group = je_by_tally.get(tally, [])
        # Filter out already-consumed JE rollcalls
        available_je = [rc for rc in je_group if rc.vote_id not in used_je]

        for i, kf_rc in enumerate(kf_group):
            if i < len(available_je):
                matched.append((kf_rc, available_je[i]))
                used_je.add(available_je[i].vote_id)
            else:
                unmatched.append(kf_rc)


def find_matches(
    kf_rollcalls: list[RollCall],
    je_rollcalls: list[RollCall],
) -> tuple[list[tuple[RollCall, RollCall]], list[RollCall]]:
    """Match KF rollcalls to JE rollcalls on (bill_number, chamber, date).

    Bill numbers are normalized (Sub-for prefixes stripped) before matching.
    When multiple rollcalls share the same (bill, chamber, date) key,
    sub-matches by tally vector (yea, nay, nv_total) to disambiguate
    multiple motions on the same bill/day.

    Returns (matched_pairs, unmatched_kf_rollcalls).
    """
    kf_groups: dict[tuple[str, str, str], list[RollCall]] = defaultdict(list)
    je_groups: dict[tuple[str, str, str], list[RollCall]] = defaultdict(list)

    for rc in kf_rollcalls:
        kf_groups[_match_key(rc)].append(rc)
    for rc in je_rollcalls:
        je_groups[_match_key(rc)].append(rc)

    matched: list[tuple[RollCall, RollCall]] = []
    unmatched: list[RollCall] = []

    for key, kf_list in kf_groups.items():
        je_list = je_groups.get(key)
        if je_list is None:
            unmatched.extend(kf_list)
            continue

        if len(kf_list) == 1 and len(je_list) == 1:
            # Simple 1:1 — no tally disambiguation needed
            matched.append((kf_list[0], je_list[0]))
        else:
            # Multi-motion: sub-match by tally vector
            _match_by_tally(kf_list, je_list, matched, unmatched)

    return matched, unmatched


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _normalize_legislator_name(name: str) -> str:
    """Normalize a legislator name for cross-source matching."""
    return normalize_name(name).lower().strip()


def compare_individual_votes(
    kf_votes: list[IndividualVote],
    je_votes: list[IndividualVote],
) -> tuple[list[VoteMismatch], list[str], list[str]]:
    """Compare individual votes matched by legislator slug.

    Uses slug matching first, then falls back to normalized name matching
    for any slugs that appear on only one side (catches remaining slug
    mismatches between KF and JE).

    Returns (mismatches, kf_only_slugs, je_only_slugs).
    """
    kf_by_slug = {v.legislator_slug: v for v in kf_votes}
    je_by_slug = {v.legislator_slug: v for v in je_votes}

    mismatches: list[VoteMismatch] = []
    kf_only_set = set(kf_by_slug) - set(je_by_slug)
    je_only_set = set(je_by_slug) - set(kf_by_slug)

    # Name-based fallback: match KF-only slugs to JE-only slugs by name.
    # JE votes typically store last-name-only ("Barrett") while KF has full
    # names ("Brad Barrett"). Three strategies: full name, last-name match,
    # then the remaining unmatched go to kf_only/je_only.
    name_matched_kf: set[str] = set()
    name_matched_je: set[str] = set()
    if kf_only_set and je_only_set:
        je_name_index: dict[str, str] = {}  # normalized_name → je_slug
        for je_slug in je_only_set:
            je_v = je_by_slug[je_slug]
            norm = _normalize_legislator_name(je_v.legislator_name)
            if norm:
                je_name_index[norm] = je_slug

        for kf_slug in kf_only_set:
            kf_v = kf_by_slug[kf_slug]
            norm = _normalize_legislator_name(kf_v.legislator_name)
            if not norm:
                continue

            # Strategy 1: full normalized name match
            je_slug = je_name_index.get(norm)

            # Strategy 2: last-name match (KF "Brad Barrett" → last word "barrett" → JE "Barrett")
            if je_slug is None:
                parts = norm.split()
                if parts:
                    last = parts[-1]
                    if last in je_name_index:
                        je_slug = je_name_index[last]

            if je_slug is not None and je_slug not in name_matched_je:
                name_matched_kf.add(kf_slug)
                name_matched_je.add(je_slug)
                # Compare their votes using the KF slug as the label
                je_v = je_by_slug[je_slug]
                kf_cat = kf_v.vote
                je_cat = je_v.vote
                if kf_cat != je_cat:
                    compatible = (
                        kf_cat == "Not Voting" and je_cat == "Absent and Not Voting"
                    ) or (kf_cat == "Absent and Not Voting" and je_cat == "Not Voting")
                    mismatches.append(
                        VoteMismatch(
                            slug=kf_slug,
                            name=kf_v.legislator_name or je_v.legislator_name,
                            kf_vote=kf_cat,
                            je_vote=je_cat,
                            compatible=compatible,
                        )
                    )

    kf_only = sorted(kf_only_set - name_matched_kf)
    je_only = sorted(je_only_set - name_matched_je)

    for slug in sorted(set(kf_by_slug) & set(je_by_slug)):
        kf_v = kf_by_slug[slug]
        je_v = je_by_slug[slug]

        kf_cat = kf_v.vote
        je_cat = je_v.vote

        # Direct match (including mapped categories from convert_to_standard)
        if kf_cat == je_cat:
            continue

        # ANV/NV compatible: KF "Not Voting" ↔ JE "Absent and Not Voting"
        compatible = (kf_cat == "Not Voting" and je_cat == "Absent and Not Voting") or (
            kf_cat == "Absent and Not Voting" and je_cat == "Not Voting"
        )

        mismatches.append(
            VoteMismatch(
                slug=slug,
                name=kf_v.legislator_name or je_v.legislator_name,
                kf_vote=kf_cat,
                je_vote=je_cat,
                compatible=compatible,
            )
        )

    return mismatches, kf_only, je_only


def compare_rollcall(
    kf_rc: RollCall,
    je_rc: RollCall,
    kf_votes: list[IndividualVote],
    je_votes: list[IndividualVote],
) -> RollCallComparison:
    """Compare a single overlapping rollcall between KF and JE."""
    yea_match = kf_rc.yea_count == je_rc.yea_count
    nay_match = kf_rc.nay_count == je_rc.nay_count
    present_match = kf_rc.present_passing_count == je_rc.present_passing_count

    # NV comparison: KF has no "Absent and Not Voting" category
    # kf_not_voting should equal je_not_voting + je_absent_not_voting
    nv_exact = kf_rc.not_voting_count == je_rc.not_voting_count
    nv_compatible = kf_rc.not_voting_count == (
        je_rc.not_voting_count + je_rc.absent_not_voting_count
    )

    passed_match = kf_rc.passed == je_rc.passed

    mismatches, kf_only, je_only = compare_individual_votes(kf_votes, je_votes)

    return RollCallComparison(
        bill_number=kf_rc.bill_number,
        chamber=kf_rc.chamber,
        vote_date=kf_rc.vote_date,
        kf_vote_id=kf_rc.vote_id,
        je_vote_id=je_rc.vote_id,
        yea_match=yea_match,
        nay_match=nay_match,
        present_match=present_match,
        nv_match=nv_exact,
        nv_compatible=nv_compatible,
        passed_match=passed_match,
        mismatches=tuple(mismatches),
        kf_only_slugs=tuple(kf_only),
        je_only_slugs=tuple(je_only),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_crossval(
    session_label: str,
    start_year: int,
    data_dir: Path,
    cache_dir: Path,
    existing_slugs: dict[str, str],
) -> CrossValReport:
    """Run full cross-validation for a biennium.

    Loads KF data from cache, JE data from CSVs, matches rollcalls, and
    compares tallies + individual votes.
    """
    output_name = data_dir.name  # e.g. "91st_2025-2026"

    print("\n  Loading KanFocus data from cache...")
    kf_rollcalls, kf_votes = load_kf_from_cache(
        cache_dir, start_year, existing_slugs, session_label
    )
    print(f"    {len(kf_rollcalls)} KF rollcalls, {len(kf_votes)} KF individual votes")

    print("  Loading kslegislature.gov data from CSVs...")
    je_rollcalls, je_votes = load_je_data(data_dir, output_name)
    print(f"    {len(je_rollcalls)} JE rollcalls, {len(je_votes)} JE individual votes")

    report = CrossValReport(session_label=session_label)
    report.total_kf_rollcalls = len(kf_rollcalls)
    report.total_je_rollcalls = len(je_rollcalls)

    if not kf_rollcalls:
        print("\n  No KanFocus data in cache. Run `tallgrass-kanfocus` first.")
        return report

    if not je_rollcalls:
        print("\n  No kslegislature.gov data. Run `tallgrass` first.")
        return report

    # Track bill number normalizations that affected matching
    for rc in kf_rollcalls:
        normalized = normalize_bill_number(rc.bill_number)
        if normalized != rc.bill_number:
            report.normalizations.append((rc.bill_number, normalized))

    print("  Matching rollcalls...")
    matched, unmatched = find_matches(kf_rollcalls, je_rollcalls)
    report.matched_rollcalls = len(matched)
    report.unmatched_kf_rollcalls = len(unmatched)
    report.unmatched_kf_bills = [
        f"{rc.bill_number} ({rc.chamber}, {rc.vote_date})" for rc in unmatched
    ]

    print(f"    {len(matched)} matched, {len(unmatched)} unmatched KF rollcalls")

    # Build vote indexes by vote_id
    kf_votes_by_id: dict[str, list[IndividualVote]] = {}
    for v in kf_votes:
        kf_votes_by_id.setdefault(v.vote_id, []).append(v)

    je_votes_by_id: dict[str, list[IndividualVote]] = {}
    for v in je_votes:
        je_votes_by_id.setdefault(v.vote_id, []).append(v)

    print("  Comparing matched rollcalls...")
    for kf_rc, je_rc in matched:
        kf_rc_votes = kf_votes_by_id.get(kf_rc.vote_id, [])
        je_rc_votes = je_votes_by_id.get(je_rc.vote_id, [])

        comp = compare_rollcall(kf_rc, je_rc, kf_rc_votes, je_rc_votes)
        report.comparisons.append(comp)

        # Classify tally result
        tallies_exact = comp.yea_match and comp.nay_match and comp.present_match and comp.nv_match
        tallies_compatible = (
            comp.yea_match and comp.nay_match and comp.present_match and comp.nv_compatible
        )

        if tallies_exact:
            report.tally_perfect += 1
        elif tallies_compatible:
            report.tally_compatible += 1
        else:
            report.tally_mismatch += 1

        if comp.passed_match:
            report.passed_agree += 1
        else:
            report.passed_disagree += 1

        # Classify individual vote result
        genuine = [m for m in comp.mismatches if not m.compatible]
        compat = [m for m in comp.mismatches if m.compatible]

        if not comp.mismatches and not comp.kf_only_slugs and not comp.je_only_slugs:
            report.individual_perfect += 1
        elif not genuine:
            report.individual_compatible += 1
        else:
            report.individual_mismatch += 1

        report.total_genuine_mismatches += len(genuine)
        report.total_compatible_mismatches += len(compat)

    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(report: CrossValReport) -> str:
    """Format a CrossValReport as a markdown string."""
    lines: list[str] = []
    lines.append(f"# KanFocus Cross-Validation Report: {report.session_label}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|------:|")
    lines.append(f"| KF rollcalls in cache | {report.total_kf_rollcalls} |")
    lines.append(f"| JE rollcalls in CSVs | {report.total_je_rollcalls} |")
    lines.append(f"| Matched rollcalls | {report.matched_rollcalls} |")
    lines.append(f"| Unmatched KF rollcalls | {report.unmatched_kf_rollcalls} |")
    lines.append("")

    if report.matched_rollcalls == 0:
        lines.append("No overlapping rollcalls found — nothing to compare.")
        return "\n".join(lines)

    # Tally agreement
    lines.append("## Tally Agreement")
    lines.append("")
    lines.append("| Category | Count | % |")
    lines.append("|----------|------:|--:|")
    total = report.matched_rollcalls
    for label, count in [
        ("Exact match", report.tally_perfect),
        ("Compatible (ANV/NV merged)", report.tally_compatible),
        ("Genuine mismatch", report.tally_mismatch),
    ]:
        pct = count / total * 100 if total else 0
        lines.append(f"| {label} | {count} | {pct:.1f}% |")
    lines.append("")

    # Passed agreement
    lines.append("## Passed/Failed Agreement")
    lines.append("")
    lines.append("| Category | Count | % |")
    lines.append("|----------|------:|--:|")
    for label, count in [
        ("Agree", report.passed_agree),
        ("Disagree", report.passed_disagree),
    ]:
        pct = count / total * 100 if total else 0
        lines.append(f"| {label} | {count} | {pct:.1f}% |")
    lines.append("")

    # Individual vote agreement
    lines.append("## Individual Vote Agreement")
    lines.append("")
    lines.append("| Category | Rollcalls | % |")
    lines.append("|----------|----------:|--:|")
    for label, count in [
        ("All votes match", report.individual_perfect),
        ("Only ANV/NV differences", report.individual_compatible),
        ("Genuine mismatches", report.individual_mismatch),
    ]:
        pct = count / total * 100 if total else 0
        lines.append(f"| {label} | {count} | {pct:.1f}% |")
    lines.append("")
    lines.append(f"Total genuine individual vote mismatches: **{report.total_genuine_mismatches}**")
    lines.append(f"Total compatible (ANV/NV) differences: **{report.total_compatible_mismatches}**")
    lines.append("")

    # Tally mismatches (detail)
    tally_mismatches = [
        c
        for c in report.comparisons
        if not (c.yea_match and c.nay_match and c.present_match and c.nv_compatible)
    ]
    if tally_mismatches:
        lines.append("## Tally Mismatches (Detail)")
        lines.append("")
        lines.append(
            "| Bill | Chamber | Date | KF Vote ID | JE Vote ID | "
            "Yea | Nay | Present | NV | NV Compat |"
        )
        lines.append(
            "|------|---------|------|------------|------------|-----|-----|---------|----|----|"
        )
        for c in tally_mismatches:
            lines.append(
                f"| {c.bill_number} | {c.chamber} | {c.vote_date} "
                f"| {c.kf_vote_id} | {c.je_vote_id} "
                f"| {'ok' if c.yea_match else 'DIFF'} "
                f"| {'ok' if c.nay_match else 'DIFF'} "
                f"| {'ok' if c.present_match else 'DIFF'} "
                f"| {'ok' if c.nv_match else 'DIFF'} "
                f"| {'ok' if c.nv_compatible else 'DIFF'} |"
            )
        lines.append("")

    # Individual vote mismatches (detail)
    comps_with_mismatches = [c for c in report.comparisons if c.mismatches]
    if comps_with_mismatches:
        lines.append("## Individual Vote Mismatches (Detail)")
        lines.append("")
        for c in comps_with_mismatches:
            genuine = [m for m in c.mismatches if not m.compatible]
            compat = [m for m in c.mismatches if m.compatible]
            lines.append(f"### {c.bill_number} — {c.chamber} — {c.vote_date}")
            lines.append("")
            if genuine:
                lines.append("**Genuine mismatches:**")
                lines.append("")
                lines.append("| Legislator | Slug | KF Vote | JE Vote |")
                lines.append("|------------|------|---------|---------|")
                for m in genuine:
                    lines.append(f"| {m.name} | {m.slug} | {m.kf_vote} | {m.je_vote} |")
                lines.append("")
            if compat:
                lines.append("**Compatible (ANV/NV):**")
                lines.append("")
                lines.append("| Legislator | Slug | KF Vote | JE Vote |")
                lines.append("|------------|------|---------|---------|")
                for m in compat:
                    lines.append(f"| {m.name} | {m.slug} | {m.kf_vote} | {m.je_vote} |")
                lines.append("")
            if c.kf_only_slugs:
                lines.append(f"**KF-only legislators:** {', '.join(c.kf_only_slugs)}")
                lines.append("")
            if c.je_only_slugs:
                lines.append(f"**JE-only legislators:** {', '.join(c.je_only_slugs)}")
                lines.append("")

    # Bill number normalizations
    if report.normalizations:
        lines.append("## Bill Number Normalizations")
        lines.append("")
        lines.append("| Original | Normalized |")
        lines.append("|----------|------------|")
        seen = set()
        for orig, norm in report.normalizations:
            key = (orig, norm)
            if key not in seen:
                seen.add(key)
                lines.append(f"| {orig} | {norm} |")
        lines.append("")

    # Unmatched KF rollcalls
    if report.unmatched_kf_bills:
        lines.append("## Unmatched KF Rollcalls")
        lines.append("")
        lines.append(
            "These KanFocus rollcalls had no matching kslegislature.gov rollcall "
            "(same bill + chamber + date):"
        )
        lines.append("")
        for bill in report.unmatched_kf_bills:
            lines.append(f"- {bill}")
        lines.append("")

    return "\n".join(lines)
