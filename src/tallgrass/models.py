"""Data classes for vote records."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IndividualVote:
    """One legislator's vote on one roll call."""

    session: str
    bill_number: str
    bill_title: str
    vote_id: str
    vote_datetime: str
    vote_date: str
    chamber: str
    motion: str
    legislator_name: str
    legislator_slug: str
    vote: str  # Yea, Nay, Present and Passing, Absent and Not Voting, Not Voting


@dataclass(frozen=True)
class RollCall:
    """Summary of one roll call vote."""

    session: str
    bill_number: str
    bill_title: str
    vote_id: str
    vote_url: str
    vote_datetime: str
    vote_date: str
    chamber: str
    motion: str
    vote_type: str
    result: str
    short_title: str
    sponsor: str
    sponsor_slugs: str = ""  # semicolon-joined legislator slugs from bill page <a> hrefs
    yea_count: int = 0
    nay_count: int = 0
    present_passing_count: int = 0
    absent_not_voting_count: int = 0
    not_voting_count: int = 0
    total_votes: int = 0
    passed: bool | None = None


@dataclass(frozen=True)
class BillAction:
    """One action in a bill's legislative history."""

    session: str
    bill_number: str
    action_code: str
    chamber: str
    committee_names: tuple[str, ...]  # frozen dataclass → tuple not list
    occurred_datetime: str
    session_date: str
    status: str
    journal_page_number: str
