"""Data classes for vote records."""

from dataclasses import dataclass
from typing import Optional


@dataclass
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


@dataclass
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
    yea_count: int = 0
    nay_count: int = 0
    present_passing_count: int = 0
    absent_not_voting_count: int = 0
    not_voting_count: int = 0
    total_votes: int = 0
    passed: Optional[bool] = None
