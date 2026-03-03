"""Intermediate data models for KanFocus vote parsing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KanFocusLegislator:
    """A legislator's vote on a single roll call, as parsed from KanFocus.

    Intermediate representation — converted to ``IndividualVote`` by output.py.
    """

    name: str  # "Steve Abrams", "Thomas C. (Tim) Owens"
    party: str  # "R", "D", "I"
    district: str  # "32nd", "8th", "113th"
    vote_category: str  # "Yea", "Nay", "Present", "Not Voting"


@dataclass(frozen=True)
class KanFocusVoteRecord:
    """One roll call vote as parsed from a KanFocus tally page.

    Intermediate representation — converted to ``RollCall`` + ``IndividualVote``
    list by output.py.
    """

    vote_num: int
    year: int
    chamber: str  # "H" or "S"
    date: str  # "MM/DD/YYYY" as displayed on page
    bill_number: str  # "SB 13", "Sub for HR 6004"
    question: str  # "On final action", "On agreeing to the amendment"
    result: str  # "Passed", "Failed"
    yea_count: int
    nay_count: int
    present_count: int
    not_voting_count: int
    legislators: tuple[KanFocusLegislator, ...]  # frozen -> tuple
    source_url: str
