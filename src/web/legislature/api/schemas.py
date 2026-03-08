"""Pydantic v2 response schemas for the Tallgrass API.

List schemas omit large text fields; detail schemas include them.
Semicolon-joined fields are parsed into JSON arrays via field validators.
Ninja's Schema base class sets from_attributes=True by default.
"""

from datetime import date, datetime

from ninja import Schema
from pydantic import field_validator


def _split_semicolons(v: str) -> list[str]:
    """Split semicolon-joined string into a list, stripping whitespace."""
    if not v:
        return []
    return [s.strip() for s in v.split(";") if s.strip()]


# -- Session -----------------------------------------------------------------


class SessionOut(Schema):
    """Session list item."""

    id: int
    state_id: str
    start_year: int
    end_year: int
    is_special: bool
    legislature_number: int
    name: str


class SessionDetail(SessionOut):
    """Session detail with aggregate counts."""

    legislator_count: int = 0
    rollcall_count: int = 0


# -- Legislator --------------------------------------------------------------


class LegislatorOut(Schema):
    """Legislator — all fields are small, no list/detail split needed."""

    id: int
    session_id: int
    name: str
    full_name: str
    legislator_slug: str
    chamber: str
    party: str
    district: str
    member_url: str
    ocd_id: str


# -- RollCall ----------------------------------------------------------------


class RollCallOut(Schema):
    """Roll call list item — omits large text fields."""

    id: int
    session_id: int
    bill_number: str
    vote_id: str
    vote_url: str
    vote_datetime: datetime | None
    vote_date: date | None
    chamber: str
    vote_type: str
    short_title: str
    sponsor: str
    sponsor_slugs: list[str]
    yea_count: int
    nay_count: int
    present_passing_count: int
    absent_not_voting_count: int
    not_voting_count: int
    total_votes: int
    passed: bool | None

    @field_validator("sponsor_slugs", mode="before")
    @classmethod
    def parse_sponsor_slugs(cls, v):
        if isinstance(v, str):
            return _split_semicolons(v)
        return v


class NestedVoteOut(Schema):
    """Vote nested inside a RollCall detail response."""

    id: int
    legislator_slug: str
    legislator_name: str
    vote: str


class RollCallDetail(RollCallOut):
    """Roll call detail — includes bill_title, motion, result, and nested votes."""

    bill_title: str
    motion: str
    result: str
    votes: list[NestedVoteOut] = []


# -- Vote --------------------------------------------------------------------


class VoteOut(Schema):
    """Individual vote with inline legislator and rollcall info."""

    id: int
    rollcall_id: int
    legislator_id: int
    vote: str
    legislator_slug: str
    legislator_name: str
    rollcall_vote_id: str
    rollcall_bill_number: str


# -- BillAction --------------------------------------------------------------


class BillActionOut(Schema):
    """Bill action — committee_names parsed from semicolons."""

    id: int
    session_id: int
    bill_number: str
    action_code: str
    chamber: str
    committee_names: list[str]
    occurred_datetime: datetime | None
    session_date: date | None
    status: str
    journal_page_number: str

    @field_validator("committee_names", mode="before")
    @classmethod
    def parse_committee_names(cls, v):
        if isinstance(v, str):
            return _split_semicolons(v)
        return v


# -- BillText ----------------------------------------------------------------


class BillTextOut(Schema):
    """Bill text list item — omits full text."""

    id: int
    session_id: int
    bill_number: str
    document_type: str
    version: str
    page_count: int
    source_url: str


class BillTextDetail(BillTextOut):
    """Bill text detail — includes full extracted text."""

    text: str


# -- ALEC Model Bill ---------------------------------------------------------


class ALECOut(Schema):
    """ALEC model bill list item — omits full text."""

    id: int
    title: str
    category: str
    bill_type: str
    date: str
    url: str
    task_force: str


class ALECDetail(ALECOut):
    """ALEC model bill detail — includes full text."""

    text: str
