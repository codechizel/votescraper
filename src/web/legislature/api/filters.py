"""FilterSchema classes for query parameter filtering.

All fields default to None — omitting a parameter skips that filter.
Uses FilterLookup annotations (django-ninja >= 1.5 style).
"""

from datetime import date
from typing import Annotated

from ninja import FilterLookup, FilterSchema


class SessionFilter(FilterSchema):
    state: str | None = None
    is_special: bool | None = None


class LegislatorFilter(FilterSchema):
    session: Annotated[int | None, FilterLookup(q="session_id")] = None
    session_name: Annotated[str | None, FilterLookup(q="session__name__icontains")] = None
    chamber: str | None = None
    party: str | None = None
    search: Annotated[str | None, FilterLookup(q=["name__icontains", "slug__icontains"])] = None


class RollCallFilter(FilterSchema):
    session: Annotated[int | None, FilterLookup(q="session_id")] = None
    chamber: str | None = None
    bill_number: str | None = None
    passed: bool | None = None
    date_from: Annotated[date | None, FilterLookup(q="vote_date__gte")] = None
    date_to: Annotated[date | None, FilterLookup(q="vote_date__lte")] = None
    search: Annotated[
        str | None, FilterLookup(q=["bill_number__icontains", "bill_title__icontains"])
    ] = None


class VoteFilter(FilterSchema):
    session: Annotated[int | None, FilterLookup(q="rollcall__session_id")] = None
    legislator: Annotated[int | None, FilterLookup(q="legislator_id")] = None
    legislator_slug: Annotated[str | None, FilterLookup(q="legislator__legislator_slug")] = None
    rollcall: Annotated[int | None, FilterLookup(q="rollcall_id")] = None
    vote: str | None = None


class BillActionFilter(FilterSchema):
    session: Annotated[int | None, FilterLookup(q="session_id")] = None
    bill_number: str | None = None
    chamber: str | None = None
    action_code: str | None = None


class BillTextFilter(FilterSchema):
    session: Annotated[int | None, FilterLookup(q="session_id")] = None
    bill_number: str | None = None
    document_type: str | None = None


class ALECFilter(FilterSchema):
    category: str | None = None
    task_force: Annotated[str | None, FilterLookup(q="task_force__icontains")] = None
    search: Annotated[str | None, FilterLookup(q=["title__icontains", "text__icontains"])] = None
