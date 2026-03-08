"""Tests for Django legislature models.

Run: uv run pytest tests/test_django_models.py -v
Requires: PostgreSQL running (just db-up)
"""

import os
import sys

import pytest

django = pytest.importorskip("django")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tallgrass_web.settings.test")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "web"))
django.setup()

from django.db import IntegrityError  # noqa: E402
from legislature.models import (  # noqa: E402
    ALECModelBill,
    BillAction,
    BillText,
    Legislator,
    RollCall,
    Session,
    State,
    Vote,
)

pytestmark = [pytest.mark.web, pytest.mark.django_db]


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def state():
    return State.objects.create(code="KS", name="Kansas")


@pytest.fixture
def session(state):
    return Session.objects.create(
        state=state,
        start_year=2025,
        end_year=2026,
        is_special=False,
        legislature_number=91,
        name="91st_2025-2026",
    )


@pytest.fixture
def legislator(session):
    return Legislator.objects.create(
        session=session,
        name="Masterson",
        full_name="Ty Masterson",
        legislator_slug="sen_masterson_ty_1",
        chamber="Senate",
        party="Republican",
        district="16",
    )


@pytest.fixture
def rollcall(session):
    return RollCall.objects.create(
        session=session,
        bill_number="SB 1",
        bill_title="Test bill",
        vote_id="je_20250320203513",
        chamber="Senate",
        yea_count=28,
        nay_count=12,
        total_votes=40,
        passed=True,
    )


# -- State -------------------------------------------------------------------


class TestState:
    """State model tests."""

    def test_create(self, state):
        assert state.code == "KS"
        assert state.name == "Kansas"

    def test_str(self, state):
        assert str(state) == "KS"

    def test_primary_key_is_code(self, state):
        fetched = State.objects.get(pk="KS")
        assert fetched == state

    def test_duplicate_code_raises(self, state):
        with pytest.raises(IntegrityError):
            State.objects.create(code="KS", name="Kansas Duplicate")


# -- Session -----------------------------------------------------------------


class TestSession:
    """Session model tests."""

    def test_create(self, session):
        assert session.start_year == 2025
        assert session.end_year == 2026
        assert session.legislature_number == 91
        assert session.is_special is False

    def test_str(self, session):
        assert "KS" in str(session)
        assert "91st_2025-2026" in str(session)

    def test_str_special(self, state):
        special = Session.objects.create(
            state=state,
            start_year=2024,
            end_year=2024,
            is_special=True,
            legislature_number=90,
            name="2024s",
        )
        assert "s" in str(special)

    def test_unique_constraint(self, session, state):
        """Duplicate (state, start_year, is_special) is rejected."""
        with pytest.raises(IntegrityError):
            Session.objects.create(
                state=state,
                start_year=2025,
                end_year=2026,
                is_special=False,
                legislature_number=91,
                name="91st_duplicate",
            )

    def test_special_same_year_allowed(self, session, state):
        """Same state+year but is_special=True is a different session."""
        special = Session.objects.create(
            state=state,
            start_year=2025,
            end_year=2025,
            is_special=True,
            legislature_number=91,
            name="2025s",
        )
        assert special.pk != session.pk

    def test_fk_to_state(self, session):
        assert session.state.code == "KS"

    def test_cascade_delete(self, session, state):
        """Deleting state cascades to session."""
        state.delete()
        assert Session.objects.count() == 0


# -- Legislator --------------------------------------------------------------


class TestLegislator:
    """Legislator model tests."""

    def test_create(self, legislator):
        assert legislator.legislator_slug == "sen_masterson_ty_1"
        assert legislator.chamber == "Senate"
        assert legislator.party == "Republican"

    def test_str(self, legislator):
        assert "sen_masterson_ty_1" in str(legislator)

    def test_empty_party_default(self, session):
        """Empty party string matches scraper convention for Independents."""
        leg = Legislator.objects.create(
            session=session,
            name="Pyle",
            legislator_slug="sen_pyle_dennis_1",
            chamber="Senate",
        )
        assert leg.party == ""

    def test_empty_ocd_id_default(self, legislator):
        assert legislator.ocd_id == ""

    def test_unique_slug_per_session(self, legislator, session):
        with pytest.raises(IntegrityError):
            Legislator.objects.create(
                session=session,
                name="Masterson Copy",
                legislator_slug="sen_masterson_ty_1",
                chamber="Senate",
            )

    def test_same_slug_different_session(self, legislator, state):
        """Same slug in a different session is fine."""
        other = Session.objects.create(
            state=state,
            start_year=2023,
            end_year=2024,
            is_special=False,
            legislature_number=90,
            name="90th_2023-2024",
        )
        leg2 = Legislator.objects.create(
            session=other,
            name="Masterson",
            legislator_slug="sen_masterson_ty_1",
            chamber="Senate",
        )
        assert leg2.pk != legislator.pk

    def test_cascade_delete(self, legislator, session):
        session.delete()
        assert Legislator.objects.count() == 0


# -- RollCall ----------------------------------------------------------------


class TestRollCall:
    """RollCall model tests."""

    def test_create(self, rollcall):
        assert rollcall.vote_id == "je_20250320203513"
        assert rollcall.passed is True
        assert rollcall.yea_count == 28

    def test_str(self, rollcall):
        assert "SB 1" in str(rollcall)
        assert "je_20250320203513" in str(rollcall)

    def test_nullable_passed(self, session):
        rc = RollCall.objects.create(
            session=session,
            bill_number="SB 2",
            vote_id="je_20250321120000",
            chamber="Senate",
            passed=None,
        )
        assert rc.passed is None

    def test_nullable_vote_datetime(self, session):
        rc = RollCall.objects.create(
            session=session,
            bill_number="SB 3",
            vote_id="je_20250322120000",
            chamber="House",
        )
        assert rc.vote_datetime is None

    def test_unique_vote_id_per_session(self, rollcall, session):
        with pytest.raises(IntegrityError):
            RollCall.objects.create(
                session=session,
                bill_number="SB 99",
                vote_id="je_20250320203513",
                chamber="House",
            )

    def test_kanfocus_vote_id(self, session):
        """KanFocus-sourced vote_ids use kf_ prefix."""
        rc = RollCall.objects.create(
            session=session,
            bill_number="SB 5",
            vote_id="kf_33_2011_S",
            chamber="Senate",
        )
        assert rc.vote_id.startswith("kf_")

    def test_cascade_delete(self, rollcall, session):
        session.delete()
        assert RollCall.objects.count() == 0


# -- Vote --------------------------------------------------------------------


class TestVote:
    """Vote model tests (individual legislator votes)."""

    def test_create(self, rollcall, legislator):
        v = Vote.objects.create(
            rollcall=rollcall,
            legislator=legislator,
            vote="Yea",
        )
        assert v.vote == "Yea"

    def test_str(self, rollcall, legislator):
        v = Vote.objects.create(
            rollcall=rollcall,
            legislator=legislator,
            vote="Nay",
        )
        s = str(v)
        assert "sen_masterson_ty_1" in s
        assert "Nay" in s

    def test_unique_per_rollcall_and_legislator(self, rollcall, legislator):
        Vote.objects.create(rollcall=rollcall, legislator=legislator, vote="Yea")
        with pytest.raises(IntegrityError):
            Vote.objects.create(rollcall=rollcall, legislator=legislator, vote="Nay")

    def test_cascade_delete_rollcall(self, rollcall, legislator):
        Vote.objects.create(rollcall=rollcall, legislator=legislator, vote="Yea")
        rollcall.delete()
        assert Vote.objects.count() == 0

    def test_cascade_delete_legislator(self, rollcall, legislator):
        Vote.objects.create(rollcall=rollcall, legislator=legislator, vote="Yea")
        legislator.delete()
        assert Vote.objects.count() == 0

    def test_all_vote_categories(self, rollcall, session):
        """All 5 canonical vote categories can be stored."""
        categories = [
            "Yea",
            "Nay",
            "Present and Passing",
            "Absent and Not Voting",
            "Not Voting",
        ]
        for i, cat in enumerate(categories):
            leg = Legislator.objects.create(
                session=session,
                name=f"Leg{i}",
                legislator_slug=f"rep_test_{i}",
                chamber="House",
            )
            Vote.objects.create(rollcall=rollcall, legislator=leg, vote=cat)
        assert Vote.objects.filter(rollcall=rollcall).count() == 5


# -- BillAction --------------------------------------------------------------


class TestBillAction:
    """BillAction model tests."""

    def test_create(self, session):
        ba = BillAction.objects.create(
            session=session,
            bill_number="SB 1",
            action_code="sign_enroll",
            chamber="Senate",
            status="Signed by Governor",
        )
        assert ba.action_code == "sign_enroll"

    def test_str(self, session):
        ba = BillAction.objects.create(
            session=session,
            bill_number="HB 2084",
            action_code="intro",
            chamber="House",
        )
        assert "HB 2084" in str(ba)

    def test_semicolon_joined_committees(self, session):
        """committee_names is stored as semicolon-joined text."""
        ba = BillAction.objects.create(
            session=session,
            bill_number="SB 10",
            action_code="refer",
            chamber="Senate",
            committee_names="Judiciary; Ways and Means",
        )
        assert ";" in ba.committee_names

    def test_nullable_datetime(self, session):
        ba = BillAction.objects.create(
            session=session,
            bill_number="SB 11",
            action_code="intro",
            chamber="Senate",
        )
        assert ba.occurred_datetime is None
        assert ba.session_date is None


# -- BillText ----------------------------------------------------------------


class TestBillText:
    """BillText model tests."""

    def test_create(self, session):
        bt = BillText.objects.create(
            session=session,
            bill_number="SB 1",
            document_type="introduced",
            version="00_0000",
            text="AN ACT concerning...",
            page_count=5,
            source_url="https://example.com/sb1.pdf",
        )
        assert bt.document_type == "introduced"
        assert bt.page_count == 5

    def test_str(self, session):
        bt = BillText.objects.create(
            session=session,
            bill_number="HB 2084",
            document_type="supp_note",
            text="Supplemental note text",
        )
        assert "HB 2084" in str(bt)
        assert "supp_note" in str(bt)

    def test_cascade_delete(self, session):
        BillText.objects.create(
            session=session,
            bill_number="SB 1",
            document_type="introduced",
            text="test",
        )
        session.delete()
        assert BillText.objects.count() == 0


# -- ALECModelBill -----------------------------------------------------------


class TestALECModelBill:
    """ALECModelBill model tests."""

    def test_create(self):
        bill = ALECModelBill.objects.create(
            title="Castle Doctrine Act",
            text="Section 1. Self-defense...",
            category="Criminal Justice",
            bill_type="Model Policy",
            url="https://alec.org/model-policy/castle-doctrine-act/",
            task_force="Public Safety",
        )
        assert bill.title == "Castle Doctrine Act"

    def test_str(self):
        bill = ALECModelBill.objects.create(
            title="Right to Work Act",
            text="Section 1...",
        )
        assert str(bill) == "Right to Work Act"

    def test_no_session_fk(self):
        """ALECModelBill has no session FK — it's a standalone corpus."""
        bill = ALECModelBill.objects.create(
            title="Test",
            text="test",
        )
        assert not hasattr(bill, "session")

    def test_empty_optional_fields(self):
        bill = ALECModelBill.objects.create(
            title="Minimal",
            text="text",
        )
        assert bill.category == ""
        assert bill.bill_type == ""
        assert bill.date == ""
        assert bill.task_force == ""


# -- Cross-model queries -----------------------------------------------------


class TestCrossModel:
    """Tests for FK relationships and queries across models."""

    def test_session_legislators_reverse(self, session, legislator):
        assert session.legislators.count() == 1
        assert session.legislators.first() == legislator

    def test_session_rollcalls_reverse(self, session, rollcall):
        assert session.rollcalls.count() == 1

    def test_rollcall_votes_reverse(self, rollcall, legislator):
        Vote.objects.create(rollcall=rollcall, legislator=legislator, vote="Yea")
        assert rollcall.votes.count() == 1

    def test_legislator_votes_reverse(self, rollcall, legislator):
        Vote.objects.create(rollcall=rollcall, legislator=legislator, vote="Nay")
        assert legislator.votes.count() == 1
        assert legislator.votes.first().vote == "Nay"

    def test_state_sessions_reverse(self, state, session):
        assert state.sessions.count() == 1
