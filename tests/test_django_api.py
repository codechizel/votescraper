"""Tests for the Tallgrass REST API (Django Ninja).

Run: uv run pytest tests/test_django_api.py -v
Requires: PostgreSQL running (just db-up)
"""

import os
import sys

import pytest

django = pytest.importorskip("django")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tallgrass_web.settings.test")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "web"))
django.setup()

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

ninja = pytest.importorskip("ninja")
from ninja.testing import TestClient  # noqa: E402

from legislature.api import api  # noqa: E402
from legislature.api.schemas import _split_semicolons  # noqa: E402

pytestmark = [pytest.mark.web, pytest.mark.django_db]

client = TestClient(api)


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
def session_special(state):
    return Session.objects.create(
        state=state,
        start_year=2024,
        end_year=2024,
        is_special=True,
        legislature_number=90,
        name="2024s",
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
        member_url="https://kslegislature.gov/li/b2025_26/members/sen_masterson_ty_1/",
        ocd_id="ocd-person/abc123",
    )


@pytest.fixture
def legislator_d(session):
    return Legislator.objects.create(
        session=session,
        name="Haley",
        full_name="Tom Haley",
        legislator_slug="rep_haley_tom_1",
        chamber="House",
        party="Democrat",
        district="42",
    )


@pytest.fixture
def rollcall(session):
    return RollCall.objects.create(
        session=session,
        bill_number="SB 1",
        bill_title="Test bill title",
        vote_id="je_20250320203513",
        chamber="Senate",
        motion="Emergency Final Action",
        vote_type="roll_call",
        result="Passed",
        short_title="Test short title",
        sponsor="Sen. Masterson",
        sponsor_slugs="sen_masterson_ty_1; sen_smith_john_1",
        yea_count=28,
        nay_count=12,
        total_votes=40,
        passed=True,
    )


@pytest.fixture
def vote(rollcall, legislator):
    return Vote.objects.create(
        rollcall=rollcall,
        legislator=legislator,
        vote="Yea",
    )


@pytest.fixture
def bill_action(session):
    return BillAction.objects.create(
        session=session,
        bill_number="SB 1",
        action_code="INTRO",
        chamber="Senate",
        committee_names="Judiciary; Federal and State Affairs",
        status="Introduced",
    )


@pytest.fixture
def bill_text(session):
    return BillText.objects.create(
        session=session,
        bill_number="SB 1",
        document_type="introduced",
        version="",
        text="AN ACT concerning the judiciary.",
        page_count=3,
        source_url="https://example.com/sb1.pdf",
    )


@pytest.fixture
def alec_bill():
    return ALECModelBill.objects.create(
        title="Model Voter ID Act",
        text="Section 1. Short Title. This act may be cited as...",
        category="Elections",
        bill_type="Model Policy",
        date="2020-01-15",
        url="https://alec.org/model-policy/voter-id-act",
        task_force="Homeland Security",
    )


# -- Health Check ------------------------------------------------------------


class TestHealthCheck:
    """Health check endpoint."""

    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


# -- Sessions ----------------------------------------------------------------


class TestSessions:
    """Session endpoints."""

    def test_list_empty(self):
        response = client.get("/sessions/")
        assert response.status_code == 200
        assert response.json()["items"] == []

    def test_list_with_data(self, session):
        response = client.get("/sessions/")
        assert response.status_code == 200
        items = response.json()["items"]
        assert len(items) == 1
        assert items[0]["name"] == "91st_2025-2026"
        assert items[0]["state_id"] == "KS"

    def test_detail(self, session, legislator, rollcall):
        response = client.get(f"/sessions/{session.id}/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "91st_2025-2026"
        assert data["legislator_count"] == 1
        assert data["rollcall_count"] == 1

    def test_detail_not_found(self):
        response = client.get("/sessions/99999/")
        assert response.status_code == 404

    def test_filter_by_state(self, session):
        response = client.get("/sessions/?state=KS")
        assert len(response.json()["items"]) == 1
        response = client.get("/sessions/?state=MO")
        assert len(response.json()["items"]) == 0

    def test_filter_by_is_special(self, session, session_special):
        response = client.get("/sessions/?is_special=true")
        items = response.json()["items"]
        assert len(items) == 1
        assert items[0]["is_special"] is True

        response = client.get("/sessions/?is_special=false")
        items = response.json()["items"]
        assert len(items) == 1
        assert items[0]["is_special"] is False


# -- Legislators -------------------------------------------------------------


class TestLegislators:
    """Legislator endpoints."""

    def test_list_empty(self):
        response = client.get("/legislators/")
        assert response.status_code == 200
        assert response.json()["items"] == []

    def test_list_with_data(self, legislator):
        response = client.get("/legislators/")
        items = response.json()["items"]
        assert len(items) == 1
        assert items[0]["legislator_slug"] == "sen_masterson_ty_1"
        assert items[0]["ocd_id"] == "ocd-person/abc123"

    def test_detail(self, legislator):
        response = client.get(f"/legislators/{legislator.id}/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Masterson"
        assert data["chamber"] == "Senate"

    def test_detail_not_found(self):
        response = client.get("/legislators/99999/")
        assert response.status_code == 404

    def test_filter_by_session(self, legislator):
        response = client.get(f"/legislators/?session={legislator.session_id}")
        assert len(response.json()["items"]) == 1

    def test_filter_by_chamber(self, legislator, legislator_d):
        response = client.get("/legislators/?chamber=Senate")
        items = response.json()["items"]
        assert len(items) == 1
        assert items[0]["chamber"] == "Senate"

    def test_filter_by_party(self, legislator, legislator_d):
        response = client.get("/legislators/?party=Democrat")
        items = response.json()["items"]
        assert len(items) == 1
        assert items[0]["party"] == "Democrat"

    def test_filter_by_search(self, legislator, legislator_d):
        response = client.get("/legislators/?search=masterson")
        items = response.json()["items"]
        assert len(items) == 1
        assert items[0]["legislator_slug"] == "sen_masterson_ty_1"

    def test_filter_by_session_name(self, legislator):
        response = client.get("/legislators/?session_name=91st")
        items = response.json()["items"]
        assert len(items) == 1


# -- Roll Calls --------------------------------------------------------------


class TestRollCalls:
    """Roll call endpoints."""

    def test_list_empty(self):
        response = client.get("/rollcalls/")
        assert response.status_code == 200
        assert response.json()["items"] == []

    def test_list_with_data(self, rollcall):
        response = client.get("/rollcalls/")
        items = response.json()["items"]
        assert len(items) == 1
        item = items[0]
        assert item["bill_number"] == "SB 1"
        assert item["vote_id"] == "je_20250320203513"
        # List omits bill_title, motion, result
        assert "bill_title" not in item
        assert "motion" not in item
        assert "result" not in item

    def test_list_sponsor_slugs_parsed(self, rollcall):
        response = client.get("/rollcalls/")
        item = response.json()["items"][0]
        assert item["sponsor_slugs"] == ["sen_masterson_ty_1", "sen_smith_john_1"]

    def test_detail(self, rollcall, vote):
        response = client.get(f"/rollcalls/{rollcall.id}/")
        assert response.status_code == 200
        data = response.json()
        assert data["bill_title"] == "Test bill title"
        assert data["motion"] == "Emergency Final Action"
        assert data["result"] == "Passed"
        assert len(data["votes"]) == 1
        assert data["votes"][0]["legislator_slug"] == "sen_masterson_ty_1"
        assert data["votes"][0]["vote"] == "Yea"

    def test_detail_not_found(self):
        response = client.get("/rollcalls/99999/")
        assert response.status_code == 404

    def test_filter_by_session(self, rollcall):
        response = client.get(f"/rollcalls/?session={rollcall.session_id}")
        assert len(response.json()["items"]) == 1

    def test_filter_by_chamber(self, rollcall):
        response = client.get("/rollcalls/?chamber=Senate")
        assert len(response.json()["items"]) == 1
        response = client.get("/rollcalls/?chamber=House")
        assert len(response.json()["items"]) == 0

    def test_filter_by_bill_number(self, rollcall):
        response = client.get("/rollcalls/?bill_number=SB 1")
        assert len(response.json()["items"]) == 1
        response = client.get("/rollcalls/?bill_number=HB 999")
        assert len(response.json()["items"]) == 0

    def test_filter_by_passed(self, rollcall):
        response = client.get("/rollcalls/?passed=true")
        assert len(response.json()["items"]) == 1
        response = client.get("/rollcalls/?passed=false")
        assert len(response.json()["items"]) == 0

    def test_filter_by_search(self, rollcall):
        response = client.get("/rollcalls/?search=SB 1")
        assert len(response.json()["items"]) == 1


# -- Votes -------------------------------------------------------------------


class TestVotes:
    """Vote endpoints."""

    def test_list_empty(self):
        response = client.get("/votes/")
        assert response.status_code == 200
        assert response.json()["items"] == []

    def test_list_with_data(self, vote):
        response = client.get("/votes/")
        items = response.json()["items"]
        assert len(items) == 1
        item = items[0]
        assert item["vote"] == "Yea"
        assert item["legislator_slug"] == "sen_masterson_ty_1"
        assert item["legislator_name"] == "Masterson"
        assert item["rollcall_vote_id"] == "je_20250320203513"
        assert item["rollcall_bill_number"] == "SB 1"

    def test_filter_by_legislator_slug(self, vote):
        response = client.get("/votes/?legislator_slug=sen_masterson_ty_1")
        assert len(response.json()["items"]) == 1
        response = client.get("/votes/?legislator_slug=nonexistent")
        assert len(response.json()["items"]) == 0

    def test_filter_by_vote_value(self, vote):
        response = client.get("/votes/?vote=Yea")
        assert len(response.json()["items"]) == 1
        response = client.get("/votes/?vote=Nay")
        assert len(response.json()["items"]) == 0

    def test_filter_by_rollcall(self, vote):
        response = client.get(f"/votes/?rollcall={vote.rollcall_id}")
        assert len(response.json()["items"]) == 1

    def test_filter_by_session(self, vote):
        response = client.get(f"/votes/?session={vote.rollcall.session_id}")
        assert len(response.json()["items"]) == 1


# -- Bill Actions ------------------------------------------------------------


class TestBillActions:
    """Bill action endpoints."""

    def test_list_empty(self):
        response = client.get("/bill-actions/")
        assert response.status_code == 200
        assert response.json()["items"] == []

    def test_list_with_data(self, bill_action):
        response = client.get("/bill-actions/")
        items = response.json()["items"]
        assert len(items) == 1
        item = items[0]
        assert item["bill_number"] == "SB 1"
        assert item["action_code"] == "INTRO"

    def test_committee_names_parsed(self, bill_action):
        response = client.get("/bill-actions/")
        item = response.json()["items"][0]
        assert item["committee_names"] == ["Judiciary", "Federal and State Affairs"]

    def test_filter_by_session(self, bill_action):
        response = client.get(f"/bill-actions/?session={bill_action.session_id}")
        assert len(response.json()["items"]) == 1

    def test_filter_by_bill_number(self, bill_action):
        response = client.get("/bill-actions/?bill_number=SB 1")
        assert len(response.json()["items"]) == 1

    def test_filter_by_chamber(self, bill_action):
        response = client.get("/bill-actions/?chamber=Senate")
        assert len(response.json()["items"]) == 1

    def test_filter_by_action_code(self, bill_action):
        response = client.get("/bill-actions/?action_code=INTRO")
        assert len(response.json()["items"]) == 1
        response = client.get("/bill-actions/?action_code=SIGN")
        assert len(response.json()["items"]) == 0


# -- Bill Texts --------------------------------------------------------------


class TestBillTexts:
    """Bill text endpoints."""

    def test_list_empty(self):
        response = client.get("/bill-texts/")
        assert response.status_code == 200
        assert response.json()["items"] == []

    def test_list_omits_text(self, bill_text):
        response = client.get("/bill-texts/")
        items = response.json()["items"]
        assert len(items) == 1
        assert "text" not in items[0]
        assert items[0]["bill_number"] == "SB 1"
        assert items[0]["page_count"] == 3

    def test_detail_includes_text(self, bill_text):
        response = client.get(f"/bill-texts/{bill_text.id}/")
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "AN ACT concerning the judiciary."

    def test_detail_not_found(self):
        response = client.get("/bill-texts/99999/")
        assert response.status_code == 404

    def test_filter_by_session(self, bill_text):
        response = client.get(f"/bill-texts/?session={bill_text.session_id}")
        assert len(response.json()["items"]) == 1

    def test_filter_by_document_type(self, bill_text):
        response = client.get("/bill-texts/?document_type=introduced")
        assert len(response.json()["items"]) == 1
        response = client.get("/bill-texts/?document_type=enrolled")
        assert len(response.json()["items"]) == 0


# -- ALEC Model Bills --------------------------------------------------------


class TestALEC:
    """ALEC model bill endpoints."""

    def test_list_empty(self):
        response = client.get("/alec/")
        assert response.status_code == 200
        assert response.json()["items"] == []

    def test_list_omits_text(self, alec_bill):
        response = client.get("/alec/")
        items = response.json()["items"]
        assert len(items) == 1
        assert "text" not in items[0]
        assert items[0]["title"] == "Model Voter ID Act"
        assert items[0]["category"] == "Elections"

    def test_detail_includes_text(self, alec_bill):
        response = client.get(f"/alec/{alec_bill.id}/")
        assert response.status_code == 200
        data = response.json()
        assert "Section 1" in data["text"]

    def test_detail_not_found(self):
        response = client.get("/alec/99999/")
        assert response.status_code == 404

    def test_filter_by_category(self, alec_bill):
        response = client.get("/alec/?category=Elections")
        assert len(response.json()["items"]) == 1
        response = client.get("/alec/?category=Healthcare")
        assert len(response.json()["items"]) == 0

    def test_filter_by_task_force(self, alec_bill):
        response = client.get("/alec/?task_force=Homeland")
        assert len(response.json()["items"]) == 1

    def test_filter_by_search(self, alec_bill):
        response = client.get("/alec/?search=Voter ID")
        assert len(response.json()["items"]) == 1
        response = client.get("/alec/?search=nonexistent")
        assert len(response.json()["items"]) == 0


# -- Pagination --------------------------------------------------------------


class TestPagination:
    """Pagination behavior across endpoints."""

    def test_default_pagination(self, legislator):
        response = client.get("/legislators/")
        data = response.json()
        assert "items" in data
        assert "count" in data
        assert data["count"] == 1

    def test_custom_limit(self, legislator, legislator_d):
        response = client.get("/legislators/?limit=1")
        data = response.json()
        assert len(data["items"]) == 1
        assert data["count"] == 2

    def test_offset(self, legislator, legislator_d):
        response = client.get("/legislators/?limit=1&offset=1")
        data = response.json()
        assert len(data["items"]) == 1
        assert data["count"] == 2

    def test_offset_beyond_data(self, legislator):
        response = client.get("/legislators/?limit=100&offset=100")
        data = response.json()
        assert len(data["items"]) == 0
        assert data["count"] == 1

    def test_count_with_filters(self, legislator, legislator_d):
        response = client.get("/legislators/?chamber=Senate")
        data = response.json()
        assert data["count"] == 1
        assert len(data["items"]) == 1


# -- Schema Validators -------------------------------------------------------


class TestSchemaValidators:
    """Schema field parsing and validation."""

    def test_split_semicolons_normal(self):
        assert _split_semicolons("a; b; c") == ["a", "b", "c"]

    def test_split_semicolons_empty(self):
        assert _split_semicolons("") == []

    def test_split_semicolons_whitespace_only(self):
        assert _split_semicolons("  ;  ;  ") == []

    def test_split_semicolons_single(self):
        assert _split_semicolons("sen_masterson_ty_1") == ["sen_masterson_ty_1"]

    def test_empty_sponsor_slugs(self, session):
        rc = RollCall.objects.create(
            session=session,
            bill_number="HB 1",
            vote_id="je_20250101000000",
            chamber="House",
            sponsor_slugs="",
            yea_count=0,
            nay_count=0,
            total_votes=0,
        )
        response = client.get(f"/rollcalls/{rc.id}/")
        assert response.json()["sponsor_slugs"] == []

    def test_nullable_passed(self, session):
        rc = RollCall.objects.create(
            session=session,
            bill_number="HB 2",
            vote_id="je_20250101000001",
            chamber="House",
            passed=None,
            yea_count=0,
            nay_count=0,
            total_votes=0,
        )
        response = client.get(f"/rollcalls/{rc.id}/")
        assert response.json()["passed"] is None

    def test_empty_committee_names(self, session):
        ba = BillAction.objects.create(
            session=session,
            bill_number="HB 1",
            action_code="INTRO",
            chamber="House",
            committee_names="",
        )
        response = client.get("/bill-actions/")
        item = next(i for i in response.json()["items"] if i["id"] == ba.id)
        assert item["committee_names"] == []
