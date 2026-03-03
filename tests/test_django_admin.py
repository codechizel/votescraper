"""Tests for Django admin registration and list views.

Run: uv run pytest tests/test_django_admin.py -v
Requires: PostgreSQL running (just db-up)
"""

import os
import sys

import pytest

django = pytest.importorskip("django")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tallgrass_web.settings.test")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "web"))
django.setup()

from django.contrib.admin.sites import site as admin_site  # noqa: E402
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

ALL_MODELS = [State, Session, Legislator, RollCall, Vote, BillAction, BillText, ALECModelBill]


# -- Registration ------------------------------------------------------------


class TestAdminRegistration:
    """All 8 models are registered in the admin site."""

    @pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
    def test_model_registered(self, model):
        assert model in admin_site._registry


class TestAdminConfig:
    """Admin ModelAdmin configuration checks."""

    def test_state_search_fields(self):
        ma = admin_site._registry[State]
        assert "code" in ma.search_fields

    def test_session_list_filter(self):
        ma = admin_site._registry[Session]
        assert "state" in ma.list_filter

    def test_legislator_list_display(self):
        ma = admin_site._registry[Legislator]
        assert "name" in ma.list_display
        assert "chamber" in ma.list_display
        assert "party" in ma.list_display

    def test_rollcall_list_display(self):
        ma = admin_site._registry[RollCall]
        assert "vote_id" in ma.list_display
        assert "bill_number" in ma.list_display
        assert "passed" in ma.list_display

    def test_vote_raw_id_fields(self):
        """Vote uses raw_id_fields — FK dropdowns unusable at 632K rows."""
        ma = admin_site._registry[Vote]
        assert "rollcall" in ma.raw_id_fields
        assert "legislator" in ma.raw_id_fields

    def test_alec_search_fields(self):
        ma = admin_site._registry[ALECModelBill]
        assert "title" in ma.search_fields
        assert "text" in ma.search_fields

    def test_bill_action_list_filter(self):
        ma = admin_site._registry[BillAction]
        assert "chamber" in ma.list_filter

    def test_bill_text_list_filter(self):
        ma = admin_site._registry[BillText]
        assert "document_type" in ma.list_filter
