"""Django admin configuration for all legislature models."""

from django.contrib import admin

from .models import ALECModelBill, BillAction, BillText, Legislator, RollCall, Session, State, Vote


@admin.register(State)
class StateAdmin(admin.ModelAdmin):
    list_display = ["code", "name"]
    search_fields = ["code", "name"]


@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    list_display = ["name", "state", "start_year", "end_year", "is_special", "legislature_number"]
    list_filter = ["state", "is_special"]
    search_fields = ["name"]


@admin.register(Legislator)
class LegislatorAdmin(admin.ModelAdmin):
    list_display = ["name", "legislator_slug", "chamber", "party", "district", "session"]
    list_filter = ["chamber", "party", "session"]
    search_fields = ["name", "legislator_slug"]


@admin.register(RollCall)
class RollCallAdmin(admin.ModelAdmin):
    list_display = [
        "vote_id",
        "bill_number",
        "chamber",
        "vote_date",
        "passed",
        "yea_count",
        "nay_count",
        "session",
    ]
    list_filter = ["chamber", "passed", "session"]
    search_fields = ["bill_number", "vote_id", "bill_title"]


@admin.register(Vote)
class VoteAdmin(admin.ModelAdmin):
    list_display = ["get_legislator_slug", "vote", "get_vote_id"]
    list_filter = ["vote"]
    raw_id_fields = ["rollcall", "legislator"]

    @admin.display(description="Legislator", ordering="legislator__legislator_slug")
    def get_legislator_slug(self, obj):
        return obj.legislator.legislator_slug

    @admin.display(description="Roll Call", ordering="rollcall__vote_id")
    def get_vote_id(self, obj):
        return obj.rollcall.vote_id


@admin.register(BillAction)
class BillActionAdmin(admin.ModelAdmin):
    list_display = ["bill_number", "action_code", "chamber", "session_date", "status", "session"]
    list_filter = ["chamber", "session"]
    search_fields = ["bill_number", "action_code"]


@admin.register(BillText)
class BillTextAdmin(admin.ModelAdmin):
    list_display = ["bill_number", "document_type", "version", "page_count", "session"]
    list_filter = ["document_type", "session"]
    search_fields = ["bill_number"]


@admin.register(ALECModelBill)
class ALECModelBillAdmin(admin.ModelAdmin):
    list_display = ["title", "category", "bill_type", "task_force", "date"]
    list_filter = ["category", "bill_type", "task_force"]
    search_fields = ["title", "text"]
