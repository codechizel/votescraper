"""Django models mapping to the Tallgrass CSV schema.

Eight models covering the 5 CSV types (votes, rollcalls, legislators,
bill_actions, bill_texts) plus 2 reference tables (State, Session) and
1 external corpus (ALECModelBill).

Field names and types match the frozen dataclasses in src/tallgrass/models.py
and src/tallgrass/text/models.py.  Semicolon-joined fields (sponsor_slugs,
committee_names) stay as text to preserve CSV round-trip fidelity.
"""

from django.db import models


class State(models.Model):
    """US state — primary key is the 2-letter code (e.g. "KS")."""

    code = models.CharField(max_length=2, primary_key=True)
    name = models.CharField(max_length=100)

    class Meta:
        ordering = ["code"]

    def __str__(self):
        return self.code


class Session(models.Model):
    """A legislative session (biennium or special)."""

    state = models.ForeignKey(State, on_delete=models.CASCADE, related_name="sessions")
    start_year = models.PositiveSmallIntegerField()
    end_year = models.PositiveSmallIntegerField()
    is_special = models.BooleanField(default=False)
    legislature_number = models.PositiveSmallIntegerField(
        help_text="e.g. 91 for the 91st Legislature"
    )
    name = models.CharField(
        max_length=50,
        help_text='Display name, e.g. "91st_2025-2026"',
    )

    class Meta:
        ordering = ["state", "start_year"]
        constraints = [
            models.UniqueConstraint(
                fields=["state", "start_year", "is_special"],
                name="unique_session",
            ),
        ]

    def __str__(self):
        suffix = "s" if self.is_special else ""
        return f"{self.state_id} {self.name}{suffix}"


class Legislator(models.Model):
    """A legislator within a session — maps to _legislators.csv."""

    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="legislators")
    name = models.CharField(max_length=200)
    full_name = models.CharField(max_length=200, default="")
    slug = models.CharField(max_length=200)
    chamber = models.CharField(max_length=10)  # "Senate" or "House"
    party = models.CharField(max_length=50, default="")  # empty = Independent at analysis time
    district = models.CharField(max_length=20, default="")
    member_url = models.URLField(max_length=500, default="")
    ocd_id = models.CharField(max_length=100, default="")  # OpenStates OCD person ID

    class Meta:
        ordering = ["session", "slug"]
        constraints = [
            models.UniqueConstraint(
                fields=["session", "slug"],
                name="unique_legislator_slug",
            ),
        ]

    def __str__(self):
        return f"{self.slug} ({self.session})"


class RollCall(models.Model):
    """A single roll call vote — maps to _rollcalls.csv."""

    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="rollcalls")
    bill_number = models.CharField(max_length=50)
    bill_title = models.TextField(default="")
    vote_id = models.CharField(max_length=100)  # je_YYYYMMDDHHMMSS or kf_N_YYYY_C
    vote_url = models.URLField(max_length=500, default="")
    vote_datetime = models.DateTimeField(null=True, blank=True)
    vote_date = models.DateField(null=True, blank=True)
    chamber = models.CharField(max_length=10)
    motion = models.TextField(default="")
    vote_type = models.CharField(max_length=100, default="")
    result = models.TextField(default="")
    short_title = models.TextField(default="")
    sponsor = models.TextField(default="")
    sponsor_slugs = models.TextField(default="")  # semicolon-joined, preserves CSV fidelity
    yea_count = models.PositiveSmallIntegerField(default=0)
    nay_count = models.PositiveSmallIntegerField(default=0)
    present_passing_count = models.PositiveSmallIntegerField(default=0)
    absent_not_voting_count = models.PositiveSmallIntegerField(default=0)
    not_voting_count = models.PositiveSmallIntegerField(default=0)
    total_votes = models.PositiveSmallIntegerField(default=0)
    passed = models.BooleanField(null=True)

    class Meta:
        ordering = ["session", "vote_id"]
        constraints = [
            models.UniqueConstraint(
                fields=["session", "vote_id"],
                name="unique_rollcall_vote_id",
            ),
        ]

    def __str__(self):
        return f"{self.bill_number} — {self.vote_id}"


class Vote(models.Model):
    """One legislator's vote on one roll call — maps to _votes.csv."""

    rollcall = models.ForeignKey(RollCall, on_delete=models.CASCADE, related_name="votes")
    legislator = models.ForeignKey(Legislator, on_delete=models.CASCADE, related_name="votes")
    vote = models.CharField(max_length=50)  # Yea, Nay, Present and Passing, etc.

    class Meta:
        ordering = ["rollcall", "legislator"]
        constraints = [
            models.UniqueConstraint(
                fields=["rollcall", "legislator"],
                name="unique_vote",
            ),
        ]

    def __str__(self):
        return f"{self.legislator.slug}: {self.vote} on {self.rollcall.vote_id}"


class BillAction(models.Model):
    """One action in a bill's legislative history — maps to _bill_actions.csv."""

    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="bill_actions")
    bill_number = models.CharField(max_length=50)
    action_code = models.CharField(max_length=50)
    chamber = models.CharField(max_length=10)
    committee_names = models.TextField(default="")  # semicolon-joined, preserves CSV fidelity
    occurred_datetime = models.DateTimeField(null=True, blank=True)
    session_date = models.DateField(null=True, blank=True)
    status = models.TextField(default="")
    journal_page_number = models.CharField(max_length=50, default="")

    class Meta:
        ordering = ["session", "bill_number", "occurred_datetime"]

    def __str__(self):
        return f"{self.bill_number} — {self.action_code}"


class BillText(models.Model):
    """Extracted text from a bill document — maps to _bill_texts.csv."""

    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="bill_texts")
    bill_number = models.CharField(max_length=50)
    document_type = models.CharField(max_length=50)  # introduced, enrolled, supp_note, etc.
    version = models.CharField(max_length=50, default="")
    text = models.TextField()
    page_count = models.PositiveSmallIntegerField(default=0)
    source_url = models.URLField(max_length=500, default="")

    class Meta:
        ordering = ["session", "bill_number", "document_type"]

    def __str__(self):
        return f"{self.bill_number} ({self.document_type})"


class ALECModelBill(models.Model):
    """An ALEC model bill or resolution — maps to alec_model_bills.csv.

    Standalone corpus (no session FK).  Joins to Kansas bill texts on
    embedding similarity, not a natural key.
    """

    title = models.TextField()
    text = models.TextField()
    category = models.CharField(max_length=200, default="")
    bill_type = models.TextField(default="")
    date = models.CharField(max_length=20, default="")  # YYYY-MM-DD or empty
    url = models.URLField(max_length=500, default="")
    task_force = models.TextField(default="")

    class Meta:
        ordering = ["title"]

    def __str__(self):
        return self.title
