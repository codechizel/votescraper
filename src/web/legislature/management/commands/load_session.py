"""Load CSVs for one session into PostgreSQL.

Usage:
    just db-load 91st_2025-2026
    just django load_session 91st_2025-2026
    just django load_session 91st_2025-2026 --dry-run
    just django load_session 91st_2025-2026 --skip-bill-text
"""

import io
import re
from datetime import date, datetime
from pathlib import Path

import polars as pl
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction

from legislature.models import (
    BillAction,
    BillText,
    Legislator,
    RollCall,
    Session,
    State,
    Vote,
)

# Default data root relative to the project root (Justfile sets CWD to project root)
DEFAULT_DATA_ROOT = Path("data/kansas")

# Session name patterns
_REGULAR_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)_(\d{4})-(\d{4})$")
_SPECIAL_RE = re.compile(r"^(\d{4})s$")


def parse_session_name(name: str) -> dict:
    """Parse a session directory name into its components.

    Returns dict with keys: legislature_number, start_year, end_year, is_special.

    Examples:
        "91st_2025-2026" → {leg_num: 91, start: 2025, end: 2026}
        "2024s"          → {leg_num: 90, start: 2024, end: 2024, special}
    """
    m = _REGULAR_RE.match(name)
    if m:
        return {
            "legislature_number": int(m.group(1)),
            "start_year": int(m.group(2)),
            "end_year": int(m.group(3)),
            "is_special": False,
        }
    m = _SPECIAL_RE.match(name)
    if m:
        year = int(m.group(1))
        return {
            "legislature_number": (year - 1879) // 2 + 18,
            "start_year": year,
            "end_year": year,
            "is_special": True,
        }
    raise ValueError(f"Cannot parse session name: {name!r}")


def parse_vote_date(s: str) -> date | None:
    """Parse MM/DD/YYYY date string from rollcalls/votes CSV."""
    if not s:
        return None
    return datetime.strptime(s, "%m/%d/%Y").date()


def parse_datetime(s: str) -> datetime | None:
    """Parse ISO datetime string (e.g. '2025-03-20T20:35:13')."""
    if not s:
        return None
    return datetime.fromisoformat(s)


def parse_date(s: str) -> date | None:
    """Parse YYYY-MM-DD date string."""
    if not s:
        return None
    return date.fromisoformat(s)


def parse_bool(s: str) -> bool | None:
    """Parse 'True'/'False'/'' to Python bool or None."""
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def parse_int(s: str) -> int:
    """Parse integer string, returning 0 for empty."""
    if not s:
        return 0
    return int(s)


def _copy_buffer(df: pl.DataFrame, table: str, columns: list[str]) -> int:
    """COPY a Polars DataFrame into a PostgreSQL table via psycopg3 COPY FROM STDIN.

    Returns the number of rows copied.
    """
    buf = io.BytesIO()
    df.select(columns).write_csv(buf, include_header=False, null_value="\\N")
    buf.seek(0)
    with connection.cursor() as cursor:
        with cursor.copy(
            f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')"
        ) as copy:
            copy.write(buf.read())
    return len(df)


class Command(BaseCommand):
    help = "Load CSVs for one session into PostgreSQL"

    def add_arguments(self, parser):
        parser.add_argument(
            "session_name",
            help='Session directory name, e.g. "91st_2025-2026"',
        )
        parser.add_argument(
            "--data-root",
            type=Path,
            default=DEFAULT_DATA_ROOT,
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate CSVs without writing to DB",
        )
        parser.add_argument(
            "--skip-bill-text",
            action="store_true",
            help="Skip loading bill_texts.csv",
        )

    def handle(self, *args, **options):
        session_name = options["session_name"]
        data_root = options["data_root"]
        dry_run = options["dry_run"]
        skip_bill_text = options["skip_bill_text"]

        # Parse session name
        try:
            info = parse_session_name(session_name)
        except ValueError as e:
            raise CommandError(str(e)) from e

        # Resolve data directory
        data_dir = data_root / session_name
        if not data_dir.is_dir():
            raise CommandError(f"Data directory not found: {data_dir}")

        # Discover CSV files
        prefix = session_name
        legislators_csv = data_dir / f"{prefix}_legislators.csv"
        rollcalls_csv = data_dir / f"{prefix}_rollcalls.csv"
        votes_csv = data_dir / f"{prefix}_votes.csv"
        bill_actions_csv = data_dir / f"{prefix}_bill_actions.csv"
        bill_texts_csv = data_dir / f"{prefix}_bill_texts.csv"

        # Validate required CSVs exist
        for path in [legislators_csv, rollcalls_csv, votes_csv]:
            if not path.exists():
                raise CommandError(f"Required CSV not found: {path}")

        # Read CSVs with Polars
        self.stdout.write(f"Reading CSVs from {data_dir}/")
        legs_df = pl.read_csv(legislators_csv, infer_schema_length=0, null_values=[""])
        rcs_df = pl.read_csv(rollcalls_csv, infer_schema_length=0, null_values=[""])
        votes_df = pl.read_csv(votes_csv, infer_schema_length=0, null_values=[""])

        bill_actions_df = None
        if bill_actions_csv.exists():
            bill_actions_df = pl.read_csv(bill_actions_csv, infer_schema_length=0, null_values=[""])

        bill_texts_df = None
        if not skip_bill_text and bill_texts_csv.exists():
            bill_texts_df = pl.read_csv(bill_texts_csv, infer_schema_length=0, null_values=[""])

        # Validate required columns
        self._validate_columns(legs_df, ["name", "legislator_slug", "chamber"], "legislators")
        self._validate_columns(rcs_df, ["bill_number", "vote_id", "chamber"], "rollcalls")
        self._validate_columns(votes_df, ["legislator_slug", "vote_id", "vote"], "votes")

        # Print summary
        self.stdout.write(f"  Legislators:  {len(legs_df):,}")
        self.stdout.write(f"  Roll calls:   {len(rcs_df):,}")
        self.stdout.write(f"  Votes:        {len(votes_df):,}")
        if bill_actions_df is not None:
            self.stdout.write(f"  Bill actions: {len(bill_actions_df):,}")
        else:
            self.stdout.write("  Bill actions: (not found, skipping)")
        if bill_texts_df is not None:
            self.stdout.write(f"  Bill texts:   {len(bill_texts_df):,}")
        elif skip_bill_text:
            self.stdout.write("  Bill texts:   (skipped)")
        else:
            self.stdout.write("  Bill texts:   (not found, skipping)")

        if dry_run:
            self.stdout.write(self.style.SUCCESS("Dry run — no database changes"))
            return

        # Load into database
        with transaction.atomic():
            # Get or create State + Session
            state, _ = State.objects.get_or_create(code="KS", defaults={"name": "Kansas"})
            session, created = Session.objects.get_or_create(
                state=state,
                start_year=info["start_year"],
                is_special=info["is_special"],
                defaults={
                    "end_year": info["end_year"],
                    "legislature_number": info["legislature_number"],
                    "name": session_name,
                },
            )
            if not created:
                # Update fields that might have changed
                session.end_year = info["end_year"]
                session.legislature_number = info["legislature_number"]
                session.name = session_name
                session.save()

            # Delete existing child data for this session
            self.stdout.write(f"Clearing existing data for {session_name}...")
            Vote.objects.filter(rollcall__session=session).delete()
            BillText.objects.filter(session=session).delete()
            BillAction.objects.filter(session=session).delete()
            RollCall.objects.filter(session=session).delete()
            Legislator.objects.filter(session=session).delete()

            # Load legislators via COPY
            leg_copy_df = self._prepare_legislators(legs_df, session.pk)
            leg_count = _copy_buffer(
                leg_copy_df,
                "legislature_legislator",
                [
                    "session_id",
                    "name",
                    "full_name",
                    "legislator_slug",
                    "chamber",
                    "party",
                    "district",
                    "member_url",
                    "ocd_id",
                ],
            )
            self.stdout.write(f"  Loaded {leg_count:,} legislators")

            # Load rollcalls via COPY
            rc_copy_df = self._prepare_rollcalls(rcs_df, session.pk)
            rc_count = _copy_buffer(
                rc_copy_df,
                "legislature_rollcall",
                [
                    "session_id",
                    "bill_number",
                    "bill_title",
                    "vote_id",
                    "vote_url",
                    "vote_datetime",
                    "vote_date",
                    "chamber",
                    "motion",
                    "vote_type",
                    "result",
                    "short_title",
                    "sponsor",
                    "sponsor_slugs",
                    "yea_count",
                    "nay_count",
                    "present_passing_count",
                    "absent_not_voting_count",
                    "not_voting_count",
                    "total_votes",
                    "passed",
                ],
            )
            self.stdout.write(f"  Loaded {rc_count:,} roll calls")

            # Load votes via bulk_create (needs FK resolution)
            vote_count, skipped = self._load_votes(votes_df, session)
            self.stdout.write(f"  Loaded {vote_count:,} votes")
            if skipped:
                self.stderr.write(self.style.WARNING(f"  Skipped {skipped:,} votes (missing FK)"))

            # Load bill actions via COPY
            if bill_actions_df is not None:
                ba_copy_df = self._prepare_bill_actions(bill_actions_df, session.pk)
                ba_count = _copy_buffer(
                    ba_copy_df,
                    "legislature_billaction",
                    [
                        "session_id",
                        "bill_number",
                        "action_code",
                        "chamber",
                        "committee_names",
                        "occurred_datetime",
                        "session_date",
                        "status",
                        "journal_page_number",
                    ],
                )
                self.stdout.write(f"  Loaded {ba_count:,} bill actions")

            # Load bill texts via COPY
            if bill_texts_df is not None:
                bt_copy_df = self._prepare_bill_texts(bill_texts_df, session.pk)
                bt_count = _copy_buffer(
                    bt_copy_df,
                    "legislature_billtext",
                    [
                        "session_id",
                        "bill_number",
                        "document_type",
                        "version",
                        "text",
                        "page_count",
                        "source_url",
                    ],
                )
                self.stdout.write(f"  Loaded {bt_count:,} bill texts")

        self.stdout.write(self.style.SUCCESS(f"Done: {session_name}"))

    def _validate_columns(self, df: pl.DataFrame, required: list[str], name: str):
        """Raise CommandError if required columns are missing."""
        missing = set(required) - set(df.columns)
        if missing:
            raise CommandError(f"{name} CSV missing required columns: {sorted(missing)}")

    def _prepare_legislators(self, df: pl.DataFrame, session_id: int) -> pl.DataFrame:
        """Prepare legislators DataFrame for COPY."""
        return df.select(
            pl.lit(session_id).alias("session_id"),
            pl.col("name").fill_null(""),
            pl.col("full_name").fill_null(""),
            pl.col("legislator_slug"),
            pl.col("chamber").fill_null(""),
            pl.col("party").fill_null(""),
            pl.col("district").fill_null(""),
            pl.col("member_url").fill_null(""),
            pl.col("ocd_id").fill_null(""),
        )

    def _prepare_rollcalls(self, df: pl.DataFrame, session_id: int) -> pl.DataFrame:
        """Prepare rollcalls DataFrame for COPY, converting types."""
        return df.select(
            pl.lit(session_id).alias("session_id"),
            pl.col("bill_number").fill_null(""),
            pl.col("bill_title").fill_null(""),
            pl.col("vote_id"),
            pl.col("vote_url").fill_null(""),
            # vote_datetime: ISO string → keep as-is for PostgreSQL parsing
            pl.col("vote_datetime").fill_null("\\N").alias("vote_datetime"),
            # vote_date: MM/DD/YYYY → YYYY-MM-DD
            pl.col("vote_date")
            .map_elements(
                lambda s: datetime.strptime(s, "%m/%d/%Y").strftime("%Y-%m-%d") if s else "\\N",
                return_dtype=pl.Utf8,
            )
            .alias("vote_date"),
            pl.col("chamber").fill_null(""),
            pl.col("motion").fill_null(""),
            pl.col("vote_type").fill_null(""),
            pl.col("result").fill_null(""),
            pl.col("short_title").fill_null(""),
            pl.col("sponsor").fill_null(""),
            pl.col("sponsor_slugs").fill_null(""),
            # Integer counts
            pl.col("yea_count").fill_null("0"),
            pl.col("nay_count").fill_null("0"),
            pl.col("present_passing_count").fill_null("0"),
            pl.col("absent_not_voting_count").fill_null("0"),
            pl.col("not_voting_count").fill_null("0"),
            pl.col("total_votes").fill_null("0"),
            # passed: True/False/None
            pl.col("passed")
            .map_elements(
                lambda s: "true" if s == "True" else ("false" if s == "False" else "\\N"),
                return_dtype=pl.Utf8,
            )
            .alias("passed"),
        )

    def _load_votes(self, df: pl.DataFrame, session) -> tuple[int, int]:
        """Load votes via bulk_create with FK resolution. Returns (loaded, skipped)."""
        # Build lookup dicts
        slug_to_pk = dict(
            Legislator.objects.filter(session=session).values_list("legislator_slug", "id")
        )
        voteid_to_pk = dict(RollCall.objects.filter(session=session).values_list("vote_id", "id"))

        vote_objects = []
        skipped = 0
        for row in df.select("legislator_slug", "vote_id", "vote").iter_rows():
            slug, vote_id, vote_cat = row
            leg_pk = slug_to_pk.get(slug)
            rc_pk = voteid_to_pk.get(vote_id)
            if leg_pk is None or rc_pk is None:
                skipped += 1
                continue
            vote_objects.append(Vote(rollcall_id=rc_pk, legislator_id=leg_pk, vote=vote_cat or ""))

        Vote.objects.bulk_create(vote_objects, batch_size=5000)
        return len(vote_objects), skipped

    def _prepare_bill_actions(self, df: pl.DataFrame, session_id: int) -> pl.DataFrame:
        """Prepare bill actions DataFrame for COPY."""
        return df.select(
            pl.lit(session_id).alias("session_id"),
            pl.col("bill_number").fill_null(""),
            pl.col("action_code").fill_null(""),
            pl.col("chamber").fill_null(""),
            pl.col("committee_names").fill_null(""),
            pl.col("occurred_datetime").fill_null("\\N").alias("occurred_datetime"),
            pl.col("session_date").fill_null("\\N").alias("session_date"),
            pl.col("status").fill_null(""),
            pl.col("journal_page_number").fill_null(""),
        )

    def _prepare_bill_texts(self, df: pl.DataFrame, session_id: int) -> pl.DataFrame:
        """Prepare bill texts DataFrame for COPY."""
        return df.select(
            pl.lit(session_id).alias("session_id"),
            pl.col("bill_number").fill_null(""),
            pl.col("document_type").fill_null(""),
            pl.col("version").fill_null(""),
            pl.col("text").fill_null(""),
            pl.col("page_count").fill_null("0"),
            pl.col("source_url").fill_null(""),
        )
