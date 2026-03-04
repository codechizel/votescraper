"""Load ALEC model legislation corpus into PostgreSQL.

Usage:
    just db-load-alec
    just django load_alec
    just django load_alec --dry-run
"""

import io
from pathlib import Path

import polars as pl
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction

from legislature.models import ALECModelBill

DEFAULT_ALEC_CSV = Path("data/external/alec/alec_model_bills.csv")


class Command(BaseCommand):
    help = "Load ALEC model legislation corpus into PostgreSQL"

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            type=Path,
            default=DEFAULT_ALEC_CSV,
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print row count without writing to DB",
        )

    def handle(self, *args, **options):
        csv_path = options["csv"]
        dry_run = options["dry_run"]

        if not csv_path.exists():
            raise CommandError(f"ALEC CSV not found: {csv_path}")

        df = pl.read_csv(csv_path, infer_schema_length=0, null_values=[""])
        self.stdout.write(f"Read {len(df):,} ALEC model bills from {csv_path}")

        # Validate required columns
        required = {"title", "text"}
        missing = required - set(df.columns)
        if missing:
            raise CommandError(f"ALEC CSV missing required columns: {sorted(missing)}")

        if dry_run:
            self.stdout.write(self.style.SUCCESS("Dry run — no database changes"))
            return

        # Prepare DataFrame for COPY
        copy_df = df.select(
            pl.col("title").fill_null(""),
            pl.col("text").fill_null(""),
            pl.col("category").fill_null(""),
            pl.col("bill_type").fill_null(""),
            pl.col("date").fill_null(""),
            pl.col("url").fill_null(""),
            pl.col("task_force").fill_null(""),
        )

        columns = ["title", "text", "category", "bill_type", "date", "url", "task_force"]

        buf = io.BytesIO()
        copy_df.write_csv(buf, include_header=False, null_value="\\N")
        buf.seek(0)

        with transaction.atomic():
            ALECModelBill.objects.all().delete()
            with connection.cursor() as cursor:
                with cursor.copy(
                    f"COPY {ALECModelBill._meta.db_table} ({', '.join(columns)}) "
                    f"FROM STDIN WITH (FORMAT csv, NULL '\\N')"
                ) as copy:
                    copy.write(buf.read())

        count = ALECModelBill.objects.count()
        self.stdout.write(self.style.SUCCESS(f"Done: loaded {count:,} ALEC model bills"))
