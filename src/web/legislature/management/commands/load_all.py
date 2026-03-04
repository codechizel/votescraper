"""Load all session CSVs + ALEC corpus into PostgreSQL.

Usage:
    just db-load-all
    just django load_all
    just django load_all --dry-run
"""

import re
from pathlib import Path

from django.core.management import call_command
from django.core.management.base import BaseCommand

DEFAULT_DATA_ROOT = Path("data/kansas")

# Sort key: regular sessions by start_year, specials after their parent
_YEAR_RE = re.compile(r"(\d{4})")


def _session_sort_key(name: str) -> tuple[int, int]:
    """Sort session directories chronologically: regular by start year, specials after parent."""
    m = _YEAR_RE.search(name)
    year = int(m.group(1)) if m else 0
    is_special = 1 if name.endswith("s") and not name[0].isalpha() else 0
    return (year, is_special)


class Command(BaseCommand):
    help = "Load all session CSVs + ALEC corpus into PostgreSQL"

    def add_arguments(self, parser):
        parser.add_argument(
            "--data-root",
            type=Path,
            default=DEFAULT_DATA_ROOT,
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate CSVs, print counts, no DB writes",
        )
        parser.add_argument(
            "--skip-bill-text",
            action="store_true",
            help="Skip bill_texts.csv for all sessions",
        )

    def handle(self, *args, **options):
        data_root = options["data_root"]
        dry_run = options["dry_run"]
        skip_bill_text = options["skip_bill_text"]

        if not data_root.is_dir():
            self.stderr.write(self.style.ERROR(f"Data root not found: {data_root}"))
            return

        # Discover session directories (skip hidden dirs and .cache)
        session_dirs = sorted(
            [d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=_session_sort_key,
        )

        if not session_dirs:
            self.stderr.write(self.style.WARNING(f"No session directories found in {data_root}"))
            return

        self.stdout.write(f"Found {len(session_dirs)} sessions in {data_root}")

        # Load each session
        for session_name in session_dirs:
            self.stdout.write(f"\n{'=' * 60}")
            self.stdout.write(f"Loading {session_name}...")
            cmd_args = [session_name, "--data-root", str(data_root)]
            if dry_run:
                cmd_args.append("--dry-run")
            if skip_bill_text:
                cmd_args.append("--skip-bill-text")
            call_command("load_session", *cmd_args, stdout=self.stdout, stderr=self.stderr)

        # Load ALEC corpus (skip if CSV not found)
        from legislature.management.commands.load_alec import DEFAULT_ALEC_CSV

        if DEFAULT_ALEC_CSV.exists():
            self.stdout.write(f"\n{'=' * 60}")
            self.stdout.write("Loading ALEC corpus...")
            alec_args = []
            if dry_run:
                alec_args.append("--dry-run")
            call_command("load_alec", *alec_args, stdout=self.stdout, stderr=self.stderr)
        else:
            self.stdout.write(f"\n{'=' * 60}")
            self.stdout.write("ALEC CSV not found, skipping")

        self.stdout.write(self.style.SUCCESS(f"\nAll done: {len(session_dirs)} sessions + ALEC"))
