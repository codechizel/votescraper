"""Command-line interface for the Kansas Legislature vote scraper."""

import argparse
from pathlib import Path

from ks_vote_scraper.config import BASE_URL, REQUEST_DELAY
from ks_vote_scraper.scraper import KSVoteScraper
from ks_vote_scraper.session import CURRENT_BIENNIUM_START, SPECIAL_SESSION_YEARS, KSSession


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ks-vote-scraper",
        description="Scrape roll call votes from the Kansas Legislature website.",
    )
    parser.add_argument(
        "year",
        nargs="?",
        type=int,
        default=CURRENT_BIENNIUM_START,
        help=f"Session start year, e.g. 2025, 2023, 2021 (default: {CURRENT_BIENNIUM_START})",
    )
    parser.add_argument(
        "--special",
        action="store_true",
        help="Scrape a special session (e.g., 2024 special session)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: data/{legislature}_{start}-{end}/)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help=f"Seconds between requests (default: {REQUEST_DELAY})",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip fetching legislator party/district info",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached pages before running",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List known session years and exit",
    )

    args = parser.parse_args(argv)

    if args.list_sessions:
        print("Known Kansas Legislature sessions:")
        print()
        print("  Regular sessions:")
        for start in range(CURRENT_BIENNIUM_START, 2010, -2):
            s = KSSession.from_year(start)
            print(f"    {s.label:22s}  {BASE_URL}{s.bills_path}")
        print()
        print("  Special sessions:")
        for year in SPECIAL_SESSION_YEARS:
            s = KSSession(start_year=year, special=True)
            print(f"    {s.label:22s}  {BASE_URL}{s.li_prefix}/")
        return

    session = KSSession.from_year(args.year, special=args.special)

    scraper = KSVoteScraper(
        session=session,
        output_dir=args.output,
        delay=args.delay,
    )

    if args.clear_cache:
        scraper.clear_cache()

    scraper.run(enrich=not args.no_enrich)
