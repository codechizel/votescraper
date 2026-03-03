"""Command-line interface for Tallgrass."""

import argparse
from pathlib import Path

from tallgrass.config import BASE_URL, REQUEST_DELAY
from tallgrass.scraper import KSVoteScraper
from tallgrass.session import CURRENT_BIENNIUM_START, SPECIAL_SESSION_YEARS, KSSession


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tallgrass",
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
        help="Output directory (default: data/kansas/{legislature}_{start}-{end}/)",
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
    parser.add_argument(
        "--merge-special",
        type=str,
        default=None,
        metavar="YEAR|all",
        help="Merge special session into parent biennium (e.g. 2020, or 'all')",
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

    if args.merge_special is not None:
        _run_merge_special(args.merge_special)
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


def _run_merge_special(arg: str) -> None:
    """Handle --merge-special: merge one or all special sessions."""
    from tallgrass.merge_special import merge_all_specials, merge_special_into_parent

    if arg.lower() == "all":
        results = merge_all_specials()
        for year, stats in sorted(results.items()):
            special = KSSession(start_year=year, special=True)
            parent = special.parent_session
            print(f"  {special.label} -> {parent.label}: {stats}")
        if not results:
            print("  No special sessions found to merge.")
        return

    year = int(arg)
    if year not in SPECIAL_SESSION_YEARS:
        print(f"Error: {year} is not a known special session year.")
        print(f"Known special sessions: {sorted(SPECIAL_SESSION_YEARS)}")
        return

    special = KSSession(start_year=year, special=True)
    parent = special.parent_session
    stats = merge_special_into_parent(year)
    print(f"Merged {special.label} -> {parent.label}:")
    print(f"  Votes added:       {stats['votes_added']}")
    print(f"  Roll calls added:  {stats['rollcalls_added']}")
    print(f"  Legislators added: {stats['legislators_added']}")
