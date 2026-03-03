"""Command-line interface for KanFocus vote data retrieval."""

import argparse

from tallgrass.kanfocus.fetcher import DEFAULT_DELAY, KanFocusFetcher
from tallgrass.kanfocus.output import convert_to_standard, merge_gap_fill, save_full
from tallgrass.kanfocus.session import session_id_for_biennium
from tallgrass.kanfocus.slugs import load_existing_slugs
from tallgrass.session import KSSession


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tallgrass-kanfocus",
        description="Scrape roll call vote data from KanFocus (kanfocus.com). "
        "Produces the same CSV format as the main tallgrass scraper.",
    )
    parser.add_argument(
        "year",
        nargs="?",
        type=int,
        help="Biennium start year (odd), e.g. 1999, 2009, 2011",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "gap-fill"],
        default="full",
        help="full: scrape all votes (default). gap-fill: fill missing votes in existing CSVs.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between requests (default: {DEFAULT_DELAY}). "
        "KanFocus is a shared paid service — be conservative.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached pages before running.",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="Show KanFocus session ID mapping and exit.",
    )

    args = parser.parse_args(argv)

    if args.list_sessions:
        _print_sessions()
        return

    if args.year is None:
        parser.error("year is required (unless using --list-sessions)")

    session = KSSession.from_year(args.year)
    cache_dir = session.data_dir / ".cache" / "kanfocus"
    fetcher = KanFocusFetcher(cache_dir=cache_dir, delay=args.delay)

    if args.clear_cache:
        fetcher.clear_cache()

    print("=" * 60)
    print(f"KanFocus Vote Scraper: {session.label}")
    print(f"Mode: {args.mode}")
    print(f"Delay: {args.delay}s between requests")
    print("=" * 60)

    # Load existing slugs for cross-referencing
    existing_slugs = load_existing_slugs(session.data_dir, session.output_name)
    if existing_slugs:
        print(f"\n  Loaded {len(existing_slugs)} existing slugs for cross-reference")

    # Fetch votes
    print("\nStep 1: Fetching votes from KanFocus...")
    records = fetcher.fetch_biennium(session.start_year)

    if not records:
        print("\nNo votes found. Exiting.")
        return

    # Convert to standard format
    print("\nStep 2: Converting to standard tallgrass format...")
    votes, rollcalls, legislators = convert_to_standard(records, session.label, existing_slugs)
    print(
        f"  {len(votes)} individual votes, {len(rollcalls)} rollcalls, "
        f"{len(legislators)} legislators"
    )

    # Save
    print("\nStep 3: Saving CSV files...")
    if args.mode == "gap-fill":
        merge_gap_fill(session.data_dir, session.output_name, votes, rollcalls, legislators)
    else:
        save_full(session.data_dir, session.output_name, votes, rollcalls, legislators)

    print(f"\nDone: {len(rollcalls)} rollcalls saved to {session.data_dir}")


def _print_sessions() -> None:
    """Print KanFocus session ID mapping table."""
    print("KanFocus Session ID Mapping:")
    print()
    print(f"  {'Session ID':<12} {'Legislature':<14} {'Years'}")
    print(f"  {'─' * 12} {'─' * 14} {'─' * 11}")
    for start in range(1999, 2027, 2):
        session = KSSession.from_year(start)
        sid = session_id_for_biennium(start)
        print(f"  {sid:<12} {session.legislature_name:<14} {start}-{start + 1}")
