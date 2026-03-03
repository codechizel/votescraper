"""Command-line interface for ALEC model legislation scraping."""

import argparse
from pathlib import Path

from tallgrass.alec.output import save_alec_bills
from tallgrass.alec.scraper import scrape_alec_corpus

# Default output location for ALEC corpus
ALEC_DATA_DIR = Path("data/external/alec")
ALEC_CACHE_DIR = ALEC_DATA_DIR / ".cache"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tallgrass-alec",
        description="Scrape ALEC model legislation from alec.org/model-policy/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ALEC_DATA_DIR,
        help=f"Output directory for CSV (default: {ALEC_DATA_DIR})",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached HTML before running",
    )

    args = parser.parse_args(argv)
    cache_dir = args.output_dir / ".cache"

    if args.clear_cache and cache_dir.exists():
        for f in cache_dir.iterdir():
            if f.is_file():
                f.unlink()
        print(f"  Cleared cache: {cache_dir}")

    print("=" * 60)
    print("ALEC Model Legislation Scraper")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Run scraping pipeline
    bills = scrape_alec_corpus(cache_dir)

    if not bills:
        print("\nNo bills extracted. Exiting.")
        return

    # Save CSV
    print("\nStep 3: Saving CSV...")
    save_alec_bills(args.output_dir, bills)

    print(f"\nDone: {len(bills)} ALEC model bills saved to {args.output_dir}")
