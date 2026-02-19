"""CSV output for scraped vote data."""

import csv
from dataclasses import asdict
from pathlib import Path

from ks_vote_scraper.models import IndividualVote, RollCall


def save_csvs(
    output_dir: Path,
    output_name: str,
    individual_votes: list[IndividualVote],
    rollcalls: list[RollCall],
    legislators: dict[str, dict],
) -> None:
    """Save all collected data to CSV files."""
    print("\n" + "=" * 60)
    print("Saving CSV files...")
    print("=" * 60)

    # Individual votes
    votes_file = output_dir / f"ks_{output_name}_votes.csv"
    with open(votes_file, "w", newline="", encoding="utf-8") as f:
        if individual_votes:
            writer = csv.DictWriter(f, fieldnames=list(asdict(individual_votes[0]).keys()))
            writer.writeheader()
            for iv in individual_votes:
                writer.writerow(asdict(iv))
    print(f"  {votes_file} ({len(individual_votes)} rows)")

    # Roll call summaries
    rollcalls_file = output_dir / f"ks_{output_name}_rollcalls.csv"
    with open(rollcalls_file, "w", newline="", encoding="utf-8") as f:
        if rollcalls:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rollcalls[0]).keys()))
            writer.writeheader()
            for rc in rollcalls:
                writer.writerow(asdict(rc))
    print(f"  {rollcalls_file} ({len(rollcalls)} rows)")

    # Legislators
    legislators_file = output_dir / f"ks_{output_name}_legislators.csv"
    with open(legislators_file, "w", newline="", encoding="utf-8") as f:
        fields = ["name", "full_name", "slug", "chamber", "party", "district", "member_url"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for slug in sorted(legislators.keys()):
            row = {k: legislators[slug].get(k, "") for k in fields}
            writer.writerow(row)
    print(f"  {legislators_file} ({len(legislators)} rows)")
