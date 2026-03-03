"""CSV export for ALEC model legislation corpus."""

import csv
from pathlib import Path

from tallgrass.alec.models import ALECModelBill

# Column order for alec_model_bills.csv
FIELDNAMES = [
    "title",
    "category",
    "bill_type",
    "date",
    "task_force",
    "url",
    "text",
]


def save_alec_bills(
    output_dir: Path,
    bills: list[ALECModelBill],
) -> Path:
    """Save ALEC model bills to CSV.

    Writes ``alec_model_bills.csv`` into ``output_dir``.
    Returns the path to the written CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "alec_model_bills.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for bill in bills:
            writer.writerow(
                {
                    "title": bill.title,
                    "category": bill.category,
                    "bill_type": bill.bill_type,
                    "date": bill.date,
                    "task_force": bill.task_force,
                    "url": bill.url,
                    "text": bill.text,
                }
            )

    print(f"  {csv_path} ({len(bills)} rows)")
    return csv_path
