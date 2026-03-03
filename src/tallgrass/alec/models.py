"""Data models for ALEC model legislation corpus."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ALECModelBill:
    """An ALEC model bill or resolution.

    Scraped from alec.org/model-policy/.  Joins to Kansas bill texts
    on embedding similarity (no natural key exists).
    """

    title: str  # model bill title
    text: str  # full text extracted from HTML
    category: str  # ALEC issue category (e.g., "Criminal Justice")
    bill_type: str  # "Model Policy", "Model Resolution", etc.
    date: str  # date finalized (YYYY-MM-DD or empty)
    url: str  # source URL for provenance
    task_force: str  # ALEC task force name
