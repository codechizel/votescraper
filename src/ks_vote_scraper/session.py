"""Kansas Legislature session/biennium URL resolution.

The KS Legislature website uses different URL prefixes for the current session
vs historical sessions:

  Current (2025-26):  /li/b2025_26/measures/bills/
  Historical (2023-24): /li_2024/b2023_24/measures/bills/
  Special (2024):     /li_2024s/...

This module encapsulates that logic so the scraper can target any session.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# Update this when a new biennium becomes the "current" session on kslegislature.gov.
# The 2027-28 session will start in January 2027.
CURRENT_BIENNIUM_START = 2025

# Known special session years
SPECIAL_SESSION_YEARS = [2024, 2021, 2020, 2016, 2013]


def _ordinal(n: int) -> str:
    """Return the ordinal string for an integer (1st, 2nd, 3rd, 4th, ..., 91st)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


@dataclass(frozen=True)
class KSSession:
    """Represents a Kansas Legislature biennium session and its URL patterns."""

    start_year: int
    special: bool = False

    @property
    def end_year(self) -> int:
        return self.start_year + 1

    @property
    def biennium_code(self) -> str:
        """e.g., 'b2025_26'"""
        return f"b{self.start_year}_{self.end_year % 100:02d}"

    @property
    def is_current(self) -> bool:
        return self.start_year == CURRENT_BIENNIUM_START and not self.special

    @property
    def legislature_number(self) -> int:
        """Kansas Legislature number (e.g., 91 for 2025-2026)."""
        return (self.start_year - 1879) // 2 + 18

    @property
    def legislature_name(self) -> str:
        """Ordinal legislature name (e.g., '91st')."""
        return _ordinal(self.legislature_number)

    @property
    def li_prefix(self) -> str:
        """The /li.../ path prefix for this session."""
        if self.special:
            return f"/li_{self.start_year}s"
        elif self.is_current:
            return f"/li/{self.biennium_code}"
        else:
            return f"/li_{self.end_year}/{self.biennium_code}"

    @property
    def bills_path(self) -> str:
        return f"{self.li_prefix}/measures/bills/"

    @property
    def senate_bills_path(self) -> str:
        return f"{self.li_prefix}/measures/bills/senate/"

    @property
    def house_bills_path(self) -> str:
        return f"{self.li_prefix}/measures/bills/house/"

    @property
    def label(self) -> str:
        """Human-readable label, e.g., '91st (2025-2026)' or '2024 Special'"""
        if self.special:
            return f"{self.start_year} Special"
        return f"{self.legislature_name} ({self.start_year}-{self.end_year})"

    @property
    def output_name(self) -> str:
        """Filesystem-safe name for output dirs/files, e.g., '91st_2025-2026' or '2024s'"""
        if self.special:
            return f"{self.start_year}s"
        return f"{self.legislature_name}_{self.start_year}-{self.end_year}"

    @property
    def bill_url_pattern(self) -> re.Pattern:
        """Compiled regex to match bill URLs within this session's paths."""
        escaped = re.escape(self.li_prefix)
        return re.compile(rf"{escaped}/measures/(sb|hb|scr|hcr|sr|hr)\d+/", re.I)

    @property
    def api_path(self) -> str:
        """API base path for this session."""
        if self.special:
            return f"/li_{self.start_year}s/api/v13/rev-1"
        elif self.is_current:
            return "/li/api/v13/rev-1"
        else:
            return f"/li_{self.end_year}/api/v13/rev-1"

    @classmethod
    def from_year(cls, year: int, special: bool = False) -> "KSSession":
        """Create a session from any year in the biennium.

        Accepts either the start or end year: 2025 and 2026 both give 2025-26.
        For special sessions, the year is used as-is.
        """
        if special:
            return cls(start_year=year, special=True)
        # Normalize: odd years are start years, even years are end years
        if year % 2 == 0:
            year = year - 1
        return cls(start_year=year)

    @classmethod
    def from_session_string(cls, session: str) -> "KSSession":
        """Create a session from a CLI-style session string like '2025-26' or '2025_26'.

        Special sessions (e.g., '2024s') are NOT handled here — use from_year() with
        special=True for those.
        """
        normalized = session.replace("_", "-")
        parts = normalized.split("-")
        return cls.from_year(int(parts[0]))

    @staticmethod
    def data_dir_for_session(session: str, special: bool = False) -> Path:
        """Convert a CLI-style session string to the data directory Path.

        Examples:
            "2025-26" -> Path("data/91st_2025-2026")
            "2023-24" -> Path("data/90th_2023-2024")
        """
        normalized = session.replace("_", "-")
        parts = normalized.split("-")
        ks = KSSession.from_year(int(parts[0]), special=special)
        return Path("data") / ks.output_name
