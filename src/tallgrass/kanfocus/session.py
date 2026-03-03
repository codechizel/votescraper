"""KanFocus session mapping and URL construction.

KanFocus uses sequential session IDs starting at 106 for the 78th legislature
(1999-2000). The vote tally URL embeds the session ID and a composite vote
identifier: ``{vote_num}{year}{chamber}``.
"""

KANFOCUS_BASE_URL = "https://kanfocus.com"

# Session ID 106 = 78th (1999-2000), 107 = 79th (2001-2002), ..., 119 = 91st (2025-2026)
_SESSION_ID_OFFSET = 106
_SESSION_ID_BASE_YEAR = 1999


def session_id_for_biennium(start_year: int) -> int:
    """Map a biennium start year to a KanFocus session ID.

    >>> session_id_for_biennium(1999)
    106
    >>> session_id_for_biennium(2025)
    119
    """
    if start_year < _SESSION_ID_BASE_YEAR or start_year % 2 != 1:
        msg = f"start_year must be an odd year >= {_SESSION_ID_BASE_YEAR}, got {start_year}"
        raise ValueError(msg)
    return (start_year - _SESSION_ID_BASE_YEAR) // 2 + _SESSION_ID_OFFSET


def vote_tally_url(session_id: int, vote_num: int, year: int, chamber: str) -> str:
    """Construct a KanFocus vote tally page URL.

    >>> vote_tally_url(112, 33, 2011, "S")
    'https://kanfocus.com/Tally_House_Alpha_112.shtml?&Unique_VoteID=332011S'
    """
    return (
        f"{KANFOCUS_BASE_URL}/Tally_House_Alpha_{session_id}.shtml"
        f"?&Unique_VoteID={vote_num}{year}{chamber}"
    )


def generate_vote_id(vote_num: int, year: int, chamber: str) -> str:
    """Generate a deterministic vote_id for KanFocus-sourced data.

    Uses ``kf_`` prefix to distinguish from ``je_`` timestamp IDs in the
    main scraper.

    >>> generate_vote_id(33, 2011, "S")
    'kf_33_2011_S'
    """
    return f"kf_{vote_num}_{year}_{chamber}"


def biennium_streams(start_year: int) -> list[tuple[int, str]]:
    """Return the 4 (year, chamber) streams for a biennium.

    Each biennium has House + Senate votes in both the odd and even years.

    >>> biennium_streams(2011)
    [(2011, 'H'), (2011, 'S'), (2012, 'H'), (2012, 'S')]
    """
    end_year = start_year + 1
    return [
        (start_year, "H"),
        (start_year, "S"),
        (end_year, "H"),
        (end_year, "S"),
    ]
