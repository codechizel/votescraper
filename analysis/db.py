"""Database loading for the analysis pipeline.

Django-free module using raw SQL + psycopg3 + Polars ``read_database()``.
PostgreSQL is the default data source; CSV is the fallback when the DB is
unavailable.

Requires ``psycopg[binary]>=3.2`` (in the ``dev`` dependency group).

Usage::

    from analysis.db import load_votes, load_rollcalls, load_legislators

    votes = load_votes(data_dir)                # DB default, CSV fallback
    votes = load_votes(data_dir, use_csv=True)  # force CSV
"""

from __future__ import annotations

import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import polars as pl

# ── Connection Management ────────────────────────────────────────────────────

_DEFAULT_DATABASE_URL = "postgresql://tallgrass:tallgrass@localhost:5432/tallgrass"


def get_connection_uri() -> str:
    """Return the database connection URI from ``DATABASE_URL`` env var."""
    return os.environ.get("DATABASE_URL", _DEFAULT_DATABASE_URL)


@lru_cache(maxsize=1)
def _cached_connection():
    """Return a cached psycopg connection (one per process)."""
    import psycopg

    return psycopg.connect(get_connection_uri(), autocommit=True)


def get_connection():
    """Return a psycopg connection, cached per process.

    Reconnects automatically if the cached connection is closed.
    """
    conn = _cached_connection()
    if conn.closed:
        _cached_connection.cache_clear()
        conn = _cached_connection()
    return conn


def db_available() -> bool:
    """Test whether the database is reachable. Returns False on any failure."""
    try:
        conn = get_connection()
        conn.execute("SELECT 1")
        return True
    except Exception:
        return False


# ── Session Name Mapping ─────────────────────────────────────────────────────


def _session_name_from_data_dir(data_dir: Path) -> str:
    """Convert a data_dir name like ``91st_2025-2026`` to DB session name.

    The DB ``Session.name`` field stores the same format (e.g. ``91st_2025-2026``).
    """
    return data_dir.name


# ── Raw DB Loading Functions ─────────────────────────────────────────────────

_VOTES_SQL = """\
SELECT
    REPLACE(s.name, '_', ' (') || ')' AS session,
    rc.bill_number, rc.bill_title, rc.vote_id,
    TO_CHAR(rc.vote_datetime AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS') AS vote_datetime,
    TO_CHAR(rc.vote_date, 'MM/DD/YYYY') AS vote_date,
    rc.chamber, rc.motion,
    l.name AS legislator_name, l.slug AS legislator_slug,
    v.vote
FROM legislature_vote v
JOIN legislature_rollcall rc ON v.rollcall_id = rc.id
JOIN legislature_legislator l ON v.legislator_id = l.id
JOIN legislature_session s ON rc.session_id = s.id
WHERE s.name = %s
"""

_ROLLCALLS_SQL = """\
SELECT
    REPLACE(s.name, '_', ' (') || ')' AS session,
    rc.bill_number, rc.bill_title, rc.vote_id, rc.vote_url,
    TO_CHAR(rc.vote_datetime AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS') AS vote_datetime,
    TO_CHAR(rc.vote_date, 'MM/DD/YYYY') AS vote_date,
    rc.chamber, rc.motion, rc.vote_type, rc.result,
    rc.short_title, rc.sponsor, rc.sponsor_slugs,
    rc.yea_count, rc.nay_count, rc.present_passing_count,
    rc.absent_not_voting_count, rc.not_voting_count, rc.total_votes,
    rc.passed
FROM legislature_rollcall rc
JOIN legislature_session s ON rc.session_id = s.id
WHERE s.name = %s
"""

_LEGISLATORS_SQL = """\
SELECT
    REPLACE(s.name, '_', ' (') || ')' AS session,
    l.name, l.full_name, l.slug, l.chamber, l.party,
    l.district, l.member_url, l.ocd_id
FROM legislature_legislator l
JOIN legislature_session s ON l.session_id = s.id
WHERE s.name = %s
"""

_BILL_TEXTS_SQL = """\
SELECT
    bt.bill_number, bt.document_type, bt.version, bt.text,
    bt.page_count, bt.source_url
FROM legislature_billtext bt
JOIN legislature_session s ON bt.session_id = s.id
WHERE s.name = %s
"""

_ALEC_SQL = """\
SELECT
    title, text, category, bill_type, date, url, task_force
FROM legislature_alecmodelbill
"""


def load_votes_db(session_name: str) -> pl.DataFrame:
    """Load denormalized votes from PostgreSQL (4-table JOIN)."""
    conn = get_connection()
    return pl.read_database(_VOTES_SQL, conn, execute_options={"parameters": [session_name]})


def load_rollcalls_db(session_name: str) -> pl.DataFrame:
    """Load rollcalls from PostgreSQL."""
    conn = get_connection()
    return pl.read_database(_ROLLCALLS_SQL, conn, execute_options={"parameters": [session_name]})


def load_legislators_db(session_name: str) -> pl.DataFrame:
    """Load legislators from PostgreSQL."""
    conn = get_connection()
    return pl.read_database(_LEGISLATORS_SQL, conn, execute_options={"parameters": [session_name]})


def load_bill_texts_db(session_name: str) -> pl.DataFrame:
    """Load bill texts from PostgreSQL."""
    conn = get_connection()
    return pl.read_database(_BILL_TEXTS_SQL, conn, execute_options={"parameters": [session_name]})


def load_alec_db() -> pl.DataFrame:
    """Load ALEC model legislation corpus from PostgreSQL."""
    conn = get_connection()
    return pl.read_database(_ALEC_SQL, conn)


# ── CSV Loading (Fallback) ──────────────────────────────────────────────────


def _load_votes_csv(data_dir: Path) -> pl.DataFrame:
    """Load votes from CSV."""
    prefix = data_dir.name
    return pl.read_csv(data_dir / f"{prefix}_votes.csv")


def _load_rollcalls_csv(data_dir: Path) -> pl.DataFrame:
    """Load rollcalls from CSV."""
    prefix = data_dir.name
    return pl.read_csv(data_dir / f"{prefix}_rollcalls.csv")


def _load_legislators_csv(data_dir: Path) -> pl.DataFrame:
    """Load legislators from CSV."""
    prefix = data_dir.name
    return pl.read_csv(data_dir / f"{prefix}_legislators.csv")


def _load_bill_texts_csv(data_dir: Path) -> pl.DataFrame:
    """Load bill texts from CSV."""
    prefix = data_dir.name
    csv_path = data_dir / f"{prefix}_bill_texts.csv"
    if not csv_path.exists():
        msg = f"Bill texts CSV not found: {csv_path}. Run `just text` first."
        raise FileNotFoundError(msg)
    return pl.read_csv(csv_path)


def _load_alec_csv(alec_dir: Path) -> pl.DataFrame:
    """Load ALEC model bills from CSV."""
    csv_path = alec_dir / "alec_model_bills.csv"
    if not csv_path.exists():
        msg = f"ALEC corpus not found: {csv_path}. Run `just alec` first."
        raise FileNotFoundError(msg)
    return pl.read_csv(csv_path)


# ── High-Level Routing Functions ─────────────────────────────────────────────


def _try_db(loader, *args) -> pl.DataFrame | None:
    """Attempt a DB load, return None on any failure."""
    try:
        df = loader(*args)
        if df.height > 0:
            return df
    except Exception:
        pass
    return None


def load_votes(data_dir: Path, *, use_csv: bool = False) -> pl.DataFrame:
    """Load votes — DB default, CSV fallback.

    Args:
        data_dir: Path to session data directory (e.g. ``data/kansas/91st_2025-2026``).
        use_csv: Force CSV loading (skip DB attempt).
    """
    if not use_csv:
        session_name = _session_name_from_data_dir(data_dir)
        df = _try_db(load_votes_db, session_name)
        if df is not None:
            return df
        _warn_fallback("votes", data_dir)
    return _load_votes_csv(data_dir)


def load_rollcalls(data_dir: Path, *, use_csv: bool = False) -> pl.DataFrame:
    """Load rollcalls — DB default, CSV fallback."""
    if not use_csv:
        session_name = _session_name_from_data_dir(data_dir)
        df = _try_db(load_rollcalls_db, session_name)
        if df is not None:
            return df
        _warn_fallback("rollcalls", data_dir)
    return _load_rollcalls_csv(data_dir)


def load_legislators(data_dir: Path, *, use_csv: bool = False) -> pl.DataFrame:
    """Load legislators with standard cleaning — DB default, CSV fallback.

    Applies the same cleaning as ``phase_utils._clean_legislators()``:
    strip leadership suffixes, fill party nulls, ensure ocd_id column.
    """
    if not use_csv:
        session_name = _session_name_from_data_dir(data_dir)
        df = _try_db(load_legislators_db, session_name)
        if df is not None:
            return _clean_legislators_df(df)
        _warn_fallback("legislators", data_dir)
    return _clean_legislators_df(_load_legislators_csv(data_dir))


def load_bill_texts(data_dir: Path, *, use_csv: bool = False) -> pl.DataFrame:
    """Load bill texts — DB default, CSV fallback."""
    if not use_csv:
        session_name = _session_name_from_data_dir(data_dir)
        df = _try_db(load_bill_texts_db, session_name)
        if df is not None:
            return df
        _warn_fallback("bill_texts", data_dir)
    return _load_bill_texts_csv(data_dir)


def load_alec(alec_dir: Path, *, use_csv: bool = False) -> pl.DataFrame:
    """Load ALEC corpus — DB default, CSV fallback."""
    if not use_csv:
        df = _try_db(load_alec_db)
        if df is not None:
            return df
        _warn_fallback("alec", alec_dir)
    return _load_alec_csv(alec_dir)


# ── Legislator Cleaning (Shared) ────────────────────────────────────────────


def _clean_legislators_df(df: pl.DataFrame) -> pl.DataFrame:
    """Clean a legislators DataFrame (works for both DB and CSV sources).

    - Strips leadership suffixes from full_name
    - Fills null/empty party to "Independent"
    - Ensures ocd_id column exists and nulls are filled
    """
    from analysis.run_context import strip_leadership_suffix

    if "ocd_id" not in df.columns:
        df = df.with_columns(pl.lit("").alias("ocd_id"))

    return df.with_columns(
        pl.col("full_name")
        .map_elements(strip_leadership_suffix, return_dtype=pl.Utf8)
        .alias("full_name"),
        pl.col("party").fill_null("Independent").replace("", "Independent").alias("party"),
        pl.col("ocd_id").fill_null("").alias("ocd_id"),
    )


# ── Warnings ─────────────────────────────────────────────────────────────────


def _warn_fallback(resource: str, path: Path) -> None:
    """Emit a warning when falling back to CSV."""
    warnings.warn(
        f"DB unavailable for {resource} — falling back to CSV ({path})",
        stacklevel=3,
    )
    print(f"  ⚠ DB unavailable for {resource}, using CSV fallback", file=sys.stderr)
