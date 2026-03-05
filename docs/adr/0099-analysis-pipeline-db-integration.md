# ADR-0099: Analysis Pipeline Database Integration (DB5)

## Status

Accepted (2026-03-04)

## Context

The analysis pipeline (27 phases) reads scraper output from CSV files in `data/kansas/{session}/`. With DB1-DB4 complete (649K votes, 8K rollcalls, 2K legislators, 1.6K bill texts, 1K ALEC model bills loaded in PostgreSQL), we can replace CSV reads with direct database queries. PostgreSQL offers indexed queries (faster than CSV scans), a single source of truth, and preparation for multi-state expansion (DB6).

## Decision

### New module: `analysis/db.py`

Django-free module using raw SQL + psycopg3 + Polars `read_database()`. No Django ORM dependency — the analysis pipeline stays independent of the web project.

**Connection management:**
- `get_connection_uri()` reads `DATABASE_URL` env var (default: `postgresql://tallgrass:tallgrass@localhost:5432/tallgrass`)
- `get_connection()` returns a cached `psycopg.Connection` (one per process, auto-reconnect)
- `db_available()` lightweight connection test for fallback logic

**DB loading functions** (raw SQL, verified column-for-column against CSV output):
- `load_votes_db(session_name)` — 4-table JOIN (Vote → RollCall → Legislator → Session)
- `load_rollcalls_db(session_name)` — RollCall → Session JOIN
- `load_legislators_db(session_name)` — Legislator → Session JOIN
- `load_bill_texts_db(session_name)` — BillText → Session JOIN
- `load_alec_db()` — standalone ALEC corpus (no session filter)

**Routing functions** with CSV fallback:
- `load_votes(data_dir, *, use_csv=False)` — tries DB, falls back to CSV on failure
- `load_rollcalls(data_dir, *, use_csv=False)`
- `load_legislators(data_dir, *, use_csv=False)` — includes standard cleaning
- `load_bill_texts(data_dir, *, use_csv=False)`
- `load_alec(alec_dir, *, use_csv=False)`

### DB is the default; CSV is the fallback

PostgreSQL is the primary data source. When the DB is unavailable (Docker down, no psycopg, connection refused), loading functions emit a warning and fall back to CSV transparently. The `--csv` CLI flag forces CSV-only mode.

### Updated call sites

9 CSV-loading call sites across 8 phase files + `phase_utils.py`:

| File | Functions updated |
|------|-----------------|
| `analysis/phase_utils.py` | `load_metadata()`, `load_legislators()` |
| `analysis/01_eda/eda.py` | `load_data()` |
| `analysis/03_mca/mca.py` | `load_raw_data()` |
| `analysis/05_irt/irt.py` | `load_metadata()` call + raw votes read |
| `analysis/13_indices/indices.py` | `load_raw_data()` |
| `analysis/15_prediction/prediction.py` | `load_vote_data()`, `load_rollcall_data()`, `load_legislator_data()` |
| `analysis/19_tsa/tsa.py` | `load_data()` |
| `analysis/20_bill_text/bill_text_data.py` | `load_bill_texts()`, `load_rollcalls()`, `load_votes()` |
| `analysis/23_model_legislation/model_legislation_data.py` | `load_alec_corpus()` |

### CLI flag

Each phase's `parse_args()` gains `--csv` (store_true). Propagated as `use_csv=args.csv`.

### Shared legislator cleaning

`_clean_legislators_df(df)` in `db.py` applies the same transformations as the old `phase_utils._clean_legislators()`: strip leadership suffixes, fill null/empty party to "Independent", ensure `ocd_id` column. Used by both DB and CSV paths.

## Consequences

- PostgreSQL becomes the default data source for all analysis phases
- No breaking changes — `--csv` flag preserves exact prior behavior
- `psycopg[binary]>=3.2` added to `dev` dependency group (already in `web` group)
- Cross-state texts in Phase 23 (`data/{state}/` directories for MO/OK/NE/CO) stay CSV-only — not in our DB
- Parquet inter-phase reads (`resolve_upstream_dir()` + `pl.read_parquet()`) unaffected
- `bill_actions` CSV not loaded by any analysis phase — no DB function needed
- New tests: unit tests (mocked DB) + integration tests (`@pytest.mark.web`)
