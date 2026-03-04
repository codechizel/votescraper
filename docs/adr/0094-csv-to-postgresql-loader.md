# ADR-0094: CSV-to-PostgreSQL Loader (DB2)

**Date:** 2026-03-04
**Status:** Accepted

## Context

DB1 (ADR-0090) scaffolded Django with 8 models matching the CSV schema. The database was empty — no data had been loaded. We need management commands to bulk-load existing CSVs into PostgreSQL, making data browsable via Django admin and queryable via SQL.

Scale: 13 biennium directories + 5 special sessions = 18 session directories. ~651K vote rows, ~8K rollcalls, ~2K legislators total. Bill texts are the largest single file (~547K lines for 91st). ALEC corpus is ~1K rows.

## Decision

Three management commands:

1. **`load_session <session_name>`** — loads all CSVs for one session (e.g. `91st_2025-2026` or `2024s`). Strategy: delete-and-reload per session inside `@transaction.atomic`. Options: `--dry-run`, `--skip-bill-text`, `--data-root`.

2. **`load_alec`** — loads the ALEC model legislation corpus (no session FK). Truncate-and-reload.

3. **`load_all`** — discovers all session directories under `data/kansas/`, loads each chronologically, then loads ALEC.

### Loading method per table

| Table | Method | Why |
|-------|--------|-----|
| Legislator | psycopg3 COPY | Small, no FK deps except Session |
| RollCall | psycopg3 COPY | Small, no FK deps except Session |
| Vote | `bulk_create(batch_size=5000)` | Needs slug→PK and vote_id→PK lookups |
| BillAction | psycopg3 COPY | Simple, Session FK only |
| BillText | psycopg3 COPY | Large but simple, Session FK only |

Vote uses `bulk_create` because it needs two FK lookups (legislator slug → Legislator.id, vote_id → RollCall.id) that are only known after parent rows are inserted. Dictionary-based resolution in Python is simple and fast enough for ~95K rows.

### Key design choices

- **Delete-and-reload, not upsert.** Simpler, guaranteed consistency, and the data is always re-derivable from CSVs. Performance is acceptable (full 91st load takes seconds).
- **Polars for CSV reading.** Consistent with the rest of the project (ADR-0002). All columns read as strings (`infer_schema_length=0`) to avoid type inference surprises.
- **Type conversions in Python/Polars.** `vote_date` MM/DD/YYYY → YYYY-MM-DD, `passed` True/False/"" → PostgreSQL boolean/NULL, integer counts with empty → 0.
- **Graceful handling of missing CSVs.** `_votes.csv`, `_rollcalls.csv`, `_legislators.csv` are required; `_bill_actions.csv` and `_bill_texts.csv` are optional (pre-89th sessions lack KLISS data; bill texts only present after `tallgrass-text` runs).
- **Orphan vote warnings.** Votes referencing unknown slugs are skipped with a warning (indicates data inconsistency between CSVs).

### Field widening (post-real-data validation)

Real data loading revealed several `CharField` limits were too small. Changed to `TextField`:

| Model | Field | Was | Max in data | Why |
|-------|-------|-----|-------------|-----|
| `ALECModelBill` | `title` | `CharField(200)` | 340 | Multi-clause resolution titles |
| `ALECModelBill` | `bill_type` | `CharField(200)` | 267 | Composite type descriptions |
| `ALECModelBill` | `task_force` | `CharField(200)` | 275 | Long task force names |
| `RollCall` | `result` | `CharField(100)` | 1,035 | ODT-era full motion text in result |
| `RollCall` | `short_title` | `CharField(500)` | 2,452 | Multi-bill omnibus titles |
| `RollCall` | `sponsor` | `CharField(200)` | 2,619 | Many co-sponsors semicolon-joined |
| `BillAction` | `status` | `CharField(200)` | 410 | Committee appointment text |

Three migrations: `0002` (ALEC fields), `0003` (RollCall fields), `0004` (BillAction status).

## Consequences

- All 13 session directories + ALEC loaded via `just db-load-all`
- 649,786 votes, 7,958 rollcalls, 2,003 legislators, 23,272 bill actions, 1,556 bill texts, 1,057 ALEC bills
- Data is browsable via Django admin at `localhost:8000/admin/`
- Idempotent — safe to re-run after re-scraping
- Unlocks DB3 (post-hook), DB4 (REST API), and DB5 (pipeline integration)
- 2,046 orphan votes skipped across all sessions (special sessions with empty legislator files, older session mismatches) — logged as warnings
