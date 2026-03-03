# ADR-0090: Django Project Scaffolding + PostgreSQL (DB1)

**Date:** 2026-03-03
**Status:** Accepted

## Context

Tallgrass stores all data as CSVs in `data/kansas/`. This works well for single-state analysis but becomes unwieldy for multi-state scaling and web-based access. The data storage deep dive (`docs/data-storage-deep-dive.md`) recommended PostgreSQL + Django as the database backend.

DB1 is the first phase of the database roadmap: scaffolding the Django project, defining models that mirror the CSV schema, configuring Django admin for browsable data access, and setting up Docker Compose for local PostgreSQL.

## Decision

### Architecture

- Django project lives at `src/web/` alongside the existing `src/tallgrass/` scraper package. They share no imports — the scraper continues writing CSVs, and a future loader (DB2) will import them into PostgreSQL.
- `src/web/tallgrass_web/` is the Django project (settings, URLs, WSGI/ASGI). `src/web/legislature/` is the single Django app containing all models.
- Settings split: `base.py` (shared), `local.py` (DEBUG), `test.py` (fast hashing). No production settings yet.

### Models (8 total)

| Model | Maps to | Key design choices |
|-------|---------|-------------------|
| `State` | reference table | CHAR(2) primary key |
| `Session` | reference table | UNIQUE(state, start_year, is_special) |
| `Legislator` | `_legislators.csv` | FK→Session, UNIQUE(session, slug), empty party default |
| `RollCall` | `_rollcalls.csv` | FK→Session, UNIQUE(session, vote_id), nullable passed |
| `Vote` | `_votes.csv` | FK→RollCall + FK→Legislator, UNIQUE(rollcall, legislator) |
| `BillAction` | `_bill_actions.csv` | FK→Session, nullable datetimes |
| `BillText` | `_bill_texts.csv` | FK→Session |
| `ALECModelBill` | `alec_model_bills.csv` | No FK (standalone corpus) |

Semicolon-joined fields (`sponsor_slugs`, `committee_names`) stay as TextField to preserve CSV round-trip fidelity — no M2M normalization.

### Dependencies

Django and psycopg are a separate `web` dependency group, not a core dependency. Scraper users don't need them. `pytest-django` is added to the dev group.

### Django admin

All 8 models registered with `list_display`, `list_filter`, `search_fields`. `Vote` uses `raw_id_fields` for FK fields (dropdowns unusable at 632K rows).

### Testing

Django tests are isolated from the existing 2458 tests:
- `DJANGO_SETTINGS_MODULE` is NOT set in `pyproject.toml` — existing tests never import Django
- Django test files use `pytest.importorskip("django")` at module top
- New `web` pytest marker; `just test-web` runs only Django tests
- `just test` continues running all non-web tests unaffected

### Infrastructure

- Docker Compose: PostgreSQL 16 only (Django runs locally via uv)
- `just db-up/down/migrate/admin/shell` recipes
- Ruff excludes auto-generated migration files
- ty treats Django imports as any (incomplete stubs)

## Consequences

**Good:**
- Database schema matches CSV schema exactly — CSV→DB loader (DB2) will be straightforward
- Admin provides browsable data access immediately after loading
- Zero impact on existing scraper or analysis pipeline
- Test isolation preserves existing 2458-test suite

**Bad:**
- Django project adds ~550 lines of new code + auto-generated migration
- Requires Docker/OrbStack for local PostgreSQL
- No data in the database yet (DB2 will add the loader)

**Neutral:**
- `src/web/` is a separate Python path (`PYTHONPATH=src/web`) — Justfile recipes handle this
- ty typechecks only `src/tallgrass/` and `analysis/`, not `src/web/`
