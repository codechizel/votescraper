# ADR-0113: Legislature number session resolution

**Date:** 2026-03-11
**Status:** Accepted

## Context

Session strings throughout the codebase (`from_session_string()`, `_normalize_session()`, `data_dir_for_session()`) only recognized 4-digit years (e.g., "2001-02", "2025") and special session suffixes ("2024s"). Passing a Kansas Legislature number like "79" caused silent misresolution:

- `from_session_string("79")` split on "-" and called `from_year(79)`, treating 79 as a calendar year (79 AD) instead of the 79th Legislature (2001-2002).
- `_normalize_session("79")` fell through all regex branches (only `^\d{4}$` matched bare numbers) and returned "79" unchanged, creating `results/kansas/79/` instead of `results/kansas/79th_2001-2002/`.
- `data_dir_for_session("79")` had the same fallthrough bug.

This produced orphan directories (`results/kansas/79/79-260311.1/`) with correctly structured but misplaced pipeline output.

## Decision

1. **Add `KSSession.from_legislature_number(n)`** — inverse of `legislature_number` property: `start_year = (n - 18) * 2 + 1879`. Clean classmethod, tested with roundtrip property.

2. **Update `from_session_string()`** — after splitting on "-", check if the parsed number is < 1000 (legislature numbers range 18-999; calendar years are 4 digits). Route small numbers to `from_legislature_number()`.

3. **Update `_normalize_session()`** — add a `^\d{1,3}$` regex branch after the existing `^\d{4}$` branch, converting legislature numbers to full biennium names via `from_legislature_number()`.

4. **Simplify `data_dir_for_session()`** — delegate to `from_session_string()` for all non-special-session inputs, eliminating duplicated parsing logic.

The < 1000 threshold is safe because:
- Kansas Legislature numbers currently range from 18 (1879) to 91 (2025-2026).
- No calendar year < 1000 is a valid biennium start year for the Kansas Legislature.

## Consequences

- `just pipeline 79`, `just eda --session 79`, and all analysis phases now correctly resolve legislature numbers.
- 11 new tests: `TestFromLegislatureNumber` (3), `TestFromSessionString` legislature number cases (3), `TestDataDirForSession.test_legislature_number` (1), `TestNormalizeSession.test_legislature_number_resolves_to_biennium` (1), plus roundtrip (1) and historical coverage (3).
- No breaking changes — all existing session string formats continue to work identically.
