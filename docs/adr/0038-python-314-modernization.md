# ADR-0038: Python 3.14 Modernization

**Date:** 2026-02-25
**Status:** Accepted

## Context

The project requires Python >=3.14 (`pyproject.toml`), but several legacy patterns remained:

1. **`from __future__ import annotations`** in all 38 `analysis/` files — a PEP 563 shim for
   deferred annotation evaluation. Python 3.14 implements PEP 649/749 natively, making this
   import unnecessary. It will be deprecated after Python 3.13 reaches EOL (~2029).

2. **`typing.TextIO` with module prefix** in `run_context.py` — used `import typing` then
   `typing.TextIO` instead of `from typing import TextIO`, inconsistent with the codebase's
   direct-import style.

3. **Missing return type hints** on three private methods in `scraper.py` — `_bill_sort_key`,
   `_parse_vote_page`, and `_print_failure_summary` lacked annotations despite the project
   convention of type hints on all function signatures.

4. **Unnecessary `f` prefix** on string segments with no interpolation in `scraper.py` and
   `run_context.py` — harmless but sloppy in a codebase that enforces ruff E/F/I/W.

5. **Version mismatch** — `pyproject.toml` and `__init__.py` used `"0.2.0"` (semver) while
   commit tags use CalVer `[vYYYY.MM.DD.N]`. Synced to `"2026.2.25"` (PEP 440 CalVer).

## Decision

- Remove `from __future__ import annotations` from all 38 analysis files.
- Replace `typing.TextIO` with direct `from typing import TextIO`.
- Add return type annotations to the three untyped methods.
- Remove unnecessary `f` prefix from non-interpolating string segments.
- Adopt CalVer (`YYYY.M.DD`) in `pyproject.toml` and `__init__.py`.

## Consequences

- **38 files touched** for the `__future__` removal — large diff but mechanical (one-line
  deletion per file). ruff reformatted all 38 for import ordering.
- No behavioral changes — PEP 649 deferred evaluation is the default in 3.14, so removing
  the `__future__` import changes nothing at runtime.
- CalVer in `pyproject.toml` means the package version is now human-readable as a date.
  The `N` counter from commit tags is not included (package version tracks the date, not
  individual commits within a day).
- Test count updated from 1,096 to 1,125 across docs (stale count predated recent additions).

**Update (2026-03-02):** 12 remaining `from __future__ import annotations` imports removed
during Python 3.14.3 upgrade (these had accumulated in files added after the original cleanup).
Also removed all 15 `# fmt: skip` workarounds on multi-exception `except` lines — PEP 758
makes bracketless `except A, B:` valid Python 3.14 syntax, eliminating the ruff format bug.
