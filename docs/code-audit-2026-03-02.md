# Tallgrass Code Audit (Static Review)

**Date:** 2026-03-02  
**Repo:** `tallgrass`  
**Reference:** `git rev-parse HEAD` → `7811177f1d097bdb30338f317b4ff23c5731de6d`  
**Method:** Static reading + repo-wide searches (no runtime execution, no code changes applied as part of this audit).

---

## Executive summary

Tallgrass is an unusually well-documented and well-tested research codebase. The scraper and analysis pipeline have clear separation of concerns, strong reproducibility primitives (`RunContext`), and a mature “deep dive + ADR” culture that captures hard-won domain lessons.

The main opportunities now are **(a) tightening a few correctness edges where integration code depends on naming conventions**, and **(b) consolidating scattered “glue code” patterns** (path resolution, manifest naming, lightweight data lookups) to reduce drift across phases.

### Highest-priority findings (actionable, likely impact)

1. **Synthesis phase manifest-key mismatch (likely bug / silent wrong numbers).**  
   The synthesis layer appears to read manifests using short keys like `"eda"` / `"indices"` / `"clustering"`, but `load_all_upstream()` stores manifests under phase IDs like `"01_eda"` / `"07_indices"` / `"05_clustering"`. This can silently fall back to defaults and produce incorrect headline numbers in the synthesis report and pipeline summary infographic.

2. **Monitoring callback mismatch with nutpie sampling (feature appears non-functional).**  
   `analysis/experiment_monitor.py` creates a PyMC callback that updates a status JSON every N draws, but hierarchical sampling uses `nutpie.sample()` which doesn’t support PyMC callbacks. Several call sites pass `callback=` and explicitly print “ignored”. This implies `just monitor` may not reflect real progress for nutpie-based runs (unless another mechanism exists elsewhere).

3. **Dead/unused helpers in `analysis/04_irt/irt.py` (maintainability).**  
   Several joint-model helper functions appear unreferenced in the current phase flow (e.g., `plot_joint_vs_chamber` and related utilities). Keeping them in the production phase file increases cognitive load and makes the “supported” surface area unclear.

### Medium-priority findings (quality / robustness)

4. **Scraper cache key risks collisions due to truncation.**  
   The cache filename is derived from the URL and truncated to `CACHE_FILENAME_MAX_LENGTH`. Collisions are unlikely but plausible over long URLs and repeated query patterns. A collision would be a correctness bug (wrong cached payload served).

5. **Analysis phases repeatedly reimplement small utilities.**  
   There is deliberate duplication (self-contained scripts that can run standalone), but some duplication looks unintentional drift (e.g., upstream directory resolution logic exists in multiple forms; leadership suffix stripping has multiple regex implementations).

6. **Special session handling in analysis appears incomplete.**  
   `KSSession.from_session_string()` explicitly doesn’t support `"2024s"`-style labels, but many analysis phases derive `data_dir` by calling it with `--session`. This is fine if analysis is never run on special sessions, but it’s a footgun if the CLI accepts those labels elsewhere.

---

## Scope and what was reviewed

### Project rules & operational constraints
- `CLAUDE.md` and `.claude/rules/*` (analysis framework, scraper architecture, analytic workflow, testing conventions, worktree rules).

### Scraper core (`src/tallgrass/`)
- `scraper.py`, `session.py`, `cli.py`, `config.py`, `models.py`, `output.py`, `odt_parser.py`.

### Analysis shared infrastructure (`analysis/`)
- `analysis/__init__.py` import redirector (PEP 302 meta-path finder)  
- `run_context.py`, `report.py`, `phase_utils.py`, `dashboard.py`  
- `experiment_monitor.py`, `experiment_runner.py`

### Representative phases
- EDA: `analysis/01_eda/eda.py`
- IRT: `analysis/04_irt/irt.py`
- Hierarchical IRT: `analysis/10_hierarchical/hierarchical.py`
- Synthesis integration: `analysis/11_synthesis/synthesis.py`, `analysis/11_synthesis/synthesis_data.py`, `analysis/11_synthesis/synthesis_detect.py`

### Repo-wide searches
- `TODO|FIXME|XXX|HACK` in `*.py` (none found)
- `except:` (none found)
- Targeted searches for likely dead helpers / questionable patterns

---

## Strengths worth preserving

### 1) Clear phase boundaries + reproducible outputs
- The scraper produces stable CSV contracts.
- The analysis pipeline is explicitly phase-ordered, and every phase writes a structured manifest and outputs to a deterministic results layout.

### 2) “Hard-won lessons” are embedded in docs and tests
- The scraper pitfalls section in `CLAUDE.md` is exactly what prevents regressions in brittle HTML parsing.
- The test inventory suggests strong coverage for both scraper edge cases and analysis invariants.

### 3) `RunContext` is a genuine platform primitive
- Capturing stdout, run metadata, primers, symlink handling, and optional download registration makes downstream report generation reliable and consistent.

---

## Correctness & bug risks (details)

### A) Synthesis manifest keys don’t match upstream storage (high priority)

**Evidence**
- `analysis/11_synthesis/synthesis_data.py` stores manifests as:
  - `upstream["manifests"]["01_eda"]`, `["07_indices"]`, `["05_clustering"]`, etc.
- `analysis/11_synthesis/synthesis.py` / `synthesis_report.py` contain accesses like:
  - `manifests.get("eda", {})`
  - `manifests.get("indices", {})`
  - `manifests.get("clustering", {})`

**Why it matters**
- This is a classic “silent fallback to defaults” risk: the code will still run, but headline numbers (roll call counts, optimal k, etc.) may be wrong or defaulted.

**Suggested direction**
- Introduce a single normalization layer in synthesis:
  - e.g., build `manifests_short = {"eda": manifests["01_eda"], "indices": manifests["07_indices"], ...}` once.
- Add a unit test for synthesis that asserts required manifest keys are present and non-empty when upstream exists.

---

### B) Experiment monitoring callback appears incompatible with nutpie sampling (high priority)

**Evidence**
- `analysis/experiment_monitor.py` defines `create_monitoring_callback()` designed for `pm.sample(callback=...)`.
- `analysis/10_hierarchical/hierarchical.py` and `analysis/experiment_runner.py` use `nutpie.sample()`.
- Several functions accept `callback=` but print that it is ignored.

**Why it matters**
- Operationally, teams rely on monitoring to decide whether to kill/restart expensive runs.
- The `just monitor` affordance can become misleading if it’s only valid for older PyMC sampling paths.

**Suggested direction**
- If nutpie provides hooks, implement them; if not:
  - downgrade callback-based monitoring to “legacy PyMC only”
  - or implement coarse monitoring at phase boundaries (compile start/end, sampling start/end, periodic “still alive” heartbeats from Python side).

---

### C) Potential dead/unused helpers in `analysis/04_irt/irt.py` (maintainability)

**Evidence**
- `plot_joint_vs_chamber` appears defined but not referenced elsewhere in repo.
- The file includes several joint-model utilities that are not called in the current “joint model” path (which uses mean/sigma equating, not joint MCMC).

**Why it matters**
- Large phase scripts already have high complexity; leaving unused helpers inside the “mainline” file increases the maintenance surface and makes it harder to know what is production-supported.

**Suggested direction**
- Move unused helpers into:
  - `analysis/experimental/` or a dedicated `analysis/04_irt/joint_experiments.py`
- Keep `irt.py` focused on the supported flow.

---

### D) Special-session analysis path likely fails (medium priority)

**Evidence**
- `KSSession.from_session_string()` does not handle `2024s` and will `int("2024s")` → error.
- Several analysis phases call `KSSession.from_session_string(args.session)` to resolve data/results roots.

**Why it matters**
- The scraper supports special sessions via `--special`, but the analysis CLI uses `--session` strings everywhere. Users can reasonably try `--session 2024s` and hit a confusing failure.

**Suggested direction**
- Decide whether analysis on special sessions is supported:
  - If yes: teach `from_session_string()` about the `s` suffix.
  - If no: fail early with a friendly error message in analysis phases.

---

## Refactoring opportunities (low risk, high leverage)

### 1) Consolidate upstream directory resolution
Today there are at least two patterns:
- `analysis.run_context.resolve_upstream_dir()` (phases)
- `analysis/11_synthesis/synthesis_data._resolve_phase_dir()` (synthesis)

Recommendation: make synthesis reuse the canonical resolver (or provide a shared resolver that covers both “run-directory mode” and “flat mode” consistently).

### 2) Consolidate small repeated parsing utilities
Examples:
- Leadership suffix stripping exists in both `run_context.py` and `phase_utils.py`.
- Vote tally parsing exists in multiple places (scraper + run_context).
- Bill motion parsing exists in multiple forms (scraper vs ODT vs analysis-level parsing).

Recommendation: create a very small `analysis/text_utils.py` / `tallgrass/text_utils.py` and import from both sides where appropriate.

### 3) Replace dict-of-dict legislator records with a typed model
The scraper’s `self.legislators: dict[str, dict]` is pragmatic, but it:
- hides required/optional fields
- makes it easy to create partial records with missing keys
- requires lots of `.get()` calls

A frozen dataclass or TypedDict would improve correctness and readability without changing runtime behavior.

---

## Performance / efficiency opportunities

These are not urgent (dataset sizes are modest), but they’re cheap wins:

### 1) Avoid repeated `polars.filter()` inside loops for label lookups
Example pattern (EDA heatmap labeling): repeatedly filtering `legislators` per slug can become O(n²) in Python overhead.

Recommendation: precompute `slug -> full_name` dict once.

### 2) Anchor selection participation rates can be vectorized
`analysis/04_irt/irt.py::select_anchors()` computes participation by iterating matrix rows and counting non-null values.

Recommendation: use `pl.sum_horizontal()` with `is_not_null()` across columns (similar to how EDA filters low-participation legislators).

### 3) Scraper cache filenames: consider hashing
Even if collisions are rare, hashing is effectively free and removes a correctness class of bugs.

---

## Testing / tooling review

### Strengths
- The test suite inventory is unusually complete for a research platform (scraper parsing edge cases, output schema invariants, report structure tests, pipeline integration tests).
- Markers (`scraper`, `integration`, `slow`) are clearly defined and used.

### Opportunities
1. **Add a synthesis manifest-key test**  
   A single unit test that builds a synthetic `upstream["manifests"]` with phase keys and asserts synthesis doesn’t use missing short keys would prevent silent regressions.

2. **Dependency grouping in `pyproject.toml`**  
   Core analysis dependencies live in `[dependency-groups].dev`. If end users are expected to run the pipeline, consider an `analysis` dependency group or an extra that is explicitly installed.

3. **Type hints in analysis code**  
   `ty` is warnings-only for analysis (reasonable). Still, obvious signature mismatches (e.g., functions returning tuples but annotated as single values) are worth cleaning up because they confuse readers and tooling.

---

## Suggested next steps (prioritized)

1. **Fix synthesis manifest key normalization** (correctness + reporting integrity)  
2. **Clarify/repair experiment monitoring under nutpie** (operability)  
3. **Move dead joint-model helpers out of `analysis/04_irt/irt.py`** (maintainability)  
4. **Harden scraper cache keys with hashing** (robustness)  
5. **Consolidate shared mini-utilities** (reduce drift)  
6. **Decide and document special-session analysis support** (UX / correctness)

---

## Appendix: files referenced most heavily

- `src/tallgrass/scraper.py`, `src/tallgrass/session.py`, `src/tallgrass/odt_parser.py`
- `analysis/run_context.py`, `analysis/report.py`, `analysis/phase_utils.py`
- `analysis/01_eda/eda.py`, `analysis/04_irt/irt.py`, `analysis/10_hierarchical/hierarchical.py`
- `analysis/11_synthesis/synthesis.py`, `analysis/11_synthesis/synthesis_data.py`, `analysis/11_synthesis/synthesis_detect.py`