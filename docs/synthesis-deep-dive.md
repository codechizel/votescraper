# Synthesis Deep Dive

A code audit, ecosystem survey, and fresh-eyes evaluation of the Synthesis phase (Phase 11).

**Date:** 2026-02-25

---

## Executive Summary

The synthesis phase is architecturally sound and — unusually for this kind of work — fills a genuine gap in the open-source landscape. No existing Python project combines IRT ideal points, network centrality, clustering assignments, classical indices, and prediction outputs into a unified per-legislator report with algorithmic detection of notable legislators. The closest academic precedent is Jakulin & Buntine (2005) at Columbia, but that was a one-off analysis, not a reusable pipeline.

The code is well-modularized (orchestrator, detector, report builder), detection logic is pure and testable, and graceful degradation is thorough. This deep dive identifies **seven issues** (two substantive, one code defect, four code quality), **nine test gaps**, and **two refactoring opportunities**. It also surveys the academic landscape for multi-method synthesis in legislative analysis — a survey that confirms the implementation's detection algorithms are novel but well-grounded in the literature.

---

## 1. Field Survey: How Do Researchers Synthesize Multi-Method Legislative Analysis?

### 1.1 There Is No Standard

The political methodology literature has no formal framework for combining IRT, network, clustering, and classical indices into a single deliverable. Researchers use one of four informal patterns:

| Pattern | Description | Examples |
|---------|-------------|----------|
| **Triangulation** | Run methods independently, report where they agree/disagree | McCarty (2010): NOMINATE vs IDEAL correlations r=0.988 |
| **Unified scorecard** | Join metrics per legislator into a flat table | Voteview.com: NOMINATE + party unity per member |
| **Multi-method paper** | Present parallel analyses in prose | Jakulin & Buntine (2005): info theory + clustering + MDS on Senate data |
| **Joint model** | Integrate within the Bayesian model | Goplerud (2024): hierarchical model jointly estimates ideal points + clusters |

Our synthesis follows the first two patterns — triangulation via narrative report, unified scorecard via the joined legislator DataFrame. This is the pragmatic choice. The joint-model approach (Goplerud) is academically more rigorous but requires a fundamentally different pipeline architecture.

### 1.2 No Open-Source Pipeline Exists

An extensive search found **no Python project** that combines IRT + network + clustering + synthesis:

- **Voteview / Rvoteview** — R-only, data access + NOMINATE scoring, no synthesis layer
- **open-source-legislation** (spartypkp) — SQL knowledge graph for bill text, not analysis
- **Jakulin's Data Mining in Politics** — Static Columbia scripts, not reusable
- **py-irt** — Educational IRT only, Python <3.12, no legislative features
- **pynominate** — Minimal DW-NOMINATE port, ~30 commits, no synthesis

**Conclusion:** This project's synthesis phase is a novel contribution. There is no existing implementation to compare against or adopt.

### 1.3 Detection Algorithms: Novel but Grounded

| Detection | Our Approach | Literature Precedent | Assessment |
|-----------|-------------|---------------------|------------|
| **Maverick** | Lowest CQ party unity, weighted_maverick tiebreaker | CQ Unity is the standard metric (Crespin, Rohde & Vander Wielen 2013) | **Sound.** Follows the field standard exactly. |
| **Bridge** | Highest betweenness within 1 SD of cross-party midpoint | Fowler (2006): betweenness identifies "people who connect communities" | **Sound with a good refinement.** The midpoint constraint prevents false positives from intra-party hubs. |
| **Paradox** | IRT percentile rank vs loyalty percentile rank gap > 0.5 | No direct precedent. Conceptually justified by Sinclair (agreement scores vs ideal points) and Gerrish & Blei (2012, issue-adjusted models) | **Novel.** Theoretically grounded but the 0.5 threshold is calibrated on one session. |
| **Annotation** | Notables + most extreme per party, deduped, max 3 | N/A (visualization convention) | **Fine.** |

### 1.4 Methods Worth Knowing About (Not Necessarily Adopting)

**Mahalanobis distance for multivariate outlier detection.** Currently, paradox detection uses a pairwise rank gap between two specific metrics. Mahalanobis distance across all metrics simultaneously would be more principled — it accounts for correlations between variables and uses the chi-squared distribution for a statistically justified threshold. However, it would detect "unusual" legislators without explaining *why* they're unusual, which is less narratively useful than the current pairwise approach. **Recommendation: don't adopt**, but document the alternative.

**Spirling & Quinn (2010) Dirichlet Process Mixture Model for faction detection.** The gold-standard method for intra-party voting blocs (JASA). Uses a nonparametric Bayesian model that estimates the number of factions rather than assuming it. Our clustering phase already found k=2 (party only) with no sub-party factions in Kansas. The DPMM would confirm this more rigorously, but at significant implementation cost for a null result. **Recommendation: defer** unless future sessions show evidence of factional structure.

**Goplerud (2024) hierarchical joint model.** Jointly estimates ideal points, bridge identification, and cluster assignments in a single model. Produces posterior probabilities of bridge membership. Academically superior to post-hoc synthesis, but requires a completely different pipeline architecture. **Recommendation: note for the roadmap** as a potential future direction, not a near-term improvement.

---

## 2. Code Audit

### 2.1 Architecture

The three-file structure is clean and well-motivated:

```
synthesis.py         (936 lines) — Orchestrator: load, join, plot, delegate
synthesis_detect.py  (370 lines) — Pure data logic: detect notables
synthesis_report.py (1091 lines) — Report builder: 30 section functions
```

**Separation of concerns is good.** The detector has no I/O, no plotting, no report building — it takes DataFrames in and returns frozen dataclasses out. This makes it testable and reusable (profiles and cross-session phases import it directly).

**Data flow is linear and easy to follow:**
1. `load_all_upstream()` → dict of parquets + manifests
2. `build_legislator_df()` → unified DataFrame per chamber (LEFT JOIN on IRT base)
3. `detect_all()` → notables dict (mavericks, bridges, paradoxes)
4. Plotting functions → new PNGs
5. `build_synthesis_report()` → 29-32 section HTML

### 2.2 Issues Found

#### Issue 1 (Substantive): Hardcoded fallback values in `plot_pipeline_summary`

**Location:** `synthesis.py:688-702`

```python
total_votes = eda.get("All", {}).get("votes_before", 882)
contested = eda.get("All", {}).get("votes_after", 491)
party_votes = indices.get("house_n_party_votes", 193) + indices.get("senate_n_party_votes", 108)
best_auc = 0.98  # Approximate from XGBoost holdout (House 0.984, Senate 0.979)
k_optimal = 2  # From clustering
```

These fallback values (882, 491, 193, 108, 0.98, 2) are from the 91st Legislature. On any other session, if manifests are missing, the pipeline infographic will silently display wrong numbers. The `best_auc` is *always* hardcoded — the commented-out code at lines 694-699 that was supposed to extract it from holdout results does nothing (the loop body is `pass`).

**Severity:** Medium. Wrong data is worse than missing data for a nontechnical audience.

**Recommendation:** Extract `best_auc` from holdout_results parquets (the data is already loaded in `upstream`). Replace numeric fallbacks with `"N/A"` or skip the infographic if manifests are incomplete. The `k_optimal` should be read from the clustering manifest.

#### Issue 2 (Substantive): Minority-party maverick detection is skipped

**Location:** `synthesis_detect.py:321-322`, `detect_all()`

```python
if majority:
    mav = detect_chamber_maverick(leg_df, majority, chamber)
```

Only the majority party's maverick is detected. In a legislature with a 72% Republican supermajority, the most interesting *Democrat* maverick is never surfaced. The minority party's mavericks are arguably *more* interesting — a Democrat who breaks ranks in a supermajority legislature is making a politically costly choice.

**Severity:** Medium. Not a bug, but a significant analytical blind spot.

**Recommendation:** Detect mavericks in *both* the majority and minority parties. The report can present them differently ("the Republican most likely to break ranks" vs "the Democrat most likely to cross the aisle"). This would require modifying `detect_all()` to iterate over all parties, and the `profiles`/`annotations` dicts to hold multiple mavericks.

#### Issue 3 (Code Defect): `report` parameter typed as `object`

**Location:** `synthesis_report.py:21`

```python
def build_synthesis_report(
    report: object,
    ...
```

The `report` parameter is typed as `object`, but it's actually a `ReportBuilder` instance. Every `_add_*` function calls `report.add()` and `report._sections` is accessed in `synthesis.py:926`. This hides type errors and prevents IDE autocompletion.

**Severity:** Low (works at runtime but violates type safety).

**Recommendation:** Import `ReportBuilder` and type the parameter correctly:
```python
from analysis.report import ReportBuilder
def build_synthesis_report(report: ReportBuilder, ...) -> None:
```

#### Issue 4 (Code Quality): Stale docstrings

**Location:** Multiple files

- `synthesis.py:4` says "all 8 analysis phases" — there are 10 upstream phases (includes beta_binomial and hierarchical)
- `synthesis_report.py:4` says "all 8 analysis phases" — same issue
- `synthesis_report.py:31` says "27-30 sections" — actual range is 29-32
- `SYNTHESIS_PRIMER` at `synthesis.py:69` lists only 8 inputs — missing beta_binomial and hierarchical
- `load_all_upstream()` docstring at line 136 says "8 upstream phases" — actually 10

**Severity:** Low but misleading for future readers.

**Recommendation:** Update all references from "8" to "10" and section counts from "27-30" to "29-32".

#### Issue 5 (Code Quality): Duplicated annotation logic in `plot_dashboard_scatter`

**Location:** `synthesis.py:370-377`

The dashboard scatter function has its own "most extreme per party" annotation logic:

```python
for party in ["Republican", "Democrat"]:
    party_df = leg_df.filter(pl.col("party") == party)
    if party_df.height > 0:
        most_extreme = party_df.sort("xi_mean", descending=(party == "Republican")).head(2)
        slugs.extend(most_extreme["legislator_slug"].to_list())
```

But `detect_annotation_slugs()` in `synthesis_detect.py:288-295` already does the same thing. The function receives `annotate_slugs` from `detect_all()` → `detect_annotation_slugs()`, but then *also* adds extreme legislators redundantly. This means the dashboard can show up to 7+ annotations (3 from detect + 4 from duplicated logic), exceeding the visual clutter threshold that `max_n=3` was designed to prevent.

**Severity:** Medium. Produces cluttered annotations on some sessions.

**Recommendation:** Remove the duplicated extreme-per-party logic from `plot_dashboard_scatter`. The caller already passes the right slugs from `detect_annotation_slugs()`.

#### Issue 6 (Code Quality): `slug.split("_")[1]` assumes slug format

**Location:** `synthesis.py:551`, `synthesis_report.py:402`, `synthesis_report.py:596`

```python
slug_short = slug.split("_")[1]  # e.g., "schreiber"
```

This assumes slugs are always `prefix_lastname`. A slug like `rep_van_dyk` would produce `"van"` instead of `"van_dyk"`. The project's actual slug format is `rep_lastname` or `sen_lastname`, so this works in practice, but it's fragile.

**Severity:** Low (works with current data, would break on compound surnames).

**Recommendation:** Use `slug.split("_", 1)[1]` to split on only the first underscore, or better, extract from `full_name` directly:
```python
slug_short = slug.split("_", 1)[1]
```

#### Issue 7 (Code Quality): `try/except ModuleNotFoundError` import pattern

**Location:** `synthesis.py:35-48`, `synthesis_report.py:14-17`

Every module has dual-import try/except blocks:
```python
try:
    from analysis.run_context import RunContext
except ModuleNotFoundError:
    from run_context import RunContext
```

This exists to support running the file directly (`python analysis/synthesis.py`) vs as a module (`python -m analysis.synthesis`). It's a pragmatic workaround, but it's duplicated across all 14 analysis phases. The PEP 302 meta-path finder in `analysis/__init__.py` (ADR-0030) was designed to solve this for cross-phase imports, but intra-module imports still use this pattern.

**Severity:** Low (cosmetic, works correctly).

**Recommendation:** Not worth changing — this pattern is stable across the codebase and the fix would touch every analysis file.

---

## 3. Test Gap Analysis

### 3.1 Current Coverage

**`test_synthesis_detect.py`:** 25 tests covering all 7 public functions in `synthesis_detect.py`. Detection logic is well-tested with realistic synthetic data and edge cases.

**`test_synthesis.py`:** 47 tests covering `synthesis_data.py` (data loading, parquet I/O, `build_legislator_df` joins, percentile ranks, hierarchical column renaming), `_extract_best_auc`, `detect_all` integration (minority mavericks, profiles deduplication, both-chamber execution), `_minority_parties`, Democrat-majority paradox, bridge-builder fallback, and Independent ideology labeling.

**`synthesis_report.py`:** 0 tests. No coverage for any of the 30 section-building functions (would require mocking ReportBuilder or testing HTML output).

### 3.2 Test Gaps

| # | Gap | Priority | Rationale |
|---|-----|----------|-----------|
| 1 | `detect_all()` integration test | High | Only function in `synthesis_detect.py` with no tests. Orchestrates all detectors on both chambers. Should verify the full output dict structure. |
| 2 | `build_legislator_df()` join correctness | High | Core data function consumed by 3 downstream phases. Should verify: base IRT join, left-join column presence, percentile rank computation, missing upstream graceful handling. |
| 3 | `ideology_label()` for Independent party | Medium | Not tested. Currently returns "moderate" for any non-R/non-D, which is correct by accident (falls through both `if` branches). Should be explicit. |
| 4 | `detect_bridge_builder()` fallback to highest betweenness | Medium | The no-one-near-midpoint fallback path is not tested. All current test cases have candidates near the midpoint. |
| 5 | `detect_metric_paradox()` Democrat majority chamber | Medium | All test data uses Republican majority. A Democrat-majority scenario would exercise the `direction` logic differently (leftward vs rightward). |
| 6 | `detect_annotation_slugs()` with ParadoxCase input | Medium | Tests only pass `NotableLegislator` objects. `ParadoxCase` has `slug` attribute but the function iterates `notables` generically — should verify it works. |
| 7 | `_read_parquet_safe()` missing file returns None | Low | Trivial but untested I/O helper used everywhere. |
| 8 | `_add_bayesian_loyalty_narrative()` most-moved calculation | Low | The shrinkage delta computation accesses `unity_score - posterior_mean` which could be wrong if the columns have different semantics across sessions. |
| 9 | `plot_profile_card()` with missing metrics | Low | What happens when a legislator has only 2 of 6 metrics available? The function handles it (builds partial bar chart), but it's untested. |

### 3.3 Recommended Test Additions

Tests 1-3 are the highest value. They cover the most critical untested code paths that downstream phases depend on:

**Test 1: `detect_all()` integration**
- Verify output dict has all 5 keys (profiles, paradoxes, annotations, mavericks, bridges)
- Verify detected notables match known synthetic data
- Verify paradox slug appears in profiles dict (the conversion logic at lines 343-357)
- Verify bridge slug is excluded from profiles if it's already the maverick

**Test 2: `build_legislator_df()` join correctness**
- Mock upstream dict with known parquets
- Verify all expected columns present after join
- Verify percentile ranks are in [0, 1]
- Verify ValueError raised when IRT is missing
- Verify graceful handling when optional phases (PCA, UMAP, beta_binomial) are absent

**Test 3: `ideology_label()` extended cases**
- Independent with positive xi → "moderate"
- Independent with negative xi → "moderate"
- Zero xi_mean for all parties → "moderate"

---

## 4. Refactoring Opportunities

### 4.1 Extract Unified Legislator Join to Shared Module

**Current state:** `build_legislator_df()` and `load_all_upstream()` live in `synthesis.py` but are imported by profiles (`analysis/25_profiles/profiles.py:69`) and cross-session (`analysis/26_cross_session/cross_session.py:80`).

**Problem:** `synthesis.py` is 936 lines, and ~200 of those (the data loading and joining functions) are pure data logic with no synthesis-specific behavior. Downstream phases importing from `synthesis.py` pull in matplotlib, the primer constant, `main()`, and everything else — creating a heavyweight dependency chain for what should be a lightweight data operation.

**Recommendation:** Extract `load_all_upstream()`, `build_legislator_df()`, `_read_parquet_safe()`, and `_read_manifest()` into a new `synthesis_data.py` module alongside `synthesis_detect.py`. This would give three focused modules:

```
synthesis_data.py     — Data loading and joining (pure I/O)
synthesis_detect.py   — Notable legislator detection (pure logic)
synthesis_report.py   — Report building (templates + sections)
synthesis.py          — Orchestrator (imports from all three)
```

**Effort:** Low. The functions are already self-contained. Move them, update imports in synthesis.py, profiles.py, and cross_session.py.

**Benefit:** Cleaner dependency graph. Downstream phases don't need to import matplotlib transitively. Easier to test the data layer in isolation.

### 4.2 Remove Dead Code in `plot_pipeline_summary`

**Location:** `synthesis.py:693-699`

```python
best_auc = 0.0
for ch_data in [
    prediction.get("chambers", {}).get("House", {}),
    prediction.get("chambers", {}).get("Senate", {}),
]:
    # Not stored directly; use hardcoded from holdout_results parquets
    pass
best_auc = 0.98
```

The loop does nothing. The variable is set to 0.0, the loop body is `pass`, then it's overwritten with 0.98. This is dead code from a failed attempt to extract AUC from manifests.

**Recommendation:** Either implement the extraction (the holdout_results parquets are already loaded in `upstream["house"]["holdout_results"]`), or remove the dead loop and add a comment explaining why the value is hardcoded.

---

## 5. Correctness Review

### 5.1 Detection Algorithms

**Maverick detection** (`detect_chamber_maverick`): Correct. Uses the CQ-standard definition. The 0.95 skip threshold is reasonable — it prevents surfacing a "maverick" with 94.8% unity when the party average is 95.2%.

**Bridge detection** (`detect_bridge_builder`): Correct with one subtlety. The midpoint is computed as the mean of per-party medians:

```python
midpoint = sum(medians.values()) / len(medians)
```

With 3+ parties (R, D, Independent), this midpoint would include the Independent party's median, which may not be meaningful. In practice, Kansas has at most 1 Independent legislator, so the median of a 1-person party is just that person's xi_mean. The effect is small but mathematically impure.

**Paradox detection** (`detect_metric_paradox`): Correct. The percentile rank normalization is the right choice — it makes the 0.5 threshold interpretable across sessions with different score scales. The direction logic correctly handles both "extreme ideologue, low loyalty" and "moderate, high loyalty" paradox directions.

### 5.2 DataFrame Joins

**`build_legislator_df`**: Correct. Uses LEFT JOIN on legislator_slug with IRT as the base table, which means:
- Every IRT-included legislator appears in the output
- Legislators filtered out of upstream phases (e.g., <20 votes in clustering but included in IRT) get null values for those columns
- No duplicate rows (legislator_slug is unique per chamber in all upstream outputs)

The percentile rank computation at line 323 is correct:

```python
df = df.with_columns((pl.col(col).rank("ordinal") / n).alias(f"{col}_percentile"))
```

This produces ranks in `(0, 1]` (not `[0, 1]`) because ordinal rank starts at 1, not 0. The value 1/n is the minimum, not 0. This is fine for the use cases (dashboard scatter, paradox detection) but worth noting.

### 5.3 Report Building

The report builder is straightforward — 30 functions that each add a section to the report. The narrative templates are well-crafted for a nontechnical audience. Dynamic data interpolation (f-strings with manifest values) is correct throughout.

One fragile pattern: several sections use `next(iter(paradoxes.values()))` to get the first (and presumably only) paradox. If `detect_all()` ever returns multiple paradoxes (one per chamber), only the first would be reported. Currently this is not a problem — each chamber's paradox is in the same dict — but the code should document this assumption.

---

## 6. Comparison with Academic Best Practices

### 6.1 What the Pipeline Does Well

**Multi-method triangulation.** The synthesis report explicitly shows where eight methods agree (party dominates) and where they disagree (paradox cases). This follows the convergent validation approach recommended by McCarty (2010).

**Nontechnical narrative.** The report is organized by story, not by method. Section titles ("The Party Line Is Everything", "Who Are the Mavericks?") are accessible to journalists and policymakers. This is rare in academic work — most legislative analysis outputs are aimed at other political scientists.

**Data-driven detection.** ADR-0008's decision to replace hardcoded legislator names with algorithmic detection was exactly right. No published legislative analysis pipeline does this — they either hardcode or require manual annotation.

**Graceful degradation.** The report generates cleanly even when upstream phases are missing. Section counts vary from 29-32. This is production-quality robustness.

### 6.2 Where It Could Be Stronger

**Cross-method agreement quantification.** The report asserts "every method confirms party dominance" but doesn't quantify the agreement. A cross-method correlation table (IRT rank vs PCA PC1 rank vs party unity rank) would strengthen the triangulation claim with numbers, not just prose. Spearman rank correlations between upstream metrics are already computable from the unified legislator DataFrame.

**Confidence intervals on detection.** The current detectors return point estimates ("this person is the maverick") without uncertainty. The IRT ideal points have credible intervals, but the detection logic ignores them. A legislator whose 95% CI overlaps with the next candidate is a weaker maverick than one with a clear separation. This is a known limitation of using point estimates from Bayesian models.

**Only one paradox per session.** `detect_metric_paradox` returns the single largest gap. In a 125-member House, there might be 2-3 genuinely paradoxical cases. The function could return a ranked list and let the report decide how many to feature.

---

## 7. Summary of Recommendations

### High Priority

| # | Recommendation | Files | Effort | Status |
|---|---------------|-------|--------|--------|
| 1 | Fix hardcoded AUC/vote-count fallbacks in `plot_pipeline_summary` | `synthesis.py` | Low | **Done** |
| 2 | Add `detect_all()` integration test | `test_synthesis.py` | Low | **Done** |
| 3 | Add `build_legislator_df()` join tests | `test_synthesis.py` | Medium | **Done** |
| 4 | Fix duplicated annotation logic in `plot_dashboard_scatter` | `synthesis.py` | Low | **Done** |

### Medium Priority

| # | Recommendation | Files | Effort | Status |
|---|---------------|-------|--------|--------|
| 5 | Detect minority-party mavericks | `synthesis_detect.py`, `synthesis_report.py` | Medium | **Done** |
| 6 | Update stale docstrings (8→10 phases, 27-30→29-32 sections) | All 3 files | Low | **Done** |
| 7 | Type `report` parameter as `ReportBuilder` | `synthesis_report.py` | Low | **Done** |
| 8 | Fix `slug.split("_")[1]` to `slug.split("_", 1)[1]` | `synthesis.py`, `synthesis_report.py` | Low | **Done** |

### Low Priority / Deferred

| # | Recommendation | Files | Effort | Status |
|---|---------------|-------|--------|--------|
| 9 | Extract data loading into `synthesis_data.py` | New file + import updates | Medium | **Done** |
| 10 | Add cross-method correlation table to report | `synthesis_report.py` | Medium | Deferred |
| 11 | Bridge detection: handle 3+ party midpoint more carefully | `synthesis_detect.py` | Low | Deferred |
| 12 | Support multiple paradoxes per session | `synthesis_detect.py`, `synthesis_report.py` | Medium | Deferred |

### Implementation Notes

Recommendations 1-9 were implemented in a single session. Key changes:

- **`synthesis_data.py` (new):** Extracted `UPSTREAM_PHASES`, `_read_parquet_safe`, `_read_manifest`, `load_all_upstream`, and `build_legislator_df` from `synthesis.py`. Updated imports in `profiles.py`, `cross_session.py`, and `analysis/__init__.py` module map.
- **`_extract_best_auc()` (new):** Dynamically reads XGBoost AUC from holdout_results parquets, replacing the dead-loop + hardcoded 0.98 value. Falls back to `"N/A"` (not a fabricated number) when prediction hasn't run (ADR-0037).
- **Minority-party mavericks:** `detect_all()` now detects the lowest-unity minority-party legislator per chamber. Report adds a "crossing the aisle" paragraph.
- **`test_synthesis.py` (new):** 47 tests covering data loading, joins, AUC extraction, `detect_all` integration, minority mavericks, Democrat-majority paradox, and edge cases.
- Test count: 975 → 1022 (47 new tests, 0 regressions).

---

## 8. References

- Crespin, Rohde & Vander Wielen (2013). "Measuring variations in party unity voting." *Party Politics* 19(3). — CQ party unity standard.
- Fowler (2006). "Legislative cosponsorship networks in the US House and Senate." *Social Networks* 28(4). — Betweenness centrality for bridge identification.
- Gerrish & Blei (2012). "How They Vote: Issue-Adjusted Models of Legislative Behavior." *NIPS*. — Issue-domain-specific ideal points; theoretical basis for paradox detection.
- Goplerud (2024). "Explaining Differences in Voting Patterns Across Voting Domains Using Hierarchical Bayesian Models." *Political Analysis* (arXiv:2312.15049). — Joint estimation of ideal points + clusters.
- Jakulin & Buntine (2005). "Analyzing the US Senate in 2003: Similarities, Clusters, and Blocs." Columbia University. — Multi-method synthesis on roll-call data.
- McCarty (2010). "Measuring Legislative Preferences." Princeton. — Comparison of NOMINATE vs Bayesian IRT methods.
- Sinclair (2015). "Agreement Scores, Ideal Points, and Legislative Polarization." — Different measures produce different rank orderings.
- Spirling & Quinn (2010). "Identifying Intraparty Voting Blocs in the U.K. House of Commons." *JASA* 105(490). — Dirichlet Process Mixture Model for faction detection.
- Victor (2025). "Network Connections and Bipartisan Cooperation." *American Politics Research*. — Cross-party network effects.
