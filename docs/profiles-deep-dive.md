# Profiles Deep Dive

A code audit, ecosystem survey, and fresh-eyes evaluation of the Profiles phase (Phase 12).

**Date:** 2026-02-25

---

## Executive Summary

The profiles phase is a well-designed consumer of upstream analysis outputs that produces per-legislator deep-dive reports. The three-file architecture (data, orchestration, report) follows the project's established pattern with clean separation of concerns.

An ecosystem survey confirms this is **novel work** — no open-source Python project generates per-legislator profiles that combine IRT ideal points, bill discrimination tiers, defection analysis, prediction-model surprises, and voting neighbors into a unified report. The closest comparisons are GovTrack.us (cosponsorship-based ideology + PageRank leadership) and the OpenStatesParser project (basic pairwise agreement), neither of which approaches this level of analytical depth.

This deep dive identifies **five issues** (two substantive, one code defect, two code quality), **eleven test gaps**, and **two refactoring opportunities**. It also surveys the academic and open-source landscape for per-legislator profiling — a survey that validates the implementation's methodological choices while identifying where the literature diverges from our approach.

---

## 1. Field Survey: How Do Others Build Legislator Profiles?

### 1.1 Open-Source Projects

| Project | Approach | Metrics | Output |
|---------|----------|---------|--------|
| **Voteview** | DW-NOMINATE (MLE) + Nokken-Poole | dim1, dim2, party_code, log_likelihood | CSV per Congress |
| **GovTrack.us** | Cosponsorship SVD + PageRank | Ideology Score, Leadership Score, report cards | Web pages |
| **OpenStatesParser** | Pairwise agreement from roll calls | `PeopleLike(ID)` similarity ranking | Python dicts |
| **ProgressiveMass** | Curated key votes, advocacy scoring | Policy alignment percentage | Gatsby web app |
| **BillTrack50** | Vote index (correct/total), letter grades | Weighted vote purity score | SaaS platform |
| **FiveThirtyEight** | Presidential agreement + probit residual | Agreement, Predicted, Plus-Minus | CSV on GitHub |

**Key observation:** The two most analytically sophisticated approaches — Voteview's NOMINATE and FiveThirtyEight's Trump/Biden Score — are both single-method scores. No open-source project combines multiple analysis methods into a unified per-legislator profile. Our profiles phase is the only implementation that synthesizes IRT ideal points, network centrality, clustering loyalty, classical indices, prediction accuracy, and bill discrimination tiers into a single report card.

GovTrack provides the most instructive cautionary tale. Their 2024 retraction of single-year report cards (used since 2013) found that Kamala Harris's ideology score fluctuated from 8th to 1st most liberal over two years — measurement noise, not real ideological change. **Lesson for us:** single-session scores should be presented with appropriate uncertainty, and the cross-session validation phase becomes more important for validating score stability.

### 1.2 Academic Approaches

The political science literature uses a convergent set of per-legislator metrics:

| Metric | Standard In | Our Equivalent |
|--------|------------|----------------|
| NOMINATE dim1/dim2 | Poole & Rosenthal (1985, 2007) | IRT `xi_mean` + PCA PC1/PC2 |
| Party Unity (CQ) | Congressional Quarterly (annual) | `unity_score` |
| Presidential Support | CQ Roll Call | Not applicable (state legislature) |
| Interest group ratings | ACU, ADA, LCV | Not applicable (no external ratings scraped) |
| Bill-domain voting patterns | Hierarchical Bayesian (arXiv:2312.15049) | Bill type breakdown via IRT discrimination |
| Voting neighbors/similarity | Standard pairwise agreement | `find_voting_neighbors()` |

A 2025 paper in *Political Analysis* (arXiv:2312.15049) validates the bill-type-breakdown approach: they extend Bayesian IRT to let legislators have *different* ideal points across voting domains (procedural vs. final passage). Our implementation uses IRT bill discrimination (`|beta_mean|`) as the domain classifier rather than vote type, which is methodologically defensible — discrimination captures how much ideology predicts the vote outcome, regardless of procedural category.

### 1.3 Advocacy Scorecard Pitfalls

The League of Conservation Voters, ACU, and similar organizations use curated vote scorecards. Academic critique of these is relevant to our methodology:

- **Vote selection bias** (LSQ 2025, DOI:10.1111/lsq.70013): Interest groups strategically select votes that "clearly separate friends from foes," inflating partisan extremity. Our approach avoids this entirely — we use all votes, weighted by IRT discrimination rather than editorial judgment.
- **Missing-as-opposing** (BillTrack50 documents three configurable options): Treating absences as opposing votes inflates scores. We correctly exclude absences from Yea/Nay analysis in `prep_votes_long()`.
- **Cardinal vs. ordinal** trap: Scorecard scores should be treated as ordinal rankings, not cardinal measures. Our use of percentile ranks for the scorecard chart (`xi_mean_percentile`, `betweenness_percentile`) correctly addresses this.

### 1.4 Visualization Best Practices

CalMatters Digital Democracy (California's legislator tracker) is the best public example of profile design. Key principles their platform demonstrates:

- Lead with accessible findings; build toward technical detail
- Use party-colored visual encoding (red/blue) consistently
- Combine narrative text with quantitative displays
- Provide both overview (scorecard) and deep-dive (specific vote tables)

Our profiles phase follows all of these principles. The main gap relative to CalMatters is the lack of policy-domain categorization — they track 6 policy areas (Education, Health, Environment, etc.), while we classify bills only by IRT discrimination tier. This is a reasonable trade-off given we don't have bill text yet (see `docs/future-bill-text-analysis.md`).

### 1.5 What the Literature Suggests We're Missing

Two capabilities that appear in the literature but not in our profiles:

1. **Temporal trajectory within session.** Nokken-Poole scores (session-specific, no career-static assumption) are the Voteview standard for session-level ideology. We have this via the cross-session phase for between-session comparison, but not within-session — the time series analysis (roadmap items 3a/3b) would add this.

2. **~~Sponsorship/cosponsorship analysis.~~** ~~GovTrack's strongest feature is their cosponsorship-based ideology score (SVD on the cosponsor matrix) and PageRank leadership score. We don't scrape sponsorship data. This would require a scraper extension.~~ **Resolved (2026-03-02).** The scraper now captures `sponsor_slugs` (semicolon-joined legislator slugs from bill page HTML, 89th+ sessions). Phase 12 profiles include a per-legislator sponsorship section: sponsored bills table, primary vs co-sponsor role, passage rate. Defection tables include the bill's sponsor for context. Full cosponsorship-network analysis (GovTrack-style SVD + PageRank) remains a future opportunity. ADR-0081.

The temporal trajectory gap is not a deficiency in the current implementation — it's a future opportunity. The within-session TSA (Phase 15, rolling PCA drift + PELT changepoints) partially addresses it.

---

## 2. Code Audit

### 2.1 Architecture

The three-file structure is clean and follows the project standard:

```
profiles_data.py   (464 lines) — Pure data logic: dataclasses, scorecard, breakdown, defections, neighbors
profiles.py        (753 lines) — Orchestrator: CLI, vote prep, 5 plotting functions, main loop
profiles_report.py (269 lines) — Report builder: intro + 7 per-legislator section functions
```

**Separation of concerns is excellent.** `profiles_data.py` has no I/O, no plotting, and no report building. Every public function takes DataFrames in and returns structured data out. This is independently testable and reusable.

**Dependency chain is clean:**
- `profiles_data.py` → `synthesis_detect.detect_all()` (detection logic reuse)
- `profiles.py` → `profiles_data`, `profiles_report`, `synthesis_data` (loading infrastructure reuse)
- `profiles_report.py` → `profiles_data.ProfileTarget`, `report.{FigureSection, TableSection, TextSection, make_gt}`

No circular dependencies. No duplicated upstream loading code.

### 2.2 Issues Found

#### Issue 1 (Substantive): Party average includes the target legislator

**Location:** `profiles_data.py:189` (`build_scorecard`) and `profiles_data.py:255-260` (`compute_bill_type_breakdown`)

In `build_scorecard()`:
```python
party_df = leg_df.filter(pl.col("party") == party)
party_avg = party_df[col].mean()
```

The target legislator's own values are included in the party average computation. For the 70+ member Republican caucus, this is negligible. But for the 11-member Democratic Senate caucus or a 3-member Independent group, the target's own extreme value pulls the average toward them, understating the gap.

**Example:** If the Senate has 4 Republicans with unity scores [0.98, 0.90, 0.82, 0.55], the party average *including* the maverick (0.55) is 0.81. Excluding them, it's 0.90. The scorecard would show the maverick at 0.55 vs party average 0.81 (gap = 0.26) when the true gap is 0.35.

**Same issue in `compute_bill_type_breakdown()`:** `party_slugs` (line 255) includes the target, so party Yea rates on high/low-discrimination bills include the target's own votes.

**Recommendation:** Exclude the target from party averages. This is a standard "leave-one-out" correction.

**Severity:** Medium. Produces slightly misleading comparisons, especially for small caucuses. The effect is proportional to 1/N_party.

#### Issue 2 (Substantive): Defection detection includes the target in the party majority calculation

**Location:** `profiles_data.py:300` (`find_defection_bills`)

```python
party_votes = votes_long.filter(pl.col("legislator_slug").is_in(party_slugs))
```

The target's own vote is included when computing the party majority direction. For a 70+ member caucus this is negligible, but for a small caucus it creates a mathematical impossibility: a legislator in a 3-member party who votes Nay while the other two vote Yea sees `party_yea_pct = 66.7%` (majority = Yea), which correctly identifies the defection. But if it's a 2-member party and one votes Yea and one Nay, `party_yea_pct = 50%` — the majority calculation fails (the code uses `> 0.5`, so ties go to Nay). This edge case exists for Independent senators (N=1) where the concept of "party majority" is undefined.

**Recommendation:** Exclude the target from party majority computation to get "what did the rest of the party do?" semantics. This is what the FiveThirtyEight Trump Score does (computes party position excluding each individual member). Handle the N=1 case (Independent) by returning empty — defection from a party of one is meaningless.

**Severity:** Low-Medium. Kansas's party sizes (70+ R, 40+ D in House; 29 R, 11 D in Senate) make this negligible in practice. Would matter if applied to other states with smaller or more fragmented caucuses.

#### Issue 3 (Code Defect): Neighbor chart spine operations outside guard

**Location:** `profiles.py:532-534` and `profiles.py:551-554`

```python
    if closest:
        # ... bar chart code ...
        ax_close.invert_yaxis()
    ax_close.spines["top"].set_visible(False)    # outside the guard
    ax_close.spines["right"].set_visible(False)
    ax_close.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax_close.set_axisbelow(True)
```

The spine/grid operations on `ax_close` and `ax_diff` run unconditionally even if the corresponding data list is empty. The function already returns `None` if *both* are empty (line 513), but if only one is empty, the other panel gets styled without data — an empty axes with grid lines and no content.

**Recommendation:** Move spine operations inside the `if closest:` / `if most_diff:` guards. Alternatively, hide the empty axes entirely with `ax.set_visible(False)`.

**Severity:** Low. Cosmetic only. The both-empty case is already handled. The one-empty case is rare but produces a visually confusing half-empty plot.

#### Issue 4 (Code Quality): `report: object` typing across all report builders

**Location:** `profiles_report.py:25, 53, 83, 101, 124, 146, 168, 206, 251`

All report builder functions use `report: object` instead of `report: ReportBuilder`. This loses type safety — the `.add()` method call and `._sections` attribute access are unchecked.

This is a project-wide pattern (synthesis_report.py has the same thing, as do all other `*_report.py` files), not profiles-specific. The root cause is likely the try/except import pattern making it awkward to import `ReportBuilder` in a way that satisfies the type checker.

**Recommendation:** Either type as `ReportBuilder` with a conditional import, or define a `Protocol` with `add()` and `_sections`. This is a project-wide refactor, not profiles-specific. Not urgent but would catch bugs at typecheck time.

**Severity:** Low. No runtime impact. Would be caught by ty if typed properly.

#### Issue 5 (Code Quality): Private attribute access `report._sections`

**Location:** `profiles.py:743`, `profiles_report.py:47`

Both files access `report._sections` (the underscore-prefixed "private" attribute) for section count logging and manifest output. `ReportBuilder` already has a `has_sections` property but not a `section_count` property.

This is also a project-wide pattern — 16 other files access `report._sections` for the same purpose.

**Recommendation:** Add a `section_count` property to `ReportBuilder` (one line: `return len(self._sections)`) and use it everywhere. This is a project-wide cleanup, not profiles-specific.

**Severity:** Low. Functional but violates encapsulation convention.

### 2.3 No Dead Code Found

Every function in all three files is called:
- All 5 `profiles_data.py` public functions are called from `profiles.py`
- All 5 plotting functions are called from `profiles.py:main()`
- All 7 report section functions are called from `profiles_report.py:build_profiles_report()`
- Both constants (`SCORECARD_METRICS`, `PARTY_COLORS`) are used
- Both dataclasses (`ProfileTarget`, `BillTypeBreakdown`) are used
- Both helper functions (`_build_slug_lookup`, `_abs_xi`, `_empty_defection_df`) are used

No dead code. No unused imports. Clean.

### 2.4 Things Done Well

- **Graceful degradation is thorough.** Every section checks for data availability before rendering. Missing bill_params → skip breakdown. No defections → skip table. No surprising votes → skip table. No neighbors → skip chart. This means profiles never crashes on partial upstream data.
- **Detection logic is reused, not duplicated.** `gather_profile_targets()` delegates to `detect_all()` from synthesis_detect rather than reimplementing detection logic.
- **Frozen dataclasses for targets and breakdowns.** Immutable data structures prevent accidental mutation.
- **Defection sorting by margin closeness** is a good editorial choice — a 55-45 defection is more interesting than an 80-20 defection.
- **Simple agreement for neighbors** is the right call for a nontechnical audience. Kappa adjusts for base rate but adds cognitive load.
- **The "dead zone" in bill type classification** (0.5 ≤ |beta_mean| ≤ 1.5) is intentional and documented in the design doc. Including "medium discrimination" bills would dilute both categories.

---

## 3. Test Gaps

The existing 36 tests (24 original + 12 added for name resolution) cover `profiles_data.py` well. The main gaps are:

### 3.1 Missing Coverage for `profiles.py` Functions

| Function | Tests | Status |
|----------|-------|--------|
| `prep_votes_long()` | 0 | **Missing** — no test validates Yea/Nay filtering, binary encoding, or chamber lowercasing |
| `plot_enhanced_scorecard()` | 0 | **Missing** — no smoke test |
| `plot_bill_type_bars()` | 0 | **Missing** — no smoke test |
| `plot_position_in_context()` | 0 | **Missing** — no smoke test |
| `plot_defection_chart()` | 0 | **Missing** — no smoke test |
| `plot_neighbor_chart()` | 0 | **Missing** — no smoke test |

**Recommendation:** Add at minimum:
1. **`test_prep_votes_long_filters_yea_nay`** — Verify only Yea/Nay rows survive, binary encoding is correct, chamber is lowercased.
2. **`test_prep_votes_long_excludes_other_categories`** — Feed in "Present and Passing", "Absent and Not Voting" rows, verify they're dropped.
3. **Smoke tests for each plot function** — Pass synthetic data, verify the function returns a `Path` (not `None`), verify the file exists on disk, clean up with `tmp_path`. These catch regressions in matplotlib API usage. Five tests, one per plot function.

### 3.2 Missing Coverage for `profiles_report.py`

| Function | Tests | Status |
|----------|-------|--------|
| `build_profiles_report()` | 0 | **Missing** |
| `_add_intro()` | 0 | **Missing** |
| `_add_target_header()` | 0 | **Missing** |
| `_add_scorecard_figure()` | 0 | **Missing** |
| `_add_defections_table()` | 0 | **Missing** |
| `_add_surprising_votes_table()` | 0 | **Missing** |
| `_add_neighbors_figure()` | 0 | **Missing** |

**Recommendation:** Add at minimum:
4. **`test_build_profiles_report_adds_sections`** — Build a report with one target and synthetic data, verify sections are added.
5. **`test_report_graceful_with_empty_data`** — Build a report where a target has no defections, no surprising votes, no breakdown — verify no crash and fewer sections.

### 3.3 Missing Edge Case Tests

6. **`test_scorecard_excludes_missing_columns`** — Pass a leg_df that's missing `accuracy` (e.g., prediction phase didn't run). Verify scorecard still works with available metrics.
7. **`test_bill_type_breakdown_missing_beta_mean`** — Pass bill_params without `beta_mean` column. Verify returns `None`.
8. **`test_defection_at_exact_50_pct`** — When `party_yea_pct == 0.5`, verify behavior. Currently ties go to Nay (the `> 0.5` threshold), so a Yea vote is a defection at exactly 50%. This is a boundary condition worth testing explicitly.
9. **`test_voting_neighbors_minimum_shared_votes`** — Two legislators share only 4 votes (below the 5-vote minimum). Verify the pair is excluded.

### 3.4 Summary

| Category | Current | Recommended | New Tests |
|----------|---------|-------------|-----------|
| `profiles_data.py` (original) | 24 | 28 | 4 edge cases |
| `profiles_data.py` (name resolution) | 12 | 12 | — (complete) |
| `profiles.py` (data prep) | 0 | 2 | `prep_votes_long` |
| `profiles.py` (plotting) | 0 | 5 | smoke tests |
| `profiles_report.py` | 0 | 2 | report integration |
| **Total** | **36** | **49** | **+13** |

---

## 4. Refactoring Opportunities

### 4.1 Leave-One-Out Party Averages (Issues 1 & 2)

Both `build_scorecard()` and `compute_bill_type_breakdown()` should exclude the target legislator from party averages. The `find_defection_bills()` function should exclude the target from the party majority computation. This is a small, localized change in each function:

```python
# In build_scorecard():
party_df = leg_df.filter(
    (pl.col("party") == party) & (pl.col("legislator_slug") != slug)
)

# In find_defection_bills():
party_votes = votes_long.filter(
    pl.col("legislator_slug").is_in(party_slugs)
    & (pl.col("legislator_slug") != slug)
)
```

Same pattern for `compute_bill_type_breakdown()`.

**Impact:** Small numerical differences for large caucuses, meaningful improvement for small caucuses. Makes the statistics cleaner — "how does this legislator compare to *the rest of* their party" rather than "to their party including themselves."

### 4.2 Neighbor Chart Empty Panel Handling

The `plot_neighbor_chart()` function creates a 2-panel subplot unconditionally. When one panel has no data, it displays as an empty axes with grid lines. Better to either:

(a) Hide the empty axes: `ax.set_visible(False)` when no data, or
(b) Only create a 1-panel figure when one list is empty.

Option (a) is simpler. The full change is moving the spine/grid operations inside the guards and adding `else: ax.set_visible(False)` for each panel.

---

## 5. Comparison: Our Implementation vs. the Field

### 5.1 What We Do Better

| Capability | Us | Best Alternative |
|-----------|-----|-----------------|
| Multi-method synthesis | 8 upstream phases joined into one profile | GovTrack: 2 methods (SVD + PageRank) |
| Bill discrimination tiers | IRT `beta_mean` classifies partisan vs routine | None found |
| Surprise detection | XGBoost prediction residuals | None in per-legislator profiles |
| Auto-detection of notable legislators | `detect_all()`: mavericks, bridges, paradoxes | Manual curation everywhere else |
| Bayesian uncertainty | HDI error bars on forest plot | Voteview's NOMINATE SEs (not per-profile) |
| Graceful degradation | Every section checks data availability | Most crash on missing data |

### 5.2 What the Field Does That We Don't

| Capability | Who Does It | Gap Assessment |
|-----------|------------|----------------|
| ~~Cosponsorship analysis~~ | ~~GovTrack (SVD + PageRank)~~ | **Partially resolved** (ADR-0081): per-legislator sponsorship stats + defection sponsor context now in profiles. Full cosponsorship-network SVD remains future work. |
| Policy-domain categorization | CalMatters (6 policy areas) | Requires bill text — see `docs/future-bill-text-analysis.md` |
| Interactive web profiles | CalMatters, GovTrack, Vote Smart | Our static HTML is appropriate for the current audience |
| Multi-year career trajectory | Voteview (DW-NOMINATE vs Nokken-Poole) | Cross-session phase handles between-biennium; within-session temporal analysis is a roadmap item |
| Interest group ratings aggregation | Vote Smart (aggregates ACU, ADA, LCV, etc.) | External data integration not in scope |
| Campaign finance integration | CalMatters, Vote Smart | Not in scope |

### 5.3 Methodological Validation from the Literature

Three of our design choices are directly supported by academic work:

1. **IRT discrimination for bill classification.** The 2025 *Political Analysis* paper on domain-based voting shows that legislators do vote differently across voting domains. Our use of `|beta_mean|` to define domains (high-disc = partisan, low-disc = routine) is a principled alternative to procedural/final-passage classification.

2. **Prediction residuals for surprise detection.** Both IRT model residuals and posterior predictive checks are standard approaches for identifying "unusual" votes. Our use of XGBoost's `confidence_error` is functionally equivalent — it identifies votes where the model was confident and wrong.

3. **Simple agreement for neighbors.** The pairwise agreement rate is the most common similarity metric in the literature for nontechnical presentation. Cohen's Kappa (used in our network phase) adjusts for chance agreement but is harder to explain. The design doc's choice of agreement over Kappa for profiles is well-justified.

---

## 6. Recommendations Summary

### Priority 1 (Substantive Correctness)

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| 1 | Party average includes target | `profiles_data.py:189, 255` | Exclude target from party filter |
| 2 | Defection majority includes target | `profiles_data.py:300` | Exclude target from party votes |

### Priority 2 (New Tests)

| # | Test | Covers |
|---|------|--------|
| 3 | `test_prep_votes_long_filters_yea_nay` | Binary encoding, category filtering |
| 4 | `test_prep_votes_long_excludes_other_categories` | Non-Yea/Nay vote exclusion |
| 5-9 | Plot function smoke tests (5 tests) | Matplotlib regression protection |
| 10 | `test_build_profiles_report_adds_sections` | Report integration |
| 11 | `test_report_graceful_with_empty_data` | Graceful degradation |
| 12 | `test_scorecard_excludes_missing_columns` | Partial upstream data |
| 13 | `test_bill_type_breakdown_missing_beta_mean` | Missing column handling |
| 14 | `test_defection_at_exact_50_pct` | Boundary condition |
| 15 | `test_voting_neighbors_minimum_shared_votes` | Minimum pair threshold |

### Priority 3 (Code Quality)

| # | Issue | Scope |
|---|-------|-------|
| 16 | Neighbor chart empty panel handling | `profiles.py:plot_neighbor_chart` |
| 17 | `report: object` → `ReportBuilder` typing | Project-wide (not profiles-specific) |
| 18 | `report._sections` → `report.section_count` property | Project-wide (not profiles-specific) |

---

## 7. Name-Based Legislator Lookup

### 7.1 Motivation

The profiles phase originally required exact legislator slugs (`--slugs rep_alcala_john_1`) — opaque identifiers from the KS Legislature website. The `--names` flag allows natural-language lookup:

```bash
just profiles --names "Masterson"
just profiles --names "Blake Carpenter,Dietrich"
just profiles --names "Masterson" --slugs rep_alcala_john_1  # combine both
```

### 7.2 Matching Algorithm

Multi-stage matching via `resolve_names()` in `profiles_data.py`:

1. **Exact full-name** (case-insensitive): `"Ty Masterson"` → direct match
2. **Last-name-only**: `"Masterson"` → finds all legislators with that last name
3. **First-name disambiguation**: `"Blake Carpenter"` → narrows multiple Carpenters to one

Name normalization reuses `normalize_name()` from `cross_session_data` (lowercases, strips leadership suffixes like "- House Minority Caucus Chair").

### 7.3 Ambiguity Handling

Each query resolves to one of three statuses:

| Status | Meaning | Behavior |
|--------|---------|----------|
| `ok` | Single match | Slug added, confirmation printed |
| `ambiguous` | Multiple matches | All slugs added, list printed |
| `no_match` | No match found | Warning printed, skipped |

Ambiguous matches include all candidates (e.g., if "Jones" matches both House and Senate members, both are profiled). Users can provide a full name to disambiguate.

### 7.4 Implementation

- `NameMatch` frozen dataclass: `query`, `status`, `matches` (list of info dicts)
- `_build_name_lookup()`: builds full-name and last-name lookup tables from `leg_dfs`
- `resolve_names()`: public API, returns `list[NameMatch]`
- `_resolve_name_args()` in `profiles.py`: CLI glue that prints status messages and collects slugs

Backward compatible — `--slugs` continues to work unchanged. Both flags can be combined.

---

## 8. References

### Open-Source Projects
- unitedstates/congress: https://github.com/unitedstates/congress
- Voteview data: https://voteview.com/data
- GovTrack methodology: https://www.govtrack.us/about/analysis
- GovTrack retraction: https://www.govtrack.us/posts/434/2024-07-26_we-retracted-our-single-year-legislator-report-cards-after-warning-about-their-unreliability
- OpenStatesParser: https://github.com/Xenocrypt/OpenStatesParser
- ProgressiveMass scorecard: https://github.com/ProgressiveMass/legislator-scorecard
- CalMatters Digital Democracy: https://calmatters.digitaldemocracy.org/
- BillTrack50 scoring: https://www.billtrack50.com/info/help/scoring-missed-votes-on-a-scorecard
- FiveThirtyEight Congress data: https://github.com/fivethirtyeight/data/tree/master/congress-trump-score

### Academic Papers
- Poole & Rosenthal, NOMINATE: Poole & Rosenthal (1985, 2007), *Congress: A Political-Economic History*
- Nokken & Poole (2004), "Congressional Party Defection in American History," *Legislative Studies Quarterly*
- Domain-based voting (hierarchical Bayesian): arXiv:2312.15049, *Political Analysis* (2025)
- Political DNA anomaly detection: PMC7462714, *PLOS ONE* (2020)
- Interest group scorecard critique: Cambridge University Press, *Studies in American Political Development* (2018)
- Interest groups and scored votes: DOI:10.1111/lsq.70013, *Legislative Studies Quarterly* (2025)
- Missing data imputation: King et al. (2001), *American Political Science Review* 95(1):49-69

### Scorecard Methodologies
- LCV: https://www.lcv.org/congressional-scorecard/about-the-scorecard/
- ACU: https://ratings.conservative.org/
