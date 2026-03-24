# Common Space Ideal Points Design Choices

**Script:** `analysis/28_common_space/common_space.py`
**Data module:** `analysis/28_common_space/common_space_data.py`
**Report:** `analysis/28_common_space/common_space_report.py`
**Deep dive:** `docs/common-space-ideal-points.md`

## Purpose

Produce a single ideological scale spanning all 14 bienniums (78th-91st, 1999-2026) so that any legislator can be compared to any other, regardless of whether they overlapped. The approach uses simultaneous affine alignment of canonical ideal points via bridge legislators.

## Assumptions

1. **Canonical ideal points are the input.** Horseshoe-corrected scores from the routing system (ADR-0109). The common space phase does not estimate ideal points — it links existing ones.

2. **Chambers are aligned separately.** House and Senate have different bills, different ideal point scales. Cross-chamber unification is an optional second step using career chamber-switchers.

3. **The 91st Legislature is the reference scale.** Most recent, best data quality, best convergence. All other bienniums are mapped onto its scale. A = 1, B = 0 for the reference.

4. **Bridge legislators' relative positions are approximately stable across sessions.** The affine model allows for scale and location differences but assumes the *ordering* of bridge legislators is preserved. Trimmed regression handles genuine movers.

5. **Per-biennium canonical scores must exist before this phase runs.** The 78th-83rd bienniums need at minimum flat IRT (Phase 05) with canonical routing. If missing, those bienniums are skipped with a warning.

## Parameters & Constants

| Constant | Value | Justification |
|----------|-------|---------------|
| `REFERENCE_SESSION` | `"91st_2025-2026"` | Most recent, best convergence, most downstream phases complete. |
| `MIN_BRIDGES` | 20 | Minimum bridge legislators for a reliable pairwise link. Matches Phase 26's `MIN_OVERLAP`. Psychometric literature recommends 20-25. |
| `TRIM_PCT` | 10 | Trim top/bottom 10% residuals from alignment. Matches Phase 26's `ALIGNMENT_TRIM_PCT`. Excludes genuine movers. |
| `N_BOOTSTRAP` | 1000 | Bootstrap iterations for uncertainty quantification. Standard in psychometric linking. |
| `BOOTSTRAP_SEED` | 42 | Reproducibility. |
| `CORRELATION_WARN` | 0.70 | Warn if any pairwise bridge correlation falls below this. Matches Phase 26. |
| `PARTY_D_MIN` | 1.5 | Minimum party separation (Cohen's d) on aligned scale. Quality gate: if d < 1.5 for any biennium post-alignment, flag as suspect. |

## Algorithm

### Step 1: Load Canonical Ideal Points

For each biennium with pipeline results, load canonical ideal points per chamber. Preferred source order:
1. Phase 07b canonical routing (`canonical_irt/canonical_ideal_points_{chamber}.parquet`)
2. Phase 06 canonical routing (same path under `06_irt_2d/`)
3. Phase 05 flat IRT (`ideal_points_{chamber}.csv`)

Record the source and tier for each chamber-session (metadata for the report).

### Step 2: Build Global Roster

Match legislators across all bienniums using the 3-phase matching from Phase 26:
1. OCD person ID (primary, when available)
2. Normalized name (`phase_utils.normalize_name()`)
3. Optional fuzzy matching (disabled by default)

Output: a global roster DataFrame with columns `global_id`, `legislator_slug`, `session`, `chamber`, `party`, `full_name`, `xi_canonical`.

### Step 3: Compute Bridge Matrix

For every pair of sessions (s, t), count shared legislators. Store as a symmetric matrix. Flag pairs below `MIN_BRIDGES`. Report the bridge coverage heatmap.

### Step 4: Simultaneous Affine Alignment

For each chamber independently:

**Decision variables:** For each non-reference session t, two parameters: A_t (scale) and B_t (shift). Reference session: A_ref = 1, B_ref = 0.

**Objective:** Minimize the sum of squared discrepancies across all bridge pairs:

```
L = sum over all (i, s, t) where legislator i served in both s and t:
    w_{s,t} * (A_s * xi_s[i] + B_s  -  A_t * xi_t[i] - B_t)^2
```

Where w_{s,t} is a weight (default: 1.0; optionally inversely proportional to temporal distance to downweight long-range bridges that may have moved).

**Solution:** This is a linear least-squares problem. Stack all bridge observations into a design matrix X and response vector y:

For each bridge observation (legislator i in sessions s and t):
```
xi_s[i] * (column for A_s) + 1 * (column for B_s) - xi_t[i] * (column for A_t) - 1 * (column for B_t) = 0
```

With reference session columns removed (A_ref = 1, B_ref = 0 substituted). Solve via `numpy.linalg.lstsq` or `scipy.optimize.least_squares`.

**Trimmed regression:** After initial fit, compute residuals. Remove observations with |residual| in the top/bottom TRIM_PCT%. Re-fit on the trimmed set.

### Step 5: Transform Scores

Apply the fitted (A_t, B_t) to each legislator's canonical score:

```
xi_common[i, t] = A_t * xi_canonical[i, t] + B_t
```

### Step 6: Bootstrap Uncertainty

Repeat Steps 4-5 with 1,000 bootstrap resamples of the bridge observations. For each resample:
1. Sample bridge legislators with replacement (within each session pair)
2. Re-estimate all (A, B) simultaneously
3. Re-transform all scores

Compute 2.5th and 97.5th percentiles of each legislator's common-space score across bootstrap iterations → 95% confidence interval.

### Step 7: Quality Gates

1. **Pairwise bridge correlation:** For each adjacent pair, Pearson r between aligned scores. Warn if r < CORRELATION_WARN.
2. **Party separation:** Cohen's d on aligned scores per biennium. Warn if d < PARTY_D_MIN.
3. **Sign consistency:** Republicans should be consistently positive (or negative) across all bienniums. A sign flip indicates a linking failure.
4. **Bootstrap stability:** If the 95% CI of any A_t includes zero or is negative, the scale may have collapsed. Flag.

### Step 8: External Validation

Where coverage exists:
- Correlate common-space career averages with Shor-McCarty scores (Phase 14)
- Correlate common-space career averages with DIME CFscores (Phase 18)
- Compare common-space trajectories with Dynamic IRT trajectories (Phase 27)

### Step 9 (Optional): Cross-Chamber Unification

Identify legislators who served in both House and Senate at different times. Use their common-space scores to estimate an affine mapping between the House and Senate scales. This is a supplementary alignment — the per-chamber scales are the primary output.

## Output

| File | Content |
|------|---------|
| `common_space_house.csv` | Every House legislator × biennium on common scale (columns: global_id, legislator_slug, full_name, party, session, chamber, xi_common, xi_common_lo, xi_common_hi, xi_canonical, source_phase, source_tier) |
| `common_space_senate.csv` | Same for Senate |
| `linking_coefficients.csv` | A_t, B_t per session with bootstrap CIs |
| `bridge_coverage.csv` | Pairwise bridge counts |
| `validation.json` | SM/DIME/Dynamic IRT correlations |
| `common_space_report.html` | Full report (see Report Sections below) |

## Report Sections

1. **Key Findings** — summary stats: N legislators, N bienniums, polarization trend
2. **Bridge Coverage** — heatmap of pairwise overlap, weakest links
3. **Linking Coefficients** — scatter of (A, B) per session with bootstrap CIs
4. **Polarization Trajectory** — party means over time on common scale
5. **Party Separation** — Cohen's d per biennium on common scale
6. **Top Movers** — legislators with largest cross-biennium shifts (genuine movers, not linking artifacts)
7. **Career Trajectories** — interactive Plotly line chart for long-serving legislators
8. **External Validation** — SM/DIME correlations, Dynamic IRT comparison
9. **Quality Gates** — pass/fail table per biennium

## Implementation Plan

### Prerequisites

Run the per-biennium pipeline on the 78th-83rd bienniums (KanFocus-only data) through at minimum Phase 05 (flat IRT) with canonical routing. This produces the canonical ideal points needed as input.

### Task Breakdown

| # | Task | Scope | Commit after? |
|---|------|-------|---------------|
| 1 | Create `analysis/28_common_space/` package scaffold | `__init__.py`, empty `common_space.py`, `common_space_data.py`, `common_space_report.py` | Yes |
| 2 | Implement `common_space_data.py` | Global roster construction, canonical score loading, bridge matrix computation, simultaneous alignment solver, bootstrap | Yes |
| 3 | Implement `common_space.py` | CLI entry point (`main()`), argparse (--chamber, --csv, --bootstrap, --reference-session), RunContext integration, quality gates | Yes |
| 4 | Implement `common_space_report.py` | HTML report builder with all 9 sections, Plotly career trajectories, bridge heatmap | Yes |
| 5 | Add Justfile recipe and pipeline integration | `just common-space`, `just cross-pipeline` update, meta-path finder entry | Yes |
| 6 | Write tests | `tests/test_common_space.py` — alignment math, bridge coverage, bootstrap, quality gates, edge cases (single-biennium, no bridges, sign flip) | Yes |
| 7 | Update CLAUDE.md and roadmap | Architecture table, phase list, completed phases table | Yes |
| 8 | Write ADR | `docs/adr/0119-common-space-ideal-points.md` | Yes (final) |

### Dependencies

```
Phase 05 (per-biennium) ──→ Phase 06/07b (canonical routing) ──→ Phase 28 (common space)
                                                                       │
Phase 26 (cross-session) ─── reuses matching logic ───────────────────┘
Phase 27 (dynamic IRT) ──── validation comparison ────────────────────┘
Phase 14 (SM) ──────────── validation comparison ─────────────────────┘
Phase 18 (DIME) ────────── validation comparison ─────────────────────┘
```

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| 78th-83rd not yet pipelined | Phase 28 gracefully skips missing bienniums; works with any subset |
| Horseshoe in early Senates | Canonical routing already handles this; Phase 28 just consumes the corrected scores |
| Weak 84th-85th bridge | Simultaneous alignment uses non-adjacent bridges to bypass; 106 bridges still well above minimum |
| Scale drift undetectable | External validation via SM/DIME; documented as known limitation |
| Alignment fails for a session | Per-session quality gates; exclude and warn rather than propagate bad data |
