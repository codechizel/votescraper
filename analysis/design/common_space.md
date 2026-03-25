# Common Space Ideal Points Design Choices

**Script:** `analysis/28_common_space/common_space.py`
**Data module:** `analysis/28_common_space/common_space_data.py`
**Report:** `analysis/28_common_space/common_space_report.py`
**Deep dive:** `docs/common-space-ideal-points.md`

## Purpose

Produce a single ideological scale spanning all 14 bienniums (78th-91st, 1999-2026) so that any legislator can be compared to any other, regardless of whether they overlapped. The approach uses simultaneous affine alignment of canonical ideal points via bridge legislators.

## Assumptions

1. **Canonical ideal points are the input.** Horseshoe-corrected scores from the routing system (ADR-0109). The common space phase does not estimate ideal points — it links existing ones.

2. **Chambers are aligned separately, then linked.** House and Senate have different bills and different ideal point scales. Each chamber is aligned independently across time (steps 2-5), then the two chambers are linked via an affine transform estimated from 54 chamber-switcher bridge legislators. The unified scale uses House as the reference; Senate scores are mapped via `xi_unified = A * xi_senate + B`. Career scores are computed both per-chamber (for within-chamber analysis) and unified (one number per legislator across both chambers).

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

Output: a global roster DataFrame with columns `person_key`, `name_norm`, `legislator_slug`, `session`, `chamber`, `party`, `full_name`, `xi_canonical`. Identity resolution uses slug-based `person_key` (strips chamber prefix: `rep_smith_greg_1` → `smith_greg_1`), not `name_norm`, because name normalization is fragile across middle initials, nicknames, and punctuation (e.g., "J.R. Claeys" vs "J. R. Claeys" vs "Joseph Claeys"). A small override table in `_PERSON_KEY_OVERRIDES` handles known slug encoding variants (8 entries). Career scores are computed per chamber and unified (cross-chamber); 59 of 658 unique legislators served in both chambers.

### Step 3: Compute Bridge Matrix

For every pair of sessions (s, t), count shared legislators. Store as a symmetric matrix. Flag pairs below `MIN_BRIDGES`. Report the bridge coverage heatmap.

### Step 4: Simultaneous Affine Alignment

For each chamber independently:

**Method: Pairwise chain linking** (Battauz 2023, GLS 1999).

For each pair of adjacent sessions (t, t+1), estimate an affine transformation using trimmed OLS on bridge legislators:

```
xi_{t+1}[i] = A_{t→t+1} * xi_t[i] + B_{t→t+1}   (for all bridge legislators i)
```

Trimmed regression: fit, remove the TRIM_PCT% most extreme residuals, re-fit.

**Chain composition:** To map session t to the reference (91st), compose the pairwise links:

```
A_total = A_{n-1→n} * A_{n-2→n-1} * ... * A_{t→t+1}
B_total follows the recursion: B_total = A_{k→k+1} * B_prev + B_{k→k+1}
```

**Why not simultaneous all-pairs?** The initial implementation used simultaneous least-squares on all bridge pairs. This produced degenerate A coefficients (0.05-0.19) that collapsed non-reference sessions to a narrow range. The solver minimized total error by shrinking A toward zero — mathematically valid but substantively wrong. Pairwise chaining avoids this because each link has a well-conditioned 1D regression. See ADR-0120 revision notes.

Non-adjacent bridges (legislators who served in sessions 2+ apart) are used for **validation**, not estimation — they cross-check the chain.

### Step 5: Transform Scores

Apply the chained (A_total, B_total) to each legislator's canonical score:

```
xi_common[i, t] = A_total[t] * xi_canonical[i, t] + B_total[t]
```

### Step 6: Uncertainty Propagation (Delta Method)

Two independent sources combined in quadrature:

1. **IRT posterior uncertainty:** `sd_irt = |A_total| * xi_sd`
2. **Chain alignment uncertainty:** Propagate Var(A), Var(B), Cov(A,B) through the chain via the delta method (Battauz 2015):
   `Var(xi_common) = A²·Var(xi) + xi²·Var(A) + Var(B) + 2·xi·Cov(A,B)`

Per-link bootstrap (N=1000) estimates Var(A), Var(B), Cov(A,B) at each step. Chain propagation multiplies the covariance matrices through the composition. Earlier sessions naturally get wider CIs — honest uncertainty that grows with chain length.

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
| `common_space_house.csv` | Every House legislator × biennium on common scale (columns: name_norm, legislator_slug, full_name, party, session, chamber, xi_common, xi_common_sd, xi_canonical) |
| `common_space_senate.csv` | Same for Senate |
| `common_space_unified.csv` | All legislators on unified cross-chamber scale (xi_unified = House scale; Senate mapped via affine transform from chamber-switchers) |
| `career_scores_house.csv` | Per-chamber career scores — House (RE meta-analysis) |
| `career_scores_senate.csv` | Per-chamber career scores — Senate (RE meta-analysis) |
| `career_scores_unified.csv` | Unified career scores — one number per legislator across both chambers, 708 unique legislators |
| `linking_coefficients.csv` | A_t, B_t per session with bootstrap CIs |
| `bridge_coverage.csv` | Pairwise bridge counts |
| `validation.json` | Quality gates and configuration |
| `common_space_report.html` | Combined report (all sections, for backward compat) |
| `common_space_overview_report.html` | Overview: key findings, bridge coverage, polarization, quality gates |
| `common_space_house_report.html` | House: ideal points table, linking coefficients, career trajectories/scores |
| `common_space_senate_report.html` | Senate: same as House |
| `common_space_unified_report.html` | Unified career scores: one number per legislator across both chambers |

## Reports

Split into four focused reports for easier navigation, plus a combined report:

### Overview (`common_space_overview_report.html`)
1. **Key Findings** — summary stats: N legislators, N bienniums, polarization trend
2. **Bridge Coverage** — heatmap of pairwise overlap, weakest links
3. **Polarization Trajectory** — party means over time on common scale (both chambers)
4. **Quality Gates** — pass/fail table per biennium (both chambers)

### Per-Chamber (`common_space_{house,senate}_report.html`)
1. **Ideal Points Table** — searchable/sortable: all legislator-sessions on common scale
2. **Linking Coefficients** — scatter of (A, B) per session with bootstrap CIs
3. **Party Separation** — Cohen's d per biennium on common scale
4. **Top Movers** — legislators with largest cross-biennium shifts
5. **Career Trajectories** — interactive Plotly line chart for long-serving legislators
6. **Career Scores** — one number per legislator (RE meta-analysis, within-chamber)
7. **Career vs Recent** — scatter comparing career-fixed score to most recent session

### Unified (`common_space_unified_report.html`)
1. **Unified Career Scores** — one number per legislator across both chambers

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
