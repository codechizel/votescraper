# ADR-0062: DIME/CFscore External Validation (Phase 18)

**Date:** 2026-02-28
**Status:** Accepted

## Context

Phase 14 validates our IRT ideal points against Shor-McCarty scores — the field standard for state legislator ideology. SM covers bienniums 84th-88th (2011-2020) and measures roll-call-based ideology. We wanted a second, completely independent validation source to strengthen the triangulation argument and extend coverage beyond 2020.

DIME (Database on Ideology, Money in Politics, and Elections) provides campaign-finance-based ideology scores (CFscores) for all Kansas state legislators through the 2022 election cycle. CFscores measure a distinct construct — who *funds* a legislator rather than how they *vote* — making the correlation with our IRT scores a genuinely independent validation test.

## Decision

Implement Phase 18 as a sibling phase to Phase 14, reusing shared infrastructure (name normalization, correlation computation, outlier detection) while adding DIME-specific logic:

1. **Coverage:** 84th-89th bienniums (6 bienniums, extending one beyond SM).
2. **Data source:** Pre-downloaded DIME CSV at `data/external/dime_recipients_1979_2024.csv` (ODC-BY license).
3. **Filtering:** Kansas state legislators, incumbents only, minimum 5 unique donors.
4. **Comparison targets:** Both static (career) and dynamic (per-cycle) CFscores.
5. **SM side-by-side:** For bienniums 84th-88th, load Phase 14 correlations for triangulation.
6. **Code reuse:** Import `compute_correlations()`, `compute_intra_party_correlations()`, `identify_outliers()`, and `normalize_our_name()` directly from Phase 14's data module.

## Consequences

**Positive:**
- Two independent validation sources (roll-call + campaign finance) provide the strongest possible credibility argument
- One extra biennium of external validation (89th, 2021-2022) beyond Shor-McCarty
- Shared infrastructure minimizes code duplication
- CLI mirrors Phase 14 for consistency (`--all-sessions`, `--irt-model`, `--run-id`)

**Negative:**
- CFscores discriminate poorly within parties (r ≈ 0.60-0.70), so intra-party correlations will be lower than SM — users must understand this is expected, not a failure
- DIME coverage ends at 2022 cycle — no validation for 90th-91st bienniums
- 144 MB CSV is cached locally (not auto-downloaded — manual placement required)
- Different expected correlation ranges vs SM require separate interpretation guidance
