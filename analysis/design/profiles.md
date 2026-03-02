# Profiles Design Choices

## Assumptions

1. **All 8 upstream phases have been run.** Profiles reads unified legislator DataFrames via `build_legislator_df()` from synthesis, which requires IRT ideal points as the base table.

2. **IRT bill_params are available.** Bill type breakdown requires `bill_params_{chamber}.parquet` from the IRT phase. If missing, the breakdown is skipped.

3. **Raw vote and rollcall CSVs are available.** Defection analysis and voting neighbor computation work from the original vote-level data, not from aggregated parquets.

4. **Detection thresholds from synthesis_detect are appropriate.** Profiles reuses `detect_all()` without modifying detection logic. If synthesis detection thresholds change (ADR-0008), profiles inherits the changes automatically.

## Parameters & Constants

| Parameter | Value | Justification | Location |
|-----------|-------|---------------|----------|
| `HIGH_DISC_THRESHOLD` | 1.5 | Matches IRT convention (ADR-0006). Bills with \|beta_mean\| > 1.5 are highly discriminating — these are the votes where ideology matters most. | `profiles_data.py` |
| `LOW_DISC_THRESHOLD` | 0.5 | Bills with \|beta_mean\| < 0.5 are low-discrimination — routine/bipartisan. | `profiles_data.py` |
| `MIN_BILLS_PER_TIER` | 3 | Need at least 3 bills in each tier to compute a meaningful Yea rate. Prevents division-by-near-zero on tiny samples. | `profiles_data.py` |
| `MAX_PROFILE_TARGETS` | 8 | Cap on total profiled legislators. Prevents excessively long reports. With 6 detections (2 chambers × 3 types) plus user extras, 8 is a reasonable limit. | `profiles_data.py` |
| Defection n (default) | 15 | Maximum defection bills shown per legislator. Sorted by closeness of party margin. | `profiles_data.py:find_defection_bills` |
| Neighbor n (default) | 5 | Top 5 most similar and 5 most different legislators. | `profiles_data.py:find_voting_neighbors` |
| Surprising n (default) | 10 | Top 10 most surprising votes per legislator. | `profiles_data.py:find_legislator_surprising_votes` |

## Methodological Choices

### 1. Scorecard uses only 0-1 normalized metrics

**Decision:** Show 6 metrics that are already on a 0-1 scale: `xi_mean_percentile`, `unity_score`, `loyalty_rate`, `maverick_rate`, `betweenness_percentile`, `accuracy`. Exclude raw dimensional values (IRT xi_mean, PCA, UMAP).

**Alternatives:** Include all available metrics. This was attempted and produced unreadable charts — Tyson's PC2 = -24.81 compressed all 0-1 metrics to invisibility. The synthesis profile card (ADR-0008) uses the same 0-1 normalization approach.

**Impact:** The scorecard is immediately interpretable. Dimensional information is available in the position-in-context plot and the synthesis report's dashboard scatter.

### 2. Bill type classification uses IRT discrimination

**Decision:** Use the IRT bill discrimination parameter (beta_mean) to classify bills into "partisan" (high-disc) and "routine" (low-disc) tiers, then compare Yea rates.

**Alternatives:** (a) Use party margin as a proxy for bill contentiousness. (b) Use vote type (final action vs procedural). IRT discrimination was chosen because it integrates both ideological content and party structure into a single parameter — a bill can have high discrimination even if the vote margin is lopsided.

### 3. Defections sorted by party margin closeness

**Decision:** Show defection bills where the party margin was tightest first. A defection on an 80-20 party vote is less interesting than a defection on a 55-45 vote.

**Alternatives:** Sort by date, bill number, or raw vote count. Margin closeness was chosen because it surfaces the defections that reveal the most about the legislator's decision-making.

### 4. Voting neighbors use simple agreement rate

**Decision:** Pairwise agreement rate (fraction of shared votes where both legislators voted the same way) computed via a pivot to wide format.

**Alternatives:** (a) Cohen's Kappa (used in network phase). (b) Cosine similarity on the vote matrix (used in UMAP). Simple agreement was chosen for interpretability — "89% agreement" is immediately meaningful to a nontechnical reader. Kappa adjusts for chance agreement, which is important for network construction but adds cognitive load in a profile card.

### 5. Chamber normalization in prep_votes_long

**Decision:** Normalize the `chamber` column to lowercase via `str.to_lowercase()` during vote data preparation.

**Alternatives:** Normalize upstream in the scraper. This was rejected because it would change the scraper's output format (CLAUDE.md rule: "ETL is separate from analysis").

### 6. Name-based legislator lookup

**Decision:** Add `--names` flag as a natural-language alternative to `--slugs`. Multi-stage matching: exact full name (case-insensitive) → last-name-only → first-name disambiguation within last-name matches.

**Alternatives:** (a) Fuzzy matching (Levenshtein distance). Rejected — adds complexity and a dependency for a small set of well-known names. (b) Interactive selection from ambiguous matches. Rejected — profiles runs non-interactively in batch pipelines. Instead, ambiguous matches include all candidates and print a clear message.

**Reuse:** Name normalization uses `normalize_name()` from `cross_session_data` (lowercases, strips leadership suffixes). No new normalization code.

### 7. Sponsorship analysis from sponsor_slugs

**Decision:** Add `compute_sponsorship_stats()` in `profiles_data.py` to identify bills where the target legislator appears in the `sponsor_slugs` semicolon-split list. Mark the first slug as primary sponsor. Display a sponsorship section in the per-legislator report between the position figure and defections table.

**Implementation:** Pure function: filter rollcalls where slug appears in the split list, derive `is_primary` from position, return DataFrame with bill_number, short_title, motion, passed, is_primary. Returns `None` if `sponsor_slugs` column is absent. Report section shows summary text ("Sponsored N bills, M as primary, passage rate X%") plus a table.

**Alternatives:** (a) Use the text `sponsor` column for name matching instead of `sponsor_slugs`. Rejected because slug matching is exact and deterministic, while name matching requires fuzzy logic and fails on committee sponsors. (b) Build a cosponsorship network (GovTrack-style SVD). Deferred — requires all-legislator sponsorship matrix, not just per-target lookup.

### 8. Defection table enrichment with sponsor

**Decision:** Add the `sponsor` column (human-readable text from rollcalls, e.g. "Senator Tyson") to the defection bills output when present. No slug resolution needed — the raw text is already display-ready.

**Graceful degradation:** Both features silently skip when `sponsor_slugs` or `sponsor` columns are missing (pre-89th, pre-rescrape data). ADR-0081.

## Downstream Implications

- **Plot filenames** use the full slug minus the `rep_`/`sen_` prefix (e.g., `scorecard_schreiber_mark_1.png`). These names change across sessions as different legislators are detected.
- **Report section count** varies from ~30-40 depending on how many legislators are detected and which have defection/surprising vote data.
- **No new upstream dependencies.** Profiles is a pure consumer of existing analysis outputs. Adding or removing an upstream phase does not require changes to profiles (missing data is handled gracefully).
