# ADR-0087: Issue-Specific Ideal Points via Topic-Stratified Flat IRT

## Status

Accepted

## Date

2026-03-03

## Context

Phase 04 produces a single overall 1D Bayesian IRT ideal point per legislator. But legislators may be conservative on fiscal issues while moderate on social policy. BT4 from the roadmap plans "issue-specific ideal points" — per-topic ideology scores.

Two approaches were considered:

1. **`issueirt` R package** (Shin 2024): Estimates 2D ideal points with Bingham directional priors per topic. Principled but: Phase 06 proved Kansas voting is fundamentally 1D (Dim 2 is noise); `issueirt` is GitHub-only (4 stars, 0 forks, pre-1.0), requires rstan (fragile R dependency), author moved to industry.

2. **Topic-stratified flat IRT**: Run the existing Phase 04 `build_irt_graph()` / `build_and_sample()` on per-topic vote subsets from Phase 18 BERTopic/CAP assignments. Zero new model code, zero new dependencies, battle-tested infrastructure.

## Decision

Implement **topic-stratified flat IRT** as Phase 19.

For each eligible topic (≥ 10 bills with roll calls):
1. Subset the EDA vote matrix to vote_ids belonging to that topic.
2. Filter legislators with ≥ 5 non-null votes in the topic.
3. Run standard 2PL IRT with anchor constraints (same model as Phase 04).
4. Sign-align per-topic ideal points against full-model IRT.

MCMC budget: 1000 draws / 1000 tune / 2 chains (smaller models converge faster). Convergence thresholds relaxed: R-hat < 1.05, ESS > 200 (vs Phase 04's 1.01 / 400).

Anchor strategy: try full model's PCA-derived anchors first. If missing from topic subset, fall back to per-topic PCA. Post-hoc sign alignment regardless.

Two taxonomies supported: BERTopic (data-driven) and CAP (standardized, cross-state comparable). Each runs independently.

No NetCDF per topic — save parquet summaries only to avoid GB of disk for 20+ per-topic posteriors.

## Consequences

### Positive

- Reuses entire Phase 04 infrastructure — zero new model code, zero new dependencies.
- Answers a meaningful substantive question: which policy areas have distinct ideological alignment?
- Cross-topic correlation matrix and outlier detection provide new analytic value.
- Two taxonomies give both data-driven and standardized perspectives.

### Negative

- Per-topic subsets are small (10-30 bills) → wider credible intervals than full model.
- No borrowing of strength across topics (each model is independent).
- BERTopic topics may not map cleanly to named policy areas.
- CAP classifications require Claude API access.

### Neutral

- Standalone phase (not in pipeline) — requires Phase 18 topics + Phase 04 IRT results.
- Graceful degradation: convergence failure on one topic logs a warning, doesn't abort.
- Available via `just issue-irt` or `uv run python analysis/22_issue_irt/issue_irt.py`.
