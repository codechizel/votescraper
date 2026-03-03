# Phase 19: Issue-Specific Ideal Points — Design Document

## Overview

Estimate per-topic Bayesian IRT ideal points to answer: "How conservative is each legislator on education vs healthcare vs taxes?" Runs the Phase 04 flat IRT model on topic-stratified vote subsets.

## Methodology

### Topic-Stratified Flat IRT

1. **Load topic assignments** from Phase 18 — BERTopic (data-driven) and/or CAP (standardized).
2. **For each eligible topic** (≥ `MIN_BILLS_PER_TOPIC` bills with roll calls):
   a. Subset the EDA wide vote matrix to `vote_id`s for bills in this topic.
   b. Filter legislators with ≥ `MIN_VOTES_IN_TOPIC` non-null votes in the subset.
   c. Run standard Phase 04 2PL IRT: `build_irt_graph()` → `build_and_sample()` via nutpie.
   d. Sign-align per-topic xi against full-model IRT (correlation-based flip).
3. **Cross-topic analysis**: assemble legislator × topic matrix, compute pairwise correlations, detect outliers.

### Why Not `issueirt`?

The `issueirt` R package (Shin 2024) estimates 2D ideal points with Bingham directional priors per topic. However:
- Phase 06 proved Kansas voting is fundamentally 1D — Dim 2 is noise.
- `issueirt` is GitHub-only (4 stars, 0 forks), pre-1.0, requires rstan.
- Author moved to industry; uncertain maintenance.

Reusing Phase 04 infrastructure means zero new model code and zero new dependencies.

## Parameters

| Parameter | Default | Justification |
|-----------|---------|---------------|
| Min bills per topic | 10 | Below this, posterior is too diffuse for meaningful estimation |
| Min legislators per topic | 10 | Small N → unreliable anchor selection and ideal points |
| Min votes in topic | 5 | Per-legislator minimum; lower than Phase 04's 20 (smaller subsets) |
| MCMC draws | 1000 | Sufficient for per-topic models (smaller than full Phase 04) |
| MCMC tune | 1000 | Standard nutpie tuning budget |
| MCMC chains | 2 | Enough for R-hat; 4 chains would be excessive for 10-20 topic models |
| R-hat threshold | 1.05 | Relaxed from Phase 04's 1.01 — smaller models are noisier |
| ESS threshold | 200 | Relaxed from Phase 04's 400 — smaller models |

## Quality Thresholds

Lower than Phase 04 — per-topic subsets are smaller and noisier:

| Quality | Phase 19 (per-topic) | Phase 04 (full model) |
|---------|---------------------|----------------------|
| Strong | r ≥ 0.80 | r ≥ 0.90 |
| Good | r ≥ 0.60 | r ≥ 0.85 |
| Moderate | r ≥ 0.40 | r ≥ 0.70 |
| Weak | r < 0.40 | r < 0.70 |

## Anchor Strategy

1. **Primary**: Use full model's PCA-derived anchors (conservative = highest PC1, liberal = lowest PC1) if both are present in the per-topic subset.
2. **Fallback**: Compute per-topic PCA on the subsetted vote matrix, select topic-specific anchors.
3. **Post-hoc**: Regardless of anchor source, sign-align per-topic ideal points against full IRT via Pearson r. If r < 0, negate all xi columns.

## Two Taxonomies

- **BERTopic** (data-driven): FastEmbed + HDBSCAN clustering from Phase 18. Topics may not map cleanly to named policy areas.
- **CAP** (standardized): Comparative Agendas Project 20-category classification via Claude API. Cross-state comparable but requires API key.

Each taxonomy runs independently. Both can be requested with `--taxonomy both`.

## Assumptions

1. **Per-topic subsets are large enough**: With 10+ bills and 10+ legislators, 2PL IRT can produce meaningful estimates. Below these thresholds, we skip.
2. **1D per topic**: Each topic's voting is primarily 1D. If a topic has cross-cutting 2D structure, PC1 captures only the dominant dimension.
3. **Full-model anchors transfer**: Legislators who are extreme overall tend to be extreme per-topic. Anchor stability checking validates this assumption.
4. **Noise from small N**: Per-topic credible intervals will be wider than full-model CIs. This is expected, not a bug.

## Limitations

- **Small N per topic**: Kansas has ~40 House and ~40 Senate members. After filtering, per-topic samples may be 15-30 legislators. Posterior uncertainty is large.
- **Topic quality**: BERTopic topics depend on bill text extraction (PDF → text) and embedding model quality.
- **Not hierarchical**: Each topic's IRT is independent. No borrowing of strength across topics. A hierarchical multi-topic model could improve estimates but adds substantial complexity.
- **CAP requires API**: CAP classifications need Claude API access and cost per bill.

## Downstream Implications

- Cross-topic correlation matrix reveals which policy areas are ideologically aligned vs cross-cutting.
- Topic outliers identify cross-pressured legislators — potentially interesting for profile deep-dives.
- Low per-topic r with full IRT = that policy area has distinct ideological alignment (not captured by overall 1D model).

## References

- Shin, M. (2024). "issueirt: Issue-Specific Ideal Point Estimation." R package (GitHub).
- Clinton, J. & Lapinski, J. (2006). "Measuring Legislative Accomplishment, 1877-1994." American Journal of Political Science 50(1): 232-249.
- Lauderdale, B. & Clark, T. (2014). "Scaling Politically Meaningful Dimensions Using Texts and Votes." American Journal of Political Science 58(3): 754-771.
