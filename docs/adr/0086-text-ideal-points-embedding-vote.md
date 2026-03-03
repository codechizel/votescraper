# ADR-0086: Text-Based Ideal Points via Embedding-Vote Approach

## Status

Accepted

## Date

2026-03-03

## Context

The BT3 roadmap item planned TBIP (Vafa et al. 2020) for text-based ideal points. TBIP models word usage conditioned on author ideal points, learning ideology from how language varies across authors. However, Kansas bills are ~92% committee-sponsored — only ~27 individual sponsors across ~38 bills in the 91st Legislature. Classic TBIP requires individual authorship to associate text variation with ideology. With so few individual authors, TBIP estimates would be unreliable.

We already have 384-dimensional bill embeddings from Phase 18 (bge-small-en-v1.5 via FastEmbed) and full vote data. The question: can we derive text-informed ideology scores using what we have?

## Decision

Implement an **embedding-vote approach** as Phase 18b:

1. Load Phase 18 bill embeddings (384-dim, cached).
2. Build a vote matrix: legislator × bill (+1 Yea, -1 Nay, 0 absent).
3. Multiply: `text_profiles = vote_matrix @ embeddings`, normalized by non-absent vote count.
4. PCA on text profiles → PC1 = text-derived ideal point.
5. Align sign with IRT convention (Republicans positive).
6. Validate against IRT ideal points (flat and hierarchical).

Quality thresholds are lower than Phase 14 external validation (strong ≥ 0.80 vs 0.90) because text scores are further removed from direct ideology measurement.

## Consequences

### Positive

- Uses existing infrastructure (Phase 18 embeddings, IRT results) — no new dependencies.
- Provides a complementary perspective: IRT measures vote direction, text ideal points measure what legislation is about.
- Interpretable: high correlation means substantive bill content aligns with voting ideology.
- Fast: no MCMC, no API calls, no model training — just matrix multiplication + PCA.

### Negative

- Not a true generative text model — cannot estimate ideal points for legislators who don't vote.
- PC1 captures the dominant dimension, which in a partisan legislature is almost always party. May not add much beyond what IRT already captures.
- Depends on embedding quality (PDF extraction noise, model choice).

### Neutral

- Standalone phase (not in pipeline) — requires bill text data from BT1 (`just text`) plus IRT results.
- Available via `just tbip` or `uv run python analysis/18b_tbip/tbip.py`.
