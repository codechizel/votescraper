# Phase 21: Text-Based Ideal Points — Design Document

## Overview

Derive text-informed ideology scores from bill text embeddings weighted by voting behavior, then compare with IRT ideal points. An embedding-vote approach — not a true TBIP (Vafa et al. 2020).

## Methodology

### Embedding-Vote Approach

1. **Load Phase 18 bill embeddings** — 384-dim bge-small-en-v1.5 via FastEmbed (cached parquet from Phase 18).
2. **Build vote matrix** — legislator × bill: +1 (Yea), -1 (Nay), 0 (absent/not voting).
3. **Compute text profiles** — `profiles = vote_matrix @ embeddings`, normalized by count of non-zero votes per legislator.
4. **PCA** — extract PC1 as the text-derived ideal point.
5. **Sign alignment** — flip text scores if Pearson r with IRT xi_mean is negative (Republicans positive convention).
6. **Validation** — Pearson r, Spearman ρ, Fisher z 95% CI against IRT ideal points.

### Why Not TBIP?

TBIP (Vafa et al. 2020) models `text_i ~ f(author_ideal_point)`, learning author ideal points from how word usage varies with ideology. This requires individual authorship. Kansas bills are ~92% committee-sponsored — only ~27 individual sponsors across ~38 bills in the 91st Legislature. Insufficient for stable authorship-based estimation.

## Parameters

| Parameter | Default | Justification |
|-----------|---------|---------------|
| Embedding model | bge-small-en-v1.5 | 384-dim, good quality-to-size ratio, matches Phase 18 |
| Min votes | 20 | Same as pipeline default; ensures stable profiles |
| Min bills | 5 | Minimum bills with both embeddings and roll calls |
| Min matched | 10 | Minimum legislators for correlation computation |
| PCA components | min(n_legislators, 384, 10) | Up to 10 for scree plot |

## Quality Thresholds

Lower than Phase 14 (SM/DIME) because text scores are further removed from ideology:

| Quality | Phase 21 (text) | Phase 14 (SM/DIME) |
|---------|-----------------|-------------------|
| Strong | r ≥ 0.80 | r ≥ 0.90 |
| Good | r ≥ 0.65 | r ≥ 0.85 |
| Moderate | r ≥ 0.50 | r ≥ 0.70 |
| Weak/Concern | r < 0.50 | r < 0.70 |

Text ideal points are twice removed from ideology: bill text → embedding → vote weighting → PCA. Each step adds noise. Phase 14 compares IRT to other direct ideology measures.

## Assumptions

1. **Linearity**: PCA captures the primary ideological dimension. If ideology is non-linear in embedding space, PC1 may miss it — but standard PCA on vote matrices also assumes linearity and works well.
2. **Vote direction carries content signal**: Voting Yea vs Nay on a bill about, say, tax policy, should push a legislator's text profile in the direction of that bill's embedding. This is a reasonable assumption when bills are substantively diverse.
3. **Embedding quality**: bge-small-en-v1.5 produces meaningful legislative text embeddings. Phase 18 validated this via topic clustering.

## Limitations

- **Not true TBIP** — does not learn a generative model of text conditioned on ideal points.
- **Committee sponsorship** — cannot distinguish sponsor ideology from institutional committee positions.
- **PDF extraction noise** — bill texts come from PDF→text extraction (pdfplumber), which is imperfect.
- **Single embedding model** — results depend on bge-small-en-v1.5 representation choices.
- **PCA linearity** — may miss non-linear ideological structure (though IRT is also essentially linear).

## Downstream Implications

Text ideal points complement but don't replace IRT:
- **High correlation (r ≥ 0.65)**: Confirms that the substantive content of bills legislators support/oppose aligns with their voting ideology.
- **Low correlation (r < 0.50)**: May indicate that voting patterns aren't well-explained by bill content alone (e.g., procedural votes, whip pressure).
- **Outliers**: Legislators whose text and IRT profiles disagree are candidates for deeper investigation — may be voting against type on specific policy areas.

## References

- Vafa, K., Naidu, S., & Blei, D.M. (2020). "Text-Based Ideal Points." ACL.
- Lauderdale, B.E. & Herzog, A. (2016). "Measuring Political Positions from Legislative Speech." Political Analysis 24(3): 374-394.
- Grimmer, J. & Stewart, B.M. (2013). "Text as Data." Political Analysis 21(3): 267-297.
