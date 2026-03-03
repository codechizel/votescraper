# ADR-0089: Model Legislation Detection (BT5)

## Status

Accepted (2026-03-03)

## Context

BT5 is the final item in the Bill Text NLP pipeline (BT1-BT5). Kansas bills already have 384-dim BGE embeddings from Phase 18. Two questions remain:

1. **Which Kansas bills match known ALEC model legislation?** (template matching)
2. **Which Kansas bills appear in neighboring states?** (cross-state diffusion)

Existing literature (LCPS "Copy/Paste/Legislate", Hertel-Fernandez, Burgess et al.) uses n-gram overlap and text similarity to identify model legislation adoption. We combine embedding-based cosine similarity (fast, semantic-aware) with n-gram overlap (precise text evidence) for robust detection.

## Decision

### Data Sources

**ALEC Model Bills**: Scrape from alec.org/model-policy/ (~1,061 model policies in plain HTML). New CLI tool `tallgrass-alec` with cached HTML storage at `data/external/alec/`.

**Neighbor State Bills**: OpenStates API v3 for bill discovery (free tier, no key required). New `OpenStatesAdapter` implements the existing `StateAdapter` Protocol. States: MO, OK, NE, CO (geographic neighbors with strongest diffusion signal).

### Architecture

Three new modules:

| Module | Purpose |
|--------|---------|
| `src/tallgrass/alec/` | ALEC corpus scraper (models, scraper, output, cli) |
| `src/tallgrass/text/openstates.py` | OpenStates multi-state adapter |
| `analysis/20_model_legislation/` | Phase 20 analysis (data, main, report) |

### Similarity Method

**Primary**: Cosine similarity on BGE embeddings (same model as Phase 18 — embeddings directly comparable). Three-tier classification:

| Tier | Threshold | Interpretation |
|------|-----------|---------------|
| Near-identical | >= 0.95 | Likely direct copy or minimal adaptation |
| Strong match | >= 0.85 | Adapted from same source |
| Related | >= 0.70 | Similar policy area, warrants investigation |

**Secondary**: 5-gram word overlap for strong matches (>= 0.85). Confirms genuine text reuse vs. topical similarity. Overlap >= 20% indicates text sharing.

### Why Not Smith-Waterman

Local alignment (Smith-Waterman) would identify exact shared passages but is O(n*m) per pair — prohibitively expensive for ~400 Kansas bills x ~1,000 ALEC bills x ~1,000 neighbor bills. Embedding similarity is O(n+m) per corpus pair. Future extension could apply local alignment only to the top 20-50 matches identified by embeddings.

### Why Not True TBIP

We already tried text-based ideal points in Phase 18b (ADR-0086) and found limitations due to ~92% committee sponsorship. Model legislation detection is a different question — it identifies provenance, not ideology. The embedding approach is appropriate for both.

## Consequences

- `just alec` builds the ALEC corpus (run once, cached)
- `just model-legislation` runs Phase 20 analysis
- OpenStates adapter enables cross-state bill text download for any state
- Phase 20 is standalone (not in pipeline) — requires BT1 bill texts + ALEC corpus
- High-similarity matches can feed into synthesis narratives (future integration)
- The OpenStates adapter could be reused for additional states beyond the initial 4

## Alternatives Considered

1. **N-gram only** (Copy/Paste/Legislate approach): Precise but misses semantically similar bills with different wording. Embeddings catch paraphrased model legislation.
2. **TF-IDF cosine**: Lower quality than dense embeddings for legislative text. BGE embeddings already available from Phase 18.
3. **LegisLATOR/BillTracker**: Existing tools but closed-source or academic-only. Our approach is transparent and reproducible.
4. **Full text alignment**: Too slow for initial corpus. Reserved for future "detail view" on top matches.
