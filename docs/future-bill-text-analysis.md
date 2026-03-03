# Future: Bill Text Analysis

> **Superseded by:** `docs/bill-text-nlp-deep-dive.md` — comprehensive survey of tools, research, data sources, and integration strategy.

Original notes on tools, techniques, and resources for a future phase that incorporates full bill text into the analysis pipeline. Currently the scraper captures vote data but not bill text; Phase 6+ uses only `short_title` via NMF topic extraction.

**Last updated:** 2026-02-25 (see deep dive for current state)

---

## Goal

Scrape or retrieve full bill text from kslegislature.gov, then use it to:

1. **Topic modeling** — Categorize bills by policy area (education, tax, criminal justice, etc.) and cross-reference with voting patterns. Which policy areas split the caucus? Which are rubber-stamps?
2. **Text-informed vote prediction** — Add bill content features to the prediction model (currently AUC=0.98 on IRT features alone). Full text should improve predictions on novel bills where legislator priors are weak.
3. **Bill similarity** — Measure how textually similar bills are to each other. Do legislators vote consistently on similar legislation, or does framing matter?
4. **Text-based ideal points** — Classic TBIP rejected (~92% committee sponsorship). **Implemented as Phase 18b** (ADR-0086): embedding-vote approach using vote-weighted bill embeddings + PCA. Validates against IRT.
5. **Interactive exploration** — "Show me how Sen. X voted on bills about Y" requires a search/retrieval layer over the bill corpus.

---

## Tools & Libraries Under Consideration

### bb25 — Bayesian BM25 Hybrid Search

- **Repo:** <https://github.com/instructkr/bb25>
- **What it is:** Rust-backed Python library that implements BM25 (keyword ranking) with Bayesian probability calibration. Converts raw BM25 scores into calibrated probabilities that blend cleanly with vector embeddings for hybrid search.
- **Performance:** 0.9149 NDCG@10 on SQuAD (vs 0.9051 for standard BM25+Dense fusion).
- **How it could be used here:**
  - **Keyword search over bill corpus** — "find all bills mentioning 'property tax'" ranked by relevance.
  - **Hybrid search** — Combine keyword matching (BM25) with semantic similarity (sentence-transformer embeddings) for queries like "education funding" that should also match "school finance" and "K-12 appropriations."
  - **Retrieval layer for interactive tools** — Power a search interface where users query bills by topic and see associated voting patterns.
  - **Feature pipeline** — Retrieve the top-N most similar past bills for each new bill, then use their voting outcomes as prediction features.
- **Fit:** bb25 is a retrieval/ranking tool, not an analysis tool. It would slot in as the search layer, sitting between the raw bill text corpus and the downstream analysis phases. Not the foundation of the text analysis pipeline, but a useful component — especially for any interactive or exploratory interface.
- **Production version:** The research repo links to [cognica-io/bayesian-bm25](https://github.com/cognica-io/bayesian-bm25) as the production-ready implementation.

### Topic Modeling (not yet evaluated)

- **BERTopic** — Transformer-based topic modeling. Would replace or complement the NMF approach currently used on short titles.
- **LDA** — Classical approach, lighter weight. May be sufficient for structured legislative text.

### Text Embeddings (not yet evaluated)

- **sentence-transformers** — For semantic similarity between bills.
- Domain-specific legal/legislative embeddings may exist; needs research.

---

## Open Questions — Answered

All four original questions have been resolved by BT1 implementation (2026-03-02, ADR-0083):

1. **Bill text availability** — PDF only, deterministic URLs. Pattern: `{li_prefix}/measures/documents/{code}_{version}.pdf`
2. **Text scope** — Introduced version + supplemental notes in Phase 1. Committee-amended and enrolled versions deferred.
3. **Corpus size** — Varies by biennium. Extracted via `pdfplumber` into `bill_texts.csv`.
4. **Integration point** — Separate CLI (`tallgrass-text`) producing a 5th CSV that joins on `bill_number`. Not integrated into the vote scraper.

---

## Related

- **BT1 implementation**: `src/tallgrass/text/` subpackage — `StateAdapter` Protocol, `KansasAdapter`, `pdfplumber` extraction (ADR-0083)
- Phase 6+ (NLP Bill Text Features): Uses `short_title` only, NMF topics. See `analysis/08_prediction/`.
- Bill text NLP deep dive: `docs/bill-text-nlp-deep-dive.md` — comprehensive survey (supersedes this document)
- Text-based ideal points: Phase 18b (ADR-0086) — embedding-vote approach, replacing classic TBIP
- Roadmap: `docs/roadmap.md` — BT1 complete, BT2-BT5 planned
