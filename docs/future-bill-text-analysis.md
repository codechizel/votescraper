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
4. **Text-based ideal points (TBIP)** — Currently rejected (no bill text). With full text available, TBIP becomes viable: jointly estimates legislator positions and bill topics from text + votes. See `docs/method-evaluation.md`.
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

## Open Questions

1. **Bill text availability** — Does kslegislature.gov serve full bill text in a scrapeable format? HTML, PDF, both? Does it vary by session?
2. **Text scope** — Full enrolled text, or just the introduced version? Amendments create multiple versions per bill.
3. **Corpus size** — How much text per biennium? Impacts model choice (LDA vs BERTopic) and embedding storage.
4. **Integration point** — Does bill text become a new scraper output (a 4th CSV?), or a separate data pipeline that joins on `bill_number`?

---

## Related

- Phase 6+ (NLP Bill Text Features): Uses `short_title` only, NMF topics. See `analysis/06_prediction/`.
- TBIP rejection: `docs/method-evaluation.md` — revisit once full text is available.
- Roadmap: `docs/roadmap.md` — this phase is not yet scheduled.
