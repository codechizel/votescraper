# ADR-0084: Bill Text Analysis — Phase 18 (BERTopic + CAP Classification)

**Date:** 2026-03-02
**Status:** Accepted

## Context

BT1 (ADR-0083) shipped: `tallgrass-text` CLI downloads Kansas bill PDFs and extracts text via `pdfplumber` into `bill_texts.csv`. Phase 08 uses NMF with K=6 fixed topics on 5-15 word `short_title`. Full bill text (1-100 pages) enables dramatically richer analysis: automatic topic discovery, validated policy classification, bill similarity, and cross-referencing with voting patterns.

Key design decisions:

1. **Embedding engine**: PyTorch-based sentence-transformers (~2 GB) vs ONNX-based FastEmbed (~50-100 MB). Legislative text doesn't need the largest models — a 384-dim BGE model captures policy semantics well.
2. **Topic modeling**: Fixed-K NMF (current, Phase 08) vs BERTopic (automatic K via density-based clustering). BERTopic on full text produces richer, more interpretable topics than NMF on short titles.
3. **Policy classification**: Unsupervised topics lack standardized labels for cross-study comparison. The Comparative Agendas Project (CAP) provides a validated 20-category taxonomy used across 1.36M+ state bills.
4. **API dependency**: CAP classification via LLM requires an API key and costs money (~$7-13/biennium). Should be optional, not required.

## Decision

### Hybrid architecture: FastEmbed + Claude API (optional)

Phase 18 (`analysis/20_bill_text/`) implements bill text NLP as four modules:

| Module | Purpose |
|--------|---------|
| `bill_text_data.py` | Data loading, text preprocessing, FastEmbed embedding with parquet cache |
| `bill_text_classify.py` | CAP classification via Claude API (optional) |
| `bill_text_report.py` | 13-section HTML report builder |
| `bill_text.py` | Main orchestration + CLI |

### FastEmbed over sentence-transformers

`fastembed` (ONNX Runtime) replaces `sentence-transformers` (PyTorch) for text embeddings:

- **BAAI/bge-small-en-v1.5**: 384-dim, ~130 MB ONNX model (vs ~2 GB PyTorch stack)
- FastEmbed downloads and caches ONNX weights automatically
- BERTopic natively supports FastEmbed via `embedding_model` parameter (since v0.17.1)
- No PyTorch = faster install, smaller footprint, no GPU/MPS complications

### BERTopic with reproducible settings

BERTopic pipeline: pre-computed embeddings → UMAP (10D) → HDBSCAN → c-TF-IDF.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| UMAP `n_components` | 10 | Standard for BERTopic clustering stage |
| UMAP `n_neighbors` | 15 | Default, good for ~800 bill corpus |
| UMAP `min_dist` | 0.0 | Tight clusters for HDBSCAN |
| UMAP `random_state` | 42 | Reproducibility |
| HDBSCAN `min_cluster_size` | 15 | CLI-configurable (`--min-cluster-size`) |
| HDBSCAN `min_samples` | 5 | Conservative noise detection |
| `nr_topics` | `"auto"` | Let HDBSCAN decide K |

UMAP `random_state=42` ensures determinism. HDBSCAN is deterministic given the same UMAP output.

### CAP classification via Claude API (optional)

20-category Comparative Agendas Project taxonomy, classified by Claude Sonnet:

- **Graceful degradation**: if `ANTHROPIC_API_KEY` not set, CAP sections skipped entirely. Phase works without any API calls.
- **Content-hash caching**: SHA-256 of (bill text + model name) → JSON cache. Subsequent runs skip API calls for already-classified bills. Cache invalidates on text change or model upgrade.
- **Lazy client creation**: `anthropic` module imported only when an uncached bill is encountered. Cached-only runs don't require the `anthropic` package at all.
- **Batch API support**: `--batch` flag uses `messages.batches.create()` for 50% cost discount (processes in <1 hour).
- **Cost**: ~800 bills × ~5K tokens in + ~100 tokens out ≈ $13 standard, ~$7 with Batch API. Per biennium.

CAP categories match the standard 20 major topics (macroeconomics, civil_rights, health, agriculture, labor, education, environment, energy, immigration, transportation, law_crime, social_welfare, housing, banking, defense, technology, foreign_trade, government_operations, public_lands, cultural_policy).

### Vote cross-reference: Rice index per topic

Join path: `bill_topics` → `rollcalls` (on `bill_number`) → `votes` (on `vote_id`) → `legislators` (on `legislator_slug`).

Key metric: **caucus-splitting score** = 1 - Rice(majority party). Topics where the majority party has low cohesion are the most analytically interesting.

### Text preprocessing for embedding quality

Boilerplate stripping before embedding:
- Enacting clauses ("Be it enacted by the Legislature...")
- Severability sections
- Effective date sections
- K.S.A. references normalized to `STATUTE_REF` token (reduces vocabulary noise)
- Section headers simplified
- Page numbers removed
- Whitespace collapsed
- Truncated to ~8000 chars (~2000 tokens, within BGE model's 512-token limit)

Supplemental notes preferred over introduced text when both exist (shorter, plain-English summaries produce better embeddings).

### Standalone, not in pipeline

Phase 18 runs with `just text-analysis` — not included in `just pipeline`. Reasons:
- Requires BT1 bill text data (separate acquisition step)
- CAP classification requires optional API key + costs money
- BERTopic/FastEmbed add ~150 MB to the dependency footprint

## Consequences

### Benefits

- **No PyTorch**: FastEmbed (ONNX) keeps the total new dependency footprint under 150 MB, vs ~2 GB for sentence-transformers + PyTorch.
- **Automatic topic discovery**: BERTopic finds natural topic count from data, replacing the arbitrary K=6 in Phase 08 NMF.
- **Standardized classification**: CAP 20-category taxonomy enables cross-study comparison and time-series analysis of policy attention.
- **Reproducible**: Embedding cache + API response cache ensure identical results across runs.
- **Graceful degradation**: Phase works fully without API key (topics + similarity); CAP classification is additive.

### Trade-offs

- **New dependencies**: `bertopic>=0.17`, `fastembed>=0.4`, `hdbscan>=0.8.31` added to core dev deps. `anthropic>=0.40` as optional `[classify]` extra.
- **API cost**: CAP classification costs ~$7-13/biennium. Mitigated by caching (one-time cost per bill) and Batch API discount.
- **Not in pipeline**: Must be run manually after BT1 data acquisition. Acceptable for an NLP phase that requires separate data prep.

### Test coverage

53 new tests in `tests/test_bill_text.py`:
- Data loading, preprocessing, embedding cache (mocked FastEmbed)
- CAP classification (mocked API), cache operations, response parsing
- Bill similarity, topic-party cohesion, CAP passage rates
- Report builder (with and without CAP), plotting smoke tests
- CLI argument parsing, constants consistency, filtering manifest

All 2113 tests pass (53 new + 2060 existing).
