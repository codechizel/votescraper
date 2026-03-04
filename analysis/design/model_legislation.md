# Phase 20: Model Legislation Detection

## Purpose

Detect Kansas bills that match known model legislation (ALEC templates) and bills that appear in neighboring states (cross-state policy diffusion). Answers: "Which Kansas bills were likely derived from model legislation?" and "Which Kansas bills appear in other states?"

## Method

### Template Matching (ALEC)

1. Scrape ALEC model policy corpus (~1,000 model bills from alec.org)
2. Embed both corpora with the same BGE model (BAAI/bge-small-en-v1.5, 384-dim) used in Phase 18
3. Compute cosine similarity matrix (Kansas x ALEC)
4. Report matches above threshold with tiered classification

### Cross-State Diffusion

1. Discover neighbor state bills via OpenStates API v3
2. Download and extract bill text (reuse BillTextFetcher)
3. Embed and compute cosine similarity (Kansas x each state)
4. Identify bills appearing in multiple states

### N-gram Overlap Confirmation

For high-similarity pairs (>= 0.85), compute 5-gram overlap as secondary evidence of genuine text reuse (vs. topical similarity). The distinction matters: two voter ID bills can be topically similar without sharing text.

## Data Sources

| Source | Access | Coverage | Update Frequency |
|--------|--------|----------|-----------------|
| ALEC model policies | Public HTML scrape | ~1,061 policies, 1995-2026 | Run `just alec` to refresh |
| OpenStates API v3 | Free API (500/day) | All 50 states, current sessions | Per-session |
| Kansas bill texts | Phase 18 (BT1) | Current biennium | Pre-requisite |

## Thresholds

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Near-identical | >= 0.95 | Standard in LID literature for direct copies |
| Strong match | >= 0.85 | Adapted from same source with minor edits |
| Related | >= 0.70 | Similar policy area, warrants investigation |
| N-gram size | 5 words | Copy/Paste/Legislate used 6-word stemmed; 5 unstemmed comparable |
| N-gram overlap | >= 0.20 | 20%+ shared 5-grams = genuine text reuse |

## Embedding Design

Uses the same BAAI/bge-small-en-v1.5 model as Phase 18. This ensures:
- Embeddings are directly comparable (same vector space)
- Kansas embeddings can be loaded from Phase 18 cache
- No model version drift between phases

Text preprocessing reuses `preprocess_for_embedding()` from Phase 18 bill_text_data.py.

## Architecture

```
src/tallgrass/alec/          — ALEC corpus scraper (new CLI: tallgrass-alec)
src/tallgrass/text/openstates.py  — OpenStates StateAdapter (multi-state discovery)
analysis/23_model_legislation/    — Phase 20 analysis
  model_legislation_data.py       — pure data functions
  model_legislation.py            — main script
  model_legislation_report.py     — HTML report builder
```

## Report Sections

1. Key Findings — top-line summary
2. Data Summary — bill counts, model, thresholds
3. ALEC Matches — interactive table (Kansas bill, ALEC title, similarity, 5-gram overlap)
4. Cross-State Matches — interactive table (Kansas bill, state, similarity)
5. Similarity Distribution — histogram with threshold lines
6. Topic Heatmap — matches by policy area
7. Match Network — Kansas bills in multiple states
8. Near-Identical Details — side-by-side text excerpts
9. Analysis Parameters

## Limitations

- **Similarity != Causation.** Bills addressing the same policy problem may independently use similar language. Near-identical matches with high n-gram overlap are the strongest evidence.
- **Embedding truncation.** Texts > 8000 chars are truncated, potentially missing model language in later sections.
- **ALEC corpus completeness.** Not all model bills are publicly posted; some may have been removed.
- **Cross-state coverage.** Only current biennium and 4 neighbor states in initial implementation.
- **No alignment scoring.** Smith-Waterman or similar local alignment would identify exact shared passages but is computationally expensive (future extension).

## Inputs

- `data/kansas/{session}/{name}_bill_texts.csv` — Kansas bill texts (from BT1)
- `data/external/alec/alec_model_bills.csv` — ALEC corpus (from `just alec`)
- `data/{state}/{session}/{name}_bill_texts.csv` — Cross-state texts (from `just text` with OpenStates)
- Phase 18 topic assignments (optional, for cross-reference)

## Outputs

- `match_summary.parquet` — unified match table
- `filtering_manifest.json` — reproducibility metadata
- `23_model_legislation_report.html` — self-contained HTML report
- Plots: similarity_distribution.png, topic_match_heatmap.png, match_network.png
