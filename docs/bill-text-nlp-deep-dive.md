# Bill Text NLP Deep Dive

A comprehensive survey of tools, techniques, data sources, and integration strategies for adding full bill text analysis to the tallgrass pipeline.

**Last updated:** 2026-03-02

---

## Executive Summary

The tallgrass pipeline currently uses only bill `short_title` (5-15 words) for text features via NMF topic modeling in Phase 08. Full bill text from the Kansas Legislature is available as PDFs at deterministic URLs — the extraction path is clear. The ecosystem has matured significantly: BERTopic is production-ready for topic modeling, sentence-transformers handle embeddings without domain-specific fine-tuning, and the TBIP model (now with a NumPyro implementation) enables text-based ideal point estimation directly compatible with our JAX/PyMC stack.

The highest-value additions are: (1) **BERTopic on full bill text** to replace NMF on short titles, producing richer policy-area topics that cross-reference with voting patterns; (2) **CAP policy classification** using an established 28-category taxonomy already validated on 1.36 million state bills; and (3) **bill text embeddings** for similarity analysis and enriched prediction features. TBIP (text-based ideal points) is a longer-term integration that would jointly model legislator ideology and bill content.

---

## 1. Current State in Tallgrass

### What exists

| Component | Location | What it does |
|-----------|----------|--------------|
| NMF topic features | `analysis/08_prediction/nlp_features.py` | TF-IDF + NMF (K=6) on `short_title`, produces 6 topic proportion columns |
| Topic visualization | `nlp_features.py:plot_topic_words()` | Multi-panel bar chart of top words per topic |
| Sponsor party feature | `analysis/08_prediction/prediction.py` | Binary `sponsor_party_R` from `sponsor_slugs` |
| NMF design decision | ADR-0012 | Why NMF over LDA or embeddings (deterministic, lightweight, zero deps) |
| Bill title extraction | `src/tallgrass/scraper.py:687` | `_extract_bill_title()` from `<h4>` |
| Short title from API | `src/tallgrass/scraper.py:1122-1125` | KLISS API `SHORTTITLE` field |
| Sponsor extraction | `src/tallgrass/scraper.py:639` | `_extract_sponsor()` parses `<a>` hrefs |

### Why NMF on short titles was the right call (and why it's now the ceiling)

ADR-0012 chose NMF because: deterministic (fixed seed), fast, non-negative weights interpretable as features, zero new dependencies. With 5-15 word titles, there's not enough signal for heavier machinery. The choice was correct for the data available.

But `short_title` is inherently limited. "Exempting the state of Kansas from certain requirements" tells you almost nothing about the bill's substance. Full bill text (1-100 pages of legislative language) contains: policy area, affected statutes, fiscal impact, definitions, implementation details. That's orders of magnitude more signal.

---

## 2. Bill Text Availability (Open Questions — Answered)

### Kansas Legislature: PDF only, deterministic URLs

Bill text is served exclusively as PDF from kslegislature.gov. The URL pattern is predictable and follows your existing session URL logic:

```
{base_url}/measures/documents/{bill_type}{number}_{version}_{revision}.pdf
```

**Examples (91st, 2025-26):**
- Introduced: `https://kslegislature.gov/li/b2025_26/measures/documents/sb55_00_0000.pdf`
- Committee amended: `.../sb281_03_0000.pdf`
- Enrolled: `.../sb394_enrolled.pdf`
- Supplemental note: `.../supp_note_hb2084_01_0000.pdf`
- Conference committee report: `.../ccrb_hb2531_01_0000.pdf`

Historical sessions follow the same pattern with `li_{end_year}/b{start}_{end}` prefix.

### Version tracking

| Suffix | Meaning | NLP relevance |
|--------|---------|---------------|
| `_00_0000` | Introduced version | Best for topic modeling (most bills never pass) |
| `_01_0000` through `_NN_0000` | Committee amendments | Version at time of vote matters for prediction |
| `_enrolled` | Final enrolled text | Best for "what actually passed" analysis |
| `supp_note_*` | Legislative Research supplemental note | Plain-English summary — potentially more useful than bill text for NLP |

### Supplemental notes: a hidden gem

Kansas Legislative Research Department writes a supplemental note for nearly every bill that advances. These are 1-5 page plain-English summaries written for legislators — not legalese. They describe: background, what the bill does, fiscal impact, proponents, opponents, committee testimony. For topic modeling and classification, supplemental notes may be *more* useful than the bills themselves: they're shorter, cleaner, and written in natural language.

### Corpus size estimate

~500-800 bills per biennium, each 1-100 pages. Supplemental notes add another ~400 shorter documents. Total corpus per biennium: probably 5,000-20,000 pages of text. Across 8 bienniums: ~40,000-160,000 pages. This is comfortably within the range of all tools discussed below — no scaling concerns.

### KLISS API: unlikely to help

The KLISS API (`/li/api/v13/rev-1`) provides bill metadata (status, sponsor, short title, history) but does not appear to serve bill text content. The API documentation is sparse and the endpoint guide is not publicly accessible. Your scraper already uses everything KLISS exposes. Bill text retrieval will require either direct PDF download or a third-party API.

---

## 3. Data Sources: How to Get Bill Text

### Option A: Direct PDF scraping (recommended)

Extend the existing scraper to download bill PDFs alongside vote data. Extract text with `pdfplumber` (pure Python, clean output from structured PDFs) or `pymupdf` (faster, C-backed).

**Pros:** No API dependency, full control, consistent with existing architecture, all versions available, supplemental notes included.

**Cons:** PDF text extraction is imperfect (layout issues, headers/footers, boilerplate). Legislative PDFs are typically well-structured (text-based, not scanned), so quality should be high.

**Implementation:** A new scraper step after `get_bill_urls()` downloads introduced version + supplemental note PDFs. Output: a 5th CSV (`{name}_bill_texts.csv`) with columns `bill_number`, `version`, `text`, `document_type` (bill/supp_note/ccrb). Joins to existing data on `bill_number`.

### Option B: LegiScan API

LegiScan (v1.91, March 2025) covers all 50 states including Kansas. The `getBillText` endpoint returns bill text with date, version, and MIME type. Base64-encoded by default; set `use_base64=False` for ASCII. Free tier: 30,000 queries/month — sufficient for Kansas-only work.

**Python client:** `legcop` on PyPI wraps the API.

**Pros:** Pre-extracted text (no PDF parsing), normalized across states, version tracking, metadata-rich.

**Cons:** External dependency, rate limits, free tier may not cover historical backfill of all 8 bienniums in one go. API key required.

**Assessment:** Good supplementary source. If direct PDF extraction proves unreliable, LegiScan is the fallback. Also useful for cross-state comparison if the project expands beyond Kansas.

### Option C: OpenStates / Plural

OpenStates (now under Plural, `pluralpolicy.com`) provides JSON bulk downloads including bill text for all 50 states. API v3 is live. Kansas is covered, but freshness depends on scraper maintenance for that state.

**Assessment:** Supplementary. Less reliable for Kansas-specific work than direct scraping or LegiScan. More useful if comparing Kansas with other states.

### Recommendation

**Start with Option A (direct PDF scraping).** You already have the URL pattern, the session logic, and the fetch infrastructure. Add `pdfplumber` to extract text. Use supplemental notes as the primary text source for topic modeling — they're shorter, cleaner, and more semantically dense than raw bill text. Fall back to LegiScan if PDF extraction quality is insufficient.

---

## 4. Text Preprocessing Challenges Specific to Legislative Language

These are known pitfalls from the literature and from inspecting Kansas bill PDFs:

1. **Amendatory language**: Bills that amend existing law use phrases like "Section 12-345 is hereby amended to read as follows:" followed by the new text, often with strikethrough/underline conventions lost in PDF extraction. Distinguishing new text from quoted existing law is a significant NLP challenge.

2. **Statutory cross-references**: Heavy use of references like "K.S.A. 79-3220" that are semantically meaningful but noise for topic models. These should be normalized (replace with "STATUTE_REF") or stripped. A regex like `K\.?S\.?A\.?\s*\d+-\d+` handles most cases.

3. **Boilerplate**: Standard enacting clauses ("Be it enacted by the Legislature of the State of Kansas:"), severability clauses, and effective date sections appear in nearly every bill. These dominate topic models if not filtered. Build a boilerplate sentence list and strip at preprocessing time.

4. **Version proliferation**: A bill that passes through committee substitute, conference committee, and enrollment may have 3-5 text versions. For topic modeling, use the introduced version (captures legislative intent, available for all bills). For vote prediction, ideally use the version on the floor at vote time — the `bill_actions.csv` HISTORY data can map vote dates to versions for 89th+ sessions.

5. **Bill length variance**: From one-paragraph resolutions to 200+ page appropriations bills. BERTopic handles this via automatic document segmentation. LDA may need manual chunking for long bills.

6. **Kansas-specific terminology**: References to Kansas statutes, agencies ("Kansas Department of Revenue"), and geographic regions that generic legal NLP models will not have seen. General sentence-transformers handle this surprisingly well — legislative language is formulaic enough that domain-specific models provide marginal improvement.

7. **PDF extraction artifacts**: Headers, footers, page numbers, line numbers embedded in legislative PDFs. `pdfplumber` handles most of these with bounding-box filtering, but validation on a sample of Kansas PDFs is essential before bulk extraction.

---

## 5. Topic Modeling: The State of the Art

### BERTopic (recommended)

**Version:** 0.17.4 (December 2025). ~7,400 GitHub stars. MIT license. Python 3.10+. Actively maintained by Maarten Grootendorst.

**Architecture:** Documents → sentence-transformer embeddings → UMAP dimensionality reduction → HDBSCAN clustering → c-TF-IDF topic representations. Every component is modular and swappable.

**Why it fits tallgrass:**
- Automatically discovers number of topics (no K selection needed — unlike NMF/LDA)
- Produces interpretable topic labels via c-TF-IDF keywords
- v0.17+ can use LLMs (GPT-4, Claude) as representation models for human-readable topic names
- Dynamic topic modeling tracks topics over time — maps directly to tracking policy attention across bienniums
- `.merge_models()` enables incremental/cross-session topic modeling
- Handles short text well (better than LDA/NMF on <100 word documents)

**Comparison to current NMF:**

| Feature | NMF (current) | BERTopic |
|---------|---------------|----------|
| Input | TF-IDF on short_title (5-15 words) | Embeddings of full bill text (1-100 pages) |
| Topics | Fixed K=6 | Automatic discovery |
| Semantics | Bag-of-words | Contextual embeddings |
| Cross-session | Independent per session | `.merge_models()` for longitudinal |
| Dependencies | scikit-learn (already installed) | sentence-transformers, hdbscan, umap-learn |
| GPU needed | No | No (CPU fine for <10K documents) |

**Dependencies:** `bertopic`, `sentence-transformers`, `hdbscan`, `umap-learn`. These are substantial additions (~500MB for sentence-transformer model). Weigh against the signal gain.

### LDA / NMF (classical)

Still viable on full bill text. If you want a lightweight approach: fit LDA on TF-IDF of full text with K=20-30 topics (coherence-optimized). Better than current NMF-on-titles but worse than BERTopic for topic quality.

**gensim Python 3.14 blocker:** gensim (the standard LDA library) only supports Python 3.8-3.11 as of its October 2025 release. Tallgrass requires Python 3.14.3+. This is a concrete incompatibility. Alternative: [tomotopy](https://bab2min.github.io/tomotopy/) is a C++-backed LDA implementation with Python bindings that may handle newer Python versions. Or skip LDA entirely and go BERTopic.

### TopicGPT / LLM-based topic modeling

Emerging approach (NAACL 2024): use LLM prompts to discover topics, then assign each document to the most relevant topic. Simpler than BERTopic for structured text. Requires API calls but no local model.

**Assessment:** Interesting for generating human-readable topic taxonomies. Not yet battle-tested at the scale of BERTopic. Worth monitoring but not recommended as primary approach today.

---

## 5. Policy Classification: The CAP Taxonomy

### Comparative Agendas Project (CAP)

The gold standard for policy area classification of legislation. 28 major topic codes covering all policy domains:

> Macroeconomics, Civil Rights, Health, Agriculture, Labor, Education, Environment, Energy, Immigration, Transportation, Law & Crime, Social Welfare, Housing, Banking, Defense, Technology, Foreign Trade, Government Operations, Public Lands, Cultural Policy, ...

### Validated on state bills

A 2025 paper in *Scientific Data* ("Policy agendas of American state legislatures") classified **1.36 million state bills** (2009-present, all 50 states) into CAP categories using transformer-based contextual word-piece embeddings. Key findings:

- Outperforms dictionary-based methods
- Consistent across states despite varying legislative language
- Publicly available labels and methodology

This is directly applicable to Kansas: the taxonomy is established, the methodology is validated at state bill scale, and the labels are interpretable by the nontechnical audience tallgrass targets.

### Implementation path

Two approaches:

1. **Fine-tune a classifier** on the published CAP-labeled state bill dataset, then apply to Kansas. More accurate, more work.
2. **Zero-shot classification** using sentence-transformers + CAP category descriptions. Less accurate, immediate gratification.
3. **LLM classification** — prompt an LLM with the CAP taxonomy + bill text, get classification. High accuracy, API cost.

**Recommendation:** Start with zero-shot classification to validate the taxonomy on Kansas data. If results are good, use them as-is. If not, fine-tune using the published dataset.

---

## 6. Text Embeddings and Bill Similarity

### Embedding model selection

| Model | Size | Domain | Speed | Quality |
|-------|------|--------|-------|---------|
| `all-MiniLM-L6-v2` | 80MB | General | Fast | Good baseline |
| `all-mpnet-base-v2` | 420MB | General | Medium | Best general-purpose |
| `pile-of-law/legalbert-large-1.7M-1` | 1.3GB | Legal (U.S. statutes + federal bills) | Slow | Best legal domain |
| `nlpaueb/legal-bert-base-uncased` | 440MB | Legal (EU + federal case law) | Medium | Good for legal |

**Recommendation:** Start with `all-mpnet-base-v2` (general-purpose, well-tested). Benchmark against PoL-BERT-Large if legal domain specificity matters. The BERTopic embedding model is a swappable component — you can A/B test without changing other code.

**Storage:** Embedding a 500-bill corpus at 768 dimensions ≈ 1.5MB per biennium. Trivial.

### Bill similarity

With embeddings, cosine similarity between bills is trivial. Applications:

1. **Consistency analysis:** Do legislators vote consistently on semantically similar bills? Compare vote vectors for bill pairs with similarity > 0.8.
2. **Framing effects:** Find bills with similar content but different framing (high text similarity, different titles). Do vote patterns differ?
3. **Model legislation detection:** Bills with >0.95 similarity across sessions or states may be model legislation (ALEC, etc.). Cross-reference with the NAACL 2024 "copycat bill" work.
4. **Prediction features:** Top-N most similar past bills → their pass/fail outcomes as prediction features.

---

## 7. Text-Based Ideal Points (TBIP)

### The model

Vafa, Naidu, and Blei (ACL 2020). An unsupervised probabilistic topic model that jointly estimates:
- **Legislator ideal points** (ideological positions) from how authors use language
- **Politicized topics** — topics where word choice varies systematically with ideology

Originally demonstrated on U.S. Senate speeches and tweets. Does not use votes or party labels — estimates ideology from text alone.

### NumPyro implementation

A NumPyro implementation is now part of the [official NumPyro tutorials](https://num.pyro.ai/en/stable/tutorials/tbip.html). This is directly compatible with the JAX/PyMC stack tallgrass already uses (nutpie compiles PyMC models to JAX). No TensorFlow dependency.

### Relevance to tallgrass

TBIP was previously rejected (no bill text — `docs/method-evaluation.md`). With full bill text, TBIP becomes a viable **cross-validation** of IRT ideal points: if TBIP ideal points (from text alone) correlate with IRT ideal points (from votes alone), that's strong evidence both are measuring the same latent construct. This is analogous to the Shor-McCarty and DIME external validations already in the pipeline.

**Caution:** TBIP was designed for *authored* text (speeches, tweets). Legislative bills are not authored by individual legislators — they're drafted by legislative staff and often co-sponsored. Sponsor information provides a link, but the text-author mapping is weaker than in the Senate speech setting. This should be treated as experimental.

### Integration

Would slot in as a standalone validation phase (like Phase 14/14b). Not a replacement for vote-based IRT — a complementary measure. Run TBIP on bill text + sponsor mapping, compare estimated ideal points with IRT xi_mean. Report correlation alongside SM and DIME validations.

---

## 8. Legal Domain Models: What Matters (and What Doesn't)

### Models evaluated

| Model | Corpus | Size | State bill coverage |
|-------|--------|------|---------------------|
| **LEGAL-BERT** | 12GB legal text (case law, EU law, contracts) | 110M params | Minimal — federal/EU focused |
| **PoL-BERT-Large** | 291GB Pile of Law (statutes, federal bills, regulations) | 335M params | U.S. State Codes included (~2.7GB) |
| **SaulLM-7B** | 30B tokens legal text (Mistral-based) | 7B params | Broad legal corpus |
| **SaulLM-54B/141B** | Scaled versions | 54B/141B params | Same corpus |

### Assessment

For **embedding** Kansas bill text (topic modeling, similarity, classification):
- General sentence-transformers (`all-mpnet-base-v2`) are sufficient. Legislative text is formulaic and well-structured — domain-specific models provide marginal improvement.
- PoL-BERT-Large is the best domain-specific option if needed. It's trained on U.S. State Codes, which overlap heavily with bill drafting language ("Section X is hereby amended to read as follows...").

For **generation/classification** (summarizing bills, generating topic labels):
- SaulLM is the open-source leader in the legal domain but is overkill for classification tasks.
- An API-based LLM (Claude, GPT-4) with a well-crafted prompt will outperform any fine-tuned model for zero-shot bill classification.

**Bottom line:** Don't over-invest in domain-specific models for the embedding step. Start with general-purpose. Reserve legal domain models for cases where general-purpose demonstrably underperforms.

---

## 9. Model Legislation Detection

### The problem

Organizations like ALEC draft "model bills" that are introduced verbatim or near-verbatim in multiple state legislatures. Detecting these reveals external influence on Kansas legislation.

### Legislative Influence Detector (LID)

Published at KDD 2016 (Burgess et al.). Uses Smith-Waterman local alignment (borrowed from bioinformatics) to find text reuse between model legislation databases and state bills. Elasticsearch-based candidate filtering, then pairwise alignment with synonym handling. Found ALEC introduced 10,370 bills across states, 1,573 enacted. A [companion tutorial](https://investigate.ai/azcentral-text-reuse-model-legislation/05-checking-for-legislative-text-reuse-using-python-solr-and-simple-text-search/) demonstrates a simpler Python + Solr approach.

### Simpler approach with embeddings

With bill text embeddings already computed (Section 6), cross-state bill similarity becomes trivial: bills with cosine similarity >0.95 from different states are model legislation candidates. This requires LegiScan or OpenStates data from other states — defer unless the project scope expands beyond Kansas.

### Integration with tallgrass

This is a natural extension of the bill similarity analysis in Phase 18. For Kansas-only analysis: flag bills whose text matches known model legislation databases. For cross-state work: detect which Kansas bills appear verbatim in other states. Both feed into the synthesis narrative ("Bill X was introduced in 12 states; Kansas was the 4th to pass it").

---

## 10. The Field: Published Research on Legislative Text + NLP

### Key papers

| Paper | Year | Contribution | Relevance |
|-------|------|-------------|-----------|
| Gerrish & Blei, "Predicting Legislative Roll Calls" | 2011 | Bayesian model combining text + votes (Ideal Point Topic Model) | Foundation for all text+vote work |
| Kraft, Jain & Rush, "An Embedding Model for Predicting Roll-Call Votes" | 2016 | Word2vec bill embeddings + legislator ideal vectors, 90.6% accuracy | Text-informed prediction |
| Vafa et al., "Text-Based Ideal Points" (ACL) | 2020 | TBIP model — ideal points from text alone | Validation of IRT (Section 7) |
| Davoodi & Goldwasser, "Understanding Political Agreement" (ACL) | 2020 | State-level legislative text agreement/disagreement prediction | State-level NLP pioneer |
| Davoodi & Goldwasser, "Extracting Winners and Losers" (ACL) | 2022 | Predicts which demographic groups gain/lose from state legislation | Policy impact extraction |
| Davoodi & Goldwasser, "State-Level Legislative Process" (NAACL) | 2024 | 50-state NLP analysis, LLMs for cross-state language normalization | Most comprehensive state-level work |
| Shin, "Issue-Specific Ideal Points" | 2024 | Hierarchical IRT with topic labels → per-policy-area ideal points | Bridges text topics and IRT |
| "Policy agendas of American state legislatures" (Scientific Data) | 2025 | 1.36M state bills classified into 28 CAP categories | Direct taxonomy for Kansas (Section 5) |
| "Text-Based Ideal Point Estimation: A Systematic Review" | 2025 | Review of 25 algorithms across 4 families | Essential method selection guide |
| "LegiGPT" | 2025 | LLM multi-stage bill classification, 85%+ over keywords | LLM classification approach |
| "TopicGPT" (NAACL) | 2024 | LLM-driven topic modeling as BERTopic alternative | Alternative topic approach |
| Spell et al. (EMNLP) | 2020 | Legislator embeddings from tweet sentiment | Social media angle |
| Burgess et al. "Legislative Influence Detector" (KDD) | 2016 | Smith-Waterman alignment for model legislation detection | Cross-state text reuse |
| Grootendorst, "BERTopic" | 2022 | Neural topic modeling with c-TF-IDF | Leading unsupervised approach |

### The Purdue state-level program (highlight)

Davoodi & Goldwasser at Purdue have produced the most comprehensive academic work on NLP for state legislatures, spanning three ACL/NAACL papers plus a dissertation. Their NAACL 2024 paper combines bill text NLP with cross-state network analysis across all 50 states, using LLMs to normalize varying legislative language. Their "winners and losers" work (ACL 2022) is particularly novel: it predicts which demographic groups benefit from or are harmed by proposed legislation based on text analysis. This is directly relevant to the tallgrass goal of making legislative analysis accessible to journalists and policymakers.

### Issue-specific ideal points (Shin 2024)

An R package (`issueirt`) that estimates topic-specific ideal points using roll call votes + user-supplied issue labels. It first estimates multidimensional ideal points, then projects onto issue-specific axes. **Key integration point:** topic labels from BERTopic/CAP classification could feed directly into this model, producing per-policy-area ideal point estimates. This bridges the text analysis (Phase 18) with the IRT infrastructure (Phase 04/10). The R dependency is already established in the pipeline (Phase 15 TSA, Phase 17 W-NOMINATE).

### Systematic review of text-based ideal points (2025)

"Computational Measurement of Political Positions" (arxiv 2511.13238, published in *Quality & Quantity*) is the first systematic review of 25 unsupervised/semi-supervised text-based ideal point estimation algorithms. Identifies four methodological families: word-frequency, topic modeling, word embedding, and LLM-based. Provides practical guidance on trade-offs (transparency, technical requirements, validation). **Essential reading before implementing TBIP or any text-based ideal point method.**

### Trend

The field has moved from "can NLP handle legislative text?" (2018-2022) to "which approach works best at scale?" (2023-2025). The current frontier is LLM-based classification displacing fine-tuned BERT models for policy categorization, while BERTopic remains the leading unsupervised option for topic discovery. State-level work has matured significantly — the Purdue program and the 2025 Scientific Data publication prove that transformer-based methods work reliably on state legislative text, not just federal.

---

## 10. Integration Strategy for Tallgrass

### Published integration patterns

The academic literature shows five main patterns for combining text with roll-call analysis. Tallgrass should use Pattern 4 as the primary approach, with Patterns 3 and 5 as future extensions.

| Pattern | Description | Example | Fit for tallgrass |
|---------|-------------|---------|-------------------|
| **1. Joint generative** | Text generates topics via LDA; topics parameterize bill positions; ideal points and positions jointly predict votes | Gerrish & Blei 2011 | Complex inference, diminishing returns over Pattern 4 |
| **2. Embedding prediction** | Bill text → averaged word embeddings; legislator ideal vectors; bilinear vote prediction | Kraft et al. 2016 | Simpler than Pattern 1, still complex |
| **3. Text-based ideal points** | Ideal points from text alone, compared post-hoc to vote-based | TBIP (Vafa et al. 2020) | Validation of IRT — Phase 18b |
| **4. Two-stage pipeline** | Extract text features independently, use as covariates in vote/ideal point models | Most common in practice | **Primary approach** — extends Phase 08 |
| **5. Issue-specific IRT** | Topic labels feed into hierarchical IRT for per-policy-area ideal points | Shin 2024 (`issueirt` R package) | Bridges text and IRT — future extension |

Pattern 4 is what Phase 08 already does with `sponsor_party_R` — extending it with richer text features is a natural evolution. Pattern 5 (issue-specific ideal points) is particularly appealing: it would produce per-policy-area ideal point estimates (e.g., "how conservative is this legislator on education vs. criminal justice?"), directly serving the nontechnical audience. The `issueirt` R package integrates via R subprocess, following the established pattern from Phase 15 and Phase 17.

### Phase architecture

Two new phases, one scraper extension:

#### Scraper extension: Bill text retrieval

Extend `KSVoteScraper` with a new step that downloads bill PDFs (introduced version + supplemental notes) and extracts text via `pdfplumber`. Output: `{name}_bill_texts.csv` with columns:

```
session, bill_number, version, document_type, text, page_count, pdf_url
```

Joins to existing data on `session` + `bill_number`. This is a 5th CSV output alongside votes, rollcalls, legislators, and bill_actions.

**Design decision:** Whether to make this a scraper step (runs with `just scrape`) or a separate retrieval step (runs before analysis). Recommendation: separate step (`just fetch-text 2025`) since it's slow (hundreds of PDF downloads) and not needed for vote data.

#### Phase 18: Bill Text Analysis

Primary bill text analysis phase. Depends on Phase 01 (EDA) for filtering manifests and Phase 08 upstream data (rollcalls with bill metadata).

1. **Load and preprocess** bill text from `bill_texts.csv`. Strip boilerplate headers/footers, section numbers. Handle encoding issues.
2. **Embed** using sentence-transformers (`all-mpnet-base-v2`). Cache embeddings per biennium.
3. **Topic modeling** with BERTopic. Automatic topic discovery. Save topic assignments per bill.
4. **Policy classification** using CAP 28-category taxonomy (zero-shot or fine-tuned). Save category per bill.
5. **Similarity matrix** — cosine similarity between all bill pairs. Identify clusters of related legislation.
6. **Cross-reference with votes** — which policy areas split the caucus? Which are rubber-stamps? Topic-specific party cohesion scores.

Report: bill topic distribution, topic-by-party heatmap, caucus-splitting topics, model legislation candidates, topic trends across bienniums.

#### Phase 18b: Text-Based Ideal Points (experimental)

Standalone validation phase (like 14/14b). Runs TBIP via NumPyro on bill text + sponsor mapping. Compares text-derived ideal points with IRT xi_mean. Reports correlation alongside SM and DIME validations.

### Dependency graph

```
Scraper → bill_texts.csv
                ↓
Phase 18 (Bill Text Analysis)
    depends on: Phase 01 (EDA), bill_texts.csv
    produces: topics, classifications, similarity, embeddings
                ↓
Phase 18b (TBIP — experimental)
    depends on: Phase 18 embeddings, Phase 04 IRT results
    produces: text-based ideal points, validation correlations
```

### Phase numbering

The pipeline currently has phases through 17. Phase 18 follows naturally. The `bill_texts.csv` fetcher is a scraper extension, not a numbered phase.

---

## 11. Dependencies and Resource Impact

### New Python dependencies

| Package | Purpose | Size | Already in project? |
|---------|---------|------|---------------------|
| `pdfplumber` | PDF text extraction | ~5MB | No |
| `sentence-transformers` | Text embeddings | ~50MB + model (~420MB) | No |
| `bertopic` | Topic modeling | ~10MB | No |
| `hdbscan` | Clustering (BERTopic dep) | ~15MB | No |
| `umap-learn` | Dim reduction (BERTopic dep) | ~20MB | Already used by Phase 03 |

`umap-learn` is already a dependency (Phase 03 UMAP). Total new: ~500MB (mostly the sentence-transformer model weight download).

### Compute

- PDF extraction: I/O-bound, ~1 sec/bill, ~10-15 min per biennium
- Embedding: ~2 sec/bill on CPU, ~5-10 min per biennium
- BERTopic fitting: ~30-60 sec per biennium on CPU
- TBIP (NumPyro MCMC): ~5-20 min depending on corpus size

All within Apple Silicon M3 Pro budget. No GPU needed.

---

## 12. Open Questions (Remaining)

The four original open questions from `docs/future-bill-text-analysis.md` are now answered:

| Question | Answer |
|----------|--------|
| Bill text availability | PDF only from kslegislature.gov, deterministic URLs. LegiScan has pre-extracted text as fallback. |
| Text scope | Multiple versions: introduced (`_00_`), amended, enrolled. Supplemental notes available. |
| Corpus size | ~500-800 bills/biennium, 1-100 pages each. Manageable without GPU. |
| Integration point | New 5th CSV (`bill_texts.csv`) joining on `bill_number`. Separate fetch step. |

### New questions

1. **Supplemental notes vs. bill text** — Are supplemental notes available for pre-89th sessions? They may not exist for all historical bienniums.
2. **Version at time of vote** — For text-informed vote prediction, we need the bill version that was on the floor when the roll call occurred. The `bill_actions.csv` HISTORY data could map vote dates to versions, but only for 89th+ sessions.
3. **Cross-biennium topic stability** — If BERTopic discovers different topics per biennium, cross-session comparison becomes harder. BERTopic's `.merge_models()` may help, or a fixed taxonomy (CAP) avoids the problem entirely.
4. **TBIP author mapping** — Bills have sponsors but are staff-drafted. Is sponsor mapping sufficient for TBIP, or does the model need true author attribution? This may limit TBIP to speeches/testimony rather than bill text.
5. **Model legislation** — Cross-state bill similarity requires LegiScan or OpenStates data from other states. Defer unless the project scope expands.

---

## 13. Recommended Phasing

### Phase 1: Data acquisition (prerequisite)

- Add `pdfplumber` dependency
- Implement PDF text fetch and extraction as a scraper extension
- Output `bill_texts.csv` with introduced version + supplemental notes
- Start with 91st biennium (current), validate, then backfill historical

### Phase 2: Topic modeling and classification

- Add `bertopic`, `sentence-transformers`, `hdbscan` dependencies
- Implement Phase 18 (Bill Text Analysis)
- BERTopic topic discovery + CAP zero-shot classification
- Cross-reference topics with voting patterns
- HTML report with topic distribution, party heatmaps, caucus-splitting analysis

### Phase 3: Similarity and prediction enrichment

- Bill similarity matrix from embeddings
- Add text-based features to Phase 08 prediction model
- Consistency analysis: same-topic vote variance
- Framing effects: similar-content, different-framing bill pairs

### Phase 4: Text-based ideal points (experimental)

- Implement Phase 18b (TBIP via NumPyro)
- Cross-validate text-derived ideal points against IRT
- Report alongside SM and DIME external validations

---

## Sources

### Tools
- [BERTopic](https://github.com/MaartenGr/BERTopic) — v0.17.4, MIT license, ~7,400 stars
- [sentence-transformers](https://www.sbert.net/) — Hugging Face, well-maintained
- [TBIP (NumPyro)](https://num.pyro.ai/en/stable/tutorials/tbip.html) — Official NumPyro tutorial
- [TBIP (original)](https://github.com/keyonvafa/tbip) — TensorFlow, ACL 2020
- [LegiScan API](https://legiscan.com/legiscan) — v1.91, free tier 30K queries/month
- [legcop](https://pypi.org/project/legcop/) — Python LegiScan client
- [OpenStates / Plural](https://docs.openstates.org/api-v3/) — All-state legislative data
- [bb25](https://github.com/instructkr/bb25) / [bayesian-bm25](https://github.com/cognica-io/bayesian-bm25) — Hybrid search
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF text extraction

### Pre-trained models
- [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) — General-purpose embeddings
- [pile-of-law/legalbert-large-1.7M-1](https://huggingface.co/pile-of-law/legalbert-large-1.7M-1) — Legal domain (PoL-BERT-Large)
- [nlpaueb/legal-bert-base-uncased](https://huggingface.co/nlpaueb/legal-bert-base-uncased) — Legal domain (LEGAL-BERT)
- [SaulLM-7B](https://arxiv.org/abs/2403.03883) — Legal LLM (Mistral-based)

### Datasets
- [Pile of Law](https://huggingface.co/datasets/pile-of-law/pile-of-law) — 291GB legal corpus (NeurIPS 2022)
- [Comparative Agendas Project](https://www.comparativeagendas.net/) — Policy topic taxonomy
- [LegiScan Kansas datasets](https://legiscan.com/KS/datasets) — Historical Kansas bill data

### Datasets and benchmarks
- [BillSum](https://github.com/FiscalNote/BillSum) — 22,218 congressional bills with summaries (first bill summarization benchmark)
- [issueirt](https://github.com/sooahnshin/issueirt) — R package for issue-specific ideal points (Shin 2024)
- [pylegiscan](https://github.com/poliquin/pylegiscan) — Python LegiScan pull API wrapper
- [tomotopy](https://bab2min.github.io/tomotopy/) — C++-backed LDA (alternative to gensim for Python 3.14+)

### Research
- Vafa, Naidu, Blei. "Text-Based Ideal Points." ACL 2020. [Paper](https://arxiv.org/abs/2005.04232)
- "Policy agendas of American state legislatures." *Scientific Data*, 2025. [Paper](https://www.nature.com/articles/s41597-025-05621-5)
- "Analysis of State-Level Legislative Process." NAACL 2024. [Paper](https://aclanthology.org/2024.naacl-long.411.pdf)
- "LegiGPT: Party Politics in Transport Policy." 2025. [Paper](https://arxiv.org/html/2506.16692)
- "TopicGPT: A Prompt-based Topic Modeling Framework." NAACL 2024. [Paper](https://aclanthology.org/2024.naacl-long.164.pdf)
- Gerrish & Blei. "Predicting Legislative Roll Calls from Text." ICML 2011.
- Grootendorst. "BERTopic: Neural topic modeling with a class-based TF-IDF procedure." 2022. [Paper](https://arxiv.org/pdf/2203.05794)
- Henderson et al. "Pile of Law: Learning Responsible Data Filtering." NeurIPS 2022. [Paper](https://arxiv.org/abs/2207.00220)
- Colombo et al. "SaulLM-7B: A pioneering Large Language Model for Law." 2024. [Paper](https://arxiv.org/abs/2403.03883)
- "NLP for the Legal Domain: A Survey of Tasks, Datasets." 2024. [Paper](https://arxiv.org/pdf/2410.21306)
- Spell et al. "An Embedding Model for Estimating Legislative Preferences." EMNLP 2020. [Paper](https://aclanthology.org/2020.emnlp-main.46/)
- Davoodi & Goldwasser. "Understanding Political Agreement and Disagreement." ACL 2020. [Paper](https://aclanthology.org/2020.acl-main.476/)
- Davoodi & Goldwasser. "Modeling U.S. State-Level Policies by Extracting Winners and Losers." ACL 2022. [Paper](https://aclanthology.org/2022.acl-long.22/)
- Davoodi & Goldwasser. "Analysis of State-Level Legislative Process." NAACL 2024. [Paper](https://aclanthology.org/2024.naacl-long.411.pdf)
- Shin. "Measuring Issue-Specific Ideal Points from Roll Call Votes." 2024. [Paper](https://sooahnshin.com/issueirt.pdf)
- "Computational Measurement of Political Positions: A Review." *Quality & Quantity*, 2025. [Paper](https://arxiv.org/abs/2511.13238)
- Kraft, Jain & Rush. "An Embedding Model for Predicting Roll-Call Votes." EMNLP 2016. [Paper](https://aclanthology.org/D16-1221.pdf)
- Burgess et al. "Legislative Influence Detector." KDD 2016. [Paper](https://www.kdd.org/kdd2016/papers/files/adf0831-burgessA.pdf)
- "BillSum: A Corpus for Automatic Summarization of US Legislation." EMNLP 2019. [Paper](https://aclanthology.org/D19-5406/)

---

## Related

- `docs/future-bill-text-analysis.md` — Original notes (superseded by this document)
- `docs/method-evaluation.md` — TBIP evaluation and rejection (revisited in Section 7)
- `docs/prediction-deep-dive.md` — Gerrish & Blei literature, current feature comparison
- `analysis/08_prediction/nlp_features.py` — Current NMF implementation
- `analysis/design/prediction.md` — NMF design decisions (lines 126-154)
- ADR-0012 — NMF over LDA/embeddings decision
