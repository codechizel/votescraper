# Bill Text Analysis (Phase 18) Design Choices

## Assumptions

1. **Supplemental notes are better for NLP than introduced text.**
   Supp notes are shorter plain-English summaries written by staff, while
   introduced text contains dense statutory language with K.S.A. references.
   Violated if a bill has no supp note (fall back to introduced text).

2. **384-dim embeddings capture legislative semantic structure.**
   `bge-small-en-v1.5` was trained on general English, not legal text.
   Violated if domain-specific models (e.g., Legal-BERT) would give
   qualitatively different topic assignments. Mitigation: BERTopic's
   c-TF-IDF layer provides domain-specific term weighting on top of
   general embeddings.

3. **HDBSCAN auto-K is appropriate for legislative text.**
   We don't know the "true" number of policy topics. HDBSCAN's density-based
   approach avoids forcing K. Violated if the corpus is too small for density
   estimation to work well (below ~100 bills).

4. **CAP 20-category taxonomy is meaningful for Kansas state legislation.**
   CAP was designed for federal legislation. Some categories (foreign_trade,
   immigration, defense) may be rare in state-level data. This is expected,
   not a failure.

## Parameters & Constants

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `DEFAULT_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | 384-dim, ONNX-native via FastEmbed, no PyTorch. MTEB rank 51 with 33M params (vs 110M for base). |
| `MAX_TOKENS_APPROX` | 8000 chars | ~2000 tokens. bge-small supports 512 token max_seq_length; truncation at 8K chars leaves headroom after tokenization. |
| `MIN_CLUSTER_SIZE` | 15 | HDBSCAN minimum cluster. 15 ≈ 2% of 800 bills. Lower = more topics, higher = fewer. CLI-adjustable. |
| `MIN_SAMPLES` | 5 | HDBSCAN min_samples. 5 is the library default. Affects noise assignment. |
| `UMAP_N_COMPONENTS` | 10 | Standard for BERTopic. Reduces 384-dim to 10 for HDBSCAN input. |
| `UMAP_N_NEIGHBORS` | 15 | Balances local/global. BERTopic default. |
| `UMAP_MIN_DIST` | 0.0 | Allows tight clusters. Standard for topic modeling. |
| `RANDOM_SEED` | 42 | Reproducibility. UMAP random_state, HDBSCAN is deterministic given same input. |
| `SIMILARITY_THRESHOLD` | 0.80 | Cosine similarity floor for reported pairs. 0.80 = strong semantic similarity. |
| `TOP_SIMILAR_PAIRS` | 30 | Limit for report readability. |
| `HEATMAP_TOP_N` | 50 | Bills shown in similarity heatmap. |

## Methodological Choices

### Why BERTopic (not LDA, NMF, or Top2Vec)

- **LDA** assumes bag-of-words, ignores word order and semantics. Poor for short documents.
- **NMF** (Phase 08) works on TF-IDF of short titles. Adequate for 5-15 word titles, not full text.
- **Top2Vec** auto-discovers topics like BERTopic but uses doc2vec embeddings (lower quality than transformer embeddings).
- **BERTopic** combines transformer embeddings + UMAP + HDBSCAN + c-TF-IDF. Best of both: semantic understanding from embeddings, interpretable labels from c-TF-IDF.

### Why FastEmbed (not sentence-transformers/HuggingFace)

- sentence-transformers requires PyTorch (~2GB). FastEmbed uses ONNX Runtime (~50-100MB).
- Same model weights (`bge-small-en-v1.5`) via different inference backend.
- BERTopic natively supports FastEmbed since v0.17.1.

### Why Claude Sonnet for CAP Classification (not fine-tuned classifier)

- Fine-tuning requires labeled training data. No labeled Kansas bill → CAP dataset exists.
- Claude Sonnet achieves high accuracy on zero-shot classification tasks.
- Caching makes it reproducible and cost-effective (one-time API cost per bill).
- Optional dependency — phase works fully without API key.

### Why Cosine Similarity (not Euclidean)

- Standard for comparing normalized text embeddings.
- Invariant to embedding magnitude (BGE model produces unit-norm vectors).

### Why Rice Index for Topic-Party Cohesion

- Rice index is the standard measure of party unity in roll call analysis (Stuart Rice, 1925).
- Already used in Phase 07 (Indices). Consistent methodology.
- Simple interpretation: 1.0 = unanimous, 0.0 = evenly split.

## Downstream Implications

- **Phase 08 (Prediction)**: NMF topics from short titles could be supplemented or replaced by BERTopic topics from full text. Not done automatically — requires conscious integration.
- **Phase 11 (Synthesis)**: Topic labels could enrich narrative synthesis. Currently not consumed.
- **Phase 21 (BT3)**: Embedding-vote text-based ideal points (ADR-0086) use the cached embeddings from this phase.
- **Phase 19 (BT4)**: Issue-specific ideal points (ADR-0087) use the BERTopic topic assignments and optional CAP classifications from this phase to subset votes by policy area.
- **Future phase (BT5)**: Temporal topic evolution and cross-state comparison build on the embeddings and topic assignments saved here.
- **CAP classifications** enable cross-state and cross-session comparisons using the standardized policy taxonomy. Useful for comparative state politics research.
