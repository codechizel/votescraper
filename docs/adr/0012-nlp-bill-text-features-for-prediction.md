# ADR-0012: NLP Bill Text Features for Prediction

**Date:** 2026-02-22
**Status:** Accepted

## Context

The prediction pipeline (Phase 6) achieves strong individual vote AUC (0.98) but moderate bill passage AUC (0.84-0.96 depending on chamber and validation method). Bill passage currently uses only structural features: IRT partisanship (beta), veto override flag, day of session, vote type, and bill prefix. The `short_title` text from the KLISS API is already present in the rollcalls CSV but unused. Adding subject-matter signal (elections, taxes, healthcare, etc.) could improve bill passage prediction.

Several implementation decisions needed to be resolved:

1. **Topic modeling method.** LDA, NMF, and sentence embeddings are all viable approaches for extracting topics from short text.
2. **Number of topics (K).** Too few loses granularity; too many creates sparse/incoherent topics on a small corpus (~200-500 documents per chamber).
3. **Which text field.** The rollcalls CSV contains both `bill_title` and `short_title`.
4. **Data leakage risk.** Bill text could theoretically encode outcome information if titles are modified post-vote.
5. **Which model gets the features.** Vote prediction (AUC=0.98) vs bill passage prediction (AUC=0.84-0.96).
6. **Module architecture.** Inline in `prediction.py` vs standalone module.

## Decision

1. **NMF (Non-negative Matrix Factorization) on TF-IDF.** NMF is deterministic (given a fixed seed), faster than LDA, produces non-negative topic weights directly usable as features, and requires zero new dependencies — scikit-learn's `TfidfVectorizer` and `NMF` are already available. LDA adds stochasticity and gains little on ~500 short documents. Sentence embeddings (e.g., sentence-transformers) would add a ~400MB model dependency for marginal benefit on 5-15 word titles.

2. **K=6 topics.** The Senate has ~194 roll calls, setting the ceiling for meaningful topics. At K=6, each topic covers ~30 documents on average — enough for stable NMF estimation. Constants: `TFIDF_MAX_DF=0.85`, `TFIDF_MIN_DF=2`, `TFIDF_MAX_FEATURES=500`, `TFIDF_NGRAM_RANGE=(1,2)`.

3. **`short_title` from the KLISS API.** Never empty (unlike `bill_title` which is null for 15 Senate amendment votes), more substantive, and lacks the formulaic "AN ACT concerning..." preamble that adds noise.

4. **No data leakage.** Bill titles are fixed at introduction — they are pre-vote public information, known before any legislator casts a ballot. Using them as features is analogous to the bill prefix (HB, SB) and vote type already in the model. The topic model is fit on all documents (not train-only) because text is pre-vote information, same as IRT ideal points.

5. **Bill passage model only.** Vote prediction already achieves AUC=0.98; adding 6 topic features to ~20 existing features provides negligible marginal lift and complicates SHAP interpretation. Bill passage (AUC=0.84-0.96) has more room for improvement and fewer features to begin with.

6. **Standalone `analysis/nlp_features.py` module.** Following the `synthesis_detect.py` pattern: pure data logic with no I/O or plotting (except a dedicated `plot_topic_words()` function). Independently testable, and could serve future phases (e.g., synthesis report could display topic distributions).

## Consequences

**Benefits:**
- Zero new dependencies — uses scikit-learn's TfidfVectorizer + NMF, already installed.
- `build_bill_features()` accepts `topic_features` as an optional parameter defaulting to `None`, maintaining full backward compatibility.
- Deterministic results (NMF with fixed random seed).
- Graceful degradation: degenerate input (nulls, empty strings, tiny corpora) returns zero-filled topic columns instead of crashing.
- 16 unit tests cover core fitting, edge cases, display names, and dataclass immutability. 4 integration tests verify topic features flow through to bill passage models.
- Topic labels are auto-generated from top words and integrated into SHAP display names for interpretable feature importance plots.

**Trade-offs:**
- 6 additional features on ~200-500 rows increases overfitting risk. Mitigated by NMF's non-negative constraint (acts as implicit regularization) and the existing CV/temporal split validation.
- Short titles (5-15 words) provide limited vocabulary for topic modeling. The `TFIDF_MAX_FEATURES=500` cap and `min_df=2` filter ensure only recurring terms contribute.
- Topic interpretability depends on the corpus — topics may not align with intuitive policy areas (e.g., a "tax" topic might blend with "budget" or "revenue"). The `plot_topic_words()` visualization helps users assess topic quality.
- NMF topics are fit per chamber, so House and Senate may discover different topic structures. This is by design (chambers vote on different bills) but means topic labels aren't comparable across chambers.
