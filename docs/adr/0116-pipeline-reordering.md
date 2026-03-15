# ADR-0116: Pipeline Reordering — Text-Analysis to Position 4, UMAP to Position 9

**Date:** 2026-03-14
**Status:** Accepted

## Context

A dependency audit (`docs/pipeline-ordering-analysis.md`) found two ordering problems in the single-biennium pipeline:

1. **Bill Text (Phase 20) at position 20** — This phase has zero upstream dependencies beyond EDA but runs after phases that consume its output. BERTopic topics from Phase 20 are needed by Phase 12 (Bipartite — topic labels on bill communities), Phase 22 (Issue IRT — topic-stratified vote subsets), and Phase 21 (TBIP — bill embeddings). Running it after these phases means they either skip the NLP features or require a second pass.

2. **UMAP (Phase 04) at position 4** — UMAP generates a nonlinear embedding that, in the pipeline, tries to overlay IRT ideal points for validation. But IRT (Phase 05) runs after UMAP, so the overlay is always missing on first run.

## Decision

Reorder the single-biennium pipeline:

**Before:** EDA → PCA → MCA → UMAP → IRT → 2D IRT → Hierarchical → PPC → Clustering → ... → TSA → Bill Text → TBIP → ...

**After:** EDA → PCA → MCA → **Bill Text** → IRT → 2D IRT → Hierarchical → Hierarchical 2D → PPC → **UMAP** → Clustering → ... → TSA → TBIP → ...

- Bill Text moves from position 20 to position 4 (after MCA, before IRT)
- UMAP moves from position 4 to position 9 (after PPC, before Clustering)

No changes to phase numbering or directory structure — only the execution order in the `pipeline` recipe in the Justfile.

## Consequences

**Positive:**
- BERTopic topics available to all downstream phases (12, 21, 22) on first pipeline run
- UMAP gets IRT ideal points for validation overlay
- Pipeline produces complete output in a single pass

**Negative:**
- Bill Text (BERTopic) adds ~2-5 min before IRT starts. Acceptable since BERTopic runs on CPU and doesn't compete with MCMC.
- UMAP moves later, so interactive exploration of the ideological landscape happens after IRT rather than before. Acceptable since UMAP is a visualization, not a prerequisite.

**Neutral:**
- Phase directory numbers unchanged (Phase 20 stays `analysis/20_bill_text/`)
- All `just <phase>` recipes continue to work independently
