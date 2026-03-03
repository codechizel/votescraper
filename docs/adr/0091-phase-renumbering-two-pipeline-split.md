# ADR-0091: Phase Renumbering + Two-Pipeline Split

## Status

Accepted (2026-03-03)

## Context

The analysis pipeline grew to 27 phases with messy numbering — letter suffixes (04b, 04c, 05b, 06b, 14b, 18b), gaps (12 jumped to 15), and 11 phases excluded from the pipeline. For open-source publication, clean sequential integers organized by logical stage make the project more approachable.

## Decision

Renumber all 27 phases as clean sequential integers 01-27, grouped by logical stage. Split into two explicit pipelines: single-biennium (01-25) and cross-biennium (26-27).

### Mapping Table

| New | Directory | Name | Old |
|-----|-----------|------|-----|
| 01 | `01_eda` | Exploratory Data Analysis | `01_eda` (same) |
| 02 | `02_pca` | Principal Component Analysis | `02_pca` (same) |
| 03 | `03_mca` | Multiple Correspondence Analysis | `02c_mca` |
| 04 | `04_umap` | UMAP Visualization | `03_umap` |
| 05 | `05_irt` | Bayesian IRT (1D) | `04_irt` |
| 06 | `06_irt_2d` | 2D Bayesian IRT | `04b_irt_2d` |
| 07 | `07_hierarchical` | Hierarchical Bayesian IRT | `10_hierarchical` |
| 08 | `08_ppc` | Posterior Predictive Checks | `04c_ppc` |
| 09 | `09_clustering` | Voting Bloc Detection | `05_clustering` |
| 10 | `10_lca` | Latent Class Analysis | `05b_lca` |
| 11 | `11_network` | Legislator Network | `06_network` |
| 12 | `12_bipartite` | Bipartite Network | `06b_network_bipartite` |
| 13 | `13_indices` | Legislative Indices | `07_indices` |
| 14 | `14_beta_binomial` | Bayesian Party Loyalty | `09_beta_binomial` |
| 15 | `15_prediction` | Vote Prediction | `08_prediction` |
| 16 | `16_wnominate` | W-NOMINATE + OC | `17_wnominate` |
| 17 | `17_external_validation` | Shor-McCarty Validation | `14_external_validation` |
| 18 | `18_dime` | DIME/CFscore Validation | `14b_external_validation_dime` |
| 19 | `19_tsa` | Time Series Analysis | `15_tsa` |
| 20 | `20_bill_text` | Bill Text NLP | `18_bill_text` |
| 21 | `21_tbip` | Text-Based Ideal Points | `18b_tbip` |
| 22 | `22_issue_irt` | Issue-Specific Ideal Points | `19_issue_irt` |
| 23 | `23_model_legislation` | Model Legislation Detection | `20_model_legislation` |
| 24 | `24_synthesis` | Narrative Synthesis | `11_synthesis` |
| 25 | `25_profiles` | Legislator Profiles | `12_profiles` |
| 26 | `26_cross_session` | Cross-Biennium Validation | `13_cross_session` |
| 27 | `27_dynamic_irt` | Dynamic Ideal Points | `16_dynamic_irt` |

### Two Pipelines

**`just pipeline 2025-26`** — Runs phases 01-25. Phases 06, 08, 16-23 gracefully skip when prerequisites are missing (no R, no bill texts, biennium out of range for SM/DIME).

**`just cross-pipeline`** — Runs phases 26-27. Separate because they require data from multiple bienniums.

### Graceful Skip Behavior

Seven phases now check prerequisites at the top of `main()` and `return` (exit 0) when missing:

| Phase | Skip condition |
|-------|---------------|
| 16 (W-NOMINATE) | R (`Rscript`) not on PATH |
| 17 (External Validation) | Biennium outside 84th-88th SM coverage |
| 18 (DIME) | DIME CSV missing or biennium outside 84th-89th |
| 20 (Bill Text) | `bill_texts.csv` not in session data dir |
| 21 (TBIP) | Phase 20 output directory missing |
| 22 (Issue IRT) | Phase 20 output directory missing |
| 23 (Model Legislation) | No bill texts and no ALEC corpus |

## Breaking Changes

- **Results directories**: Existing results use old directory names (`results/kansas/.../04_irt/`). New runs will create directories with new names. Old results are not migrated — they remain valid but won't be found by `resolve_upstream_dir()` with new names.
- **Pipeline order**: Three MCMC phases now run back-to-back (05, 06, 07) before fast voting-pattern phases. Total runtime is identical — just reordered for logical grouping.

## Files Modified

| Category | Count |
|----------|-------|
| Directory renames | 25 |
| `__init__.py` MODULE_MAP | 1 file, ~70 entries |
| `analysis_name=` strings | 25 files |
| `resolve_upstream_dir()` calls | ~55 across 22 files |
| Dashboard PHASE_ORDER | 1 file, 25 entries |
| Synthesis data | 1 file, ~18 string refs |
| Justfile | 1 file |
| Graceful skip behavior | 7 files |
| Tests | 5 files |
| Documentation | ~80 files |

## Alternatives Considered

1. **Keep current numbering** — Rejected. Letter suffixes and gaps are confusing for new contributors.
2. **Renumber without pipeline expansion** — Rejected. The 11 standalone phases should be runnable in sequence.
3. **Interleave MCMC with fast phases** — Rejected. Clean logical grouping (ideal point estimation as a stage) is more intuitive than optimizing for interleaving.
