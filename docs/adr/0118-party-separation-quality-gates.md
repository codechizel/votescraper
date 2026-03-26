# ADR-0118: Party Separation Quality Gates Across the Pipeline

**Date:** 2026-03-15
**Status:** Accepted

## Context

A deep dive into the 79th (2001-2002) PCA report revealed that PCA PC1 captures intra-Republican factionalism rather than the party divide in 7 of 14 Kansas Senate sessions (78th-83rd, 88th). This propagates through the entire pipeline:

- **PCA-informed IRT init** points the sampler at the factional axis
- **1D IRT** converges cleanly but measures the wrong dimension (party d < 1.5 vs d > 4 in normal sessions)
- **Hierarchical IRT** party means converge to near-zero separation (79th: gap = 0.589, ratio 0.22σ)
- **2D IRT** dimensions are swapped (Dim 2 separates parties, not Dim 1)
- **Tier 2 quality gate** has a circular dependency on PCA PC1
- **Dynamic IRT** inherits axis contamination from static IRT sign correction
- **Cross-session correlations** drop to r ≈ 0 at axis-swap boundaries

The root cause is that the pipeline implicitly assumes "maximum variance = ideology," with no phase validating whether its estimated axis actually separates parties. See `docs/pca-ideology-axis-instability.md` for the full analysis.

## Decision

Add party separation (Cohen's d between Republican and Democrat mean ideal points) as a quality metric at seven critical decision points in the pipeline:

### R1: Party-Aware PCA Initialization (`init_strategy.py`)

`detect_ideology_pc()` computes point-biserial correlation between each PC and party membership. When PC2 has stronger party correlation than PC1 (|r| > 0.30), auto-swap to use PC2 for ideology init.

### R2: 1D IRT Party Separation Gate (`irt.py`)

After extraction, compute Cohen's d between party means. Flag as `axis_uncertain` in convergence summary when d < 1.5. Write `convergence_summary.json` so downstream phases can consume the flag.

### R3: Tier 2 Quality Gate Fix (`canonical_ideal_points.py`)

Replace the PCA PC1 rank correlation check (circular dependency) with a direct party separation check (d > 1.5). PCA correlation kept as secondary diagnostic, not gating.

### R4: Hierarchical Minimum Separation Guard (`hierarchical.py`, `hierarchical_2d.py`)

Soft `pm.Potential` penalty when `mu_party[1] - mu_party[0] < 0.5`. Prevents near-zero party separation without hard constraining. Gap/sigma ratio logged as diagnostic.

### R5: Dynamic IRT Canonical Reference (`dynamic_irt.py`)

Sign correction prefers canonical ideal points over raw Phase 05 output. Per-period party-d check flags periods with d < 1.5 as axis-uncertain.

### R6: PCA Report Axis-Swap Warning (`pca.py`, `pca_report.py`)

Compute party-d on PC1 and PC2. When PC2 separates parties better (d > 2.0), inject a yellow warning banner in the HTML report.

### R7: 2D IRT Dimension Swap Detection (`irt_2d.py`, `hierarchical_2d.py`)

After extraction, check if Dim 2 separates parties better than Dim 1. If so, swap columns in both posterior and DataFrame, then re-run sign check.

## Consequences

**Positive:**
- Pipeline can detect and flag axis misalignment without depending on PCA axis ordering
- Tier 2 quality gate no longer has circular dependency on contaminated PCA
- Dynamic IRT uses horseshoe-corrected canonical scores for sign reference
- Human-readable warnings in PCA and IRT reports for affected sessions

**Negative:**
- Party separation threshold (d = 1.5) is empirically derived from Kansas data; may need adjustment for other legislatures
- Soft minimum separation penalty in hierarchical model adds a non-standard prior
- Dynamic IRT canonical score loading adds complexity to the data loading path

**Known limitation:** Party separation (d > 1.5) is necessary but not sufficient for correct dimension identification. The hierarchical model's party-pooling prior guarantees some party separation on Dim 1 by construction, making the party-d gate easy to pass even when the dimension doesn't match the unsupervised ideology axis. A systemic audit found 6/28 chamber-sessions where the canonical dimension passes all party-separation gates but disagrees with W-NOMINATE Dim 1 (r as low as 0.33). Manual PCA overrides (`analysis/pca_overrides.yaml`) now provide stable dimension assignments for these sessions. The W-NOMINATE gate (ADR-0123, demoted to diagnostic-only) retains the cross-validation as a reported metric without auto-swapping.

**Key references:**
- Bafumi et al. (2005), Clinton/Jackman/Rivers (2004) — IRT identification
- Lauderdale & Clark (2024) — theory-driven dimension identification (IRT-M)
- Shin/Lim/Park (2024) — L1-based ideal points (rotational invariance)

## Validation

Full 79th (2001-2002) pipeline run `79-260314.3` completed with all gates active:

- PCA axis-swap warning fired for Senate (PC2 d=5.21 > PC1 d=0.29)
- 1D IRT flagged as axis_uncertain (Senate d=1.19 < 1.5)
- Tier 2 quality gate passed 2D Dim 1 via party-d=6.05 (old PCA correlation gate would have rejected)
- Hierarchical min-sep guard improved Senate gap from 0.22σ (pre-R4) to 0.48σ
- 2D IRT dimension swap detected and corrected
- Canonical routing: Senate source `hierarchical_2d_dim1` (d=4.36) — the party-pooled 2D model

The 79th is the worst case in the dataset (all pathologies simultaneously). Pipeline now produces correct canonical ideology estimates for this session.
