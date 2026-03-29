# ADR-0130: Bootstrap Support Values for Hierarchical Clustering

**Date:** 2026-03-28
**Status:** Accepted

## Context

Phase 09 (Clustering) produces hierarchical dendrograms with global quality metrics (cophenetic correlation, silhouette scores, cross-method ARI) but no per-branch confidence measures. Users could see that "the tree is generally good" but couldn't distinguish reliable branches from fragile ones. This matters because intra-party variation in Kansas is continuous — most sub-group splits are noise, but the top-level party split is rock-solid.

The standard approach in phylogenetics is bootstrap support: resample the data, rebuild the tree, and count how often each clade appears. This directly answers "how much does confidence change as we move down a branch?"

## Decision

Add bootstrap support values to Phase 09 hierarchical clustering:

1. **Resampling:** Resample roll-call columns (votes) with replacement, B=1000 times.
2. **Rebuild:** Recompute pairwise Cohen's Kappa distance matrix from the resampled vote matrix, rebuild the dendrogram using average linkage.
3. **Count:** For each internal node (branch) in the original tree, count the fraction of bootstrap trees that contain the same clade (same set of legislators grouped below that node).
4. **Annotate:** Display bootstrap percentages on dendrogram and icicle chart plots, color-coded: green (>=95%, strong), gold (70-95%, moderate), red (<50%, unreliable).

**Vectorized Kappa:** Pairwise Kappa is computed via matrix multiplication (not pairwise loops), making 1,000 replicates for 126 legislators feasible in ~3 seconds.

**CLI flags:** `--skip-bootstrap` (skip computation), `--bootstrap-n N` (override replicate count).

**Constants:**
- `BOOTSTRAP_N_REPLICATES = 1000`
- `MIN_SHARED_VOTES_BOOTSTRAP = 10` (minimum shared votes for a pair's Kappa in bootstrap)

## Consequences

**Positive:**
- Per-branch confidence percentage directly answers "how reliable is this grouping?"
- Confirms the global metrics: 82nd House top split = 100%, but 78% of branches below 50%
- Color-coded annotations make dendrograms self-explanatory for nontechnical audiences
- Minimal runtime cost (~3-7 seconds per chamber with vectorized Kappa)

**Negative:**
- Adds ~130 lines to `clustering.py` (3 functions + 2 constants)
- Bootstrap on the vote columns tests stability to vote composition, not to legislator sampling — this is the standard phylogenetic approach but tests a specific kind of robustness

**Trade-offs:**
- B=1000 balances precision (SE of a 50% estimate = 1.6%) with runtime
- `MIN_SHARED_VOTES_BOOTSTRAP=10` is conservative; below this, Kappa estimates are too noisy to be meaningful
