# ADR-0076: Audit Findings Resolution (A6-A18)

**Date:** 2026-03-02
**Status:** Accepted

## Context

The pipeline audit (ADR-0072) identified 18 findings (A1-A18). A1-A5, A7, A12, A14-A15 were fixed immediately. This ADR resolves the remaining 13 findings (A6, A8-A11, A13, A16-A18) after deep-dive research classified them as: 1 genuine code fix, 4 report improvements, 3 document-and-accept, and 5 already resolved. Full analysis: `docs/audit-findings-deep-dive.md`.

## Decisions

### A10: Bridge-Builder Detection — Code Fix

**Problem:** `detect_bridge_builder()` used betweenness centrality, which is 66-73% zeros when the co-voting graph is disconnected (4/8 bienniums). False "bridge-builder" narratives result.

**Fix:** Connectivity-aware detection. `compute_centralities()` now computes harmonic centrality (finite for disconnected graphs) and cross-party edge fraction. `detect_bridge_builder()` accepts `network_manifest` to check `{chamber}_n_components`:
- Connected (n_components == 1): uses betweenness (original behavior)
- Disconnected (n_components >= 2): uses harmonic centrality, role label changes to "Within-Party Connector"

**Files:** `network.py`, `synthesis_detect.py`, `synthesis_data.py`, `synthesis.py`, `test_synthesis_detect.py` (4 new tests).

### A6: Surprising Votes — Report Fix

**Problem:** 90% of surprising votes are false positives (predicted Yea, actual Nay) due to 73% Yea base rate. No explanation in report.

**Fix:** New `split_surprising_by_class()` function splits surprising votes into "Surprising Nay" (FP) and "Surprising Yea" (FN). Report shows both split tables plus combined, with interpretation section explaining base-rate mechanism and reframing as "unexpected dissent" vs "unexpected support".

**Files:** `prediction.py`, `prediction_report.py`.

### A11: IRT Sensitivity — Report Fix

**Problem:** Sensitivity table showed correlations but no interpretation. ROBUST/SENSITIVE classification existed only in console output.

**Fix:** Added `_add_sensitivity_interpretation()` to IRT report. Sensitivity table now includes "Status" column (ROBUST/SENSITIVE based on |r| > 0.95). Interpretation section explains supermajority mechanism, sign flips, and field-standard convention (Clinton-Jackman-Rivers 2004).

**Files:** `irt_report.py`.

### A16: Small-Group Warning — Report Fix

**Problem:** Senate Democrat group warning (N=7-11) printed to console but not in HTML report.

**Fix:** `_generate_hierarchical_key_findings()` now checks group sizes from `group_params` DataFrame. When any party has fewer than 20 legislators, adds key finding recommending flat IRT for individual positions.

**Files:** `hierarchical_report.py`.

### A17: BiCM Backbone — Report Fix + Code

**Problem:** Senate BiCM backbone extremely sparse (73-100% isolated). Uniform p-value threshold despite 10x difference in multiple-testing burden.

**Fix:**
- New `BICM_SIGNIFICANCE_SENATE = 0.05` constant (vs 0.01 for House), reflecting fewer comparisons
- Chamber-specific threshold passed to `extract_bicm_backbone()` in main loop
- New `_add_backbone_sparsity_caveat()` warns when >50% of legislators are isolated
- Analysis parameters table shows both thresholds

**Files:** `bipartite.py`, `bipartite_report.py`.

### A8: LCA Certainty — Document and Accept

**Problem:** All legislators have max_probability > 0.99 in most bienniums.

**Resolution:** Mathematically expected with 200+ binary indicators. New `_add_membership_certainty_note()` explains when all legislators are near-certain.

**Files:** `lca_report.py`.

### A9: Clustering Party Recovery — Document and Accept

**Problem:** k=2 always recovers party labels exactly (ARI ≈ 1.0).

**Resolution:** Added `party_labels` parameter to `compare_methods()`. Computes ARI(clustering vs party). Report annotates when ARI > 0.95, confirming party is the only discrete structure.

**Files:** `clustering.py`, `clustering_report.py`.

### A18: Bill Communities Party Mirror — Document and Accept

**Problem:** Bill communities consistently find 2 communities matching the party divide.

**Resolution:** Report now stores `best_bill_modularity` and adds quality gate: when modularity < 0.10, notes weak community structure mirroring the party divide.

**Files:** `bipartite.py`, `bipartite_report.py`.

## Consequences

- All 18 audit findings (A1-A18) now resolved
- Bridge-builder detection produces valid results on disconnected graphs
- Report readers get contextual interpretation for surprising votes, IRT sensitivity, and small-group limitations
- BiCM backbone is more informative for Senate with relaxed threshold
- No breaking changes to existing APIs (all new parameters are optional with backward-compatible defaults)
