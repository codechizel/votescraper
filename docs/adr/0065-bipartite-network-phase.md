# ADR-0065: Bipartite Bill-Legislator Network (Phase 12)

**Date:** 2026-02-28
**Status:** Accepted

## Context

Phase 6 (Network) builds a **legislator-only** co-voting network weighted by Cohen's Kappa. This collapses the bipartite structure (legislators × bills) into a one-mode projection, losing all bill-centric information. The deep dive (`docs/bipartite-network-deep-dive.md`) found no published bipartite network analysis of the Kansas Legislature, and identified two high-value contributions: (1) bill polarization scores, bridge bills, and bill clustering by coalition; (2) statistically validated backbone extraction via the Bipartite Configuration Model (BiCM), which provides an analytically principled alternative to Phase 6's Kappa thresholding + disparity filter.

## Decision

Create Phase 12 as a standalone bipartite network analysis phase:

1. **Bipartite graph construction** from the binary vote matrix (Yea = edge). Legislator nodes carry party and IRT ideal points; bill nodes carry IRT discrimination and rollcall metadata.

2. **Bill polarization** scores via |pct_R_yea − pct_D_yea| per bill, with minimum voter threshold. Identifies party-line vs bipartisan votes.

3. **Bipartite betweenness centrality** and **bridge bill identification** — bills with high betweenness that connect otherwise separate partisan blocs, cross-referenced with IRT discrimination.

4. **Newman-weighted bill projection** and **Leiden community detection** to cluster bills by coalition support (not by topic).

5. **BiCM backbone extraction** using `bicm>=3.1` (MIT, PyPI, Saracco et al. 2015/2017). Fits a maximum-entropy null model preserving degree sequences, computes analytical p-values for co-voting edges, and retains only statistically significant edges (p < 0.01).

6. **Phase 6 comparison** (soft dependency): edge Jaccard overlap, community NMI/ARI, and "hidden alliances" table showing cross-party edges found by BiCM but missed by Kappa thresholding.

## Consequences

- Adds bill-centric analysis that Phase 6 cannot provide — polarization scores, bridge bills, and bill communities are new capabilities
- BiCM backbone provides a statistical test for co-voting significance, compared to Phase 6's arbitrary Kappa threshold
- New dependency: `bicm>=3.1` (pure Python, MIT, minimal dependency footprint)
- Phase 6 comparison is a soft dependency — skipped with a warning if Phase 6 hasn't run
- ~50 new tests added to the suite
- Phase runs in the pipeline after Phase 6 (`just pipeline` includes `just bipartite`)
