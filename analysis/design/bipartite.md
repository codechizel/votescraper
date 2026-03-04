# Bipartite Bill-Legislator Network (Phase 12) Design Choices

## Assumptions

1. **Yea-only edges preserve meaningful structure:** Only Yea votes create edges in the bipartite graph. Nay agreement (both voting against) is invisible. This is the standard encoding for legislative two-mode networks (Fowler 2006, Neal 2014) but means the bill projection captures co-support, not co-opposition. Phase 6's Kappa network captures both.

2. **Degree-preserving null model is appropriate:** BiCM preserves marginal degree sequences (how many bills each legislator votes Yea on, and how many Yea votes each bill receives). Edges that survive this null are co-voting relationships not explained by individual activity levels alone. This is the strongest possible null for bipartite networks (Saracco et al. 2017).

3. **Bills are independent observations:** The Newman projection and BiCM treat each bill as independent. Violated when bills are amendments to the same parent or part of a package, but roll call voting in Kansas is sufficiently independent for this approximation.

## Parameters & Constants

| Parameter | Value | Justification |
|-----------|-------|---------------|
| BICM_SIGNIFICANCE | 0.01 | Conservative p-value for backbone. Dense vote matrix generates many edges; 0.01 limits false positives. |
| BILL_POLARIZATION_MIN_VOTERS | 10 | Minimum Yea+Nay votes for polarization score. Excludes bills with too few voters for meaningful party comparison. |
| NEWMAN_PROJECTION | True | Newman (2001) 1/(k−1) discount prevents high-activity legislators from dominating bill-bill edge weights. |
| BILL_CLUSTER_RESOLUTIONS | [0.5, 1.0, 1.5, 2.0, 3.0] | Leiden resolution sweep. Coarser than Phase 6 (fewer bills than legislators). |
| TOP_BRIDGE_BILLS | 20 | Report table length for bridge bills. |
| BACKBONE_COMPARISON_THRESHOLD | 0.40 | Matches Phase 6 default Kappa threshold for fair comparison. |
| RANDOM_SEED | 42 | Project-wide reproducibility seed. |

## Methodological Choices

### Why BiCM (not FDSM)

The Fixed Degree Sequence Model (FDSM) generates null networks by random rewiring, requiring Monte Carlo samples for p-value estimation. BiCM (Saracco et al. 2015) computes p-values analytically via the Poisson-Binomial distribution — faster, exact, and deterministic. Both preserve the same null (degree sequences), but BiCM avoids Monte Carlo noise.

### Why unsigned bipartite (not signed)

Signed bipartite networks encode both Yea (+1) and Nay (-1) edges. While theoretically richer, signed backbone extraction lacks mature tooling, and the Yea-only encoding is the field standard for co-sponsorship and roll call networks. Phase 6's Kappa network already captures agreement on both Yea and Nay — Phase 12 complements it with the bill-centric perspective.

### Why Newman weighting (not simple overlap)

Simple overlap counts the number of shared legislators between two bills. This over-represents high-activity legislators who vote Yea on everything. Newman (2001) discounts each legislator's contribution by 1/(k−1), where k is their total Yea count. This is standard practice for one-mode projections of bipartite networks.

### Why Leiden (not BRIM)

Barber's bipartite modularity (BRIM) detects communities in the original bipartite graph. However, our bill projection is a one-mode weighted graph, making standard Leiden modularity optimization appropriate. BRIM would require a separate implementation and its communities mix legislator and bill nodes, complicating interpretation.

## Downstream Implications

- **Phase 11 (Synthesis):** Bill polarization scores and bridge bills could enrich the narrative with bill-centric findings (e.g., "HB 123 was the most bipartisan bill of the session").
- **Phase 6 cross-reference:** Backbone comparison quantifies how much structure the Kappa threshold misses (or invents). High Jaccard = methods agree; low Jaccard with many BiCM-only edges = Kappa threshold is too conservative.
- **No new features for downstream phases:** Phase 12 is primarily descriptive and validating. Bill communities do not feed into prediction or profile phases.
