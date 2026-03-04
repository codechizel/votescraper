# Bipartite Bill-Legislator Network Deep Dive

**Phase 12 design research — Literature survey, ecosystem evaluation, and implementation recommendations**

Last updated: 2026-02-28

---

## Executive Summary

A bipartite bill-legislator network treats legislators and bills as two distinct node types, with edges representing votes. Unlike the existing Phase 6 co-voting network (which projects into a legislator-only graph weighted by Cohen's Kappa), the bipartite representation preserves the full structure of legislative voting — which legislators supported which specific bills. This enables bill-centric analysis (bridge bills, bill clustering, bill polarization) and statistically validated projections via backbone extraction (distinguishing genuine alliances from coincidental co-voting).

The literature is clear on the core tradeoff: co-voting projections lose information (bill identity, heterogeneous bill participation, coalition composition) but gain access to mature one-mode algorithms. Bipartite methods preserve information but require specialized tools. The backbone extraction literature (Neal 2014, 2021, 2022) provides a principled bridge: project to one-mode, but use null models to retain only statistically significant edges.

**Key finding for Tallgrass**: The highest-value contribution of Phase 12 is the **bill-centric perspective** — identifying bridge bills, bill clustering by coalition support, and bill polarization scores. This is genuinely new information not available from any current phase. The legislator-centric results will largely confirm what IRT and Phase 6 already show, but the statistically validated backbone provides a cleaner network than raw Kappa thresholding.

---

## Table of Contents

1. [Foundational Literature](#1-foundational-literature)
2. [Two-Mode Projection Methods](#2-two-mode-projection-methods)
3. [Backbone Extraction: The Methodological Core](#3-backbone-extraction-the-methodological-core)
4. [Bipartite-Native Metrics](#4-bipartite-native-metrics)
5. [Community Detection on Bipartite Graphs](#5-community-detection-on-bipartite-graphs)
6. [Signed Bipartite Networks for Roll-Call Voting](#6-signed-bipartite-networks-for-roll-call-voting)
7. [Bipartite vs Co-Voting: What the Literature Says](#7-bipartite-vs-co-voting-what-the-literature-says)
8. [Integration with IRT and Ideal Points](#8-integration-with-irt-and-ideal-points)
9. [Open Source Ecosystem](#9-open-source-ecosystem)
10. [Case Studies](#10-case-studies)
11. [Kansas-Specific Considerations](#11-kansas-specific-considerations)
12. [Recommendations for Implementation](#12-recommendations-for-implementation)
13. [References](#13-references)

---

## 1. Foundational Literature

### Origins: Fowler (2006)

James H. Fowler published two foundational papers in 2006 that established legislative network analysis as a field:

- **"Connecting the Congress: A Study of Cosponsorship Networks"** (*Political Analysis* 14(4): 456–487). Mapped cosponsorship networks of all 280,000 pieces of legislation in the U.S. House and Senate from 1973–2004. Introduced a "connectedness" measure using cosponsorship frequency and the number of cosponsors per bill to infer social distance between legislators. Connectedness predicted which members passed more floor amendments and predicted roll-call vote choice even after controlling for ideology and partisanship.

- **"Legislative Cosponsorship Networks in the U.S. House and Senate"** (*Social Networks* 28: 454–465). Companion paper analyzing network structure metrics on the same data.

### The Comparative Turn: Briatte (2016)

François Briatte's **"Network Patterns of Legislative Collaboration in Twenty Parliaments"** (*Network Science* 4(2): 266–271) built 150 cosponsorship networks from 27 parliamentary chambers in 19 European countries plus Israel. Showed that methods developed for the U.S. Congress are applicable across legislative systems, and released all data and code via the [parlnet GitHub repository](https://github.com/briatte/parlnet).

### The Bipartite-Native Turn: Lo, Olivella & Imai (2023/2025)

The most methodologically important recent paper is **"A Statistical Model of Bipartite Networks: Application to Cosponsorship in the United States Senate"** (*Political Analysis*, Cambridge Core). Lo, Olivella, and Imai proposed the bipartite Mixed Membership Stochastic Blockmodel (biMMSBM), which analyzes the bipartite structure directly rather than projecting it away.

Key finding: In the 107th Senate, bipartite analysis revealed three pathways for cross-party collaboration invisible in projected networks: (1) junior senators establishing common ground through low-stakes symbolic resolutions, (2) bill-specific norms of reciprocity, and (3) shared committee memberships predicting collaborative behavior. The projected network showed 0.75 average copartisan probability vs. 0.63 in the bipartite model — projection inflates apparent polarization by ~19%.

Software: the [NetMix R package](https://CRAN.R-project.org/package=NetMix).

### Backbone Extraction: Neal (2014, 2021, 2022)

Zachary P. Neal developed the systematic framework for extracting statistically significant edges from bipartite projections:

- **"The backbone of bipartite projections"** (*Social Networks* 39: 84–97, 2014). Introduced the Stochastic Degree Sequence Model (SDSM). Demonstrated on U.S. Senate bill sponsorship data.
- **"Comparing alternatives to the fixed degree sequence model"** (*Scientific Reports* 11: 23929, 2021, with Domagalski and Sagan). Compared five null models (FFM, FRM, FCM, FDSM, SDSM); SDSM emerged as the recommended practical alternative.
- **"backbone: An R Package to Extract Network Backbones"** (*PLOS ONE* 17: e0269137, 2022). Published the definitive implementation.

### Additional Key Papers

| Author(s) | Year | Venue | Contribution |
|---|---|---|---|
| Waugh, Pei, Fowler, Mucha, Porter | 2009 | arXiv | Modularity-based polarization measurement |
| Cho & Fowler | 2010 | *J. of Politics* | Network position predicts legislative success |
| Kirkland & Gross | 2014 | *Social Networks* | Within-session temporal dynamics |
| Andris, Lee, Hamilton et al. | 2015 | *PLOS ONE* | Exponential increase in U.S. House partisanship |
| Aref & Neal | 2020 | *Scientific Reports* | Frustration index for signed network balance |
| Candellone et al. | 2024 | *Adv. Complex Sys.* | Parameter sensitivity in signed bipartite community detection |
| Lee, Kim, Park, Jin | 2025 | arXiv:2512.11610 | Joint legislator-bill embedding via Euclidean LSIRM |
| Ferraro et al. | 2025 | *npj Complexity* | Statistically validated signed bipartite projections |

---

## 2. Two-Mode Projection Methods

A bipartite (two-mode) network has two node sets — legislators (L) and bills (B) — with edges only between sets. The biadjacency matrix **B** (|L| × |B|) has B_{ij} = 1 if legislator i voted Yea on bill j.

### One-Mode Projections

**Legislator projection**: **P = B · B^T** — entry P_{ij} counts how many bills legislators i and j both voted Yea on.

**Bill projection**: **R = B^T · B** — entry R_{jk} counts how many legislators voted Yea on both bills j and k.

### Projection Methods

| Method | Formula | When to Use |
|---|---|---|
| **Simple (binary)** | w_{ij} = Σ_p 𝟙(both connected to p) | Presence/absence only |
| **Newman (2001)** | w_{ij} = Σ_p 1/(N_p − 1) | Discount large-group participation; N_p = bill degree |
| **Jaccard** | \|N(i) ∩ N(j)\| / \|N(i) ∪ N(j)\| | Normalized similarity |
| **Cosine** | dot(row_i, row_j) / (\|\|row_i\|\| · \|\|row_j\|\|) | Angle-based similarity |
| **Resource allocation** (Zhou et al. 2007) | Asymmetric flow through bipartite structure | Recommendation systems |
| **Overlap** | \|N(i) ∩ N(j)\| / min(\|N(i)\|, \|N(j)\|) | When degree asymmetry is large |

Newman's method (from "Scientific collaboration networks: II" *Physical Review E* 64, 2001) is the most widely used in legislative analysis. The 1/(N_p − 1) term discounts for group size: two legislators who both voted on a bill with 100 other co-voters share a weaker implied connection than two who were among few supporters.

### Information Loss in Projection

The fundamental problem with all projections: **different bipartite structures can produce identical unipartite projections**. Two legislator pairs can have the same co-vote count but for completely different bills. The projection destroys:

- **Bill identity**: Which specific bills created each connection
- **Bill heterogeneity**: Whether co-votes were on contested or unanimous bills
- **Coalition composition**: Whether co-voting patterns reflect genuine affinity or coincidental agreement on easy votes
- **Clique inflation**: Every bill with k Yea votes creates k(k−1)/2 edges in the projection, massively inflating density and clustering coefficients

---

## 3. Backbone Extraction: The Methodological Core

Because raw bipartite projections produce dense, nearly complete graphs, backbone extraction identifies which edges represent genuine affinity vs. statistical noise. This is the single most important methodological contribution of the bipartite network literature.

### The Core Question

"Do legislators i and j co-vote more than expected, given how many bills each votes on and how many votes each bill receives?"

### Null Model Taxonomy

A comprehensive review by Simmons et al. (2024, *PLOS Complex Systems*) identified nine primary randomization methods classified by whether row (legislator) and column (bill) degree constraints are **Fixed** (exact), **Proportional** (on average), or **Equiprobable** (unconstrained):

| Model | Row Constraint | Column Constraint | Speed | Accuracy |
|---|---|---|---|---|
| **FDSM** | Fixed | Fixed | Slow (Monte Carlo) | Gold standard |
| **SDSM** | Proportional | Proportional | Fast (analytical) | Conservative but close to FDSM |
| **FFM** | None | None | Fast | Poor (too liberal) |
| **FRM** | Fixed | None | Moderate | Incomplete (ignores bill popularity) |
| **FCM** | None | Fixed | Moderate | Incomplete (ignores legislator activity) |
| **HM** (Hypergeometric) | Fixed | Proportional | Fast | Moderate |

### SDSM: The Recommended Method

The Stochastic Degree Sequence Model constrains both row and column degree sequences *on average* rather than exactly. This allows analytical (closed-form) computation of p-values rather than Monte Carlo simulation. Key results:

- **SDSM vs FDSM** (Neal et al. 2021): SDSM offers a "statistically conservative but close approximation of the computationally-impractical FDSM" and correctly recovers community structure even with weak signal.
- **Modularity recovery**: Both FDSM and SDSM achieved modularity ~0.49 with strong within-community clustering, vs. 0.15–0.39 for simpler alternatives (FFM, FRM, FCM).
- **Computational performance**: SDSM computed 1,000,000 edge probabilities in ~0.026 seconds on standard hardware; FDSM required ~80,000 random networks taking minutes to hours.

### Disparity Filter: Serrano, Boguñá & Vespignani (2009)

Operates on the *weighted one-mode projection* rather than the bipartite network itself. For each node, tests whether each edge's weight is statistically significant relative to a null model where the node's total weight is uniformly distributed across its edges. Key property: multiscale — does not privilege large-weight edges over small-weight ones. Limitation: operates post-projection, so cannot distinguish co-occurrence caused by genuine association vs. heterogeneous degree sequences.

### Python Implementation: BiCM

The **Bipartite Configuration Model** (BiCM) package provides Python-native backbone extraction. It computes the maximum-entropy null model that preserves degree sequences, then performs statistically validated monopartite projections with p-values on every edge.

```python
from bicm import BipartiteGraph
bg = BipartiteGraph()
bg.set_biadjacency_matrix(vote_matrix)
bg.solve_tool()
projected = bg.get_rows_projection()  # legislator-legislator with p-values
```

---

## 4. Bipartite-Native Metrics

### Centrality Measures

Standard centrality measures assume any node can connect to any other. In bipartite networks, edges only cross between modes, requiring adjusted normalization.

**Faust (1997)**: "Centrality in Affiliation Networks" (*Social Networks* 19: 157–191). First systematic treatment. A legislator's centrality is partly a function of the centrality of the bills they vote on, and vice versa. Found correlations of 0.89–0.99 among degree, closeness, betweenness, and eigenvector centrality in empirical affiliation data.

**Borgatti & Everett (1997)**: "Network analysis of 2-mode data" (*Social Networks* 19(3): 243–269). Proposed adjustments:
- **Degree**: Normalize by size of the *opposite* set (maximum possible connections)
- **Closeness**: Minimum distance between same-mode nodes is 2 (always passes through an opposite-mode node)
- **Betweenness**: Normalization must use bipartite-specific maxima
- **Eigenvector**: Sensitive to normalization choices due to block structure of bipartite adjacency matrix

### Clustering Coefficients

Triangles are impossible in bipartite networks (edges only cross between modes), so clustering must be redefined.

| Measure | Author(s) | Year | What It Captures |
|---|---|---|---|
| **4-cycle coefficient** | Robins & Alexander | 2004 | Pair-level reinforcement (two legislators repeatedly co-voting) |
| **Pairwise overlap** | Latapy, Magnien & Del Vecchio | 2008 | Neighborhood similarity (Jaccard, min, or max variants) |
| **Triadic closure** | Opsahl | 2013 | Three-actor clustering — the bipartite analogue of one-mode triangles |

**Opsahl's measure** (2013, *Social Networks* 35) is generally preferred for legislative data because it captures whether three legislators who pairwise co-vote also form a genuine triad — "my allies' allies are also my allies."

### Nestedness

NODF (Almeida-Neto et al. 2008, *Oikos*) measures whether specialists interact with subsets of the partners of generalists. **Not commonly used for legislative voting** — nearly all members vote on nearly all bills, making nestedness trivially high. More applicable to cosponsorship networks where participation is voluntary and sparse.

---

## 5. Community Detection on Bipartite Graphs

### Bipartite Modularity: Barber (2007)

Barber defined a bipartite-specific null model in "Modularity and community detection in bipartite networks" (*Physical Review E* 76: 066102). Unlike Newman's standard modularity, the null model only allows edges between the two modes:

**Null model edge probability**: P_{ij} = k_i · d_j / m

where k_i is the legislator degree, d_j is the bill degree, and m is total edges. This prevents the null model from creating impossible within-mode edges.

**BRIM Algorithm** (Bipartite Recursively Induced Modules): Alternates between optimizing legislator assignments and bill assignments. Relies on matrix multiplications, making it fast. Guaranteed to converge since modularity never decreases.

### biSBM: Larremore, Clauset & Jacobs (2014)

The bipartite Stochastic Block Model (*Physical Review E* 90(1): 012805) discovers communities directly in the bipartite structure without projection. Nodes are grouped by common connection patterns with nodes of the other type. Improves resolution limit by a factor of two over general SBM. Available in [Python](https://github.com/junipertcy/bipartiteSBM).

### LPAb+ (Label Propagation)

Hybrid algorithm combining modified label propagation with multistep greedy agglomeration. Outperforms seven other methods for binary bipartite networks while retaining fast time complexity.

### Signed Bipartite Community Detection

Candellone, van Kesteren, Chelmi & Garcia-Bernardo (2024) compared SPONGE and community-spinglass algorithms on U.S. House data (1990–2022). Critical warning: "both algorithms are highly susceptible to parameter choices" and "researchers should not take communities found at face value." Dense networks (like legislative voting) showed worse performance than sparse networks.

**Implication for Tallgrass**: Bipartite community detection results should be validated against known party structure and compared with IRT/clustering findings, never presented as standalone ground truth.

---

## 6. Signed Bipartite Networks for Roll-Call Voting

### The Encoding Problem

Legislative voting creates a *signed* bipartite network, not a simple binary one:
- **Yea = +1**, **Nay = −1**, **absent = 0**

Most bipartite network research focuses on cosponsorship (binary: sponsored or not). Roll-call voting requires the signed framework.

### Ferraro et al. (2025): Signed Bipartite Projection

Published as "Statistically validated projection of bipartite signed networks" (*npj Complexity*, 2025).

Classifies dyadic motifs into concordant and discordant categories:
- **Concordant**: V++ (both Yea), V−− (both Nay) — agreement
- **Discordant**: V+− or V−+ (one Yea, one Nay) — disagreement

Computes a signature S_{ij} = concordant count minus discordant count, then tests against null models (BiSRGM-FT, BiSCM-FT) to yield p-values for every projected edge. Output: a validated **signed** projection where positive edges are genuine alliances and negative edges are genuine opposition.

**This is the methodologically cleanest approach for legislative voting data.** Applied to U.S. Congress data from the 1st through 10th Congresses.

### Relationship to Phase 6 (Existing)

The existing Phase 6 uses Cohen's Kappa on the raw vote matrix to build a one-mode co-voting network. Kappa already corrects for chance agreement (analogous to what backbone extraction does), but it does not use the bipartite structure — it treats the vote matrix as a data source for pairwise similarity rather than as a network. The signed bipartite framework would complement Phase 6 by:
1. Providing a statistically principled alternative to Kappa thresholding
2. Producing a **signed** network (allies + opponents) rather than just a positive-weight network
3. Enabling bill-side analysis that Phase 6 cannot perform

---

## 7. Bipartite vs Co-Voting: What the Literature Says

### What Bipartite Analysis Adds

| What IRT/Co-Voting Captures | What Bipartite Analysis Adds |
|---|---|
| Legislator ideology (latent trait) | Bill-level characteristics and groupings |
| Item difficulty and discrimination | Which specific bills create unusual coalitions |
| Uncertainty in estimates (posterior) | Network topology (clustering, bridges, centrality) |
| One-dimensional (or 2D) latent space | Arbitrary-dimensional community structure |
| Pairwise agreement rates | Coalition composition on specific legislation |

### Published Direct Comparisons

Gu, Mucha, Loper, Olivella & Imai (2025, *Political Analysis*): Bipartite modeling uncovers cross-party collaboration that unipartite projection obscures. Projection produces "aggregation bias" and hyper-partisanship artifacts. Differences in the degree of polarization in the underlying bipartite network "can be obscured in unipartite projections, which show strong within-party ties regardless of the underlying bipartite structure."

Sinclair, Victor, Masket & Koger (2011): Agreement scores (network-based) and ideal point estimates "capture different aspects of polarization." Agreement scores are "more conducive to assumptions typically made in social network analysis."

Tumminello, Miccichè, Lillo, Piilo & Mantegna (2011, *PLOS ONE*): Naive projection conflates structural heterogeneity with meaningful relationships. Statistically validated projections are necessary.

### When Bipartite Genuinely Adds Value

1. **Bill-level questions**: Which bills are polarizing? Which create unusual coalitions? Which serve as "bridges" between factions?
2. **Factions within parties**: Bipartite community detection can identify sub-party factions that co-voting projections mask.
3. **High baseline agreement**: In legislatures with many unanimous votes (like Kansas, 82% Yea base rate), the co-voting projection is dominated by noise from easy votes. Bipartite structure enables bill-level filtering.
4. **Cross-party collaboration on low-stakes bills**: Lo et al. (2023) found collaboration patterns completely invisible in projected networks.

### When It Is Likely Redundant

1. **One-dimensional ideology**: IRT / W-NOMINATE already estimate this well.
2. **Strongly polarized two-party division**: Both approaches recover the same two-cluster structure.
3. **Without bill-level metadata**: Bipartite bill clusters are only useful if you can label them by topic/sponsor/committee.

---

## 8. Integration with IRT and Ideal Points

### The Emerging Synthesis

IRT and bipartite network analysis were historically separate traditions — IRT from psychometrics, networks from sociology/physics. Recent work bridges them:

**LSIRM** (Jeon, Jin, Schweinberger & Baugh, 2021, *Psychometrika*): The Latent Space Item Response Model jointly embeds respondents (legislators) and items (bills) in a shared latent space. IRT captures main effects (ability/difficulty); the latent space captures residual interactions.

**Euclidean Ideal Points** (Lee, Kim, Park & Jin, 2025, arXiv:2512.11610): Directly adapts LSIRM to roll-call data. Finds that conventional ideal point models "violate the triangle inequality, producing non-metric distances." Results on the 118th House: classification accuracy 0.80 (vs 0.72 for standard IRT), silhouette coefficient 0.861 (vs 0.778), APRE 0.45 (vs 0.23). Bill locations serve as interpretable anchors, clarifying cross-cutting cleavages.

**Key insight for Tallgrass**: IRT tells you *where* a legislator sits on the ideological spectrum; bipartite analysis tells you *which specific bills* they deviate from their expected position on, and whether those deviations cluster meaningfully. The two are complementary, not redundant.

---

## 9. Open Source Ecosystem

### Python Libraries

| Library | Key Features | Bipartite Support | Status |
|---|---|---|---|
| **NetworkX** | Projections, centrality, clustering, generators | `networkx.algorithms.bipartite` module — 5 projection methods, bipartite centrality, Latapy/Robins-Alexander clustering, `biadjacency_matrix()` | Active (v3.6+) |
| **scikit-network** | Louvain (Barber modularity), SVD embedding, PageRank, diffusion | First-class bipartite — biadjacency input, `embedding_row_`/`embedding_col_`, [French Assembly use case](https://scikit-network.readthedocs.io/en/latest/use_cases/votes.html) | Active (v0.33+) |
| **BiCM** | Maximum-entropy null model, statistically validated projections | Native bipartite — `get_rows_projection()`/`get_cols_projection()` with p-values | Active (May 2025) |
| **igraph** | Fast C core, Leiden integration | `Graph.Biadjacency()`, `bipartite_projection()` | Active (v0.11+) |
| **graph-tool** | SBM inference (bipartite-native, no projection) | Via `minimize_blockmodel_dl()` with `clabel`/`pclabel` | Active (v2.92+) |
| **leidenalg** | Leiden community detection with bipartite modularity | `find_partition()` with `Bipartite()` constructor | Active (v0.11+) |
| **birankpy** | HITS, CoHITS, BGRM, BiRank | Native bipartite ranking | Maintenance mode |
| **nxviz** | Circos, Hive, Arc, Matrix plots | `group_by="bipartite"` | Active (v0.7+) |

**scikit-network** is the closest match to Tallgrass's workflow — scikit-learn-style API, sparse matrix foundation, and an [official use case](https://scikit-network.readthedocs.io/en/latest/use_cases/votes.html) analyzing French National Assembly voting as a bipartite graph.

### R Packages

| Package | Key Features | Legislative Example |
|---|---|---|
| **backbone** | SDSM, FDSM, HM, disparity filter backbone extraction | [108th US Senate vignette](https://cran.r-project.org/web/packages/backbone/vignettes/senate.html) |
| **incidentally** | Generates incidence matrices from US Congress data; pairs with backbone | `incidence.from.congress()` — 2003-present |
| **NetMix** | biMMSBM mixed-membership model | US Senate cosponsorship (*Political Analysis*) |
| **bipartite** | Ecological bipartite analysis (modularity, nestedness, null models) | Ecology-focused but generalizable |
| **tnet** | Weighted two-mode centrality (Opsahl) | Integrated into bipartite package |
| **sbm** | Bipartite SBM with variational EM | Fungus-tree example |

**backbone** is the gold standard for backbone extraction, with a published legislative vignette. Already established precedent in Tallgrass for R subprocess integration (W-NOMINATE Phase 17, TSA Phase 15 CROPS/Bai-Perron).

### Recommended Stack for Tallgrass

| Component | Tool | Why |
|---|---|---|
| Bipartite construction | NetworkX | Already a dependency; `biadjacency_matrix()` |
| Backbone extraction | BiCM (Python) | Native Python, fast, p-values on projected edges |
| Bill-side community detection | scikit-network Louvain | Barber modularity, biadjacency input |
| Bipartite centrality | NetworkX bipartite module | Degree, betweenness, closeness already adapted |
| Visualization | matplotlib + NetworkX layouts | Consistent with existing phases |
| Validation | R backbone (optional) | SDSM cross-validation via subprocess |

---

## 10. Case Studies

### US Congress

| Study | Year | Data | Key Finding |
|---|---|---|---|
| Fowler | 2006 | House/Senate 1973–2004 (280K bills) | Connectedness predicts amendment success, vote choice |
| Waugh et al. | 2009 | House 1949–2009 | Modularity reveals polarization trends; underestimation in weak-party eras |
| Andris et al. | 2015 | House 1949–2012 (5M pairs) | Partisanship increasing exponentially for 60+ years |
| Lo, Olivella, Imai | 2023 | 107th Senate cosponsorship | Bipartite reveals hidden cross-party collaboration |
| Lee et al. | 2025 | 118th House roll-call | Euclidean LSIRM: accuracy 0.80 vs IRT 0.72; bill anchors clarify cross-cutting cleavages |
| Ferraro et al. | 2025 | 1st–10th Congress | Signed bipartite projection with validated positive/negative edges |

### European / Comparative

| Study | Year | Data | Key Finding |
|---|---|---|---|
| Briatte | 2016 | 20 parliaments, 150 networks | Methods generalize across legislative systems |
| Hix et al. | 2016 | 8th European Parliament (787 MEPs) | Roll-call + Twitter co-voting networks |
| Ringe & Wilson | 2016 | European Parliament | Covoting centrality as influence measure; cue-providers always more central |

### State Legislatures

| Study | Data | Key Finding |
|---|---|---|
| Kirkland | 2014 | Multiple states | Chamber size, term limits affect collaborative structure |
| Texas study | Texas State House, 73K+ bills | Cosponsorship timing and bipartisan collaboration patterns |

### Kansas

**No published bipartite network studies of the Kansas Legislature were found.** Kansas appears in aggregate state-level datasets (Shor-McCarty, DIME/CFscores) but dedicated network analysis of Kansas roll-call voting is an open niche.

---

## 11. Kansas-Specific Considerations

### Data Characteristics

- **Dense voting matrix**: ~170 legislators × ~400 roll calls with ~82% Yea base rate. The bipartite graph is dense — most legislators vote on most bills.
- **Supermajority structure**: ~72% Republican. Intra-party variation is more interesting than inter-party.
- **Near-unanimous bills dominate**: Standard filtering removes votes with minority < 2.5%, but even after filtering, many bills pass with comfortable margins.
- **34 veto override votes**: Cross-party coalition votes — ideal candidates for bipartite bridge-bill analysis.
- **No cosponsorship data**: Unlike Congress, Kansas Legislature data from kslegislature.gov does not include structured cosponsorship information. The bipartite network must be built from roll-call votes.

### What Phase 12 Would Reveal That No Current Phase Does

1. **Bridge bills**: Which specific bills attracted unusual cross-party coalitions? The veto overrides are obvious candidates, but there may be others.
2. **Bill polarization scores**: Per-bill measure of how strongly the vote split along party lines (beyond the simple "partisan/bipartisan" binary).
3. **Bill clustering by coalition**: Bills grouped not by topic (which we lack metadata for) but by the similarity of their voting coalitions. Two bills with very different subjects that attract the same coalition are politically linked.
4. **Statistically validated legislator network**: A cleaner version of the Phase 6 network where edges are retained only if co-voting exceeds chance expectation under the BiCM null model. Compared with Phase 6's Kappa threshold approach.
5. **Signed network structure**: Positive edges (alliance) and negative edges (opposition) rather than just positive-weight agreement.

### Limitations

- **No bill text/topic data**: Bill clustering can only be interpreted by manually examining the bills in each cluster. Without topic labels, interpretation requires domain knowledge.
- **Dense voting data**: Roll-call voting is denser than cosponsorship, reducing the discriminating power of some bipartite methods. Backbone extraction becomes more important, not less.
- **Small legislature**: 40 Senate + 125 House members. Some bipartite methods (biMMSBM) may be overkill for this scale.

---

## 12. Recommendations for Implementation

### Architecture

Phase 12 should be a companion to Phase 6 (Network), not a replacement. It uses the same upstream data (vote matrices from EDA, ideal points from IRT) but produces different outputs — bill-centric analysis plus a statistically validated projection.

### Proposed Analysis Components

**Tier 1 — Core (bill-centric, genuinely new)**:
1. Build bipartite graph (legislators × roll calls, edges = Yea votes)
2. Bill-side metrics: degree (popularity), bipartite betweenness (bridge bills), polarization score (party split ratio)
3. Bill-side community detection: Barber modularity via scikit-network or BRIM
4. Bill projection: bill-bill co-support network (which bills attract similar coalitions?)

**Tier 2 — Backbone (statistically validated legislator network)**:
5. BiCM backbone extraction: statistically validated one-mode projection with p-values
6. Compare backbone network with Phase 6 Kappa-threshold network (edge overlap, community agreement)
7. Identify edges present in backbone but missing from Kappa network (hidden alliances) and vice versa

**Tier 3 — Enrichment (complementary to existing phases)**:
8. Signed bipartite projection: concordant/discordant motifs → signed legislator network
9. Bipartite clustering coefficients (Opsahl triadic closure)
10. Cross-reference bridge bills with IRT bill discrimination parameters

### What to Skip

- **Nestedness (NODF)**: Not meaningful for dense roll-call voting data.
- **biMMSBM / NetMix**: Too complex for this scale; computational overhead not justified for ~170 legislators.
- **LSIRM joint embedding**: Interesting but essentially re-derives ideal points — we already have IRT.
- **Full FDSM backbone**: Computationally expensive; SDSM/BiCM provides adequate approximation.

### Technology Stack

- **Primary**: NetworkX bipartite module + BiCM for backbone extraction + scikit-network for bill-side Louvain
- **Visualization**: matplotlib with NetworkX bipartite_layout for small subsets, spring_layout for projected networks
- **R subprocess (optional)**: backbone package SDSM for cross-validation, if desired
- **Output**: Polars DataFrames, great_tables, HTML report via existing report infrastructure

---

## 13. References

### Foundational Methods

- Barber, M.J. (2007). Modularity and community detection in bipartite networks. *Physical Review E* 76: 066102.
- Borgatti, S.P. & Everett, M.G. (1997). Network analysis of 2-mode data. *Social Networks* 19(3): 243–269.
- Everett, M.G. & Borgatti, S.P. (2013). The dual-projection approach for two-mode networks. *Social Networks* 35.
- Faust, K. (1997). Centrality in affiliation networks. *Social Networks* 19: 157–191.
- Latapy, M., Magnien, C. & Del Vecchio, N. (2008). Basic notions for the analysis of large two-mode networks. *Social Networks* 30(1): 31–48.
- Newman, M.E.J. (2001). Scientific collaboration networks: II. Shortest paths, weighted networks, and centrality. *Physical Review E* 64: 016132.
- Opsahl, T. (2013). Triadic closure in two-mode networks. *Social Networks* 35.
- Robins, G. & Alexander, M. (2004). Small worlds among interlocking directors. *Computational & Mathematical Organization Theory* 10: 69–94.
- Zhou, T., Ren, J., Medo, M. & Zhang, Y.-C. (2007). Bipartite network projection and personal recommendation. *Physical Review E* 76: 046115.

### Backbone Extraction

- Neal, Z.P. (2014). The backbone of bipartite projections. *Social Networks* 39: 84–97.
- Neal, Z.P. (2022). backbone: An R Package to Extract Network Backbones. *PLOS ONE* 17: e0269137.
- Neal, Z.P., Domagalski, R. & Sagan, B. (2021). Comparing alternatives to the fixed degree sequence model. *Scientific Reports* 11: 23929.
- Neal, Z.P. & Neal, J.W. (2024). SDSM with edge constraints. Springer.
- Saracco, F. et al. (2022). Meta-validation of bipartite network projections. *Communications Physics* 5: 76.
- Serrano, M.Á., Boguñá, M. & Vespignani, A. (2009). Extracting the multiscale backbone of complex weighted networks. *PNAS* 106(16): 6483–6488.
- Simmons, B.I. et al. (2024). Pattern detection in bipartite networks: A review. *PLOS Complex Systems*.

### Legislative Network Analysis

- Andris, C. et al. (2015). The Rise of Partisanship in the U.S. House of Representatives. *PLOS ONE*.
- Aref, S. & Neal, Z.P. (2020). Detecting coalitions by optimally partitioning signed networks. *Scientific Reports*.
- Briatte, F. (2016). Network Patterns of Legislative Collaboration in Twenty Parliaments. *Network Science* 4(2): 266–271.
- Candellone, E. et al. (2024). Community detection in bipartite signed networks is highly dependent on parameter choice. *Advances in Complex Systems*.
- Cho, W.K.T. & Fowler, J.H. (2010). Legislative Success in a Small World. *The Journal of Politics* 72(1).
- Ferraro, G. et al. (2025). Statistically validated projection of bipartite signed networks. *npj Complexity*.
- Fowler, J.H. (2006). Connecting the Congress. *Political Analysis* 14(4): 456–487.
- Fowler, J.H. (2006). Legislative Cosponsorship Networks in the US House and Senate. *Social Networks* 28: 454–465.
- Kirkland, J.H. & Gross, J.H. (2014). Measurement and theory in legislative networks. *Social Networks*.
- Lo, A., Olivella, S. & Imai, K. (2023). A Statistical Model of Bipartite Networks. *Political Analysis*.
- Ringe, N. & Wilson, S.L. (2016). Pinpointing the Powerful. *Legislative Studies Quarterly*.
- Waugh, A.S. et al. (2009). Party Polarization in Congress. arXiv:0907.3509.

### IRT + Network Integration

- Jeon, M. et al. (2021). Mapping Unobserved Item-Respondent Interactions: A Latent Space Item Response Model. *Psychometrika*.
- Lee, S. et al. (2025). Euclidean Ideal Point Estimation From Roll-Call Data via Distance-Based Bipartite Network Models. arXiv:2512.11610.

### Community Detection

- Larremore, D.B., Clauset, A. & Jacobs, A.Z. (2014). Efficiently inferring community structure in bipartite networks. *Physical Review E* 90(1): 012805.
- Guimerà, R., Sales-Pardo, M. & Amaral, L.A.N. (2007). Module identification in bipartite and directed networks. *Physical Review E* 76: 036102.

### Nestedness

- Almeida-Neto, M. et al. (2008). A consistent metric for nestedness analysis. *Oikos* 117: 1227–1239.
- Staniczenko, P.P.A. et al. (2013). The ghost of nestedness in ecological networks. *Nature Communications* 4: 1391.

### Software

- BiCM Python package: [PyPI](https://pypi.org/project/bicm/), [GitHub](https://github.com/mat701/BiCM)
- backbone R package: [CRAN](https://cran.r-project.org/package=backbone)
- bipartiteSBM: [GitHub](https://github.com/junipertcy/bipartiteSBM)
- incidentally R package: [CRAN](https://cran.r-project.org/package=incidentally)
- NetMix R package: [CRAN](https://cran.r-project.org/package=NetMix)
- NetworkX bipartite module: [Docs](https://networkx.org/documentation/stable/reference/algorithms/bipartite.html)
- scikit-network: [Docs](https://scikit-network.readthedocs.io/), [Votes use case](https://scikit-network.readthedocs.io/en/latest/use_cases/votes.html)
