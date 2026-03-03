# Network Analysis Design Choices

**Script:** `analysis/11_network/network.py`
**Constants defined at:** `analysis/11_network/network.py:45-55`

## Assumptions

1. **Pairwise voting agreement encodes legislative relationships.** Cohen's Kappa agreement matrices (from EDA) capture the strength and direction of voting similarity between every pair of legislators, correcting for the 82% Yea base rate. These pairwise relationships are naturally modeled as a weighted graph.

2. **Centrality identifies structurally important legislators.** Legislators with high betweenness centrality serve as "bridges" between voting blocs. This is distinct from IRT ideal points (which measure ideological position) and party loyalty (which measures caucus reliability). A moderate-ideology legislator may have low betweenness if they vote identically to their party, or high betweenness if they uniquely connect otherwise-disconnected groups.

3. **Community detection may find finer structure than k=2 clustering.** Clustering found k=2 (party split) as optimal, but network communities operate on a different principle — they maximize within-community edge density relative to a null model. Multi-resolution Louvain at finer resolutions may reveal subcommunities (moderate Rs, progressive Ds) that global clustering missed.

4. **Chambers are independent.** House and Senate are analyzed separately, consistent with all upstream phases. Cross-chamber comparison uses equated ideal points for node positioning but does not imply shared community structure.

5. **NaN Kappa means "no observed connection."** Unlike clustering (which filled NaN with max distance), network analysis treats NaN Kappa as absence of an edge. Unknown = no data = no connection. This is more conservative and avoids injecting artificial structure from unobserved pairs.

## Parameters & Constants

| Constant | Value | Justification | Code location |
|----------|-------|---------------|---------------|
| `KAPPA_THRESHOLD_DEFAULT` | 0.40 | "Substantial" on Landis-Koch scale; keeps only meaningful agreement | `network.py:45` |
| `KAPPA_THRESHOLD_SENSITIVITY` | [0.30, 0.40, 0.50, 0.60] | Sweeps from "fair" to "almost perfect" agreement | `network.py:46` |
| `LEIDEN_RESOLUTIONS` | [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0] | Sub-1.0 merges communities; >1.0 splits them; wide range for exploration | `network.py:47` |
| `CPM_GAMMAS` | [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50] | CPM resolution-limit-free sweep; gamma = expected internal edge density | `network.py:48` |
| `HIGH_DISC_THRESHOLD` | 1.5 | \|beta\| > 1.5 matches IRT design doc recommendation for discriminating bills | `network.py:48` |
| `TOP_BRIDGE_N` | 15 | Top bridge legislators to report; ~10% of House caucus | `network.py:49` |
| `TOP_LABEL_N` | 10 | Nodes to label on network plots; balances readability with coverage | `network.py:50` |
| `RANDOM_SEED` | 42 | Reproducibility; consistent with all upstream phases | `network.py:51` |
| `PARTY_COLORS` | R: #E81B23, D: #0015BC | Standard political colors; consistent with clustering | `network.py:52` |

## Methodological Choices

### Edge weight: Cohen's Kappa

**Decision:** Edge weights are Cohen's Kappa values from EDA agreement matrices.

**Why:** Kappa corrects for the 82% Yea base rate. Raw agreement inflates similarity between all legislator pairs. Kappa = 0.40 means agreement beyond chance at the "substantial" level. This is consistent with EDA and clustering, which both use Kappa as their primary similarity metric.

### Kappa threshold: 0.40 default with sensitivity sweep

**Decision:** Only create edges for pairs with Kappa > 0.40. Sweep thresholds [0.30, 0.40, 0.50, 0.60] for sensitivity.

**Why:** The threshold controls graph density. Too low (0.20) creates a near-complete graph where every legislator connects to every other — community detection degenerates. Too high (0.70) creates a sparse graph with many isolates — centrality measures become meaningless. The 0.40 "substantial" level on the Landis-Koch scale is a principled default.

**Sensitivity sweep:** Reports n_edges, density, n_components, and modularity at each threshold. If the qualitative findings (community structure, bridge legislators) change dramatically across thresholds, the results are threshold-dependent and should be interpreted cautiously.

### NaN Kappa: No edge (differs from clustering)

**Decision:** NaN Kappa entries produce no edge. Unknown = no observed connection.

**Why:** In clustering (distance matrix), NaN must be filled because every pair needs a distance. Max-distance fill was conservative. In networks, absence of an edge is a natural representation of "no data." This avoids injecting artificial structure. The consequence is that legislators with many NaN Kappa values (low participation overlap) may appear as low-degree nodes or isolates, which is informative.

### Community detection: Leiden + CPM

**Decision:** Use Leiden community detection (`leidenalg` via `igraph`) with both modularity optimization (RBConfigurationVertexPartition) and CPM (CPMVertexPartition) sweeps. Replaced Louvain (ADR-0029).

**Why Leiden over Louvain:** Leiden (Traag et al., 2019) is strictly superior — faster, better quality, and guarantees well-connected communities. Louvain can produce up to 25% badly connected communities. The prior concern about igraph being a C-library dependency is obsolete: igraph has been pip-installable since v0.10 (2022).

**Modularity sweep:** Resolution < 1.0 merges communities (fewer, larger groups); resolution > 1.0 splits them (more, smaller groups). Sweeping 8 values from 0.5 to 3.0 lets us observe how community structure changes from coarse (party-level) to fine (potential subcaucuses).

**CPM sweep:** The Constant Potts Model (Traag et al., 2011) is resolution-limit-free — it can detect communities of any size, including subcaucuses smaller than sqrt(2m) edges. This is critical for Kansas, where the moderate Republican wing (~30-40 legislators) falls below modularity's theoretical detection floor (Fortunato & Barthelemy, 2007). Gamma sets the expected internal edge density; higher gamma → more, smaller communities.

### Path centrality distance: 1/Kappa

**Decision:** For path-based centrality measures (betweenness, closeness), use distance = 1/Kappa.

**Why:** High Kappa (strong agreement) should correspond to short path distance. The simplest transform is reciprocal: Kappa = 0.80 → distance = 1.25; Kappa = 0.40 → distance = 2.50. Alternatives (1 - Kappa, -log(Kappa)) were considered but 1/Kappa is interpretable and monotonically relates agreement to proximity.

### High-discrimination subnetwork: |beta| > 1.5

**Decision:** Build a separate network using only bills with IRT discrimination |beta| > 1.5.

**Why:** High-discrimination bills are those where the probability of a Yea vote changes sharply with a legislator's ideal point — they are the most "partisan" or "ideologically revealing" votes. Computing Kappa on only these bills should produce a cleaner signal with less noise from unanimous or near-random votes.

**Threshold:** 1.5 matches the IRT design doc's recommendation. Approximately 20-30% of bills exceed this threshold.

### Extreme edge weight analysis: Data-driven

**Decision:** Identify the most ideologically extreme majority-party legislators by `|xi_mean|` and compare their intra-party edge weights to the party median. Replaces a prior hardcoded check on two specific senators.

**Why:** The original `check_tyson_thompson_edge_weights()` hard-coded two Senate slugs. The analysis concept — "do the most extreme majority-party members have weaker within-party connections?" — is session-independent and should work on any biennium. `check_extreme_edge_weights()` dynamically selects the `top_n` (default 2) majority-party legislators with the highest absolute IRT ideal point. It runs for both chambers, not just Senate.

**Parameters:**
- `top_n=2` — Number of extreme legislators to analyze. Keeps the output focused while surfacing the most prominent outliers.
- Majority party determined by node count in the graph (handles any party composition).
- Reports per-legislator mean/median/min intra-party edge weight and gap vs party median.

### Polarization metric: Party modularity

**Decision:** Compute modularity of the party-labeled partition (Waugh et al., 2009) as a quantitative measure of partisan polarization.

**Why:** Party assortativity (Newman's attribute assortativity) measures whether edges are preferentially within-party, but modularity captures how well the party labels partition the graph into dense communities. Higher party modularity = stronger polarization. This is a single number that can be tracked across bienniums.

### Backbone extraction: Disparity filter

**Decision:** Apply the disparity filter (Serrano et al., 2009) to extract the statistically significant backbone of the network.

**Why:** The full Kappa network at threshold 0.40 can have thousands of edges, making visualization cluttered. The disparity filter tests whether each edge weight is anomalously high given the node's degree distribution. At α=0.05, it retains only edges that are statistically significant for at least one endpoint. This preserves the structurally important connections while removing redundant ones.

**Algorithm:** For each node with degree k, compute the p-value for each edge: p = (1 - w/s)^(k-1), where w is the edge weight and s is the total weight (strength). Edges with p < α for at least one endpoint are retained.

### Bipartite network: Skipped

**Decision:** Do not build a bipartite (legislator × bill) network.

**Why:** Kappa already captures the co-voting relationship between legislators in a more statistically principled way (chance-corrected). A bipartite projection to a one-mode legislator network would approximate what Kappa already provides directly. The high-discrimination subnetwork (above) addresses the flag about bill-level signal.

## Downstream Implications

### For Prediction (Phase 7)
- Centrality measures (betweenness, eigenvector) are candidate features for vote prediction models
- Community membership may capture non-linear relationships that IRT ideal points miss
- Bridge legislators may be harder to predict (they vote with different blocs on different issues)

### For Interpretation
- Network visualization provides an intuitive complement to IRT number lines
- Community structure at multiple resolutions shows the hierarchy of legislative coalitions
- Edge weight distributions reveal how within-party and cross-party agreement differ quantitatively
- Bridge legislator annotations (red ring, betweenness centrality ranking bar chart) make structural importance visually accessible to nontechnical audiences
- Threshold sweep plots include stability zone shading and default threshold line so readers can see how robust findings are without understanding the methodology
