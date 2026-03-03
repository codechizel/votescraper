# Network Analysis Deep Dive

A literature survey, code audit, and set of recommendations for the network analysis phase of the Kansas Legislature vote analysis pipeline.

**Date:** 2026-02-25
**Scope:** `analysis/11_network/network.py` (2,731 lines), `analysis/11_network/network_report.py` (1,121 lines), `tests/test_network.py` (581 lines), `analysis/design/network.md`, ADR-0010

---

## 1. Literature Survey: Network Analysis of Legislative Voting Data

### 1.1 The Python Library Landscape

Legislative network analysis has converged on three tiers of Python tooling, each with distinct tradeoffs:

**NetworkX** (v3.6.1, Dec 2025) is the most widely used. Pure Python, BSD-licensed, pip-installable. The dict-of-dicts adjacency representation is flexible but memory-heavy. For a ~165-node Kansas Legislature graph with ~13,500 potential edges, performance is irrelevant — NetworkX handles this instantly. Starting with v3.2, NetworkX introduced a backend dispatch system; by v3.6, algorithms like `leiden_communities()` dispatch to igraph/leidenalg transparently. Built-in community detection includes Louvain (with resolution parameter), greedy modularity (Clauset-Newman-Moore), label propagation, and Girvan-Newman. All centrality measures support weighted edges.

**igraph** (v0.11.8, Jun 2025; PyPI package now called `igraph`, the old `python-igraph` name deprecated since Sep 2022) wraps a C core, making it 10-100x faster than NetworkX for algorithmic operations. Community detection algorithms include Leiden (modularity + CPM), Louvain, Walktrap, Infomap, Spinglass, and Newman's leading eigenvector — all implemented in C. The `leidenalg` package (v0.11.0, Oct 2025), maintained by Vincent Traag (the Leiden algorithm's inventor), integrates directly with igraph and supports six quality functions: modularity, Reichardt-Bornholdt (configuration and ER null models), CPM, Significance, and Surprise. All support weighted edges via `weights='weight'`.

**graph-tool** (v2.98) is the fastest of the three (C++/Boost core with OpenMP), but cannot be pip-installed — it requires conda-forge or system packages. Its primary value is the reference implementation for Stochastic Block Model (SBM) inference via minimum description length: `minimize_blockmodel_dl()` and `minimize_nested_blockmodel_dl()`. A critical gotcha: edge weights are treated as multiplicities (integers), not arbitrary reals — real-valued weights must be passed via the `recs` parameter as edge covariates. For a 165-node partisan network, this performance advantage is irrelevant and the conda-only installation is a significant friction point.

**CDlib** (v0.4.0) is a meta-library wrapping ~39 community detection algorithms behind a uniform API. It accepts NetworkX graphs and converts to igraph internally as needed. Its value-add is the evaluation toolkit (modularity, NMI, ARI, conductance) and side-by-side algorithm comparison. Installation tiers: `pip install cdlib[C]` for C-backed algorithms on unix. Weight support varies by algorithm and must be checked per-method.

Other notable libraries:
- **NetBone** — backbone extraction from weighted networks, 20 methods (6 statistical, 13 structural, 1 hybrid), accepts NetworkX graphs. Bipartite-specific methods planned but not yet integrated.
- **graspologic** (Microsoft Research) — scikit-learn-compliant graph statistics, spectral embedding, hierarchical Leiden. More oriented toward neuroscience than political science.

| Library | Version | Install | Speed | Weighted | Community Detection | Best For |
|---------|---------|---------|-------|----------|-------------------|----------|
| NetworkX | 3.6.1 | pip | Slow | Yes | Louvain, Greedy, LP | Prototyping, small graphs, ecosystem |
| igraph | 0.11.8 | pip | Fast (C) | Yes | Leiden, Louvain, Infomap, Walktrap | Production, algorithmic operations |
| leidenalg | 0.11.0 | pip (needs igraph) | Fast | Yes | 6 quality functions, CPM | Best Leiden implementation |
| graph-tool | 2.98 | conda only | Fastest (C++) | Via covariates | SBM inference (Bayesian) | Large graphs, model selection |
| CDlib | 0.4.0 | pip | Varies | Varies | 39+ algorithms | Algorithm comparison |
| NetBone | — | pip | Fast | Yes | — | Backbone extraction |

### 1.2 Community Detection Algorithms

#### Louvain vs. Leiden

Louvain (Blondel et al., 2008) was the de facto standard for a decade: greedy modularity optimization through iterative node moving and community aggregation. Near-linear time.

Leiden (Traag, Van Eck, Waltman, 2019) fixes three critical problems with Louvain:

1. **Badly connected communities.** Up to 25% of Louvain communities can be badly connected and up to 16% completely disconnected. This happens when a "bridge" node between subcommunities is moved to a different community, severing the internal connection. Leiden's refinement phase can split communities during aggregation, *guaranteeing* all communities are well-connected.

2. **Non-convergence.** Running Louvain iteratively does not converge to a stable partition. Leiden provably converges to a partition where all subsets of all communities are locally optimally assigned.

3. **Better quality.** Leiden is faster AND produces higher modularity partitions on the same graphs.

**Bottom line:** There is no reason to use Louvain over Leiden in 2026. Leiden is strictly superior — faster, better quality, connectivity guarantees. The only reason Louvain persists is inertia. The Leiden paper ([Traag et al., Nature Scientific Reports, 2019](https://www.nature.com/articles/s41598-019-41695-z)) has been cited over 5,000 times.

#### Modularity and Its Resolution Limit

Modularity (Newman & Girvan, 2004) measures the fraction of edges within communities minus the expected fraction under a null model (configuration model). Range [-0.5, 1]; values > 0.3 indicate meaningful structure.

**Resolution limit** (Fortunato & Barthelemy, 2007): modularity optimization cannot detect communities smaller than ~sqrt(2m) where m is the number of edges. For our dense Kansas network with ~4,000 edges at threshold 0.40, this means communities smaller than ~89 nodes may be missed. Since the entire House is ~125 legislators, modularity's resolution limit likely forces k=2 even when finer structure exists.

This matters for Kansas specifically. While the state has a formal two-party system, Kansas politics — particularly in earlier bienniums (84th–88th, roughly 2011–2018) — was widely understood as a *three-faction* legislature: conservative Republicans, moderate Republicans, and Democrats. The moderate Republican bloc frequently broke with party leadership on education funding, Medicaid expansion, and tax policy, sometimes forming cross-party coalitions with Democrats. Modularity optimization may be structurally incapable of detecting this moderate wing as a distinct community because the faction (~30–40 legislators) falls below the resolution limit. This makes the three-faction question one of the most important analytical targets for community detection, and it requires a method that does not suffer from the resolution limit.

Tunable resolution (gamma parameter) helps but does not eliminate the structural problem. With gamma > 1, you favor smaller communities, but the interpretation becomes less clear.

#### Constant Potts Model (CPM)

CPM is **resolution-limit-free**. The resolution parameter gamma has a direct physical interpretation: the expected internal edge density of communities. Communities are groups of nodes whose internal edge density exceeds gamma. This means:

- Sweep gamma from 0 to 1 and observe how community structure changes across scales
- At low gamma, everything merges into one community
- At high gamma, communities fragment into cliques
- Stable partitions that persist across a range of gamma values are robust

The `leidenalg` library provides `resolution_profile_analysis()` for systematic scanning, and a common heuristic is to set gamma equal to the overall network edge density.

**CPM with Leiden** is the recommended combination for legislative network analysis: no resolution limit, interpretable resolution parameter, connectivity guarantee, fast enough for any realistic network size.

For the Kansas three-faction question, CPM is particularly valuable. Sweeping gamma would reveal whether there is a stable three-community partition at intermediate resolutions — conservative R, moderate R, and D — that persists across a range of gamma values. If the moderate Republican faction exists as a genuine community (internal density exceeding gamma), CPM will find it at the right resolution without the resolution-limit confound that plagues modularity. If the intra-Republican variation is continuous rather than discrete (as the clustering phase's within-party analysis suggests), CPM will show no stable k=3 plateau — which is itself an informative negative result.

#### Stochastic Block Models (SBMs)

SBMs are generative models that assume edges are placed probabilistically based on group membership. Fitting minimizes description length (MDL) — a Bayesian model selection criterion. The number of communities is determined automatically by the model, not by a resolution parameter.

Key variants:
- **Degree-Corrected SBM (DC-SBM):** Allows degree heterogeneity within blocks. Newman (2016) showed DC-SBM likelihood maximization is equivalent to generalized modularity maximization.
- **Nested SBM:** Hierarchical block structure. Best for model selection (avoids overfitting).
- **Weighted SBM:** Edge weights as covariates via graph-tool's `recs` parameter.

When SBMs add value: when you don't want to choose a resolution parameter, when you want principled model selection, or when you suspect non-assortative structure (core-periphery patterns). For Kansas, an SBM could answer the three-faction question without any resolution parameter tuning — it would independently determine whether the data supports 2, 3, or more communities. When SBMs may not be worth the cost: when CPM already provides resolution-limit-free community detection with less installation friction, and when multi-resolution exploration (watching how structure emerges at different scales) is more valuable than a single point estimate of k.

A 2025 study found that SBM clusterings from graph-tool can produce internally disconnected clusters on large real-world networks — the same problem Leiden was designed to fix.

#### Algorithm Comparison Table

| Algorithm | Weighted | Directed | Resolution Param | Determines k | Complexity | Python Library |
|-----------|----------|----------|-----------------|-------------|------------|---------------|
| Leiden (modularity) | Yes | Yes | Yes (gamma) | No | ~O(n) | leidenalg |
| Leiden (CPM) | Yes | Yes | Yes (edge density) | No | ~O(n) | leidenalg |
| Louvain | Yes | No | Yes (gamma) | No | ~O(n) | python-louvain, igraph |
| SBM | Via covariates | Yes | No | Yes (MDL) | O(E ln² V) | graph-tool |
| Infomap | Yes | Yes | No | Yes | O(m log n) | igraph, CDlib |
| Walktrap | Yes | No | No (dendrogram) | No | O(mn²) | igraph |
| Label Propagation | Yes | Some | No | Yes | O(m) | NetworkX, igraph |
| Spectral | Yes | No | Must specify k | No | O(n³) | scikit-learn |

### 1.3 Network Metrics for Legislative Analysis

#### Centrality Measures

**Betweenness centrality** is the most informative measure for legislative networks. It identifies brokers — legislators who bridge otherwise disconnected voting blocs. [Ringe, Victor & Fowler (2016)](https://onlinelibrary.wiley.com/doi/10.1111/lsq.12129) formally proved that cue-providers (influential legislators) have higher centrality than cue-receivers in co-voting networks, and validated this on European Parliament data. Fowler's earlier cosponsorship network research found that network connectedness predicts which members will pass more amendments on the floor.

**Eigenvector centrality** identifies legislators who vote with other well-connected legislators — the "core" of the majority coalition. More meaningful than degree because it captures connection quality, not just quantity.

**Degree/Strength** is a useful baseline but highly correlated with other measures, making it partially redundant ([Valente et al., PMC, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2875682/)).

**Closeness centrality** is the least distinctive in dense legislative networks — it tends to converge with eigenvector centrality.

**PageRank** provides random-walk importance that is robust to disconnected graphs. It's a good complement to eigenvector centrality (which requires special handling for disconnected components).

#### Modularity as Polarization

[Waugh, Pei, Fowler, Mucha & Porter (2009/2011)](https://arxiv.org/abs/0907.3509) established the foundational approach: compute modularity when the network is partitioned by party. Higher modularity = stronger partisan division. They showed US Congressional polarization has been rising since the 1970s, with the current level not seen since the early 1900s.

Two derived measures:
- **Divisiveness:** The modularity Q of the *optimal* (algorithm-found) partition — how much the network "wants" to split.
- **Solidarity:** How well party labels align with the algorithm-found partition — whether party is the dominant cleavage.

#### Small-World Properties

[Fowler (2006)](https://www.journals.uchicago.edu/doi/abs/10.1017/S002238160999051X) showed Congress exhibits small-world properties in cosponsorship networks. However, these metrics are more informative for sparse cosponsorship networks than for dense co-voting networks. In a dense weighted network, clustering coefficient and average path length are less discriminating.

### 1.4 Bipartite Projection and Backbone Extraction

Legislative data is naturally bipartite (legislators × bills). Projecting to a one-mode legislator-legislator co-voting network loses information and creates the "hairball" problem — a dense, near-complete graph where community detection degenerates.

#### The Naive Projection Problem

Two legislators who both vote "yes" on 90% of bills will show high co-voting weight even without genuine political affinity — they simply both vote with the majority on routine bills. Raw co-voting counts conflate genuine alliances with shared responses to easy votes. In Kansas, with an 82% Yea base rate, the naive projection produces a nearly complete graph where every pair of legislators has high agreement simply because they all vote "yes" on most bills.

The core insight: an edge in the co-voting network should represent *more agreement than expected by chance*. Raw counts do not distinguish "we both voted yes because the bill was popular" from "we both voted yes because we share a political alliance."

#### Null Models for Bipartite Projections

The field has developed statistical tests for whether observed co-voting exceeds chance expectation. These models ask: given each legislator's overall voting rate and each bill's overall passage rate, is this specific pair's agreement significantly higher (or lower) than what random voting with those marginals would produce?

**FDSM (Fixed Degree Sequence Model):** Monte Carlo simulation that preserves *exact* row and column sums of the bipartite matrix. For each simulation, the binary vote matrix is permuted such that every legislator casts the same total number of Yea votes and every bill receives the same total number of Yea votes as in the real data. The observed pairwise agreement is then compared to the distribution of agreements across simulations. Edges that fall above the 95th percentile of the null are "significantly positive" (allies); edges below the 5th percentile are "significantly negative" (antagonists). This is the gold standard but computationally expensive — generating valid permutations of a binary matrix with fixed margins requires MCMC sampling.

**SDSM (Stochastic Degree Sequence Model):** [Neal (2014)](https://www.sciencedirect.com/science/article/abs/pii/S0378873314000343) showed that the FDSM null distribution can be closely approximated by a hypergeometric distribution. For a pair of legislators who voted on *n* shared bills, where legislator A voted Yea on *a* of them and legislator B voted Yea on *b* of them, the expected number of co-Yea votes under the SDSM null is the hypergeometric expectation:

```
P(co-Yea = k) = C(a, k) * C(n-a, b-k) / C(n, b)
```

The p-value for each edge is computed from this distribution. [Neal & Domagalski (2021)](https://www.nature.com/articles/s41598-021-03238-3) found SDSM is a conservative but close approximation of FDSM, and correctly recovers known community structure even with weak signal.

**SDSM produces a signed backbone:** edges are classified as significantly positive (allies — more agreement than expected), significantly negative (antagonists — less agreement than expected), or non-significant (noise — agreement consistent with chance). This three-way classification is more informative than a simple threshold, and the signed backbone can be used directly for community detection.

The SDSM is currently R-only (Neal's [backbone](https://cran.r-project.org/web/packages/backbone/) package, well-maintained, with vignettes demonstrating application to the 108th US Senate). However, the core computation is straightforward to implement in Python using `scipy.stats.hypergeom`. For each pair of legislators:

```python
from scipy.stats import hypergeom

# n = shared votes, a = legislator_A yea count, b = legislator_B yea count
# observed = actual co-Yea count
p_positive = 1 - hypergeom.cdf(observed - 1, n, a, b)  # right tail
p_negative = hypergeom.cdf(observed, n, a, b)            # left tail
```

This is computationally cheap — ~13,500 pairs for a 165-legislator chamber, each requiring a single CDF evaluation.

#### Backbone Extraction for Unipartite Weighted Networks

When the network is already projected (as ours is, via Kappa), backbone extraction methods identify which edges are "locally significant" — edges that carry more weight than a node's typical edge.

**Disparity filter** ([Serrano, Boguna & Vespignani, PNAS, 2009](https://www.pnas.org/doi/10.1073/pnas.0808904106)): For each node, the disparity filter tests whether the distribution of edge weights is compatible with a uniform null model. If a node has degree k and total weight s, the null assumes each edge has weight s/k. An edge with weight w is significant at level alpha if:

```
p = (1 - w/s)^(k-1) < alpha
```

Edges significant for *either* endpoint are retained. The result is a sparse subgraph that preserves locally important connections while removing noise edges. The disparity filter is multiscale — it naturally preserves hub-spoke structures and dense-cluster connections that a global threshold would destroy.

**Key advantage over global thresholding:** A global Kappa threshold (our current approach) treats all legislators equally. But a moderate Republican who bridges conservative and Democrat blocs may have weaker-than-average edge weights to *both* sides. A global threshold risks severing exactly the cross-faction edges that are most analytically interesting. The disparity filter would preserve these edges because they are locally significant for the bridge legislator, even if they fall below the global threshold.

Python implementations:
- **[NetBone](https://pypi.org/project/netbone/)** — 20 backbone extraction methods (6 statistical including disparity filter, 13 structural, 1 hybrid). Accepts NetworkX graphs or pandas DataFrames. Provides evaluation metrics for comparing backbone techniques. Unipartite weighted networks only; bipartite-specific methods planned but not yet integrated.
- **[DerwenAI/disparity_filter](https://github.com/DerwenAI/disparity_filter)** — Focused Python implementation of the Serrano et al. disparity filter, built directly on NetworkX.

#### Where Our Approach Sits

Our pipeline uses Cohen's Kappa as edge weights, which partially addresses the naive projection problem. Kappa corrects for chance agreement on binary classification — if two legislators both vote Yea 90% of the time, Kappa discounts the expected 81% co-Yea rate and measures only the *excess* agreement. This is conceptually similar to the SDSM correction, which also adjusts for marginal voting frequencies.

However, the corrections are not identical:
- **Kappa** corrects for the *aggregate* chance-agreement rate between two raters. It uses a single summary statistic (proportion of expected agreement) that conflates the per-bill voting rates into one number.
- **SDSM** conditions on the *exact* marginal voting patterns — each legislator's total Yea count and each bill's total Yea count. This is a sharper correction because it accounts for bill-specific popularity (a routine unanimous vote contributes differently than a close party-line vote).

In practice, the difference may be small because our EDA phase already filters near-unanimous votes (minority < 2.5%), which removes most of the bills where SDSM and Kappa would diverge. The remaining contested votes have more balanced marginals where the two corrections converge.

The more consequential limitation is the **global Kappa threshold** (0.40). This is an arbitrary cut that treats all legislators equally, potentially severing locally significant cross-faction bridges. The disparity filter or an SDSM backbone would provide a more principled edge selection mechanism.

### 1.5 Open Source Projects and Academic Code

| Project | Focus | Key Methodology |
|---------|-------|-----------------|
| [jonhehir/congress-voting-networks](https://github.com/jonhehir/congress-voting-networks) | U.S. House & Senate | Agreement matrices, modularity, centrality (NetworkX) |
| [unitedstates/congress](https://github.com/unitedstates/congress) | Data collection | Roll call vote scraping (Python) |
| [DerwenAI/disparity_filter](https://github.com/DerwenAI/disparity_filter) | Backbone extraction | Serrano et al. (2009) disparity filter (NetworkX) |
| [Christoph Wolfram](https://christopherwolfram.com/projects/voting-modularity/) | Partisanship analysis | Network modularity over time |
| Neal `backbone` R package | Bipartite backbones | SDSM, FDSM, universal threshold |

Key academic references:
- **Waugh et al. (2009/2011)** — Modularity as polarization metric, US Congress
- **Moody & Mucha (2013)** — Portrait of Political Party Polarization, CONCOR + modularity
- **Ringe, Victor & Fowler (2016)** — Co-voting centrality predicts legislative influence (formal proof)
- **Neal (2014, 2020, 2021)** — Backbone extraction for bipartite projections
- **Spieksma et al. (2020)** — SDSM on US Senate co-voting (Nature Scientific Reports)
- **Traag, Van Eck & Waltman (2019)** — Leiden algorithm (Nature Scientific Reports)

---

## 2. Our Implementation: What We Do Well

### 2.1 Kappa-Based Edge Construction

Using Cohen's Kappa rather than raw agreement is a critical correction for Kansas's 82% Yea base rate. This is the same insight that drives backbone extraction methods (SDSM corrects for marginal frequencies; Kappa corrects for chance agreement). The NaN-as-no-edge treatment is more conservative and appropriate than the clustering phase's NaN-as-max-distance fill.

### 2.2 Comprehensive Centrality Analysis

Computing six centrality measures (degree, weighted degree, betweenness, eigenvector, closeness, PageRank) with proper handling for disconnected graphs is thorough. The per-component eigenvector centrality fallback is the correct approach — eigenvector centrality is undefined across disconnected components. The `distance = 1/Kappa` transform for path-based measures is simple, interpretable, and monotonically correct.

### 2.3 Bridge Legislator Detection

The `identify_bridge_legislators()` function and `find_cross_party_bridge()` plot (before/after network with bridge removed) are excellent for the nontechnical audience. Showing the network *disconnection* caused by removing a bridge legislator is more intuitive than a betweenness score.

### 2.4 Multi-Resolution Community Detection

Sweeping 8 resolution values [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0] for Louvain is a reasonable approach to exploring community structure at multiple scales. The resolution plot (n_communities + modularity vs. resolution) lets the reader see where the structure stabilizes.

### 2.5 Data-Driven Extreme Edge Analysis

Replacing hardcoded `check_tyson_thompson_edge_weights()` with the data-driven `check_extreme_edge_weights()` (ADR-0010) was the right call. The new function dynamically selects the `top_n` most extreme majority-party legislators by |xi_mean| and works across any session/chamber.

### 2.6 Threshold Sensitivity Sweep

Testing edge thresholds [0.30, 0.40, 0.50, 0.60] with the 4-panel plot (n_edges, density, n_components, modularity vs. threshold) with annotated split point and stability zone is methodologically careful. This lets readers assess how robust the findings are without understanding the underlying methodology.

### 2.7 Subnetwork Analysis

Both the high-discrimination subnetwork (|beta| > 1.5) and the veto override subnetwork (34 votes) add analytical value. The high-disc network isolates the most ideologically revealing votes; the veto override analysis targets a specific cross-party coalition formation mechanism unique to Kansas politics.

---

## 3. Issues Found

### 3.1 Bug: `except` Tuple Syntax (Crash in Python 3.14+)

**Severity: Critical — will crash at runtime**

Three locations use the Python 2-style `except` syntax without parentheses around multiple exception types:

| Line | Code | Context |
|------|------|---------|
| 248 | `except ValueError, ZeroDivisionError:` | `_build_network_from_vote_subset()` |
| 434 | `except ValueError, ZeroDivisionError:` | `compute_network_summary()` |
| 487 | `except nx.NetworkXError, nx.AmbiguousSolution, np.linalg.LinAlgError:` | `compute_centralities()` |

In Python 3, `except A, B:` means "catch exception A and bind it to variable B" — it does NOT catch both exception types. This means:
- Line 248: Only catches `ValueError`; `ZeroDivisionError` is silently assigned to a local variable
- Line 434: Same issue
- Line 487: Only catches `nx.NetworkXError`; the other two exception types are silently ignored

**Fix:** Change to `except (ValueError, ZeroDivisionError):` and `except (nx.NetworkXError, nx.AmbiguousSolution, np.linalg.LinAlgError):` with parenthesized tuples. This is actually a latent crash — in Python 3.12+ the bare-comma syntax raises a `SyntaxError`, but the `from __future__ import annotations` at the top of the file may be masking this in some execution contexts.

**Update**: Testing confirms this does NOT currently crash in Python 3.14 because the comma syntax, while semantically wrong (only catches the first exception), is technically valid syntax in Python 3 — it binds the exception to the second name. However, the behavior is incorrect: `ZeroDivisionError` and `np.linalg.LinAlgError` are never caught.

### 3.2 Louvain Instead of Leiden

**Severity: Moderate — methodological impurity**

The current implementation uses Louvain (`python-louvain` package). The literature is unambiguous: Leiden is strictly superior. Louvain can produce badly connected communities (up to 25% of communities, 16% completely disconnected), does not converge under iteration, and is slower. Leiden fixes all three problems while providing a connectivity guarantee.

For our ~165-node network, the practical impact is likely small — Louvain and Leiden produce similar results on small, well-structured graphs. But the theoretical problems are real, and the fix is straightforward: replace `community_louvain.best_partition()` with `leidenalg.find_partition()` using igraph. This would add `igraph` and `leidenalg` as dependencies.

The design doc (`analysis/design/network.md`) explicitly chose Louvain to avoid the igraph C-library dependency. This was a reasonable pragmatic choice in 2024, but igraph has been pip-installable since v0.10 (2022) and the dependency concern is now outdated.

### 3.3 Missing CPM Resolution Sweep

**Severity: Moderate — missing a more principled multi-resolution approach**

The current multi-resolution sweep uses Louvain with modularity at 8 gamma values. This still inherits modularity's resolution limit — even at gamma > 1, the objective function has structural limitations. CPM (Constant Potts Model) via leidenalg is resolution-limit-free and has a more interpretable resolution parameter (expected internal edge density).

A CPM resolution sweep from 0.1 to 0.9 would reveal whether sub-party structure exists at intermediate resolutions, without the resolution limit confound.

### 3.4 No Edge Significance Testing

**Severity: Low — Kappa partially addresses this**

Our edges are thresholded by Kappa value (default 0.40). While Kappa corrects for chance agreement on binary classification, it is not a formal significance test for bipartite projection. An SDSM-like approach (testing whether observed co-voting exceeds the hypergeometric expectation given marginal voting patterns) would be more principled.

However, Kappa is a reasonable proxy, and implementing a Python SDSM from scratch (since the R `backbone` package has no Python equivalent) may not be worth the effort given the project's scope. The disparity filter (available via netbone) could serve as a complementary backbone extraction method that does not require bipartite-specific logic.

### 3.5 Kappa Distance for Betweenness: 1/Kappa vs. 1 - Kappa

**Severity: Low — both are valid, but worth documenting**

The code uses `distance = 1/Kappa` for path-based centrality measures. An alternative is `distance = 1 - Kappa`, which maps naturally to [0, 1] and is consistent with the clustering phase's distance metric. The design doc chose 1/Kappa for its monotonic relationship and interpretability. Both are valid; the important thing is consistency within the network phase, which is maintained.

Note that 1/Kappa produces larger distance ratios at the low end (Kappa 0.40 → distance 2.50 vs. Kappa 0.80 → distance 1.25), effectively amplifying differences between weak agreements. Whether this amplification is desirable depends on whether you believe weak-agreement bridges should have lower betweenness (1/Kappa) or similar betweenness (1 - Kappa).

---

## 4. Code Quality Audit

### 4.1 Duplicated Logic

`_build_network_from_vote_subset()` and `build_kappa_network()` both construct networks from Kappa matrices with similar node/edge logic. The shared structure is:

1. Convert to numpy distance matrix
2. Compute pairwise Kappa (or use precomputed)
3. Build NetworkX graph with node attributes
4. Add edges above threshold with `weight=kappa` and `distance=1/kappa`

`_build_network_from_vote_subset()` computes Kappa from a raw vote matrix subset; `build_kappa_network()` uses the precomputed Kappa matrix from EDA. The duplication is in the graph construction and attribute assignment, not the Kappa computation. A shared `_build_graph_from_kappa_array()` helper could consolidate this.

### 4.2 Community Composition Duplication

`analyze_community_composition()` computes party breakdown per community. `plot_community_network()` partially re-derives this information for labeling purposes (using `_community_label()`). The plot function should use the composition DataFrame rather than re-scanning the graph.

### 4.3 Magic Numbers

A few magic numbers remain in the code:

- `min_shared=20` in `_build_network_from_vote_subset()` — should reference the `MIN_VOTES` constant from the module or accept it as a parameter
- `top_n=2` in `check_extreme_edge_weights()` — hardcoded default, reasonable but should be a module-level constant for visibility
- Plot sizing constants (e.g., `figsize=(14, 10)`, `figsize=(12, 5)`) scattered across plot functions — could be consolidated like the clustering module's named constants

### 4.4 Dead Code

`_community_label()` has a fallback path to `"Community {id}"` when composition is not provided. In practice, composition is always provided by `analyze_community_composition()`, so this fallback is unreachable. It's harmless but could be simplified.

### 4.5 Type Annotation Gaps

Several functions return complex nested structures (dicts of dicts, tuples of DataFrames) without TypedDict or NamedTuple annotations. Key examples:

- `check_extreme_edge_weights()` returns a plain dict with string keys and mixed-type values
- `compute_network_summary()` returns a dict
- `detect_communities_multi_resolution()` returns a tuple of `(dict[float, dict[str, int]], pl.DataFrame)`

These are readable but not type-checkable. TypedDict or frozen dataclasses would improve IDE support and catch errors earlier.

---

## 5. Test Coverage Analysis

### 5.1 Current Coverage

38 tests across 10 classes. Well-tested areas:

| Area | Tests | Coverage Quality |
|------|-------|-----------------|
| Network construction | 7 | Good — threshold, NaN, self-loops, attributes |
| Centrality computation | 4 | Good — columns, ranges, star graph |
| Community detection | 3 | Good — resolution sweep, modularity range, two-clique recovery |
| Community composition | 4 | Good — n_other, Independent handling, invariant test |
| Party comparison | 2 | Good — perfect alignment, random baseline |
| Within-party network | 2 | Adequate — party filtering |
| Network summary | 2 | Adequate — keys, components |
| Threshold sweep | 2 | Adequate — all thresholds, monotonicity |
| Cross-party bridge | 3 | Good — finding, threshold sensitivity, removal impact |
| Community label | 5 | Good — all label paths |

### 5.2 Untested Functions

| Function | Risk | Priority |
|----------|------|----------|
| `check_extreme_edge_weights()` | Medium — new ADR-0010 function, data-driven selection logic | High |
| `build_high_disc_network()` | Medium — IRT-filtered subnetwork construction | Medium |
| `build_veto_override_network()` | Medium — rollcall motion filtering | Medium |
| `build_cross_chamber_network()` | Low — simple union of two graphs | Low |
| `_build_network_from_vote_subset()` | Medium — inline Kappa computation + `except` bug | High |
| All 13 plot functions | Low — visual verification only | Low |
| `network_report.py` (30+ functions) | Low — report assembly | Low |
| `main()` | Low — integration | Low |
| `compute_party_centrality_summary()` | Low — simple aggregation | Low |

**Highest priority:** `check_extreme_edge_weights()` and `_build_network_from_vote_subset()` both contain logic that could silently produce wrong results. The exception syntax bug in `_build_network_from_vote_subset()` means `ZeroDivisionError` during Kappa computation is not caught and would crash the run.

### 5.3 Missing Test Patterns

- **No property-based tests.** The community composition invariant test (counts sum to total) is a good start, but more invariants could be tested: all nodes assigned to exactly one community, modularity in [-0.5, 1], edge weights positive after thresholding.
- **No regression test for the `except` bug.** A test with perfectly correlated or constant voting vectors (which trigger `ZeroDivisionError` in `cohen_kappa_score`) would catch this.
- **No test for edge weight distribution.** The `plot_edge_weight_distribution()` function parses within-R, within-D, and cross-party edges — this categorization logic should be tested independently.

---

## 6. Recommendations

### 6.1 Fix: Exception Tuple Syntax (P0)

Three `except` statements need parenthesized tuples. This is a correctness bug that silently drops exception handling for all but the first exception type.

```python
# Before (broken):
except ValueError, ZeroDivisionError:

# After (correct):
except (ValueError, ZeroDivisionError):
```

Three locations: lines 248, 434, 487 of `network.py`.

### 6.2 Upgrade: Louvain to Leiden (P1)

Replace `python-louvain` with `igraph` + `leidenalg`. The Leiden algorithm is strictly superior: faster, better quality, connectivity guarantees. The implementation change is contained:

1. Add `igraph>=0.11` and `leidenalg>=0.11` to `pyproject.toml`
2. Replace `community_louvain.best_partition(G, weight="weight", resolution=res, random_state=RANDOM_SEED)` with:
   ```python
   import igraph as ig
   import leidenalg as la

   ig_graph = ig.Graph.from_networkx(G)
   partition = la.find_partition(
       ig_graph,
       la.RBConfigurationVertexPartition,
       resolution_parameter=res,
       weights='weight',
       seed=RANDOM_SEED,
   )
   # Convert back to {node: community_id} dict
   ```
3. Update constant names from `LOUVAIN_RESOLUTIONS` to `COMMUNITY_RESOLUTIONS`
4. Add CPM sweep alongside modularity sweep (see 6.3)
5. Remove `python-louvain` dependency
6. Create ADR documenting the switch

**Risk:** Results will change slightly (Leiden produces higher-quality partitions). Cross-session comparisons against previously-generated results will shift. This is expected and desirable.

**Note:** The design doc's concern about igraph being a "C-library dependency that complicates installation" is outdated — igraph has been pip-installable since v0.10 (2022). Pre-built wheels are available for all major platforms including Apple Silicon.

### 6.3 Enhancement: CPM Resolution Profile (P1)

Add a CPM resolution sweep alongside the existing modularity sweep. CPM is resolution-limit-free and has a physically interpretable resolution parameter (expected internal edge density). This is promoted to P1 because it directly addresses the three-faction question.

```python
# CPM sweep
for gamma in np.linspace(0.1, 0.9, 9):
    partition = la.find_partition(
        ig_graph,
        la.CPMVertexPartition,
        resolution_parameter=gamma,
        weights='weight',
        seed=RANDOM_SEED,
    )
```

The CPM results would answer the central question for Kansas: is the moderate Republican wing a genuine community? If the three-faction structure (conservative R, moderate R, D) is real, the CPM sweep should show a stable k=3 plateau across a range of gamma values — the partition persists because moderate Republicans have higher internal agreement density with each other than with either conservative Republicans or Democrats. If no such plateau exists, the intra-Republican variation is continuous, not factional, and the k=2 finding holds.

This is especially important for earlier bienniums (84th–88th) where the moderate wing was most politically active. Running CPM across all 8 bienniums would show whether the three-faction structure has collapsed over time — a politically significant finding in itself.

### 6.4 Refactor: Extract Shared Graph Builder (P2)

Consolidate duplicated graph construction logic from `build_kappa_network()` and `_build_network_from_vote_subset()` into a shared `_build_graph_from_kappa_array(kappa_arr, slugs, ip_dict, threshold, ...)` helper.

### 6.5 Tests: Exception Bug Regression + check_extreme_edge_weights (P1)

1. **Regression test for exception handling:** Create a test with constant voting vectors (all Yea or all Nay) that trigger `ZeroDivisionError` in `cohen_kappa_score`. Verify the exception is caught rather than crashing.
2. **Test for `check_extreme_edge_weights()`:** Build a synthetic graph with known extreme legislators and verify the function selects the correct ones and computes accurate edge weight statistics.

### 6.6 Enhancement: Modularity as Polarization Metric (P3)

Compute and report the modularity of the party-labeled partition (separate from the algorithm-found partition) as a formal polarization metric, following Waugh et al. (2009). This is a single additional call to `community_louvain.modularity()` (or igraph's equivalent) with the party partition. The resulting Q value is directly comparable to US Congressional polarization literature.

### 6.7 Enhancement: Backbone Extraction (P3)

Replace or supplement global Kappa thresholding with backbone extraction. Two approaches, in order of value:

**Option A: Disparity filter (unipartite, on the existing Kappa network).** The disparity filter tests whether each edge is locally significant for at least one of its endpoints. This preserves cross-faction bridges that a global threshold would sever — exactly the edges most relevant to the three-faction question. A moderate Republican's connections to both conservative Republicans and Democrats may be individually weak but locally significant (their *strongest* cross-faction ties).

Implementation: use the `netbone` package or `DerwenAI/disparity_filter` on the weighted Kappa network. A single alpha parameter (significance level, typically 0.05) replaces the arbitrary Kappa threshold. Compare the disparity-filtered backbone against the current threshold-filtered graph to assess whether bridge legislators gain or lose centrality.

**Option B: SDSM backbone (bipartite, from the raw vote matrix).** Implement the hypergeometric significance test for each legislator pair using `scipy.stats.hypergeom`. This produces a signed backbone with three edge types: significantly positive (allies), significantly negative (antagonists), and non-significant (noise). The signed backbone enables richer analysis — antagonist edges are invisible in the current Kappa-thresholded network.

Implementation is ~30 lines of scipy code per chamber. The main decision is the significance threshold (typically alpha=0.05 with Bonferroni or FDR correction for the ~13,500 pairwise tests). SDSM would be most informative as a *complement* to Kappa, not a replacement — run both and compare which edges survive each method.

Neither approach is urgent because:
- Kappa already corrects for base-rate agreement (partially addressing the SDSM concern)
- The threshold sensitivity sweep [0.30, 0.60] shows qualitative findings are robust
- The primary audience cares about findings, not edge selection methodology

But for the three-faction question specifically, the disparity filter would be genuinely useful — it naturally preserves the weak-but-locally-significant cross-faction ties that define moderate Republicans.

### 6.8 Deferred: graph-tool SBM (P4)

SBM inference via graph-tool would provide Bayesian model selection for the number of communities without any resolution parameter. For the Kansas three-faction question, an SBM would independently determine whether the data supports k=2 or k=3 communities — a clean answer without resolution parameter tuning. The degree-corrected nested SBM would also reveal whether the three-faction structure is hierarchical (moderates as a sub-block within Republicans) or flat (three independent blocs).

However, CPM via Leiden addresses the same question more flexibly: the resolution sweep shows *where* the three-faction structure emerges and whether it's stable, rather than just giving a point estimate. The conda-only installation remains a significant friction point. If CPM reveals a clear k=3 plateau, the SBM question ("is k=3 statistically justified?") becomes redundant. If CPM shows no plateau, the SBM would likely agree that k=2 is optimal.

Recommendation: defer until after the CPM implementation. If CPM results are ambiguous (no clear plateau, but hints of sub-structure), revisit SBM as a tiebreaker.

---

## 7. Summary

| Finding | Severity | Recommendation | Priority | Status |
|---------|----------|---------------|----------|--------|
| `except` tuple syntax bug (3 locations) | Critical | Fix with parenthesized tuples | P0 | **Done** (ADR-0029) |
| Louvain instead of Leiden | Moderate | Replace python-louvain with igraph+leidenalg | P1 | **Done** (ADR-0029) |
| No tests for `check_extreme_edge_weights()` | Moderate | Add unit tests | P1 | **Done** (3 tests) |
| No regression test for `except` bug | Moderate | Add ZeroDivisionError trigger test | P1 | **Done** (1 test) |
| Missing CPM resolution sweep | High | Add alongside modularity sweep — key to three-faction question | P1 | **Done** (ADR-0029) |
| Duplicated graph builder | Low | Extract shared helper | P2 | **Done** (`_graph_from_kappa_matrix()`) |
| No polarization metric | Low | Report party-partition modularity | P3 | **Done** (`compute_party_modularity()`) |
| No backbone extraction | Moderate | Disparity filter preserves cross-faction bridges | P3 | **Done** (`disparity_filter()`) |
| No SBM inference | Low | Deferred until CPM results available | P4 | Deferred |

All P0–P3 items implemented. The Louvain-to-Leiden migration is documented in ADR-0029. Test coverage increased from 34 to 53 tests (19 new tests for CPM, party modularity, disparity filter, extreme edge weights, except regression, and shared graph builder).

---

## References

1. Traag, V.A., Waltman, L. & Van Eck, N.J. From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports* 9, 5233 (2019). [doi:10.1038/s41598-019-41695-z](https://www.nature.com/articles/s41598-019-41695-z)
2. Blondel, V.D., Guillaume, J.-L., Lambiotte, R. & Lefebvre, E. Fast unfolding of communities in large networks. *J. Stat. Mech.* P10008 (2008).
3. Fortunato, S. & Barthelemy, M. Resolution limit in community detection. *PNAS* 104(1), 36-41 (2007).
4. Newman, M.E.J. & Girvan, M. Finding and evaluating community structure in networks. *Physical Review E* 69, 026113 (2004).
5. Ringe, N., Victor, J.N. & Fowler, J.H. Pinpointing the Powerful: Covoting Network Centrality as a Measure of Political Influence. *Legislative Studies Quarterly* 41(4), 739-766 (2016). [doi:10.1111/lsq.12129](https://onlinelibrary.wiley.com/doi/10.1111/lsq.12129)
6. Waugh, A.S., Pei, L., Fowler, J.H., Mucha, P.J. & Porter, M.A. Party Polarization in Congress: A Network Science Approach. *arXiv:0907.3509* (2009). [arXiv](https://arxiv.org/abs/0907.3509)
7. Moody, J. & Mucha, P.J. Portrait of Political Party Polarization. *Network Science* 1(1), 119-121 (2013). [doi:10.1017/nws.2012.3](https://www.cambridge.org/core/journals/network-science/article/abs/portrait-of-political-party-polarization1/F4D2FC8C75B3A160CE68BC3A4A4F6736)
8. Fowler, J.H. Legislative Cosponsorship Networks in the US House and Senate. *Social Networks* 28(4), 454-465 (2006).
9. Neal, Z. The backbone of bipartite projections: Inferring relationships from co-authorship, co-sponsorship, co-attendance and other co-behaviors. *Social Networks* 39, 84-97 (2014). [doi:10.1016/j.socnet.2014.06.001](https://www.sciencedirect.com/science/article/abs/pii/S0378873314000343)
10. Neal, Z., Domagalski, R. & Sagan, B. Comparing alternatives to the fixed degree sequence model for extracting the backbone of bipartite projections. *Scientific Reports* 11, 23929 (2021). [doi:10.1038/s41598-021-03238-3](https://www.nature.com/articles/s41598-021-03238-3)
11. Serrano, M.A., Boguna, M. & Vespignani, A. Extracting the multiscale backbone of complex weighted networks. *PNAS* 106(16), 6483-6488 (2009). [doi:10.1073/pnas.0808904106](https://www.pnas.org/doi/10.1073/pnas.0808904106)
12. Peixoto, T.P. Inferring modular network structure. *graph-tool documentation* (2024). [Tutorial](https://graph-tool.skewed.de/static/doc/demos/inference/inference.html)
13. Rossetti, G., Milli, L. & Cazabet, R. CDLIB: a Python Library to Extract, Compare and Evaluate Communities from Complex Networks. *Applied Network Science* 4, 52 (2019). [doi:10.1007/s41109-019-0165-9](https://link.springer.com/article/10.1007/s41109-019-0165-9)
14. Choi, S.-S., Cha, S.-H. & Tappert, C.C. A Survey of Binary Similarity and Distance Measures. *JIIS* 2010.
15. Valente, T.W., Coronges, K., Lakon, C. & Costenbader, E. How Correlated Are Network Centrality Measures? *Connect.* 28(1), 16-26 (2008). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2875682/)
