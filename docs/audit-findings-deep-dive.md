# Pipeline Audit Findings Deep Dive: A6-A18

A systematic investigation of the 8 open audit findings from the 2026-03-02 pipeline review. Each finding was traced to root cause, evaluated against the political science literature, and classified as either a code fix, a report/documentation fix, or an accepted structural limitation.

**Last updated:** 2026-03-02

---

## Summary Table

| # | Finding | Root Cause | Category | Action |
|---|---------|-----------|----------|--------|
| A6 | Prediction: 90% FP asymmetry in surprising votes | Base rate skew (73% Yea) + confidence-error ranking | Report fix | Add interpretation section + per-class split |
| A8 | LCA: degenerate class probabilities | Mathematical inevitability with 200+ binary indicators | Document | Expected behavior across all 8 bienniums |
| A9 | Clustering: trivial k=2 party split | 1D IRT bimodality + silhouette selection | Document + minor diagnostic | ARI-against-party metric; within-party parquets intentionally dead |
| A10 | Network: betweenness sparsity (66-73% zeros) | Disconnected party cliques at kappa=0.40 | Code fix | Harmonic centrality fallback for synthesis bridge detection |
| A11 | IRT: House threshold sensitivity (r=0.65-0.80) | Near-unanimous votes carry intra-party signal in supermajority legislature | Report fix | Add interpretation section with ROBUST/SENSITIVE classification |
| A16 | Hierarchical: small Senate Democrat groups | J=2 groups with N=7-11 below literature thresholds | Report fix | Surface existing warning in HTML report |
| A17 | Bipartite: sparse BiCM backbone (Senate) | Dense partisan voting fully explained by degree-preserving null model | Report fix + minor code | Chamber-adaptive threshold; density guard in report |
| A18 | Bipartite: bill communities mirror party split | Genuinely one-dimensional legislative structure | Report fix | Modularity quality gate in report |

**Verdict:** 1 genuine code fix (A10), 4 report/documentation improvements (A6, A11, A16, A17), 3 document-and-accept (A8, A9, A18).

---

## A6. Prediction: 90% False-Positive Asymmetry in Surprising Votes

### The Finding

18/20 surprising votes in the 91st House are false positives (model predicted Yea, actual Nay). The pattern holds across all 8 bienniums: FP% ranges from 55% to 90% (mean 74.7%).

### Root Cause

The `find_surprising_votes()` function (`prediction.py:966-1030`) ranks all wrong predictions by `confidence_error = |P(Yea) - actual|`. This metric is mathematically symmetric: a FP with P(Yea)=0.99 has the same confidence_error (0.99) as a FN with P(Yea)=0.01. The asymmetry comes from the probability distribution, not the metric.

Three compounding effects:

1. **Base rate skew.** With 73-82% Yea base rate, the model's probability distribution is right-skewed. Far more observations sit at P(Yea) > 0.95 than at P(Yea) < 0.05. Errors in the high-probability region produce high confidence_error values.

2. **IRT feature amplification.** The `xi_x_beta` interaction (legislator ideal point times bill discrimination) pushes predictions toward extremes. For the majority of legislator-bill pairs, this interaction predicts Yea, concentrating probability mass near P(Yea) = 1.0.

3. **Selection effect.** When ranking all errors by confidence_error, the right tail (high P(Yea), actual Nay) is fatter than the left tail (low P(Yea), actual Yea), so FPs dominate the top-N.

Cross-biennium correlation between Yea base rate and FP% is r=0.49 — confirming the mechanism.

### What the Report Shows

The surprising votes table (`prediction_report.py:604-642`) presents legislator name, bill, actual/predicted vote, P(Yea), and |Error|. There is **no interpretation section** — every other major section in the prediction report has one (`_add_vote_interpretation`, `_add_feature_importance_interpretation`, `_add_per_legislator_interpretation`, `_add_passage_interpretation`). The surprising votes section is the only one that presents raw numbers without explanation.

### Literature Context

W-NOMINATE addresses base rate effects via APRE (Aggregate Proportional Reduction in Error), which measures improvement over a majority-class predictor. The prediction deep dive (`docs/prediction-deep-dive.md:167-175`) already considered and correctly rejected class weighting for the vote model — the 82/18 split is not extreme, and AUC=0.98 regardless. The models are correctly configured.

The harm-of-class-imbalance-corrections literature (van den Goorbergh et al. 2022) confirms that class weighting degrades calibration in prediction tasks where the base rate is genuine (not a sampling artifact). Kansas's Yea base rate reflects real legislative behavior, not sampling bias.

### Assessment

This is a **report/documentation fix**, not a code or methodology issue. The mathematical behavior is correct. The deficiency is that the report does not explain the pattern.

### Remediation

1. **Add `_add_surprising_votes_interpretation()` to `prediction_report.py`.** Compute and display FP/FN counts, explain the base-rate mechanism in plain English, reframe FP-dominant surprising votes as "unexpected dissent" (legislators voting Nay on bills they were expected to support).

2. **Split surprising votes into per-class tables.** Show "Top 10 Surprising Nay Votes" (unexpected dissent) and "Top 10 Surprising Yea Votes" (unexpected support) separately, guaranteeing both error types are visible. This eliminates the base-rate-driven asymmetry in the presentation and produces more analytically useful tables.

---

## A8. LCA: Degenerate Class Probabilities

### The Finding

All 172 legislators in the 91st have max_probability=1.0 (zero membership uncertainty). K=2 recovers party split with perfect certainty.

### Root Cause: Mathematical Inevitability

This is not a 91st-specific phenomenon. It is universal across **all 8 bienniums, all 16 chamber-sessions.**

The posterior class membership probability is:

```
P(class c | votes) proportional to pi_c * PROD_j P(vote_j | class c)
```

With 200-300 independent Bernoulli indicators per legislator, each with a class profile difference of 0.3-0.7, the cumulative log-likelihood ratio overwhelms any prior. Empirical measurement on the 91st House:

| Statistic | Value |
|-----------|-------|
| Bills with class profile difference > 0.3 | 213/297 (72%) |
| Minimum cumulative |log LR| (weakest legislator) | **511.0** |
| Mean cumulative |log LR| | 2963.1 |

A log-likelihood ratio of 511 translates to posterior odds of exp(511):1 — astronomically beyond floating-point precision. Even the weakest legislator reaches |log LR| > 50 within just 4 bills. By 30 bills, every legislator has certain assignment.

A bill-count experiment confirms: with only 30 bills, zero legislators have uncertain (<0.95) membership. Our vote matrices contain 124-359 bills. **No configuration of StepMix — regardless of n_init, tolerances, or initialization — would produce uncertain posteriors with this many indicators.**

### Cross-Biennium Evidence

The audit finding's claim that "89th/86th show better differentiation at K=3/4" refers to optimal K being higher, not to membership probabilities being less certain. Even at K=3 or K=4, nearly all legislators have max_probability = 1.000000. The 86th House at K=4 has only 1 legislator below 0.95 (min max_prob = 0.869).

| Biennium | House K | Senate K | House Entropy | Senate Entropy | Straddlers |
|----------|---------|----------|---------------|----------------|------------|
| 84th-91st (all) | 2-4 | 2-3 | >= 0.998 | 1.000 | 0 |

### StepMix Configuration

The code is correctly configured and follows or exceeds field best practices:

| Parameter | Tallgrass | Field Recommendation |
|-----------|-----------|---------------------|
| `n_init` | 50 | >= 50 (Nylund-Gibson & Choi 2018) |
| `measurement` | `binary_nan` | Correct (FIML for missing data) |
| `max_iter` | 1000 | >= 500 |
| `random_state` | 42 | Fixed for reproducibility |

### Literature Context

LCA is rare in legislative roll-call analysis. The dominant tradition treats ideology as continuous (NOMINATE, IRT). The Lubke & Neale (2006) impossibility result proves you cannot empirically distinguish categorical from continuous latent structure using model fit alone — K=2 LCA and 1D IRT fit equally well by construction.

The phenomenon of LCA with 200+ indicators producing perfect classification is **under-documented** because typical LCA applications use 5-20 indicators. In a legislative context with hundreds of bills, it is mathematically inevitable.

### Assessment

**Expected behavior — document and accept.** No code changes. The interesting LCA questions are not "how certain is class membership?" (always certain) but:
- Does BIC select K=2 or K>2? (91st is the only biennium with K=2 for both chambers)
- When K>2, are extra classes distinct per the Salsa test?
- What does within-party LCA find?

### Remediation

Add a note to the LCA report explaining that entropy=1.0 is mathematically inevitable with 200+ binary indicators and strong two-party structure. Reference the bill-count threshold (~30 bills) where classification becomes certain.

---

## A9. Clustering: Trivial k=2 Party Split

### The Finding

All bienniums: k-means on 1D IRT finds k=2 = exact party labels. ARI=1.0 across methods is meaningless. Within-party optimal k (k=6-7) computed but never propagated downstream.

### Root Cause

Silhouette score on a bimodal 1D distribution always selects k=2. The IRT ideal point distribution has a clear gap between the party modes. Cross-method ARI near 1.0 (hierarchical vs k-means vs spectral) confirms only that different algorithms agree on the trivial partition.

Confirmed across all 8 bienniums: every cell is k=2 except 88th Senate k-means (k=6, an anomaly from the small chamber). The report already self-annotates: the IRT clusters plot is titled "Two Groups — And They're Exactly the Two Parties" with an explanatory callout.

### Within-Party Parquets Are Dead Output

`run_within_party_clustering()` (`clustering.py:1789-1928`) computes per-party subclusters and saves `within_party_{party}_{chamber}.parquet` files. **These files are never read by any downstream phase.** Grep confirms zero matches for `within_party` across the entire `analysis/` directory.

However, within-party subclusters are **not worth propagating**:

| Characteristic | Evidence |
|----------------|----------|
| Silhouette curves | Flat (0.57-0.64 across k=2..7) |
| Optimal k stability | Bounces from 2 to 7 across bienniums |
| Interpretation | No discrete factional structure; intra-party variation is continuous |

The design doc (`analysis/design/clustering.md:146-151`) already acknowledges this: "continuous features (IRT ideal points, party loyalty rates) will be more informative than within-party cluster labels."

### What Clustering Does Contribute

The k=2 finding itself adds no information. But the clustering phase produces genuinely valuable outputs:

1. **Party loyalty metric** — consumed by prediction (Phase 8) as a continuous feature
2. **Dendrograms and voting bloc visualizations** — narrative value for reports
3. **Veto override subgroup analysis** — reveals (or confirms absence of) bipartisan coalitions
4. **HDBSCAN noise detection** — identifies outlier legislators

### Literature Context

Spirling & Quinn (2010) used Dirichlet process mixtures for intra-party voting blocs in the UK House of Commons. Kolln & Polk (2024) cluster within parties from the start, never attempting whole-legislature clustering. The field consensus: when a single dominant dimension exists, clustering on that dimension trivially recovers party labels.

### Assessment

**The finding IS the result.** Party is the only discrete structure in Kansas legislative voting. The code already handles this well (self-aware annotations, within-party analysis, continuous features for downstream).

### Remediation

1. **Add ARI-against-party diagnostic.** When k=2 clusters match party labels with ARI > 0.95, formally report this as confirmation of party dominance rather than a validation metric. One line per method in the manifest.
2. **Document within-party parquets as intentionally dead output.** The flat silhouette curves and unstable k across bienniums mean the labels are not reliable enough to propagate.

---

## A10. Network: Betweenness Sparsity (66-73% Zeros)

### The Finding

Most legislators have betweenness=0.0 across all bienniums. Synthesis bridge-builder detection relies on betweenness, but the metric is too sparse.

### Root Cause: Disconnected Party Cliques

The co-voting graph (`network.py:227-265`) creates edges only where Cohen's Kappa > 0.40 (the default threshold). In a heavily polarized legislature, inter-party Kappa values rarely exceed 0.40. The graph splits into **disconnected party components**.

Betweenness counts shortest paths through a node. In a disconnected graph, no shortest path crosses the party boundary. Within each party clique, the dense subgraph means most nodes are interchangeable on shortest paths, so betweenness concentrates on very few nodes.

### Empirical Evidence

Sparsity worsens with increasing polarization:

| Period | Bienniums | Typical %Zero (House) | Cross-Party Edges | Assortativity |
|--------|-----------|----------------------|-------------------|---------------|
| 2011-2018 | 84th-87th | 28-39% | 0-18 | 0.67-0.91 |
| 2019-2026 | 88th-91st | 56-73% | 0-1 | 0.94-1.00 |

By contrast, eigenvector and PageRank centrality have **0% zeros** across all bienniums.

### Impact on Synthesis Bridge-Builder Detection

The `detect_bridge_builder()` function (`synthesis_detect.py:147-217`) selects the legislator with the highest betweenness whose IRT ideal point is within 1 SD of the cross-party midpoint.

**91st House result:** Jesse Borjon (R-52) — betweenness=0.0104, **zero cross-party edges**. He is labeled a "bridge-builder" who "bridges the gap between the two parties," but he has no connections to Democrats. **This is a false narrative.**

**87th House result (connected graph):** John Wilson (D-10) — betweenness=0.3014, 18 cross-party edges. This is a genuine bridge legislator.

The Phase 6 report already partially acknowledges this (conditional qualifier text when components >= 2), but the information does not propagate to Phase 11 (Synthesis).

### Alternative Centrality Measures

Empirical comparison on the 91st House (disconnected graph):

| Measure | %Zero | Handles Disconnection? |
|---------|-------|----------------------|
| Betweenness | 73.1% | No — zero across components |
| Eigenvector | 0.0% | Per-component only |
| PageRank | 0.0% | Always non-zero (teleport) |
| Harmonic | 0.0% | Yes — 1/inf = 0 natively |

Harmonic centrality handles disconnected graphs natively (Boldi & Vigna 2014). On connected graphs (87th House), it correlates with betweenness at rho=0.58, capturing similar but not identical structure.

### Assessment

This is a **genuine code fix** — the only one among the 8 findings. The synthesis bridge-builder narrative is actively misleading for 4 of 8 bienniums.

### Remediation

1. **Phase 6 (network.py):** Export harmonic centrality and cross-party edge fraction alongside existing metrics. Computation cost: negligible.

2. **Phase 11 (synthesis_data.py):** Join harmonic centrality and cross-party fraction into the unified legislator DataFrame.

3. **Phase 11 (synthesis_detect.py):** Accept a connectivity parameter. When the graph is disconnected (components >= 2), rank by harmonic centrality instead of betweenness. When connected, keep current behavior (betweenness works correctly).

4. **Phase 6 and 11 reports:** When the graph is disconnected, use "within-party connector" instead of "bridge-builder." Reserve "bridge-builder" for sessions with genuine cross-party edges.

---

## A11. IRT: House Threshold Sensitivity (r=0.65-0.80)

### The Finding

Several bienniums show sensitive House IRT when the minority vote threshold changes from 2.5% to 10%. Not a bug — near-unanimous votes carry signal — but threshold dependence is not documented in reports.

### Root Cause

The sensitivity analysis (`irt.py:2762-2871`) performs a full independent MCMC re-run at the 10% threshold with independent anchor selection. The 10% threshold removes 26-47% more votes than the 2.5% default.

| Biennium | House r | Default Votes | Sensitivity Votes | % Removed |
|----------|---------|---------------|-------------------|-----------|
| 84th | 0.737 | 260 | 175 | 33% |
| 85th | 0.668 | 228 | 127 | 44% |
| 86th | 0.770 | 236 | 183 | 22% |
| 87th | 0.750 | 263 | 193 | 27% |
| 88th | **0.647** | 141 | 104 | 26% |
| 89th | 0.691 | 278 | 184 | 34% |
| 90th | 0.975 | 322 | 221 | 31% |
| 91st | 0.802 | 297 | 229 | 23% |

6/8 House bienniums are SENSITIVE (r <= 0.95). Only 1/8 Senate bienniums is sensitive (86th, r=0.631).

### Technical Finding: Sign Flips

The `raw_pearson_r` is **negative** in every sensitive House case. The code takes |r| (line 2841-2844), which is correct, but the sign flip indicates that independent anchor selection on different vote subsets produces inverted scales. The sign flip itself is informative about anchor stability.

### Why the House Is More Sensitive

Kansas is a Republican supermajority (~72% of seats). In the 125-member House, votes where 120 Republicans vote Yea and 5 moderate Republicans break ranks have a minority of 4% — included at 2.5%, excluded at 10%. These votes carry genuine information about **intra-Republican variation**, which is the most analytically interesting dimension.

The 40-member Senate has fewer marginal votes. With 40 senators, 2.5% = 1 dissenter, 10% = 4 dissenters. The Senate typically has either very contested or truly unanimous votes, with less gray zone.

### Literature Context

The 2.5% threshold is the field standard:
- W-NOMINATE: `lop=0.025` (Poole & Rosenthal 1985, 1997)
- Clinton, Jackman, Rivers (2004): "lopsided roll calls — those where the losing side has less than 2.5% — are eliminated"
- VoteView: same convention since the 1980s
- Tallgrass Phase 17 (W-NOMINATE): uses `lop=0.025`, confirming alignment

The convention was developed for the U.S. Congress (435 House, 100 Senate), where near-unanimous votes carry no discriminating information. **State legislatures with supermajorities are different** — votes with 5-10% dissent are often the most ideologically informative.

The literature does not typically report sensitivity to this threshold. Tallgrass's 10% stress test exceeds standard practice.

### Downstream Propagation

IRT ideal points from the default (2.5%) run are consumed by at least 8 downstream phases. None perform their own threshold sensitivity analysis. However, even at r=0.647, the rank ordering is substantially preserved — legislators at the extremes stay at the extremes. Sensitivity primarily affects legislators in the middle.

### Assessment

**Report/documentation fix.** The 2.5% default is correct (field standard). The sensitivity analysis is well-designed. The gap is that the IRT report presents the results without interpretation — it is the only major section without an `_add_*_interpretation()` function.

### Remediation

1. **Add `_add_sensitivity_interpretation()` to `irt_report.py`.** Dynamically classify each chamber as ROBUST (r > 0.95) or SENSITIVE (r <= 0.95). Explain why the House is typically more sensitive (supermajority structure, intra-party signal in the 2.5-10% band). Note the field-standard convention and that sensitivity does not mean the default results are wrong.

2. **Add ROBUST/SENSITIVE classification and raw r to the sensitivity table.** Currently the table shows only the absolute correlation. Add a "Status" column and show sign flip information.

---

## A16. Hierarchical: Small Senate Democrat Groups

### The Finding

Senate Democrats range 8-11 across all bienniums. Hierarchical shrinkage unreliable. Already flagged with warnings.

### Current State

Constants at `hierarchical.py:228-238`:
- `MIN_GROUP_SIZE_WARN = 15` — triggers console warning
- `SMALL_GROUP_THRESHOLD = 20` — triggers adaptive prior (`HalfNormal(0.5)` instead of `HalfNormal(1.0)`)

The warning prints to the run log but is **not surfaced in the HTML report**. Every other major analysis section has interpretation text, but the convergence/group-size section does not flag the small-group issue.

All 8 bienniums trigger the warning. The 85th is catastrophic (R-hat 1.918). Most others have marginal R-hats (1.01-1.21) or ESS warnings.

### Literature

- James & Stein (1960): Shrinkage dominance only for J >= 3 groups. With J=2, may help or hurt.
- Gelman & Hill (2007): Recommend J >= 5 for reliable between-group variance estimation.
- Gelman (2015): For J=2, "don't try to estimate the between-group variance from the data."

The adaptive prior already follows Gelman's 2015 advice. The limitation is structural: 7-11 Senate Democrats cannot constrain `sigma_within` for their group.

### Assessment

**Known limitation — surface warning in HTML report.** No parameter changes needed; the adaptive prior is the right mitigation per the literature.

### Remediation

When any group has fewer than `SMALL_GROUP_THRESHOLD`, add a visible caveat in the Key Findings section: hierarchical shrinkage is unreliable for that chamber, and flat IRT results should be preferred for individual ideal points.

---

## A17. Bipartite: Sparse BiCM Backbone (Senate)

### The Finding

95% edge reduction, 73.8% senators isolated. Too sparse for centrality analysis.

### Root Cause: Structural Property of Dense Partisan Chambers

The BiCM null model preserves each legislator's total Yea count and each bill's total Yea votes. In the 42-member Senate with ~72% density, the null model already predicts most observed co-voting. For two senators who both vote Yea on 140/194 bills, expected overlap is ~100+ bills. Observed overlap must be substantially higher to be "surprising" — and with only ~54 remaining bills, statistical power is limited.

Senate backbone edges across bienniums:

| Biennium | Backbone Edges | % of Max Possible | Isolated Senators |
|----------|---------------|-------------------|-------------------|
| 84th | 0 | 0% | 37 (100%) |
| 85th | 18 | 2.4% | varies |
| 88th | 0 | 0% | 41 (100%) |
| 91st | 25 | 2.9% | 31 (73.8%) |

Two bienniums have zero backbone edges. The House is also sparse but less extreme.

### Technical Note: Validation Method

The code (`bipartite.py:793-841`) retrieves raw p-values from `bg.get_projected_pvals_mat()` and applies a global threshold (`p < 0.01`), rather than using FDR correction (the BiCM library default). This is actually more lenient than FDR — simulation shows that even FDR at alpha=0.20 produces zero Senate edges for Kansas-like data.

### Literature

- Saracco et al. (2017): Used FDR correction in the original BiCM work
- Neal (2022, backbone R package): Applies Holm-Bonferroni by default
- Bruno et al. (2022): Proposed "meta-validation" — sweep alpha to find signal-to-noise peak
- For dense, partisan networks, the BiCM is inherently conservative because the degree-preserving null already explains most structure

### Assessment

**Structural limitation with moderate code improvement available.**

### Remediation

1. **Chamber-adaptive threshold.** Senate uses `alpha=0.05`, House uses `alpha=0.01`. Defensible because the multiple-testing burden differs by 10x (861 vs 8385 possible edges).
2. **Density guard in report.** When >50% of nodes are isolated, explicitly state that the backbone is too sparse for legislator-level centrality analysis.
3. **Use FDR correction explicitly** for methodological correctness (even though it may produce fewer edges).

---

## A18. Bipartite: Bill Communities Mirror Party Split

### The Finding

Phase 12 consistently finds 2 bill communities = party voting pattern. Analytically redundant with clustering/IRT.

### Root Cause: Genuinely One-Dimensional Structure

The Leiden resolution sweep proves there is no intermediate structure:

| Resolution | Communities | Modularity |
|------------|-----------|-----------|
| 0.5 | 1 | 0.000 |
| 1.0 | 2 | 0.017-0.022 |
| 1.5 | 248+ | ~0.001 |

The jump from 2 to 248 communities means there is no sub-party structure in the bill projection. The 2-community modularity (0.017-0.022) is an order of magnitude below the significance threshold of 0.30 (Newman & Girvan 2004). This pattern is universal across all 8 bienniums.

Community profiles confirm the party mirror: one community of Republican-supported bills (R%=0.96, D%=0.13), one of bipartisan/Democrat-supported bills (R%=0.68, D%=0.88).

### Is Phase 12 Redundant?

**Only the community detection component is redundant.** Phase 12 produces genuinely novel outputs not available from any other phase:
- Bill polarization scores (per-bill, not per-legislator)
- Bridge bills (bipartite betweenness)
- BiCM backbone (statistically validated, even if sparse)

### Assessment

**Genuinely structural — document and accept for community detection. Phase 12 as a whole is not redundant.**

### Remediation

1. **Modularity quality gate.** When best modularity < 0.10, the report should note that bill community structure is weak and mirrors the known party divide.
2. **Redundancy note.** When bill communities equal the number of parties and profiles align with party labels, state this confirms the party-dominant structure found by IRT/clustering.

---

## Classification Summary

### Code Fixes (1)

| # | Finding | Scope | Risk |
|---|---------|-------|------|
| A10 | Synthesis bridge-builder uses betweenness on disconnected graph | 3 files, ~60 lines | Low (additive) |

### Report/Documentation Improvements (4)

| # | Finding | What to Add |
|---|---------|------------|
| A6 | Prediction FP asymmetry | Interpretation section + per-class split tables |
| A11 | IRT threshold sensitivity | Interpretation section + ROBUST/SENSITIVE classification |
| A16 | Small Senate D groups | Surface existing warning in HTML report Key Findings |
| A17 | Sparse BiCM backbone | Chamber-adaptive threshold + density guard in report |

### Document and Accept (3)

| # | Finding | Why Accepted |
|---|---------|-------------|
| A8 | LCA degenerate posteriors | Mathematical inevitability with 200+ indicators |
| A9 | Clustering trivial k=2 | The finding IS the result; party is the only discrete structure |
| A18 | Bill communities = party | Genuinely one-dimensional; no parameter fixes possible |

---

## References

- Boldi, P. & Vigna, S. (2014). Axioms for Centrality. *Internet Mathematics*, 10(3-4), 222-262.
- Bruno, M. et al. (2022). Meta-validation of bipartite network projections. *Communications Physics*, 5, 76.
- Clinton, J., Jackman, S. & Rivers, D. (2004). The Statistical Analysis of Roll Call Data. *American Political Science Review*, 98(2), 355-370.
- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models. *Bayesian Analysis*, 1(3), 515-534.
- Gelman, A. & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- James, W. & Stein, C. (1960). Estimation with Quadratic Loss. *Proceedings of the Fourth Berkeley Symposium*, 1, 361-379.
- Kolln, A. & Polk, J. (2024). Structuring Intra-Party Politics. *Comparative Political Studies*, 57(5).
- Lubke, G. & Neale, M. (2006). Distinguishing Between Latent Classes and Continuous Factors. *Multivariate Behavioral Research*, 41(4), 499-532.
- Neal, Z. (2022). backbone: An R Package to Extract Network Backbones. *PLOS ONE*, 17(5).
- Newman, M. & Girvan, M. (2004). Finding and evaluating community structure in networks. *Physical Review E*, 69(2).
- Nylund-Gibson, K. & Choi, A. (2018). Ten frequently asked questions about latent class analysis. *Translational Issues in Psychological Science*, 4(4), 440-461.
- Poole, K. & Rosenthal, H. (1985). A spatial model for legislative roll call analysis. *American Journal of Political Science*, 29(2), 357-384.
- Saracco, F. et al. (2017). Randomizing bipartite networks: the case of the World Trade Web. *New Journal of Physics*, 19(5).
- Spirling, A. & Quinn, K. (2010). Identifying Intraparty Voting Blocs in the U.K. House of Commons. *Journal of the American Statistical Association*, 105(490), 447-457.
- van den Goorbergh, R. et al. (2022). The harm of class imbalance corrections for risk prediction models. *Journal of Clinical Epidemiology*, 150, 223-232.
