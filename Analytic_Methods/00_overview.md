# Analytical Methods for Kansas Legislature Voting Data

## Purpose

This directory contains detailed documentation for every analytical method applicable to our Kansas Legislature roll call voting dataset. Each method has its own document with full mathematical foundations, Python implementation guidance, interpretation notes, and Kansas-specific considerations.

## Our Data at a Glance

| Dataset | Rows | Description |
|---------|------|-------------|
| `votes.csv` | ~75K | One row per legislator per roll call (legislator, bill, vote category) |
| `rollcalls.csv` | ~865 | One row per roll call (bill, motion, counts, result) |
| `legislators.csv` | ~172 | One row per legislator (name, party, chamber, district) |

**Key characteristics:** Two chambers (130 House, 42 Senate), two parties (Republican supermajority ~72%, Democrat ~28%), five vote categories (Yea, Nay, Present and Passing, Absent and Not Voting, Not Voting), single legislative session spanning ~11 months.

## Document Naming Scheme

```
NN_CAT_method_name.md
```

- `NN` — Two-digit ordering number (analysis pipeline order)
- `CAT` — Category code (see below)
- `method_name` — Descriptive snake_case name

### Category Codes

| Code | Category | Description |
|------|----------|-------------|
| `DATA` | Data Preparation | Vote matrix construction and preprocessing |
| `EDA` | Exploratory Data Analysis | Descriptive statistics and visualization |
| `IDX` | Index-Based Measures | Canonical political science indices |
| `DIM` | Dimensionality Reduction | Ideal point estimation and spatial models |
| `BAY` | Bayesian Methods | Probabilistic models with uncertainty quantification |
| `CLU` | Clustering & Classification | Grouping legislators by voting similarity |
| `NET` | Network Analysis | Graph-based analysis of co-voting patterns |
| `PRD` | Predictive Modeling | Machine learning vote/bill prediction |
| `TSA` | Time Series Analysis | Temporal patterns and change detection |

## Recommended Analysis Pipeline

The methods are ordered from simple to complex. Each phase builds on the previous.

### Phase 1: Foundation (Do First)

| # | Document | Method | Time to Implement |
|---|----------|--------|-------------------|
| 01 | `01_DATA_vote_matrix_construction` | Build the legislator x roll-call vote matrix | 1 hour |
| 02 | `02_EDA_descriptive_statistics` | Basic counts, distributions, passage rates | 1 hour |
| 03 | `03_EDA_vote_participation` | Absenteeism patterns and participation rates | 30 min |
| 04 | `04_EDA_agreement_matrix_heatmap` | Pairwise agreement + hierarchically clustered heatmap | 1 hour |

### Phase 2: Classical Measures (Political Science Standard)

| # | Document | Method | Time to Implement |
|---|----------|--------|-------------------|
| 05 | `05_IDX_rice_index` | Party cohesion per roll call | 30 min |
| 06 | `06_IDX_party_unity_scores` | Per-legislator party loyalty on contested votes | 30 min |
| 07 | `07_IDX_effective_number_of_parties` | Legislative fragmentation (Laakso-Taagepera) | 15 min |
| 08 | `08_IDX_loyalty_and_maverick_scores` | Identify party rebels and loyalists | 30 min |

### Phase 3: Spatial Models (Where Legislators Sit Ideologically)

| # | Document | Method | Time to Implement |
|---|----------|--------|-------------------|
| 09 | `09_DIM_principal_component_analysis` | PCA on vote matrix (Python-native ideal points) | 1 hour |
| 10 | `10_DIM_correspondence_analysis` | MCA for categorical vote data | 1 hour |
| 11 | `11_DIM_umap_tsne_visualization` | Non-linear dimensionality reduction | 30 min |
| 12 | `12_DIM_w_nominate` | Gold-standard spatial model (R via rpy2) | 2 hours |
| 13 | `13_DIM_optimal_classification` | Non-parametric ideal point estimation (R) | 1 hour |

### Phase 4: Bayesian Analysis (The Crown Jewels)

| # | Document | Method | Time to Implement |
|---|----------|--------|-------------------|
| 14 | `14_BAY_beta_binomial_party_loyalty` | Bayesian party loyalty with shrinkage | 2 hours |
| 15 | `15_BAY_irt_ideal_points` | Full Bayesian ideal point estimation (PyMC) | 3 hours |
| 16 | `16_BAY_hierarchical_legislator_model` | Legislator nested in party nested in chamber | 3 hours |
| 17 | `17_BAY_posterior_predictive_checks` | Model validation and comparison | 2 hours |

### Phase 5: Structure Discovery

| # | Document | Method | Time to Implement |
|---|----------|--------|-------------------|
| 18 | `18_CLU_hierarchical_clustering` | Dendrogram of legislator voting similarity | 1 hour |
| 19 | `19_CLU_kmeans_voting_patterns` | Partition legislators into k voting blocs | 30 min |
| 28 | `28_CLU_latent_class_mixture_models` | Discrete faction discovery (probabilistic) | 2 hours |
| 20 | `20_NET_covoting_network` | Weighted co-voting graph analysis | 2 hours |
| 21 | `21_NET_bipartite_bill_legislator` | Two-mode network of legislators and bills | 1 hour |
| 22 | `22_NET_centrality_measures` | Identify influential and bridging legislators | 1 hour |
| 23 | `23_NET_community_detection` | Data-driven faction discovery (Louvain/Leiden) | 1 hour |

### Phase 6: Prediction and Dynamics

| # | Document | Method | Time to Implement |
|---|----------|--------|-------------------|
| 24 | `24_PRD_vote_prediction` | Predict individual votes from features | 2 hours |
| 25 | `25_PRD_bill_passage_prediction` | Predict whether bills pass | 2 hours |
| 26 | `26_TSA_ideological_drift` | Track position changes within session | 2 hours |
| 27 | `27_TSA_changepoint_detection` | Detect structural breaks in voting patterns | 1 hour |

## Dependencies Between Methods

```
01_DATA_vote_matrix  ──> ALL other methods
                    ┌──> 09_PCA ──> 15_BAY_IRT (comparison)
02_EDA_descriptive ─┤
                    └──> 05_Rice ──> 06_Unity ──> 08_Maverick
04_EDA_agreement ──────> 18_CLU_hierarchical
                   ┌──> 20_NET_covoting ──> 22_NET_community
04_EDA_agreement ──┤
                   └──> 21_NET_bipartite
09_PCA ────────────────> 11_UMAP (can use PCA-reduced input)
09_PCA ────────────────> 28_CLU_latent_class (compare continuous vs discrete)
14_BAY_beta_binomial ──> 16_BAY_hierarchical
15_BAY_IRT ────────────> 17_BAY_posterior_predictive
```

## Python Environment

Core libraries needed across all analyses:

```bash
uv add pandas numpy scipy matplotlib seaborn scikit-learn
uv add networkx python-louvain prince umap-learn
uv add pymc arviz
uv add xgboost shap ruptures
```

Optional (for NOMINATE/OC via R):
```bash
# Requires R installation with wnominate, pscl, oc packages
uv add rpy2
```

## Kansas-Specific Analytical Considerations

1. **Supermajority dynamics**: Kansas has a Republican supermajority (~72% of seats). The interesting variation is often *within* the majority party, not between parties. Methods that only capture a single partisan dimension will miss intra-Republican factions (moderate vs. conservative).

2. **Bicameral structure**: The Senate (42 members) and House (130 members) vote on different roll calls, so they generally must be analyzed separately. Cross-chamber analysis is possible only for bills that received floor votes in both chambers.

3. **High passage rates**: Most votes are near-unanimous. Filtering to contested votes (where the minority side is >2.5% or >10%) is essential for methods like PCA and IRT that rely on variation to estimate positions.

4. **Veto overrides**: Kansas had 34 veto override votes in the 2025-26 session. These votes often cross party lines and are analytically rich — a natural case study for coalition analysis.

5. **Small Senate**: With only 42 senators, some methods (especially Bayesian hierarchical models) benefit from partial pooling / shrinkage to stabilize estimates.

## Key References

| Paper | Year | Method |
|-------|------|--------|
| Rice, "Behavior of Legislative Groups" | 1925 | Rice Index |
| Laakso & Taagepera, "Effective Number of Parties" | 1979 | ENP |
| Poole & Rosenthal, "A Spatial Model for Roll Call Analysis" | 1985 | NOMINATE |
| Poole, "Non-Parametric Unfolding of Binary Choice Data" | 2000 | Optimal Classification |
| Jackman, "Multidimensional Analysis via Bayesian Simulation" | 2001 | Bayesian IRT |
| Martin & Quinn, "Dynamic Ideal Point Estimation" | 2002 | Dynamic Ideal Points |
| Clinton, Jackman, Rivers, "Statistical Analysis of Roll Call Data" | 2004 | Bayesian IRT (canonical) |
| Imai, Lo, Olmsted, "Fast Estimation of Ideal Points" | 2016 | EM-based IRT |
| Rosenthal & Voeten, "Estimating Ideal Points via PCA" | 2018 | PCA |
| Goet, "Explaining Differences Across Voting Domains" | 2023 | Hierarchical Bayesian |
