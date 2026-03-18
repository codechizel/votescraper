# Appendix C: Phases at a Glance

> *The 28 analysis phases in pipeline execution order, with inputs, outputs, and the guide chapter that explains each one.*

---

## How to Use This Reference

The pipeline runs phases in a specific order (matching the dashboard sidebar). Each phase reads upstream output and produces its own. This table lists every phase with its key inputs, outputs, and the guide chapter where it's explained.

**Run command:** `just pipeline 2025-26` runs phases 01-25 (plus 07b). `just cross-pipeline` runs phases 26-27.

---

## Single-Biennium Phases (01-25)

| # | Phase | Guide | Method | Key Inputs | Key Outputs |
|---|-------|-------|--------|------------|-------------|
| 01 | **EDA** | Vol. 3, Ch. 1 | Descriptive statistics, vote matrix construction | Raw CSVs (votes, rollcalls, legislators) | Vote matrix, filtering manifest, summary stats |
| 02 | **PCA** | Vol. 3, Ch. 3 | Principal Component Analysis | Vote matrix from 01 | PC scores, scree plot, explained variance, loadings |
| 03 | **MCA** | Vol. 3, Ch. 4 | Multiple Correspondence Analysis | Vote matrix from 01 | MCA coordinates, inertia, biplot |
| 20 | **Bill Text NLP** | Vol. 7, Ch. 3 | BERTopic, embeddings, topic modeling | Bill text CSVs | Topic assignments, embeddings, companion bill detection |
| 05 | **IRT (1D)** | Vol. 4, Ch. 2-3 | Bayesian 1D Item Response Theory (nutpie NUTS) | Vote matrix from 01, PCA from 02 | Ideal points (ξ), bill params (α, β), convergence diagnostics |
| 06 | **2D IRT** | Vol. 4, Ch. 4 | Bayesian 2D IRT (M2PL with PLT identification) | Vote matrix from 01, PCA from 02 | 2D ideal points, canonical routing manifest |
| 07 | **Hierarchical IRT** | Vol. 4, Ch. 5 | Hierarchical Bayesian IRT (party-pooled) | Vote matrix from 01, canonical IRT from 06 | Party means, within-party SD, shrinkage, ICC |
| 07b | **Hierarchical 2D** | Vol. 4, Ch. 5 | Hierarchical 2D IRT (M2PL + party structure) | Vote matrix from 01, PCA from 02 | 2D hierarchical ideal points, party-pooled shrinkage |
| 08 | **PPC** | Vol. 5, Ch. 2 | Posterior predictive checks | IRT posteriors from 05/06/07 | GMP, APRE, item/person fit, LOO-CV, model comparison |
| 04 | **UMAP** | Vol. 3, Ch. 4 | UMAP dimensionality reduction | Vote matrix from 01 | 2D embedding, Procrustes validation |
| 09 | **Clustering** | Vol. 6, Ch. 1 | K-means, hierarchical, GMM, spectral, HDBSCAN | IRT from 05/06, PCA from 02 | Cluster assignments (k=2-6), ARI, silhouette, loyalty rates |
| 10 | **LCA** | Vol. 6, Ch. 2 | Latent Class Analysis | Vote matrix from 01 | Class assignments, item response probabilities, BIC, entropy |
| 11 | **Network** | Vol. 6, Ch. 3 | Co-voting network (Kappa edges, Leiden communities) | Agreement matrix from 01 | Centrality scores, community assignments, network visualization |
| 12 | **Bipartite** | Vol. 6, Ch. 4 | Bipartite legislator-bill network | Vote matrix from 01 | Bill polarization, bridge bills, bipartite betweenness |
| 13 | **Indices** | Vol. 6, Ch. 5 | Rice, party unity, maverick rate, ENP | Vote matrix from 01, party data | Unity scores, maverick rates, Rice time series, veto overrides |
| 14 | **Beta-Binomial** | Vol. 6, Ch. 6 | Empirical Bayes (Beta-Binomial conjugate) | Party unity data from 13 | Posterior loyalty, shrinkage factors, credible intervals |
| 15 | **Prediction** | Vol. 7, Ch. 1-2 | XGBoost vote + bill passage prediction | IRT from 05/06, vote matrix, bill text embeddings | AUC, SHAP values, per-legislator accuracy, surprising votes |
| 16 | **W-NOMINATE** | Vol. 5, Ch. 4 | W-NOMINATE + Optimal Classification (R subprocess) | Vote matrix from 01 | W-NOMINATE ideal points, OC classification, cross-method correlation |
| 17 | **External Validation** | Vol. 5, Ch. 3 | Shor-McCarty correlation | IRT from 05/06, external Shor-McCarty data | Pearson/Spearman correlation, within-party correlation |
| 18 | **DIME** | Vol. 5, Ch. 3 | DIME/CFscore comparison | IRT from 05/06, external DIME data | Correlation with campaign finance ideology |
| 19 | **TSA** | Vol. 8, Ch. 1 | Rolling PCA, Rice time series, PELT changepoints | Vote matrix from 01, PCA from 02 | Drift plots, changepoints, Bai-Perron CIs |
| 21 | **TBIP** | Vol. 7, Ch. 4 | Text-Based Ideal Points | Bill text embeddings from 20, vote matrix | Text ideal points, discrepancy scores |
| 22 | **Issue IRT** | Vol. 7, Ch. 4 | Topic-specific IRT | Topic assignments from 20, vote matrix | Per-topic ideal points, cross-topic correlation |
| 23 | **Model Legislation** | Vol. 7, Ch. 5 | ALEC similarity detection | Bill text embeddings from 20, ALEC corpus | Similarity scores, match classifications, detail cards |
| 24 | **Synthesis** | Vol. 9, Ch. 2 | Upstream aggregation, notable detection | 10 upstream phases | Unified legislator DF, mavericks, bridges, paradoxes, narrative report |
| 25 | **Profiles** | Vol. 9, Ch. 3 | Per-legislator deep dive | Synthesis from 24, raw votes, IRT bill params | Scorecards, defections, neighbors, surprising votes |

## Cross-Biennium Phases (26-27)

| # | Phase | Guide | Method | Key Inputs | Key Outputs |
|---|-------|-------|--------|------------|-------------|
| 26 | **Cross-Session** | Vol. 8, Ch. 3-4 | Affine alignment, conversion-replacement decomposition | IRT from adjacent sessions | Aligned ideal points, top movers, KS tests, Sankey diagrams |
| 27 | **Dynamic IRT** | Vol. 8, Ch. 2 | Dynamic state-space IRT (random walk, nutpie) | Vote matrices from all 8 bienniums | 15-year trajectories, tau estimates, polarization trend |

---

## Phase Dependencies

```
Raw CSVs
  └─→ 01 EDA (vote matrix)
        ├─→ 02 PCA
        │     ├─→ 05 IRT (1D)
        │     │     ├─→ 06 IRT (2D) → canonical routing
        │     │     │     └─→ 07 Hierarchical → 07b Hierarchical 2D
        │     │     └─→ 08 PPC (model comparison)
        │     ├─→ 09 Clustering
        │     └─→ 19 TSA (rolling PCA, changepoints)
        ├─→ 03 MCA
        ├─→ 04 UMAP
        ├─→ 10 LCA
        ├─→ 11 Network
        ├─→ 12 Bipartite
        ├─→ 13 Indices → 14 Beta-Binomial
        └─→ 20 Bill Text NLP
              ├─→ 21 TBIP
              ├─→ 22 Issue IRT
              └─→ 23 Model Legislation

  05/06 + 09 + 11 + 13 + 15 + ... → 24 Synthesis → 25 Profiles

  External: 16 W-NOMINATE, 17 Shor-McCarty, 18 DIME (validation, parallel)

  Cross-biennium: 26 Cross-Session, 27 Dynamic IRT (after all single-biennium runs)
```

---

## Runtime Reference

| Phase Group | Typical Runtime (M3 Pro) | Notes |
|-------------|-------------------------|-------|
| EDA + PCA + MCA | ~30 seconds | Fast; no MCMC |
| IRT (1D + 2D + Hierarchical) | 5-30 minutes per chamber | Depends on convergence; adaptive tuning for supermajority |
| Clustering + LCA + Network | ~2 minutes | Fast; no MCMC |
| Prediction (XGBoost) | ~1 minute | CPU-only; no GPU required |
| Text NLP + TBIP + Issue IRT | 3-10 minutes | Embedding computation is the bottleneck |
| Synthesis + Profiles | ~30 seconds | Aggregation only; no new models |
| Cross-Session (Phase 26) | ~1 minute per pair | 7 pairs for 8 bienniums |
| Dynamic IRT (Phase 27) | 15-45 minutes per chamber | ~10,000 parameters; 4 chains × 2,000 draws |
| **Full pipeline (one biennium)** | **30-90 minutes** | Sequential on Apple Silicon |

---

*Back to: [Appendices](./)*
