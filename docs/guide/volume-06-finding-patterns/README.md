# Volume 6 — Finding Patterns

> *Are there voting blocs? Who are the bridge-builders? How unified is each party?*

---

Volumes 3-5 answered the question: *where does each legislator stand?* This volume asks: *who stands together?*

Ideology scores tell you where each person is on the spectrum. But legislatures are social institutions — they run on coalitions, factions, alliances, and defections. A moderate Republican with an IRT score of +0.4 is interesting. A cluster of six moderate Republicans who consistently break ranks together on tax bills is a *story*.

This volume covers six related techniques for discovering legislative structure. **Clustering** and **latent class analysis** ask whether legislators fall into discrete groups. **Network analysis** maps the web of co-voting relationships and identifies bridge-builders. **Bipartite networks** flip the lens and ask which *bills* bring legislators together across party lines. **Classical indices** provide the simple, interpretable measures that political scientists have relied on for a century. And **empirical Bayes** applies the shrinkage logic from hierarchical models (Volume 4, Chapter 5) to party loyalty scores, giving honest uncertainty bounds on how reliable each legislator's loyalty rating really is.

---

## Chapters

| # | Title | What You'll Learn |
|---|-------|-------------------|
| 1 | [Clustering: Are There Discrete Factions?](ch01-clustering.md) | Five algorithms, one surprising answer: the legislature has two clusters, not three or four |
| 2 | [Latent Class Analysis: Testing for Hidden Groups](ch02-latent-class-analysis.md) | A probabilistic approach to group detection — and why "hidden groups" might just be points on a spectrum |
| 3 | [Co-Voting Networks: Who Votes Together?](ch03-co-voting-networks.md) | Building a web of relationships from Kappa agreement, measuring who's central and who's a bridge |
| 4 | [Bipartite Networks: Bills That Bridge the Aisle](ch04-bipartite-networks.md) | Flipping the lens from legislators to bills — which legislation creates unlikely alliances? |
| 5 | [Classical Indices: Rice, Party Unity, and the Maverick Score](ch05-classical-indices.md) | The simple metrics that political scientists have used since 1925 — and why they still matter |
| 6 | [Empirical Bayes: Shrinkage Estimates of Party Loyalty](ch06-empirical-bayes.md) | How Bayesian shrinkage gives honest uncertainty bounds on loyalty scores — even for legislators with few votes |

---

## Key Terms Introduced in This Volume

| Term | Definition |
|------|-----------|
| Hierarchical clustering | A method that builds a tree (dendrogram) by merging the most similar pairs of legislators step by step |
| K-means | An algorithm that partitions legislators into exactly *k* groups by minimizing within-group distances |
| Gaussian Mixture Model (GMM) | A soft clustering method that assigns each legislator a *probability* of belonging to each group |
| HDBSCAN | A density-based algorithm that finds clusters of varying shapes and sizes, and can label outliers as noise |
| Silhouette score | A measure of how well each legislator fits its assigned cluster — ranges from −1 (misplaced) to +1 (well placed) |
| Cophenetic correlation | How faithfully a dendrogram preserves the original pairwise distances |
| Adjusted Rand Index (ARI) | A measure of agreement between two different clusterings, corrected for chance |
| Latent Class Analysis (LCA) | A model-based method that assumes a hidden group variable generates the observed voting patterns |
| BIC (Bayesian Information Criterion) | A model-selection criterion that balances fit against complexity — lower is better |
| Salsa effect | When extra classes are just milder or hotter versions of the same pattern, not genuinely different groups |
| Network | A mathematical structure of nodes (legislators) connected by edges (co-voting relationships) |
| Centrality | A family of measures for how important a node is within a network |
| Betweenness centrality | How often a legislator lies on the shortest path between other legislators — a measure of brokerage |
| Leiden algorithm | A community-detection method that finds groups of densely connected nodes within a network |
| Bipartite network | A network with two types of nodes (legislators and bills) where edges only connect across types |
| BiCM (Bipartite Configuration Model) | A null model that tests whether co-voting patterns are explained by individual activity levels alone |
| Newman weighting | A correction that down-weights connections through high-degree nodes in bipartite projections |
| Disparity filter | A method that extracts the statistically significant edges from a weighted network |
| Rice Index | A per-vote measure of party cohesion: the absolute difference between percent Yea and percent Nay within a party |
| Party unity score | How often a legislator votes with their party majority on contested party-line votes |
| Effective Number of Parties (ENP) | A single number that captures how many parties effectively compete — the inverse of the Herfindahl index applied to seat shares |
| Maverick rate | How often a legislator defects from their party on party-line votes |
| Empirical Bayes | Estimating a prior distribution from the data itself, rather than specifying it subjectively |
| Beta-Binomial model | A Bayesian model where the prior on loyalty rates is Beta-distributed and the likelihood is Binomial |
| Shrinkage | The tendency for estimates based on little data to be pulled toward the group average |

---

*Previous: [Volume 5 — Checking Our Work](../volume-05-checking-our-work/)*

*Next: [Volume 7 — Prediction and Text](../volume-07-prediction-and-text/)*
