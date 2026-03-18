# Chapter 1: Clustering: Are There Discrete Factions?

> *Political pundits talk about "moderate Republicans" and "progressive Democrats" as if they were distinct tribes. But are there really separate factions in the Kansas Legislature — or is ideology a continuous spectrum with no natural dividing lines?*

---

## The Question Behind the Method

When you look at the IRT ideal points from Volume 4, you see a distribution — legislators spread out along a line from liberal to conservative. But a distribution can have different shapes. It might have two clear peaks with a valley between them (two distinct groups). It might have three peaks (a moderate center flanked by two poles). Or it might be a smooth, continuous spread with no natural breaks at all.

Clustering algorithms try to find the breaks. They take the voting data and ask: if I had to divide these legislators into *k* groups, where would the most natural boundaries fall? And how many groups best describe the data?

The answer in Kansas turns out to be both simple and important: **two groups, corresponding almost exactly to the two parties.** The moderate-to-conservative gradient within the Republican caucus is real, but it's a continuous spectrum — there's no sharp dividing line between "moderate Republicans" and "conservative Republicans." The parties are the factions.

## Why Five Algorithms?

If you take a photograph of a sculpture from the front, you see one thing. From the side, something else. From above, something different again. If all three angles tell the same story, you can be confident it's the sculpture's shape, not a trick of the lighting.

Tallgrass runs five different clustering algorithms on the same data. Each makes different assumptions about what "a cluster" means, so if they all find the same structure, the finding is robust. If they disagree, the disagreement itself teaches you something — usually that the structure is ambiguous.

Here's how the five algorithms differ in what they assume:

| Algorithm | What counts as a cluster? | Analogy |
|-----------|--------------------------|---------|
| **Hierarchical** | Groups that merge naturally when you build a family tree | A genealogist building a tree of relatedness |
| **K-means** | Compact, spherical groups with clear centers | Drawing circles around crowds on a football field |
| **GMM** | Overlapping bell curves where each legislator has a probability of membership | A weather map with overlapping pressure zones |
| **Spectral** | Densely connected regions in a graph | Islands of agreement in a sea of disagreement |
| **HDBSCAN** | Dense regions of any shape, with sparse regions treated as noise | Finding cities on a satellite photo without knowing how many there are |

## Algorithm 1: Hierarchical Clustering (The Family Tree)

### The Idea

Hierarchical clustering is the oldest and most intuitive approach. It was developed in the 1950s by biologists who needed to classify organisms — **Robert Sokal and Charles Michener** published the average-linkage method in 1958 as part of a broader project called "numerical taxonomy." The idea was to replace subjective expert judgment with a reproducible algorithm.

The procedure is simple: start with every legislator as their own cluster. Find the two most similar legislators and merge them into one cluster. Then find the next most similar pair (which might be two individuals, or an individual and the cluster you just created) and merge again. Repeat until everyone is in one big cluster.

The result is a **dendrogram** — a tree diagram that shows the order of merges. The height at which two legislators (or two groups) merge tells you how similar they are: low merges mean high similarity, tall merges mean the groups are quite different.

### What Goes Into the Distance?

The critical input is the **distance** between legislators. Tallgrass uses Cohen's Kappa (Volume 3, Chapter 2) as the similarity measure, then converts it to a distance:

```
distance = 1 − Kappa
```

Two legislators with Kappa = 0.80 (almost perfect agreement) have a distance of 0.20 — they're close. Two with Kappa = −0.30 (active disagreement) have a distance of 1.30 — they're far apart.

Why Kappa instead of raw agreement? Because of the 82% Yea base rate. Two random legislators agree 70% of the time purely by chance. Raw agreement would make the entire legislature look like one big cluster. Kappa strips out the chance agreement, so the distances reflect genuine ideological differences.

### How to Merge: Average Linkage

When merging two clusters (each containing multiple legislators), you need a rule for measuring the distance between them. Tallgrass uses **average linkage**: the distance between two clusters is the average distance between all pairs of legislators across the two clusters.

This choice matters more than it might seem. **Ward's method** (Joe Ward Jr., 1963), which merges clusters to minimize total within-cluster variance, is the most common default in statistics. But Ward's method assumes Euclidean distance — straight-line distances in geometric space. Our Kappa-based distances aren't Euclidean (a property called "non-metricity"). Average linkage makes no assumptions about the geometry of the distance measure, making it the methodologically appropriate choice for agreement-based distances.

### The Cophenetic Test

How do you know if the tree is a good representation of the original distances? The **cophenetic correlation** (Sokal and Rohlf, 1962) measures this. The cophenetic correlation compares the original distances between legislators (how differently they voted) with the heights at which they merge in the tree. A high value (above 0.70) means the tree faithfully represents the original data — legislators who voted similarly merge early, and those who voted differently merge late. The Kansas Legislature typically scores 0.75–0.85.

**Codebase:** `analysis/09_clustering/clustering.py` (`run_hierarchical()`, `LINKAGE_METHOD = "average"`, `COPHENETIC_THRESHOLD = 0.70`)

## Algorithm 2: K-Means (Draw Lines on the Map)

### The Idea

K-means takes a completely different approach. Instead of building a tree from the bottom up, it starts with the answer — "there are *k* groups" — and iterates until the groups stabilize.

The algorithm was developed by **Stuart Lloyd** at Bell Labs in 1957 as a technique for signal processing, though his paper wasn't published until 1982. **James MacQueen** independently published the method in 1967 and gave it the name "k-means."

The procedure: randomly place *k* center points in the data space. Assign each legislator to the nearest center. Recompute each center as the average of all legislators assigned to it. Repeat the assign-and-recompute cycle until nothing changes.

### What Goes Into K-Means?

While hierarchical clustering uses the Kappa distance matrix, k-means uses **IRT ideal points** as its input. Each legislator is represented by their ideal point score, and k-means tries to find *k* groups with the tightest possible internal spread.

The key tuning choice is *k* — how many groups. Tallgrass evaluates k = 2 through k = 7 and uses two metrics to choose:

**The elbow plot** shows the total within-cluster distance (called **inertia**) as a function of *k*. As *k* increases, inertia always decreases (more groups means each group is tighter). But there's usually a "bend" where adding more groups stops helping much. The bend is the elbow — the natural number of clusters.

**The silhouette score** (Peter Rousseeuw, 1987) gives a more formal answer. For each legislator, it computes two numbers:
- **a**: the average distance to all other legislators in the same cluster (how tight is my group?)
- **b**: the average distance to all legislators in the nearest *other* cluster (how far am I from the next group?)

The silhouette is:

```
silhouette = (b − a) / max(a, b)
```

**Plain English:** "How much closer am I to my own group than to the next closest group?"

- **+1.0** = perfectly placed (far from other clusters, close to own)
- **0.0** = on the boundary between two clusters
- **−1.0** = probably in the wrong cluster

The average silhouette across all legislators gives an overall quality score. Rousseeuw's guideline: above 0.7 is strong structure, above 0.5 is reasonable, below 0.25 means the data has no meaningful cluster structure.

Because k-means is sensitive to initialization (different starting centers can give different results), Tallgrass runs it 10 times with different random starts and keeps the best solution.

**Codebase:** `analysis/09_clustering/clustering.py` (`run_kmeans_irt()`, `K_RANGE = range(2, 8)`, `n_init=10`)

## Algorithm 3: Gaussian Mixture Models (Soft Membership)

### The Idea

K-means assigns each legislator to exactly one cluster — you're either in Group A or Group B, with no in-between. But what about the legislator who sits right on the boundary? K-means says they're 100% Group A, even though they're practically equidistant from both centers.

**Gaussian Mixture Models** (GMMs) fix this by assigning each legislator a *probability* of belonging to each cluster. A legislator might be 85% Group A and 15% Group B — a soft, probabilistic membership rather than a hard assignment.

Formally, a GMM assumes the data is generated by *k* overlapping bell curves (Gaussian distributions), each with its own center, spread, and weight. The **EM algorithm** (Dempster, Laird, and Rubin, 1977) estimates the parameters of all *k* bell curves simultaneously.

### Uncertainty-Weighted GMM

Tallgrass adds a wrinkle that standard GMM doesn't have: it **weights legislators by the precision of their IRT estimate**. A legislator with a tight credible interval (low ξ_sd, meaning their ideal point is precisely estimated) gets more weight than one with a wide interval (high ξ_sd, meaning their ideal point is uncertain).

The weight is 1/ξ_sd — legislators the model is more confident about influence the clustering more. This prevents noisy estimates from distorting the clusters.

**Codebase:** `analysis/09_clustering/clustering.py` (`run_gmm_irt()`, `GMM_COVARIANCE = "full"`, `GMM_N_INIT = 20`)

## Algorithm 4: Spectral Clustering (Islands of Agreement)

### The Idea

Spectral clustering comes from graph theory rather than geometry. Instead of measuring distances in ideological space, it works directly on the **Kappa agreement matrix** — the same matrix we visualized as a heatmap in Volume 3.

The algorithm treats the agreement matrix as a network: legislators are nodes, and each pair is connected by an edge weighted by their Kappa. It then finds the eigenvalues and eigenvectors of a matrix derived from this network (the graph Laplacian) and clusters in that eigenspace.

The advantage of spectral clustering is that it can find **non-convex clusters** — groups that aren't shaped like circles or ellipses. If two legislators agree strongly but are separated by a chain of intermediaries in ideological space, spectral clustering can still group them together.

**Codebase:** `analysis/09_clustering/clustering.py` (`run_spectral_clustering()`, `assign_labels="cluster_qr"`)

## Algorithm 5: HDBSCAN (Find the Cities)

### The Idea

The four previous algorithms all require you to specify how many groups to find. HDBSCAN doesn't. Instead, it finds clusters automatically by looking for **dense regions** separated by sparse gaps.

**Campello, Moulavi, and Sander** published HDBSCAN in 2013 as an improvement on the classic DBSCAN algorithm (Ester et al., 1996). The original DBSCAN required a global density threshold, which meant it couldn't find clusters that were denser in some places and sparser in others. HDBSCAN runs the algorithm across all possible density thresholds, builds a hierarchy of clusters, and selects the most *stable* clusters — the ones that persist across a wide range of threshold values.

A key feature: HDBSCAN can label data points as **noise** — legislators who don't belong to any cluster. In the Kansas context, these might be extreme outliers or legislators whose voting patterns don't fit either party's profile.

Tallgrass feeds HDBSCAN the first 10 principal components from PCA, giving it a richer input space than the 1D ideal points used by k-means.

**Codebase:** `analysis/09_clustering/clustering.py` (`run_hdbscan_pca()`, `HDBSCAN_MIN_CLUSTER_SIZE = 5`, `HDBSCAN_MIN_SAMPLES = 3`)

## The Verdict: k = 2

After running all five algorithms across k = 2 through k = 7, the answer is consistent and striking:

**Optimal k is 2, and the two clusters correspond almost exactly to the two parties.**

| Chamber | Method | Optimal k | Silhouette at k=2 |
|---------|--------|-----------|---------------------|
| House | Hierarchical | 2 | 0.75 |
| House | K-means (1D) | 2 | 0.82 |
| Senate | Hierarchical | 2 | 0.71 |
| Senate | K-means (1D) | 2 | 0.79 |

All Republicans land in one cluster. All Democrats land in the other. The silhouette scores are strong (above 0.7), meaning the two-group structure is well-defined — legislators are much closer to their own party than to the other.

### Cross-Method Agreement

To verify that the five methods agree, Tallgrass computes the **Adjusted Rand Index** (ARI) between every pair of clusterings. ARI corrects for chance, like Kappa does for agreement: a score of 1.0 means identical clusterings, 0.0 means they agree no more than random assignments.

Typical results: mean ARI above 0.93 for the House and above 0.90 for the Senate. The five methods agree overwhelmingly.

**Codebase:** `analysis/09_clustering/clustering.py` (`compare_methods()`)

## What About k = 3?

The pundits would tell you Kansas has three groups: conservative Republicans, moderate Republicans, and Democrats. Is there evidence for this?

Tallgrass forces k = 3 and examines the result. The third cluster is usually a handful of legislators carved off one end of the Republican distribution — the most moderate or the most conservative. But the silhouette score drops from ~0.75 to ~0.60, and the third cluster has fuzzy boundaries.

More telling: **within-party clustering** runs k-means separately on just the Republicans and just the Democrats. If there were a genuine moderate-conservative split in the Republican caucus, it would show up as a clear k = 2 within that party. Instead, silhouette scores for within-party clustering are flat — around 0.55–0.62 across k = 2 through k = 7, with no peak. The variation within each party is real but continuous, like a gradient rather than a staircase.

This is an important distinction. You can always *draw a line* at any point on a continuous spectrum and call the two sides "moderate" and "conservative." But that line is arbitrary — the data doesn't prefer one dividing point over another. When a genuine faction exists, the data will show a valley in the distribution and a clear peak in the silhouette score. Kansas doesn't show that.

**Codebase:** `analysis/09_clustering/clustering.py` (`run_within_party_clustering()`, `WITHIN_PARTY_MIN_SIZE = 15`)

## Party Loyalty: A Second Dimension

While clustering on ideology alone yields k = 2, there's another dimension worth exploring: **party loyalty**. A legislator can be ideologically moderate but vote with their party 98% of the time (a loyal moderate), or ideologically extreme but frequently defect (an independent-minded conservative).

Tallgrass computes a loyalty rate for each legislator: on votes where at least 10% of the party dissented, how often did this legislator vote with the party majority?

Plotting IRT ideal point (x-axis) against loyalty rate (y-axis) reveals a 2D landscape where the legislature's texture becomes clearer. Most legislators cluster in the upper portion (high loyalty), but the ones in the lower-left and lower-right corners — the mavericks — are the ones who make headlines.

**Codebase:** `analysis/09_clustering/clustering.py` (`compute_party_loyalty()`, `CONTESTED_PARTY_THRESHOLD = 0.10`)

## Veto Override Analysis

One window into cross-party coalitions: **veto override votes**. The Kansas governor's veto requires a two-thirds supermajority to override, which means some Republicans and Democrats must vote together. These votes are rare (typically 30–40 per biennium) but politically significant.

On veto overrides, the two-cluster structure briefly breaks down. Override coalitions don't follow party lines — they follow a different logic (usually bipartisan agreement that the governor overstepped). The override analysis confirms that the k = 2 party structure is the dominant pattern but not the only one.

---

## Key Takeaway

Five different clustering algorithms agree: the Kansas Legislature has two natural groups, and they are the two parties. The gradient from moderate to conservative within each party is real and measurable, but it's a continuous spectrum — there's no natural break point that separates "moderate Republicans" from "conservative Republicans." When pundits talk about factions, they're drawing lines on a smooth distribution. That doesn't mean the distinction is meaningless, but it means the data supports *degrees*, not *categories*.

---

*Terms introduced: hierarchical clustering, dendrogram, average linkage, cophenetic correlation, k-means, inertia, elbow plot, silhouette score, Gaussian Mixture Model, EM algorithm, soft membership, spectral clustering, HDBSCAN, density-based clustering, noise points, Adjusted Rand Index, within-party clustering, party loyalty rate*

*Next: [Latent Class Analysis: Testing for Hidden Groups](ch02-latent-class-analysis.md)*
