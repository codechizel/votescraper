# Analytic Flags

Observations, outliers, and data points flagged during quantitative analysis that warrant qualitative investigation or special handling in downstream phases. Each entry records **what** was observed, **where** (which analysis phase), **why** it matters, and **what to do about it**.

This is a living document — add entries as each analysis phase surfaces new findings.

## Flagged Legislators

### Sen. Caryn Tyson (R, District 12)

- **Phase:** PCA, IRT
- **Observation:** Extreme PC2 outlier at -24.8 (3x the next senator). PCA PC1 rank: 23rd of 32 Rs. **IRT rank: 1st (most conservative, xi=+4.17).** Jumped 22 ranks from PCA to IRT — the largest shift in the chamber.
- **Explanation:** Tyson has a 61.9% Yea rate and 74 Nay votes — more than double the Republican median. But her contrarian Nay votes are concentrated on low-discrimination bills (|beta| <= 1.5), which IRT downweights. On high-discrimination bills (|beta| > 1.5), she is 100% conservative: 63/63 Yea on R-Yea, 18/18 Nay on D-Yea. No other senator has a perfect record. Additionally, 31 of her 41 dissent votes (Nay where >80% Rs = Yea) are on negative-beta bills, meaning her dissent actually *reinforces* her conservative score rather than moderating it. This is a 1D model limitation: her two-dimensional behavior (ideology + contrarianism) is compressed into a single axis, and the axis captures the dimension that's most informative about ideology.
- **Downstream:**
  - **Clustering:** Tyson's IRT position will pull any cluster she's in toward an extreme. Consider supplementing IRT with a party loyalty metric to distinguish "ideologically extreme" from "unreliable caucus member."
  - **Prediction:** IRT ideal points will predict her partisan votes well but miss her contrarian dissent on routine bills. A 2D model would improve predictions for Tyson specifically.
  - **Interpretation:** Always present Tyson's ranking with the caveat that 1D IRT conflates "consistently conservative on partisan votes" with "most conservative overall." See `analysis/design/tyson_paradox.md` for full investigation.

### Sen. Mike Thompson (R, District 10)

- **Phase:** PCA
- **Observation:** Third-most extreme PC2 (-8.0). Same direction as Tyson but milder.
- **Explanation:** Similar pattern — higher-than-typical Nay rate on routine bills (73.4% Yea rate). Shows a softer version of the Tyson contrarian tendency.
- **Downstream:** Same as Tyson. Check if Thompson and Tyson form a recognizable caucus or voting bloc. Clustering phase should reveal whether they consistently co-vote.

### Sen. Silas Miller (D, District ?)

- **Phase:** PCA
- **Observation:** Second-most extreme PC2 (-10.9). Only 30/194 votes (15.5%) — dead last in Senate participation.
- **Explanation:** Mid-session replacement. Previously served in the House with a normal voting record. Row-mean imputation filled 85% of his Senate matrix with his average Yea rate, producing an artificial PC2 extreme. **This is an imputation artifact, not a real voting pattern.**
- **Downstream:**
  - **IRT (Phase 4):** Use Miller as a **bridging legislator**. He served in both chambers, so a joint IRT model can use his ~300+ House votes to tightly constrain his ideal point, with the Senate votes further refining it. This is the standard "bridging observations" technique in the ideal-point literature.
  - **Clustering:** Exclude from Senate clustering or flag his cluster assignment as low-confidence.
  - **General:** Any analysis with a minimum-participation filter should note that Miller barely clears the 20-vote threshold. His estimates carry much more uncertainty than typical senators.

## Flagged Voting Patterns

### PC2 as "Contrarianism on Routine Legislation"

- **Phase:** PCA (Senate)
- **Observation:** PC2 (11.2% of variance) is driven by a cluster of near-unanimous bills where 1-2 senators dissent. The top PC2 loadings are routine bills (consent calendar, waterfowl hunting regs, bond validation, National Guard education).
- **Interpretation:** This is not a traditional ideological dimension. It captures a tendency to vote against the chamber consensus on uncontroversial legislation. Tyson is the primary driver, Thompson secondary, with Miller's position artifactual.
- **Downstream:** When interpreting Senate clustering results, the Tyson/Thompson pattern may create a spurious "cluster" that is really just two contrarian voters, not a substantive ideological faction. Consider whether PC2 should be downweighted or excluded in clustering inputs.

### Sen. Silas Miller (D) — IRT Update

- **Phase:** IRT
- **Observation:** IRT ideal point xi=-0.892, HDI=[-1.341, -0.439], width=0.902 (13th widest of 42 senators). Despite having only 30/194 votes (15.5%), his HDI is not the widest — extreme conservative senators have wider intervals due to fewer discriminating bills at the tail.
- **Explanation:** IRT handles Miller's sparse data natively (absences absent from likelihood, no imputation). His 30 observed votes are consistent enough to produce a reasonably constrained estimate. The PC2 artifact from PCA does not carry over — this is exactly the improvement IRT provides over PCA for sparse legislators.
- **Downstream:**
  - **Clustering:** HDI width of 0.902 means his ideal point is less certain than most Democrats. Consider weighting by 1/xi_sd or flagging his cluster assignment.
  - **Bridging:** A joint cross-chamber IRT model could use his ~300+ House votes to tighten the Senate estimate further. Deferred to future enhancement.

### Sen. Scott Hill (R)

- **Phase:** IRT
- **Observation:** Widest HDI in Senate: width=2.028 (xi=+1.329, HDI=[+0.398, +2.426]). Well-separated from the pack (next widest is 1.412).
- **Explanation:** Likely low participation on contested votes or voting pattern that doesn't align cleanly with the 1D model. Warrants investigation.
- **Downstream:** Cluster assignment is lowest-confidence in Senate. Flag in any ranking or comparison.

### House ESS Warning — Resolved

- **Phase:** IRT
- **Observation:** With LogNormal beta prior, minimum ESS for House ideal points was 214 (threshold: 400). **After switching to Normal(0,1) beta prior, ESS min improved to 894 — well above threshold.** All convergence checks now pass.
- **Explanation:** The LogNormal prior created poor posterior geometry (bimodal beta distribution, ceiling effects). The unconstrained Normal prior resolved this, improving ESS by 4x and eliminating the only convergence warning.
- **Status:** Resolved as of 2026-02-20 (Normal(0,1) beta prior).

## Flagged Voting Patterns — IRT

### Sensitivity Analysis: Highly Robust

- **Phase:** IRT
- **Observation:** Ideal points are extremely stable across minority thresholds. Pearson r between 2.5% and 10% runs: House r=0.9982, Senate r=0.9930.
- **Interpretation:** The 1D ideological structure is not driven by borderline-contested votes. Removing them barely changes legislator positions. This validates the 2.5% default threshold.

### PCA-IRT Agreement

- **Phase:** IRT
- **Observation:** Pearson r with PCA PC1: House r=0.972, Senate r=0.939. Both above 0.90; House exceeds the 0.95 "strong" threshold.
- **Interpretation:** High agreement confirms both methods recover the same 1D structure. The Senate's lower r=0.939 reflects two things: (1) IRT weights discriminating bills more heavily (inflating Tyson/Thompson), and (2) IRT handles sparse data (Miller) without imputation artifacts. See `analysis/design/tyson_paradox.md` for a detailed investigation of the largest PCA-IRT rank divergences.
- **Downstream:** Use IRT ideal points (not PCA scores) as the primary input for clustering and network analysis. IRT provides uncertainty estimates and handles missing data properly. But be aware that IRT ideal points systematically inflate the ranking of contrarian legislators (Tyson, Thompson) relative to PCA.

## Flagged Voting Patterns — Clustering

### k=2 Optimal, k=3 Hypothesis Rejected

- **Phase:** Clustering
- **Observation:** Both hierarchical (Ward on Kappa) and k-means (on IRT) selected k=2 as optimal for both chambers. Silhouette at k=2 = 0.82 (House), 0.79 (Senate); at k=3 = 0.64 (House), 0.57 (Senate). GMM selected k=4 by BIC.
- **Explanation:** The moderate/conservative Republican distinction is continuous, not discrete. The party boundary is the dominant clustering structure. With a ~72% Republican supermajority, intra-R variation is spread smoothly across the ideal-point spectrum. GMM's k=4 likely captures distributional shape (e.g., the long right tail) rather than genuine factions.
- **Downstream:**
  - **Network:** Community detection may find finer structure than k-means because it operates on pairwise agreement edges, not centroids. Test whether Louvain/modularity recovers 3+ communities.
  - **Prediction:** Cluster labels at k=2 are equivalent to party and won't add predictive power. Consider party loyalty as a continuous feature instead.

### Tyson and Thompson Cluster Assignments

- **Phase:** Clustering
- **Observation:** Tyson (xi=+4.17, loyalty=0.417) and Thompson (xi=+3.44, loyalty=0.472) both cluster with conservative Rs (Cluster 0) at k=2. They have the two lowest party loyalty scores in the Senate.
- **Explanation:** Their extreme IRT positions dominate the 1D clustering — they're far from Democrats and firmly in the R cluster. The 2D (IRT x loyalty) scatter plot visually separates them from core party members despite same cluster assignment.
- **Downstream:** The party loyalty metric successfully distinguishes "ideologically extreme" from "reliable caucus member." For network analysis, Tyson and Thompson may have lower within-cluster edge weights than typical Rs.

### Miller and Hill Cluster Confidence

- **Phase:** Clustering
- **Observation:** Miller (xi=-2.19, loyalty=1.000) clusters with Democrats; Hill (xi=+1.44, loyalty=1.000) clusters with Republicans. Miller has perfect loyalty on his contested votes; Hill has widest HDI in Senate.
- **Explanation:** Miller's sparse data (30/194 votes) produces wider IRT uncertainty, but his cluster assignment is unambiguous (firmly D). Hill's wide HDI (2.03) means his ideal point could range from moderate to solidly conservative, but he still clusters with Rs.
- **Downstream:** Both assignments are stable across methods (ARI = 0.90+ across all pairs). No special handling needed.

### Cross-Method Agreement — Very Strong

- **Phase:** Clustering
- **Observation:** Mean ARI = 0.958 (House), 0.935 (Senate) across hierarchical/k-means/GMM. Hierarchical and k-means are perfectly aligned (ARI = 1.0 for Senate).
- **Explanation:** The 2-cluster structure is extremely robust. The high ARI despite different input spaces (Kappa distance vs IRT ideal points) and algorithms confirms the party split is the overwhelming signal.
- **Downstream:** High confidence that any network community detection finding >2 groups represents genuinely finer structure, not algorithmic noise.

### Veto Overrides — Strictly Party-Line

- **Phase:** Clustering
- **Observation:** 17 veto override votes per chamber. R cluster: 98% Yea (House), 98% Yea (Senate). D cluster: 1% Yea (House), 1% Yea (Senate). No cross-party coalition detected.
- **Explanation:** Unlike Congress where veto overrides often produce bipartisan coalitions, Kansas overrides in the 2025-26 session are strictly partisan. The R supermajority can override unilaterally without D votes.
- **Downstream:** Veto override subgroup adds no novel clustering structure. Network analysis may find override votes are among the most party-line (highest discrimination).

### Within-Party Clustering — Weakly Structured Continuous Variation

- **Phase:** Clustering (within-party)
- **Observation:** Within-party k-means on Republicans yields silhouette > 0.50 for all k in [2, 7], but the silhouette curve is essentially flat (House R: 0.597-0.605 across k=2-7; Senate R: 0.534-0.606). Optimal k is 6 (House R, 1D), 7 (House D, 1D), 3 (Senate R, 1D). Senate Democrats skipped (10 < 15 minimum).
- **Explanation:** The intra-party variation is real (silhouette > 0.50) but not strongly structured into discrete factions. The flat silhouette profile means the "optimal" k is somewhat arbitrary — a k=3 and k=6 partition have nearly identical silhouette scores. This is consistent with legislators spread across a continuous ideological spectrum within each party, rather than forming distinct moderate/conservative wings.
- **Downstream:**
  - **Network:** Community detection on the within-party agreement subgraph may reveal gradients rather than discrete communities. Consider edge-weighting by Kappa agreement and using Louvain with resolution parameter tuning.
  - **Prediction:** Use continuous features (IRT ideal points, party loyalty rates) rather than within-party cluster labels. Cluster labels add little information when the silhouette profile is flat.
  - **Interpretation:** Do not over-interpret within-party cluster labels. The k=6 (House R) and k=3 (Senate R) partitions are convenient summaries, not evidence of 6 or 3 distinct factions.

### Senate Republican k=3 — Tyson/Thompson Subcluster

- **Phase:** Clustering (within-party)
- **Observation:** Senate Republicans at k=3 (1D silhouette=0.606) show a modest peak relative to k=2 (0.534). The 2D (IRT + loyalty) analysis at k=4 reaches 0.612. Tyson (xi=+4.17, loyalty=0.417) and Thompson (xi=+3.44, loyalty=0.472) may form a distinct low-loyalty extreme.
- **Explanation:** These two senators occupy a unique position in the 2D space: extreme ideology combined with low party loyalty. Adding the loyalty dimension modestly improves cluster quality (2D sil > 1D sil at k=3-4), suggesting the loyalty axis contributes to the structure.
- **Downstream:** Network analysis should check whether Tyson-Thompson have lower within-Republican edge weights than typical R pairs, confirming their distinctiveness in pairwise agreement (not just in the IRT+loyalty feature space).

## Joint Cross-Chamber Model

### Joint MCMC Model — Failed, Replaced with Test Equating

- **Phase:** IRT (Joint)
- **Observation:** A full joint MCMC IRT model was attempted with 71 shared bills and 169 legislators. It did not converge: R-hat > 1.7, ESS < 10, despite 4 anchors (one per chamber extreme), 4 chains, target_accept=0.95, and a shared-bills-only matrix (95.5% observed). Both full-matrix (420 cols, 61.3% observed) and shared-bills-only (71 cols, 95.5%) variants failed identically.
- **Explanation:** 71 shared bills for 169 legislators gives 0.42 bills per legislator — far too few for a joint IRT model. Many legislators vote identically on the shared bills, creating a degenerate posterior. The block-diagonal structure (House-only and Senate-only columns) further complicates MCMC geometry even when removed.
- **Resolution:** Replaced with classical test equating (mean/sigma method):
  - **A = 1.136** (scale factor) from SD ratio of shared bill discrimination parameters (51 concordant / 71 shared bills)
  - **B = -0.305** (location shift) from 3 bridging legislators' per-chamber ideal points
  - Senate ideal points transformed to House scale: xi_equated = 1.136 × xi_senate - 0.305
  - Equated vs per-chamber correlations: House r = 1.000 (unchanged), Senate r = 1.000 (linear transformation)
- **Downstream:**
  - **Network:** Equated ideal points enable cross-chamber comparisons. The Senate scale is ~14% wider than House (A > 1); Tyson's equated xi = +4.43 exceeds any House member (+2.90 max).
  - **Interpretation:** Equated scores are transformed marginals, not a joint posterior. Use per-chamber models for within-chamber analyses. Equated scores for cross-chamber ranking only.
  - **Limitation:** B depends on only 3 bridging legislators. Thompson's large cross-chamber shift (House +2.43 → Senate +3.44) may reflect genuine ideology change or model noise.
- **Status:** Resolved as of 2026-02-20 (test equating). Joint MCMC deferred to future work if more shared items become available.

## Flagged Voting Patterns — Network

### Community Detection Confirms Party as Dominant Structure

- **Phase:** Network
- **Observation:** Louvain community detection at default resolution (1.0) recovers exactly 2 communities for both chambers. NMI = 1.0 and ARI = 1.0 vs party labels in both House and Senate. Community assignments are identical to k-means k=2 clusters (NMI = 1.0, ARI = 1.0).
- **Explanation:** At Kappa threshold 0.40 ("substantial" agreement), the party boundary is the dominant network structure. Within-party edge density is much higher than cross-party edge density. No misclassified legislators at any resolution that recovers 2 communities.
- **Downstream:**
  - **Prediction:** Community membership at default resolution adds no information beyond party. Higher-resolution communities (3+) may provide finer features.
  - **Interpretation:** The convergence of community detection, clustering, and simple party labels confirms the overwhelming party-line voting pattern.

### Zero Cross-Party Edges at κ=0.40

- **Phase:** Network
- **Observation:** Party assortativity = 1.0 in both chambers. At Kappa threshold 0.40, there are zero cross-party edges — the graph is two completely disconnected components (one per party). House has 4,604 edges, all within-party (94% of possible within-party edges). Senate has 495 edges, all within-party (91%).
- **Explanation:** Kappa corrects for the 82% Yea base rate. After correction, no cross-party pair reaches "substantial" (0.40) agreement. The Senate's maximum cross-party Kappa is only 0.078 (Dietrich–Pettey). The House maximum is 0.369 (Schreiber–Tom Sawyer).
- **Downstream:**
  - Community detection at default resolution trivially recovers party labels (NMI=1.0, ARI=1.0) because it's finding disconnected components, not detecting subtle structure. This result does not add information beyond assortativity=1.0.
  - Betweenness centrality is computed within each disconnected component. "High betweenness" means central within own party — not a bridge between parties. No legislator bridges R and D at this threshold.
  - All centrality measures are within-party measures. They identify intra-party structural importance, not bipartisan connectors.

### Rep. Mark Schreiber (R, House) — Most Bipartisan House Member

- **Phase:** Network
- **Observation:** Schreiber has the highest cross-party Kappa in the House: 0.369 with Tom Sawyer (D). All 10 House cross-party Kappa values above 0.30 involve Schreiber. IRT ideal point: +0.018 (essentially zero — the most centrist Republican). Party loyalty: 0.617 (low for House R).
- **Explanation:** Schreiber votes more like Democrats than any other Republican. At threshold 0.30, his cross-party edges make the House network a single connected component (vs 2 components at 0.40). He is the only legislator whose Kappa with the opposing party reaches even "fair" agreement.
- **Downstream:**
  - **Prediction:** Schreiber is likely the hardest House Republican to predict. His near-zero IRT position and low loyalty suggest he votes on issue-specific rather than party-line grounds.
  - **Interpretation:** In the Senate, the maximum cross-party Kappa is only 0.078 — the Senate is far more polarized than the House. The House has at least one genuine cross-party actor; the Senate does not.

### Tyson and Thompson — Lower Within-Republican Edge Weights

- **Phase:** Network
- **Observation:** Tyson's mean within-R edge weight = 0.493; Thompson's = 0.513. R-R median = 0.665. Both are 0.15–0.17 below median. Crucially, Tyson has only 5 out of 31 possible R-R edges (16%), meaning her Kappa with 26 other Rs falls below 0.40. Thompson keeps 27/31 (87%).
- **Explanation:** Tyson is genuinely isolated within her party — disconnected from 84% of Republican senators. Thompson is well-connected but with weaker-than-average edge weights. This quantifies the "Tyson paradox" in network terms: despite her extreme IRT position, Tyson's routine-bill dissent makes her pairwise agreement with most Rs only "fair" or below.
- **Downstream:** Tyson's 5 edges make her nearly an isolate within the R component. Thompson's position is less extreme — weaker connections, but still integrated.

### Within-Party Community Structure — Near Zero Modularity

- **Phase:** Network
- **Observation:** Within-party Louvain modularity: House R = 0.010, House D = 0.026, Senate R = 0.021, Senate D = -0.000. These are indistinguishable from random partitions.
- **Explanation:** Within-party graphs are extremely dense (House R: 93.5% of possible edges, House D: 98.3%). Louvain cannot find meaningful community structure in a near-complete graph. The multi-resolution sweep shows a sharp jump from 2–3 communities to 35+ (House) or 10+ (Senate) at resolution 1.25, with no gradual emergence of subcaucuses. This confirms clustering's finding: within-party variation is continuous, not factional.
- **Downstream:** Within-party community labels from network analysis should not be used as features — they are noise. Continue using continuous features (IRT, party loyalty) for within-party differentiation.

### Veto Override Subnetwork — Kappa Undefined for Homogeneous Votes

- **Phase:** Network
- **Observation:** Override subnetwork: House 21 edges, Senate 2 edges (at κ=0.40). The near-zero edge count is a methodological artifact, not just a substantive finding.
- **Explanation:** With 17 override votes per chamber and near-unanimous within-party voting, most legislator pairs vote identically on all shared overrides. Cohen's Kappa requires both classes (Yea and Nay) to compute — identical-vote pairs produce NaN (undefined Kappa), which maps to no edge. The ~7,100 sklearn warnings in the run log are from these NaN computations. The 21 House edges likely come from pairs where absences or rare defections produced enough variation for Kappa to be defined.
- **Downstream:** Override network is too sparse for any graph analysis. Kappa is the wrong metric for near-unanimous vote subsets. If override behavior is needed for prediction, use raw Yea/Nay counts or a party-defection indicator, not pairwise Kappa.

### High-Discrimination Subnetwork — Denser, Not More Informative

- **Phase:** Network
- **Observation:** High-disc network (|beta| > 1.5): House 4,763 edges (vs 4,604 full), Senate 537 (vs 495). More edges, not fewer.
- **Explanation:** High-discrimination bills are the most partisan — everyone votes the party line. This makes within-party Kappa values higher (more agreement on the included bills), pushing more pairs above the 0.40 threshold. The high-disc subnetwork is an even denser version of the same party-dominated structure.
- **Downstream:** Does not provide novel cross-party signal. Useful for confirming the full network is dominated by ideological agreement (not noise from unanimous votes), but the high-disc subnetwork is not a better input for finding bipartisan structure.

### Threshold Sensitivity — Schreiber Transition at κ=0.30

- **Phase:** Network
- **Observation:** House has 1 component at κ=0.30 (connected graph), then 2 components at κ=0.40 (party split). Senate has 2 components at all thresholds (0.30–0.60). At κ=0.60, both chambers split further (House: 3 components, Senate: 3 components).
- **Explanation:** The House 1→2 component transition between 0.30 and 0.40 is caused by Schreiber's cross-party edges (max κ=0.369 with Democrats). He is the sole link between parties at the lower threshold. The Senate never connects because its max cross-party κ is only 0.078.
- **Downstream:** The 0.40 threshold is substantively meaningful — it's precisely the level where cross-party agreement vanishes. The choice of threshold is not arbitrary; it corresponds to a real phase transition in the data.

## Flagged Voting Patterns — Prediction

### Vote Prediction: IRT Features Dominate

- **Phase:** Prediction
- **Observation:** XGBoost holdout AUC = 0.984 (House), 0.979 (Senate). All three models (LogReg, XGBoost, RF) perform within 0.5% of each other. Performance far exceeds majority-class baseline (72.7% House, 75.9% Senate) and party-only baseline (~75%). An initial version included vote counts (yea_count, nay_count, margin) as features — target leakage. Removing them had negligible impact (AUC unchanged), confirming the IRT features carry the signal.
- **Explanation:** The IRT ideal points (xi_mean) and bill parameters (alpha_mean, beta_mean) capture the dominant voting structure so well that model choice barely matters. The xi_x_beta interaction feature (legislator position × bill discrimination) encodes the core IRT prediction directly. Adding network centrality, PCA scores, and party loyalty provides marginal improvement over IRT alone.
- **Downstream:** For any future prediction task, IRT ideal points + bill parameters are sufficient. The additional features (centrality, PCA, loyalty) add complexity without meaningful accuracy gains. A simple IRT-based prediction (logistic on xi × beta) would likely achieve AUC > 0.95.

### Hardest-to-Predict House Legislators

- **Phase:** Prediction
- **Observation:** Bottom 5 accuracy: Helgerson (D, 0.860, xi=-0.99), Carmichael (D, 0.863, xi=-2.74), Poetter Parshall (R, 0.865, xi=+0.56), Barth (R, 0.877, xi=+0.87), Winn (D, 0.882, xi=-3.25). The list includes both centrist Republicans and strongly ideological Democrats.
- **Explanation:** Helgerson (xi=-0.99) is the most moderate Democrat — close to the party boundary, making vote direction uncertain. Carmichael and Winn are extreme Democrats who occasionally cross party lines in ways the model doesn't predict. Poetter Parshall and Barth are moderate Republicans (xi +0.5 to +0.9) whose centrist positions leave them unpredictable on contested bills.
- **Downstream:** Notably, Schreiber (xi=+0.018, previously flagged as "hardest House R to predict") does NOT appear in the bottom 10. Despite his near-zero IRT position and low party loyalty (0.617), his votes are predictable enough — possibly because his centrism is consistent rather than erratic.

### Hardest-to-Predict Senate Legislators

- **Phase:** Prediction
- **Observation:** Bottom 5 accuracy: Shallenburger (R, 0.896, xi=+1.12), Clifford (R, 0.928, xi=+0.82), Titus (R, 0.948, xi=+1.15), Blew (R, 0.953, xi=+1.90), Francisco (D, 0.953, xi=-2.71). Shallenburger is 3.2% worse than the next senator.
- **Explanation:** Shallenburger (Vice President of the Senate) may exercise procedural votes that diverge from ideological predictions. Clifford (xi=+0.82) is the most moderate Senate Republican with significant vote counts, making his direction harder to call. Note: Tyson (xi=+4.17, loyalty=0.417) does NOT appear in the bottom 10 — her contrarian pattern on routine bills is apparently consistent enough to be predictable.
- **Downstream:** The 1D IRT model captures Tyson's behavior better than expected. Her dissent, while frequent, is concentrated on low-discrimination bills where the model already assigns lower confidence. The "Tyson paradox" remains interpretively important but not a prediction problem.

### Most Surprising Votes

- **Phase:** Prediction
- **Observation:** House #1 surprise: Shannon Francis (R) voted Nay on SB 105 (Conference Committee Report) — model was 99.96% confident Yea. Senate #1 surprise: Caryn Tyson (R) voted Yea on HB 2007 (Corson amendment, rejected) — model was 99.98% confident Nay. Most surprising votes in both chambers are Republicans voting Nay on bills with near-unanimous R support.
- **Explanation:** The Tyson surprise confirms the paradox from a different angle: the model correctly treats her as ultraconservative, but on this specific amendment (Corson), she broke from the conservative position. Francis's Nay on a conference committee report likely reflects a substantive policy objection invisible to the 1D model.
- **Downstream:** Issue-specific variables (bill topic, committee of origin) are the obvious missing features that could explain these high-confidence misses. The current feature set has no bill content — only structural features (IRT difficulty/discrimination, vote type, day of session).

### Bill Passage Prediction: Moderate Performance, Small N

- **Phase:** Prediction
- **Observation:** After removing leaky features (margin, alpha_mean), bill passage performance dropped substantially. House: best holdout AUC=0.955 (XGBoost), temporal split AUC=0.858 (RF). Senate: best holdout AUC=0.931 (LogReg), temporal split still 1.000 (small-N artifact, 59 test bills). Features are now limited to beta (discrimination), vote type, bill prefix, day of session, and is_veto_override.
- **Explanation:** The initial near-perfect results were driven by alpha_mean (IRT difficulty — a near-proxy for passage outcome) and margin (derived from vote counts that determine passage). With honest features, the model has limited signal: beta captures how partisan a bill is, vote type and bill prefix capture procedural structure, but none directly encode "will this pass." The remaining signal likely comes from beta — highly discriminating (partisan) bills have more predictable outcomes.
- **Downstream:** Bill passage prediction from pre-vote features alone is a genuinely hard problem. The Senate temporal split of 1.000 is still a small-N artifact (59 test bills, ~7 failures). Cross-session validation would provide more honest estimates. Adding bill text features (NLP on titles/descriptions) could help.

### Shallenburger Name Not Stripped

- **Phase:** Prediction (data quality)
- **Observation:** `full_name` = "Tim Shallenburger - Vice President of the Senate" in per-legislator output. The scraper's leadership suffix stripping did not catch this variant.
- **Explanation:** The scraper strips suffixes like "House Minority Caucus Chair" but may not have "Vice President of the Senate" in its pattern list.
- **Downstream:** Minor cosmetic issue. Does not affect analysis (joining is on `legislator_slug`, not name). Should be fixed in the scraper for clean reporting.

### IRT Anchor Legislators — 100% Prediction Accuracy (Trivially)

- **Phase:** Prediction
- **Observation:** Avery Anderson (R, House anchor, xi=+1.0, xi_sd=0.0) and Brooklynne Mosley (D, House anchor, xi=-1.0, xi_sd=0.0) both achieve 100% prediction accuracy on 297 votes. In the Senate, Miller (D, 30 votes) and Hill (R, 30 votes) also hit 100% but via small N, not anchoring.
- **Explanation:** Anderson and Mosley are the IRT anchors — their ideal points are fixed by convention, not estimated. With xi_sd=0, the model has zero uncertainty about their positions, giving it an inherent edge. Their 100% accuracy is real (they do vote predictably) but slightly inflated relative to legislators whose xi carries estimation noise. Caiharr (R, 88 votes, 100%) is a non-anchor with perfect accuracy — genuinely predictable.
- **Downstream:** When reporting "perfect accuracy" legislators, note that anchors benefit from an informational advantage. This does not invalidate the metric — it reflects both true predictability and the anchoring design. For cross-session comparisons, anchor legislators should not be used to benchmark model quality.

### Bill Passage CV Variance — Senate Small-N Artifact

- **Phase:** Prediction
- **Observation:** Senate bill passage 5-fold CV shows extreme fold variance: LogReg AUC = [0.88, 0.77, 1.00, 1.00, 1.00] (std=0.10). Three folds achieve perfect AUC, two are below 0.90. XGBoost is worse: [0.72, 0.76, 1.00, 1.00, 1.00] (std=0.14).
- **Explanation:** With 194 bills (23 failures), each fold has ~39 bills (~4-5 failures). A fold with only 4 failures is trivially separable by chance. The 1.000 folds happened to get "easy" splits; the low folds got the harder cases. This is a fundamental small-N problem — CV variance scales inversely with test-set size.
- **Downstream:** Senate bill passage CV numbers are unreliable as point estimates. The temporal split (AUC=0.84-0.86) is more honest because it tests on a contiguous block of 59 bills. Report the temporal split as the primary metric; CV for directional comparison only. Cross-session validation (train on 2023-24, test on 2025-26) would provide the most honest estimate but requires scraping an additional session.

## Prediction Phase — Quality Assessment

**Date:** 2026-02-21. Covers the complete prediction phase after bug fixes (temporal sort, surprising bills filter, dead feature removal, target leakage removal).

### Vote Prediction: Strong and Credible

Results are internally consistent and match upstream phase expectations:

| Metric | House | Senate |
|--------|-------|--------|
| XGBoost holdout AUC | 0.984 | 0.979 |
| XGBoost holdout accuracy | 94.5% | 94.5% |
| Majority-class baseline | 72.7% | 75.9% |
| Party-only baseline | 75.5% | 75.4% |
| CV std (AUC) | 0.001 | 0.002 |

- All 3 models within 0.3% accuracy of each other — the signal is linear and well-captured by IRT features alone.
- CV is tight (std 0.001-0.002), no fold instability.
- 8 House legislators achieve 100% accuracy (all strong-ideology Rs); lowest is Helgerson at 86.0%.
- Senate range: 89.6% (Shallenburger) to 100% (6 legislators).

**Verdict:** No anomalies. AUC of 0.98 is realistic for IRT-feature-based legislative vote prediction — the IRT model *is* a vote prediction model, so high AUC validates IRT rather than indicating leakage.

### Per-Legislator Accuracy: Consistent with Prior Flags

- Hardest legislators are centrists and occasional crossovers — expected.
- Schreiber (previously flagged as "hardest House R") is NOT in the bottom 10 — his centrism is consistent, not erratic.
- Tyson (previously flagged as unpredictable) is also NOT in the bottom 10 — her routine-bill dissent targets low-discrimination bills where the model assigns lower confidence anyway.
- Shallenburger's procedural role plausibly explains his bottom ranking.

**Verdict:** The model's failures are concentrated where they should be: centrists and procedural outliers.

### Bill Passage: Honest but Fragile

| Metric | House | Senate |
|--------|-------|--------|
| Best CV AUC | 0.955 (XGBoost) | 0.931 (LogReg) |
| Temporal split AUC | 0.900-0.959 | 0.835-0.863 |
| N (total / failures) | 297 / 41 | 194 / 23 |

- Senate CV is unreliable (3 folds at AUC 1.000, 2 at 0.77-0.82). See flag above.
- Temporal split is the honest metric: House AUC ~0.90, Senate AUC ~0.84.
- Only 1 surprising bill in House (SB 125 veto override at 70.7% confidence), 3 in Senate — the model gets almost everything right, and misses are edge cases.
- With only beta, vote_type, bill_prefix, day_of_session, and is_veto_override as features, moderate AUC is expected. The hard-to-predict bills are those where procedural context or bill content matters.

**Verdict:** Honest results after leakage removal. Not strong enough for production use, but demonstrates that structural features carry some passage signal. NLP on bill text is the obvious next feature to add.

### Key Takeaway

XGBoost adds almost nothing over logistic regression on xi x beta. The IRT ideal points are doing virtually all the work. This means the prediction phase validates the IRT model rather than discovering new predictive structure. The analytically interesting output is not the model performance (which is expected to be high) but the *residuals*: which legislators and which votes the model fails on. Those residuals point to the limits of 1D ideology as an explanatory framework.

## Flagged Voting Patterns — Indices

### CQ Unity vs Clustering Loyalty — Definitional Divergence

- **Phase:** Indices
- **Observation:** Tyson's CQ party unity = 0.917 (92nd percentile for Senate) vs clustering party loyalty = 0.417 (lowest in Senate). Schreiber: CQ unity = 0.615 (lowest House R) vs clustering loyalty = 0.617 (similar). The two metrics agree for centrists but diverge sharply for contrarians.
- **Explanation:** CQ unity only counts "party votes" (majority-R opposes majority-D), which are the most partisan roll calls. On these high-stakes votes, even Tyson votes with her party 92% of the time. Clustering loyalty uses a 10% dissent threshold, capturing internal-party dissent on routine bills — where Tyson's contrarian pattern dominates. The metrics answer different questions: CQ asks "does she vote with Rs against Ds?" (yes); clustering asks "does she agree with the Rs who agree with each other?" (no).
- **Downstream:** When presenting "party loyalty" to nontechnical audiences, always specify which definition is being used. CQ unity is the standard political science metric and should be primary; clustering loyalty is a complementary within-party measure. The Tyson divergence is a useful illustration of why the definition matters.

### Schreiber — Top House Maverick (Strategic Defector)

- **Phase:** Indices
- **Observation:** Schreiber has the highest maverick rate in the House (0.385 unweighted, 0.541 weighted). His weighted maverick substantially exceeds his unweighted rate, placing him firmly above the 1:1 diagonal on the maverick landscape plot.
- **Explanation:** Schreiber's defections are concentrated on close votes where his vote could change the outcome (weighted > unweighted = strategic). This is consistent with his near-zero IRT ideal point (+0.018) and maximum cross-party Kappa (0.369 with Tom Sawyer) from the network phase.
- **Downstream:** Schreiber is the single most analytically interesting House member. He is the most centrist, most bipartisan, and most strategically independent Republican — and yet his votes are not the hardest to predict (not in bottom 10 for prediction accuracy). His pattern is consistent rather than erratic.

### Dietrich — Top Senate Maverick

- **Phase:** Indices
- **Observation:** Sen. Brenda Dietrich has the highest Senate maverick rate (0.231 unweighted, 0.362 weighted). Like Schreiber, her weighted rate substantially exceeds unweighted, indicating strategic defections on close votes.
- **Explanation:** Dietrich was not previously flagged in any upstream phase. Her IRT ideal point and clustering assignment should be checked — she may be a moderate R who escaped detection because she votes the party line on uncontested bills.
- **Downstream:** Investigate Dietrich's IRT position and network centrality. She may be the Senate analogue of Schreiber.

### Sensitivity Analysis — Perfect Stability

- **Phase:** Indices
- **Observation:** Spearman rho = 1.000 for both chambers when comparing party unity on all votes vs EDA-filtered votes. Maximum rank change = 0 for both.
- **Explanation:** The EDA filter removes near-unanimous votes, but the CQ party vote definition already excludes those (they can't be "party votes" if both parties vote the same way). The set of party votes is identical whether or not near-unanimous votes are removed from the denominator.
- **Downstream:** This confirms that the CQ party vote definition is robust to the EDA filter. No need to run sensitivity analysis on indices in future sessions — it will always be 1.0.

### Veto Overrides — Near-Perfect Party Cohesion

- **Phase:** Indices
- **Observation:** Override Rice: House R=0.959, D=0.986; Senate R=0.966, D=0.974. Both parties show near-unanimous cohesion on override votes.
- **Downstream:** Confirms the clustering finding that overrides are strictly party-line. The tiny deviations from 1.0 are from 1-2 absences or rare defections.

## Flagged Patterns — Synthesis

### Data-Driven Detection Replaces Hardcoded Legislators

- **Phase:** Synthesis
- **Observation:** Automated detection in the 2025-2026 session identifies 6 notable legislators across 3 roles: Schreiber (house maverick), Dietrich (senate maverick), Borjon (house bridge), Hill (senate bridge), Anderson (house paradox), Tyson (senate paradox). The original hardcoded list had only 3 profiles (Schreiber, Dietrich, Tyson) and 5 annotation slugs.
- **Explanation:** The detection algorithms use explicit thresholds: unity < 0.95 for mavericks, rank gap > 0.5 for paradoxes, betweenness within 1 SD of midpoint for bridges. The paradox detector found Anderson (toward the center — high loyalty but low IRT extremity) and Tyson (rightward — high IRT extremity but low loyalty). Borjon and Hill emerge as bridge-builders via betweenness centrality.
- **Downstream:** Detection thresholds (0.95 unity, 0.5 rank gap) were calibrated on the 2025-2026 session. Future sessions with different partisan dynamics (e.g., closer party balance) may need threshold adjustments. Monitor whether the detectors produce sensible results when applied to historical sessions.

### Anderson Paradox — Reverse Direction

- **Phase:** Synthesis
- **Observation:** Rep. Avery Anderson (R) is detected as a house paradox with direction "toward the center" — the reverse of the Tyson paradox. Anderson has high clustering loyalty but a moderate IRT position, meaning the rank gap comes from being loyal but not ideologically extreme.
- **Explanation:** This is a valid paradox case but less narratively interesting than the Tyson-style paradox (extreme ideology but low loyalty). The "toward the center" direction indicates the gap comes from being a reliable caucus member who isn't an ideologue — the opposite of Tyson's "reliable conservative who bucks the party."
- **Downstream:** The paradox narrative template handles both directions, but the "toward the center" case reads less dramatically. Consider whether the paradox detector should prefer "extreme ideology + low loyalty" over "high loyalty + moderate ideology" when both exceed the threshold.

## Template

```
### [Legislator Name or Pattern]

- **Phase:** [EDA | PCA | IRT | Clustering | Network | Prediction | Indices | Synthesis]
- **Observation:** What was seen in the data.
- **Explanation:** Why it happened (if known).
- **Downstream:** What to do about it in future phases.
```
