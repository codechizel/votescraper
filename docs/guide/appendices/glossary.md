# Appendix A: Glossary

> *Every technical term introduced across the nine volumes, defined in plain language and cross-referenced to where it first appears.*

---

## How to Use This Glossary

Terms are listed alphabetically. Each definition includes the volume and chapter where the term is first introduced (e.g., "Vol. 4, Ch. 2"). Terms that appear in multiple volumes are defined once and cross-referenced. Italicized terms within definitions are themselves glossary entries.

---

## A

**Absent and Not Voting (ANV)** — A vote category indicating the legislator was not present and did not cast a vote. One of five vote categories in the Kansas data. Vol. 2, Ch. 2.

**Adaptive prior** — A *prior distribution* whose parameters are adjusted based on the data characteristics (e.g., group size). In hierarchical IRT, small parties get tighter priors to prevent overfitting. Vol. 4, Ch. 5.

**Adaptive tuning** — Automatic adjustment of MCMC sampling parameters (tuning steps, initialization) for chambers with *supermajority* party compositions. Vol. 4, Ch. 4.

**Adjusted Rand Index (ARI)** — A measure of agreement between two clusterings, corrected for chance. ARI = 1.0 means perfect agreement; ARI = 0.0 means agreement expected by random assignment. Vol. 6, Ch. 1.

**Affine transformation** — A "stretch and shift" operation: y = Ax + B. Used to align *ideal point* scales across sessions. Vol. 8, Ch. 3.

**ALEC (American Legislative Exchange Council)** — An organization that produces *model legislation* adopted by state legislatures. Tallgrass compares Kansas bills against the ALEC corpus. Vol. 7, Ch. 5.

**Anchor** — A legislator or constraint used to fix the sign convention in *IRT* models. Without anchors, the model can't distinguish "liberal" from "conservative." Vol. 4, Ch. 3.

**Anchor-agreement** — An *identification strategy* that selects anchors based on the two legislators with the lowest cross-party contested-vote agreement (the most partisan legislators in each party). Used in *supermajority* chambers. Vol. 4, Ch. 3.

**Anchor-pca** — An *identification strategy* that selects anchors from the extremes of *PC1*. The default for balanced chambers. Vol. 4, Ch. 3.

**AUC-ROC (Area Under the Receiver Operating Characteristic Curve)** — A measure of classification accuracy. AUC = 1.0 means perfect prediction; AUC = 0.5 means random guessing. Vol. 7, Ch. 1.

## B

**Bai-Perron structural break test** — A formal statistical test that computes 95% *confidence intervals* on *changepoint* locations in a time series. Vol. 8, Ch. 1.

**Base rate** — The simplest possible prediction (e.g., "predict the majority vote every time"). Used as a benchmark for model accuracy. Vol. 1, Ch. 1.

**Base64 embedding** — Encoding binary data (like a PNG image) as text characters so it can be placed directly inside an HTML file. Makes reports *self-contained*. Vol. 9, Ch. 1.

**Bayesian inference** — A statistical framework that updates beliefs (priors) with evidence (data) to produce updated beliefs (posteriors). The foundation of all IRT models in Tallgrass. Vol. 1, Ch. 1.

**BERTopic** — A topic modeling algorithm that combines text *embeddings*, *UMAP* dimensionality reduction, and *HDBSCAN* clustering to discover topics in a corpus. Vol. 7, Ch. 3.

**Beta-Binomial model** — A *conjugate* Bayesian model for count data. Used in *empirical Bayes* to estimate party loyalty with appropriate uncertainty. Vol. 6, Ch. 6.

**Betweenness centrality** — The fraction of shortest paths between all pairs of nodes that pass through a given node. High betweenness means a legislator is a structural bridge. Vol. 6, Ch. 3.

**Bicameral** — Having two legislative chambers (House and Senate). Vol. 1, Ch. 2.

**Biennium** — A two-year legislative cycle. Kansas numbers its bienniums by legislature (e.g., 91st = 2025-2026). Vol. 1, Ch. 2.

**Bill discrimination** — See *discrimination (β)*.

**Bill difficulty** — See *difficulty (α)*.

**Bipartite network** — A network with two types of nodes (legislators and bills) where edges connect legislators to the bills they voted on. Vol. 6, Ch. 4.

**Bridge-builder** — A legislator with high *network centrality* near the *cross-party midpoint* — structurally positioned to connect both parties. Vol. 9, Ch. 2.

**Bridge legislator** — A lawmaker who serves in multiple bienniums, linking *ideal point* scales across sessions in *dynamic IRT*. Vol. 8, Ch. 2.

**Brier score** — A measure of probabilistic prediction accuracy. Lower is better. Brier = 0 means perfect calibration; Brier = 0.25 means random guessing on binary outcomes. Vol. 7, Ch. 1.

## C

**Calibration curve** — A plot comparing predicted probabilities to observed frequencies. A well-calibrated model's curve follows the diagonal. Vol. 7, Ch. 1.

**Canonical ideal points** — The "best available" ideology scores, selected by the *quality gate* system from among 1D IRT, 2D IRT Dimension 1, and hierarchical models. Vol. 4, Ch. 7.

**Canonical routing** — The automated process that selects which IRT model variant provides *canonical ideal points* for downstream phases. Vol. 4, Ch. 7.

**Chain** — An independent sequence of MCMC samples. Multiple chains (typically 4) are run to assess *convergence*. Vol. 4, Ch. 2.

**Changepoint** — A point in time where the statistical properties of a series (mean, variance, or both) change suddenly. Vol. 8, Ch. 1.

**Coalition labeler** — An algorithm that automatically names clusters based on party composition and ideological position (e.g., "Moderate Republicans," "Bipartisan Coalition"). Vol. 9, Ch. 2.

**Cohen's d** — An effect size measuring the separation between two groups in standard deviation units. Used to assess party separation in IRT. Vol. 5, Ch. 5.

**Cohen's Kappa (κ)** — A measure of agreement between two legislators that corrects for chance agreement. κ = 1.0 means perfect agreement beyond chance; κ = 0.0 means no agreement beyond chance. Vol. 3, Ch. 2.

**Confidence interval** — A range of values within which the true parameter value lies with a specified probability (typically 95%). In Bayesian statistics, see *credible interval*. Vol. 5, Ch. 3.

**Conjugate prior** — A *prior distribution* that, when combined with a particular *likelihood*, produces a posterior of the same mathematical family. Enables closed-form solutions. Vol. 6, Ch. 6.

**Contested vote** — A roll call where at least 2.5% of voters opposed the majority. Unanimous and near-unanimous votes are filtered out. Vol. 3, Ch. 1.

**Convergence** — The state where MCMC chains have explored the posterior distribution adequately. Assessed by *R-hat*, *ESS*, and *divergent transitions*. Vol. 4, Ch. 2.

**Conversion effect** — The component of ideological change between sessions attributable to returning legislators changing their voting behavior. Vol. 8, Ch. 4.

**Credible interval** — The Bayesian equivalent of a *confidence interval*. A 95% credible interval contains the true parameter value with 95% probability, given the data and prior. Vol. 1, Ch. 1.

**CROPS (Changepoints for a Range of Penalties)** — An automated method that scans penalty values to find the natural number of *changepoints* in a time series. Vol. 8, Ch. 1.

**Cross-party midpoint** — The average of the median *ideal points* of the two parties. Used to identify *bridge-builder* candidates. Vol. 9, Ch. 2.

**Cross-session alignment** — Putting independently estimated *ideal point* scales on a common basis using an *affine transformation*. Vol. 8, Ch. 3.

**c-TF-IDF (class-based TF-IDF)** — A variant of *TF-IDF* that measures word importance within a topic cluster rather than a single document. Used in *BERTopic*. Vol. 7, Ch. 3.

**Cumulative distribution function (CDF)** — A function showing the probability that a variable takes a value less than or equal to a given point. Used in the *KS test*. Vol. 8, Ch. 4.

## D

**Dashboard** — A session-level HTML page with sidebar navigation that embeds all phase reports in an iframe. Vol. 9, Ch. 1.

**Dashboard scatter** — A scatter plot with *ideal points* on the x-axis and *party unity* on the y-axis, sized by *weighted maverick rate*. The synthesis report's signature visualization. Vol. 9, Ch. 2.

**Defection vote** — A vote where a legislator disagreed with their party's majority position. Vol. 9, Ch. 3.

**Dendrogram** — A tree diagram showing the hierarchical nesting structure of clusters. Vol. 6, Ch. 1.

**Desposato correction** — A Monte Carlo adjustment (10,000 simulations) that removes the small-group inflation bias from the *Rice Index*. Vol. 8, Ch. 1.

**Difficulty (α)** — An IRT parameter measuring how "hard" a bill is to vote Yea on. High difficulty means even legislators inclined to vote Yea might vote Nay. Vol. 4, Ch. 1.

**Discrimination (β)** — An IRT parameter measuring how sharply a bill separates legislators along the ideological dimension. High discrimination means the bill cleanly divides liberals from conservatives. Vol. 4, Ch. 1.

**Disparity filter** — A method for extracting the statistically significant edges (backbone) from a weighted network. Vol. 6, Ch. 3.

**Divergent transitions** — MCMC samples where the numerical integrator's trajectory diverged, indicating the sampler struggled in that region of parameter space. Too many suggest model misspecification. Vol. 4, Ch. 2.

**Dynamic ideal point** — An *ideal point* that varies over time, estimated via a *state-space model*. Vol. 8, Ch. 2.

**DW-NOMINATE** — Dynamic Weighted NOMINAL Three-step Estimation. The standard method in political science for estimating legislator ideology from roll call votes. Vol. 5, Ch. 4.

## E

**Effective sample size (ESS)** — The number of independent samples equivalent to the correlated MCMC chain. Higher is better; ESS < 400 suggests insufficient sampling. Vol. 4, Ch. 2.

**Eigenvalue** — A number indicating how much variance a *principal component* captures. Large eigenvalues correspond to important dimensions. Vol. 3, Ch. 3.

**Eigenvector centrality** — A network measure where a node's importance depends on the importance of its neighbors. Similar to *PageRank*. Vol. 6, Ch. 3.

**Elbow** — The point on a curve (penalty vs. number of changepoints, or clusters vs. inertia) where the marginal gain drops sharply. Suggests the natural breakpoint. Vol. 8, Ch. 1.

**Embedding** — A vector representation of text that captures semantic meaning. Similar texts have similar embeddings. Vol. 7, Ch. 3.

**Empirical Bayes** — A method that estimates *prior distribution* parameters from the data itself, then uses those priors for Bayesian inference. A practical shortcut to full hierarchical modeling. Vol. 6, Ch. 6.

**Entropy** — In *LCA*, a measure of classification certainty. High entropy (near 1.0) means legislators are clearly assigned to one class; low entropy means assignments are ambiguous. Vol. 6, Ch. 2.

**Evolution variance (τ, tau)** — The parameter controlling how fast ideology can change between bienniums in *dynamic IRT*. Small tau means stability; large tau means volatility. Vol. 8, Ch. 2.

## F

**Feature importance** — A ranking of input variables by their contribution to a prediction model's accuracy. In Tallgrass, measured by *SHAP values*. Vol. 7, Ch. 1.

**Freshmen cohort** — Legislators serving in their first biennium. Compared to returning members in the *conversion-replacement decomposition*. Vol. 8, Ch. 4.

**Funnel geometry** — A problematic posterior shape in hierarchical models where the group-level variance and the individual offsets are tightly coupled. Solved by *non-centered parameterization*. Vol. 4, Ch. 5.

## G

**Geometric Mean Probability (GMP)** — The geometric mean of the model's predicted probability of the correct outcome across all votes. A summary of model fit. Vol. 5, Ch. 2.

**Graceful degradation** — The design principle that reports skip sections when data is unavailable rather than crashing. Vol. 9, Ch. 2.

## H

**HDBSCAN** — Hierarchical Density-Based Spatial Clustering of Applications with Noise. A clustering algorithm that identifies clusters of varying density and labels outliers as noise. Vol. 6, Ch. 1.

**Hierarchical model** — A model with multiple levels (e.g., legislators nested within parties). Enables *partial pooling* — sharing information across groups. Vol. 4, Ch. 5.

**Highest Density Interval (HDI)** — The narrowest *credible interval* containing the specified probability mass (typically 95%). Vol. 4, Ch. 2.

**Horn's parallel analysis** — A method for determining how many *principal components* to retain by comparing eigenvalues to those from random data. Vol. 3, Ch. 3.

**Horseshoe effect** — A PCA artifact in *supermajority* chambers where the first principal component captures intra-party factionalism rather than the party divide. Affects 7 of 14 Senate sessions. Vol. 3, Ch. 5.

## I

**Ideal point (ξ, xi)** — A number representing a legislator's position on the liberal-conservative spectrum, estimated from their voting pattern via *IRT*. Vol. 1, Ch. 1.

**Identification problem** — The fact that *IRT* models can't determine which direction is "conservative" without external constraints (*anchors*). Vol. 4, Ch. 3.

**Inertia** — In *k-means clustering*, the sum of squared distances from each point to its cluster center. Lower is better. Vol. 6, Ch. 1.

**Informative prior** — A *prior distribution* that encodes specific knowledge (e.g., the static IRT sign convention). Contrasts with vague priors. Vol. 8, Ch. 2.

**Intraclass Correlation Coefficient (ICC)** — A measure of how consistently a metric ranks the same legislators across two measurement occasions (e.g., two sessions). ICC > 0.75 is "good." Vol. 8, Ch. 3.

**Item Characteristic Curve (ICC)** — An S-shaped curve showing the probability of voting Yea as a function of ideology, for a specific bill. Vol. 4, Ch. 2.

**Item Response Theory (IRT)** — A family of statistical models that estimate latent traits (like ideology) from binary responses (like votes). The core method in Tallgrass. Vol. 4, Ch. 1.

## K

**K-means clustering** — An algorithm that partitions data into k groups by minimizing the distance from each point to its cluster center. Vol. 6, Ch. 1.

**KS test (Kolmogorov-Smirnov)** — A non-parametric test for whether two distributions differ significantly. Used to compare departing and incoming legislator cohorts. Vol. 8, Ch. 4.

## L

**Latent Class Analysis (LCA)** — A model-based clustering method that assumes legislators belong to unobserved classes, each with its own probability of voting Yea on each bill. Vol. 6, Ch. 2.

**Latent variable** — A quantity that cannot be directly observed but is inferred from data. Ideology is a latent variable inferred from votes. Vol. 4, Ch. 1.

**Leiden algorithm** — A community detection algorithm for networks. An improvement over the Louvain algorithm, guaranteeing well-connected communities. Vol. 6, Ch. 3.

**Loadings** — In *PCA*, the weights that define how each original variable (vote) contributes to a *principal component*. Vol. 3, Ch. 3.

**Logistic function (sigmoid)** — An S-shaped function mapping any number to the range [0, 1]. Used in IRT to convert ideology into vote probability. Vol. 4, Ch. 2.

## M

**Margin closeness** — How close a party's internal vote was to a 50-50 split. Used to rank *defection votes* by political significance. Vol. 9, Ch. 3.

**Maverick** — A legislator with the lowest *party unity score* in their caucus. Detected algorithmically in the synthesis phase. Vol. 9, Ch. 2.

**Maverick rate** — The fraction of close party-line votes where a legislator voted against their party. "Close" means the margin was tight enough that individual votes mattered. Vol. 6, Ch. 5.

**MCMC (Markov Chain Monte Carlo)** — A family of algorithms that generate samples from a posterior distribution when direct computation is impossible. Vol. 4, Ch. 2.

**Metric paradox** — A legislator whose *IRT ideology rank* and *clustering loyalty rank* disagree dramatically, indicating their defection pattern is issue-specific rather than ideological. Vol. 9, Ch. 2.

**Model legislation** — Template bills produced by organizations like *ALEC* for adoption by state legislatures. Vol. 7, Ch. 5.

**Modularity** — A measure of how well a network divides into communities. High modularity means dense connections within communities and sparse connections between them. Vol. 6, Ch. 3.

**Multiple Correspondence Analysis (MCA)** — A dimensionality reduction technique for categorical data. The categorical analogue of *PCA*. Vol. 3, Ch. 4.

## N

**Non-centered parameterization** — A reparameterization trick where individual offsets are drawn from a standard normal distribution and scaled by the group-level variance. Eliminates *funnel geometry*. Vol. 4, Ch. 5.

**NUTS (No-U-Turn Sampler)** — An advanced MCMC algorithm that automatically tunes its step size and trajectory length. The sampler used in all Tallgrass Bayesian models (via *nutpie*). Vol. 4, Ch. 2.

**nutpie** — A Rust implementation of the *NUTS* sampler. Faster than the Python-native PyMC sampler. Vol. 4, Ch. 2.

## O

**Optimal Classification (OC)** — A nonparametric method that estimates ideal points by minimizing classification errors without assuming a particular probability model. Vol. 5, Ch. 4.

## P

**PageRank** — A network centrality measure originally developed for web search. A node's importance depends on the importance of the nodes that link to it. Vol. 6, Ch. 3.

**Partial pooling** — The *hierarchical model* compromise between complete pooling (everyone shares parameters) and no pooling (everyone gets their own). Individual estimates are "shrunk" toward the group mean. Vol. 4, Ch. 5.

**Party unity score** — The fraction of party-line votes where a legislator voted with their party majority. Also called CQ Unity. Vol. 6, Ch. 5.

**Party vote** — A roll call where a majority of one party opposes a majority of the other. The CQ standard definition. Vol. 6, Ch. 5.

**PELT (Pruned Exact Linear Time)** — A *changepoint* detection algorithm that finds moments where a time series' statistical properties shift abruptly. Vol. 8, Ch. 1.

**Pipeline** — The sequence of 28 analysis phases that transforms raw vote data into reports. Vol. 1, Ch. 4.

**PLT (Positive Lower Triangular) identification** — A constraint on the discrimination matrix in *2D IRT* that resolves *rotational invariance* by requiring the matrix to be lower triangular with positive diagonal. Vol. 4, Ch. 4.

**Polarization gap** — The distance between the two parties' mean *ideal points*. A widening gap means increasing polarization. Vol. 8, Ch. 1.

**Population Stability Index (PSI)** — A metric measuring whether the distribution of a variable has shifted meaningfully between two time periods. PSI < 0.10 is stable; > 0.25 indicates significant drift. Vol. 8, Ch. 3.

**Posterior distribution** — The updated probability distribution for a parameter after combining the *prior* with the data. The "answer" in Bayesian inference. Vol. 4, Ch. 2.

**Posterior predictive check (PPC)** — A validation method that generates simulated data from the model's posterior and compares it to the observed data. If the model is good, the simulated data should look like the real data. Vol. 5, Ch. 2.

**Primer** — An auto-generated methodology description written to each phase's output directory by *RunContext*. Vol. 9, Ch. 1.

**Principal component (PC)** — A new axis created by *PCA* that captures the maximum possible variance in the data. PC1 captures the most variance, PC2 the next most, and so on. Vol. 3, Ch. 3.

**Principal Component Analysis (PCA)** — A method that finds the directions of greatest variation in high-dimensional data and projects it onto a smaller number of dimensions. Vol. 3, Ch. 3.

**Prior distribution** — A probability distribution representing beliefs about a parameter before seeing data. In *Bayesian inference*, the prior is updated by the likelihood to produce the *posterior*. Vol. 4, Ch. 2.

**Profile card** — A horizontal bar chart showing six normalized metrics for a notable legislator, comparing their values to party averages. Vol. 9, Ch. 2.

## Q

**Quality gate** — An automated check that assigns a trust level (Tier 1, 2, or 3) to IRT results based on convergence diagnostics and party separation. Vol. 5, Ch. 5.

## R

**R-hat (R̂)** — A convergence diagnostic comparing variance within and between MCMC chains. Values near 1.0 indicate convergence; values above 1.05 (or 1.01 for strict thresholds) suggest the chains haven't mixed. Vol. 4, Ch. 2.

**Random walk** — A process where each step is the previous position plus a random increment. The simplest model of drift, used in *dynamic IRT*. Vol. 8, Ch. 2.

**Reflection invariance** — The property that an *IRT* model produces identical likelihood for (ξ, β) and (-ξ, -β). The source of the *identification problem*. Vol. 4, Ch. 3.

**Replacement effect** — The component of ideological change between sessions attributable to new legislators replacing departing ones with different ideological positions. Vol. 8, Ch. 4.

**ReportBuilder** — The class that assembles *section types* into a complete HTML document with a table of contents, metadata, and consistent styling. Vol. 9, Ch. 1.

**Residual trimming** — Removing the legislators with the largest prediction errors before refitting a regression. Used in *robust fitting* for cross-session alignment. Vol. 8, Ch. 3.

**Rice Index** — A measure of party cohesion: |Yea − Nay| / (Yea + Nay). Values near 1.0 mean the party was nearly unanimous; values near 0.0 mean it split evenly. Vol. 6, Ch. 5.

**Ridgeline plot** — A visualization stacking density curves vertically (one per time period), showing how the distribution of *ideal points* changes over time. Vol. 8, Ch. 2.

**Robust fitting** — Regression that trims extreme outliers before fitting, preventing genuine movers from distorting the scale correction in *cross-session alignment*. Vol. 8, Ch. 3.

**Rolling-window PCA** — *PCA* applied to overlapping subsets of votes within a session, producing a time series of ideological positions. Vol. 8, Ch. 1.

**Run ID** — A unique identifier for a pipeline execution: `{legislature}-{YYMMDD}.{N}` (e.g., `91-260318.1`). Vol. 9, Ch. 1.

**RunContext** — A context manager that creates output directories, captures console logs, records metadata, and manages report lifecycle for each analysis phase. Vol. 9, Ch. 1.

## S

**Sankey diagram** — A flow visualization showing how legislators move between voting blocs across sessions. Band width represents the number of legislators making each transition. Vol. 8, Ch. 4.

**Scorecard** — A normalized six-metric profile (0-1 scale) summarizing a legislator's ideology, loyalty, maverick rate, network influence, and prediction accuracy. Vol. 9, Ch. 3.

**Scree plot** — A plot of *eigenvalues* in decreasing order, used to determine how many *principal components* to retain. Vol. 3, Ch. 3.

**Scrollytelling** — A progressive narrative layout where text scrolls while visualizations remain fixed, revealing the story step by step. Vol. 9, Ch. 1.

**Section type** — One of eight standardized content blocks (table, figure, text, interactive table, interactive, key findings, download, scrollytelling) that can appear in a report. Vol. 9, Ch. 1.

**Self-contained HTML** — An HTML file that embeds all its dependencies (images, CSS, JavaScript) so it works without any external files. Vol. 9, Ch. 1.

**SHAP (SHapley Additive exPlanations)** — A method from game theory that attributes a model's prediction to each input feature. Shows which features matter most and how they push the prediction. Vol. 7, Ch. 1.

**Shor-McCarty scores** — External ideology estimates from Harvard Dataverse, based on legislator surveys. Used for *external validation* of Tallgrass IRT scores. Vol. 5, Ch. 3.

**Shrinkage** — The phenomenon in *hierarchical models* and *empirical Bayes* where individual estimates are pulled toward the group mean. More data = less shrinkage. Vol. 4, Ch. 5.

**Sign convention** — The assignment of positive values to conservative legislators and negative to liberal (or vice versa). Must be fixed externally because IRT has *reflection invariance*. Vol. 4, Ch. 3.

**Silhouette score** — A measure of clustering quality: how similar each point is to its own cluster versus other clusters. Ranges from -1 (wrong cluster) to +1 (perfect fit). Vol. 6, Ch. 1.

**State-space model** — A model where a hidden state (ideology) evolves over time and is observed indirectly through data (votes). The foundation of *dynamic IRT*. Vol. 8, Ch. 2.

**Supermajority** — A party holding substantially more than 50% of seats (typically > 60%). Creates statistical challenges including the *horseshoe effect*. Vol. 1, Ch. 2.

**Surprising vote** — A vote where the prediction model was confident in the wrong answer. Measured by *confidence error*. Vol. 9, Ch. 3.

**Synthesis** — Phase 24 — the automated aggregation of ten upstream analyses into a single narrative report designed for nontechnical audiences. Vol. 9, Ch. 2.

## T

**TBIP (Text-Based Ideal Points)** — A model that estimates ideal points from the text of bills legislators support, rather than from their votes. Vol. 7, Ch. 4.

**TF-IDF (Term Frequency-Inverse Document Frequency)** — A text weighting scheme that measures how important a word is to a document relative to a corpus. Common words get low weight; distinctive words get high weight. Vol. 7, Ch. 2.

**Tier (quality gate)** — One of three trust levels for IRT results: Tier 1 (fully converged), Tier 2 (point estimates credible), Tier 3 (fallback to simpler model). Vol. 5, Ch. 5.

**Turnover decomposition** — Separating a party's ideological shift into *conversion effects* and *replacement effects*. Vol. 8, Ch. 4.

## U

**UMAP (Uniform Manifold Approximation and Projection)** — A nonlinear dimensionality reduction technique that preserves both local and global structure. Produces 2D "maps" of legislative voting patterns. Vol. 3, Ch. 4.

## V

**Vote matrix** — A legislators × votes table where each cell records how a legislator voted (1 = Yea, 0 = Nay, null = absent). The fundamental input to most analyses. Vol. 3, Ch. 1.

**Voting neighbor** — A legislator whose vote-by-vote agreement with a target is highest (closest) or lowest (most different). Computed via pairwise simple agreement. Vol. 9, Ch. 3.

## W

**W-NOMINATE** — Weighted NOMINAl Three-step Estimation. The field standard for estimating *ideal points* from roll call votes, developed by Poole and Rosenthal. Vol. 5, Ch. 4.

**WCAG 2.1 AA** — Web Content Accessibility Guidelines level AA. The accessibility standard Tallgrass reports target, including alt text on figures and sufficient color contrast. Vol. 9, Ch. 1.

**Weighted maverick rate** — *Maverick rate* weighted by vote closeness. Defections on tight votes (51-49 party split) count more than defections on lopsided ones (90-10). Vol. 6, Ch. 5.

## X

**XGBoost** — Extreme Gradient Boosting. The machine learning algorithm used for vote prediction in Phase 15. Vol. 7, Ch. 1.

---

*Back to: [Appendices](./)*
