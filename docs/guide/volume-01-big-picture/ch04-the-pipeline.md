# Chapter 4: The Pipeline: From Raw Votes to Insight

> *A guided tour of the 29-phase analysis pipeline — the assembly line that transforms a quarter-million web pages into statistical portraits of legislative ideology.*

---

## The Assembly Line

Imagine an automobile factory. Raw steel arrives at one end. At each station along the line, workers perform a specific operation: stamp the body panels, weld the frame, install the engine, paint the exterior, test the brakes. No station can work until the previous one finishes. And at the end, a finished car rolls off the line.

The Tallgrass pipeline works the same way. Raw vote data enters at one end. At each phase, a specific statistical operation transforms the data: filter, compress, estimate, validate, cluster, predict, synthesize. Each phase consumes the output of earlier phases and produces input for later ones. At the end, a comprehensive HTML report rolls off the line — complete with ideology scores, coalition maps, interactive charts, and data-driven narrative text.

The full pipeline has 29 phases, plus a data acquisition step that runs before the analysis begins. A single command kicks off the entire sequence:

```
just pipeline 2025-26
```

About 45 minutes later, you have a complete statistical analysis of one biennium of the Kansas Legislature.

This chapter walks through the pipeline at a high level. We won't dive into the mathematics — that's what Volumes 3 through 8 are for. The goal here is to give you a mental map of the whole journey, so that when we examine each phase in detail, you'll know where it fits.

## Phase 0: Data Acquisition

Before the analysis pipeline runs, the data needs to exist. This is the scraper's job.

The Tallgrass scraper visits kslegislature.gov, finds every bill that had a recorded vote, downloads every vote page, and extracts who voted what. It's a four-step process:

1. **Discover** — Find all bills in the biennium
2. **Filter** — Identify which bills had recorded votes (using the legislature's API)
3. **Parse** — Extract individual legislator votes from each vote page
4. **Enrich** — Look up each legislator's party, district, and full name

The output is a set of CSV files: one for individual votes (~95,000 rows), one for roll call summaries (~1,100 rows), one for the legislator roster (~170 rows), one for bill lifecycle actions, and one for bill text. These files are the raw material for everything that follows.

Volume 2 covers data acquisition in detail.

## The Analysis Pipeline

The 28 analysis phases organize into natural groups. Here's the map, with each group's purpose and a plain-language description of what it contributes.

### Group 1: Preparation and Exploration (Phases 1-4)

*"What does this data look like?"*

Before any fancy statistics, we need to understand the raw data — its shape, its quirks, its base rates. These four phases clean, filter, and visualize the vote matrix.

**Phase 1: Exploratory Data Analysis (EDA)**

The foundation. This phase builds the vote matrix (legislators as rows, votes as columns, Yea = 1, Nay = 0, absent = blank), applies the key filters (remove near-unanimous votes, remove legislators with too few votes), and computes basic descriptive statistics: participation rates, vote type distributions, party-line voting frequencies.

It also computes the **agreement matrix** — a measure of how similarly every pair of legislators voted, corrected for the 82% Yea base rate using Cohen's Kappa. This matrix is the starting point for network analysis and clustering later in the pipeline.

Think of this phase as a thorough physical examination before surgery. You need to know the patient's vital signs before you operate.

*Output: Filtered vote matrices (one per chamber), agreement matrices, descriptive statistics report.*

**Phase 2: Principal Component Analysis (PCA)**

PCA answers the question: *what are the most important dimensions of variation in this data?*

Imagine 600 vote columns as 600 dimensions. No human can visualize 600 dimensions. PCA finds the direction along which legislators differ the most (the first principal component, or PC1), then the next most important direction perpendicular to it (PC2), and so on. In most legislative chambers, PC1 captures the party divide — it's the liberal-to-conservative axis. PC2 often captures within-party variation.

Think of it as a photographer choosing the best angle for a group photo. The angle that shows the most difference between people is PC1.

*Output: PC scores for each legislator (their position on the main axes), eigenvalues (how much each axis explains), loadings (which votes contribute most to each axis).*

**Phase 3: Multiple Correspondence Analysis (MCA)**

A cousin of PCA designed for categorical data. Where PCA treats votes as numbers (1 and 0), MCA treats them as categories (Yea, Nay, Absent). This lets it capture patterns in absences that PCA ignores. In practice, MCA usually agrees closely with PCA — but when it doesn't, the disagreement is informative.

*Output: MCA scores per legislator, comparison to PCA results.*

**Phase 4: UMAP Visualization**

UMAP (Uniform Manifold Approximation and Projection) produces a 2D map where legislators who vote similarly are placed near each other and legislators who vote differently are placed far apart. Unlike PCA, UMAP can capture nonlinear structure — it can "unfold" curved shapes in the data that PCA would project flat.

Think of PCA as a shadow cast on a wall (a flat projection) and UMAP as uncrumpling a piece of paper (preserving neighborhoods, not straight lines).

*Output: 2D scatter plot of legislators, colored by party.*

### Group 2: Ideology Estimation (Phases 5-7b)

*"Where does each legislator fall on the political spectrum?"*

This is the mathematical heart of the project. Four phases estimate "ideal points" — numerical positions on an ideology scale — using increasingly sophisticated Bayesian models.

**Phase 5: 1D Item Response Theory (IRT)**

The workhorse. IRT borrows a framework from psychometrics — the same mathematics used to score the SAT and GRE. The analogy is direct: bills are "test questions" and legislators are "test-takers." Some questions are easy (near-unanimous bills) and some are hard (contested bills). Some questions sharply separate strong students from weak ones (high discrimination) and some don't (low discrimination). IRT figures out every legislator's "ability" (ideology) and every bill's "difficulty" and "discrimination" simultaneously.

The key equation, in plain English: *the probability that a legislator votes Yea on a bill depends on how close their ideology is to what the bill requires.* Moderate bills get Yea votes from almost everyone. Extreme bills only get Yea votes from legislators on the matching end of the spectrum.

The model is estimated using Bayesian inference — instead of a single "best guess," it produces a full distribution of plausible values for each parameter. This means every ideal point comes with an uncertainty interval: "this legislator is probably between -0.8 and -0.3, with -0.55 as the best estimate."

*Output: Ideal point estimate (with uncertainty) for every legislator, discrimination and difficulty parameters for every bill, convergence diagnostics.*

**Phase 6: 2D IRT**

Sometimes one dimension isn't enough. In chambers with large supermajorities, the most important axis of conflict isn't "left vs. right" but "establishment vs. maverick" — moderate Republicans and conservative rebels can look similar in 1D because both dissent from the party line, but for opposite reasons.

2D IRT adds a second dimension. Now each legislator has two scores (ideology and establishment loyalty) and each bill has two discrimination parameters. This is more complex to estimate and harder to identify (the model needs extra constraints to pin down the rotation), but it resolves the ambiguity that plagues 1D models in supermajority chambers.

*Output: Two-dimensional ideal points per legislator, quality gate tier assessment (Tier 1 = converged, Tier 2 = credible point estimates, Tier 3 = fall back to 1D).*

**Phase 7: Hierarchical IRT**

Flat IRT treats every legislator as independent. Hierarchical IRT recognizes that legislators come from parties, and party membership tells you something about where they'll land on the ideological spectrum. The model learns party-level averages and then adjusts each individual toward or away from their party mean based on their personal voting record.

The effect is "partial pooling" — a technique where individual estimates are pulled toward their group average, with the strength of the pull depending on how much data each individual has. A legislator with few votes gets pulled strongly toward their party average (because we don't have much evidence to go on), while a veteran with hundreds of votes stands mostly on their own record. This is particularly valuable for new members and for small caucuses (like the 9-member Senate Democratic minority).

*Output: Hierarchical ideal points (with party-level "shrinkage" — the statistical term for this pull toward the group average), intraclass correlation coefficient (how much party explains ideology), within-party variance estimates.*

**Phase 7b: Hierarchical 2D IRT**

The most sophisticated model: 2D ideal points with hierarchical party structure. This phase combines the horseshoe-correction benefits of 2D IRT with the partial-pooling benefits of hierarchical IRT. It uses informative priors from Phases 6 and 7 to help the model converge.

When this model converges, its first-dimension scores are the preferred "canonical" ideology measure for horseshoe-affected chambers.

*Output: Hierarchical 2D ideal points, canonical routing decision (which model's scores should downstream phases use).*

### Group 3: Validation (Phases 8, 16-18)

*"How do we know the model is right?"*

A model that can't be checked is a model that can't be trusted. Four phases test the IRT results from different angles.

**Phase 8: Posterior Predictive Checks + Leave-One-Out Cross-Validation**

The first check: can the model predict its own data? If we use the estimated ideal points and bill parameters to "replay" every vote, how often does the model get it right? (The answer: about 90-95% of the time, which is well above the 82% base rate.)

The second check: model comparison. How does 1D IRT compare to 2D IRT compare to hierarchical IRT? LOO-CV (Leave-One-Out Cross-Validation — hiding one vote at a time, predicting it from the rest, and measuring how often the model gets it right) gives each model a score based on how well it predicts held-out observations. This is how we decide whether the extra complexity of 2D or hierarchical models is justified by the data.

**Phase 16: W-NOMINATE**

W-NOMINATE is the most widely used ideology estimation method in political science, developed by Keith Poole and Howard Rosenthal in the 1980s. It uses maximum likelihood (not Bayesian inference) and a spatial voting model. Tallgrass runs W-NOMINATE via the R programming language and compares its results to our Bayesian IRT. High correlation (which we consistently observe) validates that both methods are measuring the same underlying construct.

**Phase 17: External Validation (Shor-McCarty)**

Boris Shor and Nolan McCarty produced the most cited external benchmark of state legislator ideology, using bridge legislators and survey data to place every state legislator in America on a common scale. Tallgrass matches Kansas legislators by name and computes the correlation. Typical results: r = 0.98 for the House, r = 0.93 for the Senate. These are among the highest external correlations reported in the state legislature literature.

**Phase 18: DIME External Validation**

DIME (Database on Ideology, Money in Politics, and Elections) uses campaign finance data — who donates to whom — to estimate ideology. This provides a completely independent check: if our vote-based scores and their donation-based scores agree, we can be more confident that both are measuring something real.

### Group 4: Pattern Detection (Phases 9-14)

*"Are there factions? Who are the bridge-builders?"*

Six phases look for structure in the voting patterns using clustering, network analysis, and classical political science metrics.

**Phase 9: Clustering**

Five different clustering algorithms search for discrete voting blocs. The consistent finding: the optimal number of clusters is 2 (the two parties), and within-party variation is continuous, not factional. There are moderate Republicans and conservative Republicans, but they blend into each other rather than forming distinct camps.

**Phase 10: Latent Class Analysis (LCA)**

A model-based alternative to clustering. LCA assumes there are hidden "classes" of legislators, each with their own probability of voting Yea on each bill. It tests whether K=2, K=3, K=4, etc. classes best fit the data. Like Phase 9, it generally confirms that the two-party split is the dominant structure.

**Phase 11: Network Analysis**

Build a network where legislators are nodes and edges connect pairs who vote together frequently (measured by Cohen's Kappa). Then use community detection algorithms to find clusters in the network. This reveals the "social structure" of voting — who is central (high betweenness centrality = a bridge between groups), who is peripheral, and who connects the two parties.

**Phase 12: Bipartite Networks**

A different kind of network: legislators on one side, bills on the other, with edges connecting legislators to bills they voted Yea on. This reveals bill-centric structure — which bills attract bipartisan support and which are strictly partisan.

**Phase 13: Classical Indices**

Standard political science metrics: the Rice Index (party cohesion), CQ Party Unity (individual loyalty), Effective Number of Parties, and the Maverick Score. These are simpler than IRT but widely used and understood. They provide a familiar vocabulary for describing legislative behavior.

**Phase 14: Beta-Binomial Party Loyalty**

An empirical Bayes approach to party loyalty estimation. The idea: a legislator with 10 party votes who defects once (90% loyalty) might just have gotten unlucky. Empirical Bayes "shrinks" this estimate toward the party average, producing more stable loyalty scores — especially for legislators with few votes. It's the same principle used in baseball to adjust batting averages for players with few at-bats.

### Group 5: Prediction and Text (Phases 15, 20-23)

*"Can we predict votes? What do the bills say?"*

Five phases use machine learning and natural language processing.

**Phase 15: Vote Prediction**

Given a legislator's ideal point, the bill's discrimination, party affiliation, and a handful of other features — can we predict how they'll vote? XGBoost (a gradient-boosted decision tree model) achieves AUC = 0.98, with the ideal-point-times-discrimination interaction as the dominant feature. A separate model predicts bill passage outcomes using bill characteristics and NLP features extracted from bill titles.

**Phase 20: Bill Text Analysis**

BERTopic (a modern topic modeling approach) identifies policy topics from bill text: education, taxes, criminal justice, healthcare, etc. The Comparative Agendas Project (CAP) taxonomy provides an alternative 20-category classification. Bill similarity analysis finds pairs of bills with nearly identical language.

**Phase 21: Text-Based Ideal Points**

Can you estimate ideology from the *content* of bills a legislator supports? This phase weights bill text embeddings by legislator votes and extracts the first principal component. Correlation with IRT ideal points validates that text-based and vote-based ideology measures agree.

**Phase 22: Issue-Specific IRT**

Run separate IRT models for each policy topic (education bills, tax bills, healthcare bills, etc.). The result: issue-specific ideal points that answer questions like "Is this legislator conservative on taxes but moderate on education?" Cross-topic correlations reveal which policy areas align ideologically and which cut across party lines.

**Phase 23: Model Legislation Detection**

Compare Kansas bills to a corpus of 1,061 ALEC (American Legislative Exchange Council) model policies and to bills from neighboring states (Missouri, Oklahoma, Nebraska, Colorado). The method: cosine similarity on text embeddings, confirmed with n-gram overlap analysis. Matches above 85% similarity suggest direct textual borrowing.

### Group 6: Temporal Analysis (Phases 19, 26-28)

*"How has the legislature changed over time?"*

Three phases track dynamics across sessions and years.

**Phase 19: Time Series Analysis**

Within a session, track party cohesion over time using rolling averages. PELT changepoint detection identifies moments where cohesion abruptly shifts — a leadership challenge, a galvanizing event, a session-ending sprint. Between sessions, compare polarization trends across the 15-year dataset.

**Phase 26: Cross-Session Validation**

Put different sessions on the same scale. Since each biennium's IRT is estimated independently, the raw scores aren't directly comparable (like different thermometers that haven't been calibrated). This phase uses "bridge legislators" — individuals who served in multiple sessions — to align the scales via an affine transformation (stretch + shift). It also decomposes ideology shifts into "conversion" (existing members changing) and "replacement" (departing members replaced by different ones).

**Phase 27: Dynamic IRT**

The most ambitious temporal model: a state-space IRT that tracks each legislator's ideal point across all bienniums simultaneously. Each legislator's ideology at time *t* is modeled as their ideology at time *t-1* plus a small random step. The model estimates how fast ideology can change (separately for each party) and produces smooth trajectories for every legislator who served across multiple sessions.

**Phase 28: Common Space Ideal Points**

Places every legislator from every biennium on a single ideological scale so that legislators who never served together can be directly compared. Uses pairwise chain affine alignment (GLS 1999, Battauz 2023): each pair of adjacent bienniums is linked via bridge legislators who served in both, then the pairwise transforms are composed into a chain reaching the reference session (91st). Bootstrap resampling provides confidence intervals. Quality gates check party separation and sign consistency. House and Senate are then linked via 54 chamber-switcher bridges into a unified cross-chamber scale. Career scores (one number per legislator) are computed via DerSimonian-Laird random-effects meta-analysis, pooling per-session scores with uncertainty. Enables questions like "Was Tim Huelskamp in 2001 more conservative than any current legislator?" and "Has the Kansas Senate polarized over the last 25 years?"

### Group 7: Synthesis and Reporting (Phases 24-25)

*"What does it all mean?"*

Two phases turn 27 phases of statistics into readable documents.

**Phase 24: Synthesis**

The grand narrative. This phase pulls together results from every upstream phase and generates a comprehensive HTML report for the session. It algorithmically identifies "notable" legislators — the biggest maverick, the most effective bridge-builder, the legislator with the largest gap between their ideology rank and their loyalty rank — and weaves them into a data-driven narrative.

The synthesis is not a template with numbers plugged in. It adapts to the data. If a horseshoe effect is present, the language changes. If no legislator qualifies as a maverick (party unity is too high), that section is skipped. The report varies between 29 and 32 sections depending on what the data supports.

**Phase 25: Legislator Profiles**

Individual deep dives. For any legislator of interest, this phase generates a personal report: their ideology score in context (what percentile of their party?), their party loyalty rate, their most surprising votes (where the model predicted one outcome and they voted the other way), their top 5 most-similar and most-different colleagues, and their voting pattern on high-discrimination versus routine bills.

## The Quality Gate System

Running through the entire pipeline is a quality control system that automatically flags unreliable results. This is critical because statistical models don't always work. Sometimes the Bayesian sampler fails to converge. Sometimes the data is too sparse. Sometimes the model's assumptions are violated.

The quality gate system has three tiers:

| Tier | Meaning | Action |
|------|---------|--------|
| **Tier 1** | Fully converged — the sampler explored the full posterior distribution | Use results with full confidence |
| **Tier 2** | Partially converged — point estimates are credible but uncertainty intervals are too wide | Use the rankings, flag the intervals |
| **Tier 3** | Failed — the model did not converge reliably | Fall back to a simpler model |

These tiers propagate through the pipeline. If Phase 6 (2D IRT) achieves Tier 1 for a chamber, downstream phases use its first-dimension scores as the canonical ideology measure. If it falls to Tier 3, they fall back to Phase 5's 1D scores. The synthesis report adjusts its language accordingly.

The quality gate system is conservative by design. It would rather give you a simple, reliable answer than a complex, unreliable one.

## The Canonical Routing Decision

With up to four different ideal point estimates per legislator (1D flat, 2D flat, 1D hierarchical, 2D hierarchical), the pipeline needs to decide which one to use for downstream phases. This decision is called **canonical routing**, and it follows a preference order:

1. **Hierarchical 2D, Dimension 1** — the most informative model, if it converged
2. **Flat 2D, Dimension 1** — separates ideology from establishment loyalty
3. **1D IRT** — the simplest and most reliable model

The routing decision is made per chamber, per session. A session might use 2D for the Senate (where horseshoe effects are common) and 1D for the House (where they're not). The decision is recorded in a manifest file and propagated to every downstream phase automatically.

## What Comes Out

At the end of the pipeline, each biennium has a results directory containing:

- **29 HTML reports** — one per phase, with plots, tables, narrative text, and interactive elements
- **A dashboard** — an index page linking to all phase reports
- **Parquet data files** — the intermediate and final results in a format other tools can consume
- **A routing manifest** — documenting which IRT model was selected as canonical for each chamber
- **Run metadata** — timestamp, git commit hash, elapsed time, and parameter choices for full reproducibility

The entire pipeline is reproducible: the same data in, the same parameters, the same results out — down to the random seed for the Bayesian sampler. If you run `just pipeline 2025-26` today and again next month, you get identical reports.

## The Cross-Biennium Pipeline

The main pipeline runs on one biennium at a time. Two additional phases run *across* bienniums:

```
just cross-pipeline
```

This executes Phase 26 (Cross-Session Validation) and Phase 27 (Dynamic IRT), which require data from multiple sessions to work. These phases compare, align, and track legislators across the full 15-year span.

## A Living System

The pipeline is not a finished product in a box. It is an evolving codebase with 118 Architecture Decision Records documenting every significant design choice, a test suite with ~2,960 tests, and a documentation system (which you're reading) that tries to keep pace with the code.

When the pipeline encounters a new challenge — a data format change on the legislature's website, a convergence failure in a particular chamber, a statistical method that underperforms — the response follows a consistent pattern: diagnose the problem, design a solution, write tests, update documentation, and record the decision in an ADR. The horseshoe effect, for example, was discovered through 2D IRT, diagnosed through PCA axis analysis, and resolved through a seven-part quality gate system documented in ADR-0118. The solution took weeks to develop and added thousands of lines of code — but the pipeline is now robust to supermajority chambers that would have produced misleading results before.

This is what "correctness over speed" means in practice. The pipeline doesn't cut corners. If a shortcut would produce wrong results, it takes the long way. If a simpler model is unreliable, it uses a more complex one. If a complex model doesn't converge, it falls back to the simpler one and tells you why.

## Looking Ahead

You now have the full map. The remaining eight volumes fill in the details:

- **Volume 2** dives into data acquisition — the scraper, the data model, and quality assurance
- **Volume 3** covers the first four phases — EDA, PCA, MCA, and UMAP
- **Volume 4** is the mathematical core — IRT explained step by step, with every equation in plain English
- **Volume 5** covers validation — how we know the numbers are right
- **Volume 6** explores clustering, networks, and classical indices
- **Volume 7** covers machine learning, text analysis, and model legislation detection
- **Volume 8** tracks change over time — dynamic ideal points and cross-session alignment
- **Volume 9** explains how the numbers become narrative reports

Each volume is self-contained enough to read on its own, but they build on each other. If a concept in Volume 6 depends on something introduced in Volume 4, we'll tell you where to look.

The journey from raw HTML to a statistical portrait of a legislature is a long one. But it starts with a single command, and now you know what that command sets in motion.

---

## Key Takeaway

The Tallgrass pipeline is a 29-phase statistical assembly line that transforms raw vote data into ideology scores, coalition maps, and narrative reports. It runs in about 45 minutes per biennium, is fully reproducible, and includes an automatic quality gate system that flags unreliable results and falls back to simpler models when necessary. The pipeline prioritizes correctness over speed: no shortcuts, no glossing over problems, no false precision.

---

*Terms introduced: pipeline, phase, vote matrix, PCA, IRT, ideal point, Bayesian inference, discrimination, difficulty, hierarchical model, partial pooling, convergence, quality gate, canonical routing, bridge legislator, horseshoe effect*

*Next: [Volume 2 — Gathering the Data](../volume-02-gathering-data/)*
