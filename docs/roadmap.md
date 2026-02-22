# Roadmap

What's been done, what's next, and what's on the horizon for the KS Vote Scraper analytics pipeline.

**Last updated:** 2026-02-22 (after NLP bill text features for Prediction)

---

## Completed Phases

| # | Phase | Date | Key Finding |
|---|-------|------|-------------|
| 1 | EDA | 2026-02-19 | 82% Yea base rate, 72% R supermajority, 34 veto overrides |
| 2 | PCA | 2026-02-19 | PC1 = party (57% variance), PC2 = contrarianism (Tyson/Thompson) |
| 3 | Bayesian IRT | 2026-02-20 | 1D ideal points converge cleanly; Tyson paradox identified |
| 4 | Clustering | 2026-02-20 | k=2 optimal (party split); intra-R variation is continuous, not factional |
| 5 | Network | 2026-02-20 | Zero cross-party edges at kappa=0.40; Schreiber sole bipartisan bridge |
| 6 | Prediction | 2026-02-20 | Vote AUC=0.98; IRT features do all the work; XGBoost adds nothing over logistic |
| 7 | Classical Indices | 2026-02-21 | Rice, CQ unity, ENP, weighted maverick; Schreiber/Dietrich top mavericks |
| 2b | UMAP | 2026-02-22 | Nonlinear ideological landscape; validates PCA/IRT; most accessible visualization |
| 6+ | NLP Bill Text Features | 2026-02-22 | NMF topic features on short_title added to bill passage prediction |
| — | Synthesis Report | 2026-02-22 | 32-section narrative HTML; joins all 8 phases into one deliverable |

---

## Next Up

### 1. Visualization Improvement Pass

**Priority:** Highest — the nontechnical audience rule was added during Phase 7 (Indices). Phases 1-6 predate it and need retrofitting. The Synthesis Report reuses upstream plots directly, so improving them improves the deliverable.

The Indices phase is the gold standard: plain-English titles ("Who Are the Most Independent Legislators?"), annotated key actors, and report text that defines every metric before showing plots. The earlier phases have good HTML report prose but their plots are still analyst-facing. 112+ total plots across 8 phases; roughly half need improvement.

**Guiding principle:** If a finding is explained in the HTML report, it should also appear visually in at least one plot. If a legislator is flagged in `docs/analytic-flags.md`, they should have a visual highlight somewhere.

#### Network Phase (highest priority — Peck example)

The network phase computes betweenness centrality and identifies "bridge" legislators, but the visualizations don't make this legible to a nontechnical reader. The centrality scatter plot (betweenness vs eigenvector) shows Peck elevated on the Y axis, but nothing explains *what that means* or *how it was derived*.

Specific improvements:
- **Annotate bridge legislators** on the community network plot — red border/halo on high-betweenness nodes (Peck, Schreiber, Thompson), with callout text: "Peck connects otherwise-separate voting blocs"
- **Add a "What is betweenness?" inset** — a simple 3-node diagram showing how removing a bridge legislator disconnects groups, next to the actual network plot
- **Replace or supplement the centrality scatter** with a ranked bar chart: "Who Holds the Most Influence in the Network?" with plain-English annotation ("higher = more paths between other legislators run through this person")
- **Highlight Schreiber's cross-party edges** — at kappa=0.30, he's the sole link between R and D. A before/after pair showing the network with and without Schreiber would be powerful
- **Label the threshold sweep plot** with event markers: "At this threshold, the network splits into two parties" instead of just showing curves

#### IRT Phase

The forest plots and trace plots are standard statistical outputs but opaque to nontechnical readers.

Specific improvements:
- **Add a Tyson spotlight subplot**: side-by-side showing (left) her voting pattern on high-discrimination bills (100% conservative) vs low-discrimination bills (50% contrarian), (right) her position on the forest plot with a callout box explaining the paradox
- **Replace or supplement trace plots** with a convergence summary panel — "The model ran 4 independent chains and they all agree" with a simple visual (overlapping distributions) instead of spaghetti lines
- **Annotate the forest plot** — highlight Tyson, Thompson, Schreiber, Miller with color/icons and brief labels explaining why they're interesting
- **Add plain-English title**: "Where Does Each Legislator Fall on the Ideological Spectrum?" instead of "IRT Ideal Points with 95% HDI"

#### Prediction Phase

SHAP beeswarm plots are cryptic for anyone who hasn't taken a machine learning course.

Specific improvements:
- **Replace SHAP beeswarm** with a simplified "What Predicts a Yea Vote?" bar chart showing top 5 features with plain-English labels (e.g., "How conservative the legislator is" instead of "xi_mean", "How partisan the bill is" instead of "beta_mean")
- **Add a "Hardest to Predict" spotlight** — scatter plot highlighting the 5-10 legislators the model struggles with most, annotated with names and brief explanations ("Shallenburger — procedural role as VP of Senate", "Helgerson — most moderate Democrat")
- **Simplify calibration plot** with annotation: "When the model says 80% chance of Yea, it's right about 80% of the time"

#### PCA Phase

PC2 is labeled "secondary dimension" with no interpretation on the plot itself.

Specific improvements:
- **Annotate PC2 axis**: "Contrarianism — legislators who vote Nay on routine, near-unanimous bills" directly on the ideological map
- **Add callout for Tyson** (PC2 = -24.8, 3x more extreme than next senator) and Thompson (-8.0)
- **Label the scree plot** with interpretation: "The sharp elbow means Kansas is essentially a one-dimensional legislature — party affiliation explains almost everything"

#### Clustering Phase

Dendrograms are hard to read; within-party subclusters lack interpretation.

Specific improvements:
- **Annotate the IRT-vs-loyalty scatter** with Tyson and Thompson's positions and a text box: "These two senators are ideologically extreme but unreliable caucus members"
- **Add a "What k=2 means" annotation** on the main cluster plot: "The data says there are exactly two groups — and they match party labels perfectly"
- **Simplify or replace dendrograms** with a more readable alternative for the report (the dendrogram can remain as supplementary)

#### EDA Phase

Mostly fine, but the heatmaps are dense and lack annotation.

Specific improvements:
- **Add name labels to heatmap axes** (currently just colored by party, no individual names visible)
- **Annotate interesting patterns**: mark Schreiber's row/column as the highest cross-party agreement

### 2. Beta-Binomial Party Loyalty (Bayesian)

**Priority:** High — experimental code already exists.

Method documented in `Analytic_Methods/14_BAY_beta_binomial_party_loyalty.md`. Experimental implementation exists at `analysis/irt_beta_experiment.py` (558 lines) but is not integrated into the main pipeline.

- Bayesian alternative to the frequentist CQ party unity from Phase 7
- Produces credible intervals on loyalty — crucial for legislators with few party votes
- Beta(alpha, beta) posterior per legislator; shrinks noisy estimates toward the group mean
- Compare Bayesian loyalty posteriors to CQ unity point estimates
- Especially useful for Miller (30 votes) and other sparse legislators

### 3. Hierarchical Bayesian Legislator Model

**Priority:** High — the "Crown Jewel" from the methods overview.

Method documented in `Analytic_Methods/16_BAY_hierarchical_legislator_model.md`. Legislators nested within party and chamber, with partial pooling:

- Models legislator-level parameters as draws from party-level distributions
- Naturally handles the R supermajority (more data = tighter party estimate)
- Quantifies how much individual legislators deviate from their party's typical behavior
- Partial pooling shrinks extreme estimates (Tyson, Miller) toward party mean — the statistically principled version of what CQ unity does informally
- Uses PyMC (already installed for IRT)

### 4. Cross-Session Scrape (2023-24)

**Priority:** High — unlocks temporal analysis and honest out-of-sample validation.

- Run `uv run ks-vote-scraper 2023` to scrape the prior biennium
- Produces a second set of 3 CSVs in `data/90th_2023-2024/`
- Then run the full 8-phase pipeline: `just scrape 2023 && just eda --session 2023-24 && ...`
- Enables all three cross-session analyses below

### 5. Cross-Session Validation

**Priority:** High — the single biggest gap in current results.

Three distinct analyses become possible once 2023-24 is scraped:

- **Prediction honesty (out-of-sample):** Train vote prediction on 2023-24, test on 2025-26 (and vice versa). This is the gold standard for prediction validation — within-session holdout (AUC=0.98) is optimistic because the model sees the same legislators and session dynamics. Cross-session tests whether the learned patterns generalize. Also solves the Senate bill passage small-N problem (59 test bills in 2025-26 is too few; stacking sessions doubles the data).
- **Temporal comparison (who moved?):** Compare IRT ideal points for returning legislators across bienniums. Who shifted ideology? Are the 2025-26 mavericks (Schreiber, Dietrich) the same people who were mavericks in 2023-24? This is the most newsworthy output for the nontechnical audience — "Senator X moved 1.2 points rightward since last session" is a concrete, actionable finding.
- **Detection threshold validation:** The synthesis detection thresholds (unity > 0.95 skip, rank gap > 0.5 for paradox, betweenness within 1 SD for bridge) were calibrated on 2025-26. Running synthesis on 2023-24 tests whether they produce sensible results on a different session with potentially different partisan dynamics. If they don't, the thresholds need to become adaptive or session-parameterized.

### 6. MCA (Multiple Correspondence Analysis)

**Priority:** Medium — alternative view on the vote matrix.

Method documented in `Analytic_Methods/10_DIM_correspondence_analysis.md`. MCA treats each vote as a categorical variable rather than numeric, which is technically more appropriate for Yea/Nay data:

- Complementary to PCA: PCA assumes continuous, MCA assumes categorical
- May reveal structure PCA misses, especially in voting patterns where abstention is meaningful
- `prince` library already in `pyproject.toml`
- Compare MCA dimensions to PCA PC1/PC2 — if they agree, PCA's linear assumption is validated

### 7. Time Series Analysis

**Priority:** Medium — adds temporal depth to static snapshots.

Two methods documented but not yet implemented:

- **Ideological drift** (`Analytic_Methods/26_TSA_ideological_drift.md`): Rolling IRT or rolling party unity within a session. Did anyone change position mid-session? Track 15-vote rolling windows.
- **Changepoint detection** (`Analytic_Methods/27_TSA_changepoint_detection.md`): Structural breaks in voting patterns. When did the session's character shift? (e.g., pre- vs post-veto override period)

Requires the `ruptures` library (already in `pyproject.toml`). Becomes much more powerful once 2023-24 data is available for cross-session comparison.

### 8. 2D Bayesian IRT Model

**Priority:** Medium — solves the Tyson paradox properly. (Formerly item #9.)

The 1D model compresses Tyson's two-dimensional behavior (ideology + contrarianism) into one axis. A 2D model would:
- Place Tyson as (very conservative on Dim 1, extreme outlier on Dim 2)
- Improve predictions for legislators with unusual PC2 patterns
- Validate whether the PCA PC2 "contrarianism" dimension has a Bayesian counterpart

This is computationally expensive (doubles MCMC time) and requires careful identification constraints for rotation.

---

## Deferred / Low Priority

### Joint Cross-Chamber IRT

A full joint MCMC model was attempted and failed: 71 shared bills for 169 legislators is too sparse. Currently using classical test equating (A=1.136, B=-0.305). Revisit only if a future session has significantly more shared bills, or if the 2023-24 session provides additional bridging data.

### Latent Class Mixture Models

Documented in `Analytic_Methods/28_CLU_latent_class_mixture_models.md`. Probabilistic alternative to k-means for discrete faction discovery. Low priority because clustering already showed within-party variation is continuous, not factional — there aren't discrete factions to discover.

### Dynamic Ideal Points (Martin-Quinn)

Track legislator positions over time within a session using a state-space model. Deferred until cross-session data is available, where it becomes much more powerful (track a legislator across bienniums).

### Bipartite Bill-Legislator Network

Documented in `Analytic_Methods/21_NET_bipartite_bill_legislator.md`. Two-mode network connecting legislators to bills they voted on. Intentionally deferred — the Kappa-based co-voting network already captures the same information more efficiently.

### Posterior Predictive Checks (Standalone)

Documented in `Analytic_Methods/17_BAY_posterior_predictive_checks.md`. Already partially integrated into the IRT phase (PPC plots in the IRT report). A standalone, cross-model PPC comparison could be useful once the hierarchical model and 2D IRT are implemented.

### Analysis Phase Primers

Each results directory should have a `README.md` explaining the analysis for non-code readers. Low priority — the HTML reports serve this role for now, but standalone primers would be useful for the `results/` directory.

### Test Suite Expansion

422 tests exist across scraper (146) and analysis (276) modules. Coverage could be expanded:
- Integration tests that run a mini end-to-end pipeline on fixture data
- Cross-session tests (once 2023-24 is scraped) to verify scripts handle multiple sessions
- Snapshot tests for HTML report output stability

---

## Explicitly Rejected

| Method | Why |
|--------|-----|
| W-NOMINATE (`Analytic_Methods/12_DIM_w_nominate.md`) | R-only; project policy is Python-only (no rpy2) |
| Optimal Classification (`Analytic_Methods/13_DIM_optimal_classification.md`) | R-only; same as above |

---

## Scraper Maintenance

| Item | When | Details |
|------|------|---------|
| Update `CURRENT_BIENNIUM_START` | 2027 | Change from 2025 to 2027 in `session.py` |
| Add special sessions | As needed | Add year to `SPECIAL_SESSION_YEARS` in `session.py` |
| Fix Shallenburger suffix | Next scraper touch | "Vice President of the Senate" not in leadership suffix strip pattern |

---

## All 29 Analytic Methods — Status

| # | Method | Category | Status |
|---|--------|----------|--------|
| 01 | Data Loading & Cleaning | DATA | Completed (EDA) |
| 02 | Vote Matrix Construction | DATA | Completed (EDA) |
| 03 | Descriptive Statistics | EDA | Completed (EDA) |
| 04 | Missing Data Analysis | EDA | Completed (EDA) |
| 05 | Rice Index | IDX | Completed (Indices) |
| 06 | Party Unity | IDX | Completed (Indices) |
| 07 | ENP | IDX | Completed (Indices) |
| 08 | Maverick Scores | IDX | Completed (Indices) |
| 09 | PCA | DIM | Completed (PCA) |
| 10 | MCA / Correspondence Analysis | DIM | **Planned** — item #6 above |
| 11 | UMAP / t-SNE | DIM | Completed (UMAP, Phase 2b) |
| 12 | W-NOMINATE | DIM | Rejected (R-only) |
| 13 | Optimal Classification | DIM | Rejected (R-only) |
| 14 | Beta-Binomial Party Loyalty | BAY | **Experimental** — item #2 above |
| 15 | Bayesian IRT (1D) | BAY | Completed (IRT) |
| 16 | Hierarchical Bayesian Model | BAY | **Planned** — item #3 above |
| 17 | Posterior Predictive Checks | BAY | Partial (embedded in IRT) |
| 18 | Hierarchical Clustering | CLU | Completed (Clustering) |
| 19 | K-Means / GMM Clustering | CLU | Completed (Clustering) |
| 20 | Co-Voting Network | NET | Completed (Network) |
| 21 | Bipartite Bill-Legislator Network | NET | Deferred (redundant) |
| 22 | Centrality Measures | NET | Completed (Network) |
| 23 | Community Detection | NET | Completed (Network) |
| 24 | Vote Prediction | PRD | Completed (Prediction) |
| 25 | SHAP Analysis | PRD | Completed (Prediction) |
| 26 | Ideological Drift | TSA | **Planned** — item #7 above |
| 27 | Changepoint Detection | TSA | **Planned** — item #7 above |
| 28 | Latent Class Mixture Models | CLU | Deferred (no discrete factions found) |

**Score: 18 completed, 2 rejected, 5 planned, 1 experimental, 2 deferred, 1 partial = 29 total**

---

## Key Architectural Decisions Still Standing

- **Polars over pandas** everywhere
- **Python over R** — no rpy2, no W-NOMINATE
- **IRT ideal points are the primary feature** — prediction confirmed this; everything else is marginal
- **Chambers analyzed separately** unless explicitly doing cross-chamber comparison
- **Tables never truncated** in analysis reports — full data for initial analysis
- **Nontechnical audience** — all visualizations self-explanatory, plain-English labels, annotated findings
