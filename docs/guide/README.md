# The Tallgrass Guide

**A plain-language documentation series for the Tallgrass legislative analysis platform**

This guide explains — in everyday language — how Tallgrass turns Kansas Legislature roll call votes into statistical measures of ideology, polarization, and legislative behavior. It is written for readers who are curious, not necessarily credentialed. Every equation is walked through step by step with analogies and examples. Every design choice is explained with the *why*, not just the *what*.

The guide is organized into nine volumes. Each volume is self-contained enough to read on its own, but they build on each other in sequence. Start with Volume 1 for the big picture, then dive into whichever topic interests you.

---

## Series at a Glance

| Volume | Title | Chapters | Audience Entry Point |
|--------|-------|----------|---------------------|
| [1](volume-01-big-picture/) | **The Big Picture** | 4 | Everyone — start here |
| [2](volume-02-gathering-data/) | **Gathering the Data** | 5 | Anyone curious about web scraping or data quality |
| [3](volume-03-first-look/) | **Your First Look at the Votes** | 5 | Anyone comfortable with percentages and averages |
| [4](volume-04-measuring-ideology/) | **Measuring Ideology** | 7 | The mathematical core — equations explained step by step |
| [5](volume-05-checking-our-work/) | **Checking Our Work** | 5 | How we know the numbers are right |
| [6](volume-06-finding-patterns/) | **Finding Patterns** | 6 | Clustering, networks, and classical political science |
| [7](volume-07-prediction-and-text/) | **Prediction and Text** | 5 | Machine learning and natural language processing |
| [8](volume-08-change-over-time/) | **Change Over Time** | 5 | Tracking ideology across sessions and years |
| [9](volume-09-telling-the-story/) | **Telling the Story** | 4 | How raw numbers become readable reports |
| [A-E](appendices/) | **Appendices** | 5 | Glossary, equations, phases, data dictionary, further reading |

**Total: 9 volumes, 46 chapters, 5 appendices**

---

## Volume Summaries

### Volume 1 — The Big Picture

*Why does any of this matter? What can you learn from how legislators vote?*

This volume introduces the project, the Kansas Legislature, and the core question: can we measure where a lawmaker stands on a left-right spectrum just by watching how they vote? It covers the 15-year span of data (2011–2026), the basic structure of a roll call vote, and a roadmap of the entire pipeline — what goes in, what comes out, and why each step exists.

**Chapters:**
1. What Is Tallgrass?
2. A Quick Tour of the Kansas Legislature
3. What Roll Call Votes Tell Us (and What They Don't)
4. The Pipeline: From Raw Votes to Insight

---

### Volume 2 — Gathering the Data

*Before analysis, you need data. Here's how we get it — reliably, reproducibly, and respectfully.*

This volume explains how Tallgrass scrapes the Kansas Legislature's website, retrieves bill text PDFs, cross-validates against a second data source (KanFocus), and loads everything into a database. It covers the ethics and mechanics of web scraping, the four-phase scraper pipeline, historical data challenges (2011–2014 ODT vote files), and data quality assurance.

**Chapters:**
1. Web Scraping: The Polite Robot
2. The Four-Phase Scraper Pipeline
3. Historical Sessions and the ODT Challenge
4. Cross-Validation: Trust but Verify (KanFocus)
5. Bill Text, ALEC Detection, and the Database

---

### Volume 3 — Your First Look at the Votes

*What does a matrix of 125 legislators × 600 votes actually look like? How do we start making sense of it?*

This volume covers exploratory data analysis and dimensionality reduction — the techniques that compress a giant spreadsheet of Yea/Nay votes into something a human can visualize and interpret. It walks through filtering, agreement measurement, PCA, MCA, and UMAP with concrete Kansas examples.

**Chapters:**
1. The Vote Matrix: Ones, Zeros, and Missing Data
2. Who Agrees with Whom? (Cohen's Kappa)
3. Compressing the Data: PCA Explained
4. Alternative Views: MCA and UMAP
5. The Horseshoe Problem (When the Map Lies)

---

### Volume 4 — Measuring Ideology

*The mathematical heart of the project. How do you assign a number to a legislator's ideology — and what does that number actually mean?*

This is the longest and most important volume. It explains Item Response Theory (IRT) from the ground up, using the analogy of a standardized test where bills are "questions" and legislators are "students." It covers the 1D model, the jump to 2D, hierarchical models with party structure, and the identification problem (why the math alone can't tell left from right). Every equation is presented first in plain English, then in notation, then walked through with a real Kansas example.

**Chapters:**
1. The Testing Analogy: Bills as Questions, Legislators as Students
2. The 1D IRT Model — Step by Step
3. Anchors and Sign: Telling Left from Right
4. When One Dimension Isn't Enough: 2D IRT
5. Partial Pooling: Hierarchical Models and Party Structure
6. The Identification Zoo: Seven Strategies
7. Canonical Ideal Points: Choosing the Best Score

---

### Volume 5 — Checking Our Work

*How do we know the model is right? What does "right" even mean for an ideology score?*

This volume covers validation — the practice of testing whether our statistical models produce trustworthy results. It explains posterior predictive checks (does the model predict what actually happened?), comparison with the political science gold standard (Shor-McCarty scores), the W-NOMINATE benchmark, and the tiered quality gate system that automatically flags unreliable results.

**Chapters:**
1. What Does Validation Mean?
2. Posterior Predictive Checks: Can the Model Predict Its Own Data?
3. The Gold Standard: Shor-McCarty External Validation
4. W-NOMINATE: Comparing to the Field Standard
5. Quality Gates: Automatic Trust Levels

---

### Volume 6 — Finding Patterns

*Are there voting blocs? Who are the bridge-builders? How unified is each party?*

This volume explains how clustering, network analysis, and classical political science indices reveal the structure of legislative coalitions. It covers hierarchical clustering, latent class analysis, co-voting networks, bipartite bill-legislator networks, and standard metrics like the Rice Index and party unity scores.

**Chapters:**
1. Clustering: Are There Discrete Factions?
2. Latent Class Analysis: Testing for Hidden Groups
3. Co-Voting Networks: Who Votes Together?
4. Bipartite Networks: Bills That Bridge the Aisle
5. Classical Indices: Rice, Party Unity, and the Maverick Score
6. Empirical Bayes: Shrinkage Estimates of Party Loyalty

---

### Volume 7 — Prediction and Text

*Can we predict how a legislator will vote? What do the words of a bill tell us about its politics?*

This volume explains the machine learning and natural language processing phases. It covers vote prediction (XGBoost), bill passage forecasting, topic modeling (BERTopic), text-based ideal points, issue-specific ideology scores, and model legislation detection (ALEC).

**Chapters:**
1. Predicting Votes: What XGBoost Learns
2. Bill Passage: Can We Forecast Outcomes?
3. Topic Modeling: What Are the Bills About?
4. Text-Based Ideology and Issue-Specific Scores
5. Model Legislation: Detecting Copy-Paste Policy

---

### Volume 8 — Change Over Time

*Legislatures aren't static. How do we track polarization trends, ideology shifts, and the effects of turnover?*

This volume explains temporal analysis — the techniques that track how the legislature changes across sessions and years. It covers time series analysis (changepoint detection), dynamic ideal points (the Martin-Quinn model), cross-session alignment, and the decomposition of ideological shifts into individual conversion versus replacement effects.

**Chapters:**
1. Time Series: Detecting When the Legislature Shifts
2. Dynamic Ideal Points: The Martin-Quinn Model
3. Cross-Session Alignment: Putting Sessions on the Same Scale
4. Conversion vs. Replacement: Why Does Ideology Change?
5. Fifteen Years of Kansas Politics (2011–2026)

---

### Volume 9 — Telling the Story

*Numbers don't speak for themselves. Here's how we turn 28 phases of analysis into a readable narrative.*

This volume explains the synthesis and reporting system — how Tallgrass automatically generates HTML reports with data-driven narratives, per-legislator profiles, and interactive visualizations. It covers the report architecture, the algorithmic detection of "notable" legislators, the horseshoe-aware narrative system, and how to read the final output.

**Chapters:**
1. The Report System: From Data to Narrative
2. Synthesis: The Session Story
3. Legislator Profiles: Individual Deep Dives
4. How to Read a Tallgrass Report

---

## Scope and Sequence

The guide follows a deliberate pedagogical sequence. Each volume builds vocabulary and intuition that later volumes rely on.

```
Volume 1: The Big Picture
  └─ Establishes: what a roll call is, what "ideology" means, pipeline overview
      │
Volume 2: Gathering the Data
  └─ Establishes: data sources, quality assurance, what the raw CSV files contain
      │
Volume 3: First Look at the Votes
  └─ Establishes: vote matrix, agreement, PCA, the horseshoe problem
      │         (readers now understand dimensionality reduction)
      │
Volume 4: Measuring Ideology  ← mathematical core
  └─ Establishes: IRT, ideal points, Bayesian inference, identification
      │         (readers now understand what an "ideal point" is and why it's hard)
      │
Volume 5: Checking Our Work
  └─ Establishes: validation, convergence, quality gates
      │         (readers now understand when to trust results)
      │
Volume 6: Finding Patterns
  └─ Establishes: clustering, networks, classical indices
      │         (readers now understand coalition structure)
      │
Volume 7: Prediction and Text
  └─ Establishes: ML, NLP, text-based scaling
      │         (readers now understand how bill content relates to ideology)
      │
Volume 8: Change Over Time
  └─ Establishes: temporal dynamics, Martin-Quinn, cross-session alignment
      │         (readers now understand ideology as a trajectory, not a snapshot)
      │
Volume 9: Telling the Story
  └─ Synthesizes everything into readable output
```

---

## Writing Principles

These principles govern every chapter in the series.

### 1. Lead with Intuition, Follow with Formalism

Every concept is introduced with a real-world analogy *before* any equation appears. The equation is then presented in three layers:

1. **Plain English** — "We multiply how much the bill separates people by where the legislator stands, then subtract how hard the bill is to pass."
2. **Notation** — P(Yea) = logit⁻¹(β · ξ − α)
3. **Worked example** — "Senator Smith has an ideal point of +0.8. This bill has a discrimination of 1.2 and a difficulty of −0.3. So the probability of a Yea vote is..."

### 2. Kansas Examples Throughout

Abstract statistics become concrete when attached to real data. Every chapter uses Kansas Legislature examples:
- The 79th Senate (2021–2022) as the extreme supermajority case study
- The 91st Legislature (2025–2026) as the "current" baseline
- Named legislators only when their voting records illustrate a statistical concept (not as political commentary)

### 3. Honest About Uncertainty

The guide never claims more precision than the data supports. When a model struggles — the horseshoe effect in supermajority chambers, convergence failures in small caucuses, PCA axis instability — the guide explains *what went wrong, why, and what we do about it*.

### 4. Code Pointers, Not Code Dumps

Each chapter references the relevant source files (e.g., "the IRT model is built in `analysis/05_irt/irt.py`") but does not reproduce large code blocks. The guide is for understanding concepts, not for reading Python.

### 5. Glossary-Linked Vocabulary

Technical terms are defined on first use and collected in a running glossary (Appendix A). Terms that have both a general and a technical meaning (e.g., "discrimination," "difficulty," "identification") are explicitly disambiguated.

---

## Appendices (Shared Across Volumes)

| Appendix | Content |
|----------|---------|
| A | **Glossary** — All technical terms with plain-language definitions |
| B | **Equation Reference** — Every equation in the series, numbered, with page/chapter cross-references |
| C | **The 28 Phases at a Glance** — One-paragraph summary of each pipeline phase |
| D | **Data Dictionary** — CSV schemas, column definitions, join keys |
| E | **Further Reading** — Academic papers, textbooks, and open-source tools cited in the guide |

---

## Chapter Writing Checklist

Each chapter should include:

- [ ] **Opening hook** — A question or scenario that motivates the topic
- [ ] **Analogy** — At least one real-world analogy for the core concept
- [ ] **Equations** — Presented in the three-layer format (English → notation → example)
- [ ] **Kansas example** — At least one worked example using real Tallgrass data
- [ ] **What can go wrong** — Honest discussion of failure modes and edge cases
- [ ] **Codebase pointer** — File paths to the relevant implementation
- [ ] **Key takeaway** — One-sentence summary at the end
- [ ] **Terms introduced** — List of new vocabulary for the glossary

---

## Detailed Chapter Outlines

Below are expanded outlines for each chapter, including the key equations, analogies, and Kansas examples to cover.

### Volume 1 — The Big Picture

#### Chapter 1: What Is Tallgrass?

- **Hook:** "What if you could read the political DNA of every Kansas legislator — not from their campaign speeches, but from every vote they've ever cast?"
- Tallgrass as a project: open-source (MIT), Python, 15 years of data
- The name: tallgrass prairie as Kansas metaphor
- What the pipeline produces: ideology scores, polarization measures, coalition maps, per-legislator profiles
- Who it's for: researchers, journalists, civic organizations, curious citizens
- **Codebase:** `src/tallgrass/` (scraper), `analysis/` (pipeline), `src/web/` (API)

#### Chapter 2: A Quick Tour of the Kansas Legislature

- Bicameral: 125 House members, 40 Senators
- Two-year bienniums (e.g., 2025–2026 = 91st Legislature)
- Types of votes: final passage, committee reports, motions, veto overrides, amendments
- The base rate problem: ~82% of votes are Yea (most legislation passes)
- Special sessions and their quirks
- **Kansas example:** The 91st Legislature by the numbers — bills introduced, votes recorded, party breakdown

#### Chapter 3: What Roll Call Votes Tell Us (and What They Don't)

- Roll calls as revealed preference (what legislators *do*, not what they *say*)
- The vote matrix: rows = legislators, columns = votes, cells = Yea/Nay/Absent
- What's missing: voice votes, committee kills, amendment negotiations, logrolling
- Strategic absences: legislators who skip controversial votes
- The 82% Yea base rate: why raw agreement percentages are misleading
- **Analogy:** Judging a restaurant by its menu (what's served) vs. its kitchen (how decisions are made)

#### Chapter 4: The Pipeline: From Raw Votes to Insight

- The 28-phase pipeline as an assembly line
- Phase groups: data prep → ideology estimation → validation → pattern detection → temporal analysis → synthesis
- What flows between phases: filtered vote matrices, ideal point estimates, quality gate flags
- The quality gate system: Tier 1 (fully trusted), Tier 2 (point estimates only), Tier 3 (fallback)
- Runtime: ~45 minutes per biennium on a MacBook Pro
- **Diagram:** Simplified pipeline flow chart (phases grouped by purpose, not numbered)

---

### Volume 2 — Gathering the Data

#### Chapter 1: Web Scraping: The Polite Robot

- **Hook:** "The Kansas Legislature publishes every roll call vote online. The catch? They're buried across thousands of web pages, in formats that change every few years."
- What web scraping is: automated browsing
- Ethics: rate limiting, caching, identifying yourself, respecting robots.txt
- The kslegislature.gov structure: biennium URLs, bill pages, vote pages
- **Analogy:** A librarian who reads every page of every book and writes down every vote in a spreadsheet — except they do it in seconds instead of months
- **Codebase:** `src/tallgrass/config.py` (rate limits), `src/tallgrass/session.py` (URL construction)

#### Chapter 2: The Four-Phase Scraper Pipeline

- Phase 1: **Discover** — Find all bills via HTML listing pages (with JavaScript fallback)
- Phase 2: **Filter** — Query the KLISS API to identify bills that actually had recorded votes
- Phase 3: **Parse** — Extract vote data from HTML vote pages (who voted what)
- Phase 4: **Enrich** — Fetch legislator metadata (party, district, full name, OpenStates ID)
- Concurrency pattern: fetch in parallel, parse sequentially
- Retry waves: exponential backoff for transient failures
- **Kansas example:** Scraping the 91st Legislature — ~1,200 bills discovered, ~600 with votes, ~3,000 individual vote pages
- **Codebase:** `src/tallgrass/scraper.py` (KSVoteScraper class)

#### Chapter 3: Historical Sessions and the ODT Challenge

- Why historical data matters: 15 years enables trend analysis
- The format break: 2011–2014 (84th–85th) votes stored as ODT (OpenDocument) files, not HTML
- ODT parsing: unzip → XML → extract vote categories
- Name resolution: ODT files use full names, not slugs — requires member directory lookup
- The special session problem: merged into parent bienniums
- **Kansas example:** The 84th Legislature (2011–2012) — 30% committee-of-the-whole votes (tally only, no individual records)
- **Codebase:** `src/tallgrass/odt_parser.py`, `src/tallgrass/merge_special.py`

#### Chapter 4: Cross-Validation: Trust but Verify (KanFocus)

- Why a second data source matters: independent verification catches scraper bugs
- KanFocus: a paid legislative tracking service with its own vote records (1999–2026)
- The cross-validation process: match votes by (bill, chamber, date), compare individual legislator votes
- Slug resolution: different name formats across sources
- Discrepancy handling: "Absent Not Voting" vs. "Not Voting" ambiguity
- **Kansas example:** Cross-validation results — match rates, resolved discrepancies
- **Codebase:** `src/tallgrass/kanfocus/crossval.py`

#### Chapter 5: Bill Text, ALEC Detection, and the Database

- Bill text retrieval: PDF download + pdfplumber extraction
- Supplemental notes vs. introduced text (supplemental notes are shorter, plain English)
- ALEC model legislation corpus: 1,061 template policies
- PostgreSQL database: 649K votes, 8K roll calls, 2K legislators
- The auto-load hook: scrape → CSV → database in one command
- **Codebase:** `src/tallgrass/text/`, `src/tallgrass/alec/`, `src/tallgrass/db_hook.py`

---

### Volume 3 — Your First Look at the Votes

#### Chapter 1: The Vote Matrix: Ones, Zeros, and Missing Data

- **Hook:** "Imagine a spreadsheet with 125 rows (one per legislator) and 600 columns (one per vote). Each cell says Yea, Nay, or nothing at all. How do you make sense of it?"
- Binary encoding: Yea = 1, Nay = 0, Absent = missing
- Filtering: remove near-unanimous votes (< 2.5% minority), require ≥ 20 votes per legislator
- Why filtering matters: unanimous votes tell us nothing about ideology
- The base rate: 82% Yea means even random voting produces high "agreement"
- **Analogy:** A multiple-choice test where 82% of the answers are (A) — the easy questions don't tell you who studied
- **Codebase:** `analysis/01_eda/eda.py`

#### Chapter 2: Who Agrees with Whom? (Cohen's Kappa)

- Raw agreement: % of votes where two legislators voted the same way
- The problem: with 82% Yea, two random legislators agree ~70% of the time
- Cohen's Kappa: corrects for chance agreement

  **Plain English:** "How much more do these two legislators agree than we'd expect by coin-flip?"

  **Equation:**
  ```
  κ = (observed agreement − expected agreement) / (1 − expected agreement)
  ```

  **Worked example:** Rep. A and Rep. B agree on 90% of votes. With an 82% Yea rate, expected agreement is ~70%. So κ = (0.90 − 0.70) / (1 − 0.70) = 0.67 — "substantial" agreement beyond chance.

- The Landis-Koch scale: < 0 = worse than chance, 0–0.20 = slight, 0.21–0.40 = fair, 0.41–0.60 = moderate, 0.61–0.80 = substantial, 0.81–1.0 = near-perfect
- **Kansas example:** Kappa agreement matrix for the 91st House — party structure immediately visible
- **Codebase:** `analysis/01_eda/eda.py` (agreement computation)

#### Chapter 3: Compressing the Data: PCA Explained

- **Hook:** "You have 600 votes. Can you summarize a legislator's entire voting record with just one or two numbers?"
- The compression analogy: PCA finds the "most important directions" in the data
- **Analogy:** A photographer taking a group photo — they look for the angle that shows the most difference between people. PCA finds that angle in voting data.
- How it works (conceptual):
  1. Center the data (subtract the average)
  2. Find the direction of maximum variation (PC1)
  3. Find the next direction, perpendicular to the first (PC2)
  4. And so on...
- What PC1 usually captures: the party divide (left ↔ right)
- What PC2 usually captures: within-party variation (establishment ↔ maverick)
- Eigenvalues and the scree plot: "how much of the variation does each component explain?"

  **Equation:**
  ```
  PC1 score for legislator i = w₁·vote₁ + w₂·vote₂ + ... + w₆₀₀·vote₆₀₀
  ```
  where the weights (w) are chosen to maximize variance.

- **Kansas example:** 91st House PCA — PC1 explains ~35% of variation, cleanly separates parties
- **Codebase:** `analysis/02_pca/pca.py`

#### Chapter 4: Alternative Views: MCA and UMAP

- **MCA (Multiple Correspondence Analysis):**
  - Like PCA but designed for categorical data (Yea/Nay/Absent as three categories, not two numbers)
  - The Greenacre correction: fixes inflated eigenvalues
  - When it differs from PCA: mainly when absences form a pattern (strategic absence)
  - **Codebase:** `analysis/03_mca/mca.py`

- **UMAP (Uniform Manifold Approximation and Projection):**
  - A nonlinear method — preserves *neighborhoods*, not distances
  - **Analogy:** If PCA is like projecting a shadow of a 3D object onto a wall (flat projection), UMAP is like uncrumpling a piece of paper — it can unfold complex shapes that PCA can't handle
  - Cosine distance: measures angle between voting records (handles legislators who missed different votes)
  - Stability: run 5 times with different random seeds, check if the picture stays the same (Procrustes alignment)
  - **Warning:** UMAP axes are meaningless — only relative positions matter
  - **Codebase:** `analysis/04_umap/umap_phase.py`

#### Chapter 5: The Horseshoe Problem (When the Map Lies)

- **Hook:** "In 7 out of 14 Kansas Senate sessions, the first principal component doesn't capture ideology at all. It captures something else entirely."
- The horseshoe effect: in supermajority chambers (e.g., 75% Republican), PCA bends the left-right spectrum into a horseshoe
- Why it happens: the most extreme Republicans and the most moderate Democrats end up at the same position on PC1 — because both dissent from the majority
- **Analogy:** Imagine sorting people by height, but you're measuring their shadow from the side. Tall people who lean left and short people who lean right cast the same shadow — the measurement confuses two different things.
- Detection: Cohen's d between party means on PC1. If d < 1.5, the horseshoe is present.
- Affected sessions: 78th–83rd Senate (2013–2018) and 88th Senate (2023–2024)
- Impact: if uncorrected, PCA-based IRT initialization points the model in the wrong direction
- Solutions (previewed here, detailed in Vol. 4): 2D IRT, alternative identification strategies, quality gates
- **Codebase:** `analysis/phase_utils.py` (horseshoe detection), `docs/pca-ideology-axis-instability.md`

---

### Volume 4 — Measuring Ideology

#### Chapter 1: The Testing Analogy: Bills as Questions, Legislators as Students

- **Hook:** "Imagine a standardized test where the questions are bills and the students are legislators. A 'hard' question is a bill that even your allies might vote against. A 'discriminating' question is one that sharply separates liberals from conservatives."
- The Item Response Theory (IRT) framework:
  - Legislators = test-takers (with an unobserved "ability" = ideology)
  - Bills = test items (with difficulty and discrimination parameters)
  - Votes = responses (Yea = "correct," in the model's framework)
- Why IRT, not just counting votes:
  - Not all votes are equally informative
  - A 98-2 vote tells us almost nothing; a 51-49 vote is gold
  - IRT automatically weights informative votes more heavily
- The three parameters:
  - ξ (xi) = ideal point — where a legislator sits on the ideology spectrum
  - α (alpha) = difficulty — how hard a bill is to pass (higher α = requires more conservative support)
  - β (beta) = discrimination — how sharply a bill separates liberals from conservatives
- **Analogy (extended):** A math test with 50 questions. Question 1 ("What is 2+2?") has low discrimination — everyone gets it right. Question 50 (a calculus proof) has high discrimination — only strong math students get it. IRT figures out both the students' abilities and the questions' difficulties *simultaneously*.

#### Chapter 2: The 1D IRT Model — Step by Step

- The core equation:

  **Plain English:** "The probability that a legislator votes Yea depends on how close their ideology is to what the bill requires."

  **Equation:**
  ```
  P(Yea | ξ, α, β) = logistic(β · ξ − α)
  ```

  where logistic(x) = 1 / (1 + e⁻ˣ)

  **Step-by-step walkthrough:**
  1. Take the legislator's ideal point (ξ = +0.8 for a moderate conservative)
  2. Multiply by the bill's discrimination (β = 1.2 for a partisan bill)
  3. Subtract the bill's difficulty (α = −0.3 for an easy-to-pass bill)
  4. Result: 1.2 × 0.8 − (−0.3) = 0.96 + 0.3 = 1.26
  5. Convert via logistic function: 1/(1 + e⁻¹·²⁶) = 0.78
  6. Interpretation: 78% probability of a Yea vote

- The Item Characteristic Curve (ICC): a plot of P(Yea) vs. ξ — an S-shaped curve
  - Steep curve = high discrimination (partisan bill)
  - Flat curve = low discrimination (routine bill)
  - Curve shifted left = low difficulty (easy to pass)
  - Curve shifted right = high difficulty (hard to pass)

- **Bayesian estimation** (simplified):
  - We don't know ξ, α, or β — we estimate them from the data
  - Prior beliefs: legislators start at 0 (neutral), bills start at 0 (average difficulty)
  - The sampler (nutpie) tries thousands of possible parameter combinations
  - It keeps the ones that best explain the observed votes
  - Result: not a single number, but a *distribution* (we're honest about uncertainty)

- Prior distributions:
  ```
  ξ ~ Normal(0, 1)    — most legislators near center, few at extremes
  α ~ Normal(0, 5)    — wide range of bill difficulty
  β ~ Normal(0, 1)    — discrimination can be positive or negative
  ```

- **Kansas example:** A worked example with 3 legislators and 3 bills from the 91st House
- **Codebase:** `analysis/05_irt/irt.py`

#### Chapter 3: Anchors and Sign: Telling Left from Right

- The identification problem:
  - **Analogy:** "If you spin a compass, north is wherever you say it is. IRT has the same problem — without an anchor, 'liberal' and 'conservative' are interchangeable."
  - Mathematically: if you flip all ξ values (multiply by −1) and all β values (multiply by −1), the predictions are identical
  - The model can't tell left from right without help

- Hard anchors: we fix two known legislators
  - Pick one Republican (from PCA PC1 extreme) → fix ξ = +1
  - Pick one Democrat (from PCA PC1 extreme) → fix ξ = −1
  - Now the scale has direction: positive = conservative, negative = liberal

- The 7 identification strategies:
  1. **anchor-pca** (default for balanced chambers): anchors from PCA extremes
  2. **anchor-agreement** (for supermajority chambers): anchors from contested-vote agreement patterns
  3. **sort-constraint**: soft ordering constraint instead of hard anchors
  4. **positive-beta**: constrain all β > 0 (assumes all bills are "conservative = Yea")
  5. **hierarchical-prior**: party-level means with ordering
  6. **unconstrained**: no identification (for diagnostics only)
  7. **external-prior**: use scores from an external dataset

- Auto-detection: Tallgrass examines party balance and automatically picks the best strategy
- Post-hoc sign validation: even with anchors, we double-check the sign by correlating ideal points with party

- **Kansas example:** The 79th Senate — why anchor-pca fails (horseshoe) and anchor-agreement succeeds
- **Codebase:** `analysis/05_irt/irt.py`, `docs/irt-identification-strategies.md`

#### Chapter 4: When One Dimension Isn't Enough: 2D IRT

- **Hook:** "Sometimes the liberal-conservative axis doesn't capture everything. In the Kansas Senate, there's a second dimension: establishment loyalty vs. independent contrarianism."
- Why 2D:
  - 1D IRT assumes all disagreement is ideological
  - But some legislators break ranks for non-ideological reasons (institutional loyalty, personal feuds, regional interests)
  - 2D IRT separates "left-right" (Dim 1) from "establishment-maverick" (Dim 2)

- The 2D equation:

  **Plain English:** "Now the probability depends on two things about the legislator: their ideology *and* their establishment loyalty."

  **Equation:**
  ```
  P(Yea | ξ, α, β) = logistic(β₁·ξ₁ + β₂·ξ₂ − α)
  ```

  Each bill now has two discrimination parameters (β₁ for ideology, β₂ for establishment), and each legislator has two scores (ξ₁ for ideology, ξ₂ for establishment).

- The rotation problem: in 2D, there are infinitely many equivalent coordinate systems (like rotating a map)
- PLT identification: we constrain the loading matrix to have a specific shape (Positive Lower Triangular)
  ```
  β[first bill, dim 2] = 0         (rotation anchor)
  β[second bill, dim 2] > 0        (positive diagonal)
  ```
  This pins down the rotation so results are interpretable.

- The tiered quality gate:
  - Tier 1 (R-hat < 1.10): fully converged, use 2D Dim 1 as canonical ideology
  - Tier 2 (R-hat < 2.50, rank correlation > 0.70): point estimates credible, flag uncertainty
  - Tier 3 (failed): fall back to 1D model
- **Kansas example:** 79th Senate — 2D reveals the establishment-maverick dimension that 1D misses
- **Codebase:** `analysis/06_irt_2d/irt_2d.py`

#### Chapter 5: Partial Pooling: Hierarchical Models and Party Structure

- **Hook:** "If you know a legislator is a Republican, that tells you something about where they'll fall on the ideology spectrum — but it doesn't tell you everything. Hierarchical models use party information as a starting point, then let the data pull legislators away from their party average."
- The concept: partial pooling between "every legislator is the same" (complete pooling) and "every legislator is completely independent" (no pooling)
- **Analogy:** Predicting the height of a person. If you know nothing, guess the population average. If you know they're a professional basketball player, start with the team average and adjust. Partial pooling is the statistical version of this common sense.

- The hierarchical IRT equation:

  **Plain English:** "Each legislator's ideology starts at their party's average, then adjusts based on their individual voting record."

  **Equations:**
  ```
  Party level:    μ_party ~ Normal(0, 2), ordered so D < R
  Within-party:   σ_within ~ HalfNormal(2.0)
  Legislator:     ξ = μ_party[party] + σ_within[party] · offset
  Offset:         offset ~ Normal(0, 1)    (non-centered parameterization)
  ```

- Non-centered parameterization:
  - **Analogy:** Instead of saying "the legislator is at position X" (centered), we say "the legislator is Y standard deviations from their party mean" (non-centered). Same thing mathematically, but the sampler navigates the space much more efficiently.
  - Why it matters: the "funnel geometry" problem — when party variance is small, centered parameterization creates narrow valleys the sampler can't navigate

- Shrinkage: legislators with few votes get pulled toward their party average more strongly
  - The Intraclass Correlation Coefficient (ICC):
    ```
    ICC = σ²_between / (σ²_between + σ²_within)
    ```
  - Kansas Senate: ICC ≈ 0.7 (strong party structure — parties explain 70% of ideological variation)

- The small-caucus problem: Kansas Senate Democrats (10–15 members) — too few for reliable hierarchical estimation
  - Adaptive priors: tighter HalfNormal(0.5) for groups < 20 members

- **Kansas example:** Shrinkage in the 91st House — a freshman Democrat with 50 votes vs. a veteran with 500
- **Codebase:** `analysis/07_hierarchical/hierarchical.py`, `analysis/07b_hierarchical_2d/hierarchical_2d.py`

#### Chapter 6: The Identification Zoo: Seven Strategies

- A deeper dive into each identification strategy with worked examples:
  1. **anchor-pca:** Find the legislator with the highest and lowest PCA score → fix as anchors
  2. **anchor-agreement:** For supermajority chambers where PCA fails — use contested-vote agreement patterns to find the most ideologically extreme legislators
  3. **sort-constraint:** Instead of hard anchors, use a soft ordering constraint (Republican mean > Democratic mean)
  4. **positive-beta:** Force all discrimination parameters positive (equivalent to assuming Yea = conservative)
  5. **hierarchical-prior:** Let the hierarchy itself identify the sign (D party mean < R party mean)
  6. **unconstrained:** No identification — useful for diagnostics (does the model find two clusters?)
  7. **external-prior:** Import ideology scores from Shor-McCarty or other external dataset

- When auto-detection picks each strategy:
  - Balanced chamber (40-60% majority) → anchor-pca
  - Supermajority (>65%) → anchor-agreement
  - External scores available → external-prior

- **Kansas example:** How identification strategy affects the 79th Senate ideal points
- **Codebase:** `docs/irt-identification-strategies.md`, `analysis/05_irt/irt.py`

#### Chapter 7: Canonical Ideal Points: Choosing the Best Score

- The routing problem: we now have up to 4 ideal point estimates per legislator (1D flat, 2D Dim 1, hierarchical 1D, hierarchical 2D Dim 1) — which one do we use?
- The canonical routing system:
  1. **Preference order:** Hierarchical 2D Dim 1 → Flat 2D Dim 1 → 1D IRT
  2. **Quality gates:** Only use a model if it passes its tier threshold
  3. **Horseshoe override:** In horseshoe-affected chambers, 2D Dim 1 is preferred (it separates ideology from establishment loyalty)

- The tiered quality gate (detailed):
  - **Tier 1:** R-hat < 1.10, ESS > 100 → use results fully, narrow credible intervals
  - **Tier 2:** R-hat < 2.50, party separation d > 1.5, rank-corr with PC1 > 0.70 → use point estimates, flag wide intervals
  - **Tier 3:** Failed → fall back to simpler model

- **Kansas example:** The routing decision for each chamber-session in the 91st Legislature
- **Codebase:** `analysis/canonical_ideal_points.py`, `docs/canonical-ideal-points.md`

---

### Volume 5 — Checking Our Work

#### Chapter 1: What Does Validation Mean?

- **Hook:** "A model that can't be checked is a model that can't be trusted. But what does it mean to 'check' an ideology score?"
- Internal validation: does the model fit its own data well?
- External validation: do our scores agree with independent measurements?
- Cross-validation: do our scores generalize to unseen data?
- The philosophical point: ideology is unobserved — there's no "ground truth" to compare against, only consistency and convergence of evidence

#### Chapter 2: Posterior Predictive Checks: Can the Model Predict Its Own Data?

- The logic: if the model is good, simulated data from the model should look like real data
- **Analogy:** If your recipe is correct, the cake you bake should look like the picture in the cookbook
- PPC statistics:
  - **Accuracy:** What fraction of votes does the model predict correctly?
  - **GMP (Geometric Mean Probability):** The average confidence of correct predictions (higher = better calibrated)
  - **APRE:** Improvement over the naive "always predict Yea" baseline

  **Equation (GMP):**
  ```
  GMP = exp( (1/N) · Σ log P(observed vote | model) )
  ```

  **Plain English:** "Take the model's confidence for each vote, average them (on a log scale), and convert back. Higher is better."

- Yen's Q3 (local independence check):

  **Plain English:** "After accounting for ideology, do any pairs of bills still have correlated residuals? If so, the model is missing something."

  **Equation:**
  ```
  Q3(j,k) = correlation(residual_j, residual_k)
  ```

  Threshold: |Q3| > 0.2 flags local dependence

- LOO-CV (Leave-One-Out Cross-Validation):
  - PSIS-LOO: an efficient approximation that doesn't require re-fitting the model for every observation
  - Stacking weights: when comparing models, LOO tells us how much weight to give each one

- **Kansas example:** PPC results for 1D vs. 2D IRT on the 91st House
- **Codebase:** `analysis/08_ppc/ppc.py`

#### Chapter 3: The Gold Standard: Shor-McCarty External Validation

- Boris Shor and Nolan McCarty's state legislator ideal points — the most widely cited external benchmark
- Method: bridge legislators (who serve in both state and national legislatures) + NPAT surveys to place all state legislators on a common scale
- Our validation: match Kansas legislators by name, compute Pearson correlation
  - House: r = 0.981 (near-perfect agreement)
  - Senate: r = 0.929 (strong, slightly lower due to smaller N)
- What discrepancies tell us: top 5 outliers analyzed individually
- Limitation: Shor-McCarty covers 2011–2020 only (career-averaged, not session-level)
- **Codebase:** `analysis/17_external_validation/external_validation.py`

#### Chapter 4: W-NOMINATE: Comparing to the Field Standard

- W-NOMINATE: the most used method in political science for measuring legislative ideology (Poole & Rosenthal)
- How it differs from Bayesian IRT:
  - Maximum likelihood (not Bayesian) — single point estimate, no uncertainty intervals
  - 2D by default, with spatial voting model
  - Different identification: polarity defined by legislator at PC1 extreme
- Our comparison: compute W-NOMINATE via R subprocess, correlate Dim 1 with our IRT
- Why we prefer Bayesian IRT: uncertainty quantification, flexible identification, hierarchical extension
- **Codebase:** `analysis/16_wnominate/wnominate.py`

#### Chapter 5: Quality Gates: Automatic Trust Levels

- MCMC convergence diagnostics:
  - **R-hat:** Are different chains (independent runs) giving the same answer? < 1.01 = good.
  - **ESS (Effective Sample Size):** After accounting for autocorrelation, how many independent samples do we really have? > 400 = good.
  - **Divergent transitions:** Did the sampler hit a region of the posterior it couldn't navigate? < 10 = acceptable.

  **Analogy (R-hat):** Four hikers start from different trailheads to find the highest peak. If they all converge on the same peak, we're confident it's the real summit. If they end up on different peaks, we can't trust the result.

- The three-tier quality gate:
  - Tier 1: converged → full trust
  - Tier 2: partially converged → trust the ranking, not the intervals
  - Tier 3: failed → fall back to simpler model
- How quality gates propagate through the pipeline: a Tier 3 result in Phase 05 changes how Phase 24 (Synthesis) reports

---

### Volume 6 — Finding Patterns

#### Chapter 1: Clustering: Are There Discrete Factions?

- **Hook:** "Are there two parties, or twenty? Is the legislature split into neat camps, or is ideology a continuous spectrum?"
- The big finding: k=2 is optimal (party split), and within-party variation is continuous, not factional
- Five clustering methods (for robustness):
  1. Hierarchical (average linkage on Kappa distance)
  2. K-means (on ideal points)
  3. Gaussian Mixture Models (with uncertainty weighting)
  4. Spectral (on Kappa affinity matrix)
  5. HDBSCAN (density-based, no k required)
- Silhouette scores: "how well does each legislator fit their assigned cluster?"

  **Equation:**
  ```
  silhouette(i) = (b(i) − a(i)) / max(a(i), b(i))
  ```
  where a(i) = average distance to own cluster, b(i) = average distance to nearest other cluster

  **Plain English:** "How much closer is this legislator to their own group than to the closest other group?"

- Within-party subclustering: House Republicans (k=6 optimal, but silhouette only 0.605 — weakly structured)
- **Analogy:** Looking for clusters of stars in the night sky — the two brightest (parties) are obvious, but within each, the stars blend into a continuous Milky Way
- **Codebase:** `analysis/09_clustering/clustering.py`

#### Chapter 2: Latent Class Analysis: Testing for Hidden Groups

- LCA: a model-based alternative to clustering
- Bernoulli mixture: each "class" has its own probability of voting Yea on each bill
- Model selection via BIC (Bayesian Information Criterion): penalizes complexity
- The Salsa effect: when K > 2, are the extra classes really different types — or just "mild, medium, hot" versions of the same thing?

  **Detection:** Spearman correlation between class profiles > 0.80 → quantitative grading, not qualitative distinction

- **Kansas example:** K=2 confirmed; K=3 shows Salsa effect (moderate R, strong R, Democrats)
- **Codebase:** `analysis/10_lca/lca.py`

#### Chapter 3: Co-Voting Networks: Who Votes Together?

- **Analogy:** "Think of each legislator as a person at a party. We draw a line between two people if they vote together often enough. The resulting web reveals who clusters together and who bridges different groups."
- Edge construction: Kappa agreement > 0.40 ("substantial" on Landis-Koch scale)
- Leiden community detection (replaces Louvain — more accurate, no resolution limit)
- Centrality measures:
  - **Betweenness:** How many shortest paths pass through this legislator? High betweenness = bridge between groups.
  - **Eigenvector:** How connected is this legislator to other well-connected legislators? High = influential.
  - **Degree:** How many connections? High = agreeable.
- Backbone extraction: disparity filter keeps only statistically significant edges
- Party modularity (Waugh et al.): single-number polarization metric

  **Equation:**
  ```
  Q = (1/2m) Σ [A(i,j) − k(i)k(j)/2m] · δ(c_i, c_j)
  ```

  **Plain English:** "How much more do legislators vote with their own party than you'd expect in a random network?"

- **Codebase:** `analysis/11_network/network.py`

#### Chapter 4: Bipartite Networks: Bills That Bridge the Aisle

- Two-mode network: legislators on one side, bills on the other
- Yea-only edges (field standard: co-support, not co-opposition)
- BiCM backbone: null model that preserves each legislator's vote count and each bill's support count
- Newman projection: project the bipartite network into a one-mode legislator network, with discount for high-activity bills
- Bill polarization: party-stratified Rice Index per bill
- Bridge bills: high connectivity, low polarization — bills where both parties voted Yea
- **Codebase:** `analysis/12_bipartite/bipartite.py`

#### Chapter 5: Classical Indices: Rice, Party Unity, and the Maverick Score

- **Rice Index:** How unified is each party on each vote?

  **Equation:**
  ```
  Rice = |Yea − Nay| / (Yea + Nay)
  ```

  **Plain English:** "If 80 out of 100 Republicans vote Yea and 20 vote Nay, Rice = |80−20|/100 = 0.60."

- **Party Unity:** What fraction of "party votes" does a legislator vote with their party?
- **Effective Number of Parties (ENP):**

  **Equation:**
  ```
  ENP = 1 / Σ(s_i²)
  ```
  where s_i is party i's seat share. Two equal parties → ENP = 2.0.

- **Maverick Score:** Defection rate, optionally weighted by constituency margin
- **Carey UNITY:** Penalizes absences (unlike standard party unity, which ignores them)
- **Kansas example:** Party unity trends across 2011–2026
- **Codebase:** `analysis/13_indices/indices.py`

#### Chapter 6: Empirical Bayes: Shrinkage Estimates of Party Loyalty

- The problem: a legislator with 10 party votes who defects once (90% loyalty) looks disloyal, but it might just be noise
- Empirical Bayes: estimate a "population" loyalty rate for each party, then shrink individual estimates toward it
- **Analogy:** A baseball player with 3 hits in 5 at-bats (0.600 average). Do you really think they'll hit .600 all season? No — you "shrink" toward the league average (~.260). Same idea.

- The Beta-Binomial model:

  **Equations:**
  ```
  Prior:      θ ~ Beta(α, β)          ← estimated from all legislators in the party
  Likelihood: k ~ Binomial(n, θ)      ← k party-line votes out of n party votes
  Posterior:  θ | k ~ Beta(α+k, β+n-k) ← updated loyalty estimate
  ```

  **Plain English:** "Start with the party average, update with individual data. Legislators with few votes stay close to the average; veterans with hundreds of votes move toward their actual rate."

- Shrinkage factor: (α+β) / (α+β+n) — decreases as more data accumulates
- **Kansas example:** Shrinkage for a 10-vote freshman vs. a 500-vote veteran
- **Codebase:** `analysis/14_betabinom/betabinom.py`

---

### Volume 7 — Prediction and Text

#### Chapter 1: Predicting Votes: What XGBoost Learns

- **Hook:** "If you know a legislator's ideology, the bill's discrimination, and a few other things — can you predict how they'll vote?"
- XGBoost: gradient-boosted decision trees (the workhorse of prediction competitions)
- **Analogy:** A committee of 200 very simple decision-makers. Each one makes a small correction to the previous group's prediction. Together, they're remarkably accurate.
- Features: ideal point, bill discrimination, party, loyalty rate, network centrality, vote type, session day
- The feature interaction: ξ × β (ideology × discrimination) — the most predictive single feature
- What's excluded and why: vote margin (leaks the target), bill difficulty α (highly correlated with passage)
- Results: AUC = 0.98, but with a caveat — IRT features come from the same votes (explanatory, not truly predictive)
- SHAP analysis: which features matter most for each prediction?
- **Codebase:** `analysis/15_prediction/prediction.py`

#### Chapter 2: Bill Passage: Can We Forecast Outcomes?

- A different prediction task: will the bill pass? (~500–600 roll calls per chamber)
- Features: bill discrimination, vote type, bill prefix (SB/HB), session day, veto override flag
- NLP features: NMF topics on bill titles (no text leakage — titles exist before the vote)
- Temporal validation: train on first 70% of session, test on last 30%
- **Kansas example:** Which bill types are hardest to predict?

#### Chapter 3: Topic Modeling: What Are the Bills About?

- BERTopic: a modern topic modeling approach
  1. Embed bill text (bge-small, 384 dimensions)
  2. Reduce dimensions (UMAP)
  3. Cluster (HDBSCAN)
  4. Extract keywords (c-TF-IDF)
- **Analogy:** Sorting a pile of 800 bills into topic folders — but instead of reading each one, you measure its "meaning fingerprint" and group similar fingerprints together
- Legislative stopwords: "shall," "statute," "pursuant" — words that appear everywhere and mean nothing for topic identification
- CAP classification: an alternative using the 20-category Comparative Agendas Project taxonomy (Claude API)
- **Kansas example:** Topic distribution for the 91st Legislature
- **Codebase:** `analysis/20_bill_text/bill_text.py`

#### Chapter 4: Text-Based Ideology and Issue-Specific Scores

- Text-based ideal points: derive ideology from the *content* of bills a legislator supports
  1. For each legislator, weight bill embeddings by their votes (+1 Yea, −1 Nay)
  2. PCA on the weighted profiles
  3. Compare to IRT ideal points
- Issue-specific IRT: run separate IRT models for each policy topic
  - "How conservative is this legislator on education?" vs. "...on taxes?"
  - Cross-topic correlations reveal which policy areas are ideologically aligned vs. cross-cutting
- **Codebase:** `analysis/21_tbip/tbip.py`, `analysis/22_issue_irt/issue_irt.py`

#### Chapter 5: Model Legislation: Detecting Copy-Paste Policy

- **Hook:** "Sometimes a Kansas bill looks suspiciously similar to a bill in Oklahoma — or to an ALEC template. Can we detect that automatically?"
- Method: cosine similarity on bill embeddings, confirmed with n-gram overlap
- ALEC corpus: 1,061 model policies
- Cross-state comparison: Missouri, Oklahoma, Nebraska, Colorado (via OpenStates API)
- Tiers: near-identical (≥ 0.95), strong match (≥ 0.85), related (≥ 0.70)
- **Caveat:** Similarity is not causation — common legal language can inflate scores
- **Codebase:** `analysis/23_model_legislation/model_legislation.py`

---

### Volume 8 — Change Over Time

#### Chapter 1: Time Series: Detecting When the Legislature Shifts

- **Hook:** "Is the Kansas Legislature more polarized today than it was in 2011? If so, when did the shift happen?"
- Rolling PCA: track the first principal component over time (75-vote windows)
- Rice Index time series: party cohesion over the course of a session
- PELT changepoint detection:
  - **Analogy:** Watching a heart monitor — the line is flat, then suddenly jumps. The changepoint is the moment it jumped.

  **Plain English:** "PELT scans the weekly party cohesion data and finds the moments where the average abruptly shifts."

- CROPS penalty selection: automated method to find the right sensitivity
- Bai-Perron confidence intervals: formal uncertainty on when the break occurred
- **Kansas example:** Detected changepoints in the 91st Legislature (e.g., after a leadership challenge)
- **Codebase:** `analysis/19_tsa/tsa.py`

#### Chapter 2: Dynamic Ideal Points: The Martin-Quinn Model

- **Hook:** "A legislator's ideology isn't frozen. People change. Can we track how?"
- The state-space model:

  **Plain English:** "Each legislator's ideology today is their ideology yesterday, plus a small random step."

  **Equations:**
  ```
  ξ[t] = ξ[t-1] + τ · innovation[t-1]
  innovation ~ Normal(0, 1)
  τ ~ HalfNormal(σ)     ← controls how fast ideology can change
  ```

- The random walk: a coin flip that takes one step left or right at each biennium
  - Small τ → ideology changes slowly (strong persistence)
  - Large τ → ideology can shift dramatically between sessions
  - Estimated from data, not assumed

- Bridge legislators: lawmakers who serve across multiple bienniums anchor the cross-biennium scale
- Polarization decomposition:

  **Equation:**
  ```
  Total shift = Conversion effect + Replacement effect
  ```

  **Plain English:** "Did the party move because existing members changed their minds (conversion)? Or because moderates retired and were replaced by ideologues (replacement)?"

- **Kansas example:** Dynamic ideal point trajectories for 5 long-serving legislators
- **Codebase:** `analysis/27_dynamic_irt/dynamic_irt.py`

#### Chapter 3: Cross-Session Alignment: Putting Sessions on the Same Scale

- The comparability problem: each biennium's IRT is estimated independently, so the scales differ
- Affine transformation:

  **Equation:**
  ```
  ξ_aligned = A · ξ_original + B
  ```

  **Plain English:** "Stretch and shift one session's scale to match another's, using legislators who appear in both."

- Robust fitting: trim the 10% most extreme residuals (outliers from real ideology change, not scale mismatch)
- Overlap requirements: at least 20 shared legislators
- **Codebase:** `analysis/26_cross_session/cross_session.py`

#### Chapter 4: Conversion vs. Replacement: Why Does Ideology Change?

- Decomposition method:
  - Returning legislators: compare their ideal point across sessions
  - Departing legislators: their "lost" ideology
  - New legislators: their "added" ideology
  - Total shift = (new cohort average − departing cohort average) × cohort size proportion + (returning legislators' shift)
- KS test: formal statistical test for whether the ideology distribution of new members differs from departing members
- **Kansas example:** The 87th → 88th Legislature transition — did the moderate Republican exodus change the party?

#### Chapter 5: Fifteen Years of Kansas Politics (2011–2026)

- A narrative synthesis of the temporal analysis results
- The Brownback era (84th–86th): tax experiment, Republican factionalism
- The moderate resurgence (87th–88th): suburban shifts
- The current era (89th–91st): polarization trends
- Lessons: what the data reveals about Kansas political dynamics
- **Note:** This chapter is descriptive, not prescriptive — we report the numbers, not our opinions

---

### Volume 9 — Telling the Story

#### Chapter 1: The Report System: From Data to Narrative

- The report architecture: HTML sections (text, plots, tables, interactive)
- RunContext: the orchestrator that tracks which phases ran and what they found
- Auto-generated primers: Purpose, Method, Inputs, Outputs, Caveats
- Theming: partisan colors (R = red, D = blue), responsive layout, accessibility (WCAG 2.1)
- **Codebase:** `analysis/report.py`

#### Chapter 2: Synthesis: The Session Story

- The 24th phase: automated narrative generation
- Notable legislator detection (algorithmic):
  - **Maverick:** Lowest party unity in their caucus
  - **Bridge-builder:** High network betweenness + centrist ideology
  - **Metric paradox:** Large gap between ideology rank and loyalty rank
- Horseshoe-aware narratives: different language when the horseshoe effect is present
- Graceful degradation: sections skip when no suitable candidate (29–32 sections per report)
- **Codebase:** `analysis/24_synthesis/synthesis.py`

#### Chapter 3: Legislator Profiles: Individual Deep Dives

- Per-legislator scorecards: 6 normalized (0–1) metrics
- Bill type breakdown: partisan vs. routine voting patterns
- Top defections: sorted by party margin closeness
- Voting neighbors: most-similar and most-different colleagues
- Surprising votes: model confidence vs. actual outcome
- **Codebase:** `analysis/25_profiles/profiles.py`

#### Chapter 4: How to Read a Tallgrass Report

- A guided tour of a real report
- What each section means and why it's there
- Common patterns to look for
- Caveats and limitations to keep in mind
- How to request custom analysis or contribute to the project

---

## Production Notes

### Estimated Scope
- **Per chapter:** 2,000–4,000 words (8–16 pages)
- **Per volume:** 10,000–28,000 words (40–110 pages)
- **Total series:** ~120,000–180,000 words (480–720 pages)
- **Appendices:** ~10,000 words shared across volumes

### File Organization
```
docs/guide/
├── README.md                    ← this file (series index)
├── volume-01-big-picture/
│   ├── README.md                ← volume introduction + chapter list
│   ├── ch01-what-is-tallgrass.md
│   ├── ch02-kansas-legislature.md
│   ├── ch03-roll-call-votes.md
│   └── ch04-the-pipeline.md
├── volume-02-gathering-data/
│   ├── README.md
│   ├── ch01-web-scraping.md
│   ├── ...
│   └── ch05-bill-text-database.md
├── ...
├── volume-09-telling-the-story/
│   ├── README.md
│   ├── ...
│   └── ch04-how-to-read-report.md
└── appendices/
    ├── glossary.md
    ├── equation-reference.md
    ├── phases-at-a-glance.md
    ├── data-dictionary.md
    └── further-reading.md
```

### Writing Order (Recommended)
1. **Volume 1** (The Big Picture) — sets up vocabulary and motivation
2. **Volume 4, Chapters 1–2** (IRT core) — the mathematical foundation everything else references
3. **Volume 3** (First Look) — PCA/EDA that feeds into IRT
4. **Volume 4, Chapters 3–7** (remaining IRT) — identification, 2D, hierarchical
5. **Volume 5** (Validation) — readers need IRT before they can understand validation
6. **Volume 2** (Gathering Data) — can be written in parallel with anything
7. **Volume 6** (Finding Patterns) — needs IRT and EDA
8. **Volume 7** (Prediction and Text) — needs everything before it
9. **Volume 8** (Change Over Time) — needs IRT and validation
10. **Volume 9** (Telling the Story) — last, wraps everything up
11. **Appendices** — accumulated throughout, finalized at end

### Style Guide
- **Voice:** Second person ("you") when walking through examples, third person for descriptions
- **Tense:** Present tense for methods ("PCA finds..."), past tense for Kansas-specific results ("The 91st House showed...")
- **Math notation:** LaTeX-style in fenced blocks, but always preceded by plain-English explanation
- **Code references:** File paths only (e.g., `analysis/05_irt/irt.py:line 234`), not code blocks
- **Figures:** Referenced by description ("Figure 3.2: PCA scree plot for the 91st House"), created during writing
- **Cross-references:** Between chapters use format "see Vol. 4, Ch. 2" or hyperlinks in digital version
