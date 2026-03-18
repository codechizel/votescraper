# Chapter 4: How to Read a Tallgrass Report

> *You've opened a Tallgrass report in your browser. It's 30 sections long. Where do you start, what matters, and what can you safely skip?*

---

## The Two Entry Points

There are two ways into a Tallgrass analysis:

**The dashboard** (`index.html`). A sidebar lists every analysis phase in pipeline order. Click a phase to load its report in the main panel. This is the entry point for someone who wants to browse — maybe you want the IRT results first, or maybe you want to jump straight to the network analysis. The dashboard is a table of contents for the entire pipeline.

**The synthesis report** (`24_synthesis_report.html`). A single narrative that draws from ten upstream phases and tells the session's story from beginning to end. This is the entry point for someone who wants the story — the key findings, the interesting legislators, the patterns that matter. If you read one thing, read this.

Most readers should start with the synthesis report. The phase-level reports are reference documents — you'll go to them when the synthesis raises a question you want to dig into.

## Anatomy of the Synthesis Report

### The First 30 Seconds: Key Findings

At the very top, before the table of contents, a highlighted box contains 2-4 bullet points. These are the report's headline findings — the equivalent of a newspaper's front-page summary:

- **The top maverick** and their defection rate
- **Bridge-builders** who connect across party lines
- **Paradox cases** where metrics disagree
- **The chamber size** and overall vote count

If you have 30 seconds, read this box. It tells you who the session's most interesting legislators are and what makes them stand out.

### The Table of Contents

A two-column numbered list of every section in the report. Each entry is a clickable link. The section count (29-32) might seem overwhelming, but you don't need to read them all — the report is designed for both sequential reading and targeted jumping.

### Section-by-Section Guide

Here's what each section type tells you and when you might want it:

**"What This Report Tells You" (Introduction)**
The opening narrative. Tells you how many votes were analyzed, how many were contested, and previews the three or four headline findings. Read this if you want the 2-minute version.

**Pipeline Summary (Infographic)**
Five boxes showing the flow from raw votes to final model accuracy. The numbers give you a feel for the scale of the data. The final box — the AUC score — tells you how predictable the legislature is: 0.98 means the model correctly predicts 98% of votes from ideology alone.

*What to look for:* The gap between "Total Votes" and "Contested Votes" tells you how many votes were unanimous or near-unanimous. In Kansas, this is typically 60-70% — most legislative business is noncontroversial.

**Party-Line Narrative**
A prose section summarizing the clustering, network, and prediction results in plain language. The core message is almost always the same: party affiliation is the dominant predictor of voting behavior. But the details vary — how strong is the party signal? Are there any cross-party edges in the network? Did clustering find subgroups beyond the two parties?

*What to look for:* If the narrative mentions k=3 or k=4 clusters, the party divide has internal structure worth exploring. If it says k=2 with near-perfect ARI match, the parties are monolithic.

**Cluster and UMAP Figures**
Visual confirmation of the party divide. The cluster plot colors legislators by IRT-derived groups; the UMAP plot projects them into 2D. In most sessions, you'll see two distinct blobs — Republicans and Democrats — with clear separation.

*What to look for:* Overlap between the blobs. If some legislators sit between the two parties, they're the centrists and potential swing votes. If the blobs are completely separated with empty space between them, the chamber is deeply polarized.

**Network Diagrams**
The co-voting network, where edges connect legislators who vote together frequently (Cohen's Kappa > 0.4). Communities are colored; notable legislators have glowing halos.

*What to look for:* Cross-party edges. If any lines connect a Republican node to a Democratic node, those legislators have unusually high agreement across party lines. In highly polarized sessions, you'll see two completely disconnected clusters.

**Dashboard Scatters**
The signature visualization. X-axis: ideology (liberal to conservative). Y-axis: party unity (0% to 100%). Point size: weighted maverick rate. Named annotations on notable legislators.

*What to look for:*
- **Vertical spread within a party** means internal disagreement — some members are loyal (top) and some are rebels (bottom), even at the same ideological position.
- **Horizontal spread within a party** means ideological diversity — moderates and extremists coexist.
- **Big dots** are the frequent rebels on close votes. Check their names.
- **Dots far from the party cloud** are the outliers. The narrative explains why.

**Mavericks Section**
Prose identifying the most rebellious legislator in each party in each chamber. For each maverick, you'll see their party unity score, how it compares to the caucus average, and a sentence explaining their pattern.

*What to look for:* The *magnitude* of the gap between the maverick's unity and the party average. A maverick at 72% in a party averaging 92% is modestly independent. A maverick at 55% is dramatically so.

**Profile Cards**
Horizontal bar charts for notable legislators (usually 2-5 of them). Six metrics on a 0-1 scale with party average markers.

*What to look for:* Bars that deviate dramatically from the party average marker. A short "Party Unity" bar tells you this legislator defects often. A tall "Network Influence" bar combined with centrist ideology tells you they're a bridge. The profile card gives you the at-a-glance shape of a legislator's behavior.

**Forest Plots**
Every legislator's IRT ideal point with 95% credible intervals, sorted by ideology. Notable legislators are called out in the caption.

*What to look for:* Overlapping credible intervals between the parties. If the most liberal Republican's interval overlaps with the most conservative Democrat's interval, those legislators are ideologically indistinguishable — they occupy the contested middle ground.

**Paradox Section**
If detected: a narrative explaining a legislator whose ideology rank and loyalty rank disagree. The three-bar visualization makes the contradiction visible.

*What to look for:* The *direction* of the paradox. "Defects rightward" means this legislator is more conservative than their loyalty suggests — they vote with Democrats on some issues despite being ideologically conservative. "Defects toward the center" means their defections pull them away from their party's extreme.

**Veto Override Analysis**
Rice indices on veto override votes, with context about the two-thirds supermajority requirement.

*What to look for:* High Rice indices on both sides (> 0.80) mean the parties circled the wagons — overrides are pure partisan exercises. Mixed or lower indices mean some cross-party coalition existed on override votes.

**Surprising Votes Table**
The 20 votes where the prediction model was most wrong. Interactive — you can sort by surprise score, filter by legislator name, or search for a specific bill.

*What to look for:* Clusters of surprising votes around specific legislators or specific policy areas. If 8 of the 20 most surprising votes involve the same 3 legislators, those legislators have behavioral patterns the model doesn't capture. If 6 involve education bills, education is the issue where party discipline breaks down.

**Full Scorecard**
An interactive table with every legislator and every metric, searchable and sortable. This is the reference section — the phone book at the back of the report.

*What to look for:* Sort by a specific column to find the extremes. Sort by "Party Unity" ascending to find all the mavericks. Sort by "Betweenness" descending to find the most structurally central legislators. Use the search box to find a specific legislator by name.

## Reading a Phase Report

Each of the 28 analysis phases produces its own report, accessible through the dashboard. These are more technical than the synthesis — they're designed for readers who want methodological detail.

### Common Elements

Every phase report has:

- **Key Findings** at the top (the 2-4 most important results)
- **Numbered sections** with a table of contents
- **Captioned figures** with alt text
- **A methodology note** explaining the specific method used
- **Download links** for raw data (CSVs, parquets)
- **A timestamp and git hash** in the footer

### The IRT Report (Phase 05)

The most important technical report. Shows:
- Forest plots of credible intervals (the visual that launched a thousand political arguments)
- Convergence diagnostics (R-hat, ESS, divergences)
- Bill discrimination histogram (which bills are most partisan)
- Posterior predictive checks (does the model reproduce the data?)

*When to reference:* When you want to know a specific legislator's ideology score, its uncertainty, or whether the model converged well.

### The Network Report (Phase 11)

Shows:
- Interactive network visualization (hover for details)
- Community detection results
- Centrality rankings
- Bipartite network (legislators × bills)

*When to reference:* When you want to understand coalition structure beyond party labels, or when the synthesis mentions a bridge-builder and you want to see their position in the network.

### The Prediction Report (Phase 15)

Shows:
- ROC curves and AUC scores per chamber
- SHAP feature importance (what drives predictions)
- Per-legislator accuracy rankings
- Calibration curves (is the model's confidence well-calibrated?)
- The full list of surprising votes

*When to reference:* When you want to understand which features drive vote prediction, or when you want the complete list of model errors (not just the top 20 in synthesis).

## Reading a Legislator Profile

If the pipeline ran Phase 25, individual legislator profiles are available. Each profile is a self-contained section within the profiles report.

### The Scorecard

Start here. Six bars, each on a 0-1 scale, with a party average marker. The scorecard is the profile's executive summary — 10 seconds of looking at bar lengths tells you the shape of this legislator's behavior.

### The Bill Type Bars

Two groups of bars: partisan bills and routine bills. Compare the legislator's Yea rate to the party average on each. The gap on partisan bills is more informative than the gap on routine bills — everyone votes similarly on easy bills.

### The Defection List

The 15 most consequential defections. Skim the bill titles to see if a pattern emerges — are the defections concentrated in a policy area? If you see "SB 123 - relating to education" and "HB 456 - school funding formula" among the top defections, education is this legislator's independence zone.

### The Neighbors

Two panels: closest (votes alike) and most different (votes opposite). If any cross-party legislators appear in the "closest" panel, that's a strong signal of genuine bipartisanship — not just occasional defections, but a sustained pattern of agreement with the other side.

### The Surprising Votes

The model's biggest prediction errors for this legislator. Look for the bills with the highest surprise scores — those are the votes that deviate most from what the legislator's overall ideology would predict.

## Common Patterns to Look For

### The Disciplined Legislature

- Both parties have Rice indices above 0.85
- The dashboard scatter shows tight, separated clusters
- k=2 clustering matches parties with ARI near 1.0
- No mavericks detected (all unity above 0.95)
- Prediction AUC above 0.97

**Interpretation:** Party is everything. The leadership has strong control, or the issues on the agenda are genuinely partisan. Individual legislators have little room to deviate.

### The Factional Legislature

- One party shows internal spread on the dashboard scatter
- k=3 or k=4 clustering finds a meaningful subgroup
- The horseshoe effect appears in PCA (Volume 3)
- Mavericks are detected, and their defection patterns have policy themes

**Interpretation:** One party has an internal divide — possibly moderate vs. conservative, rural vs. suburban, or establishment vs. insurgent. The IRT scores still separate parties, but within-party variation tells the more interesting story.

### The Polarization Shift

- Cross-session analysis (Volume 8) shows party means diverging
- Replacement effects dominate conversion effects
- The 84th-to-85th transition shows the largest replacement effect
- Later transitions show stability

**Interpretation:** The legislature polarized through personnel change (who got elected) rather than behavioral change (how incumbents vote). Once the roster stabilized, the polarization level held.

### The Swing Vote

- A legislator appears as both a maverick and a bridge-builder
- Their voting neighbors include members of both parties
- Their defection list shows they break from their party on a specific issue area
- Their prediction accuracy is lower than average (the model can't predict when they'll defect)

**Interpretation:** This legislator is the true swing vote — ideologically near the center, structurally connected to both parties, and unpredictable on the issues that matter most. On override votes (which require supermajority), this legislator's vote may be decisive.

## What to Be Cautious About

### The 84th Legislature Data Gap

The 84th Legislature (2011-2012) has approximately 30% committee-of-the-whole votes — floor votes where individual legislator positions aren't recorded. This means:
- IRT credible intervals are wider for legislators who served only the 84th
- Prediction accuracy is lower (fewer training examples)
- Network centrality may be inflated (fewer edges = more structural importance per edge)

Any claim about the 84th should carry wider error bars than claims about later sessions.

### The Horseshoe Effect in Senate Sessions

In 7 of 14 Senate sessions, the first principal component captures intra-Republican factionalism rather than the party divide (documented in Volume 3). When this happens:
- PCA scores should not be interpreted as liberal-conservative
- IRT scores are corrected via the seven-gate quality system (ADR-0118) and are generally reliable
- The synthesis report adds a warning banner explaining the issue

If you see a horseshoe warning, the IRT scores are still trustworthy — the warning is about PCA, not IRT.

### Maverick ≠ Moderate

A maverick has low party unity — they vote against their party frequently. But that doesn't necessarily mean they're moderate. A legislator could be *more* extreme than their party (defecting because the party isn't conservative enough) or could defect on a specific issue (education, guns) while being perfectly loyal on everything else.

Always check the maverick's IRT ideology score and their defection list before assuming they're a centrist. The scorecard and the bill type breakdown together reveal whether the independence is ideological (moderate on the spectrum) or issue-specific (extreme on the spectrum but defecting on particular topics).

### Prediction Accuracy ≠ Transparency

A legislator with 99% prediction accuracy isn't "transparent" — they're predictable *given their ideology*. The model knows their position and can predict their votes. But their position itself might be opaque to the public. High prediction accuracy means the IRT model understands them; it doesn't mean voters do.

Conversely, a legislator with 82% accuracy isn't "secretive" — they just have voting patterns that don't align cleanly with a single ideological dimension. They might be cross-pressured on specific issues, or they might respond to local district concerns that don't map onto the liberal-conservative spectrum.

### Statistical Significance vs. Political Significance

The paradox detection requires a 50-percentile-point rank gap. The maverick detection requires sub-0.95 unity. The bridge-builder detection requires centrality near the cross-party midpoint. These are reasonable thresholds, but they're still arbitrary lines drawn on continuous distributions.

A legislator just below the maverick threshold (0.94 unity) is behaviorally similar to one just above it (0.96). The thresholds determine who gets named in the report, not who is "really" independent. If you're investigating a specific legislator, look at their actual numbers — not just whether they crossed a detection threshold.

## How to Contribute

Tallgrass is open-source (MIT license). If you find an error, want to add a feature, or have a question about methodology:

- **Source code:** The project is on GitHub. The `analysis/` directory contains all 28 phases; `src/tallgrass/` contains the scraper.
- **Issue tracker:** Report bugs or request features through GitHub Issues.
- **Data requests:** If you want analysis for a specific session, legislator, or policy area that isn't covered by the standard pipeline, open an issue describing what you need.
- **ADRs:** Every non-obvious design decision is documented in an Architecture Decision Record (`docs/adr/`). If you're wondering "why did they do it that way?" the ADR probably explains it.

The project covers eight bienniums of Kansas legislative data (2011-2026). The scraper can be extended to other state legislatures that publish roll call votes in similar formats, but that work hasn't been done yet.

---

## Key Takeaway

Start with the synthesis report's key findings box for the 30-second summary. Use the dashboard scatter to see the full legislature at a glance. Read the maverick and bridge-builder narratives for the session's characters. Use the interactive scorecard to look up any specific legislator. Dive into phase reports when you want methodological detail, and into legislator profiles when you want the full story on an individual. Be cautious about 84th Legislature data quality, horseshoe-affected Senate sessions, and the difference between statistical detection thresholds and political significance.

---

*Terms introduced: (this chapter introduces no new technical terms — it applies the vocabulary from Chapters 1-3 to practical report reading)*

*Previous: [Legislator Profiles: Individual Deep Dives](ch03-profiles.md)*
