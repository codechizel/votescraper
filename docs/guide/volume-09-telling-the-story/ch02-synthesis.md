# Chapter 2: Synthesis: The Session Story

> *Ten analysis phases produced ideology scores, network graphs, prediction accuracy, clustering assignments, and maverick rates. Phase 24 reads all of them, finds the session's most interesting legislators, and writes a report a nonstatistician can understand.*

---

## The Problem Synthesis Solves

By the time the pipeline reaches Phase 24, the session has been analyzed from every angle. Phase 05 estimated ideology. Phase 09 found clusters. Phase 11 built networks. Phase 13 computed party unity indices. Phase 15 predicted votes. Each phase produced its own report, its own tables, its own plots.

But no single phase has the whole picture. The IRT report tells you Senator Smith has an ideal point of +1.8, but doesn't mention that she's also the most central member in the co-voting network. The clustering report tells you that k=2 matches the party divide, but doesn't name the three legislators whose behavior doesn't fit any cluster cleanly. The prediction report tells you that the model's accuracy on Representative Jones is only 82%, but doesn't explain *why* — is Jones a maverick, a bridge-builder, or just unpredictable?

Synthesis answers the question: **"If you could only read one report about this legislative session, what should it say?"**

The answer is a 29-to-32-section HTML report designed for journalists, policymakers, and citizens — people who care about Kansas politics but don't speak IRT. No new computation happens in Phase 24. It takes the numbers that earlier phases computed and turns them into a story.

## The Ten Upstream Phases

Synthesis draws from ten analysis phases, loaded via `load_all_upstream()`:

| Phase | What It Provides | Key Data |
|-------|-----------------|----------|
| **01 EDA** | Vote counts, legislator counts | Manifest with totals and filtering stats |
| **02 PCA** | First two principal components | PC1, PC2 per legislator |
| **04 UMAP** | 2D nonlinear embedding | UMAP1, UMAP2 per legislator |
| **05 IRT** | Ideology scores | xi_mean, xi_sd per legislator |
| **07 Hierarchical** | Party-pooled ideology | hier_xi_mean, hier_xi_sd, shrinkage_pct |
| **09 Clustering** | Cluster assignments, loyalty rates | cluster_k2, cluster_k3, loyalty_rate |
| **11 Network** | Centrality measures | betweenness, eigenvector, pagerank, cross_party_fraction |
| **13 Indices** | Party unity, maverick rates | unity_score, maverick_rate, weighted_maverick |
| **14 Beta-Binomial** | Bayesian loyalty estimates | posterior_mean, ci_width, shrinkage |
| **15 Prediction** | Per-legislator accuracy, surprising votes | accuracy, n_correct, surprising_votes |

One subtlety: when Phase 06 (canonical routing) has run, synthesis prefers its output over raw Phase 05 scores. Canonical routing selects the best available IRT model — Hierarchical 2D Dimension 1 if it converged, otherwise Flat 2D Dimension 1, otherwise 1D IRT. This means the "IRT" column in synthesis already reflects the pipeline's quality gate (Volume 4, Chapter 6), not just the first model that ran.

**Codebase:** `analysis/24_synthesis/synthesis_data.py` (`load_all_upstream()`, `UPSTREAM_PHASES`)

## Building the Unified Legislator DataFrame

The ten phases produce separate parquet files. Synthesis joins them into a single DataFrame per chamber — one row per legislator, with every metric from every phase available as a column.

The join uses IRT ideal points as the **base table**. If IRT didn't produce results for a chamber (convergence failure, no data), synthesis skips that chamber entirely — there's no story to tell without ideology scores.

All other phases join via LEFT JOIN on `legislator_slug`. If a phase didn't run (say, the network analysis failed because the graph was disconnected), its columns are simply missing. No crash, no error — the report adapts by skipping sections that depend on that data. This is the **graceful degradation** principle.

The analogy: imagine building a patient's medical chart. The core is their identity and vital signs (IRT). If they had blood work done (clustering), those results go in the chart. If they didn't get an X-ray (network analysis), the chart just doesn't have an X-ray section — it doesn't throw out the whole chart.

After the joins, synthesis adds **percentile ranks** for three key metrics — ideology, network betweenness, and prediction accuracy. These transform raw numbers into "this legislator is in the 85th percentile" — a format that's meaningful without knowing the underlying scale.

The final DataFrame is sorted from left to right on the ideological spectrum: the most liberal legislator first, the most conservative last.

**Codebase:** `analysis/24_synthesis/synthesis_data.py` (`build_legislator_df()`)

## Finding the Session's Characters

Every legislative session has characters — legislators whose behavior stands out from the crowd. Synthesis finds them algorithmically, using three detection strategies. No names are hardcoded. The algorithm works the same way for every session, every chamber, every biennium.

### Mavericks: The Party Rebels

A **maverick** is the legislator with the lowest party unity score in their caucus. Party unity (from Phase 13, Volume 6) measures how often a member votes with their party on party-line votes. A legislator with 65% party unity votes against their own party on more than a third of contested votes — in a system where the average is typically 90%+, that's remarkable.

The detection logic:

1. Filter to one party in one chamber
2. Find the legislator with the lowest `unity_score`
3. If there's a tie, break it by `weighted_maverick` — the rate of defection on close votes (where the margin was tight, so the defection mattered more)
4. Skip detection entirely if all members have unity above 0.95 — in an extremely disciplined caucus, the "most rebellious" member at 96% isn't meaningfully a maverick. This threshold is a design choice to avoid false positives: when even the biggest rebel votes with the party 96% of the time, labeling them a "maverick" would mislead readers into thinking the caucus has real dissent

The algorithm detects mavericks in both the majority and minority parties. In Kansas, that typically means a Republican maverick (in the majority) and a Democratic maverick (in the minority), for each chamber — potentially four mavericks total.

The subtitle is data-driven: "Voted with Republicans on 67% of party-line votes — 25 points below the caucus average of 92%." Every number comes directly from the data.

**Codebase:** `analysis/24_synthesis/synthesis_detect.py` (`detect_chamber_maverick()`, `UNITY_SKIP_THRESHOLD = 0.95`)

### Bridge-Builders: Connecting Across the Aisle

A **bridge-builder** is the legislator best positioned to connect the two parties — someone who is both ideologically centrist and structurally central in the voting network.

Finding them requires two pieces of information working together:

**Step 1: Find the cross-party midpoint.** Compute the median ideal point for each party and take their average. In a chamber where the Republican median is +1.5 and the Democratic median is -1.2, the midpoint is +0.15. Legislators near this point are ideologically between the parties.

**Step 2: Filter to nearby candidates.** Only legislators within 1.0 standard deviations of the midpoint qualify. The 1.0 standard deviation boundary is a pragmatic choice — wide enough to capture genuine centrists, narrow enough to exclude partisans who occasionally cross the aisle. This prevents a far-right Republican from being named "bridge-builder" just because they have high betweenness centrality (which could happen in a disconnected network where one member happens to link two Republican factions).

**Step 3: Rank by centrality.** Among the qualifying candidates, the legislator with the highest network centrality wins. The specific centrality measure depends on the network structure:

- **Connected graph** (most sessions): Use betweenness centrality — the number of shortest paths between other legislators that pass through this person. High betweenness means they're a structural bridge.
- **Disconnected graph** (when parties form separate components): Use harmonic centrality combined with cross-party voting fraction — legislators who vote with the other party more often despite the network being split.

The analogy: imagine a university campus where most students only eat in their dorm cafeteria. The bridge-builder is the student who eats at *every* cafeteria and knows people in all of them — they're ideologically in the middle (they live in student housing between the liberal arts dorms and the engineering dorms) and structurally central (they're on the path between any two groups).

**Codebase:** `analysis/24_synthesis/synthesis_detect.py` (`detect_bridge_builder()`, `BRIDGE_SD_TOLERANCE = 1.0`)

### Metric Paradoxes: When the Numbers Disagree

A **metric paradox** is the most counterintuitive finding in a session. It identifies a legislator whose IRT ideology rank and clustering loyalty rank disagree dramatically — someone the IRT model considers an ideological outlier but the clustering model considers loyal (or vice versa).

How can that happen? The two metrics measure different things:

- **IRT ideology** measures where you are on the left-right spectrum, estimated from all your votes
- **Clustering loyalty** measures how consistently you vote with your assigned cluster (k=2, which matches the parties)

A legislator can be the most conservative member of the Republican caucus (extreme IRT rank) but have low clustering loyalty (frequently voting with Democrats on specific issues). The IRT sees their overall position; the clustering sees their vote-by-vote consistency.

The detection logic:

1. Compute each majority-party member's percentile rank on both IRT ideology and clustering loyalty
2. Find the legislator with the largest gap between the two ranks
3. Require the gap to exceed 0.50 (50 percentile points) — otherwise it's not surprising enough to highlight. This threshold is intentionally conservative — we'd rather miss a subtle paradox than highlight someone whose metrics barely disagree
4. Determine the direction: does this legislator defect rightward (more conservative than their loyalty suggests), leftward, or toward the center?

The paradox narrative explains the contradiction in plain language: "Representative Davis ranks 3rd most conservative among 85 Republicans by IRT, but only 67th in clustering loyalty — she defects rightward on bills where even most conservatives stay with the party."

**Codebase:** `analysis/24_synthesis/synthesis_detect.py` (`detect_metric_paradox()`, `PARADOX_RANK_GAP = 0.5`, `PARADOX_MIN_PARTY_SIZE = 5`)

### The Detection Pipeline

All three detections run through a single entry point, `detect_all()`, which returns a structured dictionary:

```python
{
    "mavericks": {chamber: NotableLegislator},
    "minority_mavericks": {chamber: NotableLegislator},
    "bridges": {chamber: NotableLegislator},
    "paradoxes": {slug: ParadoxCase},
    "profiles": {slug: NotableLegislator},     # deduplicated for deep-dive
    "annotations": {chamber: [slugs]},          # for scatter plot labels
}
```

The `profiles` key deduplicates: if the House maverick and the paradox case are the same person, they appear only once. The `annotations` key collects the slugs that should be labeled on the dashboard scatter plot — typically the maverick, bridge, and most extreme party member per chamber.

**Codebase:** `analysis/24_synthesis/synthesis_detect.py` (`detect_all()`)

## Naming the Coalitions

When the clustering analysis (Phase 09) finds k=2 groups that match the parties, the labels are obvious: "Republicans" and "Democrats." But when k=3 or k=4 produces subgroups, the clusters need meaningful names.

The **coalition labeler** auto-names clusters based on two signals:

**Party composition.** If more than 80% of a cluster belongs to one party, the name starts with that party: "Moderate Republicans," "Progressive Democrats." If the cluster is mixed, it gets a nonpartisan name: "Bipartisan Coalition."

**Ideological position.** Within a party-dominated cluster, the label depends on the cluster's median ideal point relative to the party mean:
- Within 0.3 standard deviations of the party mean → "Mainstream Republicans"
- More extreme than the party mean → "Conservative Republicans" (for R) or "Progressive Democrats" (for D)
- Less extreme than the party mean → "Moderate Republicans" (for R) or "Centrist Democrats" (for D)

The algorithm is fully automatic — it produces the same labels whether the data comes from the 84th Legislature or the 91st. No human judgment about who is "moderate" or "conservative" enters the process; the labels come from the data.

**Codebase:** `analysis/24_synthesis/coalition_labeler.py` (`label_coalitions()`, `DOMINANT_PARTY_THRESHOLD = 0.80`, `MODERATE_DISTANCE = 0.3`)

## The Visualizations

Synthesis creates four types of new plots (in addition to reusing 18 plots from upstream phases):

### The Pipeline Summary

An infographic showing the flow from raw data to final output. Five colored boxes connected by arrows:

```
Total Votes → Contested Votes → Party Votes → Optimal k → AUC
```

Each box contains a number and a plain-English explanation. The first box might say "14,232 Total Votes — 91st Legislature, 2025-2026." The last box says "0.98 AUC — Model predicts votes with high confidence." A reader who looks at nothing else in the report sees this one figure and understands the pipeline's scale and accuracy.

### The Dashboard Scatter

A scatter plot with IRT ideology on the x-axis and party unity on the y-axis. Every legislator is a dot, colored by party, sized by their weighted maverick rate (big dots = frequent rebels on close votes). Notable legislators are annotated with name labels.

This is the synthesis report's signature visualization. At a glance, it shows:
- The partisan gap (two clusters of dots, separated horizontally)
- Who the mavericks are (big dots far from 1.0 on the y-axis)
- Who the bridge-builders are (dots near the center with high unity)
- The overall discipline level (are most dots clustered near the top, or spread out?)

### Profile Cards

For each notable legislator, a horizontal bar chart showing six normalized metrics on a 0-1 scale. Each bar compares the legislator's value to the party average. A maverick's card might show a high ideological rank bar but a low party unity bar — the visual immediately conveys what makes them unusual.

### Paradox Figures

For each metric paradox, a three-bar chart showing the contradiction: IRT rank (full width, because they're extreme), clustering loyalty (short bar, because it's low), and party unity (somewhere in between). An annotation explains why the metrics disagree.

## Assembling the Report

The report builder receives all the data, detections, and plots, and assembles them into 29-32 sections. The exact count depends on what was detected — if there's no metric paradox, the paradox sections are skipped; if the beta-binomial phase didn't run, the Bayesian loyalty section is omitted.

### The Linear Layout

The default report flows like a magazine article:

1. **Key Findings** — 2-4 bullet points (the executive summary)
2. **Horseshoe Warnings** — If the PCA axis instability affected this session (see Volume 3)
3. **Introduction** — "What This Report Tells You": total votes, contested votes, headline findings
4. **Pipeline Summary** — The five-box infographic
5. **Party-Line Narrative** — Clustering found k=2 = parties; networks confirmed it; prediction accuracy is 98%
6. **Clusters Figure** — The IRT-colored cluster plot from Phase 09
7. **UMAP Landscapes** — The 2D embedding from Phase 04 (one per chamber)
8. **Network Diagrams** — Community detection from Phase 11 (one per chamber)
9. **Dashboard Scatters** — The new scatter plots (one per chamber)
10. **Mavericks Narrative** — Who they are, what their numbers say
11. **Profile Cards** — Bar charts for notable legislators
12. **Forest Plots** — IRT credible intervals from Phase 05 (with notable legislators called out)
13. **Paradox Narrative and Figure** — If a paradox was detected
14. **Veto Override Analysis** — Rice indices on override votes, with supermajority context
15. **Prediction Narrative** — AUC, error patterns, where the model struggles
16. **Surprising Votes Table** — The 20 votes the model got most wrong
17. **Bayesian Loyalty** — Shrinkage analysis from Phase 14 (if available)
18. **Methodology Note** — How the analysis was done (8 phases, key parameters)
19. **Full Scorecard** — Interactive table of all legislators, all metrics, searchable and sortable

### The Scrollytelling Layout

An alternative layout activated with `--scrolly`. Same content, different presentation: six narrative "chapters" with sticky visualizations that update as the reader scrolls. The chapters flow naturally:

1. "Kansas Legislature at a Glance" (pipeline summary)
2. "Party Is Everything" (clusters, networks, scatter)
3. "The Mavericks" (forest plot, profile cards)
4. "Who Bridges the Divide?" (network communities, bridge profiles)
5. "The Paradoxes" (if detected; metric comparison, heatmap)
6. "Can We Predict Their Votes?" (SHAP, accuracy, calibration)

The scrollytelling mode is designed for readers who want a guided narrative rather than a reference document. It sacrifices random access (you can't jump to a section) for a more engaging reading experience.

**Codebase:** `analysis/24_synthesis/synthesis_report.py` (`build_synthesis_report()`, `build_scrolly_synthesis_report()`)

## Horseshoe-Aware Narratives

In the seven Senate sessions where the PCA axis captures intra-Republican factionalism rather than the party divide (the horseshoe effect documented in Volume 3), the synthesis report adjusts its language. Instead of "PC1 separates Republicans from Democrats," it says something like "PC1 captures divisions *within* the Republican caucus" and adds a warning banner explaining the implication for downstream metrics.

This is handled by `horseshoe_warning_html()` from the shared phase utilities. The function checks the horseshoe status for each chamber and returns an HTML banner (or empty string if no horseshoe was detected). Synthesis embeds the banner near the top of the report.

**Codebase:** `analysis/phase_utils.py` (`horseshoe_warning_html()`, `load_horseshoe_status()`)

## What Can Go Wrong

### Missing Phases

If fewer than the expected ten upstream phases have output, synthesis degrades gracefully. The IRT scores are required — without them, there's no base table. Everything else is optional. A report built from only IRT, PCA, and EDA will have fewer sections but will still be valid and readable.

### No Notable Legislators Detected

In an extremely disciplined legislature (all unity scores above 0.95, no centrality variation, no paradoxes), the detection algorithms find no one to highlight. The report notes this: "Party discipline is exceptionally uniform — no individual legislator stands out as a maverick or bridge-builder." This is itself a finding.

### Canonical Routing Unavailable

If Phase 06 (canonical routing) didn't run, synthesis falls back to raw Phase 05 IRT scores. These are the same underlying model, just without the quality gate that selects the best available variant (1D vs. 2D Dimension 1 vs. Hierarchical). The report is still accurate; the IRT scores are just potentially from a less optimal model variant.

---

## Key Takeaway

Synthesis (Phase 24) aggregates ten upstream analyses into a unified legislator DataFrame per chamber, algorithmically detects three types of notable legislators (mavericks by party unity, bridge-builders by centrality near the ideological midpoint, paradoxes by metric disagreement), and assembles a 29-32 section narrative report designed for nontechnical audiences. No new computation occurs — synthesis is purely an aggregation, detection, and presentation layer that turns eight analytical perspectives into a single readable story.

---

*Terms introduced: synthesis, unified legislator DataFrame, canonical routing precedence, maverick detection, bridge-builder detection (cross-party midpoint, connected vs. disconnected centrality), metric paradox (rank gap), coalition labeler (dominant party threshold, moderate distance), dashboard scatter, profile card, pipeline summary, scrollytelling layout, horseshoe-aware narrative, graceful degradation*

*Previous: [The Report System: From Data to Narrative](ch01-report-system.md)*

*Next: [Legislator Profiles: Individual Deep Dives](ch03-profiles.md)*
