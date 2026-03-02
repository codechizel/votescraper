# Report Enhancement Survey

A comprehensive review of Tallgrass's current HTML report output, gaps relative to academic standards and general-audience best practices, and opportunities from the open-source ecosystem. Based on a full inventory of all 17 phase reports (~300+ sections), a survey of the open-source legislative analysis landscape, and a review of what academics, newsrooms, and civic tech organizations report.

> **Implementation status (2026-03-02):** R1-R13 (Tier 1 + Tier 2) implemented — ADR-0069. R14-R20 (Tier 3) implemented — ADR-0071. Tier 4 remains as future backlog.

## Table of Contents

1. [Current Report Inventory](#current-report-inventory)
2. [What We're Missing: High-Impact Gaps](#high-impact-gaps)
3. [What We're Missing: Medium-Impact Gaps](#medium-impact-gaps)
4. [Visualization Upgrades: Interactive HTML](#visualization-upgrades)
5. [New Analysis Types](#new-analysis-types)
6. [General-Audience Presentation](#general-audience-presentation)
7. [Open-Source Tools and Libraries](#open-source-tools)
8. [Recommendations by Priority](#recommendations)
9. [Sources](#sources)

---

## Current Report Inventory

Tallgrass generates 17 HTML reports across its analysis pipeline. The report system uses `ReportBuilder` (Jinja2 template with embedded CSS), six section types (`TableSection`, `FigureSection`, `TextSection`, `KeyFindingsSection`, `InteractiveTableSection`, `InteractiveSection`), `make_gt()` for APA-style Polars tables via great_tables, and `make_interactive_table()` for searchable/sortable tables via ITables. Figures include both static matplotlib PNGs (base64-encoded) and interactive Plotly/PyVis HTML fragments.

| Phase | Report | Sections | Notable Features |
|-------|--------|----------|------------------|
| 01 EDA | Exploratory Data Analysis | ~22 | Agreement heatmaps, Rice preview, eigenvalue preview |
| 02 PCA | Principal Component Analysis | ~21 | Per-chamber scree, ideology scatter, sensitivity analysis |
| 02c MCA | Multiple Correspondence Analysis | ~21 | Biplots, horseshoe assessment, Greenacre correction |
| 03 UMAP | UMAP Dimensionality Reduction | ~16 | Sensitivity grid, multi-seed stability, IRT gradient |
| 04 IRT | Bayesian Item Response Theory | ~37 | Forest plots, paradox spotlight, PPC, holdout, joint model |
| 04b 2D IRT | Two-Dimensional IRT (Experimental) | ~16 | Red experimental banner, PLT identification |
| 04c PPC | Posterior Predictive Checks | ~15 | Model comparison, LOO-CV, Pareto k diagnostics |
| 05 Clustering | Cluster Analysis | ~48 | Dendrograms, icicle charts, within-party sub-factions |
| 05b LCA | Latent Class Analysis | ~29 | Profile heatmaps, Salsa effect, membership probabilities |
| 06 Network | Legislative Network Analysis | ~52 | Force-directed layouts, bridge legislators, Leiden communities |
| 06b Bipartite | Bipartite Bill-Legislator Network | ~35 | BiCM backbone, bill polarization, bridge bills |
| 07 Indices | Legislative Indices | ~43 | Rice, Unity, ENP, Maverick, co-defection heatmaps |
| 08 Prediction | Vote Prediction | ~25 | SHAP beeswarm, surprising votes, NLP topics |
| 09 Beta-Binomial | Beta-Binomial Empirical Bayes | ~15 | Shrinkage arrows, credible intervals, posteriors |
| 10 Hierarchical | Hierarchical Bayesian IRT | ~22 | Party posteriors, ICC, variance decomposition, shrinkage |
| 11 Synthesis | Legislative Synthesis | ~32 | Narrative-driven, dynamic profiles, full scorecard |
| 12 Profiles | Legislator Profiles | variable | Per-legislator radar charts, defection tables |
| 13 Cross-Session | Cross-Session Comparison | ~15 | Biggest movers, turnover, metric stability |
| 14 External Val | Shor-McCarty Validation | ~12 | Scatter plots, within-party correlations, outliers |
| 14b DIME Val | DIME CFscore Validation | ~12 | Campaign-finance ideology validation |
| 15 TSA | Time Series Analysis | ~15 | Rolling PCA drift, PELT changepoints, Bai-Perron CIs |
| 16 Dynamic IRT | Dynamic Ideal Points | ~14 | Polarization trends, trajectories, tau posteriors |
| 17 W-NOMINATE | W-NOMINATE + OC Validation | ~12 | Field-standard comparison via R subprocess |

**Total: ~300+ sections across 17+ report types.** Every report includes interpretation text blocks and an analysis parameters table. Nine reports include a "How to Read This Report" section.

---

## High-Impact Gaps

These are significant omissions relative to the combined standards of academic publications, VoteView, FiveThirtyEight, ProPublica, and GovTrack. Each would materially improve the reports for a general audience.

### 1. Executive Summary / Key Findings

**The gap:** Reports jump straight into tables and figures. No report (except Synthesis) opens with a concise "here's what we found" summary. Academic papers have abstracts. Data journalism has ledes. News readers expect the headline first.

**What to add:** A 3-5 bullet point "Key Findings" section at the top of every report. Example: "Republicans voted together 94% of the time, but a 12-member moderate faction broke ranks on 23 key votes. Sen. Masterson was the most independent Republican." Auto-generated from the results data, not hand-written.

**Who does this well:** FiveThirtyEight leads every analysis with the finding. GovTrack report cards open with the most notable stat. The Economist uses subtitle-as-finding ("Republicans are more divided than they look").

### 2. Per-Vote Spatial Visualization (Cutting Lines)

**The gap:** VoteView's signature visualization — legislators positioned on the ideology scale with a cutting line showing who voted which way on a specific bill — has no equivalent in Tallgrass. This is the single most recognizable figure in legislative analysis.

**What to add:** For each chamber's top-5 most discriminating votes (highest IRT beta), show the ideological spectrum with legislators colored by their actual vote (Yea/Nay), with the estimated cutting point marked. Immediate visual: "everyone to the right of this line voted Yea."

**Who does this well:** VoteView (the original), LegisLatio (interactive version), Clinton/Jackman/Rivers (2004) in academic publications.

### 3. Party Ideal Point Density Overlay

**The gap:** The most common figure in IRT publications — overlapping kernel density curves of Republican vs. Democrat ideal point distributions — is not in the flat IRT report. The hierarchical report has party posteriors, but the primary Phase 04 IRT report does not show the full distributional shape of each party's ideal points.

**What to add:** A single figure with smoothed density curves for each party overlaid on the same axis. Shows party separation, overlap region, and within-party spread at a glance. The overlap region is where bipartisan potential exists.

**Who does this well:** Bafumi et al. (2005), Shor & McCarty (2011), VoteView polarization plots. This is arguably the single most-published figure in ideal point estimation literature.

### 4. Predicted vs. Actual Framing ("Plus-Minus")

**The gap:** FiveThirtyEight's most effective device: "Based on their district, we expected Legislator X to vote with their party 78% of the time; they actually voted with party 92% of the time. Difference: +14." This instantly communicates whether a legislator is a team player or independent. Tallgrass has the prediction model (Phase 08) and party unity scores (Phase 07) but does not present them in this intuitive head-to-head format.

**What to add:** A table and/or dumbbell chart showing expected vs. actual party loyalty for every legislator, sorted by the gap. Positive gap = more partisan than expected; negative gap = more independent than expected. The "expected" baseline could come from the prediction model's per-legislator accuracy or from district-level partisan lean (if available).

**Who does this well:** FiveThirtyEight Trump Score, GovTrack predicted vs. actual ideology.

### 5. Absenteeism as Analysis (Not Just Filtering)

**The gap:** Tallgrass treats participation rates as a data-quality metric (filter legislators with <X% participation). But absenteeism is a finding, not just noise. Academic research shows legislators strategically skip close or controversial votes. General audiences care deeply about whether their representative shows up.

**What to add:** A dedicated section (in EDA or Indices) analyzing absence patterns: which legislators miss the most votes, which specific votes they miss, whether absences correlate with vote closeness or controversy, and whether there's a "strategic absence" signal. Present it as a ranked table with a narrative: "Sen. X missed 23 votes, including 8 of the 10 closest votes of the session."

**Who does this well:** ProPublica Represent (missed votes as a headline metric), GovTrack report cards (missed votes percentage), Ballotpedia (voting record completeness).

### 6. Swing Vote Identification

**The gap:** For close votes, who was the decisive legislator? Whose vote determined the outcome? This is the most compelling question for journalists and citizens, and Tallgrass does not answer it directly. The IRT ideal points and vote outcomes provide all the infrastructure needed.

**What to add:** For each close vote (margin ≤ 5), identify the legislator(s) whose ideal point is closest to the estimated cutting point. Label them as the "swing vote" for that bill. Aggregate across the session: who was the swing vote most often? This is immediately newsworthy.

**Who does this well:** CQ Roll Call (key vote analyses), academic pivot point analysis (Stiglitz 2010).

### 7. Interactive Visualizations

**The gap:** All figures are static matplotlib PNGs embedded as base64 in HTML. No hover tooltips, no zoom, no click-to-filter. In 2026, static charts in HTML reports feel dated. Network graphs, scatter plots, and heatmaps all benefit enormously from interactivity.

**What to add:** Replace selected high-value static figures with interactive equivalents. Not all — matplotlib is fine for density plots and bar charts. But network graphs, ideology scatters, and legislator tables would benefit from hover-to-identify, click-to-filter, and zoom.

**Key tools:**
- **Plotly**: Native Polars support (v6), standalone HTML export via `fig.write_html()`, fragment embedding via `.to_html(full_html=False)`. Strongest candidate for replacing matplotlib in high-value figures.
- **PyVis**: Interactive network graphs from NetworkX. Force-directed layout with drag, zoom, hover. Ideal for Phase 06/06b.
- **ITables**: Interactive DataTables (sort, search, filter, paginate) for Polars DataFrames. Zero dependencies since v2.6.0. Ideal for legislator tables.
- **Folium**: Interactive Leaflet maps. Choropleth of Kansas districts by ideology or party unity.

**Who does this well:** VoteView (D3.js interactive scatters), NYT election pages, WaPo interactive dashboards, every modern data journalism site.

---

## Medium-Impact Gaps

Standard in the field and partially present but not fully developed.

### 8. Item Characteristic Curves in Flat IRT

The hierarchical IRT report includes ICC plots. The primary Phase 04 flat IRT report does not. ICCs show, for each bill, the probability of a "Yea" vote as a function of ideal point. They're standard in psychometric reporting and help readers understand what individual votes "mean" on the ideological scale. Add ICCs for the 5-10 most discriminating bills per chamber.

### 9. Vote-Based Bipartisanship Index

Distinct from the existing maverick score. The maverick score measures how often a legislator breaks from their own party. A bipartisanship index measures how often a legislator votes with the majority of the opposing party on party-line votes. They're related but different: a legislator could be a maverick (breaks from party) without being bipartisan (doesn't align with the other side — just votes differently). The Lugar Center BPI is the gold standard for cosponsorship-based bipartisanship; a vote-based analogue is straightforward to compute.

### 10. Named and Described Coalitions

Phases 05 (Clustering), 05b (LCA), 06 (Network) all identify groups by number or color. The Synthesis report begins to characterize them narratively. But no report explicitly names and describes the detected coalitions in a way a journalist could quote: "the moderate Republican faction (12 members, median IRT -0.3, centered in suburban districts)." Named coalitions with member lists and characterization would be immediately useful for reporting.

### 11. Credible Interval Width Analysis

Which legislators have the most uncertainty in their ideal point estimates? The forest plots show HDI widths, but no report explicitly discusses why some legislators have wide intervals (fewer votes, centrist position, strategic absences) and what that means for interpretation. A brief "Uncertainty Spotlight" section in the IRT report would strengthen the analysis.

### 12. Individual Legislator Temporal Trajectories

Phase 16 (Dynamic IRT) produces spaghetti plots of ideal point trajectories. But small-multiple sparklines for individual legislators — showing their personal ideology path across bienniums they served — would be more readable and quotable. Martin & Quinn (2002) does this for Supreme Court justices; it's the most cited feature of their paper.

### 13. Headline-Style Section Titles

Most phase reports use descriptive titles: "Rice Index Summary," "Convergence Diagnostics," "Vote Margin Distribution." The Synthesis report uses narrative titles. Academic papers and newsrooms use finding-as-title: "Republicans Vote Together 94% of the Time" instead of "Party Unity Scores." Narrative titles make reports scannable and quotable.

### 14. Full Voting Record per Legislator

The Profiles report (Phase 12) shows defection votes and surprising votes, but not the complete voting record. General-audience sites (ProPublica, GovTrack, VoteView) all provide full vote-by-vote detail for each legislator. An expandable or paginated table showing every vote a legislator cast would serve both journalists and constituents.

---

## Visualization Upgrades

### Interactive Libraries for HTML Reports

| Library | Best For | Integration Effort | Standalone HTML |
|---------|----------|-------------------|-----------------|
| **Plotly** | Scatter plots, bar charts, heatmaps, network graphs | Medium — replace `plt.figure()` with `go.Figure()` | Yes, via `.to_html(full_html=False)` |
| **PyVis** | Network graphs (Phase 06, 06b) | Low — `from_nx(G)` on existing NetworkX objects | Yes, generates self-contained HTML |
| **ITables** | Sortable/searchable legislator tables | Low — `to_html_datatable(df)` on existing Polars DataFrames | Yes, zero dependencies since v2.6.0 |
| **Folium** | Geographic district maps | Medium — requires Kansas district GeoJSON | Yes, Leaflet-based standalone HTML |
| **Great Tables** | Publication-quality formatted tables | Already in use via `make_gt()` | Yes |
| **Altair** | Declarative statistical charts | Medium — different API from matplotlib | Yes, via `.save()` |
| **ArviZ (Plotly backend)** | Interactive MCMC diagnostics | Low — `arviz.style.use("arviz-plotly")` | Yes |

### Recommended Migration Path

**Phase 1 — Quick wins (minimal code change):**
- ITables for all legislator-level tables (sortable, searchable, paginated)
- PyVis for Phase 06/06b network visualizations (existing NetworkX graphs)
- Great Tables nanoplots (inline sparklines in tables — already using great_tables)

**Phase 2 — High-value replacements:**
- Plotly for ideology scatter plots (hover to identify legislators)
- Plotly for forest plots (hover to see HDI details)
- Plotly for SHAP figures (interactive exploration)
- Folium for district maps (if GeoJSON available)

**Phase 3 — Comprehensive upgrade:**
- ArviZ with Plotly backend for all MCMC diagnostic plots
- Plotly for all heatmaps and correlation matrices
- Altair for small-multiple/faceted charts

### New Visualization Types to Add

| Visualization | Purpose | Phase | Library |
|---------------|---------|-------|---------|
| **Beeswarm/strip plot** | 1D ideology spectrum, all legislators | 04 IRT | Plotly or seaborn |
| **Dumbbell chart** | Expected vs. actual party loyalty | 07 Indices | Plotly |
| **Ridgeline/joy plot** | Party ideal point distributions over time | 16 Dynamic | matplotlib (joypy) or Plotly |
| **Parliament/hemicircle chart** | Vote composition (Yea/Nay/Absent) | 01 EDA | Plotly or D3.js |
| **Sankey/alluvial diagram** | Bill flow (intro → committee → floor → passage) | 01 EDA | Plotly `go.Sankey()` |
| **Slope chart** | Ideology change between two sessions | 13 Cross-Session | matplotlib or Plotly |
| **Raincloud plot** | Distribution + points + box for ideal points | 04 IRT | matplotlib (PtitPrince) |
| **Choropleth map** | Districts colored by ideology/unity | New section | Folium |
| **Scrollytelling** | Progressive narrative reveal | 11 Synthesis | Quarto or custom JS |
| **Quantile dotplot** | Uncertainty communication | 04 IRT, 10 Hierarchical | matplotlib or Plotly |

---

## New Analysis Types

These are analyses that Tallgrass does not currently perform but has all the data to support.

### Absenteeism Analysis (High Priority)

**What:** Systematic analysis of voting absences — who misses votes, which votes they miss, and whether absence patterns are strategic.

**Metrics:**
- Absence rate by legislator (already computed but treated as filter, not finding)
- Absence rate by vote closeness (do legislators skip close votes?)
- Absence rate by vote type (do legislators skip certain motion types?)
- Strategic absence score: correlation between absences and predicted vote discomfort (votes where IRT model predicts high uncertainty)

**Output:** Ranked table of legislators by absence rate with context. Scatter plot of absence rate vs. vote closeness. Narrative identifying potential strategic absences.

**Literature:** Cohen & Noll (1991), "How to Vote Whether or Not to Show Up"; Rothenberg & Sanders (2000) on strategic abstention.

### Swing Vote Identification (High Priority)

**What:** For each close vote, identify the legislator whose ideal point is closest to the cutting point.

**Metrics:**
- Per close vote: the "pivot" legislator(s)
- Aggregate: who is the swing vote most often across the session
- Swing vote network: do the same legislators keep being pivotal?

**Output:** Table of close votes with identified swing legislators. Ranked "swing vote frequency" table. Narrative profiles of the most frequent swing voters.

### Bipartisanship Index (Medium Priority)

**What:** Vote-based measure of cross-party alignment, distinct from maverick score.

**Formula:** For each legislator, the fraction of party-line votes where they voted with the opposing party's majority. A maverick who votes "No" when both parties vote "Yes" is not bipartisan. A legislator who crosses the aisle to vote with the other party is.

**Output:** Ranked bipartisanship index table. Scatter of bipartisanship vs. IRT ideal point. Correlation with maverick score (should be positive but not identical).

### Cohort / Freshmen Analysis (Medium Priority)

**What:** Do newly elected legislators vote differently from incumbents? Do they converge toward the party mean over time within a session?

**Metrics:**
- First-term vs. returning legislator ideal point distributions
- Within-session ideology drift for freshmen (Phase 15 TSA infrastructure)
- Party loyalty rates for freshmen vs. incumbents

**Output:** Comparative density plots. Drift analysis for first-term members. Narrative about the "learning curve."

### Bill Outcome Analysis (Lower Priority)

**What:** Predict bill passage/failure from bill characteristics and voting patterns.

**Metrics:**
- Passage rate by motion type, chamber, party of sponsor (if available)
- Features: vote margin on related bills, sponsor's ideal point, committee membership
- Logistic regression or XGBoost on bill-level outcomes

**Literature:** Stanford CS229 (2015) achieves ~80% accuracy on congressional bill passage prediction.

### Voting Bloc Stability (Lower Priority)

**What:** Track whether clusters/coalitions persist, split, or merge across bienniums.

**Metrics:**
- Adjusted Rand Index of cluster assignments across consecutive bienniums (for returning legislators)
- Coalition membership Sankey diagram showing flow across sessions
- Individual legislator cluster stability (how often they change groups)

**Output:** Cross-session ARI table. Sankey diagram of bloc evolution. Narrative identifying stable vs. fluid coalitions.

---

## General-Audience Presentation

### What Newsrooms and Civic Tech Do Differently

| Practice | Who Does It | Current Tallgrass Status |
|----------|-------------|--------------------------|
| Single-number summary per legislator | FiveThirtyEight, GovTrack, ProPublica | Partial (maverick score, party unity — but buried in tables) |
| Finding-as-headline | FiveThirtyEight, The Economist, Pew | Synthesis report only |
| Predicted vs. actual framing | FiveThirtyEight Trump Score | Not present |
| Missed votes as headline metric | ProPublica, GovTrack | Treated as filter, not finding |
| Searchable/sortable tables | Every modern data site | Not present (static HTML tables) |
| Interactive hover/zoom | VoteView, NYT, WaPo | Not present (static matplotlib PNGs) |
| Mobile-responsive figures | All newsroom sites | Partial (CSS max-width, but figures are fixed-size PNGs) |
| Downloadable underlying data | VoteView, ProPublica, GovTrack | Not present |
| Alt text for figures | Accessibility standard | Minimal (title only, not descriptive) |
| Colorblind-safe palettes | RSS guidelines, Pew | Red/blue party coding is acceptable for political data; no other CBF issues identified |

### Design Principles for General Audiences

From the Royal Statistical Society (2023), FiveThirtyEight, Pew Research, and the 2024 arxiv survey on political visualization:

1. **Lead with the finding, not the method.** "12 Republicans broke ranks on tax votes" before "we used Ward linkage clustering on Cohen's Kappa distances."
2. **Limit each visualization to 2-3 concepts.** Our clustering report's 48 sections are thorough but overwhelming for a general reader.
3. **Use established chart types** (bar, line, scatter) for the primary message. Save novel types (beeswarm, Sankey, icicle) for supplementary exploration.
4. **Uncertainty as ranges, not point estimates.** Graduated HDIs (thin=94%, thick=50%) are widely understood. Quantile dotplots outperform density plots for decision-making.
5. **Annotate directly on charts** rather than requiring legend lookup. Label notable legislators by name on the plot.
6. **Headline-style titles** that convey the finding: "Moderate Republicans Broke Ranks on 23 Key Votes" instead of "Cluster Analysis Results."
7. **Short paragraphs, active voice, no jargon.** The Synthesis report does this well. Other phase reports still use academic framing ("the eigenvalues suggest...").

### Report Structure Recommendation

For each phase report, consider this ordering:

1. **Key Findings** (3-5 bullets, auto-generated from results)
2. **How to Read This Report** (plain-English guide — already in 9 reports)
3. **Main Findings** (the most important figures and tables)
4. **Detail Tables** (full legislator-level data, searchable/sortable)
5. **Technical Diagnostics** (convergence, sensitivity, parameters)
6. **Methodology** (what we did and why)

Currently, most reports interleave technical and substantive sections. Moving diagnostics to the end lets general readers get findings without wading through R-hat tables.

---

## Open-Source Tools and Libraries

### Legislative Data Platforms

| Platform | Coverage | Data Type | Open Source | Relevance |
|----------|----------|-----------|-------------|-----------|
| [VoteView](https://voteview.com/) | Congress, 1789-present | Roll calls, ideology scores | Data yes, viz code partial | Gold standard for ideology vis |
| [Open States](https://openstates.org/) | 50 states + DC/PR | Bills, votes, legislators | Yes (API + scrapers) | State-level data aggregator |
| [LegiScan](https://legiscan.com/) | 50 states + Congress | Bills, votes, full text | API (free tier) | Real-time bill tracking |
| [ProPublica Congress API](https://projects.propublica.org/api-docs/congress-api/) | Congress, 1995-present | Members, votes, bills | API (5k req/day) | General-audience design model |
| [@unitedstates/congress](https://github.com/unitedstates/congress) | Congress | Bills, votes, amendments | Yes (public domain) | Python scraper reference |
| [GovTrack](https://www.govtrack.us/) | Congress | Members, bills, votes, report cards | Partial | Report card design model |
| [Shor-McCarty](https://americanlegislatures.com/) | State legislatures, 1993-2020 | Ideology scores | Data yes | Already used (Phase 14) |
| [IPPSR Correlates](https://ippsr.msu.edu/public-policy/correlates-state-policy) | 50 states | Policy variables, ideology | Data yes | State-level context |
| [Polarization Research Lab](https://polarizationresearchlab.org/) | Elected officials | Approval, affect, polarization | Dashboard | Legislator-level dashboards |

**Key finding:** No open-source platform does for state legislatures what VoteView does for Congress. Tallgrass occupies a genuinely novel niche for Kansas.

### Visualization Libraries

| Library | Type | Polars Support | HTML Export | Best For |
|---------|------|---------------|-------------|----------|
| [Plotly](https://plotly.com/python/) | Interactive charts | Native (v6, via Narwhals) | Standalone or fragment | Scatter, bar, heatmap, network, Sankey |
| [Altair](https://altair-viz.github.io/) | Declarative statistical | Default Polars backend | Standalone | Small multiples, faceted charts |
| [PyVis](https://pyvis.readthedocs.io/) | Interactive networks | Via NetworkX | Standalone | Force-directed network graphs |
| [ITables](https://github.com/mwouts/itables) | Interactive tables | Yes (v2.6+) | Standalone | Searchable/sortable legislator tables |
| [Folium](https://python-visualization.github.io/folium/) | Interactive maps | GeoJSON input | Standalone | Choropleth district maps |
| [Great Tables](https://posit-dev.github.io/great-tables/) | Publication tables | First-class | Inline HTML | Already in use |
| [ArviZ](https://github.com/arviz-devs/arviz) | Bayesian diagnostics | Via xarray | Plotly backend available | MCMC traces, posteriors |
| [Quarto](https://quarto.org/) | Publishing system | Python, R, Julia | Website/dashboard | Multi-phase unified site |
| [Bokeh](https://bokeh.org/) / [Panel](https://panel.holoviz.org/) | Interactive dashboards | Via HoloViz | Standalone or server | Full dashboard apps |

### Dashboard / Publishing Frameworks

**Quarto** deserves special mention. It's an open-source publishing system from Posit (the RStudio team) that supports Python, R, and Observable JS. It can generate interactive dashboards from markdown, publish to GitHub Pages, and combine Plotly/Folium/Jupyter widgets in a single document. A Quarto dashboard could unify all 17 phase reports into a single navigable website with cross-linking between phases.

---

## Recommendations

### Tier 1: High-Impact, Lower Effort

These can be implemented without changing the analysis pipeline — they're presentation-layer enhancements.

| # | Enhancement | Phases Affected | Effort |
|---|-------------|-----------------|--------|
| 1 | **Key Findings section** at top of every report | All 17 | Low — template change + auto-generation logic |
| 2 | **ITables for legislator tables** (sort, search, filter) | All phases with full legislator tables | Low — swap `make_gt()` for `to_html_datatable()` |
| 3 | **Party ideal point density overlay** | 04 IRT | Low — one new matplotlib/Plotly figure |
| 4 | **Headline-style section titles** in Synthesis | 11 Synthesis (already partial) | Low — rename existing sections |
| 5 | **Absenteeism narrative** in EDA/Indices | 01 EDA, 07 Indices | Low — data already computed |
| 6 | **ICCs in flat IRT report** | 04 IRT | Low — code exists in Phase 10 |

### Tier 2: High-Impact, Medium Effort

These require new analysis code or significant visualization infrastructure changes.

| # | Enhancement | Phases Affected | Effort |
|---|-------------|-----------------|--------|
| 7 | **Per-vote spatial visualization** (cutting lines) | 04 IRT, new section | Medium — new figure type |
| 8 | **Swing vote identification** | 04 IRT or 07 Indices, new analysis | Medium — new computation |
| 9 | **Plotly for ideology scatter plots** | 02 PCA, 04 IRT, 05 Clustering | Medium — new rendering path in report.py |
| 10 | **PyVis for network graphs** | 06 Network, 06b Bipartite | Medium — new section type in report.py |
| 11 | **Predicted vs. actual ("plus-minus") framing** | 07 Indices or 08 Prediction | Medium — new computation + table |
| 12 | **Bipartisanship index** | 07 Indices | Medium — new metric + sections |
| 13 | **Named/described coalitions** | 05 Clustering, 11 Synthesis | Medium — labeling logic |

### Tier 3: High-Impact, Higher Effort

Substantial new features or infrastructure.

| # | Enhancement | Phases Affected | Effort |
|---|-------------|-----------------|--------|
| 14 | **Folium district maps** | New phase or EDA section | High — requires Kansas GeoJSON |
| 15 | **Full voting record per legislator** | 12 Profiles | High — large table infrastructure |
| 16 | **Quarto unified dashboard** | All phases | High — new publishing system |
| 17 | **Downloadable CSV alongside reports** | All phases | Medium — export logic + links |
| 18 | **Freshmen cohort analysis** | 13 Cross-Session or new phase | Medium-High — new analysis |
| 19 | **Voting bloc stability tracking** | 13 Cross-Session | Medium-High — new cross-session analysis |
| 20 | **Scrollytelling in Synthesis** | 11 Synthesis | High — JavaScript integration |

### Tier 4: Nice-to-Have

| # | Enhancement | Notes |
|---|-------------|-------|
| 21 | **Parliament/hemicircle charts** for vote composition | Visually striking but not analytically essential |
| 22 | **Sankey diagrams** for bill flow | Requires bill lifecycle data beyond roll calls |
| 23 | **Ridgeline plots** for temporal ideology distributions | Alternative to existing density plots |
| 24 | **Animated scatter** (Gapminder-style) for dynamic IRT | Engaging but complex |
| 25 | **Descriptive alt text** for all figures | **Done** — WCAG 2.1 AA: 132 alt-text + 8 aria-labels across 23 report builders (ADR-0079) |
| 26 | **Bill outcome prediction model** | Requires bill-level features beyond roll calls |

---

## Sources

### Academic References
- Clinton, Jackman & Rivers (2004). [The Statistical Analysis of Roll Call Data](https://www.cs.princeton.edu/courses/archive/fall09/cos597A/papers/ClintonJackmanRivers2004.pdf). APSR.
- Bafumi, Gelman, Park & Kaplan (2005). [Practical Issues in Implementing and Understanding Bayesian Ideal Point Estimation](https://sites.stat.columbia.edu/gelman/research/published/171.pdf). Political Analysis.
- Martin & Quinn (2002). [Dynamic Ideal Point Estimation via MCMC](https://www.cambridge.org/core/journals/political-analysis/article/dynamic-ideal-point-estimation-via-markov-chain-monte-carlo-for-the-us-supreme-court-19531999/2A57930D5D0C81216491B40CA2BA5D12). Political Analysis.
- Shor & McCarty (2011). [The Ideological Mapping of American Legislatures](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1676863). APSR.
- Volden & Wiseman. [Center for Effective Lawmaking Methodology](https://thelawmakers.org/methodology).
- Crespin, Rohde & Vander Wielen (2013). Adjusted party unity. Party Politics.
- Cohen & Noll (1991). How to Vote Whether or Not to Show Up.
- Stiglitz (2010). [Agenda Control](https://onlinelibrary.wiley.com/doi/10.3162/036298010791170187). Legislative Studies Quarterly.
- Wilke. [Fundamentals of Data Visualization — Uncertainty](https://clauswilke.com/dataviz/visualizing-uncertainty.html).
- Royal Statistical Society (2023). [Best Practices for Data Visualisation](https://royal-statistical-society.github.io/datavisguide/).
- arxiv (2024). [Visualization in Political Science](https://arxiv.org/abs/2405.05947). 820 visualizations analyzed.

### Legislative Data Platforms
- [VoteView / DW-NOMINATE](https://voteview.com/) — UCLA. [Data](https://voteview.com/data). [Polarization](https://voteview.com/articles/party_polarization). [WebVoteView GitHub](https://github.com/voteview/WebVoteView).
- [Open States API v3](https://docs.openstates.org/api-v3/). [pyopenstates](https://openstates.github.io/pyopenstates/). [openstates-geo](https://github.com/openstates/openstates-geo).
- [LegiScan API](https://legiscan.com/legiscan). [MCP Server](https://github.com/sh-patterson/legiscan-mcp).
- [ProPublica Congress API](https://projects.propublica.org/api-docs/congress-api/). [Represent](https://www.propublica.org/nerds/a-new-way-to-keep-an-eye-on-who-represents-you-in-congress).
- [@unitedstates/congress](https://github.com/unitedstates/congress).
- [GovTrack](https://www.govtrack.us/). [Methodology](https://www.govtrack.us/about/analysis). [Report Cards 2024](https://www.govtrack.us/congress/members/report-cards/2024).
- [Shor-McCarty / American Legislatures](https://americanlegislatures.com/). [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GZJOT3).
- [IPPSR Correlates of State Policy](https://ippsr.msu.edu/public-policy/correlates-state-policy).
- [Ballotpedia State Legislative Tracker](https://ballotpedia.org/State_Legislative_Tracker). [State Scorecards](https://ballotpedia.org/State_legislative_scorecards_in_Kansas).
- [Polarization Research Lab Dashboard](https://polarizationresearchlab.org/2024/08/12/news-americas-political-pulse-elected-official-data-and-dashboard-launch/).
- [FiveThirtyEight Trump Score](https://projects.fivethirtyeight.com/congress-trump-score/).
- [Lugar Center BPI](https://www.thelugarcenter.org/ourwork-43.html).

### Visualization Libraries
- [Plotly](https://plotly.com/python/). [Interactive HTML Export](https://plotly.com/python/interactive-html-export/). [Dumbbell Plots](https://plotly.com/python/dumbbell-plots/). [Parliament Chart Guide](https://community.plotly.com/t/show-and-tell-step-by-step-guide-for-building-parliament-charts-in-plotly/90453).
- [Altair / Vega-Lite](https://altair-viz.github.io/). [Gallery](https://altair-viz.github.io/gallery/index.html).
- [PyVis](https://pyvis.readthedocs.io/). [GitHub](https://github.com/WestHealth/pyvis).
- [ITables](https://github.com/mwouts/itables). [Jupyter Blog](https://blog.jupyter.org/make-your-pandas-or-polars-dataframes-interactive-with-itables-2-0-c64e75468fe6).
- [Folium](https://python-visualization.github.io/folium/). [Programming Historian Tutorial](https://programminghistorian.org/en/lessons/choropleth-maps-python-folium).
- [Great Tables](https://posit-dev.github.io/great-tables/). [Polars Styling](https://posit-dev.github.io/great-tables/blog/polars-styling/).
- [ArviZ](https://github.com/arviz-devs/arviz). [arviz-plots](https://python.arviz.org/projects/plots/en/latest/index.html).
- [Quarto](https://quarto.org/). [Quarto Dashboards](https://quarto.org/docs/dashboards/).
- [Bokeh](https://bokeh.org/). [Panel (HoloViz)](https://panel.holoviz.org/).
- [Flourish](https://flourish.studio/). [Parliament Charts](https://flourish.studio/visualisations/parliament-charts/). [Sankey](https://flourish.studio/visualisations/sankey-charts/).
- [Datawrapper](https://www.datawrapper.de/).
- [Clustergrammer](https://github.com/MaayanLab/clustergrammer).
- [D3 Parliament Chart](https://observablehq.com/@dkaoster/d3-parliament-chart).
- [ColorBrewer](https://colorbrewer2.org/). [Colorblind-Safe Guide](https://davidmathlogic.com/colorblind/).
- [Python Graph Gallery](https://python-graph-gallery.com/).

### Newsroom Design References
- [Pew Research 2025 Favorite Visualizations](https://www.pewresearch.org/short-reads/2025/12/15/our-favorite-data-visualizations-of-2025/).
- [Pew Small Multiples](https://www.pewresearch.org/decoded/2018/12/20/how-pew-research-center-uses-small-multiple-charts/).
- [Washington Post 2024 Visual Storytelling](https://www.washingtonpost.com/lifestyle/interactive/2024/visual-storytelling/).
- [FiveThirtyEight Visualization Lessons](https://towardsdatascience.com/data-visualization-hack-lessons-from-fivethirtyeight-graphs-e121080725a6/).
- [Economist-Style Charts in Matplotlib](https://medium.com/data-science/making-economist-style-plots-in-matplotlib-e7de6d679739).
- [Best Data Visualizations of 2025 (AnyChart)](https://www.anychart.com/blog/2026/01/09/best-data-visualizations-2025/).
