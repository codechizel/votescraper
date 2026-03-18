# Volume 9 — Telling the Story

> *You've scraped the votes, estimated ideology, validated the models, found the coalitions, predicted the outcomes, read the text, and tracked the changes. Now what? Twenty-eight analysis phases produce thousands of numbers, hundreds of plots, and dozens of tables. Someone has to turn all of that into something a journalist, a policymaker, or a citizen can read in twenty minutes and walk away understanding.*

---

Volumes 1-8 built the analytical engine — the methods that transform raw roll call votes into ideology scores, network graphs, predictions, and temporal trends. But methods don't tell stories. A forest plot of 125 credible intervals is informative to a political scientist; to a school board member wondering how their representative votes, it's impenetrable.

This volume is about the last mile: turning analysis into narrative.

Chapter 1 describes the report system itself — the eight types of content that can appear in a Tallgrass report, the orchestrator that manages output directories and metadata, and the dashboard that ties everything together. Chapter 2 covers the synthesis phase, which aggregates results from ten upstream analyses and algorithmically detects the session's most interesting legislators — the mavericks, the bridge-builders, and the paradoxes whose metrics don't agree with each other. Chapter 3 goes deeper on individual legislators, explaining the scorecard, the defection analysis, and the voting neighbor comparison that make up a profile. Chapter 4 is a guided tour of a finished report, walking through each section and explaining what to look for.

The goal throughout is the same goal that animates the whole project: make Kansas legislative data accessible to anyone who cares enough to look.

---

## Chapters

| # | Title | What You'll Learn |
|---|-------|-------------------|
| 1 | [The Report System: From Data to Narrative](ch01-report-system.md) | How eight section types, a context manager, and a dashboard combine to produce self-contained HTML reports from twenty-eight analysis phases |
| 2 | [Synthesis: The Session Story](ch02-synthesis.md) | How Phase 24 aggregates ten upstream analyses, algorithmically detects mavericks, bridge-builders, and paradox legislators, and assembles a narrative-driven report for nontechnical audiences |
| 3 | [Legislator Profiles: Individual Deep Dives](ch03-profiles.md) | How Phase 25 builds per-legislator scorecards with six metrics, classifies bills by partisanship, identifies defection votes, finds voting neighbors, and surfaces surprising predictions |
| 4 | [How to Read a Tallgrass Report](ch04-how-to-read-report.md) | A guided walk through a finished report — what each section means, what patterns to look for, what the caveats are, and how to contribute |

---

## Key Terms Introduced in This Volume

| Term | Definition |
|------|-----------|
| Section type | One of eight standardized content blocks (table, figure, text, interactive table, interactive, key findings, download, scrollytelling) that can appear in a report |
| ReportBuilder | The class that assembles sections into a complete HTML document with a table of contents, metadata, and consistent styling |
| RunContext | A context manager that creates output directories, captures console logs, records metadata, and manages report lifecycle for each analysis phase |
| Run ID | A unique identifier for a pipeline execution in the format `{legislature}-{YYMMDD}.{N}` (e.g., `91-260318.1`) |
| Dashboard | A session-level HTML page with sidebar navigation that embeds all phase reports in an iframe |
| Primer | An auto-generated methodology description written to each phase's output directory |
| Synthesis | Phase 24 — the automated aggregation of ten upstream analyses into a single narrative report |
| Notable legislator | A legislator flagged by algorithmic detection as a maverick, bridge-builder, or paradox |
| Maverick | The legislator with the lowest party unity score in their caucus — the most frequent rebel |
| Bridge-builder | The legislator with the highest network centrality near the cross-party ideological midpoint |
| Metric paradox | A legislator whose IRT ideology rank and clustering loyalty rank disagree dramatically |
| Coalition labeler | An algorithm that auto-names clusters based on party composition and ideological position |
| Scorecard | A normalized six-metric profile (0-1 scale) summarizing a legislator's ideology, loyalty, maverick rate, network influence, and prediction accuracy |
| Bill discrimination tier | Classification of bills by their IRT discrimination parameter — high-discrimination (partisan), low-discrimination (routine) |
| Defection vote | A vote where a legislator disagreed with their party's majority position |
| Voting neighbor | The legislators whose vote-by-vote agreement with a target is highest (closest) or lowest (most different) |
| Surprising vote | A vote where the prediction model was most confident in the wrong answer |
| Scrollytelling | A progressive narrative layout where text scrolls while visualizations remain fixed, revealing the story step by step |
| Graceful degradation | The design principle that reports skip sections when data is unavailable rather than crashing or showing empty content |

---

*Previous: [Volume 8 — Change Over Time](../volume-08-change-over-time/)*

*Next: [Appendices](../appendices/)*
