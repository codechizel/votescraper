# ADR-0069: Report Enhancement Infrastructure (R1-R13)

**Date:** 2026-03-01
**Status:** Accepted

## Context

The `docs/report-enhancement-survey.md` identified 26 prioritized report improvements. The top 13 (Tier 1 + Tier 2) were implemented as a single body of work. The existing report system (`analysis/report.py`) supported three section types (`TableSection`, `FigureSection`, `TextSection`) with static matplotlib PNGs and great_tables HTML. This was sufficient for initial analysis but lacked interactivity, executive summaries, and several analytical features standard in legislative analysis platforms (VoteView, FiveThirtyEight, ProPublica).

## Decision

### New Section Types

Three new frozen dataclass section types added to `analysis/report.py`:

1. **`KeyFindingsSection`** — Takes `list[str]` of HTML bullet points. Rendered above the TOC with distinct visual styling (light blue background, larger font). Every report builder now auto-generates 2-5 data-driven findings from computed results.

2. **`InteractiveTableSection`** — Stores pre-rendered ITables HTML. Uses `make_interactive_table()` helper (analogous to `make_gt()`). ITables v2.7+ with `connected=False` for fully self-contained offline mode. Replaces `make_gt()` for large legislator-level tables (>20 rows) where sort/search adds value.

3. **`InteractiveSection`** — Stores raw HTML fragments (Plotly charts, PyVis network graphs). Renders inside `<div class="interactive-container">`.

### New Dependencies

- **itables>=2.6** — Interactive DataTables for Polars DataFrames. Zero external JS dependencies in offline mode (`connected=False`). Used for legislator-ranked tables across 10+ phases.
- **plotly>=6.0** — Interactive scatter plots with hover-to-identify. Used for IRT ideal point scatter, IRT vs PCA comparison, unity vs IRT cross-reference. Embedded via `fig.to_html(full_html=False, include_plotlyjs=True)` (~3.4 MB per report, deduplicated across figures).
- **pyvis>=0.3** — Interactive force-directed network graphs from existing NetworkX objects. `net.from_nx(G)` one-liner. Used in Phase 06 Network.

### New Analytical Features

- **Bipartisanship Index (R12)** — `compute_bipartisanship_index()` in `analysis/13_indices/indices.py`. Lugar Center BPI analogue: fraction of party-line votes where legislator voted with opposing party majority. Distinct from maverick score (maverick = against own party; BPI = with opposing party). Includes BPI vs maverick scatter plot.

- **Plus-Minus Index (R11)** — `compute_plus_minus()` in `analysis/13_indices/indices.py`. Actual party unity minus party mean unity. Positive = more loyal than average. Dumbbell chart visualization.

- **Cutting Lines (R7)** — `compute_cutting_points()` in `analysis/05_irt/irt.py`. Per-bill cutting point (`alpha / beta`) where P(Yea) = 0.5. VoteView-style multi-panel visualization: legislators on ideology axis colored by actual vote, vertical line at cutting point.

- **Swing Vote Identification (R8)** — `identify_swing_votes()` in `analysis/05_irt/irt.py`. For close votes (margin <= 5), finds legislators within 0.5 IRT units of cutting point. Ranked frequency table: who is the swing vote most often.

- **Party Ideal Point Density (R3)** — `plot_party_density()` in `analysis/05_irt/irt.py`. Overlapping KDE curves per party. The single most published figure in IRT literature (Bafumi et al. 2005, Shor & McCarty 2011).

- **Item Characteristic Curves in Flat IRT (R6)** — `plot_icc_curves()` in `analysis/05_irt/irt.py`. P(Yea | theta) curves for top-5 most discriminating bills per chamber.

- **Absenteeism Analysis (R5)** — `_add_absenteeism_analysis()` in `analysis/01_eda/eda_report.py`. Strategic absence flags (ratio >= 2.0x on party-line votes with >= 3 absences).

- **Coalition Labeler (R13)** — `analysis/24_synthesis/coalition_labeler.py`. Auto-names clusters based on party composition and IRT ideal points. Rules: >80% one party = "{modifier} {party}" (Mainstream/Moderate/Conservative/Progressive based on median IRT vs party mean); mixed = "Bipartisan Coalition"/"Cross-Party Bloc".

### Presentation Enhancements

- **Key Findings (R1)** — All 22+ report builders now call `KeyFindingsSection` with auto-generated bullets. Each phase's `build_*_report()` function (or its `_generate_*_key_findings()` helper) computes 2-5 findings from results data.

- **Headline Titles (R4)** — Synthesis report uses data-driven section titles. Other phases use `subtitle` in `make_gt()` for findings.

- **ITables Conversion (R2)** — 10+ large tables converted from `make_gt()`/`TableSection` to `make_interactive_table()`/`InteractiveTableSection`. Includes: IRT ideal points, top discriminating votes, unity/maverick full tables, hierarchical shrinkage, clustering loyalty, network centrality, synthesis scorecard.

- **Plotly Interactive Scatters (R9)** — IRT ideal point scatter, IRT vs PCA comparison, unity vs IRT cross-reference. Hover shows legislator name, party, district, metric values.

- **PyVis Network Graphs (R10)** — Phase 06 Network: interactive force-directed graph with party-colored nodes, drag/zoom/hover.

## Consequences

### Positive
- Reports are immediately more useful for general audiences (searchable tables, hover-to-identify, executive summaries)
- Five new analytical features (BPI, plus-minus, cutting lines, swing votes, coalition labeling) add substantive analytical value
- Self-contained HTML preserved — `connected=False` for ITables, `include_plotlyjs=True` for Plotly, inline JS for PyVis
- Static matplotlib figures retained alongside interactive equivalents (print-friendly fallback)

### Negative
- HTML report file sizes increase (~3-5 MB for Plotly JS, ~1-2 MB for ITables JS per report)
- Three new dependencies (itables, plotly, pyvis) add to the dependency graph
- `make_interactive_table()` and `make_gt()` coexist — contributors must choose the right one (rule: interactive for >20 rows, great_tables for summary/config tables)

### Files Changed
- `analysis/report.py` — 3 new section types, `make_interactive_table()`, updated template/CSS
- `pyproject.toml` — 3 new dependencies
- 22+ `*_report.py` files — key findings, ITables conversions, interactive sections
- `analysis/05_irt/irt.py` — density, ICC, cutting points, swing votes, Plotly scatters
- `analysis/13_indices/indices.py` — BPI, plus-minus, BPI scatter, Plotly scatter
- `analysis/11_network/network.py` — PyVis generation
- `analysis/24_synthesis/coalition_labeler.py` — new module
- `analysis/01_eda/eda_report.py` — absenteeism analysis
- `tests/test_report_sections.py` — 30 new tests for section types
- `tests/test_indices.py` — 9 new tests for BPI
