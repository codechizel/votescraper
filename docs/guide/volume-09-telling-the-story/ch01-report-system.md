# Chapter 1: The Report System: From Data to Narrative

> *Twenty-eight analysis phases produce plots, tables, parquets, and CSVs. The report system turns them into something you can open in a browser and read.*

---

## Why Reports Matter

A statistical model doesn't do anyone any good if its output lives in a `.parquet` file that only Python can read. Tallgrass exists to make Kansas legislative data accessible — and "accessible" means a self-contained HTML file that a journalist can open on their laptop, a lobbyist can scroll through on their phone, and a citizen can share with a link.

Every analysis phase in the pipeline produces an HTML report. The report embeds its own plots (no external image files to lose), includes searchable tables (no separate spreadsheets), and contains all the metadata needed to reproduce the analysis (git hash, runtime, parameters). You could email it to someone with no technical background and they'd see a formatted document with a table of contents, numbered sections, and captioned figures.

This chapter describes the three pieces that make that happen: the **section types** (the building blocks), the **ReportBuilder** (the assembler), and the **RunContext** (the orchestrator that manages everything from directory creation to metadata recording).

## The Building Blocks: Section Types

Think of a report as a magazine. Each page might contain a photograph, a data table, a block of text, or an interactive chart. In Tallgrass, each of these content types has its own **section type** — a standardized container that knows how to render itself into HTML.

There are eight section types:

| Section Type | What It Contains | Analogy |
|-------------|-----------------|---------|
| **TableSection** | A pre-formatted statistical table | A printed data table in a research paper |
| **FigureSection** | A PNG image embedded as base64 | A photograph in a magazine — the image is baked into the page |
| **TextSection** | A block of HTML prose | An article or sidebar |
| **InteractiveTableSection** | A searchable, sortable table | A spreadsheet you can filter and rearrange |
| **InteractiveSection** | A Plotly chart or network visualization | An interactive infographic you can hover over and zoom |
| **KeyFindingsSection** | A bullet-point summary | The "key takeaways" box at the top of a report |
| **DownloadSection** | Links to downloadable CSV files | An appendix of raw data files |
| **ScrollySection** | A progressive narrative with sticky visuals | A multimedia story where images stay fixed while text scrolls |

### How Figures Are Embedded

When a phase creates a matplotlib plot, the report doesn't link to an external PNG file. Instead, it **base64-encodes** the image and embeds it directly in the HTML:

```html
<img src="data:image/png;base64,iVBORw0KGgo..." alt="Forest plot of ideal points"/>
```

The analogy: imagine printing a photograph directly onto the page of a book, rather than stapling a separate photo to each copy. The image is part of the document — it can't get separated, lost, or broken by a missing file path. This is what makes Tallgrass reports self-contained: you can move the HTML file anywhere and it still works.

`FigureSection` has two factory methods that handle this automatically. `from_file()` reads a PNG from disk and encodes it. `from_figure()` takes an in-memory matplotlib Figure, renders it to PNG at 150 DPI, and encodes it — the plot never touches the filesystem.

Every figure includes an `alt_text` parameter for **WCAG 2.1 AA accessibility** — screen readers can describe the chart to visually impaired users.

**Codebase:** `analysis/report.py` (`FigureSection`, `from_file()`, `from_figure()`)

### Interactive Tables

Static tables are fine for small datasets, but when a report includes 125 legislators with 10 columns each, readers need the ability to search and sort. The `InteractiveTableSection` wraps a Polars DataFrame in [ITables](https://mwouts.github.io/itables/) — a JavaScript library that adds client-side search, column sorting, and filtering to any HTML table.

The key design choice: **offline mode**. ITables can fetch its JavaScript from a CDN (a remote server), but that would mean the report needs an internet connection to function. Tallgrass uses `connected=False`, which bundles the JavaScript directly into the HTML. The file is slightly larger, but it works on a plane, in a government building with restricted internet, or in an archive ten years from now.

**Codebase:** `analysis/report.py` (`make_interactive_table()`, `InteractiveTableSection`)

### Scrollytelling

The most sophisticated section type is `ScrollySection` — a **scrollytelling** layout where narrative text scrolls on the left side of the screen while visualizations remain fixed ("sticky") on the right. As the reader scrolls through each narrative step, the visualization updates to match.

The analogy: imagine a documentary where the narrator's voice keeps going while the camera holds on a key image, then cuts to the next image when the narration moves to a new topic. The reader controls the pace by scrolling.

Under the hood, this uses [Scrollama.js](https://github.com/russellsamora/scrollama) and the browser's IntersectionObserver API. Each `ScrollyStep` has a `narrative_html` (the text that scrolls) and a `visual_id` (which visualization to show). When a step enters the viewport, the corresponding visual becomes visible and all others hide.

The synthesis report (Chapter 2) uses scrollytelling for its narrative mode — walking the reader through the session's story from "Kansas Legislature at a Glance" through "Can We Predict Their Votes?" in six chapters, each with its own sticky visualizations.

**Codebase:** `analysis/report.py` (`ScrollySection`, `ScrollyStep`, `_scrolly_init_js()`)

## The Assembler: ReportBuilder

The section types are the Lego bricks. The **ReportBuilder** is the instruction manual that snaps them together into a finished model.

### How It Works

Every analysis phase follows the same pattern:

```
1. Create a ReportBuilder (title, session, git hash)
2. Add sections one by one (figures, tables, text, key findings)
3. Call render() to produce a complete HTML document
4. Write the HTML to disk
```

The ReportBuilder handles everything the individual sections don't know about:

- **Table of contents:** Automatically generated from section titles. Each entry is a clickable link that scrolls to that section. The TOC uses a two-column layout so it doesn't dominate the page.

- **Section numbering:** Sections are numbered 1, 2, 3... in the order they were added. The number appears in both the TOC and the section header.

- **Key findings placement:** If a `KeyFindingsSection` is added, it appears above the TOC in a highlighted box — the first thing the reader sees. This is the "executive summary" of the report.

- **Metadata footer:** Every report shows when it was generated (in Central Time — `America/Chicago`), which git commit produced it (truncated to 8 characters), and how long the analysis took to run.

- **Consistent styling:** The CSS is embedded directly in the HTML (no external stylesheet to lose). The design follows SPSS/APA conventions for statistical tables: horizontal rules on top and bottom, no vertical gridlines. The maximum width is 1,100 pixels, centered on the page, with a print media query that hides the TOC and prevents awkward page breaks.

The template is rendered with Jinja2, a Python templating engine. Auto-escaping is disabled because the section content is already HTML — the sections are responsible for their own safety.

**Codebase:** `analysis/report.py` (`ReportBuilder`, `render()`, `write()`, `REPORT_CSS`, `REPORT_TEMPLATE`)

### The Helper Functions

Two helper functions simplify the most common report tasks:

**`make_gt()`** builds a statistical table using [great_tables](https://posit-dev.github.io/great-tables/) with APA styling: 2px solid borders on top and bottom, 1px column header underline, title at 16px, body at 14px. It accepts a Polars DataFrame, column labels, number formats, and an optional source note. The result is a self-contained HTML string ready to wrap in a `TableSection`.

**`make_interactive_table()`** builds a searchable/sortable table using ITables. It handles Polars-specific details like dropping nested column types (List, Struct, Array) that can't serialize to HTML, applying number formats, and renaming columns with human-readable labels.

Both return HTML strings — the report builder doesn't need to know which library generated the table.

**Codebase:** `analysis/report.py` (`make_gt()`, `make_interactive_table()`)

## The Orchestrator: RunContext

Sections and the ReportBuilder handle *content*. The **RunContext** handles *everything else*: where the files go, what metadata gets recorded, and how the phase's execution is tracked.

### The Analogy

Think of RunContext as a film set's **production manager**. The director (the analysis code) decides what scenes to shoot. The actors (the data) perform. But the production manager handles all the logistics: booking the soundstage (creating directories), keeping the production log (capturing console output), tracking the schedule (measuring elapsed time), and making sure the final cut gets delivered to the right theater (writing the report to the right path).

Every analysis phase wraps its execution in a `with RunContext(...) as ctx:` block. When the block starts, RunContext creates directories, writes a methodology primer, and starts capturing console output. When the block ends — whether successfully or after an error — RunContext writes the log, records metadata, and assembles the HTML report.

### Directory Structure

RunContext creates a predictable directory tree for every phase:

```
results/91st_2025-2026/
  91-260318.1/                    ← run directory (run ID)
    01_eda/                       ← phase directory
      plots/                      ← PNG files
      data/                       ← parquet and CSV files
      01_eda_report.html          ← the HTML report
      run_info.json               ← metadata (timing, git hash, params)
      run_log.txt                 ← captured console output
      README.md                   ← auto-generated primer
```

The **run ID** follows the format `{legislature}-{YYMMDD}.{N}`: the legislature number, the date, and a sequential counter for same-day runs. The 91st Legislature's first run on March 18, 2026 would be `91-260318.1`. A second run that day would be `91-260318.2`.

For cross-session analyses (like dynamic IRT, which spans all eight bienniums), the directory structure is flatter:

```
results/cross-session/
  27_dynamic_irt/
    260318.1/                     ← run directory (date.N only, no legislature)
      plots/
      data/
      27_dynamic_irt_report.html
```

### Symlinks: Finding the Latest

After each successful run, RunContext creates **symbolic links** (shortcuts) so you don't have to remember run IDs:

- `results/91st_2025-2026/latest` → `91-260318.1` (the most recent run)
- `results/91st_2025-2026/01_eda_report.html` → `91-260318.1/01_eda/01_eda_report.html` (quick access to any phase's report)

The symlinks update automatically. If you run the pipeline again tomorrow, `latest` will point to the new run. The old run's directory stays intact — nothing is overwritten.

### Console Capture

RunContext replaces Python's `sys.stdout` with a **tee stream** — a stream that sends output to two places simultaneously. Every `print()` statement goes to the terminal (so you can watch the analysis progress) *and* to a string buffer. When the phase finishes, the buffer is written to `run_log.txt`.

The analogy: a court reporter who types everything said in the courtroom. The judge and lawyers hear the words in real time, but the transcript is also preserved for the record.

### CSV Export and Download Links

When a phase wants to make data available for download, it calls `ctx.export_csv()`:

```python
ctx.export_csv(ideal_points_df, "ideal_points_house.csv", "IRT ideal points for House members")
```

This writes the CSV to the `data/` subdirectory and registers it for automatic inclusion in the report. When the report is assembled, a `DownloadSection` appears at the bottom with links to all exported CSVs — each with the description the phase provided.

The export function handles Polars-specific edge cases: it strips nested column types (List, Struct, Array) that CSV can't represent, and optionally renames columns using a human-readable label mapping.

**Codebase:** `analysis/run_context.py` (`RunContext`, `export_csv()`, `generate_run_id()`, `resolve_upstream_dir()`)

### Auto-Generated Primers

Each phase can provide a **primer** — a Markdown document explaining what the analysis does, what inputs it requires, what outputs it produces, and how to interpret the results. RunContext writes this to `README.md` in the phase's output directory.

The primer is the phase's self-documentation. If someone finds a directory full of parquet files and plots six months from now, the README explains what they're looking at.

### Upstream Resolution

Phase 24 (synthesis) needs output from Phase 05 (IRT). Phase 25 (profiles) needs output from Phase 24. How does each phase find its upstream data?

`resolve_upstream_dir()` checks four locations in priority order:

1. **Explicit override** — a CLI flag like `--irt-dir /custom/path`
2. **Same-run sibling** — `results/{run_id}/{upstream_phase}` (same pipeline run)
3. **Phase-level latest** — `results/{upstream_phase}/latest` (most recent run of that phase)
4. **Session-level fallback** — `results/latest/{upstream_phase}`

This means you can run the full pipeline (all phases share a run ID) or run individual phases in isolation (each finds the most recent upstream output automatically).

**Codebase:** `analysis/run_context.py` (`resolve_upstream_dir()`)

## The Dashboard: Putting It All Together

After all phases complete, the pipeline generates a **dashboard** — a single HTML page that ties every phase report together with sidebar navigation.

### How It Works

The dashboard is an `index.html` file at the root of the run directory. It has three parts:

- **Header:** The session name, run ID, total elapsed time across all phases, git hash, and timestamp.

- **Sidebar:** A scrollable list of all phases in pipeline execution order. Each entry shows the phase name, its label ("Exploratory Data Analysis," "Bayesian IRT (1D)," etc.), and how long that phase took. The first phase is highlighted by default.

- **Content area:** A single `<iframe>` that loads the selected phase's report. Clicking a sidebar entry updates the iframe's source URL — no page reload, no navigation away from the dashboard.

The analogy: a book with a table of contents on the left margin that stays visible as you flip through chapters. Click "Chapter 5: IRT" and the right side of the page shows the IRT report. Click "Chapter 24: Synthesis" and you see the narrative report.

### Phase Order

The sidebar follows **pipeline execution order**, not directory number order. This matters because the phases aren't numbered sequentially by dependency — Phase 07b (Hierarchical 2D IRT) runs after Phase 07 (Hierarchical 1D), and Phase 06 (canonical routing) runs between them. The dashboard's `PHASE_ORDER` constant defines the display sequence explicitly, matching the order a reader would naturally want to follow the analysis.

**Codebase:** `analysis/dashboard.py` (`generate_dashboard()`, `PHASE_ORDER`)

## The Styling Philosophy

The visual design of Tallgrass reports follows three principles:

**Self-contained.** Every report is a single HTML file with inline CSS and base64-embedded images. No CDN dependencies, no external stylesheets, no JavaScript that requires a network connection. The file works offline, on any device, indefinitely.

**Accessible.** WCAG 2.1 AA compliance: alt text on all figures, sufficient color contrast, semantic HTML structure. The partisan color scheme (Republican red `#E81B23`, Democrat blue `#0015BC`) is used consistently across all phases and has been tested for contrast against white backgrounds. Screen readers can navigate via the heading hierarchy and table of contents.

**Print-friendly.** A print media query hides the table of contents (which is redundant in print) and prevents section breaks from splitting a figure across pages. The 1,100px maximum width ensures the layout fits on standard paper without horizontal scrolling.

The design deliberately avoids visual complexity. There are no gradients, no shadows, no animated transitions (except in scrollytelling mode). The goal is a document that looks like it belongs in a university research library, not a marketing brochure.

## What Can Go Wrong

### Missing Upstream Data

If a phase didn't produce output (it failed, or was skipped), the synthesis and dashboard handle it gracefully. The dashboard simply omits that phase from the sidebar. The synthesis report skips sections that depend on missing data — it doesn't crash, and it documents what was unavailable.

### Large Reports

Reports with many interactive tables or high-resolution figures can exceed 10 MB. The base64 encoding of images adds ~33% overhead (a 750 KB PNG becomes a 1 MB base64 string). For sessions with both House and Senate analyses across all phases, the synthesis report can be large enough to trigger browser slowness on older machines.

### Symlink Conflicts

On operating systems that don't support symbolic links (some Windows configurations), the `latest` symlinks fail silently. The reports themselves are unaffected — you just need to navigate to the run directory by its full name instead of using `latest`.

---

## Key Takeaway

The report system transforms analysis output into self-contained HTML documents using eight section types (from static tables to scrollytelling), assembled by a ReportBuilder and orchestrated by RunContext. Each phase's output is organized into a predictable directory structure with auto-generated metadata, console logs, and methodology primers. The dashboard ties all phase reports together with sidebar navigation, giving readers a single entry point to the entire analysis.

---

*Terms introduced: section type (table, figure, text, interactive table, interactive, key findings, download, scrollytelling), ReportBuilder, RunContext, run ID, tee stream, primer, upstream resolution, dashboard, PHASE_ORDER, self-contained HTML, base64 embedding, WCAG 2.1 AA, symlink*

*Next: [Synthesis: The Session Story](ch02-synthesis.md)*
