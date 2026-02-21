# KS Vote Scraper

Scrapes Kansas Legislature roll call votes from kslegislature.gov into CSV files for statistical/Bayesian analysis.

## Commits

- **No Co-Authored-By lines.** Never append co-author trailers.
- Use conventional commits with version tags: `type(scope): description [vYYYY.MM.DD.N]`
- Never push without explicit permission.
- See `.claude/rules/commit-workflow.md` for full details.

## Commands

```bash
just scrape 2025                             # scrape (cached)
just scrape-fresh 2025                       # scrape (fresh)
just lint                                    # lint + format
just lint-check                              # check only
just sessions                                # list available sessions
just check                                   # full check
just synthesis                               # run synthesis report
uv run ks-vote-scraper 2023                  # historical session
uv run ks-vote-scraper 2024 --special        # special session
```

## Code Style

- Python 3.14+, use modern type hints (`list[str]` not `List[str]`, `X | None` not `Optional[X]`)
- Ruff: line-length 100, rules E/F/I/W
- Frozen dataclasses for data models
- Type hints on all function signatures

## Architecture

```
src/ks_vote_scraper/
  config.py    - Constants (BASE_URL, delays, workers, user agent)
  session.py   - KSSession: biennium URL resolution (current vs historical vs special)
  models.py    - IndividualVote + RollCall dataclasses
  scraper.py   - KSVoteScraper: 4-step pipeline (bill URLs → API filter → vote parse → legislator enrich)
               - FetchResult/FetchFailure/VoteLink dataclasses: typed HTTP results + vote page links
               - Module constants: _BILL_URL_RE, VOTE_CATEGORIES, _normalize_bill_code()
  output.py    - CSV export (3 files: votes, rollcalls, legislators)
  cli.py       - argparse CLI entry point
```

Pipeline: `get_bill_urls()` → `_filter_bills_with_votes()` → `get_vote_links()` → `parse_vote_pages()` → `enrich_legislators()` → `save_csvs()`

## Concurrency Pattern

All HTTP fetching uses a two-phase pattern: concurrent fetch via ThreadPoolExecutor (MAX_WORKERS=5), then sequential parse. Rate limiting is thread-safe via `threading.Lock()`. Never mutate shared state during the fetch phase.

### Retry Strategy

`_get()` returns a `FetchResult` (not raw HTML) and classifies errors for differentiated retry behavior:

- **404** → `"permanent"`, max 2 attempts (one retry for transient routing guards)
- **5xx** → `"transient"`, exponential backoff (`RETRY_DELAY * 2^attempt`)
- **Timeout** → `"timeout"`, exponential backoff
- **Connection error** → `"connection"`, fixed delay
- **Other 4xx** → `"permanent"`, no retry

Failed vote page fetches are recorded as `FetchFailure` with bill context (bill number, motion text, bill path) and written to a JSON failure manifest at the end of the run.

## HTML Parsing Pitfalls (Hard-Won Lessons)

These are real bugs that were found and fixed. Do NOT regress on them:

1. **Tag hierarchy on vote pages is NOT what you'd expect.** The vote page uses `<h2>` for bill number, `<h4>` for bill title, and `<h3>` for chamber/date/motion AND vote category headings (Yea, Nay, etc). If you search `<h2>` for title or motion data, you get nothing.

2. **Party detection via full page text will always match "Republican".** Every legislator page has a party filter dropdown containing `<option value="republican">Republican</option>`. Searching `page.get_text()` for "Republican" matches this dropdown for ALL legislators. Must parse the specific `<h2>` containing "District \d+" (e.g., `"District 27 - Republican"`).

2b. **Legislator `<h1>` is NOT the member name.** The first `<h1>` on member pages is a generic "Legislators" nav heading. The actual name is in a later `<h1>` starting with "Senator " or "Representative ". Must use `soup.find("h1", string=re.compile(r"^(Senator|Representative)\s+"))`. Also strip leadership suffixes like " - House Minority Caucus Chair".

3. **Vote category parsing requires scanning BOTH h2 and h3.** The `Yea - (33):` heading can appear as either `<h2>` or `<h3>` depending on the page. The parser correctly uses `soup.find_all(["h2", "h3", "a"])` — do not simplify this to only one tag.

4. **KLISS API response structure varies.** Sometimes it's a raw list, sometimes `{"content": [...]}`. Always handle both: `data if isinstance(data, list) else data.get("content", [])`.

## Session URL Logic

The KS Legislature uses different URL prefixes per session — this is the single trickiest part of the scraper:

- Current (2025-26): `/li/b2025_26/...`
- Historical (2023-24): `/li_2024/b2023_24/...`
- Special (2024): `/li_2024s/...`
- API paths also differ: current uses `/li/api/v13/rev-1`, historical uses `/li_{end_year}/api/v13/rev-1`

`CURRENT_BIENNIUM_START` in session.py must be updated when a new biennium becomes current (next: 2027).

## Data Model Notes

- `vote_id` encodes a timestamp: `je_20250320203513` → `2025-03-20T20:35:13`
- `bill_metadata` (short_title, sponsor) comes from the KLISS API — already fetched during pre-filtering, no extra requests needed
- `passed` is derived from result text: passed/adopted/prevailed/concurred → True; failed/rejected/sustained → False
- Vote categories: defined in `VOTE_CATEGORIES` constant — Yea, Nay, Present and Passing, Absent and Not Voting, Not Voting (exactly these 5)
- Legislator slugs encode chamber: `sen_` prefix = Senate, `rep_` prefix = House

## Output

Three CSVs in `data/{legislature}_{start}-{end}/` (e.g. `data/91st_2025-2026/`):
- `{output_name}_votes.csv` — ~68K rows, one per legislator per roll call
- `{output_name}_rollcalls.csv` — ~500 rows, one per roll call
- `{output_name}_legislators.csv` — ~172 rows, one per legislator

Cache lives in `data/{output_name}/.cache/`. Use `--clear-cache` to force fresh fetches.

The directory naming uses the Kansas Legislature numbering scheme: `(start_year - 1879) // 2 + 18` gives the legislature number (e.g., 91 for 2025-2026). Special sessions use `{year}s` (e.g., `2024s`).

When vote page fetches fail, a `failure_manifest.json` is written alongside the CSVs with per-failure context (bill number, motion, URL, status code, error type). Failed pages are never cached, so re-running the scraper automatically retries them.

## Results Directory

Analysis outputs go in `results/` (gitignored), organized by session, analysis type, and run date:

```
results/
  91st_2025-2026/
    eda/
      2026-02-19/
        plots/                  ← PNGs
        data/                   ← Parquet intermediates
        filtering_manifest.json ← What was filtered and why
        run_info.json           ← Git hash, timestamp, parameters
        run_log.txt             ← Captured console output
      latest → 2026-02-19/     ← Symlink to most recent run
    pca/                        ← Same structure
    irt/
    clustering/
    network/
    prediction/
    indices/
    synthesis/                  ← Joins all phases into one narrative report
```

All analysis scripts use `RunContext` from `analysis/run_context.py` as a context manager to get structured output. Downstream scripts read from `results/<session>/<analysis>/latest/`.

Results paths use the biennium naming scheme: `91st_2025-2026` (matching the data directory).

## Architecture Decision Records

Significant technical decisions are documented in `docs/adr/`. See `docs/adr/README.md` for the full index and template.

## Analysis Design Choices

Each analysis phase has a design document in `analysis/design/` recording the statistical assumptions, priors, thresholds, and methodological choices that shape results and carry forward into downstream phases. **Read these before interpreting results or adding a new phase.** See `analysis/design/README.md` for the index.

- `analysis/design/eda.md` — Binary encoding, filtering thresholds, agreement metrics
- `analysis/design/pca.md` — Imputation, standardization, sign convention, holdout design
- `analysis/design/irt.md` — Priors (Normal(0,1) discrimination, anchors), MCMC settings, missing data handling
- `analysis/design/clustering.md` — Three methods for robustness, party loyalty metric, k=2 finding

## Analytics

The `Analytic_Methods/` directory contains 28 documents covering every analytical method applicable to our data. See `Analytic_Methods/00_overview.md` for the full index and recommended pipeline.

### Document Naming Convention
```
NN_CAT_method_name.md
```
Categories: `DATA`, `EDA`, `IDX`, `DIM`, `BAY`, `CLU`, `NET`, `PRD`, `TSA`

### Key Data Structures for Analysis

The **vote matrix** (legislators x roll calls, binary) is the foundation. Build it from `votes.csv` by pivoting `legislator_slug` x `vote_id` with Yea=1, Nay=0, absent=NaN.

**Critical preprocessing:**
- Filter near-unanimous votes (minority < 2.5%) — they carry no ideological signal
- Filter legislators with < 20 votes — unreliable estimates
- Analyze chambers separately (House and Senate vote on different bills)

### Technology Preferences
- **Polars over pandas** for all data manipulation
- **Python over R** — no rpy2 or Rscript; use Python-native methods (PCA, Bayesian IRT) instead of R-only ones (W-NOMINATE, OC)

### Analysis Libraries
```bash
uv add polars numpy scipy matplotlib seaborn scikit-learn  # Phase 1-2
uv add networkx python-louvain prince umap-learn           # Phase 3-5
uv add pymc arviz                                          # Phase 4 (Bayesian)
uv add xgboost shap ruptures                               # Phase 6
```

### HTML Report System

Each analysis phase produces a self-contained HTML report (`{analysis}_report.html`) with SPSS/APA-style tables and embedded plots. Architecture:

- `analysis/report.py` — Generic: section types (`TableSection`, `FigureSection`, `TextSection`), `ReportBuilder`, `make_gt()` helper, Jinja2 template + CSS.
- `analysis/eda_report.py` — EDA-specific: `build_eda_report()` adds ~19 sections.
- `analysis/synthesis.py` + `analysis/synthesis_report.py` — Synthesis: loads upstream parquets from all 7 phases, joins into unified legislator DataFrames, produces 22-section narrative HTML report for nontechnical audiences.
- `RunContext` auto-writes the HTML in `finalize()` if sections were added.

Tables use great_tables with polars DataFrames (no pandas conversion). Plots are base64-embedded PNGs. See ADR-0004 for rationale.

### Analytic Flags

`docs/analytic-flags.md` is a living document that accumulates observations across analysis phases. Add an entry whenever you encounter:
- **Outlier legislators** — extreme scores, unusual voting patterns, imputation artifacts
- **Methodological flags** — legislators needing special handling (e.g., bridging observations for cross-chamber members, low-participation exclusions)
- **Unexpected patterns** — dimensions or clusters that need qualitative explanation
- **Downstream actions** — things to check or handle differently in later phases (IRT, clustering, network)

Each entry records what was observed, which phase found it, why it matters, and what to do about it. Check this file at the start of each new analysis phase for prior flags that affect the current work.

### Kansas-Specific Analysis Notes
- Republican supermajority (~72%) means intra-party variation is more interesting than inter-party
- Clustering (2026-02-20) found k=2 optimal (party split); initial k=3 hypothesis rejected — intra-R variation is continuous
- 34 veto override votes are analytically rich (cross-party coalitions, 2/3 threshold)
- Beta-Binomial and Bayesian IRT are the recommended Bayesian starting points

## Testing

No test suite yet. Verify manually:
- `uv run ruff check src/` — lint clean
- Run scraper with `--clear-cache`, check that `vote_date`, `chamber`, `motion`, `bill_title` are populated
- Check legislators CSV: party distribution includes both Republican and Democrat
- Spot-check SB 1: should show Senate, Emergency Final Action, Passed as amended
