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
  output.py    - CSV export (3 files: votes, rollcalls, legislators)
  cli.py       - argparse CLI entry point
```

Pipeline: `get_bill_urls()` → `_filter_bills_with_votes()` → `get_vote_links()` → `parse_vote_pages()` → `enrich_legislators()` → `save_csvs()`

## Concurrency Pattern

All HTTP fetching uses a two-phase pattern: concurrent fetch via ThreadPoolExecutor (MAX_WORKERS=5), then sequential parse. Rate limiting is thread-safe via `threading.Lock()`. Never mutate shared state during the fetch phase.

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
- Vote categories: Yea, Nay, Present and Passing, Absent and Not Voting, Not Voting (exactly these 5)
- Legislator slugs encode chamber: `sen_` prefix = Senate, `rep_` prefix = House

## Output

Three CSVs in `data/ks_{session}/`:
- `ks_{session}_votes.csv` — ~68K rows, one per legislator per roll call
- `ks_{session}_rollcalls.csv` — ~500 rows, one per roll call
- `ks_{session}_legislators.csv` — ~172 rows, one per legislator

Cache lives in `data/ks_{session}/.cache/`. Use `--clear-cache` to force fresh fetches.

## Testing

No test suite yet. Verify manually:
- `uv run ruff check src/` — lint clean
- Run scraper with `--clear-cache`, check that `vote_date`, `chamber`, `motion`, `bill_title` are populated
- Check legislators CSV: party distribution includes both Republican and Democrat
- Spot-check SB 1: should show Senate, Emergency Final Action, Passed as amended
