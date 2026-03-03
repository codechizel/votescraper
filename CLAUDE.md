# Tallgrass

Kansas Legislature roll call vote scraper + analysis platform. Scrapes kslegislature.gov into CSV files for statistical/Bayesian analysis. Coverage: 2011-2026 (84th-91st legislatures).

## Commits

- **No Co-Authored-By lines.** Never append co-author trailers.
- Use conventional commits with version tags: `type(scope): description [vYYYY.MM.DD.N]`
- **After every feature/fix:** update relevant docs (CLAUDE.md, ADRs, design docs) before committing. Code and docs ship in the same commit.
- Never push without explicit permission.
- See `.claude/rules/commit-workflow.md` for types, scopes, and full details.

## Commands

[Just](https://github.com/casey/just) is used as a command runner — thin aliases over `uv run` commands. The `Justfile` also sets `OMP_NUM_THREADS=6` and `OPENBLAS_NUM_THREADS=6` globally to cap thread pools on Apple Silicon (ADR-0022). Run `just --list` to see all recipes, or use the underlying `uv run` commands directly.

```bash
just scrape 2025                             # → uv run tallgrass 2025
just scrape-fresh 2025                       # → uv run tallgrass --clear-cache 2025
just text 2025                               # → uv run tallgrass-text 2025 (bill text retrieval)
just kanfocus 1999                           # → uv run tallgrass-kanfocus 1999 (KanFocus vote scrape)
just alec                                    # → uv run tallgrass-alec (ALEC model legislation scrape)
just lint                                    # → ruff check --fix + ruff format
just lint-check                              # → ruff check + ruff format --check
just typecheck                               # → ty check src/ + ty check analysis/
just sessions                                # → uv run tallgrass --list-sessions
just check                                   # → lint-check + typecheck + test (quality gate)
just test                                    # → uv run pytest tests/ -v (~2458 tests)
just test-scraper                            # → pytest -m scraper (~312 tests)
just test-fast                               # → pytest -m "not slow" (skip integration)
just monitor                                 # → check running experiment status
just roster-sync                             # → sync OpenStates legislator roster (slug→ocd_id)
just merge-special 2020                      # → merge 2020 special into 88th biennium
just merge-special all                       # → merge all 5 specials into parent bienniums
just pipeline 2025-26                        # → single-biennium pipeline (phases 01-25)
just cross-pipeline                          # → cross-biennium pipeline (phases 26-27)
uv run tallgrass 2023                  # historical session (direct)
uv run tallgrass 2024 --special        # special session (direct)
uv run tallgrass-text 2025             # bill text retrieval (direct)
uv run tallgrass-kanfocus 1999         # KanFocus scrape (direct)
uv run tallgrass-kanfocus 2011 --mode gap-fill  # fill 84th gaps
```

Analysis recipes (all pass `*args` through to the underlying script):

`just eda`, `just pca`, `just mca`, `just umap`, `just irt`, `just irt-2d`, `just ppc`, `just clustering`, `just lca`, `just network`, `just bipartite`, `just indices`, `just betabinom`, `just hierarchical`, `just synthesis`, `just profiles`, `just tsa`, `just cross-session`, `just external-validation`, `just dime`, `just dynamic-irt`, `just wnominate`, `just text-analysis`, `just tbip`, `just issue-irt`, `just model-legislation`.

Each maps to `uv run python analysis/NN_phase/phase.py`. Example: `just profiles --names "Masterson"` runs `uv run python analysis/25_profiles/profiles.py --names "Masterson"`.

`just pipeline 2025-26` runs phases 01-25 in order under a single run ID (ADR-0052, ADR-0091). Each phase gets `--run-id` automatically. Phases 06, 08, 16-23 gracefully skip when prerequisites are missing (no R, no bill texts, biennium out of range for SM/DIME).

`just cross-pipeline` runs phases 26-27 (cross-session + dynamic IRT). Separate because they require data from multiple bienniums.

Database recipes (require `uv sync --group web` + Docker):

```bash
just db-up                                   # start PostgreSQL (docker compose)
just db-down                                 # stop PostgreSQL
just db-migrate                              # apply Django migrations
just db-makemigrations                       # generate new migrations
just db-admin                                # Django admin at localhost:8000/admin/
just db-createsuperuser                      # create admin user
just db-shell                                # psql into local database
just django <cmd>                            # manage.py passthrough
just test-web                                # Django tests only (-m web)
```

## Worktrees

Git worktrees for branch isolation. See `.claude/rules/worktree-workflow.md` for full rules.

```bash
just wt-new feature-name                     # create .claude/worktrees/feature-name/
just wt-done feature-name                    # merge to main + cleanup (from main repo — prevents CWD death)
just wt-done                                 # same, but auto-detects (from inside worktree only)
```

**Hard rules:** Never `git checkout main` from a worktree. Never remove a worktree while CWD is inside it. Never `git push origin branch:main`. Claude Code sessions **must** call `just wt-done <name>` from the main repo, never from inside the worktree.

## Build Philosophy

- **Check for existing open source solutions first.** Don't reinvent the wheel, but don't force a shoehorned dependency either.

## Code Style

- Python 3.14.3+, modern type hints (`list[str]` not `List[str]`, `X | None` not `Optional[X]`)
- Ruff: line-length 100, rules E/F/I/W. Multi-exception `except` uses PEP 758 bracketless syntax: `except ValueError, TypeError:` (no parens needed in 3.14+). Ruff formats this automatically.
- ty: type checking (beta) — `src/` must pass clean; `analysis/` warnings-only for third-party stub noise
- Frozen dataclasses for data models; type hints on all function signatures
- Libraries with incomplete stubs configured as `replace-imports-with-any` in `pyproject.toml`

## Architecture

```
src/tallgrass/
  config.py     - Constants (BASE_URL, delays, workers, user agent)
  session.py    - KSSession: biennium URL resolution, STATE_DIR, data_dir/results_dir
  bills.py      - Shared bill discovery (HTML + JS fallback), used by scraper + text adapter
  models.py     - IndividualVote + RollCall + BillAction dataclasses
  scraper.py    - KSVoteScraper: 4-step pipeline (bill URLs -> API filter -> vote parse -> enrich)
  odt_parser.py - ODT vote file parser (2011-2014): pure functions, no I/O
  roster.py     - OpenStates sync: slug→ocd_id mapping, cached JSON (ADR-0085)
  output.py     - CSV export (4 files: votes, rollcalls, legislators, bill_actions)
  merge_special.py - Post-scrape special session merge into parent bienniums
  cli.py        - argparse CLI entry point
  text/
    __init__.py   - Public API re-exports
    models.py     - BillDocumentRef + BillText frozen dataclasses
    protocol.py   - StateAdapter Protocol (multi-state-ready)
    kansas.py     - KansasAdapter: bill discovery + PDF URL construction
    fetcher.py    - BillTextFetcher: concurrent download + text extraction
    extractors.py - PDF extraction + legislative text cleaning (pure functions)
    openstates.py - OpenStates multi-state adapter (MO, OK, NE, CO via API v3)
    output.py     - CSV export (bill_texts.csv)
    cli.py        - tallgrass-text entry point
  alec/
    __init__.py   - Public API re-exports
    models.py     - ALECModelBill frozen dataclass
    scraper.py    - Paginated listing scraper + bill text extraction
    output.py     - CSV export (alec_model_bills.csv)
    cli.py        - tallgrass-alec entry point
  kanfocus/
    __init__.py   - Public API re-exports
    models.py     - KanFocusVoteRecord + KanFocusLegislator frozen dataclasses
    session.py    - KanFocus session ID mapping, URL construction, vote_id generation
    parser.py     - parse_vote_page(): HTML-to-text + regex extraction (BeautifulSoup)
    slugs.py      - Slug generation from "Name, R-32nd" format + cross-ref matching
    fetcher.py    - KanFocusFetcher: Chrome cookie auth + HTTP cache + rate limiting
    output.py     - Convert intermediates → standard IndividualVote/RollCall + gap-fill merge
    cli.py        - tallgrass-kanfocus entry point + data archiving
```

Vote scraper pipeline: `get_bill_urls()` -> `_filter_bills_with_votes()` -> `get_vote_links()` -> `parse_vote_pages()` -> `enrich_legislators()` -> `save_csvs()`

Bill text pipeline: `KansasAdapter.discover_bills()` -> `BillTextFetcher.fetch_all()` -> `save_bill_texts()`

KanFocus pipeline: `KanFocusFetcher.fetch_biennium()` -> `parse_vote_page()` -> `convert_to_standard()` -> `save_csvs()` (or `merge_gap_fill()` for gap-fill mode). Coverage: 78th-91st (1999-2026). Requires Chrome with active KanFocus login (fetcher extracts cookies from Chrome's encrypted cookie database on macOS via Keychain). Parser auto-detects HTML input and converts to text via BeautifulSoup. Raw HTML archived to `data/kanfocus_archive/` after each run. See ADR-0088.

ALEC pipeline: `enumerate_bills()` -> `fetch_bill_texts()` -> `save_alec_bills()`. Scrapes alec.org/model-policy/ (~1,061 model policies). Cached HTML at `data/external/alec/.cache/`. See ADR-0089.

Static parsing helpers (all `@staticmethod` on `KSVoteScraper`): `_extract_bill_title()`, `_extract_chamber_motion_date()`, `_parse_vote_categories()`, `_extract_party_and_district()`. Each docstring references the HTML pitfalls it handles. Tests call these directly.

See `.claude/rules/scraper-architecture.md` for session coverage table, retry strategy, concurrency details, and ODT parsing.

## HTML Parsing Pitfalls (Hard-Won Lessons)

These are real bugs that were found and fixed. Do NOT regress on them:

1. **Tag hierarchy on vote pages is NOT what you'd expect.** `<h2>` = bill number, `<h4>` = bill title, `<h3>` = chamber/date/motion AND vote category headings.

2. **Party detection via full page text will always match "Republican".** Every legislator page has a party filter dropdown. Must parse the specific `<h2>` containing "District \d+".

2b. **Legislator `<h1>` is NOT the member name.** First `<h1>` is a nav heading. Must use `soup.find("h1", string=re.compile(r"^(Senator|Representative)\s+"))`. Also strip leadership suffixes.

3. **Vote category parsing requires scanning BOTH h2 and h3.** Uses `soup.find_all(["h2", "h3", "a"])` — do not simplify.

4. **KLISS API response structure varies.** Raw list or `{"content": [...]}`. Always handle both.

5. **Pre-2015 party detection uses `<h3>Party: Republican</h3>`** instead of `<h2>District N - Republican</h2>`.

6. **Pre-2021 bill lists are JavaScript-rendered.** JS fallback fetches `bills_li_{end_year}.js` data files.

6b. **Pre-2021 JS data uses two key formats.** 88th uses quoted JSON keys; 87th and earlier use unquoted JS object literal syntax.

6c. **JS data files live at `/m/` not `/s/` for all sessions except the 88th.**

7. **Pre-2015 vote pages are ODT files, not HTML.** ZIP archives with `content.xml`. House/Senate use different vote category names.

8. **Pre-2021 member directories are JavaScript-rendered.** Same unquoted-key issue as bill data.

9. **KS Legislature server returns HTML error pages with HTTP 200 for binary URLs.** `_get()` checks `content[:5]` for `<html` prefix.

10. **84th session ODTs often lack individual vote data.** ~30% are committee-of-the-whole (tally-only). Not a parser bug.

## Session URL Logic

- Current (2025-26): `/li/b2025_26/...`
- Historical (2023-24): `/li_2024/b2023_24/...`
- Special (2024): `/li_2024s/...`
- API: current uses `/li/api/v13/rev-1`, historical uses `/li_{end_year}/api/v13/rev-1`
- `CURRENT_BIENNIUM_START` in session.py must be updated when a new biennium becomes current (next: 2027).

## Data Model

- `vote_id` encodes a timestamp: `je_20250320203513` -> `2025-03-20T20:35:13`. KanFocus-sourced data uses `kf_{vote_num}_{year}_{chamber}` (e.g. `kf_33_2011_S`) — see ADR-0088.
- `passed`: passed/adopted/prevailed/concurred -> True; failed/rejected/sustained -> False; else null
- Vote categories: Yea, Nay, Present and Passing, Absent and Not Voting, Not Voting (exactly 5). KanFocus has 4 categories (no "Absent and Not Voting" — maps "Not Voting" uniformly).
- Legislator slugs: `sen_` = Senate, `rep_` = House
- Column naming: scraper CSVs use `slug` and `vote`; analysis phases rename to `legislator_slug` and expect `vote` (not `vote_category`). Each phase handles the rename at load time (ADR-0066).
- Independent party handling: scraper outputs empty string; all analysis fills to "Independent" at load time (ADR-0021)
- `sponsor_slugs`: semicolon-joined legislator slugs extracted from bill page `<a>` hrefs (e.g. `"sen_tyson_caryn_1; sen_alley_larry_1"`). Empty for committee sponsors, pre-89th sessions, and data scraped before this feature. Used by Phase 15 (`sponsor_party_R` for prediction), Phase 24 (`n_bills_sponsored` in scorecard), and Phase 25 (per-legislator sponsorship section + defection sponsor display). Text-based fallback via `phase_utils.match_sponsor_to_party()` when slugs unavailable.
- `BillAction`: KLISS API HISTORY data — `action_code`, `chamber`, `committee_names` (tuple, semicolon-joined in CSV), `occurred_datetime`, `session_date`, `status`, `journal_page_number`. Available for sessions with KLISS API (89th+); pre-KLISS sessions (84th-88th) may have limited or no HISTORY data.
- `ocd_id`: OpenStates OCD person ID (`ocd-person/{uuid}`) — stable cross-biennium legislator identifier. Populated by `enrich_legislators()` from cached `ks_slug_to_ocd.json` (run `just roster-sync` first). Empty if roster not synced. Used by Phase 26 (cross-session matching) and Phase 27 (dynamic IRT global roster) for identity resolution that handles same-name legislators and redistricting. See ADR-0085.

## Output

Five CSVs in `data/kansas/{legislature}_{start}-{end}/`:
- `{name}_votes.csv` — one per legislator per roll call (deduped by `legislator_slug` + `vote_id`)
- `{name}_rollcalls.csv` — one per roll call
- `{name}_legislators.csv` — one per legislator (8 columns: name, full_name, slug, chamber, party, district, member_url, ocd_id)
- `{name}_bill_actions.csv` — one per bill lifecycle action (KLISS API HISTORY; all bills, not just those with votes)
- `{name}_bill_texts.csv` — one per bill document (introduced + supp notes; joins to rollcalls on `bill_number`). Columns: `session`, `bill_number`, `document_type`, `version`, `text`, `page_count`, `source_url`. Generated by `tallgrass-text`, not the vote scraper.

Directory naming: `(start_year - 1879) // 2 + 18` -> legislature number. Special sessions: `{year}s`.
Special session merge: `just merge-special all` merges special session CSVs into parent biennium directories (ADR-0082). Idempotent — filters by `session` column before concat. Run after scraping specials, before running the parent's pipeline.
Cache: `data/kansas/{name}/.cache/`. Failed fetches -> `failure_manifest.json` + `missing_votes.md`. Bill text cache: `data/kansas/{name}/.cache/text/`. KanFocus cache: `data/kansas/{name}/.cache/kanfocus/` (hash-keyed HTML files, restart-safe).
KanFocus archive: `data/kanfocus_archive/{name}/` — permanent copy of raw HTML cache, auto-created after each successful run. Takes hours per biennium to rebuild; `--clear-cache` blocked unless archive exists.
External data: `data/external/shor_mccarty.tab` (Shor-McCarty scores, auto-downloaded from Harvard Dataverse).
External data: `data/external/dime_recipients_1979_2024.csv` (DIME CFscores, manually placed, ODC-BY license).
External data: `data/external/openstates/ks_slug_to_ocd.json` (OpenStates slug→ocd_id mapping, auto-synced via `just roster-sync`, CC0 license — ADR-0085).
External data: `data/external/alec/alec_model_bills.csv` (ALEC model policy corpus, scraped via `just alec`, public data — ADR-0089).

## Results Directory

Output layout (ADR-0052):

**Run-directory mode** (all biennium sessions): all phases grouped under a run ID.
`results/kansas/{session}/{run_id}/{NN_phase}/` with a session-level `latest` symlink (e.g. `results/kansas/91st_2025-2026/91-260228.1/01_eda/`). Run ID format: `{bb}-{YYMMDD}.{n}` where n starts at 1 (e.g. `91-260228.1`, second run same day: `91-260228.2`). When running a single phase without `--run-id`, a run_id is auto-generated so the directory structure is always consistent.

**Flat mode** (cross-session and special sessions): each analysis writes to its own date directory.
`results/kansas/cross-session/{aa}-vs-{bb}/{YYMMDD}.{n}/` where `aa`/`bb` are legislature numbers. Flat structure (no phase subdirectory). Example: `results/kansas/cross-session/90-vs-91/260226.1/`.

Both modes use `RunContext` for structured output, elapsed timing, HTML reports, and auto-primers. `resolve_upstream_dir()` handles 4-level path resolution (CLI override → run_id → flat/latest → latest/phase) so phases find their upstream data in either layout.

Experiments in `results/experimental_lab/YYYY-MM-DD_short-description/`. Each contains `experiment.md` (structured record from TEMPLATE.md), `run_experiment.py`, and `run_NN_description/` output directories.

## Analysis Pipeline

27 phases in numbered subdirectories (`analysis/01_eda/` through `analysis/27_dynamic_irt/`), organized by logical stage (ADR-0091). A PEP 302 meta-path finder in `analysis/__init__.py` redirects `from analysis.eda import X` to `analysis/01_eda/eda.py` — zero import changes needed (ADR-0030). Shared infrastructure (`run_context.py`, `report.py`, `phase_utils.py`) stays at the root. `phase_utils.py` provides `print_header()`, `save_fig()`, `load_metadata()`, `load_legislators()`, `normalize_name()`, `parse_sponsor_name()`, `match_sponsor_to_slug()`, and `match_sponsor_to_party()` — all phases import from here instead of defining locally.

**Single-biennium pipeline** (`just pipeline`, phases 01-25): EDA → PCA → MCA → UMAP → IRT → 2D IRT → Hierarchical IRT → PPC → Clustering → LCA → Network → Bipartite → Indices → Beta-Binomial → Prediction → W-NOMINATE → External Validation → DIME → TSA → Bill Text → TBIP → Issue IRT → Model Legislation → Synthesis → Profiles. Phases 06, 08, 16-23 gracefully skip when prerequisites are missing.

**Cross-biennium pipeline** (`just cross-pipeline`, phases 26-27): Cross-Session → Dynamic IRT. Requires data from multiple bienniums.

Phase `06_irt_2d` is 2D Bayesian IRT with PLT identification, relaxed thresholds (ADR-0054; gracefully skips in pipeline when Phase 05 output missing — ADR-0074). Phase `27_dynamic_irt` is Martin-Quinn state-space IRT across bienniums (ADR-0058; post-hoc sign correction via static IRT correlation — ADR-0068). Phase `18_dime` validates IRT against DIME/CFscores (campaign-finance ideology, 84th-89th bienniums — ADR-0062). Phase `08_ppc` is PPC + LOO-CV model comparison — loads InferenceData from all three IRT phases, runs posterior predictive checks, Yen's Q3, and PSIS-LOO (ADR-0063). Phase `16_wnominate` compares IRT to W-NOMINATE + Optimal Classification via R subprocess (ADR-0059). Phase `10_lca` is Latent Class Analysis (Bernoulli mixture on binary vote matrix via StepMix, BIC model selection, Salsa effect detection, class membership tables with IRT ideal points). Phase `12_bipartite` is bipartite bill-legislator network analysis (bill polarization, bridge bills, BiCM backbone extraction, bill communities — ADR-0065).

Phase `20_bill_text` is bill text NLP analysis — BERTopic topic modeling (FastEmbed + HDBSCAN + c-TF-IDF), optional CAP classification via Claude API, bill similarity from embeddings, and vote cross-reference (Rice index per topic × party, caucus-splitting scores). Gracefully skips when `bill_texts.csv` missing. Design: `analysis/design/bill_text.md`.

Phase `21_tbip` is text-based ideal points — embedding-vote approach (not true TBIP due to ~92% committee sponsorship). Multiplies vote matrix by Phase 20 bill embeddings, PCA on legislator text profiles, PC1 = text-derived ideal point. Validates against IRT (flat + hierarchical). Gracefully skips when Phase 20 output missing. Design: `analysis/design/tbip.md`, ADR-0086.

Phase `22_issue_irt` is issue-specific ideal points — topic-stratified flat IRT on per-topic vote subsets from Phase 20 BERTopic/CAP topics. Reuses Phase 05 `build_irt_graph()` / `build_and_sample()` — zero new model code. Two taxonomies: BERTopic (data-driven) + CAP (standardized). Relaxed thresholds (R-hat < 1.05, ESS > 200). Gracefully skips when Phase 20 output missing. Design: `analysis/design/issue_irt.md`, ADR-0087.

Phase `23_model_legislation` is model legislation detection (BT5) — cosine similarity between Kansas bill embeddings and ALEC model policy corpus + neighbor state bills (MO, OK, NE, CO via OpenStates API). Three-tier matching (near-identical >= 0.95, strong >= 0.85, related >= 0.70). N-gram overlap confirmation for strong matches. Cross-references with Phase 20 topics. Gracefully skips when bill texts + ALEC corpus both missing. Design: `analysis/design/model_legislation.md`, ADR-0089.

See `.claude/rules/analysis-framework.md` for the full pipeline, report system architecture, and design doc index. See `.claude/rules/analytic-workflow.md` for methodology rules, validation requirements, and audience guidance.

Key references:
- Design docs: `analysis/design/README.md`
- ADRs: `docs/adr/README.md` (91 decisions)
- Analysis primer: `docs/analysis-primer.md` (plain-English guide)
- How IRT works: `docs/how-irt-works.md` (general-audience explanation of anchors, identification, and MCMC divergences)
- External validation: `docs/external-validation-results.md` (5-biennium results, all 20 correlations "strong")
- DIME/CFscores: `docs/dime-cfscore-deep-dive.md` (campaign-finance ideology, V4.0 codebook/validation, 6-biennium coverage)
- Hierarchical deep dive: `docs/hierarchical-shrinkage-deep-dive.md` (over-shrinkage analysis with literature)
- PCA deep dive: `docs/pca-deep-dive.md` (literature review, code audit, implementation recommendations)
- MCA deep dive: `docs/mca-deep-dive.md` (theory survey, ecosystem evaluation, integration design)
- UMAP deep dive: `docs/umap-deep-dive.md` (literature survey, code review, implementation recommendations)
- IRT deep dive: `docs/irt-deep-dive.md` (field survey, code audit, test gaps, recommendations)
- IRT field survey: `docs/irt-field-survey.md` (identification problem, unconstrained β contribution, Python ecosystem gap)
- Clustering deep dive: `docs/clustering-deep-dive.md` (literature survey, code audit, test gaps, recommendations)
- Hierarchical IRT deep dive: `docs/hierarchical-irt-deep-dive.md` (ecosystem survey, code audit, 9 issues fixed, 35 tests, ADR-0033)
- Joint hierarchical IRT diagnosis: `docs/joint-hierarchical-irt-diagnosis.md` (bill-matching bug: vote_id vs bill_number, sign flip cascade, fix plan)
- Hierarchical PCA init experiment: `docs/hierarchical-pca-init-experiment.md` (R-hat fix, ESS threshold analysis, ADR-0044)
- 4-chain hierarchical IRT experiment: `docs/hierarchical-4-chain-experiment.md` (ESS fix, jitter mode-splitting discovery, adapt_diag fix)
- Hierarchical convergence improvement: `docs/hierarchical-convergence-improvement.md` (House vs Senate theory, β>0 constraint, 9-priority improvement plan, experiment results)
- Joint model deep dive: `docs/joint-model-deep-dive.md` (concurrent calibration failure, reparameterized LogNormal experiments, Stocking-Lord IRT linking, production recommendation)
- Positive beta experiment: `results/experimental_lab/2026-02-27_positive-beta/experiment.md` (LogNormal fixes R-hat but not ESS; joint improves but still fails; positive β necessary but not sufficient)
- Prediction deep dive: `docs/prediction-deep-dive.md` (literature survey, code audit, IRT circularity analysis, test gaps)
- Beta-Binomial deep dive: `docs/beta-binomial-deep-dive.md` (ecosystem survey, code audit, ddof fix, Tarone's test)
- Synthesis deep dive: `docs/synthesis-deep-dive.md` (field survey, code audit, detection algorithms, test gaps, refactoring)
- Profiles deep dive: `docs/profiles-deep-dive.md` (code audit, ecosystem survey, name-based lookup)
- Cross-session deep dive: `docs/cross-session-deep-dive.md` (ecosystem survey, code audit, 3 bugs fixed, 18 new tests)
- Scraper deep dive: `docs/scraper-deep-dive.md` (ecosystem comparison, code audit, data quality review, test gap analysis)
- 2D IRT deep dive: `docs/2d-irt-deep-dive.md` (ecosystem survey, PLT identification, Tyson paradox resolution, pipeline phase 06)
- TSA deep dive: `docs/tsa-deep-dive.md` (literature survey, ecosystem comparison, code audit, 7 recommendations — all resolved)
- TSA R enrichment: ADR-0061 (CROPS penalty selection + Bai-Perron CIs via R subprocess)
- Dynamic ideal points: `docs/dynamic-ideal-points-deep-dive.md` (ecosystem survey, Martin-Quinn model, state-space IRT, decomposition)
- Dynamic IRT sign correction: ADR-0068 (post-hoc per-period xi negation using static IRT correlation, 87th House sign flip case)
- Dynamic IRT convergence: ADR-0070 (informative xi_init prior from static IRT, adaptive tau for small chambers, symlink race guard, MCMC budget increase)
- W-NOMINATE + OC: `docs/w-nominate-deep-dive.md` (field-standard comparison, R subprocess, validation-only design); design: `analysis/design/wnominate.md`
- PPC + LOO-CV: design: `analysis/design/ppc.md` (manual log-likelihood, Q3 local dependence, PSIS-LOO model comparison, ADR-0063)
- LCA deep dive: `docs/latent-class-deep-dive.md` (literature survey, StepMix evaluation, Salsa effect, Lubke & Neale impossibility)
- LCA design: `analysis/design/lca.md` (Bernoulli mixture, BIC selection, FIML missing data, Salsa threshold)
- Bipartite network deep dive: `docs/bipartite-network-deep-dive.md` (literature survey, BiCM, bill-centric metrics, Kansas-specific considerations)
- Bipartite design: `analysis/design/bipartite.md` (BiCM backbone, Newman projection, bill communities, Phase 6 comparison)
- Bill text retrieval: ADR-0083 (StateAdapter Protocol, shared bill discovery, pdfplumber PDF extraction, multi-state-ready)
- Legislator identity: ADR-0085 (OpenStates OCD person IDs, slug→ocd_id mapping, 3-phase matching, same-name disambiguation)
- Bill text NLP deep dive: `docs/bill-text-nlp-deep-dive.md` (BERTopic, CAP classification, TBIP, embeddings — BT1-BT5 complete)
- Bill text analysis design: `analysis/design/bill_text.md` (BERTopic config, FastEmbed embedding, CAP taxonomy, Rice cohesion, caucus-splitting)
- Text-based ideal points: ADR-0086 (embedding-vote approach, TBIP alternative for committee-sponsored bills)
- Text-based ideal points design: `analysis/design/tbip.md` (methodology, assumptions, lower quality thresholds, limitations)
- Issue-specific ideal points: ADR-0087 (topic-stratified flat IRT, why not issueirt, thresholds, anchor strategy)
- Issue-specific ideal points design: `analysis/design/issue_irt.md` (two taxonomies, parameters, quality thresholds, assumptions)
- KanFocus vote data adapter: ADR-0088 (1999-2026 coverage, Chrome cookie auth, HTML-to-text parsing, gap-fill mode, data archiving)
- Model legislation detection: ADR-0089 (ALEC + cross-state, embedding similarity, n-gram overlap, BT5)
- Model legislation design: `analysis/design/model_legislation.md` (thresholds, data sources, architecture, limitations)
- Django project scaffolding: ADR-0090 (DB1 — 8 models, PostgreSQL, Docker Compose, admin, 63 tests)
- Data storage: `docs/data-storage-deep-dive.md` (ecosystem survey, PostgreSQL + Django recommended, multi-state scaling, migration path)
- Future bill text analysis: `docs/future-bill-text-analysis.md` (original notes, superseded by deep dive)
- Apple Silicon MCMC tuning: `docs/apple-silicon-mcmc-tuning.md` (P/E core scheduling, thread pool caps, parallel chains, batch job rules)
- Ward linkage article: `docs/ward-linkage-non-euclidean.md` (why Ward on Kappa distances is impure, the fix)
- Experiment framework: `docs/experiment-framework-deep-dive.md` (ecosystem survey, design patterns, BetaPriorSpec, PlatformCheck, monitoring)
- Nutpie deep dive: `docs/nutpie-deep-dive.md` (Rust NUTS sampler, normalizing flow adaptation, integration plan)
- Nutpie flat IRT experiment: `docs/nutpie-flat-irt-experiment.md` (Experiment 1 results: compilation, sampling, sign flip, |r|=0.994 vs PyMC)
- Nutpie hierarchical experiment: `results/experimental_lab/2026-02-27_nutpie-hierarchical/experiment.md` (Experiment 2: hierarchical per-chamber with Numba, House convergence test)
- Nutpie production migration: ADR-0051 (per-chamber hierarchical), ADR-0053 (flat IRT + joint hierarchical — all models now use nutpie)
- 84th biennium analysis: `docs/84th-biennium-analysis.md` (full pipeline review, moderate Republican faction, 2012 purge, data quality flags)
- Report enhancement survey: `docs/report-enhancement-survey.md` (current report inventory, gap analysis, open-source tools, 26 prioritized recommendations — R1-R13 implemented ADR-0069, R14-R20 implemented ADR-0071)
- Pipeline audit: ADR-0072 (8-biennium review, 18 findings, 6 fixes — except syntax, prediction leakage, sample threshold, logging)
- Code audit: ADR-0078 (phase_utils extraction, 3 bug fixes, vectorized bipartite, dead code removal, ~400 lines deduped)
- Code audit resolution: ADR-0080 (synthesis manifest key fix, 4 remaining cleanup items — monitoring, dead helpers, special sessions, cache keys)
- WCAG accessibility: ADR-0079 (alt-text on 132 figures, aria-labels on 8 interactive sections, 23 report builders)
- W-NOMINATE all-biennium run: ADR-0073 (6 R compatibility bugs fixed, all 8 bienniums validated, PPC expanded to 6/8)
- Convergence resolution: ADR-0074 (joint model off by default, 2D IRT dropped from pipeline, dynamic IRT prior fixed)
- Name matcher district tiebreaker: ADR-0075 (Phase 14 + 14b district disambiguation, shrinkage null investigation)
- Audit findings resolution: ADR-0076 (A6-A18: bridge-builder harmonic centrality, surprising vote split, IRT sensitivity interpretation, small-group warning, BiCM Senate threshold, document-and-accept annotations)
- Audit findings deep dive: `docs/audit-findings-deep-dive.md` (research classification of all 13 remaining findings)
- Worktree CWD death prevention: ADR-0077 (update-ref fix, named wt-done, forwarder pattern)
- Analytic flags: `docs/analytic-flags.md` (living document of observations)
- Field survey: `docs/landscape-legislative-vote-analysis.md`
- Method evaluation: `docs/method-evaluation.md`

## Experiment Framework

Four components eliminate code duplication in MCMC experiments (ADR-0048):

- **`analysis/07_hierarchical/model_spec.py`** — `BetaPriorSpec` frozen dataclass + `PRODUCTION_BETA`, `JOINT_BETA`. Both `build_per_chamber_model()` and `build_joint_model()` accept `beta_prior=` parameter (defaults to production). Joint model uses `JOINT_BETA` (`lognormal_reparam`: exp(Normal(0,1)) for positive discrimination without boundary geometry). Experiments pass alternative specs — no code duplication. `build_per_chamber_graph()` returns the PyMC model without sampling (importable by experiments).
- **`analysis/07_hierarchical/irt_linking.py`** — IRT scale linking for cross-chamber alignment. Implements Stocking-Lord, Haebara, Mean-Sigma, and Mean-Mean linking using shared (anchor) bills. Sign-aware anchor extraction handles unconstrained per-chamber betas. Production alternative to concurrent calibration (joint model).
- **`analysis/experiment_monitor.py`** — `PlatformCheck` (validates Apple Silicon constraints before sampling), `ExperimentLifecycle` (PID lock, process group, cleanup). `just monitor` checks if an experiment process is alive (nutpie shows its own terminal progress bar).
- **`analysis/experiment_runner.py`** — `ExperimentConfig` frozen dataclass + `run_experiment()`. Orchestrates: platform check → data load → per-chamber models → optional joint → HTML report → metrics.json. 799-line experiment scripts become ~25-line configs.

All hierarchical experiments (whether using `ExperimentRunner` or standalone scripts) produce the full production HTML report via `build_hierarchical_report()` — the same 18-22 section report as `just hierarchical` (party posteriors, ICC, variance decomposition, dispersion, shrinkage scatter/table, forest plots, convergence, cross-chamber comparison, flat vs hier comparison). The nutpie flat IRT experiment uses a bespoke report (different model type).

## Concurrency

- **Scraper**: concurrent fetch via ThreadPoolExecutor (MAX_WORKERS=5), sequential parse. Never mutate shared state during fetch.
- **MCMC (all models)**: nutpie Rust NUTS sampler — single process, Rust threads for parallel chains (ADR-0051, ADR-0053). Graph-building functions (`build_per_chamber_graph()`, `build_joint_graph()`, `build_irt_graph()`) return PyMC models without sampling. Sampling functions compile with `nutpie.compile_pymc_model()` and sample with `nutpie.sample()`. PCA-informed init via `initial_points`; `jitter_rvs` excludes the PCA-initialized variable.
- **Apple Silicon (M3 Pro, 6P+6E)**: run bienniums sequentially; cap thread pools (`OMP_NUM_THREADS=6`); never use `taskpolicy -c background`. See ADR-0022.
- **PyTensor C compiler**: PyTensor requires `clang++`/`g++` for C-compiled kernels. Without it, falls back to pure Python (~18x slower). Common failure: Xcode update requires opening Xcode.app to accept license. Justfile exports `PATH` with `/usr/bin` to prevent stripped-PATH failures.
- **R (optional)**: Required for Phase 16 (W-NOMINATE/OC: `wnominate`, `oc`, `pscl`, `jsonlite`) and Phase 19 TSA enrichment (`changepoint`, `strucchange`). Install via `brew install r` then `install.packages()`. Not managed by uv. Core pipeline works without R. R CSV files use literal "NA" for missing values — always pass `null_values="NA"` to `pl.read_csv()` when reading R output (ADR-0073).
- **StepMix / scikit-learn shim (Phase 10 LCA)**: StepMix 2.2.1 uses sklearn's private `_validate_data` and deprecated `force_all_finite` kwarg — both removed in scikit-learn 1.8. A monkey-patch in `analysis/10_lca/lca.py` (lines 45-57) restores compatibility. **TODO: remove shim when StepMix ships a fix** (check `hasattr(StepMix, "_validate_data")` — shim is already guarded and will no-op once fixed upstream).

## Testing

```bash
just test                    # 2458 tests
just test-scraper            # scraper tests only (-m scraper)
just test-fast               # skip slow/integration tests (-m "not slow")
just test-web                # Django/database tests only (-m web, requires PostgreSQL)
just check                   # full check (lint + typecheck + tests)
```

Pytest markers: `@pytest.mark.scraper` (scraper pipeline), `@pytest.mark.integration` (end-to-end/real-data), `@pytest.mark.slow` (>5s), `@pytest.mark.web` (Django, requires PostgreSQL). Registered in `pyproject.toml`.

See `.claude/rules/testing.md` for test file inventory and conventions.

## Web / Database (DB1)

Django project at `src/web/` for PostgreSQL-backed data access. Purely additive — the scraper continues writing CSVs; a future loader (DB2) will import them into PostgreSQL.

```
src/web/
  manage.py                      # Django entry point
  tallgrass_web/
    settings/
      base.py                    # DB config, apps, middleware
      local.py                   # DEBUG=True
      test.py                    # Fast hashing, test DB
    urls.py                      # Admin only (for now)
    wsgi.py / asgi.py
  legislature/
    models.py                    # 8 models: State, Session, Legislator, RollCall,
                                 #   Vote, BillAction, BillText, ALECModelBill
    admin.py                     # Admin registration for all models
    migrations/                  # Auto-generated by makemigrations
```

Dependencies: `web` group in pyproject.toml (Django 5.2 LTS + psycopg3). Not a core dependency — scraper users don't need Django. Install with `uv sync --group web`.

Justfile recipes: `just db-up` (start PostgreSQL), `just db-down` (stop), `just db-migrate` (apply migrations), `just db-makemigrations` (generate), `just db-admin` (runserver), `just db-createsuperuser`, `just db-shell` (psql), `just django <cmd>` (passthrough), `just test-web` (Django tests).

All Django recipes set `DJANGO_SETTINGS_MODULE=tallgrass_web.settings.local` and `PYTHONPATH=src/web`.

Django tests use `pytest.importorskip("django")` — existing 2458 tests never import Django. `DJANGO_SETTINGS_MODULE` is NOT set in pyproject.toml. ADR-0090.
