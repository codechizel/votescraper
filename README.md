# Tallgrass

Kansas Legislature roll call vote scraper and Bayesian analysis platform. Scrapes [kslegislature.gov](https://www.kslegislature.gov) into structured CSV files, then runs a 27-phase statistical pipeline covering IRT ideal points, network analysis, clustering, time series, and more.

**Coverage:** 2011-2026 (84th-91st Legislatures)

## Quick Start

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone https://github.com/codechizel/tallgrass.git
cd tallgrass
uv sync

# Scrape the current session
uv run tallgrass 2025

# Run the full analysis pipeline
just pipeline 2025-26
```

## Requirements

- **Python 3.14+** — install via `uv python install 3.14`
- **[uv](https://docs.astral.sh/uv/)** — package manager
- **[Just](https://github.com/casey/just)** — command runner (optional but recommended)
- **R** — only required for Phase 16 (W-NOMINATE/OC) and Phase 19 (TSA enrichment). Install `wnominate`, `pscl`, `oc`, `changepoint`, `strucchange` from CRAN.

## Scraper

```bash
uv run tallgrass 2025                   # current session (2025-26)
uv run tallgrass 2023                   # historical session (2023-24)
uv run tallgrass 2024 --special         # special session
uv run tallgrass --merge-special 2020   # merge 2020 special into parent biennium
uv run tallgrass --merge-special all    # merge all 5 specials
uv run tallgrass --list-sessions        # show all available sessions
uv run tallgrass 2025 --clear-cache     # re-fetch everything
```

### Output

Five CSV files per session in `data/kansas/{legislature}_{start}-{end}/`:

| File | Contents |
|------|----------|
| `*_votes.csv` | One row per legislator per roll call |
| `*_rollcalls.csv` | One row per roll call (bill, motion, result, tallies) |
| `*_legislators.csv` | One row per legislator (name, party, district, chamber) |
| `*_bill_actions.csv` | One row per bill lifecycle action (89th+ only) |
| `*_bill_texts.csv` | One row per bill document (via `tallgrass-text`) |

## Analysis Pipeline

27 phases covering descriptive statistics, dimensionality reduction, Bayesian modeling, network analysis, prediction, bill text NLP, and cross-session validation.

```bash
just pipeline 2025-26       # run all 25 single-biennium phases
just eda                    # single phase
just irt --n-samples 4000   # with custom arguments
```

| # | Phase | Method |
|---|-------|--------|
| 01 | EDA | Descriptive statistics, vote matrix, missingness |
| 02 | PCA | Principal component analysis |
| 03 | MCA | Multiple correspondence analysis |
| 04 | UMAP | Nonlinear dimensionality reduction |
| 05 | IRT | 1D Bayesian ideal points (PyMC + nutpie) |
| 06 | 2D IRT | 2D Bayesian IRT with PLT identification |
| 07 | Hierarchical | Hierarchical IRT with partial pooling |
| 08 | PPC | Posterior predictive checks + LOO-CV model comparison |
| 09 | Clustering | Hierarchical, k-means, GMM |
| 10 | LCA | Latent class analysis (StepMix) |
| 11 | Network | Co-voting network + community detection |
| 12 | Bipartite | Bill-legislator bipartite network |
| 13 | Indices | Rice, party unity, ENP, maverick scores |
| 14 | Beta-Binomial | Bayesian party loyalty shrinkage |
| 15 | Prediction | Vote prediction (logistic + XGBoost + SHAP) |
| 16 | W-NOMINATE | W-NOMINATE + Optimal Classification (R) |
| 17 | External Validation | Shor-McCarty score correlation |
| 18 | DIME | DIME/CFscore campaign-finance validation |
| 19 | TSA | Time series analysis + changepoint detection |
| 20 | Bill Text | BERTopic topic modeling + NLP analysis |
| 21 | TBIP | Text-based ideal points |
| 22 | Issue IRT | Issue-specific ideal points (topic-stratified) |
| 23 | Model Legislation | ALEC + cross-state bill matching |
| 24 | Synthesis | Narrative report joining all phases |
| 25 | Profiles | Per-legislator deep-dive reports |
| 26 | Cross-Session | Cross-biennium legislator matching + shift |
| 27 | Dynamic IRT | Martin-Quinn state-space IRT across bienniums |

Each phase produces an HTML report with tables, figures, and plain-English interpretation. Reports are written to `results/kansas/{session}/{run_id}/{phase}/`.

## Development

```bash
just check          # lint + typecheck + tests (quality gate)
just test           # run all ~2664 tests
just test-fast      # skip slow/integration tests
just lint           # ruff check --fix + ruff format
just typecheck      # ty check src/ + ty check analysis/
```

## Project Structure

```
src/tallgrass/     # Scraper package (config, session, models, scraper, output, CLI)
analysis/          # 27 numbered phase subdirectories + shared infrastructure
tests/             # ~2664 pytest tests (scraper + all analysis phases)
docs/              # Deep dives, ADRs, field surveys, primers
data/              # Scraped CSV output + external validation data (gitignored)
results/           # HTML reports + parquet intermediates (gitignored)
```

## Documentation

- [Analysis primer](docs/analysis-primer.md) — plain-English guide for general audiences
- [How IRT works](docs/how-irt-works.md) — general-audience explanation of Bayesian ideal points
- [Architecture decisions](docs/adr/README.md) — 96 ADRs documenting design choices
- [Design docs](analysis/design/README.md) — per-phase methodology and implementation
- [Roadmap](docs/roadmap.md) — completed phases, backlog, rejected methods

## External Data

- **Shor-McCarty scores** — auto-downloaded from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BSLEFD) on first use
- **DIME/CFscores** — must be manually downloaded from [Stanford DIME project](https://data.stanford.edu/dime) (144 MB, ODC-BY license). Place at `data/external/dime_recipients_1979_2024.csv`

## License

[MIT](LICENSE)
