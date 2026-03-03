# Tallgrass

Kansas Legislature roll call vote scraper and Bayesian analysis platform. Scrapes [kslegislature.gov](https://www.kslegislature.gov) into structured CSV files, then runs a 17-phase statistical pipeline covering IRT ideal points, network analysis, clustering, time series, and more.

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
- **R** — only required for Phases 16 (Dynamic IRT) and 17 (W-NOMINATE/OC). Install `wnominate`, `pscl`, `oc`, `changepoint`, `strucchange` from CRAN.

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

Three CSV files per session in `data/kansas/{legislature}_{start}-{end}/`:

| File | Contents |
|------|----------|
| `*_votes.csv` | One row per legislator per roll call |
| `*_rollcalls.csv` | One row per roll call (bill, motion, result, tallies) |
| `*_legislators.csv` | One row per legislator (name, party, district, chamber) |

## Analysis Pipeline

17 phases covering descriptive statistics, dimensionality reduction, Bayesian modeling, network analysis, prediction, and cross-session validation.

```bash
just pipeline 2025-26       # run all 17 phases in sequence
just eda                    # single phase
just irt --n-samples 4000   # with custom arguments
```

| # | Phase | Method |
|---|-------|--------|
| 01 | EDA | Descriptive statistics, vote matrix, missingness |
| 02 | PCA | Principal component analysis |
| 02c | MCA | Multiple correspondence analysis |
| 03 | UMAP | Nonlinear dimensionality reduction |
| 04 | IRT | 1D Bayesian ideal points (PyMC + nutpie) |
| 04b | 2D IRT | 2D Bayesian IRT with PLT identification |
| 04c | PPC | Posterior predictive checks + LOO-CV model comparison |
| 05 | Clustering | Hierarchical, k-means, GMM |
| 05b | LCA | Latent class analysis (StepMix) |
| 06 | Network | Co-voting network + community detection |
| 06b | Bipartite | Bill-legislator bipartite network |
| 07 | Indices | Rice, party unity, ENP, maverick scores |
| 08 | Prediction | Vote prediction (logistic + XGBoost + SHAP) |
| 09 | Beta-Binomial | Bayesian party loyalty shrinkage |
| 10 | Hierarchical | Hierarchical IRT with partial pooling |
| 11 | Synthesis | 30-section narrative report joining all phases |
| 12 | Profiles | Per-legislator deep-dive reports |

Additional standalone phases: cross-session validation (13), external validation against Shor-McCarty (14) and DIME/CFscores (14b), time series analysis (15), dynamic ideal points (16), W-NOMINATE + Optimal Classification (17).

Each phase produces an HTML report with tables, figures, and plain-English interpretation. Reports are written to `results/kansas/{session}/{run_id}/{phase}/`.

## Development

```bash
just check          # lint + typecheck + tests (quality gate)
just test           # run all ~1680 tests
just test-fast      # skip slow/integration tests
just lint           # ruff check --fix + ruff format
just typecheck      # ty check src/ + ty check analysis/
```

## Project Structure

```
src/tallgrass/     # Scraper package (config, session, models, scraper, output, CLI)
analysis/          # 17 numbered phase subdirectories + shared infrastructure
tests/             # ~1680 pytest tests (scraper + all analysis phases)
docs/              # Deep dives, ADRs, field surveys, primers
data/              # Scraped CSV output + external validation data (gitignored)
results/           # HTML reports + parquet intermediates (gitignored)
```

## Documentation

- [Analysis primer](docs/analysis-primer.md) — plain-English guide for general audiences
- [How IRT works](docs/how-irt-works.md) — general-audience explanation of Bayesian ideal points
- [Architecture decisions](docs/adr/README.md) — 65 ADRs documenting design choices
- [Design docs](analysis/design/README.md) — per-phase methodology and implementation
- [Roadmap](docs/roadmap.md) — completed phases, backlog, rejected methods

## External Data

- **Shor-McCarty scores** — auto-downloaded from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BSLEFD) on first use
- **DIME/CFscores** — must be manually downloaded from [Stanford DIME project](https://data.stanford.edu/dime) (144 MB, ODC-BY license). Place at `data/external/dime_recipients_1979_2024.csv`

## License

[MIT](LICENSE)
