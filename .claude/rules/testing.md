---
paths:
  - "tests/**/*.py"
---

# Testing

## Commands

```bash
just test                    # run all tests (~1803)
just test-scraper            # scraper tests only (-m scraper, ~264)
just test-fast               # skip slow tests (-m "not slow", ~1445)
just check                   # full check (lint + typecheck + tests)
uv run pytest tests/ -v      # pytest directly
uv run pytest tests/ -m integration  # integration tests only
uv run pytest tests/ -m "not scraper"  # analysis tests only
```

## Markers

Registered in `pyproject.toml`. Module-level `pytestmark` variables (not per-class decorators).

- `@pytest.mark.scraper` — scraper pipeline tests (8 files, ~264 tests)
- `@pytest.mark.integration` — end-to-end and real-data tests (~29 tests)
- `@pytest.mark.slow` — tests that take >5 seconds (~24 tests)

## Conventions

- Class-based test organization with docstrings including run command
- `# -- Section --` headers to group related tests
- Inline fixtures (HTML parsing uses inline BeautifulSoup, not separate files)
- CLI tests use monkeypatched FakeScraper class
- RunContext tests use `tmp_path` for isolated filesystem operations

## Scraper Test Files

- `tests/conftest.py` — shared KSSession fixtures (current, historical, special)
- `tests/test_session.py` — session URL resolution, biennium logic, uses_odt, js_data_paths (~40 tests)
- `tests/test_scraper_pure.py` — pure functions: bill codes, datetime parsing, result derivation, JS parsing (~45 tests)
- `tests/test_scraper_html.py` — HTML parsing via static methods (`_extract_bill_title`, `_extract_chamber_motion_date`, `_parse_vote_categories`, `_extract_party_and_district`), pre-2015 party detection, odt_view links (~32 tests)
- `tests/test_models.py` — dataclass construction and immutability, VoteLink.is_odt (~8 tests)
- `tests/test_odt_parser.py` — ODT vote parsing: XML, metadata, body text, name resolution (~47 tests)
- `tests/test_scraper_http.py` — HTTP layer: _get() retries, error classification, cache, _fetch_many() waves, rate limiting, KLISS API (~44 tests)
- `tests/test_output.py` — CSV export: filenames, headers, row counts, roundtrip (~10 tests)
- `tests/test_cli.py` — argument parsing with monkeypatched scraper (~17 tests)

## Analysis Test Files

- `tests/test_run_context.py` — TeeStream, session normalization, strip_leadership_suffix, lifecycle, generate_run_id, resolve_upstream_dir, run-directory mode, auto-run-id, symlink race guard (~76 tests)
- `tests/test_eda.py` — vote matrix, filtering, agreement, Rice, party-line, integrity, new diagnostics (~28 tests)
- `tests/test_report.py` — section rendering, format parsing, ReportBuilder, make_gt, elapsed (~38 tests)
- `tests/test_report_sections.py` — DownloadSection, ScrollyStep, ScrollySection rendering, ReportBuilder scrolly integration, CSS styles (~18 tests)
- `tests/test_irt.py` — IRT data prep, anchor selection, sensitivity, forest, paradox detection, convergence diagnostics, posterior extraction, equating (~73 tests)
- `tests/test_umap_viz.py` — imputation, orientation, embedding, Procrustes, validation, trustworthiness, sensitivity sweep, stability, three-party (~40 tests)
- `tests/test_nlp_features.py` — TF-IDF + NMF fitting, edge cases, display names (~16 tests)
- `tests/test_pca.py` — imputation, PC1 orientation, extreme PC2 detection (~16 tests)
- `tests/test_mca.py` — categorical matrix, Polars-to-pandas, MCA fitting, orientation, eigenvalues, contributions, cos², horseshoe detection, constants (~34 tests)
- `tests/test_prediction.py` — vote/bill features, model training, SHAP, NLP integration, holdout eval, baselines, proper scoring rules (~54 tests)
- `tests/test_beta_binomial.py` — method of moments, posteriors, shrinkage, edge cases (~26 tests)
- `tests/test_hierarchical.py` — hierarchical data prep, model structure, variance decomposition, small-group warning, joint ordering, rescaling fallback, Independent exclusion, sign convention fix, bill-matching, adaptive priors (~48 tests)
- `tests/test_profiles.py` — profile targets, scorecard, bill-type breakdown, defections, name resolution, full voting record (~44 tests)
- `tests/test_cross_session.py` — matching, IRT alignment, shift, stability, PSI, ICC, fuzzy matching, prediction transfer, detection, freshmen cohort, bloc stability, plot smoke tests, report (~111 tests)
- `tests/test_clustering.py` — party loyalty, cross-method ARI, within-party, kappa distance, hierarchical, spectral, HDBSCAN, characterization (~70 tests)
- `tests/test_network.py` — network construction, centrality, Leiden/CPM community detection, bridges, threshold sweep, polarization, disparity filter backbone, extreme edge weights (~53 tests)
- `tests/test_bipartite.py` — bipartite graph construction, bill polarization, bipartite betweenness, bill projection, bill communities, BiCM backbone extraction, backbone comparison, constants (~50 tests)
- `tests/test_indices.py` — Rice formula, party votes, ENP, unity/maverick, co-defection, Carey UNITY, fractured votes (~37 tests)
- `tests/test_synthesis.py` — synthesis data loading, build_legislator_df joins, _extract_best_auc, detect_all integration, minority mavericks, Democrat-majority paradox (~47 tests)
- `tests/test_synthesis_detect.py` — maverick, bridge-builder, metric paradox detection, annotation slugs, threshold constants, percentile modes (~33 tests)
- `tests/test_external_validation.py` — SM name normalization, parsing, biennium filtering, matching, correlations, outliers (~65 tests)
- `tests/test_external_validation_dime.py` — DIME name normalization, parsing, biennium filtering, min-givers filter, matching, overlap detection, correlation reuse (~43 tests)
- `tests/test_tsa.py` — rolling PCA drift, sign alignment, party trajectories, early-vs-late, Rice timeseries, weekly aggregation, PELT changepoints, joint detection, penalty sensitivity, vote matrix, veto cross-reference, Desposato correction, short-session warnings, imputation sensitivity, variance-change detection, CROPS parsing, Bai-Perron parsing, elbow detection, PELT/BP merge, R package check (~85 tests)
- `tests/test_dynamic_irt.py` — global roster, name dedup, cross-biennium stacking, bridge coverage, model structure, polarization decomposition, top movers, sign correction, static correlation, graph construction (adaptive tau, informative prior), ridgeline plot, report smoke tests (~73 tests)
- `tests/test_ppc.py` — log-likelihood (1D/2D/hierarchical), PPC battery, item/person fit, Q3 local dependence, LOO-CV, Pareto k, data alignment (~60 tests)
- `tests/test_wnominate.py` — vote matrix conversion, polarity selection, result parsing, sign alignment, three-way correlations, comparison table, eigenvalues (~25 tests)
- `tests/test_lca.py` — vote matrix construction, class enumeration, BIC selection, Salsa effect detection, IRT cross-validation, ARI comparison, within-party LCA, discriminating bills, constants consistency (~37 tests)
- `tests/test_irt_2d.py` — sign-flip logic, PCA correlation, convergence constants (~16 tests)
- `tests/test_bill_actions.py` — BillAction dataclass construction, CSV export (headers, committee join, roundtrip, row counts) (~17 tests)
- `tests/test_bill_lifecycle.py` — bill lifecycle Sankey data: action categorization, stage transitions, chamber filtering, empty/edge cases (~22 tests)

## Integration & Structure Test Files

- `tests/test_report_structure.py` — HTML report skeleton: TOC anchors, section ordering, numbering, container types, duplicate IDs, empty report, CSS embedding, make_gt integration (~22 tests)
- `tests/test_dashboard.py` — phase ordering, dashboard generation, sidebar/iframe rendering, elapsed formatting, git hash display, missing phase handling (~14 tests)
- `tests/test_integration_pipeline.py` — end-to-end: synthetic data → EDA → PCA pipeline chain, RunContext lifecycle, upstream resolution, generate_run_id (~26 tests). Marked `@pytest.mark.integration`.
- `tests/test_data_integrity.py` — real-data CSV structural tests (~24 tests). Marked `@pytest.mark.integration` + `@pytest.mark.slow`.

## Manual Verification

- Run scraper with `--clear-cache`, check `vote_date`, `chamber`, `motion`, `bill_title` populated
- Check legislators CSV: party includes both Republican and Democrat
- Spot-check SB 1: Senate, Emergency Final Action, Passed as amended
