---
paths:
  - "analysis/**/*.py"
---

# Analytic Workflow Rules

## Mandatory Workflow Order
1. **Always run EDA first.** Record: row counts, missingness rates, close-vote fraction, unanimous-vote fraction, chamber split sizes.
2. **PCA first, Bayesian second.** PCA is cheap — sanity-check before investing in MCMC.
3. **Canonical baseline: 1D Bayesian IRT** on Yea/Nay only. All other models compared against this.

## Filtering and Reproducibility
- All filtering decisions must be **explicit constants**, not magic numbers.
- Save a **filtering manifest** with every output: which votes dropped, which legislators excluded, why.
- Default filters: contested threshold < 2.5%, min votes < 20 (both centralized in `analysis/tuning.py`).
- **Sensitivity analyses are mandatory**: at least two filter settings per core model.

## Validation
- Every fitted model must include at least one: holdout prediction (20%), posterior predictive check (Bayesian), classification accuracy.
- Report both accuracy AND AUC-ROC (accuracy misleading with 82% Yea base rate).

## Report Completeness
- **Never truncate tables.** Show all rows. Truncation happens for articles/presentations, not during analysis.

## Separation of Concerns
- **ETL (scraping) is separate from analysis.** Never modify scraper code for analysis needs.
- Analysis scripts load data via `analysis/db.py` — PostgreSQL by default, CSV fallback. `--csv` flag forces CSV-only. Intermediates go in results directory.

## Audience: Nontechnical Consumers
- Final outputs consumed by journalists, policymakers, engaged citizens.
- Every plot must be **self-explanatory** without statistics knowledge: clear titles, plain-English labels, intuitive colors (red=Republican, blue=Democrat).
- Prefer narrative-friendly visualizations (ranked bars, annotated scatters) over abstract plots (dendrograms, scree plots).
- **Annotate findings directly** on plots — label key actors, add callout boxes.
- Tables: plain-English headers with interpretive context.
- Reports: lead with accessible findings, build toward technical detail.

## Runtime Timing as a Sanity Check
- Check runtime when reviewing results. Unexpected speed changes indicate bugs or regressions.
- Baselines (91st, M3 Pro): EDA ~30s, PCA ~15s, IRT ~10-20min/chamber, prediction ~2-5min, synthesis ~30s.

## Kansas-Specific Defaults
- Analyze chambers separately unless explicitly doing cross-chamber comparison.
- Use Cohen's Kappa (not raw agreement) for similarity thresholds.
- Analyze 34 veto override votes as a separate subgroup.
