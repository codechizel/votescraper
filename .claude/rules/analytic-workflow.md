# Analytic Workflow Rules

## Technology Preferences
- **Polars over pandas.** Use polars for all data manipulation. Never use pandas unless a downstream library strictly requires a pandas DataFrame (e.g., seaborn, pymc).
- **Python over R.** All analysis in pure Python. No R, no rpy2, no Rscript calls. When an R-only method exists (W-NOMINATE, OC), use the Python alternative (PCA, Bayesian IRT).

## Method Documentation
- One method per file in `Analytic_Methods/`
- Naming convention: `NN_CAT_method_name.md`
- Each file includes: purpose, assumptions, inputs, outputs, validation steps

## Mandatory Workflow Order
1. **Always run EDA first.** Before any model, record: row counts, missingness rates, close-vote fraction, unanimous-vote fraction, chamber split sizes.
2. **Canonical baseline: 1D Bayesian IRT** on Yea/Nay only. All other models are compared against this.
3. **PCA first, Bayesian second.** PCA is cheap and fast — use it to sanity-check before investing in MCMC.

## Filtering and Reproducibility
- All filtering decisions (unanimous threshold, min participation, chamber separation) must be **explicit constants**, not magic numbers buried in code.
- Save a **filtering manifest** with every analysis output: which votes were dropped, which legislators were excluded, and why.
- Default filters: drop votes where minority < 2.5%, drop legislators with < 20 votes.
- **Sensitivity analyses are mandatory**: run the core model with at least two filter settings (e.g., minority < 2.5% vs < 10%, final passage only vs all motions).

## Validation
- Every fitted model must include at least one validation:
  - Holdout prediction (random 20% of vote observations)
  - Posterior predictive check (for Bayesian models)
  - Classification accuracy against observed votes
- Report both accuracy AND AUC-ROC (accuracy alone is misleading with 82% Yea base rate).

## Separation of Concerns
- **ETL (scraping) is separate from analysis.** Never modify scraper code to accommodate analysis needs. Instead, transform data in analysis scripts.
- Analysis scripts read from `data/ks_{session}/` CSVs. Intermediate analysis artifacts go in a separate output directory.

## Kansas-Specific Defaults
- Always analyze chambers separately unless explicitly doing cross-chamber comparison.
- Use Cohen's Kappa (not raw agreement) when thresholding similarity for networks or clustering.
- The 34 veto override votes should be analyzed as a separate subgroup in addition to the full dataset.
