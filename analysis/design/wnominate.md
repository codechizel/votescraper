# W-NOMINATE + Optimal Classification Design Choices

**Script:** `analysis/17_wnominate/wnominate.py`
**Data module:** `analysis/17_wnominate/wnominate_data.py`
**R script:** `analysis/17_wnominate/wnominate.R`
**Report builder:** `analysis/17_wnominate/wnominate_report.py`
**Constants defined at:** top of `wnominate_data.py`

## Assumptions

1. **Validation-only phase.** W-NOMINATE and OC results do NOT feed into synthesis, profiles, or any downstream phase. They exist solely to validate our Bayesian IRT ideal points against field-standard methods. This simplifies the dependency graph and avoids circular validation.

2. **Chambers analyzed separately**, consistent with all upstream phases.

3. **W-NOMINATE uses MLE, not Bayesian.** The parametric bootstrap SEs are not posterior credible intervals. Do not compare WNOM SEs to IRT posterior widths — different uncertainty concepts.

4. **OC is nonparametric.** No distributional assumptions about error terms. Slightly lower correlation with parametric methods (IRT, WNOM) is expected and normal.

5. **Polarity identification via PCA PC1** (highest PC1 legislator with ≥50% participation). This is equivalent to the convention in `pscl::ideal` and matches our data-driven approach across all phases.

## Parameters & Constants

| Constant | Value | Justification |
|----------|-------|---------------|
| `ROLLCALL_YEA` | 1 | pscl convention: 1 = Yea |
| `ROLLCALL_NAY` | 6 | pscl convention: 6 = Nay |
| `ROLLCALL_MISSING` | 9 | pscl convention: 9 = Missing/Not Voting |
| `ROLLCALL_NOT_IN_LEGIS` | NA | Must differ from `ROLLCALL_MISSING` to avoid pscl "codes not unique" error (ADR-0073) |
| `WNOMINATE_DIMS` | 2 | Standard 2D estimation. Dim 1 dominates; dim 2 is diagnostic. |
| `MIN_LEGISLATORS` | 10 | Chamber must have ≥10 legislators to run. Safety guard. |
| `MIN_VOTES` | 20 | Passed to R: legislators with <20 votes excluded by W-NOMINATE. |
| `LOP_THRESHOLD` | 0.025 | Lopsided vote filter: votes with <2.5% minority excluded. Matches our EDA filter. |
| `PARTY_COLORS` | R=#E81B23, D=#0015BC, I=#999999 | Consistent with all prior phases. |

## R Compatibility Notes (ADR-0073)

Discovered during all-biennium production run (84th-91st). These are version-dependent behaviors in the R packages:

1. **`notInLegis` must be unique.** pscl's `rollcall()` requires all code values to be distinct. Setting `notInLegis=NA` avoids collision with `missing=9`.

2. **Polarity must be a vector of length `dims`.** For 2D W-NOMINATE and OC, `polarity` must be `rep(idx, dims)`, not a scalar. Scalar works for 1D only.

3. **OC `legislators` is a matrix, not a data.frame** (oc 1.2.1). Use `oc_leg[, "coord1D"]` column indexing, not `$coord1D` list-style access.

4. **R CSV "NA" strings.** R's `write.csv()` writes `NA` for missing values. Polars requires `null_values="NA"` to parse these correctly.

5. **Fit statistics as strings.** R's `jsonlite::write_json()` serializes numeric values that may be read back as strings. Always cast to float with NA-safe conversion.

## Methodological Choices

### W-NOMINATE and OC bundled in one phase

Both methods use the same input (pscl rollcall object), same polarity legislator, and same R session. Running them together adds ~10s to a ~30s R call. Separate phases would duplicate all the data loading, conversion, and reporting infrastructure.

### R subprocess over rpy2

The `wnominate` and `oc` packages are R-only with complex Fortran backends. Using rpy2 would require:
- Compiling rpy2 against the user's R installation
- Managing R shared library paths on macOS
- Debugging opaque segfaults when R/Python ABI mismatches occur

A subprocess call with CSV I/O is simpler, more portable, and matches the pattern used for emIRT in Phase 16. The ~10ms CSV serialization overhead is negligible compared to the W-NOMINATE MLE optimization.

### Polarity via PCA PC1 (not external knowledge)

We could hardcode a known conservative legislator (e.g., Masterson in the 91st). But:
- This breaks for historical bienniums where we don't know legislators.
- PCA PC1 is the data-driven equivalent — it identifies the dominant ideological dimension automatically.
- `pscl::ideal` uses this same convention (Jackman 2001).
- Post-estimation sign alignment against IRT guarantees correct direction regardless of initial polarity.

### Sign alignment against IRT (not PCA)

After W-NOMINATE and OC estimation, we check Pearson r of dim1 against IRT xi_mean. If negative, we flip. IRT is the primary upstream — aligning to it ensures consistent direction across the pipeline. Aligning to PCA would add an extra indirection.

### 3 trials for W-NOMINATE

W-NOMINATE uses a random start for MLE optimization. Multiple trials (default 3) guard against local optima. With 2D estimation and ~100+ legislators, 3 trials provide sufficient coverage without excessive runtime.

## Downstream Implications

None. This is a terminal validation phase — no other phase reads from `17_wnominate/` output.

The correlation results may inform narrative claims in external-facing documents (e.g., "our IRT ideal points correlate at r=0.98 with W-NOMINATE"), but this is a human interpretation step, not a pipeline dependency.
