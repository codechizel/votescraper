# Confirmatory Factor Analysis, IRT, and the Two-Dimension Problem

**Date:** 2026-03-30

---

## Motivation

The tallgrass pipeline has a recurring challenge: Kansas legislative roll-call data is genuinely two-dimensional — one dimension captures left-right ideology, the other captures establishment-vs-contrarian behavior — but we need a clean 1D ideology score for canonical ideal points. This document evaluates whether Confirmatory Factor Analysis (CFA) with two constrained factors would improve ideology extraction compared to the current pipeline (PCA → Fisher's LDA → Bayesian IRT), and surveys the broader landscape of methods for separating dimensions in binary response data.

The question arose naturally: if we *know* the data has two factors, why not specify that structure upfront (confirmatory) rather than discover it post-hoc (exploratory)?

---

## Background: The Two-Factor Structure

### What EGA and PCA Tell Us

Across 28 Kansas chamber-sessions (2011-2026), the data consistently reveals two latent dimensions:

1. **Ideology (party divide):** The liberal-conservative spectrum. In balanced chambers, this is PC1 by a wide margin (eigenvalue ratio λ₁/λ₂ > 5). In supermajority chambers, it can slip to PC2 or be distributed across PC2-PC5.

2. **Establishment-contrarian axis:** Intra-party factionalism — leadership loyalists vs. rebels. In supermajority chambers (75%+ Republican), this axis can dominate PC1 because within-Republican variance exceeds between-party variance.

The Tyson paradox (see `analysis/design/tyson_paradox.md`) is the clearest illustration: Sen. Caryn Tyson votes with perfect 100% consistency on high-discrimination partisan bills but dissents on 50% of routine bipartisan legislation. She is simultaneously the most ideologically conservative senator (Dim 1) and the most contrarian (Dim 2). A 1D model can only capture one dimension and picks ideology because it carries more Fisher information — producing a ranking that is statistically correct but substantively counterintuitive to chamber observers.

### What the Pipeline Currently Does

The current architecture uses a multi-method sequential approach:

```
Phase 02 (PCA)     → Unsupervised dimensionality discovery
                    → Fisher's LDA projects PCA scores to ideology_score + establishment_score
Phase 02b (EGA)    → Network-based dimensionality estimate (advisory K)
Phase 05 (1D IRT)  → Bayesian 2PL ideal points on single ideology dimension
Phase 06 (2D IRT)  → Experimental M2PL with PLT identification (both dimensions)
Phase 07b (H2D)    → Hierarchical 2D with party-level pooling
Canonical Routing   → Horseshoe detection → tiered convergence → per-chamber source selection
```

Fisher's LDA (ADR-0129) was the most recent improvement. It finds the optimal party-separating direction across all PCs automatically, producing Cohen's d improvements of +20% to +253% over the best single PC. This solved the axis instability problem where PC1 captured factionalism instead of ideology in 4 of 14 Senate sessions.

### What Remains Unsatisfying

Three things:

1. **No formal hypothesis test.** PCA, EGA, and IRT each estimate dimensionality independently, but there is no unified confirmatory model that tests the specific hypothesis "these data are driven by two orthogonal factors: ideology and establishment." We have exploratory evidence and ad-hoc routing, not a principled structural test.

2. **Dim 2 convergence is weak.** The 2D IRT models (Phase 06, 07b) use relaxed convergence thresholds (R-hat < 1.05 vs 1.01 for 1D, ESS > 200 vs 400). Dim 2 has ~11% variance — enough to detect in aggregate but too little for precise individual estimation. Most legislators have non-informative posteriors on Dim 2.

3. **Bill-level assignment is implicit.** The pipeline doesn't formally model *which* bills belong to *which* dimension. IRT's discrimination parameters (β) implicitly weight bills by ideological informativeness, and the contested threshold (10%) removes lopsided votes, but there is no explicit bill-to-factor assignment.

---

## What CFA Would Give Us

### The Basic Idea

Confirmatory Factor Analysis specifies the factor structure *before* estimation:

- **Number of factors:** 2 (ideology + establishment)
- **Loading constraints:** Which bills load on which factors (zero-loading constraints)
- **Factor correlation:** Orthogonal (uncorrelated) or oblique (correlated)

The researcher pre-classifies each bill: "this bill loads on the ideology factor," "this bill loads on the establishment factor," or "this bill loads on both." CFA then estimates the loadings and tests whether the specified structure fits the observed data, reporting fit indices (CFI, TLI, RMSEA, SRMR) that have well-established cutoff norms in psychometrics.

### CFA on Binary Data: The IRT Equivalence

This is the central theoretical insight, and it cuts both ways.

CFA with a probit (or logit) link on binary outcomes is **mathematically equivalent** to a multidimensional IRT (MIRT) model. The equivalence was established by Takane & de Leeuw (1987) and formalized by Kamata & Bauer (2008) and Wirth & Edwards (2007):

| CFA Parameterization | IRT Parameterization | Relationship |
|---------------------|---------------------|--------------|
| Factor loading λⱼ | Discrimination aⱼ | aⱼ ≈ λⱼ × 1.7 (logit-probit scaling) |
| Threshold τⱼ | Difficulty bⱼ | bⱼ = -τⱼ/aⱼ |
| Factor score ηᵢ | Ability/ideal point θᵢ | Same quantity, different name |
| Factor correlation Φ | Latent trait correlation | Same |

The implication: **a 2-factor CFA with probit link on binary vote data would produce ideal points statistically indistinguishable from our 2D M2PL IRT model** (Phase 06), assuming equivalent identification constraints. The models are the same — the differences are in parameterization, identification conventions, and reporting traditions.

This means CFA cannot produce a *better* ideology dimension than 2D IRT. They are estimating the same latent structure. What CFA offers is a different **identification strategy** (zero-loading constraints instead of PLT constraints) and a different **validation framework** (fit indices instead of convergence diagnostics).

### What CFA Specifically Adds

**1. Formal fit indices.** CFI > 0.95, RMSEA < 0.06, and SRMR < 0.08 are widely understood thresholds. The pipeline currently has no single metric that says "the 2-factor structure fits well" vs "it doesn't" — horseshoe detection and convergence tiering are functional substitutes, but they lack the simplicity and familiarity of standard SEM fit indices.

**2. Modification indices.** CFA produces modification indices that identify which constraints are most harmful to model fit — essentially, which bills are misclassified. If a bill assigned to the "ideology" factor actually loads strongly on the "establishment" factor, the modification index flags it. This is useful for iteratively refining the bill classification.

**3. Factor score extraction with known structure.** CFA factor scores are estimated conditional on the specified structure, not a post-hoc rotation. If the structure is correct, these scores have better properties than EFA factor scores rotated to match a target (no rotation indeterminacy).

**4. Multi-group CFA for cross-session comparisons.** CFA naturally extends to measurement invariance testing: does the factor structure hold across sessions? If the same loading constraints fit the 85th and 91st Senates equally well, the ideology dimension is stable over time. This is harder to assess with the current ad-hoc approach.

---

## Why CFA Probably Doesn't Solve Our Core Problem

### The Bill Classification Problem

CFA requires specifying *a priori* which bills load on which factors. For the 2-factor model:

- Bills that load on Factor 1 (ideology) only: partisan bills (party-line votes)
- Bills that load on Factor 2 (establishment) only: intra-party procedural votes
- Bills that load on both: bills with both partisan and procedural components

This classification is the hard part. Options:

| Classification Method | Approach | Problem |
|-----------------------|----------|---------|
| Vote margin | Close margins = ideology, lopsided = establishment | Confounds difficulty with dimension |
| Party-line rate | High party unity = ideology, low = establishment | Circular — defines ideology by party |
| Committee assignment | Judiciary = ideology, Rules = establishment | Kansas committees don't cleanly separate dimensions |
| Topic coding | Tax/abortion = ideology, budget/admin = establishment | Requires external bill metadata; many bills span topics |
| IRT discrimination | High β = ideology, low β = establishment | Circular — uses IRT to constrain IRT |
| EGA communities | Community 1 = ideology, Community 2 = establishment | EGA's communities are exploratory, not confirmatory |

Every classification method is either circular (uses the outcome to define the structure), imprecise (committees and topics don't cleanly separate), or both. This is the fundamental asymmetry between CFA and LDA: **LDA classifies legislators (by party), not bills. CFA classifies bills (by loading constraints), not legislators.** Party labels are observed, well-defined, and unambiguous. Bill-to-dimension assignments are latent, ambiguous, and require exactly the substantive knowledge we are trying to extract.

Fisher's LDA sidesteps this entirely by operating in PCA space using party labels — it finds the party-separating direction without ever classifying individual bills. CFA cannot use this shortcut because its loading matrix operates at the bill level.

### CFA Misspecification is Worse Than EFA Imprecision

If the bill classification is wrong — even partially — CFA produces distorted factor scores. A bill with strong ideology content assigned to the establishment factor forces the model to explain ideological variance through establishment scores, contaminating both factors. CFA's strength (structural constraints) becomes a liability when the constraints are wrong.

EFA/PCA + LDA is more robust to this: PCA extracts whatever structure exists without constraints, and LDA finds the party axis in the resulting space. If a bill has mixed loadings, PCA distributes its variance across components naturally. LDA then weights those components by their party relevance. No bill-level classification needed.

Simulation studies in psychometrics (Schmitt & Sass, 2011; Asparouhov & Muthén, 2009) consistently show that misspecified CFA produces more biased factor scores than well-rotated EFA when the true cross-loadings are non-trivial. Legislative vote data, where many bills have both partisan and procedural components, is exactly this scenario.

### The 2D IRT Already Does What CFA Would Do

Phase 06's M2PL model with PLT identification is estimating the same 2-factor structure that CFA would estimate, just with different identification constraints:

| Feature | CFA Approach | Current M2PL (Phase 06) |
|---------|-------------|------------------------|
| Number of dimensions | Specified: 2 | Specified: 2 |
| Loading structure | Zero constraints on subset of bills | PLT: β[0,1]=0, β[1,1]>0 |
| Factor identification | Loading pattern + scale anchors | PLT rotation + PCA-informed init |
| Estimation | WLSMV or FIML (frequentist) or MCMC (Bayesian) | MCMC via nutpie (Bayesian) |
| Factor correlation | Estimable (oblique) or fixed (orthogonal) | Implicitly orthogonal via PLT |
| Fit assessment | CFI, RMSEA, SRMR | R-hat, ESS, divergences, party-d |
| Factor scores | Regression or Bartlett | Posterior means + HDIs |

The CFA identification (zero-loading constraints) would require bill classification — the hard problem we just discussed. The PLT identification (lower-triangular discrimination matrix) avoids this by using a mathematical constraint that doesn't require substantive knowledge about individual bills. The trade-off: PLT is weaker identification (more rotation freedom in the posterior), but it doesn't inject classification bias.

---

## The Bifactor Model: The Most Promising CFA Variant

If any confirmatory approach merits implementation, it is the **bifactor model** (Gibbons & Hedeker, 1992; Reise, 2012). The bifactor structure avoids the bill-classification problem that undermines standard CFA.

### Structure

```
P(Yea_ij = 1 | θ_Gi, θ_Si) = logit⁻¹(a_Gj × θ_Gi + a_Sj × θ_Si - d_j)
```

- **θ_G (general factor):** Loads on ALL bills. Captures what is common to all voting behavior — the broadest ideological dimension.
- **θ_S (specific factors):** Load on SUBSETS of bills. Capture domain-specific variation after removing the general factor.
- **Orthogonality:** General and specific factors are uncorrelated by construction.

The general factor θ_G is the "pure ideology" score — what remains when domain-specific patterns (fiscal policy quirks, social issue voting, procedural contrarianism) have been partitioned out.

### Why This Solves the Classification Problem (Partially)

The general factor doesn't require bill classification — it loads on everything. Only the specific factors require bill groupings, and the general factor scores are robust to misspecification of specific factors (Reise, Bonifay, & Haviland, 2013). You could group bills by committee, by topic, or even randomly, and the general factor would remain stable because it captures the shared variance across all groupings.

This is fundamentally different from standard 2-factor CFA, where both factors require correct bill classification. In the bifactor model, the factor we care about most (general ideology) is the one that needs no classification at all.

### The Kansas Application

For Kansas roll-call data, a bifactor IRT model might look like:

- **General factor (θ_G):** All ~200-400 contested bills per chamber. This is the "pure ideology" score. It captures the common thread across fiscal, social, procedural, and governance votes.
- **Specific factor 1 (θ_S1):** Fiscal/tax bills (identified by committee or topic code). Captures fiscal conservatism beyond general ideology.
- **Specific factor 2 (θ_S2):** Social/cultural bills. Captures social conservatism beyond general ideology.
- **Specific factor 3 (θ_S3):** Procedural/governance bills. Captures the contrarian/establishment axis — the dimension that currently confounds Tyson's ranking.

The general factor should produce a Tyson score that is conservative but not extreme — her perfect record on high-discrimination bills contributes, but the contrarian pattern on procedural bills is partitioned into θ_S3 rather than inflating θ_G. This is exactly what we want.

### Practical Considerations

**Implementation in PyMC.** The bifactor IRT model is straightforward to specify in the existing PyMC framework:

```python
# Discrimination parameters
a_general = pm.Normal("a_general", mu=0, sigma=1, shape=n_bills)  # all bills
a_specific = pm.Normal("a_specific", mu=0, sigma=1, shape=(n_bills, n_specific))
# Zero out specific loadings for bills not in each group
a_specific_masked = a_specific * bill_group_mask  # binary mask

# Ideal points
theta_general = pm.Normal("theta_general", mu=0, sigma=1, shape=n_legislators)
theta_specific = pm.Normal("theta_specific", mu=0, sigma=1, shape=(n_legislators, n_specific))

# Likelihood
logit_p = (
    a_general[None, :] * theta_general[:, None]
    + pt.sum(a_specific_masked[None, :, :] * theta_specific[:, None, :], axis=-1)
    - difficulty[None, :]
)
pm.Bernoulli("votes", logit_p=logit_p, observed=vote_matrix)
```

**Identification.** The general factor is identified by loading on all items (no rotation indeterminacy for the general factor). Specific factors are identified by being orthogonal to the general factor and loading on non-overlapping bill subsets. Additional anchoring (e.g., fixing one legislator per factor) may be needed for MCMC stability.

**Bill grouping.** The minimal grouping for Kansas would be 2 specific factors: "partisan" (bills with party-line rate > 80%) and "bipartisan" (party-line rate < 50%). This avoids the need for external topic metadata. More refined groupings (committee-based, topic-based) would be better but require metadata integration.

**Computational cost.** The bifactor model adds n_specific × (n_bills + n_legislators) parameters. For n_specific = 3, this roughly doubles the parameter count from the current 2D M2PL. Convergence will be slower, and the relaxed thresholds already used for Phase 06 would likely need further relaxation.

**The ECV diagnostic.** Explained Common Variance (ECV) — the proportion of common variance attributable to the general factor — indicates whether the bifactor model is overkill. If ECV > 0.70, a unidimensional model captures most of the action. If ECV < 0.60, the specific factors carry substantial information. This metric would be valuable for the pipeline's dimensionality assessment, complementing EGA's K estimate.

### Limitations

- **Specific factors can be weak.** If most variance is general (ECV > 0.80), the specific factors will have poor estimation precision — the same Dim 2 convergence problem we already see in Phase 06.
- **Bill grouping still matters** for specific factors (just not for the general factor). Bad groupings produce uninterpretable specific factors.
- **The bifactor model is very flexible** and can overfit. Rodriguez, Reise, & Haviland (2016) warn that bifactor models almost always fit better than alternatives in terms of CFI/RMSEA, even when the improvement is spurious.
- **Integration with canonical routing** would require rethinking the horseshoe detection logic. The general factor might not exhibit the horseshoe at all (since establishment variance is partitioned out), which could simplify or obsolete the current routing system.

---

## How CFA/Bifactor Relates to What We Already Do

### The Supervision Spectrum

The pipeline has been moving along a supervision spectrum over its development:

| Approach | Year | Supervision | What It Uses |
|----------|------|-------------|-------------|
| Raw PCA | Baseline | None | Variance ordering |
| Contested threshold | Early | Implicit (vote-level) | Margin as proxy for partisanship |
| `detect_ideology_pc()` | ADR-0118 | Component-level | Point-biserial correlation with party |
| Manual PCA overrides | ADR-0118 | Full human judgment | Expert session-by-session review |
| Fisher's LDA | ADR-0129 | Legislator-level (party labels) | Optimal party-separating direction |
| 2D IRT | ADR-0054 | None (but init from PCA) | PLT identification on 2D latent space |
| **CFA** | (Proposed) | Bill-level | Zero-loading constraints per bill |
| **Bifactor IRT** | (Proposed) | Bill-group-level | General factor + specific group loadings |

Fisher's LDA and CFA occupy different points on this spectrum. LDA supervises at the legislator level (party membership); CFA supervises at the bill level (loading constraints). Neither is strictly "more supervised" — they use different types of external information.

The bifactor model sits between: it is unsupervised for the general factor (no bill classification needed) and supervised for specific factors (bill groupings required). This makes it a natural extension of the current philosophy: automatic where possible (general ideology), manual where necessary (specific domains).

### What CFA Would Replace vs. Complement

| Current Component | CFA Would... | Bifactor Would... |
|-------------------|-------------|-------------------|
| Phase 02 PCA | Complement (PCA remains for unsupervised discovery) | Complement |
| Fisher's LDA | Compete (CFA defines ideology differently) | Partially replace (general factor ≈ LDA ideology) |
| Phase 05 1D IRT | Be equivalent (same model, different parameterization) | Subsume (general factor is the "better 1D") |
| Phase 06 2D IRT | Be equivalent (same model, different identification) | Subsume (general + specific = better 2D) |
| Canonical routing | Potentially simplify (if general factor avoids horseshoe) | Potentially simplify |
| Phase 02b EGA | Complement (CFA tests, EGA explores) | Complement |

---

## The DW-NOMINATE Precedent

It is worth noting how the field's dominant scaling method handles this problem. DW-NOMINATE:

1. Fits a 2D spatial voting model (always — regardless of whether the second dimension is "needed")
2. Reports Dimension 1 as the primary ideology score
3. Identifies dimensions through SVD initialization (variance-ordered) + iterative MLE refinement
4. Does not attempt to label or constrain dimensions — Dim 1 is "whatever explains the most"

The tallgrass pipeline's approach (ADR-0109 canonical routing) follows this precedent: fit 2D, extract Dim 1, use tiered quality gates to decide when 2D is trustworthy. The key difference is that DW-NOMINATE's Dim 1 is variance-ordered (and therefore vulnerable to axis instability in supermajority chambers — ADR-0127 confirmed this), while tallgrass's canonical routing uses Fisher's LDA to ensure Dim 1 is the *party-separating* direction, not the *maximum-variance* direction.

CFA would be a departure from this precedent: instead of extracting the first dimension from a general 2D model, CFA would constrain the model so that Dim 1 *must* be ideology by construction. This is philosophically closer to Lauderdale & Clark's (2024) IRT-M, which encodes theoretical relationships between items and dimensions through zero constraints.

The bifactor model offers a third path: the general factor captures *everything common* (which is dominated by ideology in roll-call data), and specific factors capture *everything domain-specific*. No need to pre-assign bills to "ideology" — the general factor finds it automatically because ideology is the largest common factor.

---

## Practical Recommendation

### Don't implement standard 2-factor CFA

The bill-classification problem is too severe, and the statistical equivalence with 2D IRT means no improvement in ideal point quality. The fit indices (CFI, RMSEA) would be genuinely useful for model validation, but they can be approximated from the existing Bayesian models using posterior predictive checks (already in Phase 08) without the full CFA apparatus.

### Consider a bifactor IRT as a future Phase 06b experiment

The bifactor model's general factor avoids the bill-classification problem and could produce a "purer" ideology score than the current 2D Dim 1 extraction. The specific factors would formalize the establishment-contrarian axis and potentially resolve the Dim 2 convergence problems (by constraining specific factors to load on bill subsets rather than all bills).

**Prerequisites before implementation:**
1. Bill metadata (committee assignments or topic codes) in the database for specific-factor grouping
2. ECV diagnostic added to Phase 02b or Phase 08 to assess whether bifactor structure is warranted
3. Simulation study on synthetic Kansas-like data to validate that the general factor recovers known ideology scores under realistic signal-to-noise ratios

**Expected benefit:** If ECV is in the 0.50-0.70 range (meaningful specific factors), the bifactor general factor should outperform both 1D IRT (which conflates general and specific) and 2D Dim 1 (which is identified by rotation, not by structural constraints). If ECV > 0.80, the pipeline's current 1D IRT is already adequate and bifactor adds complexity without gain.

### Immediate wins without CFA

Two CFA-adjacent improvements that don't require the full CFA apparatus:

1. **Compute ECV from the existing 2D IRT output.** The explained common variance ratio can be approximated from Phase 06's discrimination parameters: ECV ≈ Σ(a₁²) / [Σ(a₁²) + Σ(a₂²)], where a₁ and a₂ are the Dim 1 and Dim 2 discrimination columns. This tells us how much of the discriminating variance is general (ideology) vs. specific (establishment). No new model needed.

2. **Add a TEFI comparison of 1-factor vs 2-factor solutions to Phase 02b EGA.** EGA already computes TEFI for K=1..5. Reporting the TEFI difference between K=1 and K=2 (and its bootstrap CI) provides a principled dimensionality test that is more interpretable than the current advisory K estimate.

---

## Summary Table: Methods Compared

| Method | Type | Bill Classification? | Ideology Score | Dim 2 Score | Formal Fit Test | In Pipeline? |
|--------|------|---------------------|---------------|-------------|-----------------|-------------|
| PCA | Unsupervised | No | PC1 (or PC2) | PC2 (or PC1) | Parallel analysis | Phase 02 |
| Fisher's LDA | Supervised (party) | No | LDA projection | Orthogonal complement | LOO-CV accuracy | Phase 02 |
| EGA | Unsupervised (network) | No | — | — | TEFI + bootEGA | Phase 02b |
| 1D IRT (2PL) | Generative model | No (implicit via β) | ξ posterior mean | — | PPC, R-hat | Phase 05 |
| 2D IRT (M2PL) | Generative model | No (PLT identification) | Dim 1 posterior mean | Dim 2 posterior mean | PPC, R-hat, party-d | Phase 06 |
| **Standard CFA** | **Confirmatory** | **Yes (zero constraints)** | **Factor 1 score** | **Factor 2 score** | **CFI, RMSEA, SRMR** | **No** |
| **Bifactor IRT** | **Confirmatory (partial)** | **General: No, Specific: Yes** | **General factor θ_G** | **Specific factor θ_S** | **ECV + fit indices** | **No (proposed)** |
| DW-NOMINATE | Spatial voting | No (variance-ordered) | Dim 1 | Dim 2 | GMP, PRE | Phase 16 |
| IRT-M (Lauderdale-Clark) | Theory-constrained IRT | Yes (zero constraints) | Theory-defined Dim 1 | Theory-defined Dim 2 | PPC | No |

---

## Key Takeaways

1. **CFA on binary data is mathematically equivalent to multidimensional IRT.** Switching from IRT to CFA changes the parameterization and identification strategy but not the underlying statistical model. Factor scores from CFA ≈ ideal points from MIRT.

2. **CFA's constraint apparatus (zero loadings) requires bill-level classification** — which is the hardest part of the problem. Fisher's LDA avoids this by classifying legislators (by party) instead of bills. This asymmetry is why LDA works well and CFA would struggle.

3. **The bifactor model is the most promising confirmatory variant** because its general factor loads on all bills (no classification needed) and naturally captures "pure ideology" with domain-specific variance partitioned into specific factors.

4. **The current pipeline (PCA → LDA → IRT → canonical routing) is well-founded** and has been validated through 130 ADRs and extensive empirical testing. CFA would not improve the core ideology scores — it would provide an alternative validation framework (fit indices) for the same latent structure.

5. **The Dim 2 convergence problem is structural, not methodological.** It persists across 1D IRT, 2D IRT, hierarchical models, and W-NOMINATE. CFA and bifactor IRT would face the same challenge: the establishment dimension has ~11% variance and produces non-informative posteriors for most legislators.

6. **The most impactful near-term improvements** are (a) computing ECV from existing Phase 06 output and (b) adding TEFI-based dimensionality comparison to Phase 02b — both achievable without new models.

---

## Literature

- Asparouhov, T., & Muthén, B. (2009). Exploratory structural equation modeling. *Structural Equation Modeling*, 16(3), 397-438.
- Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika*, 75(4), 581-612.
- Clinton, J., Jackman, S., & Rivers, D. (2004). The statistical analysis of roll call data. *APSR*, 98(2), 355-370.
- Gibbons, R. D., & Hedeker, D. (1992). Full-information item bi-factor analysis. *Psychometrika*, 57(3), 423-436.
- Kamata, A., & Bauer, D. J. (2008). A note on the relation between factor analytic and item response theory models. *Structural Equation Modeling*, 15(1), 136-153.
- Lauderdale, B. E., & Clark, T. S. (2024). Measurement that matches theory: Theory-driven identification in IRT models. *APSR*.
- Poole, K. T., & Rosenthal, H. (1997). *Congress: A Political-Economic History of Roll Call Voting*. Oxford University Press.
- Reise, S. P. (2012). The rediscovery of bifactor measurement models. *Multivariate Behavioral Research*, 47(5), 667-696.
- Reise, S. P., Bonifay, W. E., & Haviland, M. G. (2013). Scoring and modeling psychological measures in the presence of multidimensionality. *Journal of Personality Assessment*, 95(2), 129-140.
- Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor models: Calculating and interpreting statistical indices. *Psychological Methods*, 21(2), 137-150.
- Schmitt, T. A., & Sass, D. A. (2011). Rotation criteria and hypothesis testing for exploratory factor analysis. *Journal of Multivariate Analysis*, 102(3), 422-436.
- Takane, Y., & de Leeuw, J. (1987). On the relationship between item response theory and factor analysis of discretized variables. *Psychometrika*, 52(3), 393-408.
- Vafa, K., Naidu, S., & Blei, D. M. (2020). Text-based ideal points. *ACL 2020*.
- Wirth, R. J., & Edwards, M. C. (2007). Item factor analysis: Current approaches and future directions. *Psychological Methods*, 12(1), 58-79.
