# Analysis Design Choices

Every analysis phase makes statistical and methodological choices — priors, thresholds, imputation strategies, encoding rules — that shape the results and carry forward into downstream phases. These documents record those choices so they can be known, questioned, and accounted for.

**These are not ADRs.** ADRs (in `docs/adr/`) document *architectural* decisions (file formats, library choices, directory layout). Design docs here record *statistical* decisions: the assumptions baked into the numbers.

## Reading Order

Read these sequentially. Each phase inherits data from the previous one, so assumptions compound.

| Phase | Document | Key Downstream Effects |
|-------|----------|----------------------|
| 1. EDA | [eda.md](eda.md) | Binary encoding, filtering thresholds, chamber separation |
| 2. PCA | [pca.md](pca.md) | Imputation method, scaling, PC1 sign convention |
| 3. IRT | [irt.md](irt.md) | Priors, anchor selection, missing data handling |
| 4. Clustering | [clustering.md](clustering.md) | Three methods for robustness, party loyalty metric, Ward on Kappa, 3 dendrogram alternatives |
| 4b. LCA | [lca.md](lca.md) | Bernoulli mixture on binary votes, BIC model selection, Salsa effect detection, StepMix FIML |
| 5. Network | [network.md](network.md) | Kappa threshold, Leiden community detection, centrality measures, NaN = no edge |
| 5b. Bipartite Network | [bipartite.md](bipartite.md) | BiCM backbone, Newman projection, bill polarization, bridge bills, Leiden bill communities |
| 6. Prediction | [prediction.md](prediction.md) | XGBoost primary, skip cluster/community labels, IRT features dominate, NLP topic features (NMF on short_title), 82% base rate |
| 7. Indices | [indices.md](indices.md) | CQ-standard party votes, Rice on all votes, Carey UNITY, weighted maverick by chamber margin |
| 8. Synthesis | [synthesis.md](synthesis.md) | Data-driven detection thresholds, graceful degradation, template narratives |
| 9. Profiles | [profiles.md](profiles.md) | Per-legislator deep-dives: 0-1 scorecard metrics, bill discrimination tiers, defection sorting, agreement vs Kappa |
| 2b. UMAP | [umap.md](umap.md) | Cosine metric, n_neighbors default, Procrustes sensitivity, Spearman validation |
| 2c. MCA | [mca.md](mca.md) | Categorical encoding (Yea/Nay/Absent), Greenacre correction, prince library, horseshoe detection, PCA validation |
| 7b. Beta-Binomial | [beta_binomial.md](beta_binomial.md) | Empirical Bayes, per-party-per-chamber priors, method of moments, shrinkage factor |
| 8. Hierarchical IRT | [hierarchical.md](hierarchical.md) | 2-level partial pooling, ordering constraint, non-centered parameterization, ICC, shrinkage vs flat |
| 9. Cross-Session | [cross_session.md](cross_session.md) | Affine IRT alignment, name matching, shift thresholds, prediction transfer, detection validation |
| 10. External Validation | [external_validation.md](external_validation.md) | SM name matching, correlation methodology, career-fixed vs session-specific, outlier z-scores |
| 10b. DIME External Validation | [external_validation_dime.md](external_validation_dime.md) | DIME/CFscore matching, min-givers filter, incumbent-only, cycle-to-biennium mapping |
| 11. TSA | [tsa.md](tsa.md) | Rolling PCA drift, PELT changepoint detection, weekly Rice aggregation, penalty sensitivity |
| 12. Dynamic IRT | [dynamic_irt.md](dynamic_irt.md) | State-space IRT, random walk evolution, per-party tau, polarization decomposition, bridge coverage, post-hoc sign correction (ADR-0068) |
| 13. W-NOMINATE + OC | [wnominate.md](wnominate.md) | Validation-only; R subprocess, polarity via PCA, sign alignment, 3×3 correlation matrix |
| 14. PPC + LOO-CV | [ppc.md](ppc.md) | Validation-only; manual log-likelihood, Q3 local dependence, PSIS-LOO model comparison, graceful degradation |

## Pipeline Phase (Experimental)

| Document | Summary |
|----------|---------|
| [irt_2d.md](irt_2d.md) | 2D M2PL IRT with PLT identification. Pipeline phase 04b with relaxed convergence thresholds (ADR-0046, ADR-0054). |

## Investigations

| Document | Summary |
|----------|---------|
| [beta_prior_investigation.md](beta_prior_investigation.md) | LogNormal β prior creates a blind spot for D-Yea bills. Normal(0,1) fixes it with +3.5% holdout accuracy. |
| [tyson_paradox.md](tyson_paradox.md) | Why the senator with the most Nay votes ranks as the most conservative in IRT. A case study in 1D model limitations. |

## How to Use These

- **Before running a phase:** Read its design doc to understand what assumptions are in play.
- **Before interpreting results:** Check the "Downstream Implications" section of all upstream phases.
- **When results look wrong:** The design doc lists what to check first.
- **When adding a new phase:** Create a design doc following the template below.

## Template

```markdown
# [Phase Name] Design Choices

## Assumptions
What the method assumes about the data and the world.

## Parameters & Constants
Named constants, values, justification, and where they live in code.

## Methodological Choices
For each non-obvious choice: what was decided, what alternatives exist, and what impact it has.

## Downstream Implications
What later phases need to know or account for.
```
