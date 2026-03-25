# Analysis Design Choices

Every analysis phase makes statistical and methodological choices — priors, thresholds, imputation strategies, encoding rules — that shape the results and carry forward into downstream phases. These documents record those choices so they can be known, questioned, and accounted for.

**These are not ADRs.** ADRs (in `docs/adr/`) document *architectural* decisions (file formats, library choices, directory layout). Design docs here record *statistical* decisions: the assumptions baked into the numbers.

## Reading Order

Read these sequentially. Each phase inherits data from the previous one, so assumptions compound.

| Phase | Document | Key Downstream Effects |
|-------|----------|----------------------|
| 01. EDA | [eda.md](eda.md) | Binary encoding, filtering thresholds, chamber separation |
| 02. PCA | [pca.md](pca.md) | Imputation method, scaling, PC1 sign convention, multidimensional interpretation (ADR-0115) |
| 03. MCA | [mca.md](mca.md) | Categorical encoding (Yea/Nay/Absent), Greenacre correction, prince library, horseshoe detection, PCA validation |
| 04. UMAP | [umap.md](umap.md) | Cosine metric, n_neighbors default, Procrustes sensitivity, Spearman validation |
| 05. IRT | [irt.md](irt.md) | Priors, anchor selection, missing data handling |
| 06. 2D IRT | [irt_2d.md](irt_2d.md) | 2D M2PL IRT with PLT identification. Relaxed convergence thresholds (ADR-0046, ADR-0054). |
| 07. Hierarchical IRT | [hierarchical.md](hierarchical.md) | 2-level partial pooling, ordering constraint, non-centered parameterization, ICC, shrinkage vs flat |
| 07b. Hierarchical 2D IRT | [hierarchical_2d.md](hierarchical_2d.md) | Party-pooled M2PL with PLT, informative priors from Phases 06+07, canonical routing preference |
| 08. PPC + LOO-CV | [ppc.md](ppc.md) | Validation-only; manual log-likelihood, Q3 local dependence, PSIS-LOO model comparison, graceful degradation |
| 09. Clustering | [clustering.md](clustering.md) | Three methods for robustness, party loyalty metric, Ward on Kappa, 3 dendrogram alternatives |
| 10. LCA | [lca.md](lca.md) | Bernoulli mixture on binary votes, BIC model selection, Salsa effect detection, StepMix FIML |
| 11. Network | [network.md](network.md) | Kappa threshold, Leiden community detection, centrality measures, NaN = no edge |
| 12. Bipartite Network | [bipartite.md](bipartite.md) | BiCM backbone, Newman projection, bill polarization, bridge bills, Leiden bill communities |
| 13. Indices | [indices.md](indices.md) | CQ-standard party votes, Rice on all votes, Carey UNITY, weighted maverick by chamber margin |
| 14. Beta-Binomial | [beta_binomial.md](beta_binomial.md) | Empirical Bayes, per-party-per-chamber priors, method of moments, shrinkage factor |
| 15. Prediction | [prediction.md](prediction.md) | XGBoost primary, skip cluster/community labels, IRT features dominate, NLP topic features (NMF on short_title), 82% base rate |
| 16. W-NOMINATE + OC | [wnominate.md](wnominate.md) | Validation-only; R subprocess, polarity via PCA, sign alignment, 3×3 correlation matrix |
| 17. External Validation | [external_validation.md](external_validation.md) | SM name matching, correlation methodology, career-fixed vs session-specific, outlier z-scores |
| 18. DIME External Validation | [external_validation_dime.md](external_validation_dime.md) | DIME/CFscore matching, min-givers filter, incumbent-only, cycle-to-biennium mapping |
| 19. TSA | [tsa.md](tsa.md) | Rolling PCA drift, PELT changepoint detection, weekly Rice aggregation, penalty sensitivity |
| 20. Bill Text Analysis | [bill_text.md](bill_text.md) | BERTopic topics (FastEmbed + HDBSCAN), CAP classification (Claude API, optional), bill similarity, caucus-splitting scores |
| 21. Text-Based Ideal Points | [tbip.md](tbip.md) | Embedding-vote approach (not TBIP), vote-weighted bill embeddings + PCA, lower quality thresholds, IRT validation |
| 22. Issue-Specific Ideal Points | [issue_irt.md](issue_irt.md) | Topic-stratified flat IRT on per-topic vote subsets, two taxonomies (BERTopic/CAP), relaxed convergence, anchor stability |
| 23. Model Legislation Detection | [model_legislation.md](model_legislation.md) | ALEC corpus matching, cross-state diffusion (MO/OK/NE/CO), cosine similarity, n-gram overlap confirmation |
| 24. Synthesis | [synthesis.md](synthesis.md) | Data-driven detection thresholds, graceful degradation, template narratives |
| 25. Profiles | [profiles.md](profiles.md) | Per-legislator deep-dives: 0-1 scorecard metrics, bill discrimination tiers, defection sorting, agreement vs Kappa |
| 26. Cross-Session | [cross_session.md](cross_session.md) | Affine IRT alignment, name matching, shift thresholds, prediction transfer, detection validation |
| 27. Dynamic IRT | [dynamic_irt.md](dynamic_irt.md) | State-space IRT, random walk evolution, per-party tau, polarization decomposition, bridge coverage, post-hoc sign correction (ADR-0068) |
| 28. Common Space | [common_space.md](common_space.md) | Pairwise chain affine alignment, delta-method uncertainty, cross-chamber unification via chamber-switchers, career scores via RE meta-analysis (ADR-0120) |

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
