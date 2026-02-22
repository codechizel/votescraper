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
| 4. Clustering | [clustering.md](clustering.md) | Three methods for robustness, party loyalty metric, Ward on Kappa |
| 5. Network | [network.md](network.md) | Kappa threshold, Louvain multi-resolution, centrality measures, NaN = no edge |
| 6. Prediction | [prediction.md](prediction.md) | XGBoost primary, skip cluster/community labels, IRT features dominate, 82% base rate |
| 7. Indices | [indices.md](indices.md) | CQ-standard party votes, Rice on all votes, weighted maverick by chamber margin |
| 8. Synthesis | [synthesis.md](synthesis.md) | Data-driven detection thresholds, graceful degradation, template narratives |
| 2b. UMAP | [umap.md](umap.md) | Cosine metric, n_neighbors default, Procrustes sensitivity, Spearman validation |

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
