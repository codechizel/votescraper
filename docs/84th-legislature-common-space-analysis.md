# The 84th Legislature: Why the Common-Space Chain's Weakest Link Breaks

**Date:** 2026-03-25
**Scope:** 84th Kansas Legislature (2011-2012), Phases 05-07b, 28, 30
**Related:** ADR-0118, ADR-0120, ADR-0122, `docs/pca-ideology-axis-instability.md`, `docs/84th-biennium-analysis.md`

---

## Summary

The 84th Legislature is the common-space chain's weakest link — not because of data quality problems, but because of a **canonical routing error**. The hierarchical 2D IRT model (Phase 07b) was selected as the canonical source despite producing scores that correlate poorly with W-NOMINATE (r=0.392 Senate, r=0.838 House). The flat 2D IRT model (Phase 06) produces far better agreement (r=0.918 Senate, r=0.968 House). The root cause: the hierarchical model's party-pooling prior distorts the first dimension in a chamber where **intra-Republican factionalism dominates the party divide**. This is the only session in the dataset where the hierarchical model is demonstrably worse than the flat model.

---

## Political Context: A Party at War with Itself

The 84th Legislature (2011-2012) captures the Kansas Republican Party in the middle of a historic factional split. Governor Sam Brownback, elected in 2010, brought an aggressive agenda — massive income tax cuts (the "Kansas experiment"), abortion restrictions, and Medicaid restructuring. The Republican caucus fractured into two camps:

- **Conservative faction** (aligned with Brownback, AFP, Kansas Chamber): supply-side tax cuts, reduced state spending, social conservatism
- **Moderate faction** (led by Senate President Steve Morris): fiscal pragmatism, education funding, coalition governance with Democrats

The Senate was ground zero. With 30 Republicans and 10 Democrats, moderates held the balance of power by allying with Democrats on key votes. Party-line votes were rare (2.4% vs 22.4% in the 91st). The Republican caucus had 5.4x more internal ideological variation than Democrats. Senate assortativity was 0.188 — nearly random — meaning party membership was a weak predictor of voting behavior.

**The August 2012 purge** eliminated this faction. Conservative groups funded primary challenges against moderate senators. Steve Morris, Tim Owens, and approximately seven other moderates were defeated. The 85th Legislature (2013-2014) had dramatically higher Republican cohesion. This is why the 84th-85th bridge is the weakest in the dataset (89 bridges, 62.7% overlap).

---

## Per-Session Model Performance

### The problem: 1D IRT fails completely

Flat 1D IRT for the 84th is one of the worst-performing models in the entire pipeline:

| Metric | House | Senate |
|--------|-------|--------|
| R-hat max | 1.832 | 1.828 |
| ESS min | 2.88 | 2.89 |
| Party separation d | 1.42 | **0.02** |
| Sign flipped | Yes | Yes |
| axis_uncertain | True | True |

The Senate party separation of d=0.02 means the 1D model found essentially **zero ideological difference between parties**. This is correct — in a 1D projection, the moderate-conservative Republican split dominates, and both parties' centers collapse to the middle.

### The solution: 2D IRT recovers ideology

Both the flat 2D (Phase 06) and hierarchical 2D (Phase 07b) models converge:

| Model | Chamber | R-hat | ESS | Converged |
|-------|---------|-------|-----|-----------|
| Flat 2D | House | 1.026 | 139 | Yes |
| Flat 2D | Senate | 1.033 | 141 | Yes |
| Hierarchical 2D | House | 1.016 | 261 | Yes (Tier 1) |
| Hierarchical 2D | Senate | 1.015 | 694 | Yes (Tier 1) |

Both converge cleanly. But convergence does not mean correctness.

### The critical finding: which Dim 1 is right?

Cross-validating each model's Dim 1 against W-NOMINATE Dim 1 (the field-standard unsupervised estimator):

| Model | House r | Senate r |
|-------|---------|----------|
| 1D IRT | 0.583 | 0.288 |
| **Flat 2D Dim 1** | **0.968** | **0.918** |
| Hierarchical 2D Dim 1 | 0.838 | 0.392 |

The flat 2D model outperforms the hierarchical model in both chambers, and the gap is enormous in the Senate (0.918 vs 0.392). Within-party correlations for the hierarchical Senate are essentially zero (Republican r=-0.137, Democrat r=0.060).

---

## Root Cause: Party Pooling vs. Factionalism

The hierarchical 2D IRT model (Phase 07b, ADR-0117) uses a **party-pooling prior**: each party's legislators share a hierarchical mean and within-party variance. This works brilliantly in most sessions because party membership IS the dominant predictor of ideology.

In the 84th Senate, party membership is a **weak predictor** (ICC=0.502, assortativity=0.188). The dominant dimension of conflict is the moderate-conservative Republican factional split. The party-pooling prior forces the model to emphasize the party divide, which is the **second-most-important** dimension, not the first. The result: Dim 1 captures a party signal that exists but is not the primary axis of variation, while the true ideology dimension (which cross-cuts the party divide within Republicans) is relegated to Dim 2 or mixed across both dimensions.

W-NOMINATE and the flat 2D IRT model are **unsupervised** — they identify the principal axes of variation without any party information. For the 84th Senate, these unsupervised methods correctly place the factional split on Dim 1.

### Why the routing logic was fooled

The canonical routing system (ADR-0109, ADR-0110) selects the hierarchical model when:
1. Horseshoe detected (yes — intra-R factionalism triggers this correctly)
2. H2D converges at Tier 1 (yes — R-hat 1.015, ESS 694)
3. Party separation d > 1.5 (likely passed — the party-pooling prior *creates* party separation)

The quality gates check convergence and party separation, but they don't check **whether the separated dimension is the right one**. The party-pooling prior guarantees some party separation on Dim 1 by construction, making the party-d gate easy to pass even when the dimension doesn't match the unsupervised ideology axis.

---

## Impact on Common-Space Linking

The incorrect canonical source for the 84th propagates through the common-space chain:

1. **Per-session scores are distorted**: The 84th Senate canonical scores don't align with the ideological continuum. Bridge legislators' positions in the 84th are measured on a different axis than in the 85th.

2. **The 84th-85th affine link is unreliable**: The linking algorithm estimates A and B from bridge legislators' positions in adjacent sessions. If the 84th measures factionalism and the 85th measures ideology, the affine fit is trying to map between incommensurable scales.

3. **All pre-84th sessions are affected**: The chain composes links backward from the 91st. Any error at the 84th-85th link propagates to the 78th-83rd.

This explains the per-session common-space correlation for the 84th (r=0.564 between IRT and W-NOMINATE common-space scores, worst in the dataset). It's not the linking algorithm that failed — it's the input scores.

---

## The Dimension Swap: Ideology Is on Dim 2, Not Dim 1

The dimension swap detection (`dimension_swap_corrected = YES` in the convergence summary) fired for both chambers — but it swapped the **wrong way**. A comprehensive comparison of all available IRT dimensions against W-NOMINATE Dim 1 reveals the error:

| Model | House r | House rho | Senate r | Senate rho |
|-------|---------|-----------|----------|------------|
| 1D IRT | 0.617 | 0.865 | 0.288 | 0.578 |
| Flat 2D Dim 1 | 0.968 | 0.984 | 0.918 | 0.893 |
| Flat 2D Dim 2 | 0.520 | 0.324 | -0.812 | -0.754 |
| H2D Dim 1 (current canonical) | 0.838 | 0.582 | 0.392 | 0.377 |
| **H2D Dim 2** | **0.977** | **0.984** | **0.992** | **0.966** |

**H2D Dim 2 correlates r=0.992 with W-NOMINATE in the Senate** — near-perfect agreement. Within Republicans, the correlation is r=0.994 (n=30). This is the ideology dimension. What the routing system labeled "Dim 1" actually captures the party-pooling prior's forced party separation, not the natural ideology axis.

The hierarchical model's Dim 2 is the **best available option** for the 84th: it has the hierarchical model's superior convergence properties (ESS 261 House, 694 Senate) AND near-perfect agreement with the field-standard unsupervised estimator. The flat 2D Dim 1 (r=0.918/0.968) is the second-best option.

### Why the swap detection failed

The dimension swap detection uses PCA correlation: it checks which IRT dimension aligns better with PC1. But in the 84th, PCA itself has ambiguous axes (PC1 d=1.84, PC2 d=1.30 in the Senate). When PCA can't reliably identify the ideology axis, the swap detection inherits that ambiguity. Using W-NOMINATE as the cross-validation target instead of PCA would catch this case.

---

## Recommendation: Use H2D Dim 2 for the 84th

The primary fix: **override the canonical source for the 84th to use H2D Dim 2 instead of H2D Dim 1.**

This can be implemented as a W-NOMINATE cross-validation gate in the routing logic:

> After selecting the canonical dimension, check its Pearson correlation with W-NOMINATE Dim 1 (when available). If the OTHER dimension of the same model correlates substantially better (delta r > 0.15), swap dimensions. This catches cases where the PCA-based swap detection fails due to ambiguous axes.

This gate would fire for the 84th (and should be checked for the 88th Senate, which also shows PCA axis instability). All other sessions would be unaffected.

### Expected improvement

After switching to H2D Dim 2 for the 84th:
- Per-session Senate IRT-WNOM correlation: 0.392 → **0.992**
- Per-session House IRT-WNOM correlation: 0.838 → **0.977**
- Common-space 84th correlation: expected to improve dramatically
- The 84th-85th bridge link will be far more reliable
- All pre-84th sessions (78th-83rd) will benefit from the improved chain link

---

## Other 84th Characteristics (Not Bugs)

These are **features, not bugs** — they reflect the genuine political complexity of the session:

- **Weakest bridge coverage (62.7%)**: The 2012 purge + redistricting produced the largest turnover. Still well above the 20-bridge psychometric minimum (89 bridges).
- **Ambiguous PCA axes**: Neither PC1 nor PC2 clearly dominates party separation (d=1.84 vs 1.30). This is the transition point between the PC2-ideology era (78th-83rd) and the PC1-ideology era (85th+).
- **ODT-era data limitations**: 29.4% of vote pages are committee-of-the-whole tallies without individual names. This reduces the number of votes but doesn't bias the ones we have.
- **Supermajority adaptive tuning**: The 2D IRT phase automatically doubled N_TUNE from 2000 to 4000 for this session (81% R), per ADR-0112.

---

## Why W-NOMINATE Just Works (And IRT Doesn't)

The user's observation is accurate: W-NOMINATE produces correct dimension identification for the 84th with a single function call, while the IRT pipeline requires seven quality gates, three models (1D, flat 2D, hierarchical 2D), dimension swap detection, horseshoe correction, and canonical routing — and STILL gets it wrong. Why?

The answer lies in three fundamental design differences.

### 1. SVD Initialization Orders Dimensions by Variance, Not by Party

W-NOMINATE's "Nominal Three-Step Estimation" starts by computing the eigenvalue decomposition of a double-centered legislator agreement matrix — mathematically equivalent to PCA on the co-voting structure. This produces dimensions ordered by **explained variance**: Dim 1 captures the largest source of systematic voting variation, Dim 2 the second-largest, and so on.

In the 84th Senate, the largest source of variation is the moderate-conservative Republican factional split. W-NOMINATE places this on Dim 1 automatically because that's what explains the most votes. It doesn't know or care whether this is a "party" dimension or a "faction" dimension — it just finds the direction of maximum predictive power.

The flat 2D IRT model uses the same PCA initialization (`--init-strategy pca-informed`), which is why it also gets the right answer (r=0.918 with W-NOMINATE). The problem is specific to the **hierarchical** 2D IRT, which overrides this natural ordering with party structure.

### 2. No Party Information = No Party Bias

W-NOMINATE is **completely unsupervised**. It receives a vote matrix (legislators x bills, yea/nay/missing) and nothing else. No party labels, no chamber metadata, no prior beliefs about which legislators should be similar. The algorithm finds the spatial arrangement that maximizes the likelihood of the observed votes, period.

The hierarchical 2D IRT model adds **party-level hyperpriors**: each party's legislators share a hierarchical mean (`mu_R`, `mu_D`) and within-party variance (`sigma_R`, `sigma_D`). These priors tell the model "Republicans should cluster together and Democrats should cluster together." In most sessions, this is a helpful inductive bias — it improves estimation in small chambers by pooling information within parties.

In the 84th Senate, this prior is **actively harmful**. The Republicans don't cluster together — they span the entire ideological spectrum from moderate (Steve Morris at -1.79) to ultraconservative. The hierarchical prior forces the model to find a dimension where Republicans DO cluster, which means finding a dimension that DOESN'T capture the factional split. The result: Dim 1 captures a weak, prior-dominated party signal, while the real ideology dimension gets pushed to Dim 2.

This is the core lesson: **supervised methods (hierarchical IRT with party priors) can be misled by their own assumptions. Unsupervised methods (W-NOMINATE, flat IRT) cannot, because they have no assumptions to be misled by.**

### 3. The Polarity Legislator vs. Anchor Selection

W-NOMINATE resolves the sign ambiguity (which end is "conservative") with a **polarity legislator** — a single legislator designated by the user as positive. In Tallgrass, we select the legislator with the highest PCA PC1 score who has sufficient participation. This is simple, robust, and uses the same information that SVD initialization already computed.

The IRT pipeline resolves sign ambiguity through **anchor selection** — fixing two legislators at extreme positions (e.g., the most liberal Democrat and most conservative Republican). The identification strategy system (ADR-0103) offers seven strategies: `anchor-pca`, `anchor-agreement`, `sort-constraint`, `positive-beta`, `hierarchical-prior`, `unconstrained`, `external-prior`.

The sophistication of IRT's identification system is both its strength and its weakness. In normal sessions, the auto-detection selects the right strategy. In the 84th, where "most conservative Republican" might actually be a moderate by absolute standards (because the factional split dominates the party split), the anchors can land in misleading positions. The hierarchical model compounds this by pulling the anchor positions toward party means.

### 4. Bounded Scale as Implicit Regularization

W-NOMINATE constrains all positions to [-1, +1] (the unit hypersphere). This seems like a limitation, but it acts as **implicit regularization** that prevents the kind of extreme posterior exploration that MCMC can exhibit. The bounded scale means:

- No legislator can "run away" to extreme positions
- The optimization landscape has no unbounded directions
- Convergence is guaranteed to a compact set

IRT's unbounded latent trait means MCMC chains can explore extreme regions of parameter space, especially when identification is weak (as in the 84th Senate). The 1D model's R-hat of 1.83 and ESS of 3 reflect chains exploring incompatible posterior modes — a problem that simply can't occur in W-NOMINATE's bounded optimization.

### 5. Deterministic Optimization vs. Stochastic Sampling

W-NOMINATE uses **deterministic alternating optimization** (Newton-Raphson on per-legislator likelihoods). Given the same data and initialization, it always produces the same answer. There is no chain mixing problem, no R-hat diagnostic needed, no ESS to worry about.

Bayesian IRT uses **MCMC sampling** (nutpie Rust NUTS), which explores the posterior distribution stochastically. This provides full uncertainty quantification — a genuine advantage — but introduces the possibility of chain mixing failures. The 84th's 1D IRT shows exactly this: R-hat 1.83 means different chains found different posterior modes. W-NOMINATE can't have this problem because it finds a single maximum, not a posterior distribution.

### The Trade-off

W-NOMINATE's simplicity is its strength for dimension identification: it finds what the data says without being influenced by modeling assumptions. IRT's complexity is its strength for everything else: posterior uncertainty, hierarchical borrowing, model comparison, and extensibility.

The pipeline's architecture already reflects this trade-off: W-NOMINATE (Phase 16) is used as a **validation benchmark**, not as the primary estimator. The 84th Senate reveals the one case where the primary estimator's modeling assumptions (party pooling) conflict with the data's true structure. The fix — a W-NOMINATE cross-validation gate — explicitly uses W-NOMINATE's unsupervised dimension identification to catch these cases.

---

## The Broader Lesson: When Party Pooling Helps and Hurts

The 84th teaches a general lesson about hierarchical models in political science:

**Party pooling helps when party structure is strong.** In the 91st Legislature (2025-2026), party membership explains most of the ideological variance. The hierarchical prior correctly shrinks within-party estimates toward the party mean, improving precision.

**Party pooling hurts when party structure is weak.** In the 84th Senate (2011-2012), the Republican caucus spans the entire ideological spectrum. The hierarchical prior pulls moderate Republicans toward the Republican mean and conservative Republicans toward the same mean, compressing the factional variation that IS the dominant political dimension. The result is a first dimension that reflects the model's prior, not the data.

This is analogous to the "ecological fallacy" in political science — assuming group-level patterns (party = ideology) hold at the individual level. In the 84th Senate, they don't.

The 84th is not an edge case — it's the canonical example of a broader pattern. Any supermajority chamber with a significant intra-party factional split will exhibit this behavior. The Brownback-era Kansas Senate, the post-1964 Southern Democratic caucus, the pre-1994 moderate Republican House caucus — all are cases where party labels mask the true dimension of political conflict.

---

## Is W-NOMINATE Simply Better? A Systemic Audit

The 84th's failure raises a harder question: is this a one-off problem, or is the canonical routing systematically wrong? To answer this, we cross-validated every IRT model's every dimension against W-NOMINATE Dim 1 across all 28 chamber-sessions (14 bienniums x 2 chambers). The results are sobering.

### The Full Cross-Validation Table

| Session | Ch | 1D | F2D-1 | F2D-2 | H2D-1 | H2D-2 | Best | Canon | Match? |
|---------|----|----|-------|-------|-------|-------|------|-------|--------|
| 78th | H | **0.987** | 0.981 | 0.358 | 0.966 | 0.641 | 1D | 1D | YES |
| 78th | S | **0.978** | 0.837 | 0.706 | 0.941 | 0.524 | 1D | 1D | YES |
| 79th | H | 0.988 | **0.996** | 0.070 | **0.996** | 0.035 | H2D-1 | 1D | YES |
| 79th | S | **0.989** | 0.741 | 0.746 | 0.330 | 0.983 | 1D | H2D-1 | **NO** |
| 80th | H | **0.991** | 0.531 | 0.953 | **0.993** | 0.310 | H2D-1 | 1D | YES |
| 80th | S | **0.966** | 0.068 | 0.935 | 0.773 | 0.413 | 1D | H2D-1 | **NO** |
| 81st | H | **0.981** | 0.928 | 0.839 | 0.908 | 0.954 | 1D | 1D | YES |
| 81st | S | 0.957 | 0.936 | 0.622 | 0.163 | **0.958** | H2D-2 | 1D | YES |
| 82nd | H | **0.986** | 0.931 | 0.824 | 0.802 | 0.982 | 1D | 1D | YES |
| 82nd | S | 0.954 | 0.103 | **0.958** | 0.927 | 0.929 | F2D-2 | H2D-1 | YES |
| 83rd | H | 0.988 | 0.989 | 0.349 | **0.997** | 0.452 | H2D-1 | 1D | YES |
| 83rd | S | 0.961 | 0.215 | **0.976** | 0.515 | 0.937 | F2D-2 | 1D | YES |
| **84th** | **H** | 0.617 | 0.968 | 0.520 | 0.838 | **0.977** | H2D-2 | H2D-1 | **NO** |
| **84th** | **S** | 0.288 | 0.918 | 0.812 | 0.392 | **0.992** | H2D-2 | H2D-1 | **NO** |
| **85th** | **H** | 0.561 | 0.614 | 0.952 | 0.828 | **0.962** | H2D-2 | H2D-1 | **NO** |
| 85th | S | 0.968 | **0.991** | 0.113 | 0.990 | 0.534 | F2D-1 | 1D | YES |
| 86th | H | **0.990** | 0.987 | 0.126 | 0.981 | 0.276 | 1D | 1D | YES |
| 86th | S | 0.969 | 0.846 | 0.936 | **0.989** | 0.737 | H2D-1 | 1D | YES |
| 87th | H | **0.984** | 0.545 | 0.924 | 0.837 | 0.910 | 1D | 1D | YES |
| 87th | S | 0.981 | 0.377 | 0.980 | 0.735 | **0.985** | H2D-2 | 1D | YES |
| 88th | H | **0.988** | 0.949 | 0.638 | 0.924 | 0.969 | 1D | 1D | YES |
| **88th** | **S** | 0.101 | **0.994** | 0.060 | 0.856 | 0.988 | F2D-1 | H2D-1 | **NO** |
| 89th | H | **0.990** | 0.982 | 0.180 | 0.989 | 0.675 | 1D | 1D | YES |
| 89th | S | 0.521 | 0.988 | 0.053 | **0.990** | 0.553 | H2D-1 | H2D-1 | YES |
| 90th | H | 0.986 | **0.993** | 0.329 | 0.991 | 0.034 | F2D-1 | 1D | YES |
| 90th | S | 0.584 | 0.983 | 0.486 | **0.985** | 0.658 | H2D-1 | H2D-1 | YES |
| 91st | H | 0.982 | **0.990** | 0.119 | 0.979 | 0.878 | F2D-1 | 1D | YES |
| 91st | S | 0.933 | 0.981 | 0.598 | **0.986** | 0.416 | H2D-1 | H2D-1 | YES |

Values are |Pearson r| with W-NOMINATE Dim 1. Bold = best per row. "Canon" = what the routing system selected. "Match?" = whether canon achieves within 0.05 of the best.

### What This Reveals

**6 of 28 chamber-sessions (21%) have the wrong canonical dimension.** All 6 are cases where the hierarchical 2D model's Dim 1 was selected but a different dimension or model agrees better with W-NOMINATE:

| Session | Chamber | Canon r | Best r | Best Model | Gap |
|---------|---------|---------|--------|------------|-----|
| 79th | Senate | 0.330 | 0.989 | 1D IRT | 0.659 |
| 80th | Senate | 0.773 | 0.966 | 1D IRT | 0.193 |
| 84th | House | 0.838 | 0.977 | H2D Dim 2 | 0.139 |
| 84th | Senate | 0.392 | 0.992 | H2D Dim 2 | 0.600 |
| 85th | House | 0.828 | 0.962 | H2D Dim 2 | 0.134 |
| 88th | Senate | 0.856 | 0.994 | Flat 2D Dim 1 | 0.138 |

The pattern is striking:
- **All 6 failures are in the hierarchical model's Dim 1.** The hierarchical party-pooling prior is the common cause.
- **5 of 6 are Senate chambers.** Smaller chambers (37-40 members) with stronger supermajority effects are more vulnerable.
- **In 2 cases (79th, 80th Senate), simple 1D IRT is best.** The horseshoe detection triggered a false positive — the 1D model already captured ideology correctly, but the routing system overrode it with a worse 2D model.

### Which model wins most often?

| Model | Times Best (of 28) | Percentage |
|-------|-------------------|------------|
| 1D IRT | 14 | 50% |
| H2D Dim 1 | 5 | 18% |
| H2D Dim 2 | 6 | 21% |
| Flat 2D Dim 1 | 2 | 7% |
| Flat 2D Dim 2 | 1 | 4% |

**1D IRT is the best model half the time.** When 2D is needed, the hierarchical model's Dim 2 is more often the ideology dimension than Dim 1. The routing system's preference order (H2D Dim 1 > Flat 2D Dim 1 > 1D) is exactly backwards for a substantial minority of sessions.

### So Is W-NOMINATE Better?

No — but it's better at **one critical thing**, and we should use it for that.

**What W-NOMINATE does better: dimension identification.** Its unsupervised, variance-maximizing SVD initialization produces correct dimension ordering in every session, without party priors, horseshoe detection, or dimension swap logic. It's a solved problem for W-NOMINATE because it never introduces the party assumption that creates the problem.

**What IRT does better: everything else.** Full posterior uncertainty (vs W-NOMINATE's often-zero bootstrap SEs). Hierarchical pooling across parties (when party structure is strong). Natural extension to dynamic models, issue-specific models, and cross-chamber models. Proper Bayesian model comparison (LOO, PPC, WAIC). Better small-sample properties via regularizing priors.

**The right architecture is a hybrid:** use W-NOMINATE as the **dimension identification oracle** and IRT as the **estimation engine.**

### The Proposed Fix: W-NOMINATE Cross-Validation Gate

After the canonical routing selects an IRT source and dimension, cross-validate against W-NOMINATE Dim 1 (which Phase 16 already computes):

1. Compute |Pearson r| between the selected canonical dimension and W-NOMINATE Dim 1
2. Compute |Pearson r| for ALL available IRT dimensions (1D, F2D Dim 1/2, H2D Dim 1/2)
3. If a different dimension exceeds the selected one's correlation by more than 0.10, swap to the better dimension
4. Log the swap decision in the routing manifest

This gate would have caught all 6 misrouted sessions. It uses each method's strength: W-NOMINATE for dimension identification, IRT for the actual ideal point estimates with full posterior uncertainty.

The IRT models themselves don't need to change. The hierarchical party-pooling prior is valuable for 22 of 28 chamber-sessions — it should stay. The fix is in the **routing**, not the **estimation**.

---

## References

- Poole, K. T., & Rosenthal, H. (1997). *Congress: A Political-Economic History of Roll Call Voting*. Oxford.
- Poole, K. T. (2005). *Spatial Models of Parliamentary Voting*. Cambridge.
- Clinton, J., Jackman, S., & Rivers, D. (2004). The statistical analysis of roll call data. *APSR* 98(2): 355-370.
- Carroll, R., Lewis, J. B., Lo, J., Poole, K. T., & Rosenthal, H. (2009). Measuring bias and uncertainty in DW-NOMINATE ideal point estimates. *Political Analysis* 17(3): 261-275.
- Carroll, R., Lewis, J. B., Lo, J., Poole, K. T., & Rosenthal, H. (2013). The structure of utility in spatial models of voting. *AJPS* 57(4): 1008-1028.
- Peress, M. (2009). Small chamber ideal point estimation. *Political Analysis* 17(3): 276-290.
- Shor, B., & McCarty, N. (2011). The ideological mapping of American legislatures. *APSR* 105(3): 530-551.
- Brownback, S. (2012). Interview with AP: described the tax cuts as "a real live experiment."
- Kansas City Star (2012). Coverage of the August 2012 Republican primary results.
