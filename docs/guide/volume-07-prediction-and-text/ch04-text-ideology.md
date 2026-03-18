# Chapter 4: Text-Based Ideology and Issue-Specific Scores

> *IRT measures ideology from vote patterns. But what if you could also measure it from the words of the bills legislators support? And what if a legislator's ideology changes depending on the topic — conservative on taxes but moderate on education?*

---

## Two Ideas, One Chapter

This chapter covers two related analyses that extend ideology measurement in different directions.

The first — **text-based ideal points** — asks whether you can derive ideology from the *content* of the bills a legislator supports, rather than just from the voting pattern. It takes the bill embeddings from Chapter 3 and uses them to build an alternative ideology score that comes from the semantic meaning of legislation, not from the Yea/Nay matrix.

The second — **issue-specific IRT** — runs separate IRT models for each policy topic. Instead of asking "how conservative is Senator Smith overall?", it asks "how conservative is Senator Smith on education? On taxation? On criminal justice?" The answer isn't always the same — and the differences reveal which policy areas are ideologically aligned (voting conservative on taxes predicts voting conservative on criminal justice) and which are cross-cutting (education votes don't track the overall liberal-conservative divide as cleanly).

## Part I: Text-Based Ideal Points

### The Idea

Volume 4 estimated ideology from the pattern of votes:

```
Votes → IRT → Ideal points
```

Chapter 3 showed that bills have meaning — each one is a 384-dimensional semantic fingerprint. What if we combined the two?

```
Bill embeddings + Vote directions → Text profiles → PCA → Text-based ideal points
```

The intuition: a legislator who votes Yea on tax cuts, gun rights, and school choice is supporting a *cluster of policy content*. A legislator who votes Yea on Medicaid expansion, environmental regulation, and education funding is supporting a *different cluster*. The content of the bills they support tells you something about their ideology — something that goes beyond the binary Yea/Nay pattern.

### Why Not True TBIP?

The academic version of this idea is called **TBIP** (Text-Based Ideal Points), published by **Vafa, Naidu, and Blei** in 2020. TBIP fits a sophisticated Bayesian model that jointly estimates ideal points from both votes and the text of bills each legislator *sponsors*.

Tallgrass can't use true TBIP because of a data constraint: **about 92% of Kansas bills are committee-sponsored**. Only ~27 individual legislators sponsor ~38 bills in a typical session — far too few for the model to estimate individual text profiles from sponsorship alone. True TBIP needs a legislature where most bills have individual sponsors (like the U.S. Congress). Kansas's committee-centric process makes sponsorship data too sparse.

Instead, Tallgrass uses a **vote-weighted embedding approach** — a simpler but more broadly applicable method.

### The Vote-Weighted Embedding Profile

The method works in six steps:

**Step 1: Load bill embeddings.** From Chapter 3, every bill has a 384-dimensional embedding vector — its meaning fingerprint.

**Step 2: Build the vote matrix.** For each legislator-bill pair, encode the vote as +1 (Yea), -1 (Nay), or 0 (absent/not voting).

**Step 3: Compute the weighted average.** For each legislator, multiply each bill's embedding by their vote (+1 or -1) and average across all bills they voted on:

```
text_profile[legislator] = sum(vote[bill] × embedding[bill]) / n_votes_cast
```

**Plain English:** "Take the meaning fingerprint of every bill this legislator voted on. If they voted Yea, add the fingerprint. If they voted Nay, subtract it. Average the result."

A legislator who consistently votes Yea on tax-related bills will have a text profile that "points toward" the tax cluster in embedding space. A legislator who consistently votes Yea on education bills will point toward the education cluster. The profile is a 384-dimensional summary of *what kinds of legislation this person supports*.

**Step 4: PCA on the profiles.** With ~125 legislators and 384-dimensional profiles, run PCA to extract the dominant dimension of variation. PC1 — the direction along which legislators' text profiles vary the most — becomes the text-based ideal point.

**Step 5: Align sign convention.** PCA doesn't know which direction is "conservative" and which is "liberal." Tallgrass checks the correlation between PC1 and the IRT ideal points from Volume 4. If the correlation is negative (text-conservative maps to IRT-liberal), flip the sign.

**Step 6: Compare to IRT.** Compute the Pearson correlation between text-based scores and IRT ideal points. This tells you how much the content of supported legislation agrees with the voting-pattern-based ideology.

**Codebase:** `analysis/21_tbip/tbip.py` (full pipeline), `analysis/21_tbip/tbip_data.py` (`build_vote_embedding_profiles()`)

### What the Correlation Tells You

The correlation between text-based and IRT ideal points typically falls in the 0.65-0.85 range. This is lower than the external validation correlations from Volume 5 (where IRT correlated 0.90+ with Shor-McCarty scores), and that's expected. The text score is **twice removed** from the underlying ideology:

```
Ideology → Votes → IRT ideal point       (one step of inference)
Ideology → Votes → Bill content → Embedding → PCA → Text ideal point  (four steps of inference)
```

Each step adds noise. The embedding model wasn't trained on legislative text. The vote weighting averages across all bills (diluting topic-specific signal). PCA extracts only the first component (discarding secondary dimensions). Given all this, a correlation of 0.75 is genuinely strong — it means the content of supported legislation captures most of the same ideological variation that the vote pattern does.

Tallgrass uses adjusted quality thresholds for text-based scores:

| Pearson r | Quality Label | Interpretation |
|-----------|--------------|----------------|
| >= 0.80 | Strong | Text profiles capture ideology well |
| 0.65-0.80 | Good | Expected range — text adds signal |
| 0.50-0.65 | Moderate | Captures partisan direction, less within-party variation |
| < 0.50 | Weak | Text and votes may capture different dimensions |

### Outliers: When Text and Votes Disagree

The most interesting legislators are those where the text-based and IRT scores **diverge**. Tallgrass identifies the top 5 outliers per chamber — legislators whose text profile puts them in a different ideological position than their vote pattern.

The **discrepancy** is measured in standardized units:

```
discrepancy_z = |z_score(text) - z_score(IRT)|
```

A legislator with a large discrepancy might be someone who votes the party line on most bills (moderate IRT score) but consistently supports legislation about a specific issue area that pulls their text profile in an unexpected direction. For example, a moderate Republican who votes with the party on procedural and budget matters but sponsors or supports unusually progressive education bills would show up as a text-IRT outlier.

These outliers connect back to the "issue-specific" analysis in Part II — they're often the legislators whose ideology genuinely varies by topic.

**Codebase:** `analysis/21_tbip/tbip_data.py` (`identify_outliers()`, `OUTLIER_TOP_N = 5`)

## Part II: Issue-Specific IRT

### The Motivation

The IRT models from Volume 4 give each legislator a single ideology score. But nobody is uniformly conservative or uniformly liberal across every issue. A legislator might be economically conservative (favoring tax cuts, opposing spending increases) but socially moderate (supporting education funding, opposing strict criminal sentencing). A single number compresses these distinctions into a blurry average.

Issue-specific IRT de-compresses the picture. It takes the topic assignments from Chapter 3, subsets the votes by topic, and runs a separate 1D IRT model on each topic. Each legislator gets a per-topic ideal point: one for education bills, one for tax bills, one for criminal justice, and so on.

### The Method

**Step 1: Select eligible topics.** Not every topic has enough bills for IRT to work. The model requires at least 10 bills with roll call votes per topic and at least 10 legislators with 5+ votes in that topic. Topics that fall short are skipped.

**Step 2: Subset the vote matrix.** For each eligible topic, extract only the votes on bills assigned to that topic. The resulting sub-matrix is smaller — maybe 15-40 bills instead of 500 — but still has the legislator-bill structure that IRT needs.

**Step 3: Fit 1D IRT per topic.** The same flat IRT model from Volume 4, Chapter 2 (2PL with anchor constraints, nutpie Rust NUTS sampler) is applied to each subset. The model estimates per-topic ideal points and per-bill parameters.

**Step 4: Sign-align to the full IRT.** Each topic's IRT is fit independently, so the sign is arbitrary. Tallgrass correlates each topic's ideal points with the full-model ideal points and flips the sign if necessary, ensuring that "positive" always means "more conservative" across all topics.

**Codebase:** `analysis/22_issue_irt/issue_irt.py` (full pipeline), `analysis/22_issue_irt/issue_irt_data.py` (`subset_vote_matrix_for_topic()`)

### Relaxed Convergence Standards

The per-topic models are smaller and noisier than the full model. With only 15-40 bills (instead of 500+), the posterior is less precise, and the MCMC sampler may struggle to converge as tightly. Tallgrass relaxes the convergence standards accordingly:

| Diagnostic | Full IRT (Phase 05) | Per-Topic IRT (Phase 22) |
|-----------|---------------------|--------------------------|
| **R-hat threshold** | < 1.01 | < 1.05 |
| **ESS threshold** | > 400 | > 200 |
| **Chains** | 2 | 2 |
| **Samples per chain** | 2000 | 1000 |
| **Tuning steps** | 1000 | 1000 |

The relaxation is principled: smaller datasets produce noisier posteriors, and demanding tight convergence on 15 bills would cause most topics to fail. The thresholds are still meaningful — R-hat < 1.05 means the chains agree to within 5%, and ESS > 200 ensures stable mean and credible interval estimates.

**Codebase:** `analysis/22_issue_irt/issue_irt.py` (`RHAT_THRESHOLD = 1.05`, `ESS_THRESHOLD = 200`)

### Cross-Topic Correlations: The Ideological Map of Policy

The payoff comes from comparing the per-topic ideal points to each other. For every pair of topics, Tallgrass computes the Pearson correlation between their ideal point vectors:

```
r(education, taxation) = correlation of per-legislator ideal points across the two topics
```

The result is a **cross-topic correlation matrix** — a heatmap that shows which policy areas sort legislators the same way and which don't.

**Highly correlated topics** (r > 0.80): These topics are ideologically aligned. A legislator who is conservative on taxation is also conservative on criminal justice — the two issues track the same underlying dimension. In Kansas, most topic pairs fall in this range, reflecting the strength of the partisan divide.

**Moderately correlated topics** (0.40 < r < 0.80): These topics partially track the main ideological dimension but introduce some independent variation. Education often falls in this range — it broadly tracks the left-right divide but has enough cross-cutting dynamics (rural vs. suburban, establishment vs. populist) to produce distinct patterns.

**Weakly correlated or uncorrelated topics** (r < 0.40): These topics define a **different** ideological axis. A legislator's position on this topic tells you little about their position on the main dimension. Agriculture is sometimes in this category — voting on water rights and agricultural subsidies may align more with a rural/urban divide than with the traditional liberal/conservative spectrum.

### Cross-Pressured Legislators

The issue-specific analysis identifies **cross-pressured legislators** — those whose per-topic ideal point differs substantially from their overall score. These are computed as z-score discrepancies:

```
discrepancy_z = |z(topic_xi) - z(overall_xi)|
```

The top 10 outliers per topic are reported. A cross-pressured legislator might be:
- A conservative Republican who is moderate on education (personally invested in school funding)
- A moderate Democrat who is hawkish on criminal justice (represents a high-crime district)
- A rural Republican who votes liberally on agriculture (defending farming subsidies against free-market ideologues)

These are the legislators whose single-number IRT score is most misleading — and they're often the ones who make the most interesting political decisions.

### Anchor Stability

Each per-topic IRT uses the same anchor strategy as the full model (Volume 4, Chapter 3), with the anchors chosen from the full-model extremes. But do these anchors stay at the extremes in every topic?

Tallgrass checks **anchor stability** by tracking where the full-model conservative and liberal anchors rank in each per-topic distribution. If the conservative anchor is the most conservative legislator on taxation (rank 1 out of 85), that's stable. If they're only the 30th most conservative on education, the anchor has "drifted" — education sorts legislators by a different criterion than the overall model.

Anchor drift is a signal, not an error. It tells you that the topic uses a different ideological axis, and the per-topic scores should be interpreted with that in mind.

**Codebase:** `analysis/22_issue_irt/issue_irt.py` (`check_anchor_stability()`)

## What Can Go Wrong

### Too Few Bills Per Topic

BERTopic might discover 15 topics but only 5 have enough bills with roll calls for issue-specific IRT. The analysis reports which topics were eligible and which were skipped, so the reader knows what's covered.

### Convergence Failures on Small Subsets

Even with relaxed thresholds, some topics with 10-15 bills may fail to converge. Tallgrass marks these as non-converged and reports them with appropriate caveats. The ideal points are still reported (posterior means are usually reasonable even with imperfect convergence), but the credible intervals should not be trusted.

### Sign Alignment Ambiguity

If a per-topic IRT's ideal points have near-zero correlation with the full model (because the topic captures a genuinely different dimension), the sign alignment becomes arbitrary. Tallgrass flags topics where the alignment correlation is below 0.30 — these topics may need manual inspection to determine which direction is "conservative."

---

## Key Takeaway

Text-based ideal points validate IRT from a completely different angle: the content of the bills legislators support tells a consistent story about their ideology, with typical correlations of 0.65-0.85. Issue-specific IRT reveals that ideology is not one-dimensional across all policy areas — some topics (taxation, criminal justice) track the main partisan divide closely, while others (education, agriculture) introduce independent variation. Cross-pressured legislators — those whose per-topic scores diverge from their overall position — are often the most politically interesting members of the chamber.

---

*Terms introduced: text-based ideal point, vote-weighted embedding profile, TBIP (Text-Based Ideal Points), committee sponsorship constraint, signal attenuation, issue-specific IRT, cross-topic correlation, cross-pressured legislator, anchor stability, anchor drift*

*Previous: [Topic Modeling: What Are the Bills About?](ch03-topic-modeling.md)*

*Next: [Model Legislation: Detecting Copy-Paste Policy](ch05-model-legislation.md)*
