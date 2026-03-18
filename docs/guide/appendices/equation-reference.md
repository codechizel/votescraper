# Appendix B: Equation Reference

> *Every equation in the Tallgrass Guide, collected in one place with plain-English summaries and cross-references.*

---

## How to Use This Reference

Equations are grouped by topic and listed in the order they first appear in the guide. Each entry includes:
- The equation in code-block notation (matching the guide chapters)
- A plain-English translation
- The volume and chapter where it's explained in full

For worked examples and step-by-step walkthroughs, see the referenced chapter.

---

## Agreement and Similarity

### Cohen's Kappa (Vol. 3, Ch. 2)

```
κ = (p_observed − p_expected) / (1 − p_expected)
```

**Plain English:** "How much do two legislators agree, beyond what you'd expect by random chance?"

- κ = 1.0 → perfect agreement
- κ = 0.0 → agreement no better than chance
- κ < 0.0 → systematic disagreement

### Cosine Similarity (Vol. 7, Ch. 3)

```
similarity(A, B) = (embedding_A · embedding_B) / (||embedding_A|| × ||embedding_B||)
```

**Plain English:** "How similar are two documents' meanings?" The dot product of their vector representations, normalized by their lengths. Ranges from -1 (opposite) to +1 (identical).

---

## Dimensionality Reduction

### PCA Score (Vol. 3, Ch. 3)

```
PC1 score for legislator i = w₁ · vote₁ + w₂ · vote₂ + ... + wₙ · voteₙ
```

**Plain English:** "A weighted average of all votes, where the weights (loadings) are chosen to capture the most variation across legislators." Bills that split the chamber contribute more weight; unanimous bills contribute almost nothing.

---

## Item Response Theory (IRT)

### The Core Equation: 1D IRT (Vol. 4, Ch. 2)

```
P(Yea) = logistic(β · ξ − α)
```

**Plain English:** "The probability a legislator votes Yea depends on three things: their ideology (ξ), how sharply the bill divides legislators (β), and how hard the bill is to pass (α)."

- ξ (xi) = legislator's ideal point (latent ideology)
- α (alpha) = bill difficulty (high α → harder to vote Yea)
- β (beta) = bill discrimination (high β → sharply separates liberal from conservative)
- logistic(x) = 1 / (1 + exp(-x)) — maps any number to [0, 1]

### The Tipping Point (Vol. 4, Ch. 2)

```
Tipping point = α / β
```

**Plain English:** "The ideology score at which a legislator has exactly 50% probability of voting Yea." Legislators to the right of the tipping point are more likely to vote Yea; legislators to the left are more likely to vote Nay.

### 2D IRT (Vol. 4, Ch. 4)

```
P(Yea) = logistic(β₁·ξ₁ + β₂·ξ₂ − α)
```

**Plain English:** "Same as 1D, but the legislator has two ideology dimensions and the bill has two discrimination parameters." Dimension 1 typically captures the party divide; Dimension 2 captures a secondary axis (often within-party variation).

### IRT Priors (Vol. 4, Ch. 2)

```
ξ  ~  Normal(0, 1)       — ideal points: most legislators near center
α  ~  Normal(0, 5)       — difficulty: wide range allowed
β  ~  Normal(0, 1)       — discrimination: moderate values expected
```

**Plain English:** "Before seeing any data, we assume ideal points cluster around zero, difficulty can vary widely, and discrimination is moderate. The data then updates these starting assumptions."

### Reflection Invariance (Vol. 4, Ch. 3)

```
P(Yea) = logistic((−β) · (−ξ) − α) = logistic(β · ξ − α)
```

**Plain English:** "Flipping the sign of both ideology and discrimination produces the same vote probabilities. This is why IRT needs anchors — the model can't tell left from right on its own."

---

## Hierarchical IRT

### The Hierarchical Ideal Point (Vol. 4, Ch. 5)

```
μ_party  ~  Normal(0, 2), ordered so that Democrat < Republican
σ_within  ~  HalfNormal(σ_scale)
offset_i  ~  Normal(0, 1)
ξ_i  =  μ_party[p_i]  +  σ_within[p_i]  ·  offset_i
```

**Plain English:** "Each party has an average ideology (μ) and a spread (σ). Each legislator's ideal point equals their party's average plus an individual offset, scaled by the party's spread." This is *non-centered parameterization* — sampling the offset (always standard normal) rather than ξ directly.

### Intraclass Correlation (Vol. 4, Ch. 5)

```
ICC = σ²_between / (σ²_between + σ²_within)
```

**Plain English:** "What fraction of the total variation in ideology is explained by party membership?" ICC near 1.0 means party explains almost everything; ICC near 0.5 means within-party variation is as large as between-party variation.

### Minimum-Separation Penalty (Vol. 4, Ch. 5)

```
Penalty = switch(μ_R − μ_D > 0.5, 0.0, −100.0)
```

**Plain English:** "A soft constraint requiring the Republican party mean to be at least 0.5 units above the Democrat party mean. Violations are penalized heavily (-100 log-probability), making them effectively impossible."

---

## Validation Metrics

### Geometric Mean Probability (Vol. 5, Ch. 2)

```
GMP = exp(mean(log(p_correct)))
```

**Plain English:** "The geometric average of the model's confidence in the correct outcome across all votes." GMP of 0.90 means the model assigns, on average, 90% probability to what actually happened. Penalizes confident wrong answers more than uncertain ones.

### APRE (Vol. 5, Ch. 2)

```
APRE = (model_accuracy − baseline_accuracy) / (1 − baseline_accuracy)
```

**Plain English:** "How much does the model reduce errors compared to always predicting the majority? APRE = 1.0 means the model makes no errors; APRE = 0.0 means it's no better than the baseline."

### Cohen's d for Party Separation (Vol. 5, Ch. 5)

```
Cohen's d = |mean(R) − mean(D)| / pooled_SD
```

**Plain English:** "How far apart are the two parties in standard deviation units?" d > 1.5 indicates strong party separation — the typical threshold for accepting IRT results.

---

## Party Cohesion and Indices

### Rice Index (Vol. 6, Ch. 5)

```
Rice Index = |Yea − Nay| / (Yea + Nay)
```

**Plain English:** "What fraction of the party voted together, beyond 50-50?" Rice = 1.0 means unanimous; Rice = 0.0 means perfectly split.

### Caucus-Splitting Score (Vol. 7, Ch. 3)

```
Caucus-splitting score = 1 − Rice_Index(majority party)
```

**Plain English:** "How much did a bill divide the majority party?" High scores mean the party split; low scores mean the party held together.

---

## Empirical Bayes (Beta-Binomial)

### Binomial Likelihood (Vol. 6, Ch. 6)

```
P(y | θ, n) = C(n,y) · θ^y · (1−θ)^(n−y)
```

**Plain English:** "The probability of observing y party-line votes out of n total, given a true loyalty rate of θ."

### Method of Moments (Vol. 6, Ch. 6)

```
α + β = μ(1 − μ) / v − 1
α = μ · (α + β)
β = (1 − μ) · (α + β)
```

**Plain English:** "Estimate the Beta prior's shape parameters from the data's mean (μ) and variance (v). Higher α + β means a more concentrated prior — the group is more certain about typical loyalty rates."

### Posterior (Vol. 6, Ch. 6)

```
Posterior = Beta(α + y, β + (n − y))
Posterior mean = (α + y) / (α + y + β + (n − y))
```

**Plain English:** "Add the prior's pseudo-counts (α, β) to the observed counts (y successes, n-y failures). The posterior mean is the weighted average of the prior mean and the observed rate."

### Shrinkage Factor (Vol. 6, Ch. 6)

```
shrinkage = (α + β) / (α + β + n)
Posterior mean = (1 − shrinkage) × (raw rate) + shrinkage × (prior mean)
```

**Plain English:** "The posterior mean is a weighted average of the raw rate and the prior mean. The weight on the prior (shrinkage) decreases as the sample size (n) increases. Many votes → trust the data; few votes → lean on the group average."

### Credible Interval (Vol. 6, Ch. 6)

```
CI = [Beta.ppf(0.025, α_post, β_post), Beta.ppf(0.975, α_post, β_post)]
```

**Plain English:** "The 95% credible interval is the range from the 2.5th to the 97.5th percentile of the Beta posterior distribution."

---

## Dynamic IRT and Temporal Models

### Evolution Equation (Vol. 8, Ch. 2)

```
xi[t] = xi[t-1] + tau · innovation[t-1]
innovation ~ Normal(0, 1)
```

**Plain English:** "A legislator's ideology at time t equals their ideology at time t-1, plus a random step whose size is controlled by tau. Small tau = slow drift; large tau = rapid change."

### Initial Ideal Point Prior (Vol. 8, Ch. 2)

```
xi_init ~ Normal(mu = static_IRT_mean, sigma = 1.5)
```

**Plain English:** "The starting ideology is centered on the static IRT estimate from Volume 4, with enough slack (σ = 1.5) that the dynamic model can find its own scale. This transfers the sign convention from static to dynamic IRT."

### Affine Alignment (Vol. 8, Ch. 3)

```
xi_aligned = A · xi_original + B
```

**Plain English:** "Stretch (A) and shift (B) one session's ideal points to match another session's scale, calibrated using legislators who served in both sessions."

### Conversion-Replacement Decomposition (Vol. 8, Ch. 4)

```
Total shift = Conversion effect + Replacement effect
Conversion = mean(xi_returning in B) − mean(xi_returning in A, aligned)
Replacement = Total shift − Conversion
```

**Plain English:** "A party's ideological shift equals how much the returning members changed (conversion) plus how different the newcomers are from the people they replaced (replacement)."

---

## Text and Prediction

### Text-Based Legislator Profile (Vol. 7, Ch. 4)

```
text_profile[legislator] = sum(vote[bill] × embedding[bill]) / n_votes_cast
```

**Plain English:** "A legislator's text-based ideology is the vote-weighted average of the bills they supported. Yea votes pull toward a bill's semantic content; Nay votes push away."

### Z-Score Discrepancy (Vol. 7, Ch. 4)

```
discrepancy_z = |z_score(text) − z_score(IRT)|
```

**Plain English:** "How different is a legislator's text-based ideology from their vote-based ideology, measured in standard deviations? Large discrepancies flag legislators whose textual associations don't match their voting patterns."

### Interaction Feature (Vol. 7, Ch. 1)

```
xi_x_beta = xi_mean × beta_mean
```

**Plain English:** "The product of a legislator's ideology and a bill's discrimination. High positive values mean a conservative legislator facing a bill that separates along the conservative-liberal axis — a strong predictor of Yea."

---

*Back to: [Appendices](./)*
