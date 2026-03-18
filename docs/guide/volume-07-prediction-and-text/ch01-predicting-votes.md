# Chapter 1: Predicting Votes: What XGBoost Learns

> *If you know a legislator's ideology, the bill's discrimination, and a few other things — can you predict how they'll vote before the roll is called?*

---

## The Ultimate Test of Understanding

Throughout this guide, we've built up a picture of the Kansas Legislature: ideology scores from IRT, coalitions from clustering, relationships from network analysis. But here's a question that puts all of it to the test: **can we predict individual votes?**

Measurement and prediction are different games. Measurement asks "what happened?" — it looks at how legislators voted and assigns them ideology scores. Prediction asks "what *will* happen?" — given what we know about a legislator and a bill, can we say how the legislator will vote before the roll call?

Getting prediction right means we truly understand the mechanics of legislative voting, not just its surface patterns. Getting it wrong in interesting ways — the *surprising* votes where the model is confident but the legislator does the opposite — tells us which legislators are genuinely unpredictable and which votes defy the simple left-right logic.

## XGBoost: A Committee of Weak Experts

The prediction engine is **XGBoost** (eXtreme Gradient Boosting), the algorithm that has dominated tabular data competitions for the past decade. If you've heard of Kaggle (the data science competition platform), you've encountered XGBoost — it's won more Kaggle competitions than any other single algorithm.

### The Analogy: A Panel of Advisors

Imagine you're trying to predict how Senator Smith will vote on HB 400. You gather a panel of 200 very simple advisors. Each advisor can only ask one question about one feature of the data — something like "Is the legislator more conservative than +0.3?" or "Does this bill have discrimination above 1.0?" — and based on the answer, make a tentative prediction.

The first advisor gives their best guess. It's mediocre — they're making a complex prediction from a single data point. The second advisor looks at *where the first advisor was wrong* and focuses their one question on correcting those mistakes. The third advisor corrects the second advisor's remaining errors. And so on, through all 200.

By the end, the 200 simple advisors — each individually unimpressive — collectively form a remarkably accurate prediction system. Each advisor contributes a small correction to the previous group's combined prediction. Together, their corrections add up.

This is gradient boosting. The "gradient" is the mathematical direction of each correction. The "boosting" is the process of building each new advisor to fix the previous advisors' mistakes. And "XGBoost" is a particular implementation, created by **Tianqi Chen** in 2014, that does this with exceptional speed and several clever tricks for preventing overfitting.

### How XGBoost Differs from Logistic Regression

Tallgrass actually trains three models on each chamber: logistic regression, XGBoost, and random forest. Logistic regression draws a single straight line through the feature space (it asks: "is a weighted sum of the features above or below a threshold?"). Random forest builds 200 independent decision trees and averages their predictions. XGBoost builds 200 sequential trees, each correcting the last.

The key advantage of XGBoost over logistic regression is that it can learn **nonlinear relationships** and **interactions** between features without being told. Logistic regression needs you to explicitly create an interaction term (like ideology times discrimination) to capture the way those features work together. XGBoost discovers interactions on its own — if the combination of high ideology and high discrimination predicts Yea, it learns that pattern directly from the tree structure.

**Codebase:** `analysis/15_prediction/prediction.py` (`XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1)`)

## The Features: What the Model Knows

What information does the model get to work with? Think of it as the briefing packet for our panel of advisors. For each legislator-bill pair, the model sees 14 core features plus a handful of one-hot encoded vote types:

### Legislator Features

| Feature | Plain English | Where It Comes From |
|---------|--------------|---------------------|
| `xi_mean` | How conservative the legislator is | Phase 05: IRT ideal point |
| `xi_sd` | How uncertain the ideology estimate is | Phase 05: IRT posterior SD |
| `party_binary` | Republican (1) or Democrat (0) | Legislator roster |
| `loyalty_rate` | How often they vote with their party | Phase 09: Clustering |
| `betweenness` | How much they bridge voting blocs | Phase 11: Network |
| `eigenvector` | How influential in the voting network | Phase 11: Network |
| `pagerank` | How central in the voting network | Phase 11: Network |
| `PC1`, `PC2` | Primary and secondary voting dimensions | Phase 02: PCA |

### Bill Features

| Feature | Plain English | Where It Comes From |
|---------|--------------|---------------------|
| `alpha_mean` | How easy the bill is to pass | Phase 05: IRT difficulty |
| `beta_mean` | How partisan the bill is | Phase 05: IRT discrimination |
| `is_veto_override` | Whether it's a veto override vote | Roll call metadata |
| `day_of_session` | How far into the session | Vote date (ordinal) |

### The Crucial Interaction

One feature stands above all others: **`xi_x_beta`** — the product of the legislator's ideal point and the bill's discrimination. This is the IRT interaction term from Volume 4:

```
xi_x_beta = xi_mean × beta_mean
```

**Plain English:** "How well does this legislator's ideology match what this bill is testing?"

A conservative legislator (xi = +1.5) on a highly discriminating bill (beta = +2.0) gets xi_x_beta = +3.0 — a strong positive signal toward Yea. A liberal legislator (xi = -1.0) on the same bill gets xi_x_beta = -2.0 — a strong negative signal toward Nay. The interaction captures the core IRT mechanism: the probability of a Yea vote depends not just on how conservative you are or how partisan the bill is, but on the *product* of the two.

### What's Excluded (and Why)

Some features are deliberately left out because they would **leak the answer**:

- **`yea_count`, `nay_count`, `margin`** — These directly encode the outcome. Knowing that a bill passed 70-55 makes it easy to predict individual Yea votes, but it's cheating: you can't know the vote count before the vote happens.
- **Cluster labels** — Redundant with party and IRT scores. Including them would just double-count information the model already has.
- **`alpha_mean` (bill difficulty)** — Used for vote prediction but excluded from bill passage prediction, because difficulty is estimated *from* the passage outcome.

Only Yea and Nay votes are included. Absences, "Present" votes, and "Not Voting" entries are dropped — the model predicts the direction of votes actually cast.

**Codebase:** `analysis/15_prediction/prediction.py` (features built in `extract_vote_features()`, exclusions documented in `EXCLUDED_VOTE_FEATURES`)

## The Evaluation Framework

### The 82% Problem

Before looking at results, we need to acknowledge a baseline problem. About 82% of all votes in the Kansas Legislature are Yea. A model that always predicts Yea, without looking at any data, would be "82% accurate."

This makes accuracy a terrible metric for this task. An accuracy of 90% sounds impressive until you realize it's only 8 percentage points better than a strategy that ignores every feature.

That's why Tallgrass uses **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve) as the primary metric. AUC measures something different: how well the model *ranks* Yea votes above Nay votes across all possible thresholds. A perfect model has AUC = 1.0. A coin flip has AUC = 0.5. And critically, AUC is not inflated by the 82% base rate.

### Three Baselines

To interpret the model's performance honestly, Tallgrass computes three baselines:

1. **Majority class** (always predict Yea): Accuracy ~82%, AUC = 0.50. This is the floor.
2. **Party-only** (use only the party label): Accuracy ~85%, AUC ~0.75-0.85. This is the "cheap prediction" — it costs nothing to know a legislator's party.
3. **Full model** (XGBoost with all features): Accuracy ~95-97%, AUC ~0.98.

The interesting question is the gap between the party-only baseline and the full model. That gap is what all our IRT, network, and PCA analysis adds beyond simply knowing which jersey each legislator wears.

### Train/Test Split and Cross-Validation

The evaluation is rigorous in a specific, standard way:

1. **Stratified split:** 80% of legislator-bill pairs are randomly assigned to training, 20% to a holdout test set. The stratification ensures both sets have the same ratio of Yea to Nay votes.
2. **5-fold cross-validation** on the training set: the training data is split into 5 parts, and the model is trained on 4 parts and evaluated on the 5th, rotating through all 5 combinations. This gives confidence intervals on training performance.
3. **Holdout evaluation:** The final models are evaluated on the 20% holdout that was never touched during training. This is the number that matters.

**Codebase:** `analysis/15_prediction/prediction.py` (`N_SPLITS = 5`, `TEST_SIZE = 0.20`, `RANDOM_SEED = 42`)

## What the Model Learns

### SHAP: Opening the Black Box

XGBoost is sometimes called a "black box" — it makes accurate predictions but doesn't naturally explain why. **SHAP** (SHapley Additive exPlanations) opens that box.

SHAP comes from **cooperative game theory** — specifically from the work of **Lloyd Shapley** (1953), who won the Nobel Prize in Economics for his contributions. In a cooperative game, the question is: how much does each player contribute to the team's total payoff? Shapley's insight was to compute each player's contribution by considering every possible team they could join, averaging their marginal contribution across all those teams.

SHAP applies this to machine learning: each "player" is a feature, the "team" is the full model, and the "payoff" is the prediction. For a specific prediction, SHAP tells you how much each feature pushed the prediction toward Yea or toward Nay.

### The Feature Ranking

Across Kansas sessions, the SHAP analysis consistently reveals the same hierarchy:

**The xi_x_beta interaction dominates.** The product of ideology and discrimination is by far the most important feature — typically 3-5 times more important than the next feature. This shouldn't be surprising: it's the core IRT mechanism. The model is rediscovering the fundamental equation from Volume 4: the probability of a Yea vote depends on the product of ideology and discrimination.

**Bill discrimination (beta_mean) ranks second.** How partisan a bill is matters a great deal — not just through the interaction term but on its own. Bills with near-zero discrimination (procedural motions, unanimous resolutions) are easy to predict: everyone votes Yea. Bills with high discrimination are where the action is.

**Ideology (xi_mean) ranks third.** A legislator's position on the spectrum matters, but it matters most through the interaction with discrimination. Knowing someone is conservative doesn't help much if the bill isn't partisan.

**Party (party_binary) ranks surprisingly low** — typically 5th-8th. This confirms that IRT has already captured what party membership provides. Party is a crude proxy for ideology; the continuous IRT ideal point is a precise measure. Once the model has xi_mean, knowing the party label adds little.

**Network centrality and PCA contribute at the margins.** Betweenness, eigenvector centrality, and PageRank each add small amounts of predictive power, mostly for the cross-aisle votes that ideology alone can't explain.

### A Worked SHAP Example

Consider a specific prediction: Senator Jones (xi = +0.6, moderate Republican) on HB 250 (beta = +1.8, highly discriminating):

```
Base prediction:     0.82  (the overall Yea rate — the model's starting point)
xi_x_beta = +1.08:  +0.11  (ideology × discrimination pushes toward Yea)
beta_mean = +1.8:   +0.04  (high-discrimination bill, slight Yea lean)
loyalty_rate = 0.87: -0.02  (slightly below party average, tiny Nay push)
is_veto_override:   +0.00  (not a veto override, no effect)
...
Final prediction:    0.93  (93% probability of Yea)
```

The SHAP values are additive: the base rate (0.82) plus all the SHAP values equals the final predicted probability (0.93). Each value tells you exactly how much that feature contributed.

**Codebase:** `analysis/15_prediction/prediction.py` (`shap.TreeExplainer`, `TOP_SHAP_FEATURES = 15`)

## Per-Legislator Accuracy: Who Defies the Model?

The aggregate AUC of 0.98 masks an important pattern: the model doesn't predict everyone equally well. Some legislators are easy targets; others are frustratingly unpredictable.

Tallgrass computes per-legislator accuracy on the holdout set, requiring at least 10 holdout votes for a reliable estimate. The results form a characteristic **U-shape** when plotted against ideology:

- **High accuracy at the extremes** (xi < -1.0 or xi > +1.5): Legislators at the poles vote predictably along party lines. The model gets them right 98%+ of the time.
- **Lower accuracy in the middle** (xi near 0 to +0.5): Moderates are harder to predict because they sometimes vote with their party and sometimes cross the aisle. Their votes depend on bill-specific factors that the model only partially captures.

### The Hardest to Predict

The model identifies the 8 most difficult legislators in each chamber and provides plain-English explanations for why they're hard:

| Pattern | Explanation |
|---------|-------------|
| **Moderate (close to the boundary)** | Their ideal point sits between the parties — some bills pull them left, some right |
| **Centrist for their party** | Not extreme, but within their party's range — they break ranks on specific issues |
| **Strongly positioned but occasional crossover** | Far from the center but defects on a few key issues — the model expects party-line votes and gets surprised |
| **Pattern doesn't fit the one-dimensional model** | Their voting may reflect a second dimension (establishment vs. maverick) that the 1D model can't capture |

These labels connect back to upstream findings. The "moderates" are often the same legislators identified as bridges in the network analysis (Volume 6, Chapter 3). The "doesn't fit 1D" legislators are often the ones the 2D IRT model captures better (Volume 4, Chapter 4). The prediction analysis confirms what the measurement phases found, from a completely different angle.

**Codebase:** `analysis/15_prediction/prediction.py` (`detect_hardest_legislators()`, `MIN_VOTES_RELIABLE = 10`, `HARDEST_N = 8`)

## Surprising Votes: When the Model Is Confidently Wrong

The most interesting output isn't the model's successes — it's its failures. A **surprising vote** is one where the model was highly confident but wrong.

The model assigns each vote a probability: "I'm 95% sure Senator Smith will vote Yea on HB 400." If Senator Smith votes Nay, that's a surprising vote. The **confidence error** — the gap between the predicted probability and the actual outcome — measures just how surprising it was.

Tallgrass splits surprising votes into two categories:

**Surprising Nays** (the model predicted Yea but got Nay): These are unexpected dissent. A legislator who "should" have voted with their party or with their ideology — based on everything the model knows — chose not to. These are the most common type of surprise because of the 82% Yea base rate: the model predicts Yea for most votes, so most of its errors are false positives.

**Surprising Yeas** (the model predicted Nay but got Yea): These are unexpected support. A legislator who the model expected to vote against a bill instead supported it. These are rarer but often more politically interesting — they might indicate logrolling, personal relationships, or issue-specific positions that cross party lines.

The top 20 surprising votes per chamber are reported as a ranked table, sorted by confidence error. These votes are computed exclusively on the holdout set — the model never saw them during training, so the surprises are genuine, not artifacts of overfitting.

**Codebase:** `analysis/15_prediction/prediction.py` (`find_surprising_votes()`, `TOP_SURPRISING_N = 20`)

## The Caveat: Explanatory, Not Truly Predictive

This section is the most important in the chapter, and it requires honesty.

The IRT features — xi_mean, alpha_mean, beta_mean, and their interaction — are estimated from the same votes that the model is trying to predict. The ideal point of Senator Smith is computed from *all* of Senator Smith's votes. The difficulty and discrimination of HB 400 are computed from *all* votes on HB 400. When the model uses these features to predict Senator Smith's vote on HB 400, it's partially circular: the features already encode information about the outcome.

This doesn't mean the model is useless. The 80/20 train-test split mitigates the problem — the model is evaluated on votes it never saw during training. But the features themselves still "know" about the test set, because IRT was fit on the entire vote matrix. True out-of-sample prediction would require fitting IRT on only the training votes, which would produce different (and slightly less accurate) ideal point estimates.

Tallgrass is transparent about this. The report labels the vote prediction as **explanatory** — it measures how well the features explain voting patterns, not how well the system would perform as a real-time vote forecaster. The IRT model's dominance as a feature is partially tautological: IRT was designed to model exactly these voting patterns, so of course it predicts them well.

The per-legislator accuracy and surprising vote analyses are still genuinely informative, because they test specific predictions against held-out data. But the headline AUC of 0.98 should be read as "the IRT model captures nearly all the variance in voting" rather than "we can predict votes with 98% accuracy in real time."

## The Metrics at a Glance

| Metric | What It Measures | Typical Value | Baseline |
|--------|-----------------|---------------|----------|
| **AUC-ROC** | Ranking quality (primary metric) | ~0.98 | 0.50 (random), 0.75-0.85 (party-only) |
| **Accuracy** | Fraction correct | ~95-97% | ~82% (majority class) |
| **Brier score** | Probability calibration (lower = better) | ~0.03-0.05 | 0.25 (random) |
| **Log-loss** | Cross-entropy (lower = better) | ~0.10-0.15 | 0.693 (random) |
| **Precision** | Of predicted Yeas, how many are correct | ~0.97 | — |
| **Recall** | Of actual Yeas, how many are caught | ~0.98 | — |

---

## Key Takeaway

XGBoost achieves a 0.98 AUC for vote prediction, with the IRT interaction term (ideology times discrimination) dominating all other features. This confirms that IRT captures the fundamental structure of legislative voting. Party labels add little once continuous ideology is included. The model's failures are informative: surprising votes reveal cross-pressured legislators and issue-specific defections, while the hardest-to-predict legislators tend to be the same moderates and bridge-builders identified by upstream analyses. The caveat: IRT features are estimated from the same votes being predicted, making this explanatory rather than truly predictive.

---

*Terms introduced: XGBoost, gradient boosting, AUC-ROC, ROC curve, SHAP (SHapley Additive exPlanations), feature interaction, confidence error, surprising vote, explanatory vs. predictive, Brier score, calibration curve, stratified train/test split, k-fold cross-validation*

*Next: [Bill Passage: Can We Forecast Outcomes?](ch02-bill-passage.md)*
