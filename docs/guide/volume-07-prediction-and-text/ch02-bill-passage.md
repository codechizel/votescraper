# Chapter 2: Bill Passage: Can We Forecast Outcomes?

> *Chapter 1 asked "how will this legislator vote?" Chapter 2 asks a harder question: "will this bill pass?" It's harder because you can't use the most powerful features — you don't get to peek at individual legislator scores when predicting a bill-level outcome.*

---

## A Different Kind of Prediction

Vote prediction (Chapter 1) is a legislator-bill task with 30,000-60,000 data points per chamber. Bill passage prediction is a **bill-level** task with only 250-500 data points per chamber — roughly 100 times fewer. Every bill in the training set becomes a single row: pass or fail.

This changes the game entirely. With 30,000+ examples, XGBoost can learn subtle patterns and fine-grained interactions. With 300 examples, the model must rely on broader, more robust features. Overfitting becomes a real risk — the model could memorize the training bills instead of learning generalizable patterns.

The analogy: predicting individual votes is like predicting whether each student will pass each question on a test (lots of data, fine-grained). Predicting bill passage is like predicting whether the entire class will pass the test (fewer observations, higher stakes per observation).

## The Feature Set: Working Without Ideology Scores

Bill passage prediction can't use legislator-level features — there's no specific legislator to characterize. It can't use bill difficulty (alpha_mean), because difficulty is estimated from the very passage outcome it's trying to predict. And it can't use vote counts, because those *are* the outcome.

What's left?

### The Core Features

| Feature | Plain English | Why It's Not Leakage |
|---------|--------------|---------------------|
| `beta_mean` | How partisan the bill is | Discrimination captures how the bill *divides* the chamber, not whether it passes |
| `is_veto_override` | Whether it's a veto override vote | Known before the vote occurs |
| `day_of_session` | How far into the session | Calendar date, known in advance |
| `sponsor_party_R` | Sponsored by a Republican | Sponsor information is public before the vote |

### Bill Type Indicators

Each bill gets a one-hot encoded **prefix** — HB (House Bill), SB (Senate Bill), HCR (House Concurrent Resolution), SCR (Senate Concurrent Resolution), and so on. Different bill types have different passage rates. Resolutions, for example, pass at much higher rates than substantive bills. The prefix is known before the vote and doesn't leak the outcome.

Similarly, **vote type** indicators (final passage, committee report, motion to override, etc.) are one-hot encoded. A veto override vote has a different base rate than a routine final passage vote.

### NMF Topic Features: What's the Bill About?

The most interesting features come from **topic modeling** on bill short titles. Tallgrass applies **Non-negative Matrix Factorization** (NMF) — a simpler cousin of the BERTopic system described in Chapter 3 — to extract six topics from the short titles of all bills in a chamber.

NMF works like a recipe decomposition. Imagine you have 400 dishes (bills) and you want to describe them in terms of 6 basic flavor profiles (topics). NMF takes the ingredient list for each dish (the words in each title) and finds the 6 combinations of ingredients that best explain all the dishes. Each dish gets a proportion for each flavor: "this bill is 60% Topic 1 (taxation/revenue), 20% Topic 3 (education), 20% other."

Concretely:

1. Build a **TF-IDF matrix** from bill short titles. Each title becomes a bag of words, weighted by how informative each word is (common words get downweighted).
2. Apply NMF to decompose this matrix into 6 topic components.
3. Each bill gets 6 features (topic_0 through topic_5) representing its proportion in each topic.

The TF-IDF parameters are deliberately conservative:

```
Vocabulary: up to 500 terms (unigrams + bigrams)
Min document frequency: 2 (term must appear in 2+ titles)
Max document frequency: 0.85 (terms in >85% of titles are filtered)
Stopwords: standard English list
```

Why NMF instead of the more sophisticated BERTopic (Chapter 3)? Two reasons. First, NMF is **deterministic** — same input, same output, no random seed sensitivity. Second, NMF operates on short titles (5-15 words each), which are too short for embedding-based methods to work well. BERTopic shines on full bill text; NMF is the right tool for titles.

**Why isn't this target leakage?** Bill titles exist before the vote happens. The words in "AN ACT concerning taxation; relating to income tax credits for child care" don't tell you whether the bill passed. They tell you what the bill is *about*, which is legitimately predictive — some topics pass more easily than others.

**Codebase:** `analysis/15_prediction/nlp_features.py` (`NMF_N_TOPICS = 6`, `TFIDF_MAX_FEATURES = 500`, `TFIDF_NGRAM_RANGE = (1, 2)`)

## Temporal Validation: Can Early Patterns Predict Late Bills?

Standard cross-validation randomly shuffles the data, which means training and test sets contain bills from throughout the session. This is fine for measuring explanatory power, but it doesn't answer the question a real forecaster would ask: **can patterns from the first part of the session predict what happens later?**

Tallgrass runs a **temporal split** in addition to standard cross-validation:

1. Sort all bills by vote date.
2. Train on the first 70% of the session chronologically.
3. Test on the last 30%.

This is a harder test. The legislative calendar isn't random — the types of bills that come to a vote change as the session progresses. Early in the session, routine bills and committee reports dominate (high pass rate). Late in the session, contentious bills that were deferred or amended surface (lower pass rate, higher partisanship). A model trained only on the calm early weeks must generalize to the storm of the final weeks.

If the temporal split AUC is close to the random-split AUC, the model is learning genuine patterns that generalize across the session. If there's a substantial drop, the model may be learning session-phase artifacts rather than fundamental bill characteristics.

**Codebase:** `analysis/15_prediction/prediction.py` (temporal split in `train_passage_models()`, 70/30 chronological)

## The Small-Sample Challenge

With only 250-500 bills per chamber, the passage model hits several statistical roadblocks that the vote model never encounters:

### Class Imbalance

Most bills pass. The passage rate is typically 75-85%, depending on the chamber and session. With 300 bills, that means only 45-75 failures. The Nay class is small enough that a standard train/test split might leave only 10-15 failures in the test set — too few for reliable metrics.

Tallgrass handles this with **adaptive k-fold cross-validation**:

```
n_splits = min(5, max(2, minority_class_count))
```

If the minority class has fewer than 5 members, the number of folds is reduced to ensure each fold contains at least one example of each class. If only one class exists (all bills pass), prediction is skipped entirely — there's nothing to learn.

### Safeguards

- Skip prediction if fewer than 20 observations total
- Skip if only one class exists
- Skip if the minority class has fewer than 2 members
- Report wide confidence intervals honestly

These guardrails mean that in some chambers (especially the Senate, with fewer bills), the passage model may not run at all. That's an honest outcome — better to report "insufficient data for reliable passage prediction" than to train on 15 bills and pretend the results are meaningful.

## Results: What Bill Features Predict Passage?

### SHAP for Bill Passage

The SHAP analysis for bill passage tells a different story than vote prediction. Without legislator features to dominate, the bill-level features reveal their relative importance:

**Bill discrimination (beta_mean) usually leads.** How partisan the bill is turns out to be the strongest predictor of passage. This makes intuitive sense: in a chamber where one party has a supermajority, a bill that cleanly divides liberals from conservatives will pass if it favors the majority party and fail if it doesn't. Beta captures this asymmetry.

**Vote type matters.** Veto overrides, which require two-thirds supermajority, have a substantially different base rate than routine final passages. The `is_veto_override` flag and vote type indicators capture this structural difference.

**Bill prefix captures institutional patterns.** Senate bills (SB) and House bills (HB) may have different passage rates in the originating vs. receiving chamber. Joint resolutions and concurrent resolutions have their own dynamics.

**Topic features contribute modestly.** The NMF topics typically rank 4th-8th in SHAP importance. Some topics (e.g., taxation, criminal justice) have lower passage rates than others (e.g., naming ceremonies, administrative adjustments). This effect is real but smaller than the structural features.

**Day of session is weakly predictive.** Bills voted on later in the session tend to be more contentious (lower passage rates), but this effect is modest and noisy.

### Stratified Accuracy by Bill Type

One of the most useful outputs: how well the model predicts passage *by bill prefix*. Typical findings:

| Bill Type | Passage Rate | Model Accuracy | Notes |
|-----------|-------------|----------------|-------|
| HB (House Bill) | ~80% | ~85% | Core legislation — moderate difficulty |
| SB (Senate Bill) | ~75% | ~80% | Often more contentious in House |
| HCR/SCR (Resolutions) | ~95% | ~95% | Nearly always pass — easy to predict |
| Veto overrides | ~40-60% | ~65% | Hardest to predict — outcome depends on specific coalitions |

Resolutions are the model's easiest targets and veto overrides are its hardest. This mirrors the conceptual distinction: resolutions are pro forma, while veto overrides depend on complex cross-party dynamics that bill-level features can't fully capture.

## Surprising Bills: Outcomes Nobody Expected

Just as Chapter 1 identified surprising votes, the passage model identifies **surprising bills** — legislation that the model confidently predicted would pass but failed, or confidently predicted would fail but passed.

A surprising failure might be a bill that looked routine (low discrimination, bipartisan sponsor, common bill type) but hit unexpected opposition. A surprising passage might be a highly partisan bill that shouldn't have attracted enough cross-party support but did.

These surprises often point to **political dynamics that the model can't see**: backroom negotiations, leadership arm-twisting, last-minute amendments that changed the bill's character, or single-issue coalitions that form around specific legislation.

**Codebase:** `analysis/15_prediction/prediction.py` (`find_surprising_bills()`)

## Comparing the Two Prediction Tasks

| Dimension | Vote Prediction (Ch 1) | Bill Passage (Ch 2) |
|-----------|----------------------|---------------------|
| **Unit of analysis** | Legislator × bill pair | Bill |
| **Sample size** | 30,000-60,000 | 250-500 |
| **Strongest feature** | xi_x_beta (IRT interaction) | beta_mean (bill discrimination) |
| **AUC range** | ~0.98 | ~0.70-0.85 |
| **Key caveat** | IRT features partially circular | Small sample, wide confidence intervals |
| **Temporal validation** | Not applicable (random split) | 70/30 chronological split |
| **NLP features** | Not included (redundant at 0.98 AUC) | Included (topic_0 through topic_5) |
| **Primary value** | Confirms IRT dominance | Identifies bill types and topics that defy prediction |

The passage model's lower AUC isn't a failure — it's an honest reflection of a harder problem with less information. Bill-level prediction without legislator features is like forecasting an election without polling individual voters. The model captures the structural regularities (partisan bills in a supermajority pass, veto overrides are uncertain), but the bill-specific dynamics that determine close outcomes are genuinely unpredictable from features alone.

---

## Key Takeaway

Bill passage prediction is a harder, more honest test than vote prediction. With 100 times fewer observations and no legislator features, the model relies on bill discrimination, vote type, bill prefix, and topic features — achieving AUC around 0.70-0.85. Temporal validation tests whether patterns from the first 70% of the session generalize to the last 30%. The model's limitations are instructive: veto overrides and close partisan bills are genuinely hard to predict without knowing the specific coalition dynamics at play.

---

*Terms introduced: bill passage prediction, temporal validation, adaptive k-fold, NMF (Non-negative Matrix Factorization), TF-IDF (Term Frequency-Inverse Document Frequency), bill prefix, stratified accuracy, class imbalance, surprising bills*

*Previous: [Predicting Votes: What XGBoost Learns](ch01-predicting-votes.md)*

*Next: [Topic Modeling: What Are the Bills About?](ch03-topic-modeling.md)*
