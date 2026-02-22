# Prediction Design Choices

**Script:** `analysis/prediction.py`
**Constants defined at:** `analysis/prediction.py:48-60`

## Assumptions

1. **Vote prediction is a binary classification task.** Each observation is a (legislator, roll call) pair with target Yea=1, Nay=0. Absent/not-voting observations are excluded — they match the IRT and EDA encoding where only substantive votes are modeled. Including absences as a third class would require a multinomial model and confound ideological signal with attendance patterns.

2. **Bill passage prediction is a separate binary classification task.** Each observation is a roll call with target `passed` from the rollcalls CSV. The unit of analysis shifts from (legislator, vote) to (vote), reducing the dataset from ~68K to ~500 rows. This small sample requires regularization and careful validation.

3. **Features from upstream phases carry forward.** IRT ideal points, party loyalty, PCA scores, and network centrality are treated as fixed features (not re-estimated). Any uncertainty in these features (e.g., xi_sd) is captured as an input feature, not propagated through the prediction model.

4. **Chambers are independent.** Models are trained separately for House and Senate. Cross-chamber prediction is not attempted (different legislators, different bill sets).

## Parameters & Constants

| Constant | Value | Justification | Code location |
|----------|-------|---------------|---------------|
| `RANDOM_SEED` | 42 | Reproducibility; consistent across all phases | `prediction.py:48` |
| `N_SPLITS` | 5 | Standard StratifiedKFold; 5 is standard for ~68K rows | `prediction.py:49` |
| `TEST_SIZE` | 0.20 | 80/20 holdout; sufficient test set (~13K observations) | `prediction.py:50` |
| `N_ESTIMATORS_XGB` | 200 | XGBoost trees; 200 balances bias-variance for this data size | `prediction.py:51` |
| `N_ESTIMATORS_RF` | 200 | Random Forest trees; matches XGBoost for fair comparison | `prediction.py:52` |
| `XGB_MAX_DEPTH` | 6 | Moderate depth; prevents overfitting on ~20 features | `prediction.py:53` |
| `XGB_LEARNING_RATE` | 0.1 | Standard; slower learning with 200 trees | `prediction.py:54` |
| `TOP_SHAP_FEATURES` | 15 | Features to show in SHAP plots; ~20 total features | `prediction.py:55` |
| `TOP_SURPRISING_N` | 20 | Most surprising votes/bills to report per chamber | `prediction.py:56` |
| `MINORITY_THRESHOLD` | 0.025 | Inherited from EDA; filter near-unanimous votes | `prediction.py:57` |
| `MIN_VOTES` | 20 | Inherited from EDA; minimum votes per legislator | `prediction.py:58` |

## Methodological Choices

### Target encoding: Yea/Nay only

**Decision:** Exclude absent/not-voting observations. Target is binary: Yea=1, Nay=0.

**Why:** Matches the IRT encoding (absences are missing, not a category). Including absences would require a multinomial model and conflate ideology with attendance. The 5 vote categories (Yea, Nay, Present and Passing, Absent and Not Voting, Not Voting) collapse to 2.

**Impact:** ~68K total vote observations → ~60K Yea/Nay observations after excluding absences.

### Feature selection: skip cluster labels and community labels

**Decision:** Do not include cluster assignments (k=2) or community assignments as features.

**Why:**
- Cluster labels at k=2 are identical to party (ARI=1.0). Adding them is redundant with the `party` feature.
- Community labels at default resolution also equal party (NMI=1.0, ARI=1.0).
- Within-party community labels are noise (modularity ≈ 0 for all caucuses).

**What to use instead:** `party` (binary), `loyalty_rate` (continuous), IRT ideal points, and centrality measures capture the same information without redundancy.

### Feature selection: include IRT uncertainty

**Decision:** Include `xi_sd` (posterior standard deviation of ideal point) as a feature.

**Why:** Legislators with wide HDI intervals (high xi_sd) are harder for IRT to place, which likely makes them harder to predict. xi_sd serves as a proxy for "how well the 1D model captures this legislator's behavior."

### Primary model: XGBoost

**Decision:** XGBoost is the primary model; logistic regression and random forest are comparisons.

**Why:** XGBoost handles mixed feature types, captures non-linear interactions (e.g., xi × beta), is SHAP-compatible via TreeExplainer, and consistently performs well on tabular data. Logistic regression provides a linear baseline. Random forest provides a non-boosted tree comparison.

**Alternatives considered:**
- Neural networks — rejected: tabular data with ~20 features doesn't benefit from deep learning
- LightGBM/CatBoost — similar to XGBoost; XGBoost chosen per method doc specification

### Interaction feature: xi × beta

**Decision:** Include `xi_x_beta = xi_mean × beta_mean` as an explicit interaction feature.

**Why:** In the IRT model, the probability of a Yea vote is a function of xi × beta (ideal point × discrimination). Making this interaction explicit helps the logistic regression baseline. XGBoost can discover it automatically, but the explicit feature improves interpretability.

### Feature exclusion: vote counts and margin (target leakage)

**Decision:** Exclude `yea_count`, `nay_count`, and `margin` from the vote-level feature set. Exclude `margin` and `alpha_mean` from the bill-level feature set.

**Why — vote-level:** `yea_count` and `nay_count` are the roll call tallies that *include* the target legislator's vote. If legislator X voted Yea, their vote is part of `yea_count`. The model learns "when yea_count is high, predict Yea" — circular reasoning. `margin = |yea - nay| / total` encodes how lopsided the vote was, which is unavailable at prediction time. These features would inflate accuracy by giving the model post-hoc information about the vote outcome.

**Why — bill-level:** `margin` is derived from the vote counts that determine passage (bills pass when yea > nay). Including it is equivalent to telling the model the answer. `alpha_mean` (IRT difficulty) is a near-proxy for passage — it measures how easy a bill was to pass, estimated from the same votes that determine the `passed` label. Failed bills: alpha = +0.3 to +1.1; passed bills: alpha = -1.7 to -2.4. Nearly separable by alpha alone.

**What remains:** For vote-level: IRT bill parameters (alpha, beta), vote type, day of session. For bill-level: beta (discrimination — measures partisanship, not outcome), vote type, bill prefix, day of session, is_veto_override.

### Override handling: binary indicator

**Decision:** Use `is_veto_override` as a binary indicator, not Kappa-based network features.

**Why:** The override subnetwork is too sparse for Kappa computation (most pairs have undefined Kappa due to near-unanimous within-party voting). A simple binary flag captures the key information: override votes have different dynamics (2/3 threshold, higher party loyalty).

### Validation: 5-fold CV + 20% holdout

**Decision:** Use 5-fold StratifiedKFold cross-validation for model selection and hyperparameter comparison, plus a held-out 20% test set for final evaluation.

**Why:** CV provides uncertainty estimates (mean ± std across folds). The holdout provides an unbiased final evaluation. Stratification ensures each fold has the same Yea/Nay ratio.

### Bill passage: temporal split

**Decision:** In addition to random CV, evaluate bill passage models with a temporal split: train on first 70% of roll calls chronologically, test on last 30%.

**Why:** In practice, passage prediction would be used for future bills. Temporal validation tests whether early-session patterns generalize to late-session votes. With ~500 rows, the test set is small (~150 rows) but sufficient for directional assessment.

### Base rate: AUC-ROC mandatory

**Decision:** Report AUC-ROC alongside accuracy for all models. Include a majority-class baseline.

**Why:** The 82% Yea base rate means a "predict all Yea" classifier achieves 82% accuracy. Accuracy alone is misleading. AUC-ROC measures discrimination independent of the base rate. A party-only baseline (predict by party median) provides a more informative comparison than majority-class.

### NLP topic features: NMF on bill short_title

**Decision:** Add NMF topic features (K=6) from TF-IDF on `short_title` to the bill passage model only.

**Why NMF, not LDA or embeddings:**
- NMF is deterministic (given a fixed random seed), faster than LDA, and produces non-negative topic weights that are directly interpretable as feature values.
- LDA requires Bayesian inference (Gibbs sampling or variational), adds stochasticity, and gains little on ~500 short documents.
- Sentence embeddings (e.g., sentence-transformers) would add a large dependency (~400MB model) for marginal benefit on 5-15 word titles.
- NMF is already available via scikit-learn — zero new dependencies.

**Why K=6:** The Senate has ~194 roll calls. With 6 topics, each topic covers ~30 documents on average — enough for stable NMF estimation. Higher K risks sparse topics; lower K loses granularity. K=6 balances coherence with discrimination.

**Why `short_title`, not `bill_title`:** `short_title` comes from the KLISS API and is never empty. `bill_title` can be empty for some roll calls and contains formulaic preamble ("AN ACT concerning...") that adds noise without signal.

**Data leakage assessment:** Bill titles are public information known before any votes are cast. The text is fixed at bill introduction — it does not change based on vote outcomes. Using it as a feature is analogous to using the bill prefix (HB, SB) or vote type, which are already in the model. The topic model is fit on all documents (not train-only) because the text is pre-vote information, same as IRT ideal points which are also pre-computed on all data.

**Scope:** Topic features are added to the bill passage model only, not the individual vote model (which already achieves AUC=0.98 — marginal topic signal would not justify the added complexity).

**Constants:**

| Constant | Value | Justification |
|----------|-------|---------------|
| `NMF_N_TOPICS` | 6 | Senate N=194 ceiling; balances granularity vs coherence |
| `TFIDF_MAX_DF` | 0.85 | Exclude terms appearing in >85% of docs (stop-word-like) |
| `TFIDF_MIN_DF` | 2 | Require term in at least 2 docs (avoid hapax legomena) |
| `TFIDF_MAX_FEATURES` | 500 | Cap vocabulary; short titles have limited vocabulary |
| `TFIDF_NGRAM_RANGE` | (1, 2) | Unigrams + bigrams capture phrases like "income tax" |
| `NMF_TOP_WORDS` | 5 | Words per topic for display/labeling |

## Downstream Implications

### For interpretation
- SHAP values identify which features drive individual predictions, enabling case-study analysis of legislators like Tyson and Schreiber
- Per-legislator accuracy reveals who the model struggles with — these legislators may warrant 2D or issue-specific models
- Surprising votes identify specific (legislator, bill) pairs where behavior deviates from the model's expectations — these are the most analytically interesting observations

### For future work
- NLP topic features (NMF on `short_title`) are now included in the bill passage model; if AUC remains low, consider temporal features (session progression) or cross-session stacking
- If per-legislator accuracy is bimodal, consider legislator-specific models or mixture-of-experts
- Bill passage prediction on ~500 rows is inherently limited; more sessions would improve this
