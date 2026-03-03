# Prediction Deep Dive: Implementation Review & Literature Comparison

**Date:** 2026-02-25
**Scope:** `analysis/15_prediction/prediction.py` (~1,988 lines), `analysis/15_prediction/prediction_report.py` (~960 lines), `analysis/15_prediction/nlp_features.py` (~238 lines), `tests/test_prediction.py` (~943 lines)
**Status:** All recommendations implemented (2026-02-25). ADR-0031. 958 tests passing.

This document steps back from the implementation to ask: are we doing prediction right? It surveys the political science and machine learning literature, evaluates open-source alternatives, audits our code for correctness and completeness, and identifies concrete improvements.

---

## 1. Literature Grounding

### 1.1 Foundational References

Legislative vote prediction spans political science, NLP, and machine learning. Our implementation draws on — or independently converges with — established practices from all three fields:

| Reference | What It Prescribes | Our Compliance |
|-----------|-------------------|----------------|
| [Poole & Rosenthal 1985 (NOMINATE)](https://legacy.voteview.com/pdf/nominate.pdf) | Spatial voting model: vote = f(ideal point × bill position) | `xi_x_beta` interaction is exactly this |
| [Clinton, Jackman & Rivers 2004](https://www.cs.princeton.edu/courses/archive/fall09/cos597A/papers/ClintonJackmanRivers2004.pdf) | Bayesian IRT as the modern ideal-point method; absences excluded | IRT ideal points as features; Yea/Nay only |
| [Gerrish & Blei 2011 (ICML)](https://www.cs.columbia.edu/~blei/papers/GerrishBlei2011.pdf) | Topic-factorized ideal points; bill text predicts votes | NMF topic features on `short_title` for passage model |
| [Kornilova & Eidelman 2018 (ACL)](https://aclanthology.org/P18-2081/) | Models fail across sessions without metadata; party augmentation essential | Party + IRT features; temporal split validation |
| [Karimi et al. 2019 (ASONAM)](https://ieeexplore.ieee.org/document/9073191/) | Multi-factor: party, co-voting network, bill text | Party + network centrality + NLP topics |
| [Immer et al. 2020 (KDD)](https://dl.acm.org/doi/10.1145/3394486.3403277) | Sub-matrix factorization for real-time prediction | Different paradigm (collaborative filtering); we use supervised learning |

**Verdict:** Our feature set (IRT ideal points + party + network centrality + PCA + NLP topics) covers more upstream signals than any single published system we found. The `xi_x_beta` interaction makes the IRT spatial model explicit, which is textbook.

### 1.2 How Our Approach Compares to Published Systems

| System | Features | Models | CV Strategy | Best AUC/Accuracy | Our Comparison |
|--------|----------|--------|-------------|-------------------|----------------|
| **Kraft et al. 2016** | Bill text embeddings + legislator vectors | Neural embeddings | Temporal (train on Congress N, test N+1) | ~83% accuracy | We use IRT features instead of learned embeddings; richer feature set |
| **Karimi et al. 2019** | Party, Doc2Vec, co-voting network | RF, NN | 10-fold CV | F1=0.76 | We include IRT (they don't); they use Doc2Vec (we use NMF) |
| **Immer et al. 2020** | Matrix factorization | GLM on sub-matrices | Temporal (real-time) | 93% accuracy (Swiss) | Different task (real-time partial-result prediction) |
| **VPF 2025** | Multiple feature sets, 5 countries | DT, RF, MLP, SVM, NB, XGBoost | 5-fold CV | 85% precision | Closest to our approach; we add IRT/SHAP |
| **Our system** | IRT + party + PCA + network + NLP | LogReg, XGBoost, RF | 5-fold stratified + 20% holdout + temporal | AUC ~0.98 (vote), ~0.60-0.80 (passage) | Strongest feature set for single-session prediction |

### 1.3 The Open-Source Landscape

**Dedicated Python packages for legislative vote prediction are rare.** Most published work releases code as one-off research repos rather than maintained libraries:

- **predikon** ([PyPI](https://pypi.org/project/predikon/)) — Sub-matrix factorization for real-time vote prediction. Installable library from Immer et al. 2020. Different paradigm (collaborative filtering on partial results), not directly applicable to our pre-vote prediction task.
- **py-irt** ([PyPI](https://pypi.org/project/py-irt/)) — Bayesian IRT in Python (Pyro-based). We use PyMC instead, but py-irt is worth noting as an alternative.
- **girth** ([PyPI](https://pypi.org/project/girth/)) — Classical IRT estimation (MML, joint MLE). Pure Python, lighter weight.
- **Voteview data** ([voteview.com/data](https://voteview.com/data)) — The canonical source for US congressional roll-call data + DW-NOMINATE scores. Not a prediction library, but the standard dataset.

**No maintained Python library exists for the specific task of "supervised vote prediction from a feature matrix."** This means our approach — building features from upstream phases and using scikit-learn/XGBoost — is the standard practice, not a gap. The field simply uses general-purpose ML tools.

---

## 2. Code Audit

### 2.1 What We're Doing Right

**Feature engineering is careful and well-documented:**
- Target leakage prevention is explicit and correct: `yea_count`, `nay_count`, `margin` excluded from vote features; `margin`, `alpha_mean` excluded from bill features. These exclusions are documented both in code comments and in `design/prediction.md`.
- The `xi_x_beta` interaction makes the IRT spatial model explicit, helping the logistic regression baseline capture non-linear signal that tree models find automatically.
- Feature sources span all upstream phases (IRT, PCA, clustering, network) — broader than any single published system.

**Validation strategy is solid:**
- 5-fold stratified CV + 20% holdout is standard and appropriate.
- Two baselines (majority-class and party-only) are reported — this is exactly what the literature recommends.
- Temporal split for bill passage tests generalization to future votes.
- AUC-ROC is the primary metric, correctly acknowledging the 82% Yea base rate.

**Interpretability is strong:**
- SHAP with plain-English feature names serves the nontechnical audience well.
- Per-legislator accuracy identifies who the model struggles with, with data-driven explanations.
- Surprising votes/bills pinpoint the most analytically interesting observations.

**Edge case handling is thorough:**
- Bill passage gracefully skips when: <20 rows, single class, minority class <2 members.
- `n_splits` is dynamically adjusted for small minority classes.
- NLP features gracefully degrade with small/empty corpora.
- Empty surprising-bills result returns a properly typed DataFrame.

### 2.2 Potential Issues

#### Issue 1: IRT Ideal Points as Features — The Circularity Question

**This is the single most important methodological concern in our pipeline.**

IRT ideal points (`xi_mean`, `xi_sd`) and bill parameters (`alpha_mean`, `beta_mean`) are estimated from the *same vote matrix* we're predicting. When we use them as features to predict votes in that matrix, there is a circularity:

1. IRT sees legislator X voted Yea on bill Y → estimates X's ideal point
2. Prediction model uses X's ideal point to predict X's vote on bill Y → high accuracy

This is **not strict target leakage** (we're not feeding in the target variable directly), but it is **information leakage through the feature pipeline**. The model's high AUC (~0.98) may partly reflect this circularity rather than genuine predictive power.

**Context:** This circularity is *ubiquitous* in the literature. DW-NOMINATE scores are routinely used in analyses of the same votes they were estimated from. The Gerrish & Blei papers explicitly noted that pure ideal-point models cannot predict on held-out bills because they lack bill parameters. Our design doc acknowledges this ("Features from upstream phases carry forward... not re-estimated").

**How serious is it?** For our purposes (understanding what drives Kansas legislative voting), it is not a fatal flaw. The SHAP values are genuinely informative — they tell us *which upstream signals* matter most. The per-legislator accuracy genuinely identifies who the 1D model fails on. The circularity means we should not claim "we can predict 98% of votes from pre-vote information alone" — we should say "a model combining IRT ideal points, party, and network features explains 98% of the variation in Kansas voting."

**Recommendation:** Add a clearly worded caveat to the report and design doc:

> *Note: IRT features are estimated from the same vote matrix used for prediction. The high AUC reflects explanatory power (how well these features describe voting patterns) rather than true out-of-sample predictive accuracy. For genuine prediction of future votes, ideal points would need to be estimated from prior sessions only.*

No code change needed — this is a documentation/interpretation issue.

#### Issue 2: Surprising Votes Computed on Full Dataset, Not Holdout

`find_surprising_votes()` and `compute_per_legislator_accuracy()` are called on the **full** `vote_features` DataFrame, not just the holdout test set. This means the model evaluates its own training data — inflating accuracy for in-sample observations and potentially missing genuinely surprising out-of-sample votes.

**Impact:** The per-legislator accuracy numbers are optimistically biased. Surprising votes may include some that were surprising only because the model memorized them.

**Recommendation:** Evaluate surprising votes and per-legislator accuracy on the holdout test set only. This requires filtering `vote_features` to test-set indices before calling these functions:

```python
# In main(), after training:
test_indices = vote_result["test_indices"]  # Need to capture these
test_features = vote_features[test_indices]
leg_accuracy = compute_per_legislator_accuracy(xgb_model, test_features, ...)
surprising = find_surprising_votes(xgb_model, test_features, ...)
```

This would reduce the sample size for per-legislator analysis (from ~60K to ~12K), but the results would be methodologically sound.

**Severity:** Moderate. The current approach is common in exploratory analysis but would not pass peer review for a prediction claim.

#### Issue 3: `find_surprising_votes()` Returns Empty DataFrame Without Schema

When there are zero wrong predictions, `find_surprising_votes()` returns `pl.DataFrame()` — an empty DataFrame with **no schema**. By contrast, `find_surprising_bills()` correctly returns a typed empty DataFrame. This inconsistency could cause downstream issues (e.g., in the report builder when checking for expected columns).

**Location:** `prediction.py:954`

**Recommendation:** Match the pattern used in `find_surprising_bills()`:

```python
if wrong_mask.sum() == 0:
    return pl.DataFrame(
        schema={
            "legislator_slug": pl.Utf8,
            "full_name": pl.Utf8,
            "party": pl.Utf8,
            "vote_id": pl.Utf8,
            "bill_number": pl.Utf8,
            "motion": pl.Utf8,
            "actual": pl.Int64,
            "predicted": pl.Int64,
            "y_prob": pl.Float64,
            "confidence_error": pl.Float64,
        }
    )
```

**Severity:** Low (unlikely to trigger in practice — the model always makes some mistakes).

#### Issue 4: Missing Brier Score and Log-Loss

The literature unanimously recommends **proper scoring rules** (Brier score, log-loss) for evaluating probabilistic predictions. We compute accuracy, AUC, precision, recall, and F1 — but not Brier score or log-loss.

- **Brier score** (`sklearn.metrics.brier_score_loss`) measures calibration + discrimination jointly. It directly answers "how good are the model's probability estimates?"
- **Log-loss** (`sklearn.metrics.log_loss`) penalizes confident wrong predictions more severely.

Our calibration curve plot implicitly checks calibration visually, but reporting the numeric Brier score would be more rigorous and more comparable across sessions.

**Recommendation:** Add Brier score to the CV results, holdout evaluation, and passage models. Two lines per model evaluation:

```python
from sklearn.metrics import brier_score_loss, log_loss
fold_results[f"{name}_brier"] = brier_score_loss(y_val, y_prob)
fold_results[f"{name}_logloss"] = log_loss(y_val, y_prob)
```

**Severity:** Low (improvement to completeness, not a correctness issue).

#### Issue 5: No `class_weight` or Imbalance Handling

With an 82% Yea base rate, the minority class (Nay) is underrepresented. None of the three models use `class_weight="balanced"` or any other imbalance correction.

**Context:** For the vote prediction model, this is probably fine — the 82/18 split is not extreme, and the model achieves AUC=0.98 regardless. For the bill passage model, sessions with near-100% passage rates (e.g., 85th House = 100%) are already handled by the graceful-skip logic.

**Recommendation:** No change for vote prediction. For bill passage, consider adding `scale_pos_weight` to XGBoost and `class_weight="balanced"` to LogReg/RF when the minority class is <20% of the data. This would improve recall for failed-bill detection.

**Severity:** Low.

### 2.3 Phase Number Discrepancy

The module docstring says "Phase 7" but the directory is `15_prediction`. The internal phase numbering (Phase 1-9 within the module) is a separate concept from the pipeline phase number, but the docstring header should match the pipeline. This also appears in the design doc path references.

**Recommendation:** Update the docstring header from "Phase 7" to "Phase 8" to match the directory structure.

**Severity:** Cosmetic.

---

## 3. Refactoring Opportunities

### 3.1 Repeated Model Training in CV

In `train_vote_models()`, `_make_models()` is called inside the CV loop (line 583) — creating 3 fresh model instances per fold — and then again outside the loop (line 598) for final training. This is correct behavior (sklearn models are mutable, so re-creating them is the right thing to do), but the pattern is repeated identically in `train_passage_models()` and `_temporal_split_eval()`.

**Recommendation:** This is fine as-is. Extracting a shared `_cross_validate()` helper would save ~30 lines but would add indirection for no functional benefit. The current duplication is clear and each function handles its own edge cases differently.

### 3.2 One-Hot Encoding Pattern

The one-hot encoding pattern for `vote_type` and `bill_prefix` is manual (lines 371-376, 457-462, 464-469):

```python
for vt in vote_types:
    safe_name = f"vt_{vt.lower().replace(' ', '_').replace('/', '_')}"
    df = df.with_columns(
        pl.when(pl.col("vote_type") == vt).then(1).otherwise(0).alias(safe_name)
    )
```

This works correctly but is repeated three times. Polars has `df.to_dummies()` but it doesn't offer the `vt_`/`pfx_` prefix control we need.

**Recommendation:** A small helper would reduce repetition:

```python
def _one_hot(df: pl.DataFrame, col: str, prefix: str) -> pl.DataFrame:
    values = df.select(col).drop_nulls().unique().sort(col)[col].to_list()
    for val in values:
        safe = f"{prefix}{val.lower().replace(' ', '_').replace('/', '_')}"
        df = df.with_columns(
            pl.when(pl.col(col) == val).then(1).otherwise(0).alias(safe)
        )
    return df
```

**Severity:** Low. Nice-to-have, not urgent.

### 3.3 `_compute_day_of_session()` Date Format Fallback Chain

The try/except chain for date parsing (lines 286-294) handles both `MM/DD/YYYY` (real data) and `YYYY-MM-DD` (tests) formats. This works but is fragile — any third format will silently fall through to `cast(pl.Date)` which may produce wrong results.

**Recommendation:** Use `pl.col.str.to_date(strict=False)` with explicit format detection:

```python
def _compute_day_of_session(vote_date_col: pl.Series) -> pl.Series:
    sample = vote_date_col.drop_nulls().head(1).to_list()
    if sample and "/" in sample[0]:
        dates = vote_date_col.str.to_date("%m/%d/%Y")
    else:
        dates = vote_date_col.str.to_date("%Y-%m-%d")
    return (dates - dates.min()).dt.total_days().alias("day_of_session")
```

**Severity:** Low. Current code works for all known inputs.

### 3.4 Dead Code Check

No dead code found. All public functions are called from either `main()` or tests. All imports are used. The `plot_surprising_votes()` function is called in `main()` but not tested directly — it's an I/O function (writes PNG), so this is expected.

---

## 4. Test Coverage Analysis

### 4.1 Current Coverage (38 tests)

| Area | Tests | Coverage |
|------|-------|----------|
| Vote feature engineering | 4 | Shape, NaN, target encoding, expected columns |
| Bill feature engineering | 3 | Shape, target binary, one-hot columns |
| Vote model training | 3 | Model count, CV keys, AUC > random |
| Passage model training | 2 | Model count, small-sample handling |
| Surprising bills | 2 | Empty schema, schema with data |
| Per-legislator accuracy | 2 | Row count, accuracy range |
| Surprising votes | 2 | Top-N, expected columns |
| SHAP | 2 | Shape, feature names |
| NLP topic integration | 4 | Topic columns, backward compat, feature count, training |
| Hardest legislators | 14 | Count, sorting, explanations, edge cases |

### 4.2 Test Gaps

#### Gap 1: No test for `evaluate_holdout()`

The holdout evaluation function is called in `main()` but never tested directly. It's a simple function, but a test would catch regressions.

```python
class TestEvaluateHoldout:
    def test_returns_all_models(self, trained_models):
        results = evaluate_holdout(
            trained_models["models"],
            trained_models["X_test"],
            trained_models["y_test"],
        )
        assert len(results) == 3
        assert all("accuracy" in r for r in results)
        assert all("auc" in r for r in results)

    def test_metrics_in_range(self, trained_models):
        results = evaluate_holdout(
            trained_models["models"],
            trained_models["X_test"],
            trained_models["y_test"],
        )
        for r in results:
            assert 0 <= r["accuracy"] <= 1
            assert 0 <= r["auc"] <= 1
```

#### Gap 2: No test for temporal split

`_temporal_split_eval()` is tested only indirectly through `train_passage_models()`. A direct test would verify the chronological ordering and the 70/30 split.

```python
class TestTemporalSplit:
    def test_returns_results(self, synthetic_rollcalls, synthetic_bill_params):
        features = build_bill_features(synthetic_rollcalls, synthetic_bill_params, "House")
        feature_cols = _get_feature_cols(features, "passed_binary", ["vote_id", "bill_number", "vote_date"])
        results = _temporal_split_eval(features, feature_cols, "House")
        if results:  # May be empty if too few rows
            assert len(results) == 3  # 3 models
            assert all("train_size" in r for r in results)
            assert all("test_size" in r for r in results)
```

#### Gap 3: No test for `_compute_day_of_session()` edge cases

The function handles two date formats but there's no test for the fallback behavior or for the minimum-date anchoring.

```python
class TestComputeDayOfSession:
    def test_yyyy_mm_dd_format(self):
        dates = pl.Series(["2025-01-10", "2025-01-11", "2025-01-15"])
        result = _compute_day_of_session(dates)
        assert result.to_list() == [0, 1, 5]

    def test_mm_dd_yyyy_format(self):
        dates = pl.Series(["01/10/2025", "01/11/2025", "01/15/2025"])
        result = _compute_day_of_session(dates)
        assert result.to_list() == [0, 1, 5]
```

#### Gap 4: No test for baselines in `train_vote_models()`

The party-only and majority-class baselines are computed but never tested. A test should verify that the party-only baseline is reasonable (>50% for a polarized legislature).

```python
class TestBaselines:
    def test_majority_baseline_above_50(self, trained_models):
        baselines = trained_models["baselines"]
        assert baselines["majority_class_acc"] >= 0.5

    def test_party_baseline_above_majority(self, trained_models):
        baselines = trained_models["baselines"]
        assert baselines["party_only_acc"] >= baselines["majority_class_acc"]

    def test_xgb_beats_party_baseline(self, trained_models):
        baselines = trained_models["baselines"]
        cv_df = pl.DataFrame(trained_models["cv_results"])
        xgb_acc = float(cv_df["XGBoost_accuracy"].mean())
        assert xgb_acc > baselines["party_only_acc"]
```

#### Gap 5: No test for `find_surprising_votes()` with zero wrong predictions

The function returns an untyped empty DataFrame when no predictions are wrong. This is the Issue 3 bug — a test would have caught it.

#### Gap 6: No test for the report builder

`prediction_report.py` has no dedicated test file. The 25 section builders are tested only by running the full pipeline. At minimum, `_add_data_summary()` and `_add_model_comparison_table()` should have unit tests to catch regressions in table formatting.

### 4.3 Tests Added (2026-02-25)

| Class | Tests | What It Covers |
|-------|-------|---------------|
| `TestEvaluateHoldout` | 3 | All models returned, metrics in range, proper scoring rules |
| `TestComputeDayOfSession` | 3 | ISO format, US format, single-date edge case |
| `TestBaselines` | 3 | Majority ≥50%, party > majority, party AUC > random |
| `TestProperScoringRules` | 3 | Brier + log-loss present in CV, valid ranges |
| `TestTestIndices` | 2 | test_indices present, proportion matches TEST_SIZE |
| `TestSurprisingVotesEmptySchema` | 1 | Perfect model returns typed empty DataFrame |
| `TestTemporalSplit` | 1 | Returns results for all 3 models with expected keys |

Total added: 16 new tests. Current: 54 prediction tests (958 total).

---

## 5. Comparison with Open-Source Alternatives

### 5.1 Scikit-Learn Patterns

Our use of scikit-learn is idiomatic and standard:
- `StratifiedKFold` for CV — correct.
- `train_test_split` with `stratify=y` — correct.
- `XGBClassifier` with `eval_metric="logloss"` — correct (avoids the deprecated default).
- `LogisticRegression(solver="lbfgs", max_iter=5000)` — correct for convergence on this data.
- `RandomForestClassifier(n_jobs=-1)` — fine, though on Apple Silicon with 6P+6E cores, this may oversubscribe (see ADR-0022).

**One gap:** We don't use `sklearn.pipeline.Pipeline`. Wrapping the feature engineering + model into a pipeline would make the code more portable but adds complexity for no functional benefit in our single-pipeline architecture.

### 5.2 What Others Do That We Don't

| Technique | Who Uses It | Applicable to Us? | Recommendation |
|-----------|------------|-------------------|----------------|
| **Bill text embeddings** (Doc2Vec, BERT) | Kraft 2016, Karimi 2019 | Partially. We use NMF on short_title, which is lighter. Full bill text is not available from KS Legislature API. | No change — short_title is all we have |
| **Legislator embeddings** (learned vectors) | Kraft 2016 | Interesting but requires more data. IRT ideal points serve the same purpose with principled Bayesian estimation. | No change — IRT is superior for our data size |
| **Leave-one-bill-out CV** | Gerrish & Blei 2012 | Yes — would test generalization to new legislation. Currently we only do random and temporal splits. | Consider adding as a sensitivity check |
| **Leave-one-legislator-out CV** | Various | Yes — would test cold-start prediction for new members. | Consider adding for cross-session analysis |
| **LightGBM** | VPF 2025 | Marginal improvement over XGBoost on this data size. Handles categoricals natively. | Not worth the added dependency |
| **Calibration (Platt/Isotonic)** | Standard ML practice | We check calibration visually but don't explicitly calibrate. XGBoost probabilities are already well-calibrated for this task. | Consider `CalibratedClassifierCV` if calibration curve shows systematic bias |
| **SMOTE / class weighting** | Standard for imbalanced data | 82/18 split is mild. Tree models handle it well. | No change for vote model; consider for passage model |

### 5.3 The Kaggle/UCI Congressional Votes Dataset

The [UCI Congressional Voting Records dataset](https://www.kaggle.com/datasets/devvret/congressional-voting-records) (1984 House, 16 binary features, 435 members) is the most common benchmark for legislative vote classification. It's a party-prediction task, not a vote-prediction task — classifiers predict Republican/Democrat from vote patterns.

This is not comparable to our setup. We predict individual votes from legislator + bill features, not party from votes. The UCI dataset is useful for benchmarking classifiers but not for evaluating our methodology.

---

## 6. Summary of Findings

### What's Working Well

1. **Feature engineering** is the strongest aspect — broader upstream integration than any published system.
2. **Target leakage prevention** is thorough and well-documented.
3. **Validation strategy** (5-fold CV + holdout + temporal + two baselines) meets literature standards.
4. **Interpretability** (SHAP, per-legislator, surprising votes) serves the nontechnical audience.
5. **Edge case handling** for bill passage (small samples, single class, degenerate splits) is robust.
6. **NLP topic features** are well-designed — NMF over LDA is the right call for determinism and speed on short titles.
7. **Test coverage** for hardest-legislator detection is particularly thorough (14 tests).

### What Needs Attention

| Finding | Severity | Status |
|---------|----------|--------|
| IRT circularity caveat missing from report | **High** (interpretation) | **Implemented.** Caveat added to report + design doc. |
| Surprising votes/per-legislator accuracy on full data | **Moderate** | **Implemented.** Now evaluated on 20% holdout only via `test_indices`. |
| `find_surprising_votes()` empty DataFrame has no schema | **Low** (bug) | **Fixed.** Returns typed schema matching `find_surprising_bills()`. |
| Missing Brier score / log-loss | **Low** | **Implemented.** Added to vote CV, holdout, passage CV, temporal split, and report tables. |
| Phase number "7" vs directory "08" | **Cosmetic** | **Fixed.** Docstring updated. |
| ~12 test gaps | **Medium** | **Implemented.** 16 new tests added (54 total, up from 38). |

### Expected Output Changes After Implementation

These changes will cause the following differences when re-running the prediction phase:

1. **Per-legislator accuracy values will decrease slightly.** Previously computed on all ~60K observations (including training data); now computed on ~12K holdout observations only. This removes in-sample inflation.
2. **Surprising votes may differ.** Previously drawn from all observations; now drawn from holdout only. Some previously-flagged surprising votes were surprising because the model memorized them, not because they were genuinely unexpected.
3. **Report tables gain two columns.** BRIER and LOGLOSS appear in vote CV, holdout, passage CV, and temporal split tables. Lower is better for both.
4. **Report includes a new caveat paragraph** in the Vote Prediction Interpretation section explaining the IRT circularity.
5. **No changes to bill passage results.** Passage models were already small enough that holdout-only was the only option.

### What We Explicitly Chose Not To Do (and Why)

- **Bill text embeddings (BERT/Doc2Vec):** Full bill text is not available from the KS Legislature API. Short titles are 5-15 words — NMF captures the signal without a 400MB model dependency.
- **2D ideal points / multi-dimensional scaling:** The 1D IRT model is the canonical baseline. Per-legislator accuracy already identifies where 1D fails. A 2D model is a possible Phase 15 extension.
- **Neural networks:** With ~20 tabular features and 60K rows, deep learning offers no advantage over gradient boosted trees. XGBoost is the right tool.
- **LightGBM/CatBoost:** Marginal improvement over XGBoost for this data size. Not worth the added dependency.
- **Leave-one-bill-out / leave-one-legislator-out CV:** Valuable for different questions (generalization to new legislation / new members) but our primary question is "what drives voting in this session?" Standard CV answers that question.

---

## 7. Literature Bibliography

### Spatial Voting & Ideal Points

- Poole, K. T., & Rosenthal, H. (1985). "A Spatial Model for Legislative Roll Call Analysis." *American Journal of Political Science*, 29(2), 357-384.
- Clinton, J., Jackman, S., & Rivers, D. (2004). "The Statistical Analysis of Roll Call Data." *American Political Science Review*, 98(2), 355-370.
- Imai, K., Lo, J., & Olmsted, J. (2016). "Fast Estimation of Ideal Points with Massive Data." *American Political Science Review*, 110(4), 631-656.

### Vote Prediction with ML/NLP

- Gerrish, S., & Blei, D. M. (2011). "Predicting Legislative Roll Calls from Text." *Proceedings of the 28th International Conference on Machine Learning (ICML)*.
- Gerrish, S., & Blei, D. M. (2012). "How They Vote: Issue-Adjusted Models of Legislative Behavior." *Advances in Neural Information Processing Systems (NeurIPS)*.
- Kraft, P., Jain, H., & Rush, A. M. (2016). "An Embedding Model for Predicting Roll-Call Votes." *Proceedings of EMNLP*.
- Kornilova, A., & Eidelman, V. (2018). "Party Matters: Enhancing Legislative Embeddings with Author Attributes for Vote Prediction." *Proceedings of ACL*.
- Karimi, F., et al. (2019). "Multi-Factor Congressional Vote Prediction Model." *Proceedings of ASONAM*.

### Real-Time & Matrix Factorization

- Immer, A., et al. (2020). "Sub-Matrix Factorization for Real-Time Vote Prediction." *Proceedings of KDD*.

### Frameworks & Surveys

- VPF (2025). "Framework of Voting Prediction of Parliament Members." *arXiv:2505.12535*.
- Political Actor Agent (2024). "LLM-based agent simulation for roll-call prediction." *arXiv:2412.07144*.

### Evaluation & Methodology

- Desposato, S. (2005). "Correcting for Small Group Inflation of Indexes of Agreement." *British Journal of Political Science*.
- Carrubba, C. J., et al. (2006). "Off the Record: Unrecorded Legislative Votes, Selection Bias and Roll-Call Vote Analysis." *British Journal of Political Science*.
