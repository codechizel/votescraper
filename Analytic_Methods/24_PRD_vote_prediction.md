# Vote Prediction (Individual-Level)

**Category:** Predictive Modeling
**Prerequisites:** `01_DATA_vote_matrix_construction`, ideally after PCA and index-based measures
**Complexity:** Medium
**Related:** `25_PRD_bill_passage_prediction`

## What It Measures

Vote prediction models answer: given what we know about a legislator (party, ideology, district) and a bill (subject, sponsor, previous votes), can we predict how the legislator will vote? This is fundamentally a binary classification problem — Yea or Nay — with ~68K training examples.

Unlike ideal point models (which estimate latent ideology), predictive models are evaluated purely on out-of-sample accuracy. They reveal which features are most predictive of legislative behavior and can forecast votes on new bills.

## Questions It Answers

- How accurately can we predict individual votes from observable features?
- Which features matter most: party, estimated ideology, bill type, sponsor party, or vote timing?
- Are some legislators more predictable than others?
- Are some bills more predictable than others?
- What is the accuracy ceiling — how much of legislative voting is systematic vs. idiosyncratic?

## Input Features

### Legislator Features
| Feature | Source | Type |
|---------|--------|------|
| `party` | legislators.csv | Binary (R/D) |
| `chamber` | legislators.csv | Binary (House/Senate) |
| `district` | legislators.csv | Numeric |
| `ideal_point_pc1` | PCA analysis | Continuous |
| `party_unity_score` | Unity analysis | Continuous |
| `participation_rate` | Participation analysis | Continuous |

### Bill/Roll Call Features
| Feature | Source | Type |
|---------|--------|------|
| `vote_type` | rollcalls.csv | Categorical (8 types) |
| `sponsor_party` | rollcalls.csv (parsed) | Binary (R/D) |
| `bill_chamber` | rollcalls.csv | Binary (House/Senate) |
| `yea_count_prev` | rollcalls.csv (lag) | Numeric |
| `is_bipartisan_bill` | Computed | Binary |
| `days_into_session` | Computed from date | Numeric |

### Interaction Features
| Feature | Description |
|---------|-------------|
| `party_x_sponsor_party` | Does legislator's party match bill sponsor's party? |
| `ideal_x_bill_difficulty` | Interaction of ideology with bill contestedness |

## Python Implementation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

def prepare_prediction_data(
    votes: pd.DataFrame,
    rollcalls: pd.DataFrame,
    legislators: pd.DataFrame,
    pca_scores: pd.DataFrame | None = None,
    unity_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target for vote prediction."""
    # Keep only Yea/Nay votes
    df = votes[votes["vote"].isin(["Yea", "Nay"])].copy()
    df["target"] = (df["vote"] == "Yea").astype(int)

    # Merge legislator features
    leg_feats = legislators.set_index("slug")[["party", "chamber", "district"]]
    df = df.merge(leg_feats, left_on="legislator_slug", right_index=True, how="left")

    # Merge roll call features
    rc_feats = rollcalls.set_index("vote_id")[
        ["vote_type", "sponsor", "yea_count", "nay_count", "total_votes", "vote_datetime"]
    ]
    df = df.merge(rc_feats, left_on="vote_id", right_index=True, how="left")

    # Engineered features
    df["sponsor_party"] = df["sponsor"].apply(
        lambda x: "Republican" if isinstance(x, str) and "Senator" in x else "Unknown"
    )  # Simplified — would need proper parsing
    df["margin"] = abs(df["yea_count"] - df["nay_count"]) / df["total_votes"]
    df["is_contested"] = (df["margin"] < 0.5).astype(int)
    df["same_party_sponsor"] = (df["party"] == df["sponsor_party"]).astype(int)

    # Temporal features
    df["vote_dt"] = pd.to_datetime(df["vote_datetime"])
    df["day_of_session"] = (df["vote_dt"] - df["vote_dt"].min()).dt.days
    df["month"] = df["vote_dt"].dt.month

    # Add PCA scores if available
    if pca_scores is not None:
        df = df.merge(
            pca_scores[["PC1", "PC2"]],
            left_on="legislator_slug",
            right_index=True,
            how="left",
        )

    # Add unity scores if available
    if unity_scores is not None:
        df = df.merge(
            unity_scores[["legislator_slug", "unity_score"]],
            on="legislator_slug",
            how="left",
        )

    # Encode categoricals
    le_party = LabelEncoder()
    df["party_encoded"] = le_party.fit_transform(df["party"])

    le_chamber = LabelEncoder()
    df["chamber_encoded"] = le_chamber.fit_transform(df["chamber"])

    le_votetype = LabelEncoder()
    df["vote_type_encoded"] = le_votetype.fit_transform(df["vote_type"].fillna("Unknown"))

    # Select features
    feature_cols = [
        "party_encoded", "chamber_encoded", "vote_type_encoded",
        "margin", "is_contested", "day_of_session", "month",
    ]
    if "PC1" in df.columns:
        feature_cols.extend(["PC1", "PC2"])
    if "unity_score" in df.columns:
        feature_cols.append("unity_score")

    X = df[feature_cols].fillna(0)
    y = df["target"]

    return X, y


def run_vote_prediction(
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Run multiple classifiers and compare performance."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    # Baseline: always predict majority class
    baseline_accuracy = max(y.mean(), 1 - y.mean())
    print(f"Baseline (majority class): {baseline_accuracy:.3f}")

    # Party-only baseline
    party_accuracy = (X["party_encoded"] == y).mean()  # Rough proxy
    print(f"Party-only baseline: ~{party_accuracy:.3f}")

    results = {"baseline": baseline_accuracy}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        results[name] = {
            "accuracy_mean": scores.mean(),
            "accuracy_std": scores.std(),
            "auc_mean": auc_scores.mean(),
            "auc_std": auc_scores.std(),
        }
        print(f"{name}: Acc={scores.mean():.3f}(+/-{scores.std():.3f}), "
              f"AUC={auc_scores.mean():.3f}(+/-{auc_scores.std():.3f})")

    return results
```

### Feature Importance (SHAP)

```python
def analyze_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    save_path: str | None = None,
):
    """Use SHAP to understand which features drive predictions."""
    import shap

    # Train a gradient boosting model
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")


def per_legislator_accuracy(
    X: pd.DataFrame,
    y: pd.Series,
    legislator_slugs: pd.Series,
) -> pd.DataFrame:
    """Compute prediction accuracy per legislator."""
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)

    df = pd.DataFrame({
        "legislator_slug": legislator_slugs,
        "actual": y,
        "predicted": predictions,
        "correct": (predictions == y),
    })

    accuracy_by_leg = df.groupby("legislator_slug")["correct"].mean().sort_values()
    return accuracy_by_leg
```

## Interpretation Guide

### Accuracy Benchmarks

| Model | Expected Accuracy | Interpretation |
|-------|-------------------|----------------|
| Majority class baseline | ~82% | Always predict Yea (since 82% of votes are Yea) |
| Party-only | ~85-88% | Using just party affiliation |
| Logistic Regression (all features) | ~88-92% | Linear model with ideology and bill features |
| Gradient Boosting (all features) | ~90-94% | Captures non-linear interactions |
| Theoretical ceiling | ~96-98% | Some votes are genuinely unpredictable |

### Feature Importance Ranking (Typical)

1. **Ideal point / PC1** (if available) — strongest single predictor
2. **Party** — second strongest (highly correlated with ideal point)
3. **Vote margin / is_contested** — contested votes are harder to predict
4. **Vote type** — some vote types (e.g., Conference Committee) have different dynamics
5. **Day of session** — votes at session end may differ from early-session votes

### SHAP Value Interpretation

- **Positive SHAP value**: Feature pushes prediction toward Yea
- **Negative SHAP value**: Feature pushes prediction toward Nay
- **Spread of SHAP values**: How much the feature matters across different predictions

### Per-Legislator Predictability

- **Highly predictable legislators** (accuracy > 95%): Pure party voters. Their ideology completely determines their vote.
- **Unpredictable legislators** (accuracy < 85%): Mavericks, moderates, or issue-specific voters. Their behavior depends on the specific bill.

## Kansas-Specific Considerations

- **The high Yea baseline (82%) makes accuracy metrics misleading.** Use AUC-ROC instead of raw accuracy for model comparison. AUC is independent of class balance.
- **Filter to contested votes for more meaningful predictions.** Predicting Yea on near-unanimous votes is trivial and inflates accuracy. The interesting predictions are on contested votes where parties divide.
- **Party and ideal point will dominate.** In a partisan legislature, these features will explain >85% of votes. The value of this analysis is in quantifying the remaining ~15%.
- **Leave-one-vote-out validation** is more realistic than random CV, since it simulates predicting a new bill.

## Feasibility Assessment

- **Data size**: ~68K individual votes = excellent for ML
- **Compute time**: Seconds for logistic regression, minutes for gradient boosting with SHAP
- **Libraries**: `scikit-learn`, `xgboost` or `lightgbm`, `shap`
- **Difficulty**: Medium

## Key References

- Kornilova, Anastassia, and Daniel Eidelman. "Party Matters: Enhancing Legislative Embeddings with Author Information for Vote Prediction." 2019. https://arxiv.org/abs/1911.06467
- Gerrish, Sean, and David M. Blei. "Predicting Legislative Roll Calls from Text." *ICML*, 2011.
- Stanford CS229 Project: "Predicting Bill Passage in Congress." https://cs229.stanford.edu/proj2015/242_report.pdf
