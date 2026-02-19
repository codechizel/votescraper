# Bill Passage Prediction

**Category:** Predictive Modeling
**Prerequisites:** `01_DATA_vote_matrix_construction`, `02_EDA_descriptive_statistics`
**Complexity:** Medium
**Related:** `24_PRD_vote_prediction`

## What It Measures

Bill passage prediction operates at the roll-call level rather than the individual-vote level. Given a bill's characteristics (sponsor, type, subject, timing), can we predict whether it will pass? This is a harder problem than individual vote prediction because the unit of analysis is the roll call (~500-800 observations) rather than individual votes (~68K).

## Questions It Answers

- Which bill characteristics predict passage?
- Does sponsor party matter? Number of sponsors?
- Are certain vote types (Final Action vs. Conference Committee) more predictable?
- Are early-session bills more or less likely to pass than late-session bills?
- Can we identify bills that are "surprising" — predicted to pass but failed, or vice versa?

## Input Features

| Feature | Source | Type | Notes |
|---------|--------|------|-------|
| `sponsor_party` | rollcalls.csv | Categorical | Majority vs. minority party sponsor |
| `n_sponsors` | rollcalls.csv | Numeric | Number of sponsors/cosponsors |
| `vote_type` | rollcalls.csv | Categorical | Final Action, Conference Committee, etc. |
| `chamber` | rollcalls.csv | Binary | House or Senate |
| `day_of_session` | Computed | Numeric | When in the session timeline |
| `bill_prefix` | Extracted from bill_number | Categorical | SB, HB, SCR, etc. |
| `is_bipartisan_sponsor` | Computed | Binary | Are sponsors from both parties? |
| `prior_votes_on_bill` | Computed | Numeric | How many prior roll calls on this bill? |

### Target Variable

`passed` — Boolean from rollcalls.csv. True for passed/adopted/prevailed/concurred; False for failed/rejected/sustained; None for procedural (excluded).

## Python Implementation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def prepare_bill_passage_data(rollcalls: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix for bill passage prediction."""
    # Exclude votes with no clear pass/fail outcome
    df = rollcalls.dropna(subset=["passed"]).copy()
    df["target"] = df["passed"].astype(int)

    # Parse features
    df["bill_prefix"] = df["bill_number"].str.extract(r"^([A-Z]+)")
    df["n_sponsors"] = df["sponsor"].apply(
        lambda x: len(str(x).split(";")) if pd.notna(x) else 0
    )

    # Temporal features
    df["vote_dt"] = pd.to_datetime(df["vote_datetime"])
    df["day_of_session"] = (df["vote_dt"] - df["vote_dt"].min()).dt.days
    df["month"] = df["vote_dt"].dt.month
    df["hour"] = df["vote_dt"].dt.hour

    # Vote margin features
    df["total_present"] = df["yea_count"] + df["nay_count"]
    df["absence_rate"] = 1 - df["total_present"] / df["total_votes"]

    # Prior votes on same bill
    df = df.sort_values("vote_datetime")
    df["prior_votes_on_bill"] = df.groupby("bill_number").cumcount()

    # Encode categoricals
    from sklearn.preprocessing import LabelEncoder
    for col in ["vote_type", "chamber", "bill_prefix"]:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].fillna("Unknown"))

    feature_cols = [
        "vote_type_encoded", "chamber_encoded", "bill_prefix_encoded",
        "n_sponsors", "day_of_session", "month", "hour",
        "absence_rate", "prior_votes_on_bill",
    ]

    X = df[feature_cols].fillna(0)
    y = df["target"]

    return X, y


def run_bill_passage_prediction(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train and evaluate bill passage models."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline = max(y.mean(), 1 - y.mean())
    print(f"Baseline: {baseline:.3f} (always predict {'pass' if y.mean() > 0.5 else 'fail'})")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        results[name] = {"accuracy": acc.mean(), "auc": auc.mean()}
        print(f"{name}: Acc={acc.mean():.3f}, AUC={auc.mean():.3f}")

    return results


def find_surprising_outcomes(
    X: pd.DataFrame,
    y: pd.Series,
    rollcalls: pd.DataFrame,
) -> pd.DataFrame:
    """Find bills where the model's prediction was most wrong."""
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]

    df = rollcalls.dropna(subset=["passed"]).copy()
    df["predicted_pass_prob"] = probs
    df["surprise"] = abs(df["passed"].astype(float) - probs)

    return df.nlargest(20, "surprise")[
        ["bill_number", "bill_title", "motion", "passed", "predicted_pass_prob", "surprise"]
    ]
```

## Interpretation Guide

- **High baseline accuracy** (>80%) is expected — most bills that make it to a floor vote pass. The challenge is predicting the failures.
- **Surprising outcomes** (bills predicted to pass that failed, or vice versa) are the most analytically interesting. Investigate what made these bills unusual.
- **Vote type is likely the strongest predictor**: Conference Committee votes almost always pass (they're pre-negotiated); Veto Override votes are the most uncertain.
- **Limited by sample size**: ~500-800 roll calls is a modest dataset for ML. Use regularization and simple models.

## Kansas-Specific Considerations

- **Most bills pass** in a supermajority legislature. The interesting predictions are on the ~10-15% that fail.
- **Veto overrides (34 votes) are a natural subanalysis** — what predicts successful vs. failed overrides?
- **Bill text features** would substantially improve predictions but require additional scraping beyond the current pipeline.
- **Consider a temporal train/test split** (train on first half of session, predict second half) for more realistic evaluation.

## Feasibility Assessment

- **Data size**: ~500-800 roll calls = small for ML but workable with simple models
- **Compute time**: Seconds
- **Libraries**: `scikit-learn`, `xgboost`
- **Difficulty**: Medium (feature engineering is the main challenge)

## Key References

- Yano, Tae, et al. "Textual Predictors of Bill Survival in Congressional Committees." *NAACL*, 2012.
- Gerrish, Sean, and David M. Blei. "Predicting Legislative Roll Calls from Text." *ICML*, 2011.
- Nay, John J. "Predicting and Understanding Law-Making with Word Vectors and an Ensemble Model." *PLOS ONE* 12(5), 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5425031/
