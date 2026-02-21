"""
Kansas Legislature — Vote & Bill Passage Prediction (Phase 7)

Trains XGBoost, Logistic Regression, and Random Forest models to predict
individual legislator votes (Yea/Nay) and bill passage outcomes.  Features
come from all upstream phases: IRT ideal points, PCA scores, party loyalty,
and network centrality.

Usage:
  uv run python analysis/prediction.py [--session 2025-26] [--skip-bill-passage]

Outputs (in results/<session>/prediction/<date>/):
  - data/:   Parquet files (features, per-legislator accuracy, surprising votes)
  - plots/:  PNG visualizations (ROC, SHAP, confusion matrices, calibration)
  - filtering_manifest.json, run_info.json, run_log.txt
  - prediction_report.html
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

try:
    from analysis.run_context import RunContext
except ModuleNotFoundError:
    from run_context import RunContext

try:
    from analysis.prediction_report import build_prediction_report
except ModuleNotFoundError:
    from prediction_report import build_prediction_report  # type: ignore[no-redef]


# ── Constants ────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.20
N_ESTIMATORS_XGB = 200
N_ESTIMATORS_RF = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
TOP_SHAP_FEATURES = 15
TOP_SURPRISING_N = 20
MINORITY_THRESHOLD = 0.025
MIN_VOTES = 20
PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC"}

PREDICTION_PRIMER = """\
# Vote & Bill Passage Prediction

## Purpose

Predicts individual legislator votes (Yea/Nay) and bill passage outcomes using
features from all upstream analysis phases. The primary goals are:
1. Quantify how predictable Kansas legislative voting is
2. Identify which features drive predictions (via SHAP)
3. Find the most surprising votes — where model expectations diverge from reality

## Method

### Vote Prediction (Individual Level)
Three models trained per chamber on ~30K-60K (legislator, roll call) pairs:
- **Logistic Regression** — linear baseline (C=1.0)
- **XGBoost** — primary model (200 trees, depth 6)
- **Random Forest** — non-boosted tree comparison (200 trees)

Features combine legislator attributes (IRT ideal point, party loyalty, PCA scores,
network centrality) with bill attributes (IRT difficulty/discrimination, vote type)
and a legislator-bill interaction (xi × beta). Vote counts (yea/nay/margin) are
intentionally excluded — they encode the outcome and would constitute target leakage.

### Bill Passage Prediction (Bill Level)
Same three models on ~250-500 roll calls per chamber. Smaller sample requires
careful regularization. Temporal split validation supplements random CV.

### Validation
- 5-fold stratified cross-validation (mean ± std for all metrics)
- 20% holdout test set for final evaluation
- Baselines: majority-class (always Yea) and party-only (predict by party median)

## Inputs

| File | Source | Contents |
|------|--------|----------|
| `ks_*_votes.csv` | Scraper | Individual vote records |
| `ks_*_rollcalls.csv` | Scraper | Roll call metadata and results |
| `ks_*_legislators.csv` | Scraper | Legislator demographics |
| `ideal_points_{chamber}.parquet` | IRT | Legislator ideal points |
| `bill_params_{chamber}.parquet` | IRT | Bill difficulty/discrimination |
| `party_loyalty_{chamber}.parquet` | Clustering | Party loyalty rates |
| `centrality_{chamber}.parquet` | Network | Centrality measures |
| `pc_scores_{chamber}.parquet` | PCA | Principal component scores |

## Outputs

| File | Contents |
|------|----------|
| `vote_features_{chamber}.parquet` | Engineered feature matrix (vote-level) |
| `bill_features_{chamber}.parquet` | Engineered feature matrix (bill-level) |
| `cv_results_{chamber}.parquet` | Cross-validation metrics per model per fold |
| `holdout_results_{chamber}.parquet` | Holdout test set metrics |
| `per_legislator_accuracy_{chamber}.parquet` | Per-legislator prediction accuracy |
| `surprising_votes_{chamber}.parquet` | Most surprising individual votes |
| `surprising_bills_{chamber}.parquet` | Most surprising bill passage outcomes |
| `passage_cv_results_{chamber}.parquet` | Bill passage CV metrics |
| `temporal_split_{chamber}.parquet` | Bill passage temporal split metrics |
| `prediction_report.html` | Self-contained HTML report |

## Interpretation Guide

- **AUC-ROC > 0.90** indicates strong discrimination; expected given IRT features
- **SHAP beeswarm** shows how each feature pushes predictions toward Yea/Nay
- **Per-legislator accuracy** identifies who the model struggles with (Tyson, Schreiber)
- **Surprising votes** are the most analytically interesting: high-confidence wrong predictions
- **Bill passage AUC** will be lower than vote AUC due to small sample (~500 rows)

## Caveats

- 82% Yea base rate inflates accuracy; always compare against baselines
- SHAP values explain model behavior, not causal mechanisms
- Bill passage prediction on ~500 rows has wide confidence intervals
- Temporal split test set (~150 rows) is small; interpret directionally
"""


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Vote Prediction")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--irt-dir", default=None, help="Override IRT results directory")
    parser.add_argument("--clustering-dir", default=None, help="Override clustering results dir")
    parser.add_argument("--network-dir", default=None, help="Override network results directory")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument(
        "--skip-bill-passage",
        action="store_true",
        help="Skip bill passage prediction (vote prediction only)",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    width = 80
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Phase 1: Load Data ──────────────────────────────────────────────────────


def load_vote_data(data_dir: Path, session_slug: str) -> pl.DataFrame:
    """Load individual votes CSV."""
    return pl.read_csv(data_dir / f"ks_{session_slug}_votes.csv")


def load_rollcall_data(data_dir: Path, session_slug: str) -> pl.DataFrame:
    """Load rollcalls CSV."""
    return pl.read_csv(data_dir / f"ks_{session_slug}_rollcalls.csv")


def load_legislator_data(data_dir: Path, session_slug: str) -> pl.DataFrame:
    """Load legislators CSV."""
    return pl.read_csv(data_dir / f"ks_{session_slug}_legislators.csv")


def load_ideal_points(irt_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load IRT ideal points for both chambers."""
    house = pl.read_parquet(irt_dir / "data" / "ideal_points_house.parquet")
    senate = pl.read_parquet(irt_dir / "data" / "ideal_points_senate.parquet")
    return house, senate


def load_bill_params(irt_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load IRT bill parameters for both chambers."""
    house = pl.read_parquet(irt_dir / "data" / "bill_params_house.parquet")
    senate = pl.read_parquet(irt_dir / "data" / "bill_params_senate.parquet")
    return house, senate


def load_party_loyalty(clustering_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load party loyalty rates from clustering phase."""
    house = pl.read_parquet(clustering_dir / "data" / "party_loyalty_house.parquet")
    senate = pl.read_parquet(clustering_dir / "data" / "party_loyalty_senate.parquet")
    return house, senate


def load_centrality(network_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load centrality measures from network phase."""
    house = pl.read_parquet(network_dir / "data" / "centrality_house.parquet")
    senate = pl.read_parquet(network_dir / "data" / "centrality_senate.parquet")
    return house, senate


def load_pc_scores(pca_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load PCA scores from PCA phase."""
    house = pl.read_parquet(pca_dir / "data" / "pc_scores_house.parquet")
    senate = pl.read_parquet(pca_dir / "data" / "pc_scores_senate.parquet")
    return house, senate


# ── Phase 2: Feature Engineering — Vote-Level ───────────────────────────────


def _compute_day_of_session(vote_date_col: pl.Series) -> pl.Series:
    """Convert vote_date strings to ordinal day-of-session (1 = first vote day)."""
    # Try both date formats: MM/DD/YYYY (real data) and YYYY-MM-DD (tests)
    try:
        dates = vote_date_col.str.to_date("%m/%d/%Y")
    except Exception:
        try:
            dates = vote_date_col.str.to_date("%Y-%m-%d")
        except Exception:
            # If already a date type, use directly
            dates = vote_date_col.cast(pl.Date)
    min_date = dates.min()
    return (dates - min_date).dt.total_days().alias("day_of_session")


def build_vote_features(
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
    ideal_points: pl.DataFrame,
    bill_params: pl.DataFrame,
    party_loyalty: pl.DataFrame,
    centrality: pl.DataFrame,
    pc_scores: pl.DataFrame,
    chamber: str,
) -> pl.DataFrame:
    """Build the vote-level feature matrix for one chamber.

    Returns a DataFrame with features + target column 'vote_binary' (1=Yea, 0=Nay).
    Rows with absent/not-voting are excluded.
    """
    # Filter to chamber
    chamber_votes = votes.filter(pl.col("chamber") == chamber)

    # Keep only Yea/Nay
    chamber_votes = chamber_votes.filter(pl.col("vote").is_in(["Yea", "Nay"]))

    # Create binary target
    chamber_votes = chamber_votes.with_columns(
        pl.when(pl.col("vote") == "Yea").then(1).otherwise(0).alias("vote_binary")
    )

    # Join legislator features: party
    leg_features = legislators.select(
        pl.col("slug").alias("legislator_slug"),
        pl.when(pl.col("party") == "Republican").then(1).otherwise(0).alias("party_binary"),
    )
    chamber_votes = chamber_votes.join(leg_features, on="legislator_slug", how="left")

    # Join IRT ideal points
    ip_features = ideal_points.select("legislator_slug", "xi_mean", "xi_sd")
    chamber_votes = chamber_votes.join(ip_features, on="legislator_slug", how="left")

    # Join party loyalty
    loyalty_features = party_loyalty.select("legislator_slug", "loyalty_rate")
    chamber_votes = chamber_votes.join(loyalty_features, on="legislator_slug", how="left")

    # Join PCA scores (PC1, PC2)
    pca_features = pc_scores.select("legislator_slug", "PC1", "PC2")
    chamber_votes = chamber_votes.join(pca_features, on="legislator_slug", how="left")

    # Join centrality measures
    cent_features = centrality.select("legislator_slug", "betweenness", "eigenvector", "pagerank")
    chamber_votes = chamber_votes.join(cent_features, on="legislator_slug", how="left")

    # Join bill/roll call features (vote counts excluded — they encode the outcome)
    rc_features = rollcalls.select(
        "vote_id",
        "vote_type",
        "vote_date",
    )
    chamber_votes = chamber_votes.join(rc_features, on="vote_id", how="left", suffix="_rc")

    # Join IRT bill params
    bp_features = bill_params.select("vote_id", "alpha_mean", "beta_mean", "is_veto_override")
    chamber_votes = chamber_votes.join(bp_features, on="vote_id", how="left")

    # Compute derived features
    # Use the vote_date from the right (rollcalls) join if present
    date_col = "vote_date_rc" if "vote_date_rc" in chamber_votes.columns else "vote_date"
    chamber_votes = chamber_votes.with_columns(
        # Day of session
        _compute_day_of_session(chamber_votes[date_col]),
        # Interaction: xi × beta
        (pl.col("xi_mean") * pl.col("beta_mean")).alias("xi_x_beta"),
    )

    # One-hot encode vote_type
    vote_types = chamber_votes.select("vote_type").unique().sort("vote_type")["vote_type"].to_list()
    for vt in vote_types:
        safe_name = f"vt_{vt.lower().replace(' ', '_').replace('/', '_')}"
        chamber_votes = chamber_votes.with_columns(
            pl.when(pl.col("vote_type") == vt).then(1).otherwise(0).alias(safe_name)
        )

    # Select final feature columns + metadata
    # NOTE: yea_count, nay_count, margin are intentionally excluded — they encode
    # the vote outcome (target leakage). The model must predict from pre-vote
    # features only: legislator attributes + bill structural parameters.
    feature_cols = [
        "party_binary",
        "xi_mean",
        "xi_sd",
        "loyalty_rate",
        "PC1",
        "PC2",
        "betweenness",
        "eigenvector",
        "pagerank",
        "alpha_mean",
        "beta_mean",
        "is_veto_override",
        "day_of_session",
        "xi_x_beta",
    ]
    # Add one-hot vote_type columns
    vt_cols = [c for c in chamber_votes.columns if c.startswith("vt_")]
    feature_cols.extend(vt_cols)

    metadata_cols = ["legislator_slug", "vote_id", "vote_binary"]

    # Keep only rows where all features are non-null
    keep_cols = [c for c in feature_cols + metadata_cols if c in chamber_votes.columns]
    result = chamber_votes.select(keep_cols).drop_nulls(subset=feature_cols)

    return result


# ── Phase 3: Feature Engineering — Bill-Level ────────────────────────────────


def build_bill_features(
    rollcalls: pl.DataFrame,
    bill_params: pl.DataFrame,
    chamber: str,
) -> pl.DataFrame:
    """Build the bill-level feature matrix for passage prediction.

    Returns a DataFrame with features + target column 'passed_binary'.
    """
    # Filter to chamber
    rc = rollcalls.filter(pl.col("chamber") == chamber)

    # Create binary target from 'passed' column
    # Handle both boolean (True/False), string ("true"/"false"), and numeric (1.0/0.0)
    passed_col = rc["passed"]
    if passed_col.dtype == pl.Boolean:
        rc = rc.with_columns(pl.col("passed").cast(pl.Int32).alias("passed_binary"))
    elif passed_col.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
        rc = rc.with_columns(pl.col("passed").cast(pl.Int32).alias("passed_binary"))
    else:
        rc = rc.with_columns(
            pl.when(pl.col("passed").cast(pl.Utf8).str.to_lowercase() == "true")
            .then(1)
            .otherwise(0)
            .alias("passed_binary")
        )

    # Join IRT bill params
    bp = bill_params.select("vote_id", "alpha_mean", "beta_mean", "is_veto_override")
    rc = rc.join(bp, on="vote_id", how="left")

    # Compute derived features
    # NOTE: margin excluded — it's derived from vote counts that determine passage
    # (target leakage). alpha_mean also excluded — IRT difficulty is a near-proxy
    # for the passage outcome it was estimated from.
    rc = rc.with_columns(
        # Day of session
        _compute_day_of_session(rc["vote_date"]),
        # Chamber binary (1=House, 0=Senate)
        pl.when(pl.col("chamber") == "House").then(1).otherwise(0).alias("chamber_binary"),
    )

    # Extract bill prefix (SB, HB, SCR, HCR, etc.)
    rc = rc.with_columns(pl.col("bill_number").str.extract(r"^([A-Z]+)", 1).alias("bill_prefix"))

    # One-hot encode vote_type
    vote_types = rc.select("vote_type").unique().sort("vote_type")["vote_type"].to_list()
    for vt in vote_types:
        safe_name = f"vt_{vt.lower().replace(' ', '_').replace('/', '_')}"
        rc = rc.with_columns(
            pl.when(pl.col("vote_type") == vt).then(1).otherwise(0).alias(safe_name)
        )

    # One-hot encode bill prefix
    prefixes = rc.select("bill_prefix").drop_nulls().unique().sort("bill_prefix")
    prefix_list = prefixes["bill_prefix"].to_list()
    for pfx in prefix_list:
        rc = rc.with_columns(
            pl.when(pl.col("bill_prefix") == pfx).then(1).otherwise(0).alias(f"pfx_{pfx.lower()}")
        )

    # Select feature columns
    # NOTE: alpha_mean and margin excluded — alpha (IRT difficulty) is a near-proxy
    # for passage outcome, and margin is derived from the vote counts that determine
    # passage. Both constitute target leakage. beta (discrimination) is retained
    # because it measures how partisan a bill is, not whether it passes.
    feature_cols = [
        "beta_mean",
        "is_veto_override",
        "day_of_session",
        "chamber_binary",
    ]
    vt_cols = [c for c in rc.columns if c.startswith("vt_")]
    pfx_cols = [c for c in rc.columns if c.startswith("pfx_")]
    feature_cols.extend(vt_cols)
    feature_cols.extend(pfx_cols)

    metadata_cols = ["vote_id", "bill_number", "passed_binary", "vote_date"]
    keep_cols = [c for c in feature_cols + metadata_cols if c in rc.columns]
    result = rc.select(keep_cols).drop_nulls(subset=feature_cols)

    return result


# ── Phase 4: Model Training — Vote Prediction ───────────────────────────────


def _get_feature_cols(df: pl.DataFrame, target: str, meta_cols: list[str]) -> list[str]:
    """Get feature column names (everything except target and metadata)."""
    exclude = {target} | set(meta_cols)
    return [c for c in df.columns if c not in exclude]


def _make_models() -> dict[str, object]:
    """Create the three model instances."""
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=5000, random_state=RANDOM_SEED, solver="lbfgs"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=N_ESTIMATORS_XGB,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            verbosity=0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS_RF,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def train_vote_models(
    features_df: pl.DataFrame,
    chamber: str,
) -> dict:
    """Train vote prediction models with cross-validation.

    Returns dict with:
      - 'models': {name: fitted_model}
      - 'cv_results': list of per-fold dicts
      - 'X_train', 'X_test', 'y_train', 'y_test': holdout arrays
      - 'feature_names': list of feature column names
      - 'baselines': majority and party-only baseline metrics
    """
    feature_cols = _get_feature_cols(features_df, "vote_binary", ["legislator_slug", "vote_id"])
    X = features_df.select(feature_cols).to_numpy().astype(np.float64)
    y = features_df["vote_binary"].to_numpy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Baselines
    majority_class = int(np.bincount(y_train).argmax())
    majority_preds = np.full_like(y_test, majority_class)
    majority_acc = accuracy_score(y_test, majority_preds)

    # Party-only baseline: use party_binary feature (first column by convention)
    party_idx = feature_cols.index("party_binary") if "party_binary" in feature_cols else 0
    party_only_preds = X_test[:, party_idx].astype(int)
    party_only_acc = accuracy_score(y_test, party_only_preds)
    party_only_auc = roc_auc_score(y_test, X_test[:, party_idx])

    baselines = {
        "majority_class_acc": float(majority_acc),
        "party_only_acc": float(party_only_acc),
        "party_only_auc": float(party_only_auc),
    }

    # Cross-validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    cv_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        fold_results = {"fold": fold_idx}

        for name, model in _make_models().items():
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            fold_results[f"{name}_accuracy"] = accuracy_score(y_val, y_pred)
            fold_results[f"{name}_auc"] = roc_auc_score(y_val, y_prob)
            fold_results[f"{name}_precision"] = precision_score(y_val, y_pred, zero_division=0)
            fold_results[f"{name}_recall"] = recall_score(y_val, y_pred, zero_division=0)
            fold_results[f"{name}_f1"] = f1_score(y_val, y_pred, zero_division=0)

        cv_results.append(fold_results)

    # Train final models on full training set
    final_models = {}
    for name, model in _make_models().items():
        model.fit(X_train, y_train)
        final_models[name] = model

    return {
        "models": final_models,
        "cv_results": cv_results,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_cols,
        "baselines": baselines,
    }


def evaluate_holdout(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[dict]:
    """Evaluate all models on the holdout test set."""
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "auc": roc_auc_score(y_test, y_prob),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
        )

    return results


# ── Phase 5: Model Training — Bill Passage ───────────────────────────────────


def train_passage_models(
    features_df: pl.DataFrame,
    chamber: str,
) -> dict:
    """Train bill passage prediction models.

    Returns dict with models, cv_results, holdout data, and temporal split results.
    """
    feature_cols = _get_feature_cols(
        features_df, "passed_binary", ["vote_id", "bill_number", "vote_date"]
    )
    X = features_df.select(feature_cols).to_numpy().astype(np.float64)
    y = features_df["passed_binary"].to_numpy()

    if len(y) < 20:
        print(f"  WARNING: Only {len(y)} observations for {chamber} passage — skipping")
        return {"skipped": True, "reason": f"Only {len(y)} observations"}

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Cross-validation
    n_splits = min(N_SPLITS, max(2, int(np.unique(y_train, return_counts=True)[1].min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    cv_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        fold_results = {"fold": fold_idx}

        # Skip fold if training set has only one class
        if len(np.unique(y_tr)) < 2:
            continue

        for name, model in _make_models().items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            fold_results[f"{name}_accuracy"] = accuracy_score(y_val, y_pred)
            try:
                fold_results[f"{name}_auc"] = roc_auc_score(y_val, y_prob)
            except ValueError:
                fold_results[f"{name}_auc"] = float("nan")
            fold_results[f"{name}_precision"] = precision_score(y_val, y_pred, zero_division=0)
            fold_results[f"{name}_recall"] = recall_score(y_val, y_pred, zero_division=0)
            fold_results[f"{name}_f1"] = f1_score(y_val, y_pred, zero_division=0)

        cv_results.append(fold_results)

    # Train final models
    final_models = {}
    for name, model in _make_models().items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        final_models[name] = model

    # Temporal split: train on first 70%, test on last 30%
    temporal_results = _temporal_split_eval(features_df, feature_cols, chamber)

    return {
        "models": final_models,
        "cv_results": cv_results,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_cols,
        "temporal_results": temporal_results,
    }


def _temporal_split_eval(
    features_df: pl.DataFrame,
    feature_cols: list[str],
    chamber: str,
) -> list[dict]:
    """Evaluate passage models with a temporal train/test split."""
    # Sort by vote_date
    sorted_df = features_df.sort("vote_date")
    n = sorted_df.height
    split_idx = int(n * 0.70)

    if split_idx < 10 or (n - split_idx) < 5:
        return []

    train_df = sorted_df.head(split_idx)
    test_df = sorted_df.tail(n - split_idx)

    X_train = train_df.select(feature_cols).to_numpy().astype(np.float64)
    y_train = train_df["passed_binary"].to_numpy()
    X_test = test_df.select(feature_cols).to_numpy().astype(np.float64)
    y_test = test_df["passed_binary"].to_numpy()

    results = []
    for name, model in _make_models().items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        row = {"model": name, "accuracy": accuracy_score(y_test, y_pred)}
        try:
            row["auc"] = roc_auc_score(y_test, y_prob)
        except ValueError:
            row["auc"] = float("nan")
        row["precision"] = precision_score(y_test, y_pred, zero_division=0)
        row["recall"] = recall_score(y_test, y_pred, zero_division=0)
        row["f1"] = f1_score(y_test, y_pred, zero_division=0)
        row["train_size"] = split_idx
        row["test_size"] = n - split_idx
        results.append(row)

    return results


# ── Phase 6: SHAP Analysis ──────────────────────────────────────────────────


def compute_shap_values(
    model: XGBClassifier,
    X: np.ndarray,
    feature_names: list[str],
) -> shap.Explanation:
    """Compute SHAP values for the XGBoost model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap_values.feature_names = feature_names
    return shap_values


# ── Phase 7: Per-Legislator Analysis ─────────────────────────────────────────


def compute_per_legislator_accuracy(
    model: object,
    features_df: pl.DataFrame,
    feature_cols: list[str],
    ideal_points: pl.DataFrame,
) -> pl.DataFrame:
    """Compute prediction accuracy per legislator.

    Returns DataFrame with: legislator_slug, full_name, party, xi_mean,
    n_votes, n_correct, accuracy, hardest_vote_ids.
    """
    X = features_df.select(feature_cols).to_numpy().astype(np.float64)
    y = features_df["vote_binary"].to_numpy()
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Add predictions back to dataframe
    pred_df = features_df.with_columns(
        pl.Series("y_pred", y_pred),
        pl.Series("y_prob", y_prob),
        pl.Series("correct", (y_pred == y).astype(int)),
        pl.Series("confidence_error", np.abs(y_prob - y.astype(float))),
    )

    # Group by legislator — basic accuracy
    leg_acc = pred_df.group_by("legislator_slug").agg(
        pl.col("correct").sum().alias("n_correct"),
        pl.col("correct").count().alias("n_votes"),
        pl.col("correct").mean().alias("accuracy"),
    )

    # Build hardest vote IDs separately (top 5 by confidence error per legislator)
    hardest = (
        pred_df.sort("confidence_error", descending=True)
        .group_by("legislator_slug")
        .head(5)
        .group_by("legislator_slug")
        .agg(pl.col("vote_id").alias("hardest_vote_ids"))
    )
    leg_acc = leg_acc.join(hardest, on="legislator_slug", how="left")

    # Join legislator metadata
    ip_meta = ideal_points.select("legislator_slug", "full_name", "party", "xi_mean")
    leg_acc = leg_acc.join(ip_meta, on="legislator_slug", how="left")

    return leg_acc.sort("accuracy")


# ── Phase 8: Surprising Votes / Bills ────────────────────────────────────────


def find_surprising_votes(
    model: object,
    features_df: pl.DataFrame,
    feature_cols: list[str],
    rollcalls: pl.DataFrame,
    ideal_points: pl.DataFrame,
    top_n: int = TOP_SURPRISING_N,
) -> pl.DataFrame:
    """Find votes where the model was most confident but wrong."""
    X = features_df.select(feature_cols).to_numpy().astype(np.float64)
    y = features_df["vote_binary"].to_numpy()
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # Confidence error = |predicted probability - actual|
    confidence_error = np.abs(y_prob - y.astype(float))

    # Only look at wrong predictions
    wrong_mask = y_pred != y
    if wrong_mask.sum() == 0:
        return pl.DataFrame()

    wrong_df = features_df.filter(pl.Series(wrong_mask)).with_columns(
        pl.Series("y_prob", y_prob[wrong_mask]),
        pl.Series("confidence_error", confidence_error[wrong_mask]),
        pl.Series("predicted", y_pred[wrong_mask]),
        pl.Series("actual", y[wrong_mask]),
    )

    # Take top N most confident wrong predictions
    surprising = wrong_df.sort("confidence_error", descending=True).head(top_n)

    # Enrich with metadata
    rc_meta = rollcalls.select("vote_id", "bill_number", "motion")
    surprising = surprising.join(rc_meta, on="vote_id", how="left")

    ip_meta = ideal_points.select("legislator_slug", "full_name", "party")
    surprising = surprising.join(ip_meta, on="legislator_slug", how="left")

    keep_cols = [
        "legislator_slug",
        "full_name",
        "party",
        "vote_id",
        "bill_number",
        "motion",
        "actual",
        "predicted",
        "y_prob",
        "confidence_error",
    ]
    return surprising.select([c for c in keep_cols if c in surprising.columns])


def find_surprising_bills(
    model: object,
    features_df: pl.DataFrame,
    feature_cols: list[str],
    rollcalls: pl.DataFrame,
    top_n: int = TOP_SURPRISING_N,
) -> pl.DataFrame:
    """Find bills where passage prediction was most wrong."""
    X = features_df.select(feature_cols).to_numpy().astype(np.float64)
    y = features_df["passed_binary"].to_numpy()
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    confidence_error = np.abs(y_prob - y.astype(float))

    result_df = features_df.select("vote_id", "bill_number", "passed_binary").with_columns(
        pl.Series("y_prob", y_prob),
        pl.Series("predicted", y_pred),
        pl.Series("confidence_error", confidence_error),
    )

    # Enrich with rollcall metadata
    rc_meta = rollcalls.select("vote_id", "motion", "vote_type", "yea_count", "nay_count")
    result_df = result_df.join(rc_meta, on="vote_id", how="left")

    return result_df.sort("confidence_error", descending=True).head(top_n)


# ── Phase 9: Plots ───────────────────────────────────────────────────────────


def plot_roc_curves(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    chamber: str,
    out_path: Path,
) -> None:
    """Plot ROC curves for all models on one figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"Logistic Regression": "#2196F3", "XGBoost": "#FF9800", "Random Forest": "#4CAF50"}

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=colors.get(name, "#999"),
            linewidth=2,
            label=f"{name} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{chamber} — Vote Prediction ROC Curves", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)

    save_fig(fig, out_path)


def plot_model_comparison(
    cv_results: list[dict],
    baselines: dict,
    chamber: str,
    out_path: Path,
) -> None:
    """Bar chart comparing model accuracy and AUC across CV folds."""
    model_names = ["Logistic Regression", "XGBoost", "Random Forest"]
    metrics = ["accuracy", "auc"]
    metric_labels = {"accuracy": "Accuracy", "auc": "AUC-ROC"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        for name in model_names:
            key = f"{name}_{metric}"
            vals = [fold[key] for fold in cv_results]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        x = np.arange(len(model_names))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            color=["#2196F3", "#FF9800", "#4CAF50"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add baseline lines
        if metric == "accuracy":
            maj = baselines["majority_class_acc"]
            pty = baselines["party_only_acc"]
            ax.axhline(
                maj,
                color="red",
                linestyle="--",
                linewidth=1,
                label=f"Majority baseline ({maj:.3f})",
            )
            ax.axhline(
                pty, color="purple", linestyle="--", linewidth=1, label=f"Party-only ({pty:.3f})"
            )
        elif metric == "auc":
            ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random (0.500)")
            ax.axhline(
                baselines["party_only_auc"],
                color="purple",
                linestyle="--",
                linewidth=1,
                label=f"Party-only ({baselines['party_only_auc']:.3f})",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel(metric_labels[metric], fontsize=12)
        ax.set_title(f"{metric_labels[metric]}", fontsize=13)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(f"{chamber} — Model Comparison (5-fold CV)", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_shap_summary(
    shap_values: shap.Explanation,
    chamber: str,
    out_path: Path,
) -> None:
    """SHAP beeswarm plot."""
    shap.plots.beeswarm(shap_values, max_display=TOP_SHAP_FEATURES, show=False, plot_size=(10, 8))
    fig = plt.gcf()
    fig.suptitle(f"{chamber} — SHAP Beeswarm (XGBoost)", fontsize=14, y=1.02)
    save_fig(fig, out_path)


def plot_shap_bar(
    shap_values: shap.Explanation,
    chamber: str,
    out_path: Path,
) -> None:
    """SHAP mean |SHAP| bar plot."""
    shap.plots.bar(shap_values, max_display=TOP_SHAP_FEATURES, show=False)
    fig = plt.gcf()
    fig.suptitle(
        f"{chamber} — Mean |SHAP| Feature Importance (XGBoost)",
        fontsize=14,
        y=1.02,
    )
    save_fig(fig, out_path)


def plot_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    chamber: str,
    out_path: Path,
) -> None:
    """XGBoost native feature importance (gain)."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[-TOP_SHAP_FEATURES:]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(sorted_idx)),
        importances[sorted_idx],
        color="#FF9800",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title(f"{chamber} — XGBoost Native Feature Importance", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, out_path)


def plot_per_legislator_accuracy(
    leg_accuracy: pl.DataFrame,
    chamber: str,
    out_path: Path,
) -> None:
    """Scatter: IRT ideal point (x) vs prediction accuracy (y), colored by party."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for party, color in PARTY_COLORS.items():
        subset = leg_accuracy.filter(pl.col("party") == party)
        if subset.height == 0:
            continue
        ax.scatter(
            subset["xi_mean"].to_numpy(),
            subset["accuracy"].to_numpy(),
            c=color,
            label=party,
            alpha=0.7,
            s=40,
            edgecolors="black",
            linewidths=0.3,
        )

    # Label bottom 5 by accuracy
    bottom5 = leg_accuracy.sort("accuracy").head(5)
    for row in bottom5.iter_rows(named=True):
        ax.annotate(
            row.get("full_name", row["legislator_slug"]),
            (row["xi_mean"], row["accuracy"]),
            fontsize=7,
            alpha=0.8,
            textcoords="offset points",
            xytext=(5, -5),
        )

    ax.set_xlabel("IRT Ideal Point (xi_mean)", fontsize=12)
    ax.set_ylabel("Prediction Accuracy", fontsize=12)
    ax.set_title(f"{chamber} — Per-Legislator Accuracy vs Ideology", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, out_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    chamber: str,
    model_name: str,
    out_path: Path,
) -> None:
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    labels = ["Nay (0)", "Yea (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"{chamber} — {model_name} Confusion Matrix", fontsize=13)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=14,
                fontweight="bold",
            )

    fig.tight_layout()
    save_fig(fig, out_path)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    chamber: str,
    model_name: str,
    out_path: Path,
) -> None:
    """Plot calibration (reliability) curve."""
    fig, ax = plt.subplots(figsize=(7, 6))

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    ax.plot(prob_pred, prob_true, "o-", color="#FF9800", linewidth=2, label=model_name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(f"{chamber} — {model_name} Calibration Curve", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, out_path)


def plot_passage_roc(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    chamber: str,
    out_path: Path,
) -> None:
    """Plot ROC curves for bill passage prediction."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"Logistic Regression": "#2196F3", "XGBoost": "#FF9800", "Random Forest": "#4CAF50"}

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                color=colors.get(name, "#999"),
                linewidth=2,
                label=f"{name} (AUC = {roc_auc:.3f})",
            )
        except ValueError:
            ax.plot(
                [], [], color=colors.get(name, "#999"), linewidth=2, label=f"{name} (AUC = N/A)"
            )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{chamber} — Bill Passage ROC Curves", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)

    save_fig(fig, out_path)


def plot_surprising_votes(
    surprising_df: pl.DataFrame,
    chamber: str,
    out_path: Path,
) -> None:
    """Strip plot showing surprising votes by confidence error."""
    if surprising_df.height == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    for row in surprising_df.iter_rows(named=True):
        name = row.get("full_name", row.get("legislator_slug", "?"))
        bill = row.get("bill_number", "?")
        labels.append(f"{name}\n{bill}")

    y_pos = range(min(surprising_df.height, TOP_SURPRISING_N))
    errors = surprising_df["confidence_error"].head(TOP_SURPRISING_N).to_numpy()

    colors = []
    for row in surprising_df.head(TOP_SURPRISING_N).iter_rows(named=True):
        party = row.get("party", "Unknown")
        colors.append(PARTY_COLORS.get(party, "#999999"))

    ax.barh(list(y_pos), errors, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels[: len(y_pos)], fontsize=8)
    ax.set_xlabel("Confidence Error (|P(Yea) - Actual|)", fontsize=12)
    ax.set_title(f"{chamber} — Most Surprising Votes (XGBoost)", fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # Add party legend
    handles = [Patch(facecolor=c, label=p) for p, c in PARTY_COLORS.items()]
    ax.legend(handles=handles, fontsize=9, loc="lower right")

    fig.tight_layout()
    save_fig(fig, out_path)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    session_slug = args.session.replace("-", "_")
    data_dir = Path(args.data_dir) if args.data_dir else Path(f"data/ks_{session_slug}")

    # Resolve upstream directories
    session_full = args.session.replace("_", "-")
    # Normalize to full year format: 2025-26 -> 2025-2026
    parts = session_full.split("-")
    if len(parts) == 2 and len(parts[1]) == 2:
        century = parts[0][:2]
        session_full = f"{parts[0]}-{century}{parts[1]}"

    results_root = Path("results") / session_full
    irt_dir = Path(args.irt_dir) if args.irt_dir else results_root / "irt" / "latest"
    clustering_dir = (
        Path(args.clustering_dir) if args.clustering_dir else results_root / "clustering" / "latest"
    )
    network_dir = (
        Path(args.network_dir) if args.network_dir else results_root / "network" / "latest"
    )
    pca_dir = Path(args.pca_dir) if args.pca_dir else results_root / "pca" / "latest"

    with RunContext(
        session=args.session,
        analysis_name="prediction",
        params=vars(args),
        primer=PREDICTION_PRIMER,
    ) as ctx:
        # ── Phase 1: Load Data ──────────────────────────────────────────
        print_header("PHASE 1: LOADING DATA")

        print("  Loading vote data...")
        votes = load_vote_data(data_dir, session_slug)
        rollcalls = load_rollcall_data(data_dir, session_slug)
        legislators = load_legislator_data(data_dir, session_slug)
        print(f"    Votes: {votes.height:,} rows")
        print(f"    Roll calls: {rollcalls.height:,} rows")
        print(f"    Legislators: {legislators.height:,} rows")

        print("  Loading upstream results...")
        ip_house, ip_senate = load_ideal_points(irt_dir)
        bp_house, bp_senate = load_bill_params(irt_dir)
        loyalty_house, loyalty_senate = load_party_loyalty(clustering_dir)
        cent_house, cent_senate = load_centrality(network_dir)
        pc_house, pc_senate = load_pc_scores(pca_dir)

        # Organize by chamber
        chamber_data = {
            "House": {
                "ideal_points": ip_house,
                "bill_params": bp_house,
                "party_loyalty": loyalty_house,
                "centrality": cent_house,
                "pc_scores": pc_house,
            },
            "Senate": {
                "ideal_points": ip_senate,
                "bill_params": bp_senate,
                "party_loyalty": loyalty_senate,
                "centrality": cent_senate,
                "pc_scores": pc_senate,
            },
        }

        results = {}

        for chamber in ["House", "Senate"]:
            cd = chamber_data[chamber]

            # ── Phase 2: Build Vote Features ────────────────────────────
            print_header(f"PHASE 2: VOTE FEATURES — {chamber.upper()}")

            vote_features = build_vote_features(
                votes=votes,
                rollcalls=rollcalls,
                legislators=legislators,
                ideal_points=cd["ideal_points"],
                bill_params=cd["bill_params"],
                party_loyalty=cd["party_loyalty"],
                centrality=cd["centrality"],
                pc_scores=cd["pc_scores"],
                chamber=chamber,
            )
            print(f"  Feature matrix: {vote_features.height:,} rows × {vote_features.width} cols")
            print(f"  Yea rate: {vote_features['vote_binary'].mean():.3f}")

            # Save features
            vote_features.write_parquet(ctx.data_dir / f"vote_features_{chamber.lower()}.parquet")

            # ── Phase 4: Train Vote Models ──────────────────────────────
            print_header(f"PHASE 4: VOTE MODELS — {chamber.upper()}")

            vote_result = train_vote_models(vote_features, chamber)
            print("  Baselines:")
            print(f"    Majority class: {vote_result['baselines']['majority_class_acc']:.3f}")
            print(f"    Party only:     {vote_result['baselines']['party_only_acc']:.3f}")

            # Print CV results
            cv_df = pl.DataFrame(vote_result["cv_results"])
            for name in ["Logistic Regression", "XGBoost", "Random Forest"]:
                acc_mean = cv_df[f"{name}_accuracy"].mean()
                auc_mean = cv_df[f"{name}_auc"].mean()
                acc_std = cv_df[f"{name}_accuracy"].std()
                auc_std = cv_df[f"{name}_auc"].std()
                print(
                    f"  {name}: Acc={acc_mean:.3f}±{acc_std:.3f}, AUC={auc_mean:.3f}±{auc_std:.3f}"
                )

            cv_df.write_parquet(ctx.data_dir / f"cv_results_{chamber.lower()}.parquet")

            # Holdout evaluation
            holdout = evaluate_holdout(
                vote_result["models"],
                vote_result["X_test"],
                vote_result["y_test"],
            )
            holdout_df = pl.DataFrame(holdout)
            holdout_df.write_parquet(ctx.data_dir / f"holdout_results_{chamber.lower()}.parquet")
            print("  Holdout results:")
            for row in holdout:
                print(f"    {row['model']}: Acc={row['accuracy']:.3f}, AUC={row['auc']:.3f}")

            # ── Phase 6: SHAP ───────────────────────────────────────────
            print_header(f"PHASE 6: SHAP ANALYSIS — {chamber.upper()}")

            xgb_model = vote_result["models"]["XGBoost"]
            # Use a sample for SHAP (full dataset can be slow)
            n_shap = min(5000, vote_result["X_test"].shape[0])
            shap_sample = vote_result["X_test"][:n_shap]
            shap_values = compute_shap_values(xgb_model, shap_sample, vote_result["feature_names"])
            print(f"  SHAP computed on {n_shap} test samples")

            # ── Phase 7: Per-Legislator Accuracy ────────────────────────
            print_header(f"PHASE 7: PER-LEGISLATOR ACCURACY — {chamber.upper()}")

            leg_accuracy = compute_per_legislator_accuracy(
                xgb_model,
                vote_features,
                vote_result["feature_names"],
                cd["ideal_points"],
            )
            leg_accuracy.write_parquet(
                ctx.data_dir / f"per_legislator_accuracy_{chamber.lower()}.parquet"
            )
            print(f"  {leg_accuracy.height} legislators evaluated")
            bottom5 = leg_accuracy.sort("accuracy").head(5)
            print("  Bottom 5 accuracy:")
            for row in bottom5.iter_rows(named=True):
                name = row.get("full_name", row["legislator_slug"])
                print(f"    {name}: {row['accuracy']:.3f} ({row['n_correct']}/{row['n_votes']})")

            # ── Phase 8: Surprising Votes ───────────────────────────────
            print_header(f"PHASE 8: SURPRISING VOTES — {chamber.upper()}")

            surprising = find_surprising_votes(
                xgb_model,
                vote_features,
                vote_result["feature_names"],
                rollcalls,
                cd["ideal_points"],
                top_n=TOP_SURPRISING_N,
            )
            surprising.write_parquet(ctx.data_dir / f"surprising_votes_{chamber.lower()}.parquet")
            print(f"  Top {surprising.height} surprising votes identified")

            # ── Phase 9: Plots ──────────────────────────────────────────
            print_header(f"PHASE 9: PLOTS — {chamber.upper()}")

            plot_roc_curves(
                vote_result["models"],
                vote_result["X_test"],
                vote_result["y_test"],
                chamber,
                ctx.plots_dir / f"roc_curves_{chamber.lower()}.png",
            )

            plot_model_comparison(
                vote_result["cv_results"],
                vote_result["baselines"],
                chamber,
                ctx.plots_dir / f"model_comparison_{chamber.lower()}.png",
            )

            xgb_preds = xgb_model.predict(vote_result["X_test"])
            xgb_probs = xgb_model.predict_proba(vote_result["X_test"])[:, 1]

            plot_confusion_matrix(
                vote_result["y_test"],
                xgb_preds,
                chamber,
                "XGBoost",
                ctx.plots_dir / f"confusion_matrix_{chamber.lower()}.png",
            )

            plot_calibration_curve(
                vote_result["y_test"],
                xgb_probs,
                chamber,
                "XGBoost",
                ctx.plots_dir / f"calibration_{chamber.lower()}.png",
            )

            plot_shap_summary(
                shap_values,
                chamber,
                ctx.plots_dir / f"shap_beeswarm_{chamber.lower()}.png",
            )

            plot_shap_bar(
                shap_values,
                chamber,
                ctx.plots_dir / f"shap_bar_{chamber.lower()}.png",
            )

            plot_feature_importance(
                xgb_model,
                vote_result["feature_names"],
                chamber,
                ctx.plots_dir / f"feature_importance_{chamber.lower()}.png",
            )

            plot_per_legislator_accuracy(
                leg_accuracy,
                chamber,
                ctx.plots_dir / f"per_legislator_accuracy_{chamber.lower()}.png",
            )

            plot_surprising_votes(
                surprising,
                chamber,
                ctx.plots_dir / f"surprising_votes_{chamber.lower()}.png",
            )

            # Store results for report
            results[chamber] = {
                "vote_features": vote_features,
                "vote_result": vote_result,
                "holdout": holdout_df,
                "shap_values": shap_values,
                "leg_accuracy": leg_accuracy,
                "surprising_votes": surprising,
            }

        # ── Phase 5: Bill Passage Prediction ────────────────────────────
        if not args.skip_bill_passage:
            for chamber in ["House", "Senate"]:
                cd = chamber_data[chamber]

                print_header(f"PHASE 5: BILL PASSAGE — {chamber.upper()}")

                bill_features = build_bill_features(rollcalls, cd["bill_params"], chamber)
                n_rows = bill_features.height
                n_cols = bill_features.width
                print(f"  Bill feature matrix: {n_rows:,} rows × {n_cols} cols")

                if bill_features.height < 20:
                    print(f"  WARNING: Too few observations ({bill_features.height}), skipping")
                    results[chamber]["passage_skipped"] = True
                    continue

                passage_rate = bill_features["passed_binary"].mean()
                print(f"  Passage rate: {passage_rate:.3f}")

                bill_features.write_parquet(
                    ctx.data_dir / f"bill_features_{chamber.lower()}.parquet"
                )

                passage_result = train_passage_models(bill_features, chamber)

                if passage_result.get("skipped"):
                    results[chamber]["passage_skipped"] = True
                    continue

                # Print passage CV results
                passage_cv_df = pl.DataFrame(passage_result["cv_results"])
                for name in ["Logistic Regression", "XGBoost", "Random Forest"]:
                    acc_key = f"{name}_accuracy"
                    auc_key = f"{name}_auc"
                    if acc_key in passage_cv_df.columns:
                        acc_mean = passage_cv_df[acc_key].mean()
                        auc_vals = passage_cv_df[auc_key].drop_nulls().drop_nans()
                        auc_mean = auc_vals.mean() if auc_vals.len() > 0 else float("nan")
                        print(f"  {name}: Acc={acc_mean:.3f}, AUC={auc_mean:.3f}")

                passage_cv_df.write_parquet(
                    ctx.data_dir / f"passage_cv_results_{chamber.lower()}.parquet"
                )

                # Holdout
                passage_holdout = evaluate_holdout(
                    passage_result["models"],
                    passage_result["X_test"],
                    passage_result["y_test"],
                )
                passage_holdout_df = pl.DataFrame(passage_holdout)
                passage_holdout_df.write_parquet(
                    ctx.data_dir / f"passage_holdout_{chamber.lower()}.parquet"
                )

                # Temporal split
                if passage_result["temporal_results"]:
                    temporal_df = pl.DataFrame(passage_result["temporal_results"])
                    temporal_df.write_parquet(
                        ctx.data_dir / f"temporal_split_{chamber.lower()}.parquet"
                    )
                    print("  Temporal split results:")
                    for row in passage_result["temporal_results"]:
                        m = row["model"]
                        a, u = row["accuracy"], row["auc"]
                        print(f"    {m}: Acc={a:.3f}, AUC={u:.3f}")

                # Surprising bills
                surprising_bills = find_surprising_bills(
                    passage_result["models"]["XGBoost"],
                    bill_features,
                    passage_result["feature_names"],
                    rollcalls,
                    top_n=TOP_SURPRISING_N,
                )
                surprising_bills.write_parquet(
                    ctx.data_dir / f"surprising_bills_{chamber.lower()}.parquet"
                )

                # Passage ROC plot
                plot_passage_roc(
                    passage_result["models"],
                    passage_result["X_test"],
                    passage_result["y_test"],
                    chamber,
                    ctx.plots_dir / f"passage_roc_{chamber.lower()}.png",
                )

                results[chamber]["passage_result"] = passage_result
                results[chamber]["passage_cv"] = passage_cv_df
                results[chamber]["passage_holdout"] = passage_holdout_df
                results[chamber]["temporal_results"] = passage_result["temporal_results"]
                results[chamber]["surprising_bills"] = surprising_bills
                results[chamber]["bill_features"] = bill_features

        # ── Filtering Manifest ──────────────────────────────────────────
        print_header("FILTERING MANIFEST")

        manifest = {
            "minority_threshold": MINORITY_THRESHOLD,
            "min_votes": MIN_VOTES,
            "test_size": TEST_SIZE,
            "n_splits": N_SPLITS,
            "random_seed": RANDOM_SEED,
            "chambers": {},
        }

        for chamber in ["House", "Senate"]:
            r = results[chamber]
            vf = r["vote_features"]
            manifest["chambers"][chamber] = {
                "vote_observations": vf.height,
                "unique_legislators": vf["legislator_slug"].n_unique(),
                "unique_rollcalls": vf["vote_id"].n_unique(),
                "yea_rate": float(vf["vote_binary"].mean()),
                "n_features": len(r["vote_result"]["feature_names"]),
                "feature_names": r["vote_result"]["feature_names"],
            }

        manifest_path = ctx.run_dir / "filtering_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print("  Saved: filtering_manifest.json")

        # ── HTML Report ─────────────────────────────────────────────────
        print_header("HTML REPORT")

        build_prediction_report(
            ctx.report,
            results=results,
            plots_dir=ctx.plots_dir,
            skip_bill_passage=args.skip_bill_passage,
        )

        print(f"\n  Report sections: {len(ctx.report._sections)}")
        print("  Done!")


if __name__ == "__main__":
    main()
