"""
Kansas Legislature — Vote & Bill Passage Prediction (Phase 8)

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

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
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
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir, strip_leadership_suffix
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir, strip_leadership_suffix

try:
    from analysis.prediction_report import build_prediction_report
except ModuleNotFoundError:
    from prediction_report import build_prediction_report  # type: ignore[no-redef]

try:
    from analysis.nlp_features import (
        fit_topic_features,
        get_topic_display_names,
        plot_topic_words,
    )
except ModuleNotFoundError:
    from nlp_features import (  # type: ignore[no-redef]
        fit_topic_features,
        get_topic_display_names,
        plot_topic_words,
    )

try:
    from analysis.phase_utils import (
        match_sponsor_to_slug,
        print_header,
        save_fig,
    )
except ImportError:
    from phase_utils import (  # type: ignore[no-redef]
        match_sponsor_to_slug,
        print_header,
        save_fig,
    )


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
MIN_VOTES_RELIABLE = 10  # minimum holdout votes for reliable per-legislator accuracy
PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}
HARDEST_N = 8

# Plain-English feature names for SHAP and feature importance plots (nontechnical audience)
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "xi_mean": "How conservative the legislator is",
    "beta_mean": "How partisan the bill is",
    "alpha_mean": "How easy the bill is to pass",
    "xi_sd": "How uncertain the ideology estimate is",
    "xi_x_beta": "Legislator\u2013bill ideology match",
    "party_binary": "Party (Republican or Democrat)",
    "loyalty_rate": "How often they vote with their party",
    "betweenness": "How much they bridge voting blocs",
    "eigenvector": "How influential in the voting network",
    "pagerank": "How central in the voting network",
    "PC1": "Primary left\u2013right position",
    "PC2": "Secondary voting dimension",
    "day_of_session": "How far into the session",
    "is_veto_override": "Whether it\u2019s a veto override vote",
    "sponsor_party_R": "Sponsored by a Republican",
}

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
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--skip-bill-passage",
        action="store_true",
        help="Skip bill passage prediction (vote prediction only)",
    )
    return parser.parse_args()


# ── Phase 1: Load Data ──────────────────────────────────────────────────────


def load_vote_data(data_dir: Path) -> pl.DataFrame:
    """Load individual votes CSV."""
    return pl.read_csv(data_dir / f"{data_dir.name}_votes.csv")


def load_rollcall_data(data_dir: Path) -> pl.DataFrame:
    """Load rollcalls CSV."""
    rc = pl.read_csv(data_dir / f"{data_dir.name}_rollcalls.csv")
    if "vote_type" in rc.columns:
        rc = rc.with_columns(pl.col("vote_type").fill_null("Unknown").alias("vote_type"))
    return rc


def load_legislator_data(data_dir: Path) -> pl.DataFrame:
    """Load legislators CSV."""
    legislators = pl.read_csv(data_dir / f"{data_dir.name}_legislators.csv")
    return legislators.with_columns(
        pl.col("full_name")
        .map_elements(strip_leadership_suffix, return_dtype=pl.Utf8)
        .alias("full_name"),
        pl.col("party").fill_null("Independent").replace("", "Independent").alias("party"),
    )


def _load_parquet_pair(
    base_dir: Path, prefix: str
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load a house/senate parquet pair, returning None for missing files."""
    results: list[pl.DataFrame | None] = []
    for ch in ("house", "senate"):
        path = base_dir / "data" / f"{prefix}_{ch}.parquet"
        if path.exists():
            results.append(pl.read_parquet(path))
        else:
            results.append(None)
    return results[0], results[1]


def load_ideal_points(
    irt_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load IRT ideal points for both chambers."""
    return _load_parquet_pair(irt_dir, "ideal_points")


def load_bill_params(
    irt_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load IRT bill parameters for both chambers."""
    return _load_parquet_pair(irt_dir, "bill_params")


def load_party_loyalty(
    clustering_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load party loyalty rates from clustering phase."""
    return _load_parquet_pair(clustering_dir, "party_loyalty")


def load_centrality(
    network_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load centrality measures from network phase."""
    return _load_parquet_pair(network_dir, "centrality")


def load_pc_scores(
    pca_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load PCA scores from PCA phase."""
    return _load_parquet_pair(pca_dir, "pc_scores")


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
    topic_features: pl.DataFrame | None = None,
    legislators: pl.DataFrame | None = None,
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

    # Join IRT bill params (only beta and override — alpha excluded as target leakage)
    bp = bill_params.select("vote_id", "beta_mean", "is_veto_override")
    rc = rc.join(bp, on="vote_id", how="left")

    # Compute derived features
    # NOTE: margin and alpha_mean excluded — target leakage (see design/prediction.md).
    # chamber_binary excluded — constant within per-chamber models (zero information).
    rc = rc.with_columns(
        _compute_day_of_session(rc["vote_date"]),
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

    # Sponsor party feature: 1 if primary sponsor is Republican, 0 otherwise
    has_sponsor_party = False
    if legislators is not None:
        # Slug-based matching (preferred — unambiguous)
        if "sponsor_slugs" in rc.columns:
            # Extract first slug from semicolon-joined string
            rc = rc.with_columns(
                pl.col("sponsor_slugs").str.split("; ").list.first().alias("_first_slug")
            )
            # Determine the slug column name in legislators
            slug_col = "legislator_slug" if "legislator_slug" in legislators.columns else "slug"
            slug_party = legislators.select(
                pl.col(slug_col).alias("_first_slug"), pl.col("party")
            ).unique(subset=["_first_slug"])
            rc = rc.join(slug_party, on="_first_slug", how="left")
            rc = rc.with_columns(
                pl.when(pl.col("party") == "Republican")
                .then(1)
                .otherwise(0)
                .alias("sponsor_party_R")
            )
            rc = rc.drop("_first_slug", "party")
            has_sponsor_party = rc["sponsor_party_R"].sum() > 0

        # Text-based fallback for old data without sponsor_slugs:
        # resolve sponsor text → slug, then use the same slug-based join path
        if not has_sponsor_party and "sponsor" in rc.columns:
            resolved_slugs = [
                match_sponsor_to_slug(row.get("sponsor", ""), legislators) or ""
                for row in rc.iter_rows(named=True)
            ]
            rc = rc.with_columns(pl.Series("_first_slug", resolved_slugs))
            slug_col = "legislator_slug" if "legislator_slug" in legislators.columns else "slug"
            slug_party = legislators.select(
                pl.col(slug_col).alias("_first_slug"), pl.col("party")
            ).unique(subset=["_first_slug"])
            rc = rc.join(slug_party, on="_first_slug", how="left")
            rc = rc.with_columns(
                pl.when(pl.col("party") == "Republican")
                .then(1)
                .otherwise(0)
                .alias("sponsor_party_R")
            )
            rc = rc.drop("_first_slug", "party")
            has_sponsor_party = rc["sponsor_party_R"].sum() > 0

    # Join NLP topic features if provided
    if topic_features is not None:
        # topic_features must have vote_id + topic_* columns
        topic_cols_to_join = [c for c in topic_features.columns if c.startswith("topic_")]
        if topic_cols_to_join and "vote_id" in topic_features.columns:
            rc = rc.join(
                topic_features.select(["vote_id"] + topic_cols_to_join),
                on="vote_id",
                how="left",
            )

    feature_cols = [
        "beta_mean",
        "is_veto_override",
        "day_of_session",
    ]
    if has_sponsor_party:
        feature_cols.append("sponsor_party_R")
    vt_cols = [c for c in rc.columns if c.startswith("vt_")]
    pfx_cols = [c for c in rc.columns if c.startswith("pfx_")]
    topic_cols = [c for c in rc.columns if c.startswith("topic_")]
    feature_cols.extend(vt_cols)
    feature_cols.extend(pfx_cols)
    feature_cols.extend(topic_cols)

    metadata_cols = ["vote_id", "bill_number", "passed_binary", "vote_date", "bill_prefix"]
    keep_cols = [c for c in feature_cols + metadata_cols if c in rc.columns]
    result = rc.select(keep_cols).drop_nulls(subset=feature_cols + ["passed_binary"])

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
            n_jobs=-1,
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

    # Train/test split — use indices so callers can reconstruct the holdout DataFrame
    all_indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        all_indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

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
            fold_results[f"{name}_brier"] = brier_score_loss(y_val, y_prob)
            fold_results[f"{name}_logloss"] = log_loss(y_val, y_prob)

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
        "test_indices": test_indices,
        "feature_names": feature_cols,
        "baselines": baselines,
    }


def evaluate_holdout(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[dict]:
    """Evaluate all models on the holdout test set."""
    n_classes = len(np.unique(y_test))
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "auc": roc_auc_score(y_test, y_prob) if n_classes > 1 else float("nan"),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "brier": brier_score_loss(y_test, y_prob),
                "logloss": log_loss(y_test, y_prob, labels=[0, 1]),
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
        features_df, "passed_binary", ["vote_id", "bill_number", "vote_date", "bill_prefix"]
    )
    X = features_df.select(feature_cols).to_numpy().astype(np.float64)
    y = features_df["passed_binary"].to_numpy()

    if len(y) < 20:
        print(f"  WARNING: Only {len(y)} observations for {chamber} passage — skipping")
        return {"skipped": True, "reason": f"Only {len(y)} observations"}

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        label = int(y[0])
        print(f"  WARNING: All bills have same outcome ({label}) for {chamber} — skipping passage")
        return {"skipped": True, "reason": f"Only one class ({label}) in data"}
    min_class_count = int(counts.min())
    if min_class_count < 2:
        print(
            f"  WARNING: Minority class has only {min_class_count} member(s) for {chamber}"
            " — skipping passage (need ≥2 for stratified split)"
        )
        return {"skipped": True, "reason": f"Minority class has {min_class_count} member(s)"}

    # Train/test split — keep indices so callers can filter original DataFrame
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
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
            fold_results[f"{name}_brier"] = brier_score_loss(y_val, y_prob)
            fold_results[f"{name}_logloss"] = log_loss(y_val, y_prob, labels=[0, 1])

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
        "test_indices": idx_test,
        "feature_names": feature_cols,
        "temporal_results": temporal_results,
    }


def _temporal_split_eval(
    features_df: pl.DataFrame,
    feature_cols: list[str],
    chamber: str,
) -> list[dict]:
    """Evaluate passage models with a temporal train/test split."""
    # Sort by actual date (string sorting on MM/DD/YYYY is wrong across years)
    try:
        sorted_df = features_df.with_columns(
            pl.col("vote_date").str.to_date("%m/%d/%Y").alias("_sort_date")
        )
    except Exception:
        sorted_df = features_df.with_columns(
            pl.col("vote_date").str.to_date("%Y-%m-%d").alias("_sort_date")
        )
    sorted_df = sorted_df.sort("_sort_date").drop("_sort_date")
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
        row["brier"] = brier_score_loss(y_test, y_prob)
        row["logloss"] = log_loss(y_test, y_prob, labels=[0, 1])
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

    # Flag legislators with too few holdout votes for reliable accuracy
    leg_acc = leg_acc.with_columns(
        (pl.col("n_votes") >= MIN_VOTES_RELIABLE).alias("reliable"),
    )

    # Join legislator metadata
    ip_meta = ideal_points.select("legislator_slug", "full_name", "party", "xi_mean")
    leg_acc = leg_acc.join(ip_meta, on="legislator_slug", how="left")

    return leg_acc.sort("accuracy")


@dataclass(frozen=True)
class HardestLegislator:
    """A legislator the model struggles to predict, with a plain-English explanation."""

    slug: str
    full_name: str
    party: str
    xi_mean: float
    accuracy: float
    n_votes: int
    explanation: str


def detect_hardest_legislators(
    leg_accuracy: pl.DataFrame,
    n: int = HARDEST_N,
) -> list[HardestLegislator]:
    """Identify the N legislators the model struggles with most.

    Returns a list of HardestLegislator sorted by accuracy ascending (worst first).
    Each entry includes a data-driven plain-English explanation based on IRT position.
    """
    if leg_accuracy.height == 0:
        return []

    # Only rank legislators with enough holdout votes for reliable accuracy
    if "reliable" in leg_accuracy.columns:
        reliable = leg_accuracy.filter(pl.col("reliable"))
    else:
        reliable = leg_accuracy
    if reliable.height == 0:
        reliable = leg_accuracy  # fallback if none are reliable
    bottom = reliable.sort("accuracy").head(n)
    results: list[HardestLegislator] = []

    # Compute cross-party midpoint for explanation logic
    r_vals = leg_accuracy.filter(pl.col("party") == "Republican")["xi_mean"]
    d_vals = leg_accuracy.filter(pl.col("party") == "Democrat")["xi_mean"]
    if r_vals.len() > 0 and d_vals.len() > 0:
        midpoint = (r_vals.min() + d_vals.max()) / 2
    else:
        midpoint = 0.0

    # Compute party medians for "centrist for their party" logic
    party_medians: dict[str, float] = {}
    for party in ["Republican", "Democrat"]:
        vals = leg_accuracy.filter(pl.col("party") == party)["xi_mean"]
        if vals.len() > 0:
            party_medians[party] = float(vals.median())

    for row in bottom.iter_rows(named=True):
        xi = row.get("xi_mean")
        party = row.get("party", "Unknown")

        if xi is not None and midpoint is not None and abs(xi - midpoint) < 0.5:
            explanation = "Moderate \u2014 close to the boundary between parties"
        elif xi is not None and party in party_medians:
            median = party_medians[party]
            # "Centrist for their party" = closer to midpoint than their party median
            if abs(xi - midpoint) < abs(median - midpoint) * 0.7:
                explanation = f"Centrist {party} \u2014 less ideologically committed than most"
            elif party == "Republican" and xi > median:
                explanation = "Strongly conservative but occasionally crosses party lines"
            elif party == "Democrat" and xi < median:
                explanation = "Strongly liberal but occasionally crosses party lines"
            else:
                explanation = "Voting pattern doesn\u2019t fit the one-dimensional model"
        else:
            explanation = "Voting pattern doesn\u2019t fit the one-dimensional model"

        results.append(
            HardestLegislator(
                slug=row["legislator_slug"],
                full_name=row.get("full_name") or row["legislator_slug"],
                party=party,
                xi_mean=float(xi) if xi is not None else 0.0,
                accuracy=float(row["accuracy"]),
                n_votes=int(row["n_votes"]),
                explanation=explanation,
            )
        )

    return results


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
    empty_schema = {
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
    if wrong_mask.sum() == 0:
        return pl.DataFrame(schema=empty_schema)

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

    keep_cols = list(empty_schema.keys())
    return surprising.select([c for c in keep_cols if c in surprising.columns])


def split_surprising_by_class(
    surprising: pl.DataFrame,
    top_n: int = TOP_SURPRISING_N,
) -> dict[str, pl.DataFrame]:
    """Split surprising votes into false positives (surprising Nay) and false negatives.

    - Surprising Nay (FP): predicted=1, actual=0 — expected Yea, got Nay
    - Surprising Yea (FN): predicted=0, actual=1 — expected Nay, got Yea

    Returns dict with "surprising_nay" and "surprising_yea" DataFrames.
    """
    if surprising.height == 0:
        return {"surprising_nay": surprising.clone(), "surprising_yea": surprising.clone()}

    fp = surprising.filter((pl.col("predicted") == 1) & (pl.col("actual") == 0)).head(top_n)
    fn = surprising.filter((pl.col("predicted") == 0) & (pl.col("actual") == 1)).head(top_n)
    return {"surprising_nay": fp, "surprising_yea": fn}


def find_surprising_bills(
    model: object,
    features_df: pl.DataFrame,
    feature_cols: list[str],
    rollcalls: pl.DataFrame,
    top_n: int = TOP_SURPRISING_N,
) -> pl.DataFrame:
    """Find bills where passage prediction was most confidently wrong."""
    X = features_df.select(feature_cols).to_numpy().astype(np.float64)
    y = features_df["passed_binary"].to_numpy()
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    confidence_error = np.abs(y_prob - y.astype(float))

    # Only look at wrong predictions (consistent with find_surprising_votes)
    wrong_mask = y_pred != y
    if wrong_mask.sum() == 0:
        return pl.DataFrame(
            schema={
                "vote_id": pl.Utf8,
                "bill_number": pl.Utf8,
                "passed_binary": pl.Int64,
                "y_prob": pl.Float64,
                "predicted": pl.Float64,
                "confidence_error": pl.Float64,
                "motion": pl.Utf8,
                "vote_type": pl.Utf8,
                "yea_count": pl.Int64,
                "nay_count": pl.Int64,
            }
        )

    wrong_df = (
        features_df.filter(pl.Series(wrong_mask))
        .select("vote_id", "bill_number", "passed_binary")
        .with_columns(
            pl.Series("y_prob", y_prob[wrong_mask]),
            pl.Series("predicted", y_pred[wrong_mask]),
            pl.Series("confidence_error", confidence_error[wrong_mask]),
        )
    )

    # Enrich with rollcall metadata
    rc_meta = rollcalls.select("vote_id", "motion", "vote_type", "yea_count", "nay_count")
    wrong_df = wrong_df.join(rc_meta, on="vote_id", how="left")

    return wrong_df.sort("confidence_error", descending=True).head(top_n)


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


def compute_stratified_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bill_prefixes: list[str],
) -> pl.DataFrame:
    """Compute accuracy and passage rate stratified by bill prefix.

    Returns a DataFrame with columns: prefix, count, accuracy, passage_rate,
    sorted by count descending.
    """
    df = pl.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "bill_prefix": bill_prefixes,
        }
    )
    result = (
        df.group_by("bill_prefix")
        .agg(
            pl.len().alias("count"),
            (pl.col("y_true") == pl.col("y_pred")).mean().alias("accuracy"),
            pl.col("y_true").mean().alias("passage_rate"),
        )
        .sort("count", descending=True)
        .rename({"bill_prefix": "prefix"})
    )
    return result


def _rename_shap_features(shap_values: shap.Explanation) -> shap.Explanation:
    """Return a copy of SHAP values with plain-English feature names."""
    import copy

    sv = copy.deepcopy(shap_values)
    if sv.feature_names is not None:
        sv.feature_names = [
            FEATURE_DISPLAY_NAMES.get(f, f.replace("_", " ").replace("vt ", "Vote type: "))
            for f in sv.feature_names
        ]
    return sv


def plot_shap_summary(
    shap_values: shap.Explanation,
    chamber: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    """SHAP beeswarm plot with plain-English feature names."""
    sv = _rename_shap_features(shap_values)
    shap.plots.beeswarm(sv, max_display=TOP_SHAP_FEATURES, show=False, plot_size=(10, 8))
    fig = plt.gcf()
    fig.suptitle(title or f"{chamber} \u2014 What Predicts a Yea Vote?", fontsize=14, y=1.02)
    save_fig(fig, out_path)


def plot_shap_bar(
    shap_values: shap.Explanation,
    chamber: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    """SHAP mean |SHAP| bar plot with plain-English feature names."""
    sv = _rename_shap_features(shap_values)
    shap.plots.bar(sv, max_display=TOP_SHAP_FEATURES, show=False)
    fig = plt.gcf()
    fig.suptitle(
        title or f"{chamber} \u2014 What Predicts a Yea Vote? (Feature Importance)",
        fontsize=14,
        y=1.02,
    )
    save_fig(fig, out_path)


def plot_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    chamber: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    """XGBoost native feature importance (gain) with plain-English names."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[-TOP_SHAP_FEATURES:]

    display_names = [
        FEATURE_DISPLAY_NAMES.get(
            feature_names[i],
            feature_names[i].replace("_", " ").replace("vt ", "Vote type: "),
        )
        for i in sorted_idx
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(sorted_idx)),
        importances[sorted_idx],
        color="#FF9800",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(display_names, fontsize=10)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title(title or f"{chamber} \u2014 What Matters Most for Predicting Votes?", fontsize=14)
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

    # Label bottom 5 by accuracy with callout boxes
    bottom5 = leg_accuracy.sort("accuracy").head(5)
    for row in bottom5.iter_rows(named=True):
        name = row.get("full_name") or row["legislator_slug"]
        name_part = name.split(" - ")[0] if name else ""
        last_name = name_part.split()[-1] if name_part else "?"
        ax.annotate(
            last_name,
            (row["xi_mean"], row["accuracy"]),
            fontsize=8,
            fontweight="bold",
            textcoords="offset points",
            xytext=(8, -8),
            bbox={"boxstyle": "round,pad=0.3", "fc": "wheat", "alpha": 0.7},
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
        )

    ax.set_xlabel("Ideology (Liberal \u2190 \u2192 Conservative)", fontsize=12)
    ax.set_ylabel("Prediction Accuracy", fontsize=12)
    ax.set_title(f"{chamber} \u2014 Some Legislators Are Harder to Predict", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, out_path)


def plot_hardest_to_predict(
    hardest: list[HardestLegislator],
    chamber: str,
    chamber_median_accuracy: float,
    out_path: Path,
) -> None:
    """Horizontal dot chart spotlighting the legislators the model struggles with most."""
    if not hardest:
        return

    fig, ax = plt.subplots(figsize=(12, max(4, 0.6 * len(hardest) + 1.5)))

    labels = []
    accuracies = []
    colors = []
    for h in hardest:
        # Strip leadership suffixes ("- Vice President of the Senate") before extracting last name
        name_part = h.full_name.split(" - ")[0] if h.full_name else ""
        last_name = name_part.split()[-1] if name_part else "?"
        party_initial = h.party[0] if h.party else "?"
        labels.append(f"{last_name} ({party_initial})")
        accuracies.append(h.accuracy)
        colors.append(PARTY_COLORS.get(h.party, "#999999"))

    y_pos = list(range(len(hardest)))

    ax.scatter(
        accuracies,
        y_pos,
        c=colors,
        s=120,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    # Annotate each dot with its explanation
    for i, h in enumerate(hardest):
        ax.annotate(
            h.explanation,
            (accuracies[i], i),
            textcoords="offset points",
            xytext=(12, 0),
            fontsize=8,
            fontstyle="italic",
            color="#666666",
            va="center",
        )

    # Vertical dashed line at chamber median
    ax.axvline(
        chamber_median_accuracy,
        color="#888888",
        linestyle="--",
        linewidth=1,
        zorder=1,
    )
    # Label the median line at the bottom of the plot
    ax.text(
        chamber_median_accuracy,
        len(hardest) - 0.4,
        f" Median: {chamber_median_accuracy:.1%}",
        fontsize=8,
        color="#888888",
        va="top",
    )

    ax.set_xlabel("Prediction Accuracy", fontsize=12)
    ax.set_title(
        "The model correctly predicts ~95% of all votes. These legislators are the exceptions.",
        fontsize=9,
        fontstyle="italic",
        color="#555555",
        pad=4,
    )
    fig.suptitle(
        f"{chamber} \u2014 Which Legislators Does the Model Struggle With Most?",
        fontsize=14,
        y=0.99,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 0.58, 0.96])
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
    ax.set_xlabel("Model's Predicted Chance of Yea", fontsize=12)
    ax.set_ylabel("Actual Yea Rate", fontsize=12)
    ax.set_title(f"{chamber} \u2014 {model_name} Calibration Curve", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)

    # Add annotation explaining what good calibration means
    ax.annotate(
        "When the model says 80% chance\nof Yea, it's right about 80%\nof the time",
        xy=(0.8, 0.8),
        xycoords="data",
        xytext=(0.25, 0.85),
        textcoords="data",
        fontsize=9,
        fontstyle="italic",
        color="#555555",
        bbox={"boxstyle": "round,pad=0.4", "fc": "lightyellow", "alpha": 0.8, "ec": "#cccccc"},
        arrowprops={"arrowstyle": "->", "color": "#888888", "lw": 1.2},
    )

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
        name = row.get("full_name") or row.get("legislator_slug") or "?"
        # Strip leadership suffixes ("- Vice President of the Senate") before extracting last name
        name_part = name.split(" - ")[0] if name else ""
        last_name = name_part.split()[-1] if name_part else "?"
        bill = row.get("bill_number", "?")
        labels.append(f"{last_name}\n{bill}")

    y_pos = range(min(surprising_df.height, TOP_SURPRISING_N))
    errors = surprising_df["confidence_error"].head(TOP_SURPRISING_N).to_numpy()

    colors = []
    for row in surprising_df.head(TOP_SURPRISING_N).iter_rows(named=True):
        party = row.get("party", "Unknown")
        colors.append(PARTY_COLORS.get(party, "#999999"))

    ax.barh(list(y_pos), errors, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels[: len(y_pos)], fontsize=8)
    ax.set_xlabel("How Surprising (higher = more unexpected)", fontsize=12)
    ax.set_title(f"{chamber} \u2014 The Votes Nobody Expected", fontsize=14)
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

    from tallgrass.session import KSSession

    ks = KSSession.from_session_string(args.session)
    data_dir = Path(args.data_dir) if args.data_dir else ks.data_dir

    results_root = ks.results_dir
    irt_dir = resolve_upstream_dir(
        "04_irt",
        results_root,
        args.run_id,
        Path(args.irt_dir) if args.irt_dir else None,
    )
    clustering_dir = resolve_upstream_dir(
        "05_clustering",
        results_root,
        args.run_id,
        Path(args.clustering_dir) if args.clustering_dir else None,
    )
    network_dir = resolve_upstream_dir(
        "06_network",
        results_root,
        args.run_id,
        Path(args.network_dir) if args.network_dir else None,
    )
    pca_dir = resolve_upstream_dir(
        "02_pca",
        results_root,
        args.run_id,
        Path(args.pca_dir) if args.pca_dir else None,
    )

    with RunContext(
        session=args.session,
        analysis_name="08_prediction",
        params=vars(args),
        primer=PREDICTION_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        # ── Phase 1: Load Data ──────────────────────────────────────────
        print_header("PHASE 1: LOADING DATA")

        print("  Loading vote data...")
        votes = load_vote_data(data_dir)
        rollcalls = load_rollcall_data(data_dir)
        legislators = load_legislator_data(data_dir)
        print(f"    Votes: {votes.height:,} rows")
        print(f"    Roll calls: {rollcalls.height:,} rows")
        print(f"    Legislators: {legislators.height:,} rows")

        print("  Loading upstream results...")
        ip_house, ip_senate = load_ideal_points(irt_dir)
        bp_house, bp_senate = load_bill_params(irt_dir)
        loyalty_house, loyalty_senate = load_party_loyalty(clustering_dir)
        cent_house, cent_senate = load_centrality(network_dir)
        pc_house, pc_senate = load_pc_scores(pca_dir)

        # IRT is required — skip phase entirely if both chambers missing
        if ip_house is None and ip_senate is None:
            print("Phase 08 (Prediction): skipping — no IRT ideal points available")
            return

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

            if cd["ideal_points"] is None:
                print(f"\n  {chamber}: IRT not available — skipping")
                continue

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

            # Evaluate on holdout only to avoid in-sample inflation (ADR-0031)
            holdout_features = vote_features[vote_result["test_indices"].tolist()]
            print(
                f"  Evaluating on holdout set: {holdout_features.height:,} observations "
                f"(out of {vote_features.height:,} total)"
            )

            leg_accuracy = compute_per_legislator_accuracy(
                xgb_model,
                holdout_features,
                vote_result["feature_names"],
                cd["ideal_points"],
            )
            leg_accuracy.write_parquet(
                ctx.data_dir / f"per_legislator_accuracy_{chamber.lower()}.parquet"
            )
            ctx.export_csv(
                leg_accuracy,
                f"per_legislator_accuracy_{chamber.lower()}.csv",
                f"Per-legislator prediction accuracy for {chamber}",
            )
            print(f"  {leg_accuracy.height} legislators evaluated")
            bottom5 = leg_accuracy.sort("accuracy").head(5)
            print("  Bottom 5 accuracy:")
            for row in bottom5.iter_rows(named=True):
                name = row.get("full_name", row["legislator_slug"])
                print(f"    {name}: {row['accuracy']:.3f} ({row['n_correct']}/{row['n_votes']})")

            hardest = detect_hardest_legislators(leg_accuracy)
            chamber_median_accuracy = float(leg_accuracy["accuracy"].median())
            print(f"  Hardest to predict: {len(hardest)} legislators")
            for h in hardest:
                print(f"    {h.full_name}: {h.accuracy:.3f} — {h.explanation}")

            # ── Phase 8: Surprising Votes ───────────────────────────────
            print_header(f"PHASE 8: SURPRISING VOTES — {chamber.upper()}")

            # Evaluate on holdout only — in-sample predictions are biased (ADR-0031)
            surprising = find_surprising_votes(
                xgb_model,
                holdout_features,
                vote_result["feature_names"],
                rollcalls,
                cd["ideal_points"],
                top_n=TOP_SURPRISING_N,
            )
            surprising.write_parquet(ctx.data_dir / f"surprising_votes_{chamber.lower()}.parquet")
            ctx.export_csv(
                surprising,
                f"surprising_votes_{chamber.lower()}.csv",
                f"Most surprising votes in {chamber}",
            )
            print(f"  Top {surprising.height} surprising votes identified (holdout only)")

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

            plot_hardest_to_predict(
                hardest,
                chamber,
                chamber_median_accuracy,
                ctx.plots_dir / f"hardest_to_predict_{chamber.lower()}.png",
            )

            plot_surprising_votes(
                surprising,
                chamber,
                ctx.plots_dir / f"surprising_votes_{chamber.lower()}.png",
            )

            # Store results for report
            surprising_split = split_surprising_by_class(surprising)
            results[chamber] = {
                "vote_features": vote_features,
                "vote_result": vote_result,
                "holdout": holdout_df,
                "shap_values": shap_values,
                "leg_accuracy": leg_accuracy,
                "hardest": hardest,
                "chamber_median_accuracy": chamber_median_accuracy,
                "surprising_votes": surprising,
                "surprising_nay": surprising_split["surprising_nay"],
                "surprising_yea": surprising_split["surprising_yea"],
            }

        # ── Phase 5: Bill Passage Prediction ────────────────────────────
        topic_models: dict[str, object] = {}
        if not args.skip_bill_passage:
            for chamber in ["House", "Senate"]:
                cd = chamber_data[chamber]

                print_header(f"PHASE 5: BILL PASSAGE — {chamber.upper()}")

                # Fit NLP topic model on bill short_title text
                chamber_rc = rollcalls.filter(pl.col("chamber") == chamber)
                print("  Fitting NLP topic model on bill titles...")
                topic_df, topic_model = fit_topic_features(chamber_rc["short_title"])
                topic_models[chamber] = topic_model

                # Attach vote_id for joining
                topic_df = topic_df.with_columns(chamber_rc["vote_id"])
                topic_df.write_parquet(ctx.data_dir / f"topic_features_{chamber.lower()}.parquet")
                print(
                    f"    Topics: {topic_model.n_topics}, "
                    f"Vocab: {topic_model.vocabulary_size}, "
                    f"Docs: {topic_model.n_documents}"
                )
                for i in range(topic_model.n_topics):
                    col = f"topic_{i}"
                    words = topic_model.topic_top_words.get(col, [])
                    print(f"    {topic_model.topic_labels[i]}: {', '.join(words)}")

                # Plot topic words
                plot_topic_words(
                    topic_model,
                    chamber,
                    ctx.plots_dir / f"topic_words_{chamber.lower()}.png",
                )

                # Update FEATURE_DISPLAY_NAMES with topic labels
                display_names = get_topic_display_names(topic_model)
                FEATURE_DISPLAY_NAMES.update(display_names)

                bill_features = build_bill_features(
                    rollcalls,
                    cd["bill_params"],
                    chamber,
                    topic_features=topic_df,
                    legislators=legislators,
                )
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

                # Surprising bills — evaluate on holdout test set only (avoid in-sample leakage)
                test_bill_features = bill_features[passage_result["test_indices"].tolist()]
                surprising_bills = find_surprising_bills(
                    passage_result["models"]["XGBoost"],
                    test_bill_features,
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

                # Passage SHAP analysis
                passage_xgb = passage_result["models"]["XGBoost"]
                n_passage_shap = min(500, passage_result["X_test"].shape[0])
                passage_shap_sample = passage_result["X_test"][:n_passage_shap]
                passage_shap_values = compute_shap_values(
                    passage_xgb, passage_shap_sample, passage_result["feature_names"]
                )
                print(f"  Passage SHAP computed on {n_passage_shap} test samples")

                plot_shap_summary(
                    passage_shap_values,
                    chamber,
                    ctx.plots_dir / f"shap_passage_{chamber.lower()}.png",
                    title=f"{chamber} \u2014 What Predicts Bill Passage?",
                )
                plot_shap_bar(
                    passage_shap_values,
                    chamber,
                    ctx.plots_dir / f"shap_bar_passage_{chamber.lower()}.png",
                    title=f"{chamber} \u2014 What Predicts Bill Passage? (Feature Importance)",
                )
                plot_feature_importance(
                    passage_xgb,
                    passage_result["feature_names"],
                    chamber,
                    ctx.plots_dir / f"passage_importance_{chamber.lower()}.png",
                    title=f"{chamber} \u2014 What Matters Most for Bill Passage?",
                )

                # Stratified accuracy by bill prefix
                passage_preds = passage_xgb.predict(passage_result["X_test"])
                if "bill_prefix" in bill_features.columns:
                    test_bill_prefixes = (
                        bill_features[passage_result["test_indices"].tolist()]
                        .get_column("bill_prefix")
                        .to_list()
                    )
                    stratified = compute_stratified_accuracy(
                        passage_result["y_test"], passage_preds, test_bill_prefixes
                    )
                    stratified.write_parquet(
                        ctx.data_dir / f"stratified_accuracy_{chamber.lower()}.parquet"
                    )
                    ctx.export_csv(
                        stratified,
                        f"stratified_accuracy_{chamber.lower()}.csv",
                        f"Passage accuracy by bill prefix for {chamber}",
                    )
                    print(f"  Stratified accuracy: {stratified.height} prefixes")
                    for row in stratified.iter_rows(named=True):
                        print(
                            f"    {row['prefix']}: "
                            f"acc={row['accuracy']:.3f}, "
                            f"rate={row['passage_rate']:.3f}, "
                            f"n={row['count']}"
                        )
                else:
                    stratified = None

                results[chamber]["passage_result"] = passage_result
                results[chamber]["passage_cv"] = passage_cv_df
                results[chamber]["passage_holdout"] = passage_holdout_df
                results[chamber]["temporal_results"] = passage_result["temporal_results"]
                results[chamber]["surprising_bills"] = surprising_bills
                results[chamber]["bill_features"] = bill_features
                results[chamber]["passage_shap_values"] = passage_shap_values
                results[chamber]["stratified_accuracy"] = stratified

        if not results:
            print("Phase 08 (Prediction): skipping — no chambers with IRT data")
            return

        # ── Filtering Manifest ──────────────────────────────────────────
        print_header("FILTERING MANIFEST")

        manifest: dict = {
            "test_size": TEST_SIZE,
            "n_splits": N_SPLITS,
            "random_seed": RANDOM_SEED,
            "note": "Minority/participation filtering done upstream in EDA/IRT",
            "chambers": {},
        }

        # Add NLP metadata if topic models were fitted
        if topic_models:
            nlp_meta = {}
            for chamber, tm in topic_models.items():
                nlp_meta[chamber] = {
                    "n_topics": tm.n_topics,
                    "vocabulary_size": tm.vocabulary_size,
                    "n_documents": tm.n_documents,
                    "topic_labels": tm.topic_labels,
                }
            manifest["nlp_topic_models"] = nlp_meta

        for chamber in ["House", "Senate"]:
            if chamber not in results:
                continue
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
            topic_models=topic_models if topic_models else None,
        )

        print(f"\n  Report sections: {len(ctx.report._sections)}")
        print("  Done!")


if __name__ == "__main__":
    main()
