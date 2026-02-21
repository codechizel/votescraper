"""Prediction-specific HTML report builder.

Builds ~25 sections (tables, figures, and text) for the prediction analysis report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from prediction.py):
    from analysis.prediction_report import build_prediction_report
    build_prediction_report(ctx.report, results=results, ...)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

try:
    from analysis.report import FigureSection, ReportBuilder, TableSection, TextSection, make_gt
except ModuleNotFoundError:
    from report import (  # type: ignore[no-redef]
        FigureSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
    )


def build_prediction_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    plots_dir: Path,
    skip_bill_passage: bool = False,
) -> None:
    """Build the full prediction HTML report by adding sections to the ReportBuilder."""
    chambers = [c for c in ["House", "Senate"] if c in results]

    _add_data_summary(report, results, chambers)
    _add_how_to_read(report)

    for chamber in chambers:
        _add_feature_summary(report, results[chamber], chamber)

    for chamber in chambers:
        _add_model_comparison_table(report, results[chamber], chamber)

    for chamber in chambers:
        _add_roc_figure(report, plots_dir, chamber)

    for chamber in chambers:
        _add_confusion_matrix_figure(report, plots_dir, chamber)

    for chamber in chambers:
        _add_calibration_figure(report, plots_dir, chamber)

    for chamber in chambers:
        _add_holdout_table(report, results[chamber], chamber)

    _add_vote_interpretation(report, results, chambers)

    # SHAP
    for chamber in chambers:
        _add_shap_beeswarm_figure(report, plots_dir, chamber)

    for chamber in chambers:
        _add_shap_bar_figure(report, plots_dir, chamber)

    for chamber in chambers:
        _add_feature_importance_figure(report, plots_dir, chamber)

    _add_feature_importance_interpretation(report, results, chambers)

    # Per-legislator
    for chamber in chambers:
        _add_per_legislator_table(report, results[chamber], chamber)

    for chamber in chambers:
        _add_per_legislator_figure(report, plots_dir, chamber)

    for chamber in chambers:
        _add_hardest_legislators_table(report, results[chamber], chamber)

    _add_per_legislator_interpretation(report, results, chambers)

    # Surprising votes
    for chamber in chambers:
        _add_surprising_votes_table(report, results[chamber], chamber)

    # Bill passage
    if not skip_bill_passage:
        for chamber in chambers:
            _add_passage_model_table(report, results[chamber], chamber)

        for chamber in chambers:
            _add_passage_roc_figure(report, plots_dir, chamber)

        for chamber in chambers:
            _add_temporal_split_table(report, results[chamber], chamber)

        for chamber in chambers:
            _add_surprising_bills_table(report, results[chamber], chamber)

        _add_passage_interpretation(report, results, chambers)

    _add_downstream_findings(report, results, chambers)
    _add_parameters_table(report, results, chambers)

    print(f"  Report: {len(report._sections)} sections added")


# ── Section Builders ──────────────────────────────────────────────────────────


def _add_data_summary(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    rows = []
    for chamber in chambers:
        r = results[chamber]
        vf = r["vote_features"]
        rows.append(
            {
                "Chamber": chamber,
                "Vote Observations": vf.height,
                "Legislators": vf["legislator_slug"].n_unique(),
                "Roll Calls": vf["vote_id"].n_unique(),
                "Yea Rate": float(vf["vote_binary"].mean()),
                "Features": len(r["vote_result"]["feature_names"]),
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Prediction Data Summary",
        number_formats={"Yea Rate": ".3f"},
    )
    report.add(TableSection(id="data-summary", title="Data Summary", html=html))


def _add_how_to_read(report: ReportBuilder) -> None:
    html = """
    <p><strong>This report presents vote prediction and bill passage prediction results
    for both chambers.</strong></p>
    <ul>
    <li><strong>Vote Prediction:</strong> Given a legislator and a roll call, predict Yea/Nay.
    Three models compared: Logistic Regression (linear baseline), XGBoost (primary), Random
    Forest (non-boosted comparison).</li>
    <li><strong>AUC-ROC</strong> is the primary metric
    (82% Yea base rate makes accuracy misleading).
    AUC measures how well the model ranks Yea votes above Nay votes.</li>
    <li><strong>SHAP values</strong> explain feature importance: positive SHAP pushes toward Yea,
    negative pushes toward Nay. The beeswarm plot shows the distribution of SHAP values across
    all predictions.</li>
    <li><strong>Per-legislator accuracy</strong> identifies who the model struggles with.
    Legislators near the center (moderate IRT) or with low party loyalty are typically hardest.</li>
    <li><strong>Surprising votes</strong> are high-confidence wrong predictions — the most
    analytically interesting observations.</li>
    <li><strong>Bill Passage</strong> predicts whether a roll call passes, using bill-level features
    only. Smaller sample (~250-500 rows per chamber) yields wider confidence intervals.</li>
    </ul>
    """
    report.add(TextSection(id="how-to-read", title="How to Read This Report", html=html))


def _add_feature_summary(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    feature_names = result["vote_result"]["feature_names"]
    feature_sources = {
        "party_binary": "legislators.csv",
        "xi_mean": "IRT ideal points",
        "xi_sd": "IRT ideal points",
        "loyalty_rate": "Clustering",
        "PC1": "PCA",
        "PC2": "PCA",
        "betweenness": "Network centrality",
        "eigenvector": "Network centrality",
        "pagerank": "Network centrality",
        "alpha_mean": "IRT bill params",
        "beta_mean": "IRT bill params",
        "is_veto_override": "IRT bill params",
        "day_of_session": "Computed",
        "xi_x_beta": "Interaction (xi × beta)",
    }

    rows = []
    for name in feature_names:
        source = feature_sources.get(name, "Vote type (one-hot)")
        rows.append({"Feature": name, "Source": source})

    df = pl.DataFrame(rows)
    html = make_gt(df, title=f"{chamber} — Features ({len(feature_names)} total)")
    report.add(
        TableSection(
            id=f"features-{chamber.lower()}",
            title=f"{chamber} Feature Summary",
            html=html,
        )
    )


def _add_model_comparison_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    cv = result["vote_result"]["cv_results"]
    cv_df = pl.DataFrame(cv)

    rows = []
    for name in ["Logistic Regression", "XGBoost", "Random Forest"]:
        row = {"Model": name}
        for metric in ["accuracy", "auc", "precision", "recall", "f1"]:
            key = f"{name}_{metric}"
            vals = cv_df[key].to_numpy()
            row[f"{metric.upper()} Mean"] = float(np.mean(vals))
            row[f"{metric.upper()} Std"] = float(np.std(vals))
        rows.append(row)

    df = pl.DataFrame(rows)
    number_fmts = {}
    for col in df.columns:
        if col != "Model":
            number_fmts[col] = ".4f"

    html = make_gt(
        df,
        title=f"{chamber} — Vote Prediction: 5-Fold CV Results",
        number_formats=number_fmts,
        source_note=(
            f"Baselines — Majority class: "
            f"{result['vote_result']['baselines']['majority_class_acc']:.3f}"
            f", Party-only: "
            f"{result['vote_result']['baselines']['party_only_acc']:.3f}"
        ),
    )
    report.add(
        TableSection(
            id=f"model-comparison-{chamber.lower()}",
            title=f"{chamber} Model Comparison — Vote Prediction",
            html=html,
        )
    )


def _add_roc_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"roc_curves_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"roc-{chamber.lower()}",
                f"{chamber} ROC Curves — Vote Prediction",
                path,
            )
        )


def _add_confusion_matrix_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"confusion_matrix_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"confusion-{chamber.lower()}",
                f"{chamber} Confusion Matrix — XGBoost",
                path,
            )
        )


def _add_calibration_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"calibration_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"calibration-{chamber.lower()}",
                f"{chamber} Calibration Curve — XGBoost",
                path,
            )
        )


def _add_holdout_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    holdout = result.get("holdout")
    if holdout is None:
        return

    number_fmts = {c: ".4f" for c in holdout.columns if c != "model"}
    html = make_gt(
        holdout,
        title=f"{chamber} — Vote Prediction: Holdout Test Results",
        number_formats=number_fmts,
    )
    report.add(
        TableSection(
            id=f"holdout-{chamber.lower()}",
            title=f"{chamber} Holdout Results",
            html=html,
        )
    )


def _add_vote_interpretation(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    parts = ["<p><strong>Vote Prediction Interpretation:</strong></p><ul>"]

    for chamber in chambers:
        r = results[chamber]
        baselines = r["vote_result"]["baselines"]
        cv = r["vote_result"]["cv_results"]
        cv_df = pl.DataFrame(cv)
        xgb_auc = float(cv_df["XGBoost_auc"].mean())
        xgb_acc = float(cv_df["XGBoost_accuracy"].mean())

        parts.append(
            f"<li><strong>{chamber}:</strong> XGBoost achieves AUC={xgb_auc:.3f} and "
            f"accuracy={xgb_acc:.3f} vs majority baseline {baselines['majority_class_acc']:.3f} "
            f"and party-only baseline {baselines['party_only_acc']:.3f}.</li>"
        )

    parts.append("</ul>")
    parts.append(
        "<p>The large gap between party-only and XGBoost demonstrates that IRT ideal points, "
        "bill parameters, and other features capture significant predictive signal beyond party "
        "affiliation alone. The high AUC indicates strong discrimination between Yea and Nay "
        "votes.</p>"
    )

    report.add(
        TextSection(
            id="vote-interpretation",
            title="Vote Prediction Interpretation",
            html="\n".join(parts),
        )
    )


def _add_shap_beeswarm_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"shap_beeswarm_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"shap-beeswarm-{chamber.lower()}",
                f"{chamber} SHAP Summary (Beeswarm)",
                path,
                caption=(
                    "Each dot is one prediction. X-axis = SHAP value "
                    "(positive = pushes toward Yea). Color = feature value."
                ),
            )
        )


def _add_shap_bar_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"shap_bar_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"shap-bar-{chamber.lower()}",
                f"{chamber} SHAP Feature Importance (Bar)",
                path,
                caption=(
                    "Mean absolute SHAP value per feature. "
                    "Higher = more influential on predictions."
                ),
            )
        )


def _add_feature_importance_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"feature_importance_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"feature-importance-{chamber.lower()}",
                f"{chamber} XGBoost Native Feature Importance",
                path,
                caption=(
                    "XGBoost gain-based feature importance. "
                    "Complementary to SHAP (gain = split quality, "
                    "SHAP = prediction impact)."
                ),
            )
        )


def _add_feature_importance_interpretation(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    html = """
    <p><strong>Feature Importance Interpretation:</strong></p>
    <ul>
    <li><strong>xi_x_beta (ideal point × discrimination)</strong> is expected to be the top
    feature — this is the core IRT interaction that determines vote probability.</li>
    <li><strong>beta_mean (bill discrimination)</strong> captures how much a bill separates
    legislators ideologically. High-discrimination bills are more predictable.</li>
    <li><strong>xi_mean (ideal point)</strong> captures the legislator's ideological position.
    Combined with beta, it determines whether a legislator is on the "Yea" or "Nay" side.</li>
    <li><strong>alpha_mean (bill difficulty)</strong> captures how easy or hard a bill is to
    pass. Easy bills (negative alpha) have more predictable Yea votes.</li>
    <li><strong>loyalty_rate</strong> captures party discipline. Low-loyalty legislators
    (Tyson, Schreiber) are harder to predict.</li>
    <li>If <strong>party_binary</strong> ranks below IRT features, it confirms that IRT
    subsumes the party signal — the model has learned a more nuanced representation than
    simple party affiliation.</li>
    </ul>
    """
    report.add(
        TextSection(
            id="feature-interpretation",
            title="Feature Importance Interpretation",
            html=html,
        )
    )


def _add_per_legislator_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    leg_acc = result.get("leg_accuracy")
    if leg_acc is None or leg_acc.height == 0:
        return

    # Select display columns (exclude list columns that don't render in tables)
    display_cols = [
        "legislator_slug",
        "full_name",
        "party",
        "xi_mean",
        "n_votes",
        "n_correct",
        "accuracy",
    ]
    display_df = leg_acc.select([c for c in display_cols if c in leg_acc.columns]).sort("accuracy")

    html = make_gt(
        display_df,
        title=f"{chamber} — Per-Legislator Prediction Accuracy ({display_df.height} legislators)",
        number_formats={"xi_mean": ".3f", "accuracy": ".3f"},
        source_note="Sorted by accuracy (ascending). All legislators shown.",
    )
    report.add(
        TableSection(
            id=f"per-legislator-{chamber.lower()}",
            title=f"{chamber} Per-Legislator Accuracy",
            html=html,
        )
    )


def _add_per_legislator_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"per_legislator_accuracy_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"per-leg-fig-{chamber.lower()}",
                f"{chamber} Per-Legislator Accuracy vs IRT",
                path,
                caption=(
                    "Each dot is one legislator. X = IRT ideal point, Y = prediction accuracy."
                ),
            )
        )


def _add_hardest_legislators_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    leg_acc = result.get("leg_accuracy")
    if leg_acc is None or leg_acc.height == 0:
        return

    bottom15 = leg_acc.sort("accuracy").head(15)
    display_cols = [
        "legislator_slug",
        "full_name",
        "party",
        "xi_mean",
        "n_votes",
        "n_correct",
        "accuracy",
    ]
    display_df = bottom15.select([c for c in display_cols if c in bottom15.columns])

    html = make_gt(
        display_df,
        title=f"{chamber} — Hardest-to-Predict Legislators (Bottom 15)",
        number_formats={"xi_mean": ".3f", "accuracy": ".3f"},
    )
    report.add(
        TableSection(
            id=f"hardest-legislators-{chamber.lower()}",
            title=f"{chamber} Hardest-to-Predict Legislators",
            html=html,
        )
    )


def _add_per_legislator_interpretation(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    parts = ["<p><strong>Per-Legislator Interpretation:</strong></p><ul>"]
    parts.append(
        "<li>Legislators with <strong>low accuracy</strong> tend to fall into two categories: "
        "(1) moderates near the center of the ideological spectrum whose votes are genuinely "
        "uncertain, and (2) contrarians like Tyson who vote against their party on routine "
        "bills despite being ideologically extreme.</li>"
    )
    parts.append(
        "<li>The <strong>IRT × accuracy</strong> scatter plot should show a U-shape: legislators "
        "at the extremes (far left or far right) are easiest to predict, while those near the "
        "center are hardest.</li>"
    )
    parts.append(
        "<li>Legislators flagged in upstream phases (Tyson, Thompson, Schreiber, Miller) are "
        "expected to appear in the hardest-to-predict list.</li>"
    )
    parts.append("</ul>")

    report.add(
        TextSection(
            id="per-legislator-interpretation",
            title="Per-Legislator Interpretation",
            html="\n".join(parts),
        )
    )


def _add_surprising_votes_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    surprising = result.get("surprising_votes")
    if surprising is None or surprising.height == 0:
        return

    display_cols = [
        "full_name",
        "party",
        "bill_number",
        "motion",
        "actual",
        "predicted",
        "y_prob",
        "confidence_error",
    ]
    display_df = surprising.select([c for c in display_cols if c in surprising.columns])

    html = make_gt(
        display_df,
        title=f"{chamber} — Most Surprising Votes (Top {display_df.height})",
        column_labels={
            "full_name": "Legislator",
            "y_prob": "P(Yea)",
            "confidence_error": "|Error|",
        },
        number_formats={"y_prob": ".3f", "confidence_error": ".3f"},
        source_note="Votes where the model was most confident but wrong. actual: 1=Yea, 0=Nay.",
    )
    report.add(
        TableSection(
            id=f"surprising-votes-{chamber.lower()}",
            title=f"{chamber} Surprising Votes",
            html=html,
        )
    )


def _add_passage_model_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    passage_cv = result.get("passage_cv")
    if passage_cv is None or result.get("passage_skipped"):
        return

    rows = []
    for name in ["Logistic Regression", "XGBoost", "Random Forest"]:
        row = {"Model": name}
        for metric in ["accuracy", "auc", "precision", "recall", "f1"]:
            key = f"{name}_{metric}"
            if key in passage_cv.columns:
                vals = passage_cv[key].drop_nulls().drop_nans().to_numpy()
                row[f"{metric.upper()} Mean"] = float(np.mean(vals)) if len(vals) > 0 else None
                row[f"{metric.upper()} Std"] = float(np.std(vals)) if len(vals) > 0 else None
        rows.append(row)

    df = pl.DataFrame(rows)
    number_fmts = {c: ".4f" for c in df.columns if c != "Model"}

    html = make_gt(
        df,
        title=f"{chamber} — Bill Passage: CV Results",
        number_formats=number_fmts,
    )
    report.add(
        TableSection(
            id=f"passage-model-{chamber.lower()}",
            title=f"{chamber} Model Comparison — Bill Passage",
            html=html,
        )
    )


def _add_passage_roc_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"passage_roc_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"passage-roc-{chamber.lower()}",
                f"{chamber} Passage ROC Curves",
                path,
            )
        )


def _add_temporal_split_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    temporal = result.get("temporal_results")
    if not temporal:
        return

    df = pl.DataFrame(temporal)
    number_fmts = {c: ".4f" for c in df.columns if c not in {"model", "train_size", "test_size"}}

    html = make_gt(
        df,
        title=f"{chamber} — Bill Passage: Temporal Split Results",
        number_formats=number_fmts,
        source_note="Train on first 70% of session chronologically, test on last 30%.",
    )
    report.add(
        TableSection(
            id=f"temporal-{chamber.lower()}",
            title=f"{chamber} Temporal Split Results",
            html=html,
        )
    )


def _add_surprising_bills_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    surprising = result.get("surprising_bills")
    if surprising is None or surprising.height == 0:
        return

    display_cols = [
        "bill_number",
        "motion",
        "vote_type",
        "passed_binary",
        "y_prob",
        "confidence_error",
        "yea_count",
        "nay_count",
    ]
    display_df = surprising.select([c for c in display_cols if c in surprising.columns])

    html = make_gt(
        display_df,
        title=f"{chamber} — Most Surprising Bills (Top {display_df.height})",
        column_labels={
            "passed_binary": "Passed",
            "y_prob": "P(Pass)",
            "confidence_error": "|Error|",
        },
        number_formats={"y_prob": ".3f", "confidence_error": ".3f"},
        source_note="Bills where the passage model was most confident but wrong.",
    )
    report.add(
        TableSection(
            id=f"surprising-bills-{chamber.lower()}",
            title=f"{chamber} Surprising Bills",
            html=html,
        )
    )


def _add_passage_interpretation(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    html = """
    <p><strong>Bill Passage Interpretation:</strong></p>
    <ul>
    <li>Bill passage prediction operates on a much smaller dataset (~250-500 roll calls per
    chamber) compared to vote prediction (~30K-60K observations). This means wider confidence
    intervals and more overfitting risk.</li>
    <li>The <strong>temporal split</strong> tests whether early-session voting patterns generalize
    to late-session votes. A large drop in temporal vs random CV suggests the legislature's
    behavior shifts over the session (e.g., more contentious votes near deadlines).</li>
    <li><strong>Surprising bills</strong> are bills the model was confident would pass/fail but
    didn't — these may represent unexpected coalitions or procedural surprises.</li>
    <li>Expected performance: bill passage AUC will be lower than vote AUC because we lack
    legislator-level features at the bill level. The model sees bill characteristics but not
    who will vote on it.</li>
    </ul>
    """
    report.add(
        TextSection(
            id="passage-interpretation",
            title="Bill Passage Interpretation",
            html=html,
        )
    )


def _add_downstream_findings(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    parts = ["<p><strong>Downstream Findings and Implications:</strong></p><ul>"]
    parts.append(
        "<li><strong>Cluster labels confirmed redundant:</strong> As predicted by the clustering "
        "phase (k=2 = party), cluster labels were excluded from features. The model achieves "
        "strong performance using party as a binary feature combined with continuous metrics "
        "(IRT, loyalty, centrality).</li>"
    )
    parts.append(
        "<li><strong>IRT features dominate:</strong> If xi_x_beta and beta_mean rank as top "
        "SHAP features, this validates the IRT model as the strongest single source of "
        "predictive signal.</li>"
    )
    parts.append(
        "<li><strong>Per-legislator accuracy patterns:</strong> Legislators flagged in upstream "
        "phases (Tyson as contrarian, Schreiber as most bipartisan, Miller as sparse) should "
        "appear in the hardest-to-predict list, validating those flags.</li>"
    )
    parts.append(
        "<li><strong>Model limitations:</strong> The 1D IRT model compresses Tyson's "
        "two-dimensional behavior (extreme on partisan votes, contrarian on routine votes) "
        "into a single axis. A 2D model or issue-specific features could improve predictions "
        "for legislators like Tyson and Thompson.</li>"
    )
    parts.append("</ul>")

    report.add(
        TextSection(
            id="downstream-findings",
            title="Downstream Findings",
            html="\n".join(parts),
        )
    )


def _add_parameters_table(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    try:
        from analysis.prediction import (
            MIN_VOTES,
            MINORITY_THRESHOLD,
            N_ESTIMATORS_RF,
            N_ESTIMATORS_XGB,
            N_SPLITS,
            RANDOM_SEED,
            TEST_SIZE,
            TOP_SHAP_FEATURES,
            TOP_SURPRISING_N,
            XGB_LEARNING_RATE,
            XGB_MAX_DEPTH,
        )
    except ModuleNotFoundError:
        from prediction import (  # type: ignore[no-redef]
            MIN_VOTES,
            MINORITY_THRESHOLD,
            N_ESTIMATORS_RF,
            N_ESTIMATORS_XGB,
            N_SPLITS,
            RANDOM_SEED,
            TEST_SIZE,
            TOP_SHAP_FEATURES,
            TOP_SURPRISING_N,
            XGB_LEARNING_RATE,
            XGB_MAX_DEPTH,
        )

    rows = [
        {"Parameter": "RANDOM_SEED", "Value": str(RANDOM_SEED)},
        {"Parameter": "N_SPLITS", "Value": str(N_SPLITS)},
        {"Parameter": "TEST_SIZE", "Value": str(TEST_SIZE)},
        {"Parameter": "N_ESTIMATORS_XGB", "Value": str(N_ESTIMATORS_XGB)},
        {"Parameter": "N_ESTIMATORS_RF", "Value": str(N_ESTIMATORS_RF)},
        {"Parameter": "XGB_MAX_DEPTH", "Value": str(XGB_MAX_DEPTH)},
        {"Parameter": "XGB_LEARNING_RATE", "Value": str(XGB_LEARNING_RATE)},
        {"Parameter": "TOP_SHAP_FEATURES", "Value": str(TOP_SHAP_FEATURES)},
        {"Parameter": "TOP_SURPRISING_N", "Value": str(TOP_SURPRISING_N)},
        {"Parameter": "MINORITY_THRESHOLD", "Value": str(MINORITY_THRESHOLD)},
        {"Parameter": "MIN_VOTES", "Value": str(MIN_VOTES)},
    ]

    df = pl.DataFrame(rows)
    html = make_gt(df, title="Analysis Parameters")
    report.add(
        TableSection(
            id="parameters",
            title="Analysis Parameters",
            html=html,
        )
    )
