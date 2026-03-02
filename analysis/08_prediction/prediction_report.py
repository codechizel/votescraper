"""Prediction-specific HTML report builder.

Builds ~25 sections (tables, figures, and text) for the prediction analysis report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from prediction.py):
    from analysis.prediction_report import build_prediction_report
    build_prediction_report(ctx.report, results=results, ...)
"""

from pathlib import Path

import numpy as np
import polars as pl

try:
    from analysis.report import (
        FigureSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
    )
except ModuleNotFoundError:
    from report import (  # type: ignore[no-redef]
        FigureSection,
        KeyFindingsSection,
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
    topic_models: dict[str, object] | None = None,
) -> None:
    """Build the full prediction HTML report by adding sections to the ReportBuilder."""
    chambers = [c for c in ["House", "Senate"] if c in results]

    findings = _generate_prediction_key_findings(results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

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

    _add_feature_importance_interpretation(report)

    # Per-legislator
    for chamber in chambers:
        _add_per_legislator_table(report, results[chamber], chamber)

    for chamber in chambers:
        _add_per_legislator_figure(report, plots_dir, chamber)

    for chamber in chambers:
        _add_hardest_to_predict_figure(report, plots_dir, chamber)

    for chamber in chambers:
        _add_hardest_legislators_table(report, results[chamber], chamber)

    _add_per_legislator_interpretation(report)

    # Surprising votes
    for chamber in chambers:
        _add_surprising_votes_table(report, results[chamber], chamber)
    _add_surprising_votes_interpretation(report, results)

    # Bill passage
    if not skip_bill_passage:
        for chamber in chambers:
            _add_passage_model_table(report, results[chamber], chamber)

        for chamber in chambers:
            _add_passage_roc_figure(report, plots_dir, chamber)

        for chamber in chambers:
            _add_passage_shap_beeswarm_figure(report, plots_dir, chamber)

        for chamber in chambers:
            _add_passage_shap_bar_figure(report, plots_dir, chamber)

        for chamber in chambers:
            _add_passage_feature_importance_figure(report, plots_dir, chamber)

        for chamber in chambers:
            _add_stratified_accuracy_table(report, results[chamber], chamber)

        # NLP topic features sections
        if topic_models:
            _add_nlp_interpretation(report)
            for chamber in chambers:
                if chamber in topic_models:
                    _add_topic_summary_table(report, topic_models[chamber], chamber)
            for chamber in chambers:
                _add_topic_words_figure(report, plots_dir, chamber)

        for chamber in chambers:
            _add_temporal_split_table(report, results[chamber], chamber)

        for chamber in chambers:
            _add_surprising_bills_table(report, results[chamber], chamber)

        _add_passage_interpretation(report, has_topics=topic_models is not None)

    _add_downstream_findings(report)
    _add_parameters_table(report)

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
        if name.startswith("topic_"):
            source = "NLP (bill title topics)"
        else:
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
        for metric in ["accuracy", "auc", "precision", "recall", "f1", "brier", "logloss"]:
            key = f"{name}_{metric}"
            if key in cv_df.columns:
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
            f". Brier: lower is better (0=perfect). Log-loss: lower is better."
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
                alt_text=(
                    f"ROC curves comparing Logistic Regression, XGBoost, and Random Forest "
                    f"for {chamber} vote prediction. All models well above the diagonal baseline."
                ),
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
                alt_text=(
                    f"Confusion matrix heatmap for {chamber} XGBoost vote predictions. "
                    f"Shows counts of true positives, false positives, true negatives, "
                    f"and false negatives."
                ),
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
                alt_text=(
                    f"Calibration curve for {chamber} XGBoost model. Compares predicted "
                    f"probabilities to observed frequencies against the perfectly "
                    f"calibrated diagonal."
                ),
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
    parts.append(
        '<p class="caveat"><strong>Methodological note:</strong> IRT features (ideal points, '
        "bill parameters) are estimated from the same vote matrix used for prediction. "
        "The high AUC therefore reflects <em>explanatory power</em> (how well these features "
        "describe voting patterns) rather than true out-of-sample predictive accuracy. "
        "For genuine prediction of future votes, ideal points would need to be estimated "
        "from prior sessions only. Per-legislator accuracy and surprising votes are evaluated "
        "on the 20% holdout set to mitigate in-sample bias.</p>"
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
                alt_text=(
                    f"Beeswarm plot of SHAP values for {chamber} vote predictions. "
                    f"Each dot is one prediction colored by feature value, showing how "
                    f"each feature pushes toward Yea or Nay."
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
                alt_text=(
                    f"Horizontal bar chart of mean absolute SHAP values for {chamber}. "
                    f"Features ranked by importance, with IRT interaction typically dominant."
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
                alt_text=(
                    f"Bar chart of XGBoost gain-based feature importance for {chamber}. "
                    f"Ranks features by how much they improve split quality in the model."
                ),
            )
        )


def _add_feature_importance_interpretation(
    report: ReportBuilder,
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
    are harder to predict.</li>
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
                alt_text=(
                    f"Scatter plot of per-legislator prediction accuracy vs IRT ideal point "
                    f"for {chamber}. Moderates near the center tend to have lower accuracy."
                ),
            )
        )


def _add_hardest_to_predict_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"hardest_to_predict_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"hardest-to-predict-{chamber.lower()}",
                f"{chamber} Hardest-to-Predict Legislators (Spotlight)",
                path,
                caption=(
                    "Legislators the model struggles with most, "
                    "with a plain-English explanation for each."
                ),
                alt_text=(
                    f"Annotated chart spotlighting the hardest-to-predict {chamber} "
                    f"legislators with explanations for why each is difficult to model."
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
) -> None:
    parts = ["<p><strong>Per-Legislator Interpretation:</strong></p><ul>"]
    parts.append(
        "<li>Legislators with <strong>low accuracy</strong> tend to fall into two categories: "
        "(1) moderates near the center of the ideological spectrum whose votes are genuinely "
        "uncertain, and (2) contrarians who vote against their party on routine "
        "bills despite being ideologically extreme.</li>"
    )
    parts.append(
        "<li>The <strong>IRT × accuracy</strong> scatter plot should show a U-shape: legislators "
        "at the extremes (far left or far right) are easiest to predict, while those near the "
        "center are hardest.</li>"
    )
    parts.append(
        "<li>Legislators flagged in upstream phases (contrarians, bipartisan bridge-builders, "
        "sparse-vote members) are expected to appear in the hardest-to-predict list.</li>"
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

    # Show split tables when available
    surprising_nay = result.get("surprising_nay")
    surprising_yea = result.get("surprising_yea")

    if surprising_nay is not None and surprising_nay.height > 0:
        nay_df = surprising_nay.select([c for c in display_cols if c in surprising_nay.columns])
        html_nay = make_gt(
            nay_df,
            title=f"{chamber} — Surprising Nay Votes (Top {nay_df.height})",
            subtitle="Predicted Yea, actual Nay — unexpected dissent",
            column_labels={
                "full_name": "Legislator",
                "y_prob": "P(Yea)",
                "confidence_error": "|Error|",
            },
            number_formats={"y_prob": ".3f", "confidence_error": ".3f"},
            source_note=(
                "False positives: model predicted Yea with high confidence but legislator "
                "voted Nay. These are genuine surprises — dissent against expectations."
            ),
        )
        report.add(
            TableSection(
                id=f"surprising-nay-{chamber.lower()}",
                title=f"{chamber} Surprising Nay Votes",
                html=html_nay,
            )
        )

    if surprising_yea is not None and surprising_yea.height > 0:
        yea_df = surprising_yea.select([c for c in display_cols if c in surprising_yea.columns])
        html_yea = make_gt(
            yea_df,
            title=f"{chamber} — Surprising Yea Votes (Top {yea_df.height})",
            subtitle="Predicted Nay, actual Yea — unexpected support",
            column_labels={
                "full_name": "Legislator",
                "y_prob": "P(Yea)",
                "confidence_error": "|Error|",
            },
            number_formats={"y_prob": ".3f", "confidence_error": ".3f"},
            source_note=(
                "False negatives: model predicted Nay with high confidence but legislator "
                "voted Yea. Rarer than surprising Nay due to high Yea base rate."
            ),
        )
        report.add(
            TableSection(
                id=f"surprising-yea-{chamber.lower()}",
                title=f"{chamber} Surprising Yea Votes",
                html=html_yea,
            )
        )

    # Always show the combined table as well
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
            title=f"{chamber} Surprising Votes (Combined)",
            html=html,
        )
    )


def _add_surprising_votes_interpretation(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Text block: Interpret surprising votes with base-rate context."""
    # Compute FP/FN stats across chambers
    parts = []
    for chamber, result in results.items():
        surprising = result.get("surprising_votes")
        if surprising is None or surprising.height == 0:
            continue
        if "actual" not in surprising.columns or "predicted" not in surprising.columns:
            continue
        n_fp = surprising.filter((pl.col("predicted") == 1) & (pl.col("actual") == 0)).height
        n_fn = surprising.filter((pl.col("predicted") == 0) & (pl.col("actual") == 1)).height
        total = surprising.height
        fp_pct = n_fp / total * 100 if total > 0 else 0
        parts.append(
            f"<li><strong>{chamber}:</strong> {n_fp} surprising Nay ({fp_pct:.0f}%) vs "
            f"{n_fn} surprising Yea</li>"
        )

    if not parts:
        return

    report.add(
        TextSection(
            id="surprising-votes-interpretation",
            title="Interpreting Surprising Votes",
            html=(
                "<p>Surprising votes are high-confidence errors — cases where the model was "
                "most confident but wrong. The split between error types reveals a base-rate "
                "effect:</p>"
                "<ul>" + "".join(parts) + "</ul>"
                "<p><strong>Why the imbalance?</strong> With a ~73% Yea base rate, the model "
                "predicts Yea on most votes. When it is wrong, the error is almost always "
                "a false positive (predicted Yea, actual Nay). This is not a model flaw — it "
                "reflects the legislature's strong tendency to pass bills.</p>"
                "<p><strong>How to read the tables:</strong></p>"
                "<ul>"
                "<li><strong>Surprising Nay</strong> (predicted Yea, actual Nay): Unexpected "
                "dissent. A legislator the model expected to support a bill voted against it. "
                "These are genuine surprises worth investigating.</li>"
                "<li><strong>Surprising Yea</strong> (predicted Nay, actual Yea): Unexpected "
                "support. A legislator the model expected to oppose a bill voted for it. "
                "Rarer but often more politically interesting.</li>"
                "</ul>"
            ),
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
        for metric in ["accuracy", "auc", "precision", "recall", "f1", "brier", "logloss"]:
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
        source_note="Brier: lower is better (0=perfect). Log-loss: lower is better.",
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
                alt_text=(
                    f"ROC curves for {chamber} bill passage prediction models. "
                    f"Compares three classifiers against the diagonal chance baseline."
                ),
            )
        )


def _add_passage_shap_beeswarm_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"shap_passage_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"passage-shap-beeswarm-{chamber.lower()}",
                f"{chamber} Passage SHAP Beeswarm",
                path,
                alt_text=(
                    f"SHAP beeswarm plot for {chamber} bill passage prediction. "
                    f"Each dot is a bill; position shows feature impact on passage probability."
                ),
            )
        )


def _add_passage_shap_bar_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"shap_bar_passage_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"passage-shap-bar-{chamber.lower()}",
                f"{chamber} Passage Feature Importance (SHAP)",
                path,
                alt_text=(
                    f"Mean absolute SHAP values for {chamber} bill passage features. "
                    f"Longer bars indicate features with greater influence on passage prediction."
                ),
            )
        )


def _add_passage_feature_importance_figure(
    report: ReportBuilder, plots_dir: Path, chamber: str
) -> None:
    path = plots_dir / f"passage_importance_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"passage-importance-{chamber.lower()}",
                f"{chamber} Passage Feature Importance (XGBoost Gain)",
                path,
                alt_text=(
                    f"XGBoost native feature importance (gain) for {chamber} bill passage. "
                    f"Shows which features contribute most to the passage model's splits."
                ),
            )
        )


def _add_stratified_accuracy_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    stratified = result.get("stratified_accuracy")
    if stratified is None or stratified.height == 0:
        return

    number_fmts = {"accuracy": ".3f", "passage_rate": ".3f"}
    html = make_gt(
        stratified,
        title=f"{chamber} — Passage Accuracy by Bill Prefix",
        column_labels={
            "prefix": "Bill Prefix",
            "count": "Count",
            "accuracy": "Accuracy",
            "passage_rate": "Passage Rate",
        },
        number_formats=number_fmts,
        source_note="Accuracy on holdout test set, stratified by bill type prefix (HB, SB, etc.).",
    )
    report.add(
        TableSection(
            id=f"stratified-accuracy-{chamber.lower()}",
            title=f"{chamber} Passage Accuracy by Bill Prefix",
            html=html,
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


def _add_nlp_interpretation(report: ReportBuilder) -> None:
    html = """
    <p><strong>Bill Title Topic Modeling (NLP):</strong></p>
    <p>We applied NMF (Non-negative Matrix Factorization) topic modeling to the KLISS API
    <code>short_title</code> field to extract subject-matter signal from bill titles.
    This is the same text visible on the legislature's website — it captures what each bill
    is <em>about</em> (taxes, elections, healthcare, etc.).</p>
    <ul>
    <li><strong>Method:</strong> TF-IDF vectorization (unigrams + bigrams, max 500 features)
    followed by NMF with K=6 topics. NMF was chosen over LDA because it is deterministic,
    faster, and produces non-negative topic weights suitable as model features.</li>
    <li><strong>Not target leakage:</strong> Bill titles are known before any votes are cast —
    they are pre-vote public information, analogous to the bill prefix (HB, SB) already in the
    model.</li>
    <li><strong>Scope:</strong> Topic features are added to the bill passage model only (not
    the individual vote model, which already achieves AUC=0.98).</li>
    </ul>
    """
    report.add(
        TextSection(
            id="nlp-interpretation",
            title="Bill Title Topic Modeling (NLP)",
            html=html,
        )
    )


def _add_topic_summary_table(report: ReportBuilder, topic_model: object, chamber: str) -> None:
    rows = []
    for i in range(topic_model.n_topics):
        col = f"topic_{i}"
        words = topic_model.topic_top_words.get(col, [])
        rows.append(
            {
                "Topic": topic_model.topic_labels[i],
                "Top Words": ", ".join(words),
            }
        )
    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} — NMF Topic Summary (K={topic_model.n_topics})",
        source_note=f"Vocabulary: {topic_model.vocabulary_size} terms, "
        f"Documents: {topic_model.n_documents} bill titles.",
    )
    report.add(
        TableSection(
            id=f"topic-summary-{chamber.lower()}",
            title=f"{chamber} Topic Summary",
            html=html,
        )
    )


def _add_topic_words_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"topic_words_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"topic-words-{chamber.lower()}",
                f"{chamber} NMF Topic Top Words",
                path,
                caption="Top words per NMF topic extracted from bill short_title text.",
                alt_text=(
                    f"Bar chart of top words per NMF topic for {chamber} bill titles. "
                    f"Each topic cluster reveals a policy area such as taxes or elections."
                ),
            )
        )


def _add_passage_interpretation(
    report: ReportBuilder,
    has_topics: bool = False,
) -> None:
    parts = [
        "<p><strong>Bill Passage Interpretation:</strong></p>",
        "<ul>",
        "<li>Bill passage prediction operates on a much smaller dataset (~250-500 roll calls per "
        "chamber) compared to vote prediction (~30K-60K observations). This means wider confidence "
        "intervals and more overfitting risk.</li>",
        "<li>The <strong>temporal split</strong> tests whether early-session voting patterns "
        "generalize to late-session votes. A large drop in temporal vs random CV suggests the "
        "legislature's behavior shifts over the session (e.g., more contentious votes near "
        "deadlines).</li>",
        "<li><strong>Surprising bills</strong> are bills the model was confident would pass/fail "
        "but didn't — these may represent unexpected coalitions or procedural surprises.</li>",
        "<li>Expected performance: bill passage AUC will be lower than vote AUC because we lack "
        "legislator-level features at the bill level. The model sees bill characteristics but not "
        "who will vote on it.</li>",
        "<li><strong>Sponsor party</strong> indicates whether the bill's primary sponsor is a "
        "Republican. In a Republican supermajority, Republican-sponsored bills may have higher "
        "passage rates, making this a useful structural feature.</li>",
        "<li><strong>Stratified accuracy by bill prefix</strong> (HB vs SB) reveals whether the "
        "model performs differently on House bills vs Senate bills. Different passage rates and "
        "procedural paths can create systematic prediction gaps.</li>",
    ]
    if has_topics:
        parts.append(
            "<li><strong>Bill title topic features</strong> (NMF on TF-IDF) capture "
            "subject-matter signal — e.g., tax bills, election bills, or healthcare bills may "
            "have systematically different passage rates. These complement the structural features "
            "(partisanship, vote type, bill prefix) already in the model.</li>"
        )
    parts.append("</ul>")
    html = "\n".join(parts)
    report.add(
        TextSection(
            id="passage-interpretation",
            title="Bill Passage Interpretation",
            html=html,
        )
    )


def _generate_prediction_key_findings(results: dict[str, dict]) -> list[str]:
    """Generate 2-4 key findings from prediction results."""
    findings: list[str] = []

    for chamber in ["House", "Senate"]:
        if chamber not in results:
            continue
        result = results[chamber]

        # Best model accuracy and AUC
        models = result.get("model_comparison")
        if models is not None and hasattr(models, "height") and models.height > 0:
            best = models.sort("accuracy", descending=True).head(1)
            model_name = best["model"][0]
            acc = float(best["accuracy"][0])
            auc = float(best["auc_roc"][0]) if "auc_roc" in best.columns else None
            detail = f", AUC-ROC = {auc:.3f}" if auc is not None else ""
            findings.append(
                f"{chamber} best model: <strong>{model_name}</strong> "
                f"(accuracy = {acc:.1%}{detail})."
            )
        elif result.get("vote_model"):
            vm = result["vote_model"]
            acc = vm.get("accuracy")
            auc = vm.get("auc_roc")
            if acc is not None:
                detail = f", AUC-ROC = {auc:.3f}" if auc is not None else ""
                findings.append(
                    f"{chamber} XGBoost vote prediction: <strong>{acc:.1%}</strong> "
                    f"accuracy{detail}."
                )

        # Top SHAP feature
        shap_data = result.get("shap_importance")
        if shap_data is not None and hasattr(shap_data, "height") and shap_data.height > 0:
            top_feature = shap_data.sort("mean_abs_shap", descending=True).head(1)
            feat_name = top_feature["feature"][0]
            findings.append(f"{chamber} top predictive feature: <strong>{feat_name}</strong>.")

        break  # First chamber only

    return findings


def _add_downstream_findings(
    report: ReportBuilder,
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
        "phases (contrarians, bipartisan bridge-builders, sparse-vote members) should "
        "appear in the hardest-to-predict list, validating those flags.</li>"
    )
    parts.append(
        "<li><strong>Model limitations:</strong> The 1D IRT model compresses multi-dimensional "
        "behavior (extreme on partisan votes, contrarian on routine votes) "
        "into a single axis. A 2D model or issue-specific features could improve predictions "
        "for legislators with paradoxical voting patterns.</li>"
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
) -> None:
    try:
        from analysis.prediction import (
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

    try:
        from analysis.nlp_features import (
            NMF_N_TOPICS,
            NMF_TOP_WORDS,
            TFIDF_MAX_DF,
            TFIDF_MAX_FEATURES,
            TFIDF_MIN_DF,
            TFIDF_NGRAM_RANGE,
        )
    except ModuleNotFoundError:
        from nlp_features import (  # type: ignore[no-redef]
            NMF_N_TOPICS,
            NMF_TOP_WORDS,
            TFIDF_MAX_DF,
            TFIDF_MAX_FEATURES,
            TFIDF_MIN_DF,
            TFIDF_NGRAM_RANGE,
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
        {"Parameter": "NMF_N_TOPICS", "Value": str(NMF_N_TOPICS)},
        {"Parameter": "TFIDF_MAX_DF", "Value": str(TFIDF_MAX_DF)},
        {"Parameter": "TFIDF_MIN_DF", "Value": str(TFIDF_MIN_DF)},
        {"Parameter": "TFIDF_MAX_FEATURES", "Value": str(TFIDF_MAX_FEATURES)},
        {"Parameter": "TFIDF_NGRAM_RANGE", "Value": str(TFIDF_NGRAM_RANGE)},
        {"Parameter": "NMF_TOP_WORDS", "Value": str(NMF_TOP_WORDS)},
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
