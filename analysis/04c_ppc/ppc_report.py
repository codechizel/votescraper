"""HTML report builder for Phase 4c Posterior Predictive Checks.

Produces a self-contained HTML report with ~13 sections covering PPC battery,
item/person fit, Q3 local dependence, LOO-CV model comparison, and Pareto k
diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def build_ppc_report(
    report: ReportBuilder,
    *,
    all_results: dict[str, dict],
    session: str,
    skip_loo: bool,
    skip_q3: bool,
    plots_dir: Path,
) -> None:
    """Build the full PPC + LOO-CV HTML report."""
    findings = _generate_ppc_key_findings(all_results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_how_to_read(report)
    _add_executive_summary(report, all_results, session)

    for ch_name, ch_data in sorted(all_results.items()):
        _add_calibration_plot(report, plots_dir, ch_name)
        _add_item_fit(report, ch_data, plots_dir, ch_name, session)
        _add_person_fit(report, ch_data, plots_dir, ch_name, session)
        _add_margins(report, plots_dir, ch_name)

        if not skip_q3 and ch_data.get("q3_results"):
            _add_q3_summary(report, ch_data, ch_name)

    if not skip_loo:
        _add_loo_comparison(report, all_results, plots_dir, session)
        _add_pareto_k(report, all_results, plots_dir)

    _add_model_ranking(report, all_results, skip_loo, skip_q3)
    _add_methodology(report)

    print(f"  Report: {len(report._sections)} sections added")


# ── Section Builders ────────────────────────────────────────────────────────


def _add_how_to_read(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="how-to-read",
            title="How to Read This Report",
            html="""
<div style="background: #f0f7ff; padding: 1em; border-radius: 6px; margin-bottom: 1em;">
<p><strong>This report checks whether our statistical models reproduce the voting
patterns they were trained on.</strong></p>

<p>Think of it like this: if our model truly captures how legislators vote, then
simulated legislatures should produce voting statistics similar to the real Kansas
Legislature. We check this by simulating 500 "fake" legislatures from each model
and comparing key statistics.</p>

<p><strong>Key metrics:</strong></p>
<ul>
<li><strong>Bayesian p-value</strong>: Should be between 0.1 and 0.9. Values near 0 or 1
mean the model is systematically wrong about that statistic.</li>
<li><strong>GMP</strong> (Geometric Mean Probability): How confident the model is in correct
predictions. Higher is better; above 0.7 is good.</li>
<li><strong>APRE</strong> (Aggregate Proportional Reduction in Error): How much better the
model is than just guessing the most common outcome. Above 0.3 is meaningful.</li>
<li><strong>Misfitting items/persons</strong>: Specific votes or legislators the model
struggles with. Under 5% is normal.</li>
</ul>

<p>If LOO-CV (leave-one-out cross-validation) was computed, the report also ranks
models by out-of-sample predictive accuracy. A model with higher ELPD predicts
better on unseen data.</p>
</div>
""",
        )
    )


def _add_executive_summary(
    report: ReportBuilder,
    all_results: dict[str, dict],
    session: str,
) -> None:
    rows = []
    for ch_name, ch_data in sorted(all_results.items()):
        for model_name, model_data in ch_data["models"].items():
            ppc = model_data["ppc"]
            row: dict[str, Any] = {
                "Chamber": ch_name,
                "Model": model_name,
                "Yea Rate (obs)": ppc["observed_yea_rate"],
                "Yea Rate (rep)": ppc["replicated_yea_rate_mean"],
                "p-value": ppc["bayesian_p_yea_rate"],
                "Accuracy": ppc["mean_accuracy"],
                "GMP": ppc["mean_gmp"],
                "APRE": ppc["apre"],
                "Items Misfit": (
                    f"{model_data['item_fit']['n_misfitting']}/{model_data['item_fit']['n_votes']}"
                ),
                "Persons Misfit": (
                    f"{model_data['person_fit']['n_misfitting']}"
                    f"/{model_data['person_fit']['n_legislators']}"
                ),
            }
            if "loo" in model_data:
                row["ELPD"] = model_data["loo"]["elpd_loo"]
                row["p_loo"] = model_data["loo"]["p_loo"]
            rows.append(row)

    if not rows:
        return

    df = pl.DataFrame(rows)

    # Determine numeric columns for formatting
    num_formats = {
        "Yea Rate (obs)": ".3f",
        "Yea Rate (rep)": ".3f",
        "p-value": ".3f",
        "Accuracy": ".3f",
        "GMP": ".3f",
        "APRE": ".3f",
    }
    if "ELPD" in df.columns:
        num_formats["ELPD"] = ".1f"
        num_formats["p_loo"] = ".1f"

    html = make_gt(
        df,
        title="PPC Executive Summary",
        subtitle=f"Session {session} — All models, both chambers",
        number_formats=num_formats,
        source_note=(
            "p-value: Bayesian posterior p-value for Yea rate calibration. "
            "GMP: Geometric Mean Probability. "
            "APRE: Aggregate Proportional Reduction in Error."
        ),
    )
    report.add(TableSection(id="executive-summary", title="Executive Summary", html=html))


def _add_calibration_plot(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    ch = chamber.lower()
    path = plots_dir / f"calibration_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"calibration-{ch}",
                f"{chamber} — Yea Rate Calibration",
                path,
                caption="Histograms show the distribution of replicated Yea rates (500 draws). "
                "Red line = observed rate. p-value shown in title.",
                alt_text=(
                    f"Histogram of replicated Yea rates from 500 posterior draws for {chamber}. "
                    "Red vertical line marks the observed rate for calibration comparison."
                ),
            )
        )


def _add_item_fit(
    report: ReportBuilder,
    ch_data: dict,
    plots_dir: Path,
    chamber: str,
    session: str,
) -> None:
    ch = chamber.lower()

    # Plot
    path = plots_dir / f"item_fit_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"item-fit-{ch}",
                f"{chamber} — Item Endorsement Rates",
                path,
                caption="Each point is one roll call. Diagonal = perfect prediction. "
                "Red X marks misfitting items (observed outside 95% replicated interval).",
                alt_text=(
                    f"Scatter plot of observed vs predicted item endorsement rates for {chamber}. "
                    "Most points fall near the diagonal; red X marks flag misfitting items."
                ),
            )
        )

    # Summary table
    rows = []
    for model_name, model_data in ch_data["models"].items():
        item = model_data["item_fit"]
        rows.append(
            {
                "Model": model_name,
                "N Votes": item["n_votes"],
                "Misfitting": item["n_misfitting"],
                "Misfit %": item["misfit_pct"],
            }
        )
    if rows:
        df = pl.DataFrame(rows)
        html = make_gt(
            df,
            title=f"{chamber} — Item Fit Summary",
            number_formats={"Misfit %": ".1f"},
            source_note="Misfitting: observed endorsement rate outside 95% replicated interval.",
        )
        report.add(
            TableSection(
                id=f"item-fit-table-{ch}", title=f"{chamber} — Item Fit Summary", html=html
            )
        )


def _add_person_fit(
    report: ReportBuilder,
    ch_data: dict,
    plots_dir: Path,
    chamber: str,
    session: str,
) -> None:
    ch = chamber.lower()

    path = plots_dir / f"person_fit_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"person-fit-{ch}",
                f"{chamber} — Person Total Scores",
                path,
                caption=(
                    "Each point is one legislator. Diagonal = perfect prediction. "
                    "Red X marks misfitting legislators "
                    "(observed outside 95% replicated interval)."
                ),
                alt_text=(
                    f"Scatter plot of observed vs predicted person total scores for {chamber}. "
                    "Most points fall near the diagonal; red X marks flag misfitting legislators."
                ),
            )
        )

    rows = []
    for model_name, model_data in ch_data["models"].items():
        person = model_data["person_fit"]
        rows.append(
            {
                "Model": model_name,
                "N Legislators": person["n_legislators"],
                "Misfitting": person["n_misfitting"],
                "Misfit %": person["misfit_pct"],
            }
        )
    if rows:
        df = pl.DataFrame(rows)
        html = make_gt(
            df,
            title=f"{chamber} — Person Fit Summary",
            number_formats={"Misfit %": ".1f"},
            source_note="Misfitting: observed Yea count outside 95% replicated interval.",
        )
        report.add(
            TableSection(
                id=f"person-fit-table-{ch}", title=f"{chamber} — Person Fit Summary", html=html
            )
        )


def _add_margins(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    ch = chamber.lower()
    path = plots_dir / f"margins_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"margins-{ch}",
                f"{chamber} — Vote Margin Distributions",
                path,
                caption="Gray = observed margins. Colored = replicated mean margins per model.",
                alt_text=(
                    f"Overlaid histograms of vote margin distributions for {chamber}. "
                    "Gray shows observed margins; colored overlays show "
                    "replicated margins per model."
                ),
            )
        )


def _add_q3_summary(
    report: ReportBuilder,
    ch_data: dict,
    chamber: str,
) -> None:
    ch = chamber.lower()
    q3_results = ch_data.get("q3_results", {})
    if not q3_results:
        return

    rows = []
    for model_name, q3 in q3_results.items():
        rows.append(
            {
                "Model": model_name,
                "Item Pairs": q3["n_pairs"],
                "Violations (|Q3|>0.2)": q3["n_violations"],
                "Violation Rate": q3["violation_rate"],
                "Max |Q3|": q3["max_abs_q3"],
                "Mean |Q3|": q3["mean_abs_q3"],
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} — Yen's Q3 Local Dependence",
        subtitle="Residual correlations between items after conditioning on ability",
        number_formats={
            "Violation Rate": ".3f",
            "Max |Q3|": ".3f",
            "Mean |Q3|": ".3f",
        },
        source_note="Q3 > 0.2 indicates local dependence not explained by the latent trait. "
        "If 1D shows violations that 2D resolves, the second dimension is justified.",
    )
    report.add(TableSection(id=f"q3-{ch}", title=f"{chamber} — Q3 Local Dependence", html=html))


def _add_loo_comparison(
    report: ReportBuilder,
    all_results: dict[str, dict],
    plots_dir: Path,
    session: str,
) -> None:
    # Comparison tables
    for ch_name, ch_data in sorted(all_results.items()):
        ch = ch_name.lower()
        models_with_loo = {name: data for name, data in ch_data["models"].items() if "loo" in data}
        if not models_with_loo:
            continue

        rows = []
        for model_name, model_data in models_with_loo.items():
            loo = model_data["loo"]
            pk = loo["pareto_k"]
            rows.append(
                {
                    "Model": model_name,
                    "ELPD": loo["elpd_loo"],
                    "SE": loo["se"],
                    "p_loo": loo["p_loo"],
                    "k Good": pk["good"],
                    "k OK": pk["ok"],
                    "k Bad": pk["bad"],
                    "k Very Bad": pk["very_bad"],
                }
            )

        df = pl.DataFrame(rows)
        html = make_gt(
            df,
            title=f"{ch_name} — LOO-CV Model Comparison",
            subtitle=f"Session {session}",
            number_formats={"ELPD": ".1f", "SE": ".1f", "p_loo": ".1f"},
            source_note=(
                "ELPD: expected log pointwise predictive density "
                "(higher = better). p_loo: effective number of "
                "parameters. Pareto k: good (<0.5), OK (0.5-0.7), "
                "bad (0.7-1.0), very bad (>1.0)."
            ),
        )
        report.add(
            TableSection(id=f"loo-table-{ch}", title=f"{ch_name} — LOO-CV Comparison", html=html)
        )

        # Comparison plot
        path = plots_dir / f"loo_comparison_{ch}.png"
        if path.exists():
            report.add(
                FigureSection.from_file(
                    f"loo-plot-{ch}",
                    f"{ch_name} — LOO-CV Forest Plot",
                    path,
                    caption="ELPD differences with standard errors. "
                    "Higher ELPD = better out-of-sample prediction.",
                    alt_text=(
                        f"Forest plot comparing LOO-CV ELPD across models for {ch_name}. "
                        "Error bars show standard errors; higher ELPD indicates better prediction."
                    ),
                )
            )


def _add_pareto_k(
    report: ReportBuilder,
    all_results: dict[str, dict],
    plots_dir: Path,
) -> None:
    for ch_name, ch_data in sorted(all_results.items()):
        ch = ch_name.lower()
        path = plots_dir / f"pareto_k_{ch}.png"
        if path.exists():
            report.add(
                FigureSection.from_file(
                    f"pareto-k-{ch}",
                    f"{ch_name} — Pareto k Diagnostics",
                    path,
                    caption="Each point is one observation. Colors: green (<0.5 good), "
                    "yellow (0.5-0.7 ok), orange (0.7-1.0 bad), red (>1.0 very bad). "
                    "Observations with k > 0.7 have unreliable LOO estimates.",
                    alt_text=(
                        f"Scatter plot of Pareto k diagnostic values for {ch_name}. "
                        "Points colored green, yellow, orange, or red by reliability threshold."
                    ),
                )
            )


def _add_model_ranking(
    report: ReportBuilder,
    all_results: dict[str, dict],
    skip_loo: bool,
    skip_q3: bool,
) -> None:
    parts = ['<div style="background: #f9f9f0; padding: 1em; border-radius: 6px;">']
    parts.append("<h3>Model Ranking and Interpretation</h3>")

    for ch_name, ch_data in sorted(all_results.items()):
        parts.append(f"<h4>{ch_name}</h4>")
        models = ch_data["models"]

        # Rank by GMP
        ranked = sorted(models.items(), key=lambda x: x[1]["ppc"]["mean_gmp"], reverse=True)
        parts.append("<p><strong>Ranked by GMP (probabilistic accuracy):</strong></p><ol>")
        for name, data in ranked:
            ppc = data["ppc"]
            parts.append(
                f"<li>{name}: GMP = {ppc['mean_gmp']:.3f}, "
                f"Accuracy = {ppc['mean_accuracy']:.3f}, "
                f"APRE = {ppc['apre']:.3f}</li>"
            )
        parts.append("</ol>")

        # LOO ranking
        if not skip_loo:
            loo_models = {n: d for n, d in models.items() if "loo" in d}
            if len(loo_models) >= 2:
                loo_ranked = sorted(
                    loo_models.items(),
                    key=lambda x: x[1]["loo"]["elpd_loo"],
                    reverse=True,
                )
                parts.append(
                    "<p><strong>Ranked by ELPD (out-of-sample prediction):</strong></p><ol>"
                )
                for name, data in loo_ranked:
                    loo = data["loo"]
                    parts.append(
                        f"<li>{name}: ELPD = {loo['elpd_loo']:.1f} (SE = {loo['se']:.1f})</li>"
                    )
                parts.append("</ol>")

        # Q3 interpretation
        if not skip_q3:
            q3_results = ch_data.get("q3_results", {})
            if q3_results:
                parts.append("<p><strong>Dimensionality assessment (Q3):</strong></p><ul>")
                for name, q3 in q3_results.items():
                    verdict = (
                        "no major violations"
                        if q3["violation_rate"] < 0.05
                        else "violations present"
                    )
                    parts.append(
                        f"<li>{name}: {q3['n_violations']}/{q3['n_pairs']} violations "
                        f"({100 * q3['violation_rate']:.1f}%) — {verdict}</li>"
                    )
                parts.append("</ul>")

                # Dimensionality conclusion
                flat_q3 = q3_results.get("Flat 1D", {})
                twod_q3 = q3_results.get("2D IRT", {})
                if flat_q3 and twod_q3:
                    flat_vr = flat_q3.get("violation_rate", 0)
                    twod_vr = twod_q3.get("violation_rate", 0)
                    if flat_vr > 0.05 and twod_vr < 0.05:
                        parts.append(
                            "<p><em>The 1D model shows Q3 violations that the 2D model resolves, "
                            "supporting the need for a second dimension.</em></p>"
                        )
                    elif flat_vr < 0.05:
                        parts.append(
                            "<p><em>The 1D model shows few Q3 violations, suggesting a single "
                            "dimension is sufficient for this chamber.</em></p>"
                        )

    parts.append("</div>")
    report.add(
        TextSection(
            id="model-ranking",
            title="Model Ranking and Interpretation",
            html="\n".join(parts),
        )
    )


def _generate_ppc_key_findings(all_results: dict[str, dict]) -> list[str]:
    """Generate 2-4 key findings from PPC results."""
    findings: list[str] = []

    # Best model by GMP across all chambers
    best_gmp = 0.0
    best_model = ""
    best_chamber = ""
    for ch_name, ch_data in all_results.items():
        for model_name, model_data in ch_data.get("models", {}).items():
            ppc = model_data.get("ppc", {})
            gmp = ppc.get("mean_gmp", 0)
            if gmp > best_gmp:
                best_gmp = gmp
                best_model = model_name
                best_chamber = ch_name

    if best_model:
        findings.append(
            f"Best calibrated model: <strong>{best_model}</strong> "
            f"({best_chamber}, GMP = {best_gmp:.3f})."
        )

    # LOO-CV winner (if available)
    for ch_name, ch_data in sorted(all_results.items()):
        models_with_loo = {n: d for n, d in ch_data.get("models", {}).items() if "loo" in d}
        if len(models_with_loo) >= 2:
            loo_ranked = sorted(
                models_with_loo.items(),
                key=lambda x: x[1]["loo"]["elpd_loo"],
                reverse=True,
            )
            winner = loo_ranked[0]
            findings.append(
                f"{ch_name} LOO-CV winner: <strong>{winner[0]}</strong> "
                f"(ELPD = {winner[1]['loo']['elpd_loo']:.1f})."
            )
            break

    # Overall misfit rate
    total_items = 0
    total_misfit = 0
    for ch_data in all_results.values():
        for model_data in ch_data.get("models", {}).values():
            item = model_data.get("item_fit", {})
            total_items += item.get("n_votes", 0)
            total_misfit += item.get("n_misfitting", 0)
    if total_items > 0:
        misfit_pct = 100 * total_misfit / total_items
        findings.append(
            f"Item misfit rate: <strong>{misfit_pct:.1f}%</strong> "
            f"({total_misfit}/{total_items} votes across all models)."
        )

    return findings


def _add_methodology(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="methodology",
            title="Methodology",
            html="""
<h3>Posterior Predictive Checks</h3>
<p>For each model, 500 posterior draws are sampled. At each draw, a replicated
dataset is generated from the Bernoulli likelihood using the drawn parameters.
Summary statistics (Yea rate, accuracy, GMP, APRE) are computed on each
replicated dataset and compared to the observed data.</p>

<p><strong>Bayesian p-value</strong>: Fraction of replications where the statistic
exceeds the observed value. Well-calibrated models produce p-values near 0.5;
extreme values (< 0.1 or > 0.9) indicate systematic misfit.</p>

<h3>Item and Person Fit</h3>
<p>Per-item endorsement rates and per-person total scores are compared between
observed and replicated data. Items/persons are flagged as misfitting when the
observed value falls outside the 95% interval of replicated values.</p>

<h3>Yen's Q3</h3>
<p>For each posterior draw, residuals (observed - predicted probability) are
computed for each observation. The correlation matrix of residuals across items
is the Q3 matrix. Item pairs with |Q3| > 0.2 indicate local dependence not
captured by the latent trait. This is key for dimensionality assessment: if
1D shows Q3 violations that 2D resolves, the second dimension is empirically
justified.</p>

<h3>LOO-CV</h3>
<p>Leave-one-out cross-validation via Pareto-smoothed importance sampling
(PSIS-LOO; Vehtari, Gelman, & Gabry 2017). Estimates expected log pointwise
predictive density (ELPD) without refitting. Models are compared using
az.compare() which provides stacking weights. Pareto k diagnostics flag
observations where importance sampling is unreliable.</p>

<h3>References</h3>
<ul>
<li>Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
evaluation using leave-one-out cross-validation and WAIC. <em>Statistics and
Computing</em>, 27(5), 1413-1432.</li>
<li>Yen, W. M. (1993). Scaling performance assessments: Strategies for managing
local item dependence. <em>Journal of Educational Measurement</em>, 30(3), 187-213.</li>
</ul>
""",
        )
    )
