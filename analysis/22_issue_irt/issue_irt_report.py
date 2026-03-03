"""Issue-Specific Ideal Points (Phase 19) HTML report builder.

Builds sections for the issue IRT report: key findings, topic overview,
per-topic scatter plots, cross-topic heatmap, ideological profiles,
outlier table, convergence summary, correlation summary, anchor stability,
methodology, and analysis parameters.

Usage (called from issue_irt.py):
    from analysis.issue_irt_report import build_issue_irt_report
    build_issue_irt_report(ctx.report, all_taxonomy_results=..., ...)
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

try:
    from analysis.issue_irt_data import (
        ESS_THRESHOLD,
        GOOD_CORRELATION,
        MIN_BILLS_PER_TOPIC,
        MIN_LEGISLATORS_PER_TOPIC,
        MIN_VOTES_IN_TOPIC,
        MODERATE_CORRELATION,
        OUTLIER_TOP_N,
        RHAT_THRESHOLD,
        STRONG_CORRELATION,
    )
except ModuleNotFoundError:
    from issue_irt_data import (  # type: ignore[no-redef]
        ESS_THRESHOLD,
        GOOD_CORRELATION,
        MIN_BILLS_PER_TOPIC,
        MIN_LEGISLATORS_PER_TOPIC,
        MIN_VOTES_IN_TOPIC,
        MODERATE_CORRELATION,
        OUTLIER_TOP_N,
        RHAT_THRESHOLD,
        STRONG_CORRELATION,
    )


def build_issue_irt_report(
    report: ReportBuilder,
    *,
    all_taxonomy_results: dict[str, dict],
    plots_dir: Path,
    session: str,
    taxonomies: list[str],
    args: dict,
) -> None:
    """Build the full issue-specific IRT HTML report."""
    findings = _generate_key_findings(all_taxonomy_results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_methodology(report)

    for taxonomy in taxonomies:
        tax_results = all_taxonomy_results.get(taxonomy, {})
        if not tax_results:
            continue

        tax_label = "BERTopic" if taxonomy == "bertopic" else "CAP"

        if len(taxonomies) > 1:
            report.add(
                TextSection(
                    id=f"taxonomy-{taxonomy}",
                    title=f"Taxonomy: {tax_label}",
                    html=f"<p>Results using <strong>{tax_label}</strong> topic assignments.</p>",
                )
            )

        # Topic overview table
        _add_topic_overview(report, tax_results, taxonomy)

        # Per-topic scatter plots
        for ch_key, ch_data in sorted(tax_results.items()):
            chamber = ch_data["chamber"]
            for tid, tr in sorted(ch_data["topic_results"].items()):
                _add_topic_scatter(report, plots_dir, tid, chamber, tr["label"])

        # Cross-topic heatmap
        for ch_key, ch_data in sorted(tax_results.items()):
            chamber = ch_data["chamber"]
            _add_heatmap_figure(report, plots_dir, chamber, taxonomy)

        # Ideological profile heatmap
        for ch_key, ch_data in sorted(tax_results.items()):
            chamber = ch_data["chamber"]
            _add_profile_figure(report, plots_dir, chamber, taxonomy)

        # Outlier table
        _add_outlier_table(report, tax_results, taxonomy)

        # Convergence summary
        _add_convergence_summary(report, tax_results, taxonomy)

        # Correlation summary
        _add_correlation_summary(report, tax_results, taxonomy)

        # Anchor stability
        _add_anchor_stability(report, tax_results, taxonomy)

    _add_interpretation_guide(report)
    _add_analysis_parameters(report, session, taxonomies, args)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _generate_key_findings(all_taxonomy_results: dict[str, dict]) -> list[str]:
    """Generate 2-4 key findings from issue IRT results."""
    findings: list[str] = []

    n_topics_total = 0
    all_rs: list[float] = []
    most_sep_topic = ""
    most_sep_r = 1.0
    least_sep_topic = ""
    least_sep_r = 0.0

    for tax_results in all_taxonomy_results.values():
        for ch_data in tax_results.values():
            for tid, tr in ch_data["topic_results"].items():
                n_topics_total += 1
                corr = tr.get("correlation", {})
                r = corr.get("pearson_r")
                if r is not None and not np.isnan(r):
                    all_rs.append(abs(r))
                    label = tr["label"][:40]
                    if abs(r) < most_sep_r:
                        most_sep_r = abs(r)
                        most_sep_topic = label
                    if abs(r) > least_sep_r:
                        least_sep_r = abs(r)
                        least_sep_topic = label

    if n_topics_total > 0:
        findings.append(
            f"<strong>{n_topics_total}</strong> topic-specific IRT models estimated "
            f"across all chambers and taxonomies."
        )

    if all_rs:
        mean_r = sum(all_rs) / len(all_rs)
        findings.append(
            f"Mean |r| with full IRT: <strong>{mean_r:.3f}</strong> "
            f"({len(all_rs)} topic-chamber pairs)."
        )

    if most_sep_topic and most_sep_r < 0.95:
        findings.append(
            f"Most distinctive topic: <strong>{most_sep_topic}</strong> "
            f"(r = {most_sep_r:.3f} with full IRT)."
        )

    if least_sep_topic and least_sep_r > most_sep_r:
        findings.append(
            f"Most aligned topic: <strong>{least_sep_topic}</strong> "
            f"(r = {least_sep_r:.3f} with full IRT)."
        )

    return findings


def _add_methodology(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="methodology",
            title="Methodology",
            html=(
                "<p>This phase estimates <strong>per-topic Bayesian IRT ideal points</strong> "
                "by running the same 2PL IRT model from Phase 04 on topic-stratified vote "
                "subsets.</p>"
                "<ol>"
                "<li><strong>Load topic assignments</strong> from Phase 18 (BERTopic) "
                "and/or CAP classifications.</li>"
                "<li><strong>For each eligible topic</strong> (enough bills with roll calls):</li>"
                "<ul>"
                "<li>Subset the EDA vote matrix to vote_ids belonging to that topic.</li>"
                "<li>Filter legislators with enough non-null votes in the topic.</li>"
                "<li>Run standard 2PL IRT (anchor-identified, nutpie Rust NUTS sampler).</li>"
                "<li>Sign-align against the full-model IRT ideal points.</li>"
                "</ul>"
                "<li><strong>Cross-topic analysis</strong>: pairwise correlations between "
                "topic ideal points, ideological profile heatmap, outlier detection.</li>"
                "</ol>"
                "<p><strong>Why stratified flat IRT?</strong> Phase 04b already showed Kansas "
                "voting is fundamentally 1D. Per-topic subsets are too small for multi-dimensional "
                "models. Reusing the battle-tested Phase 04 infrastructure means zero new model "
                "code and zero new dependencies.</p>"
            ),
        )
    )


def _add_topic_overview(
    report: ReportBuilder,
    tax_results: dict[str, dict],
    taxonomy: str,
) -> None:
    rows = []
    for ch_key, ch_data in sorted(tax_results.items()):
        chamber = ch_data["chamber"]
        eligibility = ch_data.get("eligibility_report")
        if eligibility is None:
            continue

        for row in eligibility.iter_rows(named=True):
            status = "Modeled" if row.get("eligible", False) else "Skipped"
            reason = ""
            if not row.get("eligible", False):
                n_rc = row.get("n_rollcall_bills", 0)
                reason = f"Only {n_rc} roll-call bills"

            # Check if we have results for this topic
            tid = row["topic_id"]
            topic_r = ch_data["topic_results"].get(tid, {})
            if topic_r and status == "Modeled":
                corr = topic_r.get("correlation", {})
                r_val = corr.get("pearson_r", float("nan"))
                if np.isnan(r_val):
                    status = "Failed"
                    reason = "MCMC or extraction failed"

            rows.append(
                {
                    "Chamber": chamber,
                    "Topic": row.get("topic_label", f"Topic {tid}")[:50],
                    "Bills": row.get("n_bills", 0),
                    "Roll-Call Bills": row.get("n_rollcall_bills", 0),
                    "Status": status,
                    "Reason": reason,
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    tax_label = "BERTopic" if taxonomy == "bertopic" else "CAP"
    html = make_gt(
        df,
        title=f"Topic Overview ({tax_label})",
        subtitle="Eligible and skipped topics by chamber",
        source_note=(
            f"Minimum {MIN_BILLS_PER_TOPIC} roll-call bills required per topic. "
            f"Minimum {MIN_LEGISLATORS_PER_TOPIC} legislators with ≥ {MIN_VOTES_IN_TOPIC} votes."
        ),
    )
    report.add(
        TableSection(
            id=f"topic-overview-{taxonomy}",
            title=f"Topic Overview ({tax_label})",
            html=html,
        )
    )


def _add_topic_scatter(
    report: ReportBuilder,
    plots_dir: Path,
    topic_id: int,
    chamber: str,
    label: str,
) -> None:
    ch = chamber.lower()
    path = plots_dir / f"scatter_t{topic_id}_{ch}.png"
    if path.exists():
        short_label = label[:50]
        report.add(
            FigureSection.from_file(
                f"fig-scatter-t{topic_id}-{ch}",
                f"{chamber} — {short_label}",
                path,
                caption=(
                    f"Per-topic ideal point vs full IRT ideal point for {chamber}. "
                    "Each dot is a legislator, colored by party. Tight clustering "
                    "along the diagonal = topic tracks overall ideology."
                ),
                alt_text=(
                    f"Scatter plot of topic {topic_id} ideal points vs full IRT "
                    f"for {chamber}. Points colored by party."
                ),
            )
        )


def _add_heatmap_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
    taxonomy: str,
) -> None:
    ch = chamber.lower()
    path = plots_dir / f"heatmap_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-heatmap-{taxonomy}-{ch}",
                f"{chamber} — Cross-Topic Correlation Heatmap",
                path,
                caption=(
                    "Pairwise Pearson r between per-topic ideal points. "
                    "High correlations (red) = topics that sort legislators the same way. "
                    "Low/negative correlations (blue) = cross-cutting issues."
                ),
                alt_text=(
                    f"Heatmap showing pairwise correlations between topic ideal points "
                    f"for {chamber}. Most topics show positive correlations."
                ),
            )
        )


def _add_profile_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
    taxonomy: str,
) -> None:
    ch = chamber.lower()
    path = plots_dir / f"profile_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-profile-{taxonomy}-{ch}",
                f"{chamber} — Ideological Profiles by Topic",
                path,
                caption=(
                    "Each row is a legislator (sorted by overall IRT ideal point), "
                    "each column is a topic. Colors show z-standardized topic ideal points: "
                    "red = more conservative than topic mean, blue = more liberal. "
                    "Horizontal streaks = consistent across topics. "
                    "Breaks in the pattern = topic-specific divergence."
                ),
                alt_text=(
                    f"Heatmap of legislator ideological profiles across topics for {chamber}. "
                    "Rows sorted by full IRT ideal point, columns are topics."
                ),
            )
        )


def _add_outlier_table(
    report: ReportBuilder,
    tax_results: dict[str, dict],
    taxonomy: str,
) -> None:
    rows = []
    for ch_key, ch_data in sorted(tax_results.items()):
        chamber = ch_data["chamber"]
        for tid, tr in sorted(ch_data["topic_results"].items()):
            outliers = tr.get("outliers")
            if outliers is None or outliers.height == 0:
                continue
            label = tr["label"][:40]
            for row in outliers.iter_rows(named=True):
                rows.append(
                    {
                        "Chamber": chamber,
                        "Topic": label,
                        "Name": row.get("full_name", row.get("legislator_slug", "")),
                        "Party": row.get("party", ""),
                        "Topic xi": row.get("xi_topic", float("nan")),
                        "Full xi": row.get("xi_full", float("nan")),
                        "Discrepancy (z)": row.get("discrepancy_z", float("nan")),
                    }
                )

    if not rows:
        return

    df = pl.DataFrame(rows)
    tax_label = "BERTopic" if taxonomy == "bertopic" else "CAP"
    html = make_gt(
        df,
        title=f"Topic Outliers ({tax_label})",
        subtitle=(
            f"Top {OUTLIER_TOP_N} deviators per topic — "
            "legislators furthest from overall position"
        ),
        number_formats={
            "Topic xi": ".3f",
            "Full xi": ".3f",
            "Discrepancy (z)": ".2f",
        },
        source_note=(
            "Both topic and full ideal points are z-standardized before computing "
            "discrepancy. High discrepancy = legislator is cross-pressured on this "
            "policy area."
        ),
    )
    report.add(
        TableSection(
            id=f"outliers-{taxonomy}",
            title=f"Topic Outliers ({tax_label})",
            html=html,
        )
    )


def _add_convergence_summary(
    report: ReportBuilder,
    tax_results: dict[str, dict],
    taxonomy: str,
) -> None:
    rows = []
    for ch_key, ch_data in sorted(tax_results.items()):
        chamber = ch_data["chamber"]
        for tid, tr in sorted(ch_data["topic_results"].items()):
            diag = tr.get("convergence", {})
            label = tr["label"][:40]
            rows.append(
                {
                    "Chamber": chamber,
                    "Topic": label,
                    "R-hat (xi)": diag.get("xi_rhat_max", float("nan")),
                    "ESS (xi)": diag.get("xi_ess_min", float("nan")),
                    "Divergences": diag.get("divergences", "?"),
                    "Time (s)": tr.get("sampling_time", float("nan")),
                    "Converged": "Yes" if tr.get("converged", False) else "No",
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    tax_label = "BERTopic" if taxonomy == "bertopic" else "CAP"
    html = make_gt(
        df,
        title=f"Convergence Summary ({tax_label})",
        subtitle="MCMC diagnostics per topic",
        number_formats={
            "R-hat (xi)": ".4f",
            "ESS (xi)": ".0f",
            "Time (s)": ".1f",
        },
        source_note=(
            f"Thresholds: R-hat < {RHAT_THRESHOLD}, ESS > {ESS_THRESHOLD}. "
            "Relaxed from Phase 04 (smaller per-topic models). "
            "nutpie Rust NUTS sampler."
        ),
    )
    report.add(
        TableSection(
            id=f"convergence-{taxonomy}",
            title=f"Convergence Summary ({tax_label})",
            html=html,
        )
    )


def _add_correlation_summary(
    report: ReportBuilder,
    tax_results: dict[str, dict],
    taxonomy: str,
) -> None:
    rows = []
    for ch_key, ch_data in sorted(tax_results.items()):
        chamber = ch_data["chamber"]
        for tid, tr in sorted(ch_data["topic_results"].items()):
            corr = tr.get("correlation", {})
            if corr.get("quality") == "insufficient_data":
                continue
            label = tr["label"][:40]
            rows.append(
                {
                    "Chamber": chamber,
                    "Topic": label,
                    "n": corr.get("n", 0),
                    "Pearson r": corr.get("pearson_r", float("nan")),
                    "Spearman ρ": corr.get("spearman_rho", float("nan")),
                    "CI Lower": corr.get("ci_lower", float("nan")),
                    "CI Upper": corr.get("ci_upper", float("nan")),
                    "Quality": corr.get("quality", ""),
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    tax_label = "BERTopic" if taxonomy == "bertopic" else "CAP"
    html = make_gt(
        df,
        title=f"Per-Topic Correlations with Full IRT ({tax_label})",
        subtitle="How well each topic tracks overall ideology",
        number_formats={
            "Pearson r": ".3f",
            "Spearman ρ": ".3f",
            "CI Lower": ".3f",
            "CI Upper": ".3f",
        },
        source_note=(
            f"Quality: strong (r ≥ {STRONG_CORRELATION}), "
            f"good ({GOOD_CORRELATION} ≤ r < {STRONG_CORRELATION}), "
            f"moderate ({MODERATE_CORRELATION} ≤ r < {GOOD_CORRELATION}), "
            f"weak (r < {MODERATE_CORRELATION}). "
            "CI = 95% Fisher z confidence interval. "
            "Lower r = topic introduces distinct ideological variation."
        ),
    )
    report.add(
        TableSection(
            id=f"correlation-{taxonomy}",
            title=f"Per-Topic Correlation Summary ({tax_label})",
            html=html,
        )
    )


def _add_anchor_stability(
    report: ReportBuilder,
    tax_results: dict[str, dict],
    taxonomy: str,
) -> None:
    # Check if any anchor stability data exists
    has_data = False
    for ch_data in tax_results.values():
        for tr in ch_data["topic_results"].values():
            if tr.get("anchor_slugs"):
                has_data = True
                break

    if not has_data:
        return

    report.add(
        TextSection(
            id=f"anchor-stability-note-{taxonomy}",
            title="Anchor Stability",
            html=(
                "<p>Per-topic IRT uses the same anchors as the full model (from PCA PC1 "
                "extremes). Stable anchors should remain near the top/bottom of the "
                "per-topic ideal point ranking. If anchors move toward the center in a "
                "topic, that topic may have a different ideological axis.</p>"
                "<p>Anchor stability data is saved in per-chamber parquet files. "
                "Consult <code>anchor_stability_*.parquet</code> for percentile rankings.</p>"
            ),
        )
    )


def _add_interpretation_guide(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="interpretation",
            title="Interpretation Guide",
            html=(
                "<p>How to interpret issue-specific ideal points:</p>"
                "<table style='width:100%; border-collapse:collapse; font-size:14px; "
                "margin-top:12px;'>"
                "<thead><tr>"
                "<th style='text-align:left; border-bottom:2px solid #333; padding:6px;'>"
                "Per-Topic r</th>"
                "<th style='text-align:left; border-bottom:2px solid #333; padding:6px;'>"
                "Interpretation</th>"
                "</tr></thead><tbody>"
                f"<tr><td style='padding:4px 6px;'>≥ {STRONG_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Strong — topic tracks overall ideology</td></tr>"
                f"<tr><td style='padding:4px 6px;'>{GOOD_CORRELATION}–{STRONG_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Good — some topic-specific divergence</td></tr>"
                f"<tr><td style='padding:4px 6px;'>{MODERATE_CORRELATION}–{GOOD_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>"
                "Moderate — meaningful policy-area differences</td></tr>"
                f"<tr><td style='padding:4px 6px;'>< {MODERATE_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Weak — topic has distinct ideological axis</td></tr>"
                "</tbody></table>"
                "<p style='margin-top:12px;'><strong>Cross-topic correlations:</strong> "
                "High pairwise r between topics = legislators are consistently ranked "
                "regardless of policy area. Low/negative r = cross-cutting issue that "
                "reshuffles the ideological ordering.</p>"
                "<p><strong>Outliers:</strong> Legislators with high discrepancy between "
                "their topic-specific and overall ideal points are cross-pressured — they "
                "deviate from their party's typical position on that policy area.</p>"
            ),
        )
    )


def _add_analysis_parameters(
    report: ReportBuilder,
    session: str,
    taxonomies: list[str],
    args: dict,
) -> None:
    df = pl.DataFrame(
        {
            "Parameter": [
                "Session",
                "Taxonomies",
                "MCMC Draws",
                "MCMC Tune",
                "MCMC Chains",
                "Min Bills per Topic",
                "Min Legislators per Topic",
                "Min Votes in Topic",
                "R-hat Threshold",
                "ESS Threshold",
                "Outlier Top-N",
                "Strong Correlation",
                "Good Correlation",
                "Moderate Correlation",
                "Sign Convention",
            ],
            "Value": [
                session,
                ", ".join(taxonomies),
                str(args.get("n_samples", 1000)),
                str(args.get("n_tune", 1000)),
                str(args.get("n_chains", 2)),
                str(args.get("min_bills", MIN_BILLS_PER_TOPIC)),
                str(args.get("min_legislators", MIN_LEGISLATORS_PER_TOPIC)),
                str(args.get("min_votes_in_topic", MIN_VOTES_IN_TOPIC)),
                str(RHAT_THRESHOLD),
                str(ESS_THRESHOLD),
                str(OUTLIER_TOP_N),
                str(STRONG_CORRELATION),
                str(GOOD_CORRELATION),
                str(MODERATE_CORRELATION),
                "Aligned with full IRT (Republicans positive)",
            ],
            "Description": [
                "Legislative session analyzed",
                "Topic taxonomy source(s)",
                "Posterior draws per chain per topic",
                "Tuning steps per chain (discarded)",
                "Independent MCMC chains per topic",
                "Topics with fewer bills are skipped",
                "Topics with fewer eligible legislators are skipped",
                "Per-legislator minimum non-null votes within topic",
                "Maximum R-hat for convergence (relaxed from Phase 04)",
                "Minimum bulk ESS for convergence (relaxed from Phase 04)",
                "Number of outliers reported per topic",
                f"Pearson |r| ≥ {STRONG_CORRELATION}",
                f"Pearson |r| ≥ {GOOD_CORRELATION}",
                f"Pearson |r| ≥ {MODERATE_CORRELATION}",
                "Per-topic xi negated if r < 0 with full IRT",
            ],
        }
    )
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="See analysis/design/issue_irt.md for methodology justification.",
    )
    report.add(TableSection(id="analysis-params", title="Analysis Parameters", html=html))
