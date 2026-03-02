"""Beta-Binomial-specific HTML report builder.

Builds ~15-20 sections (tables, figures, and text) for the Bayesian party
loyalty report. Each section is a small function that slices/aggregates polars
DataFrames and calls make_gt() or FigureSection.from_file().

Usage (called from beta_binomial.py):
    from analysis.beta_binomial_report import build_beta_binomial_report
    build_beta_binomial_report(ctx.report, chamber_results=..., ...)
"""

from pathlib import Path

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

# Duplicated from beta_binomial.py — cannot import because beta_binomial.py
# imports this module at top level, creating a circular dependency.
MIN_PARTY_VOTES = 3
CI_LEVEL = 0.95


def build_beta_binomial_report(
    report: ReportBuilder,
    *,
    chamber_results: dict[str, dict],
    all_loyalty: dict[str, pl.DataFrame],
    plots_dir: Path,
) -> None:
    """Build the full beta-binomial HTML report by adding sections to the ReportBuilder."""
    findings = _generate_beta_binomial_key_findings(chamber_results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_what_is_shrinkage(report)
    _add_how_to_read(report)

    for chamber in sorted(all_loyalty.keys()):
        loyalty_df = all_loyalty[chamber]
        _add_full_table(report, loyalty_df, chamber)
        _add_shrinkage_arrows_figure(report, plots_dir, chamber)
        _add_credible_intervals_figure(report, plots_dir, chamber)
        _add_posterior_distributions_figure(report, plots_dir, chamber)
        _add_raw_vs_bayesian_figure(report, plots_dir, chamber)
        _add_interpretation(report, loyalty_df, chamber)

    # Cross-chamber comparison
    if len(all_loyalty) > 1:
        _add_cross_chamber_comparison(report, all_loyalty)

    _add_analysis_parameters(report)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_what_is_shrinkage(report: ReportBuilder) -> None:
    """Text block: plain-English explanation of Bayesian shrinkage."""
    report.add(
        TextSection(
            id="what-is-shrinkage",
            title="What Is Bayesian Shrinkage?",
            html=(
                "<p>Imagine you're rating restaurants. A restaurant with 500 five-star reviews "
                "is probably genuinely excellent. But what about a restaurant with 3 five-star "
                "reviews? It <em>might</em> be excellent — or those 3 reviewers might just be "
                "the owner's friends.</p>"
                "<p>Bayesian shrinkage applies this same logic to party loyalty scores. When a "
                "legislator has cast hundreds of party votes, their raw loyalty rate is reliable "
                "— it's based on a lot of data. But when a legislator has only cast a handful of "
                "party votes (e.g., they joined mid-session, or they missed many votes), their "
                "raw rate is unreliable.</p>"
                "<p>The Bayesian approach <strong>pulls uncertain estimates toward the party "
                "average</strong>. Think of it as a rubber band: legislators with few votes get "
                "pulled more strongly toward the party mean. Legislators with many votes barely "
                "move — their data speaks for itself.</p>"
                "<p>The result: more reliable loyalty estimates for everyone, with honest "
                "uncertainty bars that tell you how confident the analysis is about each "
                "legislator.</p>"
            ),
        )
    )


def _add_how_to_read(report: ReportBuilder) -> None:
    """Text block: how to interpret the report."""
    report.add(
        TextSection(
            id="how-to-read",
            title="How to Read This Report",
            html=(
                "<p>This report shows four views of the same data:</p>"
                "<ul>"
                "<li><strong>Shrinkage Arrows</strong> — How much did each estimate move? "
                "Arrows show the shift from raw rate to Bayesian estimate, plotted against "
                "sample size. Legislators near the bottom (few votes) move the most.</li>"
                "<li><strong>Credible Intervals</strong> — How certain are we? Horizontal bars "
                "show the 95% range of plausible loyalty values. Wider bars = less certainty.</li>"
                "<li><strong>Posterior Distributions</strong> — Three example legislators showing "
                "how the bell curve shape changes with sample size. Tall, narrow curves mean high "
                "confidence; wide, flat curves mean low confidence.</li>"
                "<li><strong>Raw vs. Bayesian Scatter</strong> — A direct comparison. Points on "
                "the diagonal didn't change. Points pulled toward the center were shrunk.</li>"
                "</ul>"
                "<p>The table shows all legislators with their raw and Bayesian loyalty "
                "estimates, credible intervals, and shrinkage factors. "
                "It is never truncated — all legislators "
                "are shown.</p>"
            ),
        )
    )


def _add_full_table(
    report: ReportBuilder,
    loyalty_df: pl.DataFrame,
    chamber: str,
) -> None:
    """Table: ALL legislators with posterior stats — never truncated."""
    if loyalty_df.height == 0:
        return

    display = loyalty_df.sort("posterior_mean").select(
        "full_name",
        "party",
        "district",
        "raw_loyalty",
        "posterior_mean",
        "ci_lower",
        "ci_upper",
        "ci_width",
        "shrinkage",
        "n_party_votes",
    )

    html = make_gt(
        display,
        title=f"{chamber} — Bayesian Party Loyalty ({display.height} legislators)",
        subtitle="All legislators shown (never truncated), sorted by posterior mean",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "district": "District",
            "raw_loyalty": "Raw Loyalty",
            "posterior_mean": "Bayesian Estimate",
            "ci_lower": "CI Lower",
            "ci_upper": "CI Upper",
            "ci_width": "CI Width",
            "shrinkage": "Shrinkage",
            "n_party_votes": "N Party Votes",
        },
        number_formats={
            "raw_loyalty": ".3f",
            "posterior_mean": ".3f",
            "ci_lower": ".3f",
            "ci_upper": ".3f",
            "ci_width": ".3f",
            "shrinkage": ".3f",
        },
        source_note=(
            "Raw Loyalty = CQ-standard party unity (votes with party / party votes present). "
            "Bayesian Estimate = posterior mean after shrinkage toward party average. "
            f"CI = {CI_LEVEL * 100:.0f}% credible interval. "
            "Shrinkage: 0 = no change, 1 = fully pulled to party mean."
        ),
    )
    report.add(
        TableSection(
            id=f"loyalty-table-{chamber.lower()}",
            title=f"{chamber} Bayesian Loyalty Table",
            html=html,
        )
    )


def _add_shrinkage_arrows_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"shrinkage_arrows_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-shrinkage-{chamber.lower()}",
                f"{chamber} Shrinkage Arrows",
                path,
                caption=(
                    "Each arrow shows the shift from raw loyalty rate (faded dot) to Bayesian "
                    "estimate (bright dot). Legislators with few party votes (near the bottom) "
                    "experience the most shrinkage — their estimates move furthest toward the "
                    "party mean."
                ),
                alt_text=(
                    "Arrow plot showing Bayesian shrinkage for "
                    f"{chamber} legislators. Arrows point from raw loyalty "
                    "to Bayesian estimate; longer arrows indicate "
                    "more shrinkage."
                ),
            )
        )


def _add_credible_intervals_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"credible_intervals_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-ci-{chamber.lower()}",
                f"{chamber} Credible Intervals",
                path,
                caption=(
                    f"Each horizontal bar shows the {CI_LEVEL * 100:.0f}% credible interval for a "
                    "legislator's true loyalty. Wider bars mean less certainty. The dot marks "
                    "the posterior mean (best estimate)."
                ),
                alt_text=(
                    f"Forest plot of 95% credible intervals for party loyalty in {chamber}. "
                    "Each bar represents one legislator; wider bars indicate greater uncertainty."
                ),
            )
        )


def _add_posterior_distributions_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"posterior_distributions_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-posteriors-{chamber.lower()}",
                f"{chamber} Posterior Distributions",
                path,
                caption=(
                    "Posterior Beta distributions for three legislators: the most shrunk, "
                    "the least shrunk, and the lowest loyalty. Taller and narrower curves "
                    "represent more certainty (more data). Wide, flat curves represent "
                    "high uncertainty (fewer votes)."
                ),
                alt_text=(
                    "Density curves of posterior Beta distributions for "
                    f"three {chamber} legislators. Narrow curves indicate "
                    "high certainty; wide curves indicate fewer votes."
                ),
            )
        )


def _add_raw_vs_bayesian_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"raw_vs_bayesian_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-raw-vs-bayes-{chamber.lower()}",
                f"{chamber} Raw vs. Bayesian Loyalty",
                path,
                caption=(
                    "Each dot is a legislator. The diagonal line marks 'no change' — points "
                    "on the line were not affected by shrinkage. Points pulled toward the "
                    "center (party mean) experienced shrinkage. Larger dots = more party votes."
                ),
                alt_text=(
                    "Scatter plot comparing raw and Bayesian loyalty "
                    f"estimates for {chamber}. Points near the diagonal "
                    "were minimally shrunk; displaced points show "
                    "shrinkage effect."
                ),
            )
        )


def _add_interpretation(
    report: ReportBuilder,
    loyalty_df: pl.DataFrame,
    chamber: str,
) -> None:
    """Text: chamber-specific interpretation."""
    if loyalty_df.height == 0:
        return

    # Compute summary stats for the narrative
    parts = []

    for party in ["Republican", "Democrat"]:
        party_sub = loyalty_df.filter(pl.col("party") == party)
        if party_sub.height == 0:
            continue

        prior_mean = float(party_sub["prior_mean"][0])
        mean_shrink = float(party_sub["shrinkage"].mean())
        mean_ci = float(party_sub["ci_width"].mean())

        parts.append(
            f"<p><strong>{chamber} {party}s</strong> (n={party_sub.height}): "
            f"The party prior loyalty is {prior_mean:.1%}. "
            f"Average shrinkage factor is {mean_shrink:.3f} "
            f"(meaning estimates moved about {mean_shrink:.1%} of the way from the raw rate "
            f"to the party mean). Average credible interval width is {mean_ci:.3f}.</p>"
        )

    # Most affected legislator
    most_affected = loyalty_df.with_columns(
        (pl.col("raw_loyalty") - pl.col("posterior_mean")).abs().alias("abs_delta")
    ).sort("abs_delta", descending=True)

    if most_affected.height > 0:
        top = most_affected.row(0, named=True)
        delta = top["posterior_mean"] - top["raw_loyalty"]
        direction = "up" if delta > 0 else "down"
        parts.append(
            f"<p>The most affected legislator is <strong>{top['full_name']}</strong> "
            f"({top['party']}), whose estimate moved {direction} by "
            f"{abs(delta):.3f} (from {top['raw_loyalty']:.3f} to "
            f"{top['posterior_mean']:.3f}) based on {top['n_party_votes']} party votes.</p>"
        )

    report.add(
        TextSection(
            id=f"interpretation-{chamber.lower()}",
            title=f"{chamber} — What Changed?",
            html="".join(parts),
        )
    )


def _add_cross_chamber_comparison(
    report: ReportBuilder,
    all_loyalty: dict[str, pl.DataFrame],
) -> None:
    """Table: Compare party hyperparameters across chambers."""
    rows = []
    for chamber in sorted(all_loyalty.keys()):
        loyalty_df = all_loyalty[chamber]
        for party in ["Republican", "Democrat"]:
            party_sub = loyalty_df.filter(pl.col("party") == party)
            if party_sub.height == 0:
                continue
            rows.append(
                {
                    "Chamber": chamber,
                    "Party": party,
                    "N Legislators": party_sub.height,
                    "Prior Alpha": float(party_sub["alpha_prior"][0]),
                    "Prior Beta": float(party_sub["beta_prior"][0]),
                    "Prior Kappa": float(party_sub["prior_kappa"][0]),
                    "Prior Mean Loyalty": float(party_sub["prior_mean"][0]),
                    "Mean Shrinkage": float(party_sub["shrinkage"].mean()),
                    "Mean CI Width": float(party_sub["ci_width"].mean()),
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Cross-Chamber Comparison of Party Priors",
        subtitle="How do Republican and Democratic loyalty patterns differ between chambers?",
        number_formats={
            "Prior Alpha": ".2f",
            "Prior Beta": ".2f",
            "Prior Kappa": ".1f",
            "Prior Mean Loyalty": ".3f",
            "Mean Shrinkage": ".3f",
            "Mean CI Width": ".3f",
        },
        source_note=(
            "Alpha and Beta are the empirical Bayes prior parameters. "
            "Kappa (alpha+beta) is the effective prior sample size — "
            "higher = less within-party variance. "
            "Prior mean = alpha / (alpha + beta)."
        ),
    )
    report.add(
        TableSection(
            id="cross-chamber",
            title="Cross-Chamber Comparison",
            html=html,
        )
    )


def _generate_beta_binomial_key_findings(
    chamber_results: dict[str, dict],
) -> list[str]:
    """Generate 2-4 key findings from beta-binomial results."""
    findings: list[str] = []

    for chamber, result in chamber_results.items():
        shrinkage = result.get("shrinkage_summary", {})
        min_shrink = shrinkage.get("min_shrinkage")
        max_shrink = shrinkage.get("max_shrinkage")
        if min_shrink is not None and max_shrink is not None:
            findings.append(
                f"{chamber} shrinkage range: <strong>{min_shrink:.1%}</strong> to "
                f"<strong>{max_shrink:.1%}</strong>."
            )

        # Most/least loyal from loyalty data
        loyalty_df = result.get("loyalty_df")
        if loyalty_df is not None and loyalty_df.height > 0:
            sorted_loy = loyalty_df.sort("posterior_mean", descending=True)
            most = sorted_loy.head(1)
            least = sorted_loy.tail(1)
            most_name = most["full_name"][0] if "full_name" in most.columns else "N/A"
            most_rate = float(most["posterior_mean"][0])
            least_name = least["full_name"][0] if "full_name" in least.columns else "N/A"
            least_rate = float(least["posterior_mean"][0])
            findings.append(
                f"{chamber} most loyal: <strong>{most_name}</strong> ({most_rate:.0%}), "
                f"least loyal: <strong>{least_name}</strong> ({least_rate:.0%})."
            )

        break  # First chamber only

    return findings


def _add_analysis_parameters(report: ReportBuilder) -> None:
    """Table: All constants and settings used in this run."""
    df = pl.DataFrame(
        {
            "Parameter": [
                "Minimum Party Votes",
                "Credible Interval Level",
                "Hyperparameter Estimation",
                "Prior Fallback",
                "Group Structure",
            ],
            "Value": [
                str(MIN_PARTY_VOTES),
                f"{CI_LEVEL:.0%}",
                "Method of moments (empirical Bayes)",
                "Beta(1, 1) — uniform",
                "Per party per chamber (4 groups)",
            ],
            "Description": [
                f"Legislators with fewer than {MIN_PARTY_VOTES} party votes are excluded",
                "Width of the credible interval (equal-tailed)",
                "Beta prior estimated from observed loyalty rates within each group",
                "Used when variance exceeds Beta family limits or data is degenerate",
                "Separate priors for House R, House D, Senate R, Senate D",
            ],
        }
    )
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="See analysis/design/beta_binomial.md for justification.",
    )
    report.add(
        TableSection(
            id="analysis-params",
            title="Analysis Parameters",
            html=html,
        )
    )
