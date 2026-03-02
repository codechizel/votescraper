"""LCA-specific HTML report builder.

Builds ~18 sections (tables, figures, and text) for the LCA report.
Each section is a small function that slices/aggregates results and calls
make_gt() or FigureSection.from_file().

Usage (called from lca.py):
    from analysis.lca_report import build_lca_report
    build_lca_report(ctx.report, results=results, plots_dir=plots_dir)
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


def build_lca_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    plots_dir: Path,
) -> None:
    """Build the full LCA HTML report by adding sections to the ReportBuilder."""
    findings = _generate_lca_key_findings(results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_data_summary(report, results)

    for chamber in results:
        _add_enumeration_table(report, results[chamber], chamber)
        _add_bic_elbow_figure(report, plots_dir, chamber)
        _add_optimal_k_summary(report, results[chamber], chamber)
        _add_composition_table(report, results[chamber], chamber)
        _add_class_membership_table(report, results[chamber], chamber)
        _add_membership_certainty_note(report, results[chamber], chamber)
        _add_profile_heatmap_figure(report, plots_dir, chamber)
        _add_membership_figure(report, plots_dir, chamber)
        _add_irt_crossval(report, results[chamber], chamber)
        _add_irt_boxplot_figure(report, plots_dir, chamber)
        _add_salsa_assessment(report, results[chamber], chamber)
        _add_salsa_matrix_figure(report, plots_dir, chamber)
        _add_clustering_agreement(report, results[chamber], chamber)
        _add_within_party_results(report, results[chamber], chamber)
        _add_discriminating_bills(report, results[chamber], chamber)

    _add_analysis_parameters(report)
    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ──────────────────────────────────────────────────


def _add_data_summary(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Data dimensions per chamber."""
    rows = []
    for chamber, result in results.items():
        rows.append(
            {
                "Chamber": chamber,
                "Legislators": result["n_legislators"],
                "Votes": result["n_votes"],
                "Missing (%)": round(result["pct_missing"], 1),
                "Optimal K": result["optimal_k"],
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="LCA Data Summary",
        subtitle="Upstream data dimensions and BIC-selected class count per chamber",
    )
    report.add(TableSection(id="data-summary", title="Data Summary", html=html))


def _add_enumeration_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: BIC/AIC/entropy for all K values."""
    enum_data = result.get("enumeration", [])
    if not enum_data:
        return

    rows = []
    for r in enum_data:
        rows.append(
            {
                "K": r["k"],
                "BIC": round(r["bic"], 1),
                "AIC": round(r["aic"], 1),
                "Log-Likelihood": round(r["log_likelihood"], 1),
                "Entropy": round(r["entropy"], 3),
                "Converged": "Yes" if r["converged"] else "No",
            }
        )

    df = pl.DataFrame(rows)
    optimal_k = result["optimal_k"]
    html = make_gt(
        df,
        title=f"{chamber} — Class Enumeration (K=1 to K={len(enum_data)})",
        subtitle=f"BIC minimum at K={optimal_k}",
        number_formats={"BIC": ".1f", "AIC": ".1f", "Log-Likelihood": ".1f", "Entropy": ".3f"},
    )
    report.add(
        TableSection(
            id=f"enumeration-{chamber.lower()}",
            title=f"{chamber} Class Enumeration",
            html=html,
        )
    )


def _add_bic_elbow_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"bic_elbow_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-bic-elbow-{chamber.lower()}",
                f"{chamber} BIC/AIC Elbow Plot",
                path,
                caption=(
                    f"Information criteria by number of latent classes ({chamber}). "
                    "Red dashed line marks BIC minimum. Green line shows entropy "
                    "(classification certainty, right axis)."
                ),
                alt_text=(
                    f"Line chart showing BIC and AIC by number of latent classes for {chamber}. "
                    "BIC minimum identifies the optimal class count."
                ),
            )
        )


def _add_optimal_k_summary(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Text: Summary of optimal K selection and its implications."""
    optimal_k = result["optimal_k"]
    rationale = result.get("rationale", "")
    salsa = result.get("salsa", {})
    salsa_verdict = salsa.get("verdict", "")

    lines = [f"<p><strong>Optimal K = {optimal_k}</strong></p>"]
    lines.append(f"<p>{rationale}</p>")

    if optimal_k == 2:
        lines.append(
            "<p><em>Interpretation:</em> BIC selects the two-party split as the only "
            "discrete structure. This confirms Phase 5's finding that within-party "
            "variation is continuous, not factional. No discrete moderate-Republican "
            "or progressive-Democrat blocs exist in the data.</p>"
        )
    elif optimal_k > 2:
        lines.append(
            f"<p><em>Interpretation:</em> BIC selects K={optimal_k}, suggesting "
            "possible sub-party structure. The Salsa effect test below determines "
            "whether these are qualitatively distinct factions or just quantitative "
            "grading along the same ideological dimension.</p>"
        )
        if salsa_verdict:
            lines.append(f"<p>{salsa_verdict}</p>")

    html = "\n".join(lines)
    report.add(
        TextSection(
            id=f"optimal-k-{chamber.lower()}",
            title=f"{chamber} Optimal K Summary",
            html=html,
        )
    )


def _add_composition_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Party composition per class."""
    composition = result.get("composition", [])
    if not composition:
        return

    df = pl.DataFrame(composition)
    html = make_gt(
        df,
        title=f"{chamber} — Class Composition by Party",
        subtitle="Number of legislators per party in each latent class",
        column_labels={
            "class": "Class",
            "n": "Total",
            "Republican": "Republican",
            "Democrat": "Democrat",
            "Independent": "Independent",
        },
    )
    report.add(
        TableSection(
            id=f"composition-{chamber.lower()}",
            title=f"{chamber} Class Composition",
            html=html,
        )
    )


def _add_class_membership_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Every legislator's class assignment, party, and IRT ideal point."""
    membership = result.get("membership", [])
    if not membership:
        return

    rows = []
    for m in membership:
        row = {
            "Name": m["Name"],
            "Party": m["Party"],
            "Class": m["Class"],
            "Max P": round(m["Max P"], 3),
        }
        if m.get("IRT xi") is not None:
            row["IRT xi"] = round(m["IRT xi"], 3)
        else:
            row["IRT xi"] = None
        rows.append(row)

    df = pl.DataFrame(rows)
    optimal_k = result.get("optimal_k", 2)
    html = make_gt(
        df,
        title=f"{chamber} — Class Membership ({len(rows)} legislators)",
        subtitle=(
            f"All legislators assigned to K={optimal_k} latent classes, "
            "sorted by class then IRT ideal point. Max P = classification certainty."
        ),
        number_formats={"IRT xi": ".3f", "Max P": ".3f"},
    )
    report.add(
        TableSection(
            id=f"membership-{chamber.lower()}",
            title=f"{chamber} Class Membership",
            html=html,
        )
    )


def _add_membership_certainty_note(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Add note when all legislators have near-certain class assignments."""
    membership = result.get("membership", [])
    if not membership:
        return

    max_probs = [m.get("Max P", 0) for m in membership]
    if not max_probs:
        return

    all_certain = all(p > 0.99 for p in max_probs)
    if not all_certain:
        return

    n_bills = result.get("n_votes", 0)
    report.add(
        TextSection(
            id=f"membership-certainty-{chamber.lower()}",
            title=f"{chamber} Classification Certainty Note",
            html=(
                f"<p><strong>All {len(max_probs)} legislators</strong> have maximum class "
                f"probability &gt; 0.99. This is mathematically expected with {n_bills}+ "
                f"binary indicators — approximately 30 discriminating bills are sufficient "
                f"for near-certain classification. The absence of uncertain ('straddler') "
                f"legislators does not indicate a model problem; it reflects the high "
                f"dimensionality of the vote matrix relative to the number of latent classes.</p>"
            ),
        )
    )


def _add_profile_heatmap_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"profile_heatmap_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-profile-heatmap-{chamber.lower()}",
                f"{chamber} Class Profiles",
                path,
                caption=(
                    f"P(Yea | Class) for the most discriminating votes ({chamber}). "
                    "Red = high probability of voting Yea, blue = low. Bills with the "
                    "largest range across classes are shown."
                ),
                alt_text=(
                    f"Heatmap of class response profiles for {chamber}. "
                    "Rows are discriminating votes; columns are latent classes. "
                    "Color intensity shows probability of voting Yea."
                ),
            )
        )


def _add_membership_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"membership_hist_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-membership-{chamber.lower()}",
                f"{chamber} Membership Probabilities",
                path,
                caption=(
                    f"Distribution of maximum class membership probabilities ({chamber}). "
                    "Legislators with max P < 0.7 are 'straddlers' — uncertain class "
                    "assignment, potentially cross-cutting voters."
                ),
                alt_text=(
                    "Histogram of maximum class membership probabilities "
                    f"for {chamber}. Most legislators have near-certain "
                    "assignments; low values indicate straddlers."
                ),
            )
        )


def _add_irt_crossval(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: IRT ideal point statistics per class + straddler count."""
    irt_cv = result.get("irt_cv", {})
    class_stats = irt_cv.get("class_stats", [])
    if not class_stats:
        return

    rows = []
    for cs in class_stats:
        rows.append(
            {
                "Class": cs["class"] + 1,
                "N": cs["n"],
                "Mean xi": round(cs["mean_xi"], 3) if cs["mean_xi"] is not None else None,
                "Median xi": round(cs["median_xi"], 3) if cs["median_xi"] is not None else None,
                "SD xi": round(cs["sd_xi"], 3) if cs["sd_xi"] is not None else None,
                "Min xi": round(cs["min_xi"], 3) if cs["min_xi"] is not None else None,
                "Max xi": round(cs["max_xi"], 3) if cs["max_xi"] is not None else None,
            }
        )

    df = pl.DataFrame(rows)
    monotonic = irt_cv.get("is_monotonic", False)
    n_straddlers = irt_cv.get("n_straddlers", 0)

    html = make_gt(
        df,
        title=f"{chamber} — IRT Cross-Validation",
        subtitle=(
            f"IRT ideal points by LCA class. "
            f"Monotonic: {'Yes' if monotonic else 'No'}. "
            f"Straddlers (max P < 0.7): {n_straddlers}."
        ),
        number_formats={
            "Mean xi": ".3f",
            "Median xi": ".3f",
            "SD xi": ".3f",
            "Min xi": ".3f",
            "Max xi": ".3f",
        },
    )
    report.add(
        TableSection(
            id=f"irt-crossval-{chamber.lower()}",
            title=f"{chamber} IRT Cross-Validation",
            html=html,
        )
    )


def _add_irt_boxplot_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"irt_boxplot_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-irt-boxplot-{chamber.lower()}",
                f"{chamber} IRT by LCA Class",
                path,
                caption=(
                    f"IRT ideal points by latent class ({chamber}). "
                    "Points colored by party. If classes are monotonically ordered "
                    "in IRT space, LCA is recovering the same one-dimensional structure."
                ),
                alt_text=(
                    f"Box plot of IRT ideal points grouped by LCA class for {chamber}. "
                    "Points colored by party show whether classes align with ideology."
                ),
            )
        )


def _add_salsa_assessment(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Text: Salsa effect assessment."""
    salsa = result.get("salsa", {})
    verdict = salsa.get("verdict", "")
    if not verdict:
        return

    optimal_k = result.get("optimal_k", 0)
    lines = [f"<p><strong>Salsa Effect Test ({chamber})</strong></p>"]

    if optimal_k <= 2:
        lines.append(
            "<p>With K ≤ 2, the Salsa effect test is not applicable — there is "
            "only one pair of profiles to compare.</p>"
        )
    else:
        lines.append(f"<p>{verdict}</p>")

    mean_corr = salsa.get("mean_correlation")
    min_corr = salsa.get("min_correlation")
    if mean_corr is not None and optimal_k > 1:
        lines.append(
            f"<p>Profile correlations: mean Spearman r = {mean_corr:.3f}, min = {min_corr:.3f}.</p>"
        )

    report.add(
        TextSection(
            id=f"salsa-{chamber.lower()}",
            title=f"{chamber} Salsa Effect Assessment",
            html="\n".join(lines),
        )
    )


def _add_salsa_matrix_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"salsa_matrix_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-salsa-matrix-{chamber.lower()}",
                f"{chamber} Profile Correlation Matrix",
                path,
                caption=(
                    f"Pairwise Spearman correlations between class P(Yea) profiles "
                    f"({chamber}). Values > 0.80 indicate Salsa effect (quantitative "
                    "grading, not qualitative distinction)."
                ),
                alt_text=(
                    "Heatmap of pairwise Spearman correlations between "
                    f"class profiles for {chamber}. "
                    "High correlations indicate classes differ in degree, "
                    "not kind."
                ),
            )
        )


def _add_clustering_agreement(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: ARI between LCA and Phase 5 methods."""
    ari_scores = result.get("ari_scores", {})
    if not ari_scores:
        return

    rows = [{"Method Pair": pair, "ARI": round(ari, 3)} for pair, ari in sorted(ari_scores.items())]

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} — LCA vs Phase 5 Clustering Agreement",
        subtitle="Adjusted Rand Index (ARI). ARI > 0.7 = strong agreement.",
        number_formats={"ARI": ".3f"},
    )
    report.add(
        TableSection(
            id=f"ari-{chamber.lower()}",
            title=f"{chamber} Clustering Agreement",
            html=html,
        )
    )


def _add_within_party_results(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table/text: Within-party LCA results."""
    within_party = result.get("within_party", {})
    if not within_party:
        return

    lines = [f"<p><strong>Within-Party LCA ({chamber})</strong></p>"]

    for party, wp in within_party.items():
        if wp.get("skipped", True):
            lines.append(f"<p><em>{party}:</em> {wp.get('reason', 'Skipped')}</p>")
        else:
            optimal_k = wp.get("optimal_k", 1)
            n = wp.get("n_legislators", 0)
            n_votes = wp.get("n_votes", 0)
            rationale = wp.get("rationale", "")
            lines.append(
                f"<p><em>{party} ({n} legislators, {n_votes} informative votes):</em> "
                f"Optimal K = {optimal_k}. {rationale}</p>"
            )
            if "salsa" in wp:
                lines.append(f"<p>{wp['salsa']['verdict']}</p>")

    report.add(
        TextSection(
            id=f"within-party-{chamber.lower()}",
            title=f"{chamber} Within-Party LCA",
            html="\n".join(lines),
        )
    )


def _add_discriminating_bills(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Most discriminating bills (largest P(Yea) range across classes)."""
    disc_bills = result.get("discriminating_bills", [])
    if not disc_bills:
        return

    optimal_k = result.get("optimal_k", 2)
    rows = []
    for b in disc_bills[:20]:  # Top 20
        row: dict = {
            "Vote ID": b["vote_id"],
            "Range": round(b["range"], 3),
        }
        for c, p in enumerate(b["profiles"]):
            row[f"Class {c + 1}"] = round(p, 3)
        rows.append(row)

    df = pl.DataFrame(rows)
    number_fmts = {"Range": ".3f"}
    for c in range(optimal_k):
        number_fmts[f"Class {c + 1}"] = ".3f"

    html = make_gt(
        df,
        title=f"{chamber} — Most Discriminating Votes",
        subtitle="Votes with the largest P(Yea) range across classes",
        number_formats=number_fmts,
    )
    report.add(
        TableSection(
            id=f"discriminating-{chamber.lower()}",
            title=f"{chamber} Discriminating Votes",
            html=html,
        )
    )


def _generate_lca_key_findings(results: dict[str, dict]) -> list[str]:
    """Generate 2-4 key findings from LCA results."""
    findings: list[str] = []

    for chamber, result in results.items():
        optimal_k = result.get("optimal_k")
        if optimal_k is not None:
            findings.append(
                f"{chamber} BIC selects <strong>K = {optimal_k}</strong> latent classes."
            )

        salsa = result.get("salsa", {})
        if salsa.get("detected"):
            findings.append(
                f"{chamber} <strong>Salsa effect detected</strong> — classes are "
                f"quantitative grading, not qualitatively distinct factions."
            )
        elif optimal_k and optimal_k > 2 and salsa.get("mean_correlation") is not None:
            findings.append(
                f"{chamber} Salsa effect not detected "
                f"(mean profile r = {salsa['mean_correlation']:.3f})."
            )

        break  # First chamber only

    # Clustering agreement
    for chamber, result in results.items():
        ari_scores = result.get("ari_scores", {})
        if ari_scores:
            best_pair = max(ari_scores, key=ari_scores.get)
            best_ari = ari_scores[best_pair]
            findings.append(
                f"Strongest LCA agreement: <strong>{best_pair}</strong> (ARI = {best_ari:.3f})."
            )
            break

    return findings


def _add_analysis_parameters(report: ReportBuilder) -> None:
    """Table: Analysis constants and parameters."""
    from analysis.lca import (
        K_MAX,
        MAX_ITER,
        MIN_CLASS_FRACTION,
        MIN_VOTES,
        MINORITY_THRESHOLD,
        N_INIT,
        RANDOM_SEED,
        SALSA_THRESHOLD,
    )

    rows = [
        {"Parameter": "K_MAX", "Value": str(K_MAX), "Description": "Max classes"},
        {"Parameter": "N_INIT", "Value": str(N_INIT), "Description": "Random starts"},
        {"Parameter": "MAX_ITER", "Value": str(MAX_ITER), "Description": "Max EM iters"},
        {"Parameter": "RANDOM_SEED", "Value": str(RANDOM_SEED), "Description": "Seed"},
        {"Parameter": "MIN_VOTES", "Value": str(MIN_VOTES), "Description": "Min votes"},
        {
            "Parameter": "MINORITY_THRESHOLD",
            "Value": str(MINORITY_THRESHOLD),
            "Description": "Near-unanimous filter",
        },
        {
            "Parameter": "MIN_CLASS_FRACTION",
            "Value": str(MIN_CLASS_FRACTION),
            "Description": "Small class warning",
        },
        {
            "Parameter": "SALSA_THRESHOLD",
            "Value": str(SALSA_THRESHOLD),
            "Description": "Salsa effect threshold",
        },
    ]

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Analysis Parameters",
        subtitle="Constants used in the LCA analysis",
    )
    report.add(
        TableSection(
            id="parameters",
            title="Analysis Parameters",
            html=html,
        )
    )
