"""Bill text analysis HTML report builder (Phase 18).

Builds ~13 sections: key findings, data summary, topic distribution,
topic keywords, party × topic heatmap, caucus-splitting topics,
CAP classification (conditional), bill similarity, NMF comparison,
analysis parameters, and optionally bill summaries.

CAP sections (7-9, 13) are conditionally rendered only when CAP
classification was performed (requires ANTHROPIC_API_KEY).

Usage (called from bill_text.py):
    from analysis.bill_text_report import build_bill_text_report
    build_bill_text_report(ctx.report, results=results, plots_dir=plots_dir)
"""

from pathlib import Path

import polars as pl

try:
    from analysis.report import (
        FigureSection,
        InteractiveTableSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
        make_interactive_table,
    )
except ModuleNotFoundError:
    from report import (  # type: ignore[no-redef]
        FigureSection,
        InteractiveTableSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
        make_interactive_table,
    )


def build_bill_text_report(
    report: ReportBuilder,
    *,
    results: dict,
    plots_dir: Path,
    has_cap: bool = False,
) -> None:
    """Build the full Phase 18 HTML report."""
    findings = _generate_key_findings(results, has_cap)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_data_summary(report, results)
    _add_topic_distribution(report, results, plots_dir)
    _add_topic_keywords(report, results)
    _add_topic_party_heatmap(report, results, plots_dir)
    _add_caucus_splitting(report, results, plots_dir)

    if has_cap:
        _add_cap_distribution(report, results, plots_dir)
        _add_cap_party_breakdown(report, results, plots_dir)
        _add_cap_passage_rate(report, results)

    _add_bill_similarity(report, results, plots_dir)
    _add_nmf_comparison(report, results)
    _add_analysis_parameters(report, results)

    if has_cap and "cap_classifications" in results:
        _add_bill_summaries(report, results)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ──────────────────────────────────────────────────


def _generate_key_findings(results: dict, has_cap: bool) -> list[str]:
    """Extract 3-5 data-driven findings for KeyFindingsSection."""
    findings: list[str] = []

    # Number of topics discovered
    n_topics = results.get("n_topics", 0)
    n_bills = results.get("n_bills_analyzed", 0)
    if n_topics > 0:
        findings.append(
            f"BERTopic discovered <strong>{n_topics} topics</strong> across "
            f"{n_bills} bills using density-based clustering on text embeddings."
        )

    # Most caucus-splitting topic
    if "caucus_splitting" in results and len(results["caucus_splitting"]) > 0:
        top_split = results["caucus_splitting"][0]
        findings.append(
            f"Most caucus-splitting topic: <strong>{top_split['topic_label']}</strong> "
            f"(majority-party cohesion = {top_split['majority_rice']:.0%})."
        )

    # Top CAP category
    if has_cap and "cap_classifications" in results:
        cap_df = results["cap_classifications"]
        if len(cap_df) > 0:
            top_cat = cap_df.group_by("cap_label").len().sort("len", descending=True).row(0)
            findings.append(
                f"Most common policy area: <strong>{top_cat[0]}</strong> "
                f"({top_cat[1]} bills, {top_cat[1] / len(cap_df):.0%} of classified bills)."
            )

    # Text source breakdown
    if "text_source_counts" in results:
        sc = results["text_source_counts"]
        supp = sc.get("supp_note", 0)
        intro = sc.get("introduced", 0)
        if supp > 0:
            findings.append(
                f"Text sources: {supp} supplemental notes, {intro} introduced texts "
                f"(supplemental notes preferred for NLP — shorter, plain-English summaries)."
            )

    return findings


def _add_data_summary(report: ReportBuilder, results: dict) -> None:
    """Table: bills analyzed, text sources, average length."""
    rows = [
        {
            "Metric": "Bills analyzed",
            "Value": str(results.get("n_bills_analyzed", 0)),
        },
        {
            "Metric": "Supplemental notes used",
            "Value": str(results.get("text_source_counts", {}).get("supp_note", 0)),
        },
        {
            "Metric": "Introduced texts used",
            "Value": str(results.get("text_source_counts", {}).get("introduced", 0)),
        },
        {
            "Metric": "Avg text length (chars)",
            "Value": f"{results.get('avg_text_length', 0):,.0f}",
        },
        {
            "Metric": "Embedding model",
            "Value": results.get("embedding_model", "BAAI/bge-small-en-v1.5"),
        },
        {
            "Metric": "Embedding dimensions",
            "Value": str(results.get("embedding_dim", 384)),
        },
    ]

    df = pl.DataFrame(rows)
    html = make_gt(df, title="Data Summary", subtitle="Bill text corpus overview")
    report.add(TableSection(id="data-summary", title="Data Summary", html=html))


def _add_topic_distribution(
    report: ReportBuilder,
    results: dict,
    plots_dir: Path,
) -> None:
    """Figure: topic distribution bar chart."""
    for suffix in ["all", "house", "senate"]:
        path = plots_dir / f"topic_distribution_{suffix}.png"
        if path.exists():
            label = {"all": "All Bills", "house": "House", "senate": "Senate"}[suffix]
            report.add(
                FigureSection.from_file(
                    id=f"topic-dist-{suffix}",
                    title=f"Topic Distribution — {label}",
                    path=path,
                    caption=f"Number of bills per BERTopic cluster ({label}).",
                    alt_text=(
                        f"Bar chart showing the number of bills assigned to each "
                        f"BERTopic topic for {label} bills."
                    ),
                )
            )


def _add_topic_keywords(report: ReportBuilder, results: dict) -> None:
    """Table: top words per topic."""
    topic_info = results.get("topic_info")
    if topic_info is None or len(topic_info) == 0:
        return

    if isinstance(topic_info, list):
        df = pl.DataFrame(topic_info)
    else:
        df = topic_info

    if len(df) == 0:
        return

    # Ensure we have the right columns
    display_cols = [c for c in ["topic_id", "topic_label", "count", "top_words"] if c in df.columns]
    if not display_cols:
        return

    df_display = df.select(display_cols)
    html = make_gt(
        df_display,
        title="Topic Keywords",
        subtitle="Top c-TF-IDF words per topic (BERTopic)",
        column_labels={
            "topic_id": "Topic",
            "topic_label": "Label",
            "count": "Bills",
            "top_words": "Top Words",
        },
    )
    report.add(TableSection(id="topic-keywords", title="Topic Keywords", html=html))


def _add_topic_party_heatmap(
    report: ReportBuilder,
    results: dict,
    plots_dir: Path,
) -> None:
    """Figure: topic × party cohesion heatmap."""
    for suffix in ["house", "senate"]:
        path = plots_dir / f"topic_party_heatmap_{suffix}.png"
        if path.exists():
            chamber = suffix.title()
            report.add(
                FigureSection.from_file(
                    id=f"topic-party-{suffix}",
                    title=f"Topic × Party Cohesion — {chamber}",
                    path=path,
                    caption=(
                        f"Heatmap of Rice index per topic and party ({chamber}). "
                        f"Lower values indicate topics that split the party internally."
                    ),
                    alt_text=(
                        f"Heatmap showing Rice index cohesion scores for each topic "
                        f"by political party in the {chamber}."
                    ),
                )
            )


def _add_caucus_splitting(
    report: ReportBuilder,
    results: dict,
    plots_dir: Path,
) -> None:
    """Table + figure: topics ranked by intra-majority-party dissent."""
    splitting = results.get("caucus_splitting")
    if not splitting:
        return

    rows = []
    for item in splitting:
        rows.append(
            {
                "Topic": item["topic_label"],
                "Bills": item["n_bills"],
                "Majority Rice": f"{item['majority_rice']:.2%}",
                "Minority Rice": f"{item['minority_rice']:.2%}",
                "Split Score": f"{item['split_score']:.2%}",
            }
        )

    if rows:
        df = pl.DataFrame(rows)
        html = make_gt(
            df,
            title="Caucus-Splitting Topics",
            subtitle=(
                "Topics ranked by how much they split the majority party "
                "(Split Score = 1 - majority Rice index)"
            ),
        )
        report.add(
            TableSection(
                id="caucus-splitting",
                title="Caucus-Splitting Topics",
                html=html,
            )
        )

    for suffix in ["house", "senate"]:
        path = plots_dir / f"caucus_splitting_topics_{suffix}.png"
        if path.exists():
            chamber = suffix.title()
            report.add(
                FigureSection.from_file(
                    id=f"caucus-split-fig-{suffix}",
                    title=f"Caucus-Splitting Topics — {chamber}",
                    path=path,
                    caption=f"Topics ranked by majority-party dissent ({chamber}).",
                    alt_text=(
                        f"Horizontal bar chart showing topics ranked by caucus-splitting "
                        f"score in the {chamber}."
                    ),
                )
            )


def _add_cap_distribution(
    report: ReportBuilder,
    results: dict,
    plots_dir: Path,
) -> None:
    """Figure: CAP 20-category distribution."""
    path = plots_dir / "cap_category_distribution.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id="cap-dist",
                title="CAP Policy Classification",
                path=path,
                caption=(
                    "Distribution of bills across 20 Comparative Agendas Project categories, "
                    "classified by Claude Sonnet."
                ),
                alt_text=(
                    "Bar chart showing the number of bills classified into each of the "
                    "20 CAP major topic categories."
                ),
            )
        )


def _add_cap_party_breakdown(
    report: ReportBuilder,
    results: dict,
    plots_dir: Path,
) -> None:
    """Figure: CAP categories by sponsor party."""
    path = plots_dir / "cap_party_breakdown.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id="cap-party",
                title="CAP Categories by Sponsor Party",
                path=path,
                caption="Which policy areas does each party sponsor?",
                alt_text=(
                    "Stacked bar chart showing CAP categories broken down by "
                    "sponsor party (Republican, Democrat, other)."
                ),
            )
        )


def _add_cap_passage_rate(report: ReportBuilder, results: dict) -> None:
    """Table: passage rate per CAP category."""
    cap_passage = results.get("cap_passage_rates")
    if cap_passage is None or len(cap_passage) == 0:
        return

    if isinstance(cap_passage, list):
        df = pl.DataFrame(cap_passage)
    else:
        df = cap_passage

    html = make_gt(
        df,
        title="CAP Category Passage Rates",
        subtitle="Which policy areas pass? Which die?",
        column_labels={
            "cap_label": "Category",
            "n_bills": "Bills",
            "n_votes": "Roll Calls",
            "passage_rate": "Passage Rate",
        },
        number_formats={"passage_rate": ".0%"},
    )
    report.add(TableSection(id="cap-passage", title="CAP Passage Rates", html=html))


def _add_bill_similarity(
    report: ReportBuilder,
    results: dict,
    plots_dir: Path,
) -> None:
    """Figure: clustered similarity heatmap."""
    path = plots_dir / "bill_similarity_heatmap.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id="similarity",
                title="Bill Similarity Clusters",
                path=path,
                caption=(
                    "Cosine similarity matrix of bill text embeddings, clustered "
                    "hierarchically.  Bright squares indicate semantically similar bills."
                ),
                alt_text=(
                    "Heatmap showing pairwise cosine similarity between bill text "
                    "embeddings, with hierarchical clustering applied."
                ),
            )
        )

    # Top similar pairs table
    sim_pairs = results.get("top_similar_pairs")
    if sim_pairs is not None and len(sim_pairs) > 0:
        if isinstance(sim_pairs, list):
            df = pl.DataFrame(sim_pairs)
        else:
            df = sim_pairs

        html = make_gt(
            df,
            title="Most Similar Bill Pairs",
            subtitle="Top bill pairs by cosine similarity of text embeddings",
            column_labels={
                "bill_a": "Bill A",
                "bill_b": "Bill B",
                "similarity": "Cosine Similarity",
            },
            number_formats={"similarity": ".3f"},
        )
        report.add(TableSection(id="sim-pairs", title="Most Similar Bill Pairs", html=html))


def _add_nmf_comparison(report: ReportBuilder, results: dict) -> None:
    """Text: how BERTopic compares to Phase 08 NMF topics."""
    nmf_comparison = results.get("nmf_comparison_text")
    if not nmf_comparison:
        nmf_comparison = (
            "<p>Phase 08 (Prediction) uses NMF with K=6 fixed topics on short bill titles. "
            "BERTopic uses full bill text with automatic topic discovery via density-based "
            "clustering. BERTopic typically finds more nuanced topics because it operates "
            "on longer, richer text and does not require a fixed K.</p>"
        )

    html = f"""<div style="padding: 1em; background: #f8f9fa; border-radius: 6px;">
<h4>BERTopic vs NMF (Phase 08)</h4>
{nmf_comparison}
</div>"""
    report.add(TextSection(id="nmf-comparison", title="Comparison to NMF Topics", html=html))


def _add_analysis_parameters(report: ReportBuilder, results: dict) -> None:
    """Text: all analysis parameters for reproducibility."""
    params = results.get("parameters", {})
    rows_html = ""
    for key, value in sorted(params.items()):
        rows_html += f"<tr><td><code>{key}</code></td><td>{value}</td></tr>\n"

    html = f"""<div style="padding: 1em; background: #fafafa; border: 1px solid #ddd;
    border-radius: 6px;">
<h4>Analysis Parameters</h4>
<table style="width: 100%; border-collapse: collapse;">
<thead><tr><th style="text-align: left; padding: 4px;">Parameter</th>
<th style="text-align: left; padding: 4px;">Value</th></tr></thead>
<tbody>
{rows_html}
</tbody></table>
</div>"""
    report.add(TextSection(id="parameters", title="Analysis Parameters", html=html))


def _add_bill_summaries(report: ReportBuilder, results: dict) -> None:
    """Interactive table: Claude-generated 1-sentence summaries per bill."""
    cap_df = results.get("cap_classifications")
    if cap_df is None or len(cap_df) == 0:
        return

    if isinstance(cap_df, pl.DataFrame):
        display_df = cap_df.select(
            [
                "bill_number",
                "cap_label",
                "cap_confidence",
                "bill_summary",
            ]
        )
    else:
        return

    html = make_interactive_table(
        display_df,
        caption="Bill summaries generated by Claude Sonnet during CAP classification.",
        column_labels={
            "bill_number": "Bill",
            "cap_label": "Category",
            "cap_confidence": "Confidence",
            "bill_summary": "Summary",
        },
    )
    report.add(
        InteractiveTableSection(
            id="bill-summaries",
            title="Bill Summaries",
            html=html,
        )
    )
