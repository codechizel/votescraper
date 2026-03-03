"""HTML report builder for model legislation detection (Phase 20).

Builds a self-contained report with:
  - Key findings summary
  - ALEC match table (interactive, searchable)
  - Cross-state match table (interactive)
  - Similarity distribution histogram
  - Topic heatmap (matches by policy area)
  - Detail cards for near-identical matches
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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


def build_model_legislation_report(
    report: ReportBuilder,
    *,
    results: dict,
    plots_dir: Path,
) -> None:
    """Build the Phase 20 model legislation detection report.

    Args:
        report: ReportBuilder instance from RunContext.
        results: Dict with analysis results (see below for expected keys).
        plots_dir: Path to directory containing generated plots.
    """
    # Key findings
    findings = _generate_key_findings(results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    # Data summary
    _add_data_summary(report, results)

    # ALEC matches
    _add_alec_matches(report, results)

    # Cross-state matches
    _add_cross_state_matches(report, results)

    # Similarity distribution
    _add_similarity_distribution(report, plots_dir)

    # Topic heatmap
    _add_topic_heatmap(report, results, plots_dir)

    # Detail cards for near-identical matches
    _add_detail_cards(report, results)

    # Analysis parameters
    _add_analysis_parameters(report, results)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private Section Builders ────────────────────────────────────────────────


def _generate_key_findings(results: dict) -> list[str]:
    """Extract 3-5 data-driven key findings."""
    findings: list[str] = []

    n_ks = results.get("n_kansas_bills", 0)
    n_alec = results.get("n_alec_bills", 0)
    n_alec_matches = results.get("n_alec_matches", 0)
    n_near_identical = results.get("n_near_identical", 0)
    n_strong = results.get("n_strong_matches", 0)
    n_cross_state_matches = results.get("n_cross_state_matches", 0)
    states_with_matches = results.get("states_with_matches", [])

    findings.append(f"Compared {n_ks} Kansas bills against {n_alec} ALEC model policies")

    if n_alec_matches > 0:
        parts = []
        if n_near_identical > 0:
            parts.append(f"{n_near_identical} near-identical (>= 0.95)")
        if n_strong > 0:
            parts.append(f"{n_strong} strong matches (>= 0.85)")
        total_related = n_alec_matches - n_near_identical - n_strong
        if total_related > 0:
            parts.append(f"{total_related} related (>= 0.70)")
        findings.append(f"Found {n_alec_matches} ALEC matches: {'; '.join(parts)}")
    else:
        findings.append("No Kansas bills matched ALEC model policies above the 0.70 threshold")

    if n_cross_state_matches > 0:
        findings.append(
            f"Found {n_cross_state_matches} cross-state matches "
            f"across {', '.join(s.upper() for s in states_with_matches)}"
        )

    # Top ALEC match highlight
    top_alec = results.get("top_alec_match")
    if top_alec:
        findings.append(
            f"Strongest ALEC match: {top_alec['ks_bill']} — "
            f'"{top_alec["match_label"]}" (similarity: {top_alec["similarity"]:.3f})'
        )

    return findings


def _add_data_summary(report: ReportBuilder, results: dict) -> None:
    """Add data summary table."""
    import polars as pl

    rows = [
        ("Kansas bills analyzed", str(results.get("n_kansas_bills", 0))),
        ("ALEC model bills", str(results.get("n_alec_bills", 0))),
        ("Embedding model", str(results.get("embedding_model", ""))),
        ("Embedding dimensions", str(results.get("embedding_dim", ""))),
        ("Similarity threshold", str(results.get("threshold", 0.70))),
        ("N-gram size", str(results.get("ngram_size", 5))),
    ]

    # Add cross-state info
    states = results.get("cross_states", [])
    if states:
        for state in states:
            n = results.get(f"n_{state}_bills", 0)
            rows.append((f"{state.upper()} bills", str(n)))

    df = pl.DataFrame({"Metric": [r[0] for r in rows], "Value": [r[1] for r in rows]})
    html = make_gt(df, title="Data Summary")
    report.add(TableSection(id="data-summary", title="Data Summary", html=html))


def _add_alec_matches(report: ReportBuilder, results: dict) -> None:
    """Add ALEC match table (interactive, searchable)."""
    import polars as pl

    alec_summary = results.get("alec_match_summary")
    if alec_summary is None or len(alec_summary) == 0:
        report.add(
            TextSection(
                id="alec-matches",
                title="ALEC Matches",
                html="<p>No Kansas bills matched ALEC model policies above the threshold.</p>",
            )
        )
        return

    # Format for display
    display_df = alec_summary.select(
        [
            pl.col("ks_bill").alias("Kansas Bill"),
            pl.col("match_label").alias("ALEC Model Bill"),
            pl.col("similarity").round(3).alias("Similarity"),
            pl.col("match_tier").alias("Match Tier"),
            pl.col("ngram_overlap").round(3).alias("5-gram Overlap"),
            pl.col("topic").alias("Topic"),
        ]
    )

    html = make_interactive_table(
        display_df,
        title="ALEC Model Legislation Matches",
        number_formats={"Similarity": "{:.3f}", "5-gram Overlap": "{:.3f}"},
        caption="Kansas bills matched against ALEC model policy corpus. "
        "Sorted by similarity (highest first).",
    )
    report.add(
        InteractiveTableSection(
            id="alec-matches",
            title="ALEC Matches",
            html=html,
        )
    )


def _add_cross_state_matches(report: ReportBuilder, results: dict) -> None:
    """Add cross-state match table."""
    import polars as pl

    cross_summary = results.get("cross_state_match_summary")
    if cross_summary is None or len(cross_summary) == 0:
        report.add(
            TextSection(
                id="cross-state-matches",
                title="Cross-State Matches",
                html="<p>No cross-state matches found above the threshold.</p>",
            )
        )
        return

    display_df = cross_summary.select(
        [
            pl.col("ks_bill").alias("Kansas Bill"),
            pl.col("source").alias("State"),
            pl.col("match_id").alias("Match Bill"),
            pl.col("similarity").round(3).alias("Similarity"),
            pl.col("match_tier").alias("Match Tier"),
            pl.col("topic").alias("Topic"),
        ]
    )

    html = make_interactive_table(
        display_df,
        title="Cross-State Bill Matches",
        number_formats={"Similarity": "{:.3f}"},
        caption="Kansas bills matched against neighbor state bills. "
        "Sorted by similarity (highest first).",
    )
    report.add(
        InteractiveTableSection(
            id="cross-state-matches",
            title="Cross-State Matches",
            html=html,
        )
    )


def _add_similarity_distribution(report: ReportBuilder, plots_dir: Path) -> None:
    """Add similarity distribution histogram."""
    hist_path = plots_dir / "similarity_distribution.png"
    if hist_path.exists():
        report.add(
            FigureSection.from_file(
                id="similarity-dist",
                title="Similarity Distribution",
                path=hist_path,
                caption="Distribution of maximum similarity scores per Kansas bill "
                "across all comparison corpora (ALEC + neighbor states).",
                alt_text="Histogram showing the distribution of maximum cosine similarity "
                "scores for each Kansas bill against ALEC and neighbor state bills.",
            )
        )


def _add_topic_heatmap(report: ReportBuilder, results: dict, plots_dir: Path) -> None:
    """Add topic heatmap showing matches by policy area."""
    heatmap_path = plots_dir / "topic_match_heatmap.png"
    if heatmap_path.exists():
        report.add(
            FigureSection.from_file(
                id="topic-heatmap",
                title="Matches by Policy Area",
                path=heatmap_path,
                caption="Count of model legislation matches by BERTopic/CAP policy area.",
                alt_text="Heatmap showing which policy areas have the most model "
                "legislation matches across ALEC and neighbor states.",
            )
        )

    # Also add match network if available
    network_path = plots_dir / "match_network.png"
    if network_path.exists():
        report.add(
            FigureSection.from_file(
                id="match-network",
                title="Multi-State Match Network",
                path=network_path,
                caption="Kansas bills matched in 2+ states, showing diffusion patterns.",
                alt_text="Network diagram showing Kansas bills that appear in multiple "
                "neighbor states, with edges weighted by cosine similarity.",
            )
        )


def _add_detail_cards(report: ReportBuilder, results: dict) -> None:
    """Add detail cards for near-identical matches (>= 0.95)."""
    near_identical = results.get("near_identical_details", [])
    if not near_identical:
        return

    cards_html = ['<div class="detail-cards">']
    for match in near_identical:
        ks_bill = match.get("ks_bill", "")
        match_label = match.get("match_label", "")
        similarity = match.get("similarity", 0.0)
        source = match.get("source", "")
        ngram = match.get("ngram_overlap")
        ks_excerpt = match.get("ks_excerpt", "")
        match_excerpt = match.get("match_excerpt", "")

        ngram_str = f" | 5-gram overlap: {ngram:.1%}" if ngram is not None else ""

        cards_html.append(f"""
<div class="match-card" style="border: 1px solid #ccc; margin: 1em 0; padding: 1em;
  border-radius: 4px; background: #fafafa;">
  <h4>{ks_bill} &larr; {source}: {match_label}</h4>
  <p><strong>Similarity: {similarity:.3f}</strong>{ngram_str}</p>
  <div style="display: flex; gap: 1em;">
    <div style="flex: 1;">
      <h5>Kansas ({ks_bill})</h5>
      <pre style="white-space: pre-wrap; font-size: 0.85em; max-height: 200px;
        overflow-y: auto; background: #f0f0f0; padding: 0.5em;">{ks_excerpt}</pre>
    </div>
    <div style="flex: 1;">
      <h5>{match_label}</h5>
      <pre style="white-space: pre-wrap; font-size: 0.85em; max-height: 200px;
        overflow-y: auto; background: #f0f0f0; padding: 0.5em;">{match_excerpt}</pre>
    </div>
  </div>
</div>""")

    cards_html.append("</div>")

    report.add(
        TextSection(
            id="near-identical-details",
            title="Near-Identical Match Details",
            html="\n".join(cards_html),
            caption="Side-by-side text excerpts for near-identical matches (>= 0.95 similarity).",
        )
    )


def _add_analysis_parameters(report: ReportBuilder, results: dict) -> None:
    """Add analysis parameters section."""
    params = results.get("parameters", {})
    if not params:
        return

    import polars as pl

    rows = [(k, str(v)) for k, v in sorted(params.items())]
    df = pl.DataFrame({"Parameter": [r[0] for r in rows], "Value": [r[1] for r in rows]})
    html = make_gt(df, title="Analysis Parameters")
    report.add(TableSection(id="parameters", title="Analysis Parameters", html=html))
