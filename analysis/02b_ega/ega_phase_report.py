"""HTML report builder for EGA Phase 02b."""

from pathlib import Path

try:
    from analysis.report import (
        FigureSection,
        KeyFindingsSection,
        ReportBuilder,
        TextSection,
    )
except ModuleNotFoundError:
    from report import (  # type: ignore[no-redef]
        FigureSection,
        KeyFindingsSection,
        ReportBuilder,
        TextSection,
    )


def build_ega_report(
    all_results: dict[str, dict],
    plots_dir: Path,
    data_dir: Path,
    run_dir: Path,
) -> Path:
    """Build the EGA HTML report from phase results."""
    report = ReportBuilder(title="Exploratory Graph Analysis (EGA)")

    # Key findings
    findings: list[str] = []
    for chamber, res in all_results.items():
        summary = res["summary"]
        k = summary["n_communities"]
        uni = summary["unidimensional"]
        edges = summary["glasso_n_edges"]
        findings.append(
            f"<b>{chamber.title()}</b>: EGA finds <b>K={k}</b> dimension{'s' if k != 1 else ''}"
            f" ({edges} network edges)" + (" — unidimensional check passed" if uni else "")
        )
        if "boot_modal_k" in summary:
            findings.append(
                f"  bootEGA modal K={summary['boot_modal_k']} "
                f"({summary['boot_n']} replicates, "
                f"{summary.get('n_unstable_items', 0)} unstable items)"
            )
        findings.append(f"  TEFI best K={summary['tefi_best_k']}")
        if "uva_n_redundant_pairs" in summary:
            findings.append(
                f"  UVA: {summary['uva_n_redundant_pairs']} redundant pairs, "
                f"{summary['uva_n_suggested_removals']} suggested removals"
            )

    report.add(KeyFindingsSection(findings=findings))

    # Per-chamber sections
    for chamber, res in all_results.items():
        ch = chamber.title()

        # Network plot
        net_path = plots_dir / f"ega_network_{chamber}.png"
        if net_path.exists():
            report.add(
                FigureSection.from_file(
                    id=f"network-{chamber}",
                    title=f"GLASSO Network — {ch}",
                    path=net_path,
                    alt_text=f"GLASSO partial correlation network for {ch} colored by community",
                )
            )

        # TEFI curve
        tefi_path = plots_dir / f"tefi_curve_{chamber}.png"
        if tefi_path.exists():
            report.add(
                FigureSection.from_file(
                    id=f"tefi-{chamber}",
                    title=f"TEFI Dimensionality Comparison — {ch}",
                    path=tefi_path,
                    alt_text=f"TEFI values across K=1 to K=5 for {ch}, lower is better",
                )
            )

        # Bootstrap histogram
        boot_path = plots_dir / f"boot_k_histogram_{chamber}.png"
        if boot_path.exists():
            report.add(
                FigureSection.from_file(
                    id=f"boot-{chamber}",
                    title=f"bootEGA Dimension Frequency — {ch}",
                    path=boot_path,
                    alt_text=f"Histogram of K values found across bootstrap replicates for {ch}",
                )
            )

        # Item stability
        stab_path = plots_dir / f"item_stability_{chamber}.png"
        if stab_path.exists():
            report.add(
                FigureSection.from_file(
                    id=f"stability-{chamber}",
                    title=f"Item Stability — {ch}",
                    path=stab_path,
                    alt_text=f"Per-bill stability from bootEGA for {ch}, red below 0.70",
                )
            )

    # Method details
    method_html = """
    <h3>Method Details</h3>
    <p><b>Exploratory Graph Analysis (EGA)</b> estimates dimensionality via:</p>
    <ol>
    <li><b>Tetrachoric correlations</b> — correct for binary (Yea/Nay) data</li>
    <li><b>GLASSO + EBIC</b> — sparse partial correlation network
        (conditional dependencies, not marginal)</li>
    <li><b>Community detection</b> — Walktrap or Leiden on the GLASSO network</li>
    <li><b>Unidimensional check</b> — Louvain on zero-order correlations</li>
    </ol>
    <p><b>bootEGA</b> runs 500 bootstrap replicates to assess stability.
    Items with stability &lt; 0.70 are dimensionally ambiguous.</p>
    <p><b>TEFI</b> (Total Entropy Fit Index) uses Von Neumann entropy to
    compare dimensional structures. Lower = better fit.</p>
    <p><b>UVA</b> (Unique Variable Analysis) detects redundant bill pairs
    via weighted topological overlap (wTO &gt; threshold).</p>
    <p>Reference: Golino et al. (2020), <i>Psychological Methods</i>, 25(3).</p>
    """
    report.add(TextSection(id="methods", title="Methods", html=method_html))

    report_path = run_dir / "02b_ega_report.html"
    report.write(report_path)
    return report_path
