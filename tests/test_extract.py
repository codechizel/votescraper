"""Tests for bespoke report extraction."""

from pathlib import Path

import pytest

from tallgrass.extract.extractor import (
    _collect_and_strip_dependencies,
    extract_sections,
    generate_slug,
    infer_phase_name,
    list_sections,
    parse_report_css,
    parse_report_session,
    parse_report_title,
    render_extracted,
    write_extracted,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

MINIMAL_REPORT = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>06_IRT_2D Report — 91st_2025-2026</title>
</head>
<body>
  <header>
    <h1>06_IRT_2D Report</h1>
    <div class="meta">
      <span>Session: <strong>91st_2025-2026</strong></span>
      <span>Generated: 2026-03-09 10:00 CDT</span>
    </div>
  </header>

  <section class="report-section" id="experimental-status">
    <h2><span class="section-number">1.</span> Experimental Status</h2>
    <div class="text-container">This is experimental.</div>
  </section>

  <section class="report-section" id="model-spec">
    <h2><span class="section-number">2.</span> Model Specification</h2>
    <div class="text-container">M2PL model details.</div>
  </section>

  <section class="report-section" id="convergence-house">
    <h2><span class="section-number">3.</span> Convergence — House</h2>
    <div class="table-container">Convergence table here.</div>
  </section>

  <footer>Test footer</footer>
</body>
</html>"""


PLOTLY_SECTION_CONTENT = """\
    <div class="interactive-container">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <div id="plotly-div-1">chart 1</div>
    </div>"""

PLOTLY_SECTION_CONTENT_2 = """\
    <div class="interactive-container">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <div id="plotly-div-2">chart 2</div>
    </div>"""


REPORT_WITH_PLOTLY = (
    """\
<!DOCTYPE html>
<html lang="en">
<head><title>Plotly Report</title></head>
<body>
  <header>
    <div class="meta">
      <span>Session: <strong>91st_2025-2026</strong></span>
    </div>
  </header>

  <section class="report-section" id="scatter-house">
    <h2><span class="section-number">1.</span> Scatter House</h2>
"""
    + PLOTLY_SECTION_CONTENT
    + """
  </section>

  <section class="report-section" id="scatter-senate">
    <h2><span class="section-number">2.</span> Scatter Senate</h2>
"""
    + PLOTLY_SECTION_CONTENT_2
    + """
  </section>
</body>
</html>"""
)


REPORT_WITH_DATATABLES = """\
<!DOCTYPE html>
<html lang="en">
<head><title>DataTables Report</title></head>
<body>
  <header>
    <div class="meta">
      <span>Session: <strong>91st_2025-2026</strong></span>
    </div>
  </header>

  <section class="report-section" id="table-section">
    <h2><span class="section-number">1.</span> Interactive Table</h2>
    <link href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css" rel="stylesheet" />
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <table>data</table>
  </section>
</body>
</html>"""


# ── Parsing Tests ────────────────────────────────────────────────────────────


class TestListSections:
    def test_parses_all_sections(self):
        sections = list_sections(MINIMAL_REPORT)
        assert len(sections) == 3

    def test_section_ids(self):
        sections = list_sections(MINIMAL_REPORT)
        assert [s.id for s in sections] == [
            "experimental-status",
            "model-spec",
            "convergence-house",
        ]

    def test_section_numbers(self):
        sections = list_sections(MINIMAL_REPORT)
        assert [s.number for s in sections] == [1, 2, 3]

    def test_section_titles(self):
        sections = list_sections(MINIMAL_REPORT)
        assert sections[0].title == "Experimental Status"
        assert sections[2].title == "Convergence — House"

    def test_section_content(self):
        sections = list_sections(MINIMAL_REPORT)
        assert "This is experimental." in sections[0].content
        assert "M2PL model details." in sections[1].content

    def test_empty_html(self):
        assert list_sections("") == []
        assert list_sections("<html><body></body></html>") == []


class TestParseReportTitle:
    def test_extracts_title(self):
        assert parse_report_title(MINIMAL_REPORT) == "06_IRT_2D Report — 91st_2025-2026"

    def test_missing_title(self):
        assert parse_report_title("<html></html>") == "Unknown Report"


class TestParseReportSession:
    def test_extracts_session(self):
        assert parse_report_session(MINIMAL_REPORT) == "91st_2025-2026"

    def test_missing_session(self):
        assert parse_report_session("<html></html>") == ""


class TestParseReportCSS:
    def test_extracts_css(self):
        html = "<html><head><style>body { color: red; }</style></head></html>"
        assert parse_report_css(html) == "body { color: red; }"

    def test_missing_style(self):
        assert parse_report_css("<html></html>") == ""

    def test_extracts_from_report(self):
        css = parse_report_css(MINIMAL_REPORT)
        # MINIMAL_REPORT has no <style> tag.
        assert css == ""

    def test_extracts_from_styled_report(self):
        styled = MINIMAL_REPORT.replace("<head>", "<head><style>.test { margin: 0; }</style>")
        css = parse_report_css(styled)
        assert ".test" in css


class TestInferPhaseName:
    def test_standard_path(self):
        assert infer_phase_name(Path("results/06_irt_2d/report.html")) == "06_irt_2d"

    def test_nested_path(self):
        p = Path("results/kansas/91st_2025-2026/91-260307.1/06_irt_2d/06_irt_2d_report.html")
        assert infer_phase_name(p) == "06_irt_2d"


# ── Extraction Tests ─────────────────────────────────────────────────────────


class TestExtractSections:
    def test_extract_by_number(self):
        source = Path("results/06_irt_2d/report.html")
        result = extract_sections(MINIMAL_REPORT, ["1"], source)
        assert len(result) == 1
        assert result[0].id == "experimental-status"
        assert result[0].number == 1

    def test_extract_by_id(self):
        source = Path("results/06_irt_2d/report.html")
        result = extract_sections(MINIMAL_REPORT, ["model-spec"], source)
        assert len(result) == 1
        assert result[0].title == "Model Specification"

    def test_extract_multiple(self):
        source = Path("results/06_irt_2d/report.html")
        result = extract_sections(MINIMAL_REPORT, ["1", "3"], source)
        assert len(result) == 2
        assert result[0].id == "experimental-status"
        assert result[1].id == "convergence-house"

    def test_preserves_order(self):
        source = Path("results/06_irt_2d/report.html")
        result = extract_sections(MINIMAL_REPORT, ["3", "1"], source)
        assert result[0].id == "convergence-house"
        assert result[1].id == "experimental-status"

    def test_unknown_section_raises(self):
        source = Path("results/06_irt_2d/report.html")
        with pytest.raises(ValueError, match="Section 'nonexistent' not found"):
            extract_sections(MINIMAL_REPORT, ["nonexistent"], source)

    def test_unknown_number_raises(self):
        source = Path("results/06_irt_2d/report.html")
        with pytest.raises(ValueError, match="Section '99' not found"):
            extract_sections(MINIMAL_REPORT, ["99"], source)

    def test_source_metadata(self):
        source = Path("results/06_irt_2d/report.html")
        result = extract_sections(MINIMAL_REPORT, ["1"], source)
        assert result[0].source_report == "06_irt_2d"
        assert result[0].source_session == "91st_2025-2026"
        assert result[0].source_path == source


# ── Dependency Deduplication Tests ───────────────────────────────────────────


class TestDependencyDedup:
    def test_plotly_dedup(self):
        source = Path("results/06_irt_2d/report.html")
        sections = extract_sections(REPORT_WITH_PLOTLY, ["1", "2"], source)
        cleaned, deps = _collect_and_strip_dependencies(sections)

        # One Plotly script in head deps.
        plotly_deps = [d for d in deps if "plotly" in d]
        assert len(plotly_deps) == 1

        # No Plotly scripts left in section content.
        for s in cleaned:
            assert "cdn.plot.ly" not in s.content

        # Chart content preserved.
        assert "plotly-div-1" in cleaned[0].content
        assert "plotly-div-2" in cleaned[1].content

    def test_datatables_dedup(self):
        source = Path("results/01_eda/report.html")
        sections = extract_sections(REPORT_WITH_DATATABLES, ["1"], source)
        cleaned, deps = _collect_and_strip_dependencies(sections)

        # DataTables script and CSS in head deps.
        dt_deps = [d for d in deps if "datatables" in d.lower()]
        assert len(dt_deps) == 2  # one CSS, one JS

        # No DataTables tags left in section content.
        for s in cleaned:
            assert "datatables" not in s.content.lower()

    def test_no_deps_returns_empty(self):
        source = Path("results/06_irt_2d/report.html")
        sections = extract_sections(MINIMAL_REPORT, ["1"], source)
        cleaned, deps = _collect_and_strip_dependencies(sections)
        assert deps == []
        assert len(cleaned) == 1


# ── Rendering Tests ──────────────────────────────────────────────────────────


class TestRenderExtracted:
    def test_single_section_no_toc(self):
        source = Path("results/06_irt_2d/report.html")
        sections = extract_sections(MINIMAL_REPORT, ["1"], source)
        html = render_extracted(sections, title="Test")
        assert '<nav class="toc">' not in html

    def test_multi_section_has_toc(self):
        source = Path("results/06_irt_2d/report.html")
        sections = extract_sections(MINIMAL_REPORT, ["1", "2"], source)
        html = render_extracted(sections, title="Test")
        assert '<nav class="toc">' in html
        assert "Experimental Status" in html
        assert "Model Specification" in html

    def test_preserves_original_numbering(self):
        source = Path("results/06_irt_2d/report.html")
        sections = extract_sections(MINIMAL_REPORT, ["3"], source)
        html = render_extracted(sections, title="Test")
        assert '<span class="section-number">3.</span>' in html

    def test_custom_title(self):
        source = Path("results/06_irt_2d/report.html")
        sections = extract_sections(MINIMAL_REPORT, ["1"], source)
        html = render_extracted(sections, title="Custom Title")
        assert "<title>Custom Title</title>" in html
        assert "<h1>Custom Title</h1>" in html

    def test_default_title(self):
        source = Path("results/06_irt_2d/report.html")
        sections = extract_sections(MINIMAL_REPORT, ["1"], source)
        html = render_extracted(sections)
        assert "<title>Extracted Report</title>" in html

    def test_provenance_shown_for_multi_source(self):
        source_a = Path("results/06_irt_2d/report.html")
        source_b = Path("results/01_eda/report.html")
        sections_a = extract_sections(MINIMAL_REPORT, ["1"], source_a)
        sections_b = extract_sections(MINIMAL_REPORT, ["2"], source_b)
        html = render_extracted(sections_a + sections_b, title="Combined")
        assert "provenance" in html
        assert "06_irt_2d" in html
        assert "01_eda" in html

    def test_provenance_hidden_for_single_source(self):
        source = Path("results/06_irt_2d/report.html")
        sections = extract_sections(MINIMAL_REPORT, ["1", "2"], source)
        html = render_extracted(sections, title="Single Source")
        # CSS class exists in <style> but no provenance <p> elements in the body.
        assert '<p class="provenance">' not in html

    def test_empty_sections_raises(self):
        with pytest.raises(ValueError, match="No sections to render"):
            render_extracted([])

    def test_contains_fallback_css(self):
        source = Path("results/06_irt_2d/report.html")
        sections = extract_sections(MINIMAL_REPORT, ["1"], source)
        html = render_extracted(sections)
        # Fallback CSS includes distinctive rules.
        assert "report-section" in html
        assert "Segoe UI" in html

    def test_source_css_used_when_provided(self):
        source = Path("results/06_irt_2d/report.html")
        sections = extract_sections(MINIMAL_REPORT, ["1"], source)
        html = render_extracted(sections, source_css="body { font-family: CustomFont; }")
        assert "CustomFont" in html

    def test_plotly_deps_in_head(self):
        source = Path("results/06_irt_2d/report.html")
        sections = extract_sections(REPORT_WITH_PLOTLY, ["1", "2"], source)
        html = render_extracted(sections, title="Plotly Test")
        # Plotly script should appear in <head>, not in sections.
        head_end = html.index("</head>")
        body_start = html.index("<body>")
        head_content = html[:head_end]
        body_content = html[body_start:]
        assert "cdn.plot.ly" in head_content
        assert "cdn.plot.ly" not in body_content


# ── Slug and Output Path Tests ───────────────────────────────────────────────


class TestSlugGeneration:
    def test_simple_title(self):
        assert generate_slug("My Report") == "my-report"

    def test_special_characters(self):
        assert generate_slug("79th Senate: Three Factions!") == "79th-senate-three-factions"

    def test_leading_trailing_stripped(self):
        assert generate_slug("  spaces  ") == "spaces"

    def test_empty_string(self):
        assert generate_slug("") == "extracted"

    def test_unicode(self):
        assert generate_slug("café résumé") == "caf-r-sum"


class TestWriteExtracted:
    def test_writes_file(self, tmp_path: Path):
        out = tmp_path / "bespoke" / "2026-03-10" / "test.html"
        result = write_extracted("<html>test</html>", out)
        assert result == out
        assert out.read_text() == "<html>test</html>"

    def test_creates_latest_symlink(self, tmp_path: Path):
        out = tmp_path / "bespoke" / "2026-03-10" / "test.html"
        write_extracted("<html>test</html>", out)
        latest = tmp_path / "bespoke" / "latest"
        assert latest.is_symlink()
        assert latest.resolve().name == "2026-03-10"

    def test_updates_existing_symlink(self, tmp_path: Path):
        # Write first extraction.
        out1 = tmp_path / "bespoke" / "2026-03-09" / "old.html"
        write_extracted("<html>old</html>", out1)
        # Write second — latest should update.
        out2 = tmp_path / "bespoke" / "2026-03-10" / "new.html"
        write_extracted("<html>new</html>", out2)
        latest = tmp_path / "bespoke" / "latest"
        assert latest.resolve().name == "2026-03-10"


# ── CLI Tests ────────────────────────────────────────────────────────────────


class TestCLIParsing:
    def test_list_mode(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        report = tmp_path / "report.html"
        report.write_text(MINIMAL_REPORT)

        from tallgrass.extract.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--list", str(report)])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "experimental-status" in captured.out
        assert "model-spec" in captured.out
        assert "3 sections found" in captured.out

    def test_extract_single(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        report = tmp_path / "06_irt_2d" / "report.html"
        report.parent.mkdir()
        report.write_text(MINIMAL_REPORT)

        # Redirect output to tmp_path.
        monkeypatch.chdir(tmp_path)

        from tallgrass.extract.cli import main

        main([str(report), "--section", "1", "--output", "test.html"])

        # Find the output file under results/bespoke/.
        output_files = list((tmp_path / "results" / "bespoke").rglob("test.html"))
        assert len(output_files) == 1
        content = output_files[0].read_text()
        assert "Experimental Status" in content

    def test_extract_multi_report(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        report_a = tmp_path / "06_irt_2d" / "a.html"
        report_a.parent.mkdir()
        report_a.write_text(MINIMAL_REPORT)

        report_b = tmp_path / "01_eda" / "b.html"
        report_b.parent.mkdir()
        report_b.write_text(MINIMAL_REPORT)

        monkeypatch.chdir(tmp_path)

        from tallgrass.extract.cli import main

        main(
            [
                str(report_a),
                "--section",
                "1",
                str(report_b),
                "--section",
                "2",
                "--title",
                "Combined",
                "--output",
                "combined.html",
            ]
        )

        output_files = list((tmp_path / "results" / "bespoke").rglob("combined.html"))
        assert len(output_files) == 1
        content = output_files[0].read_text()
        assert "Combined" in content
        assert "06_irt_2d" in content
        assert "01_eda" in content

    def test_no_args_shows_help(self, capsys: pytest.CaptureFixture[str]):
        from tallgrass.extract.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    def test_section_without_report_errors(self, capsys: pytest.CaptureFixture[str]):
        from tallgrass.extract.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--section", "1"])
        assert exc_info.value.code != 0
