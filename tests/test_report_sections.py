"""
Tests for new report section types added in the report enhancement (WU0).

Covers InteractiveTableSection, InteractiveSection, KeyFindingsSection,
make_interactive_table() helper, and ReportBuilder integration.

Run: uv run pytest tests/test_report_sections.py -v
"""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.report import (
    DownloadSection,
    InteractiveSection,
    InteractiveTableSection,
    KeyFindingsSection,
    ReportBuilder,
    ScrollySection,
    ScrollyStep,
    TextSection,
    make_interactive_table,
)

# ── InteractiveTableSection ──────────────────────────────────────────────────


class TestInteractiveTableSection:
    """Searchable/sortable table section (ITables)."""

    def test_render_basic(self):
        section = InteractiveTableSection(id="it1", title="Scores", html="<table>data</table>")
        html = section.render()
        assert '<div class="interactive-table-container" id="it1">' in html
        assert "<table>data</table>" in html
        assert "</div>" in html

    def test_render_with_caption(self):
        section = InteractiveTableSection(
            id="it1", title="Scores", html="<table></table>", caption="All rows shown"
        )
        html = section.render()
        assert '<p class="caption">All rows shown</p>' in html

    def test_render_without_caption(self):
        section = InteractiveTableSection(id="it1", title="Scores", html="<table></table>")
        html = section.render()
        assert "caption" not in html

    def test_frozen(self):
        section = InteractiveTableSection(id="it1", title="Scores", html="<table></table>")
        with pytest.raises(AttributeError):
            section.id = "it2"  # type: ignore[misc]


# ── InteractiveSection ───────────────────────────────────────────────────────


class TestInteractiveSection:
    """Raw HTML fragment for interactive content (Plotly/PyVis)."""

    def test_render_basic(self):
        section = InteractiveSection(id="plotly1", title="Scatter", html="<div>plotly chart</div>")
        html = section.render()
        assert '<div class="interactive-container" id="plotly1">' in html
        assert "<div>plotly chart</div>" in html

    def test_render_with_caption(self):
        section = InteractiveSection(
            id="p1", title="Chart", html="<div></div>", caption="Hover for details"
        )
        html = section.render()
        assert '<p class="caption">Hover for details</p>' in html

    def test_render_without_caption(self):
        section = InteractiveSection(id="p1", title="Chart", html="<div></div>")
        html = section.render()
        assert "caption" not in html

    def test_frozen(self):
        section = InteractiveSection(id="p1", title="Chart", html="<div></div>")
        with pytest.raises(AttributeError):
            section.title = "New"  # type: ignore[misc]


# ── KeyFindingsSection ───────────────────────────────────────────────────────


class TestKeyFindingsSection:
    """Bullet-point key findings rendered above the TOC."""

    def test_render_basic(self):
        section = KeyFindingsSection(findings=["Finding 1", "Finding 2"])
        html = section.render()
        assert '<div class="key-findings">' in html
        assert "<h2>Key Findings</h2>" in html
        assert "<li>Finding 1</li>" in html
        assert "<li>Finding 2</li>" in html

    def test_render_single_finding(self):
        section = KeyFindingsSection(findings=["Only one finding."])
        html = section.render()
        assert "<li>Only one finding.</li>" in html

    def test_render_empty_findings(self):
        section = KeyFindingsSection(findings=[])
        html = section.render()
        assert '<div class="key-findings">' in html
        assert "<ul>" in html

    def test_frozen(self):
        section = KeyFindingsSection(findings=["A"])
        with pytest.raises(AttributeError):
            section.findings = ["B"]  # type: ignore[misc]


# ── make_interactive_table() ─────────────────────────────────────────────────


class TestMakeInteractiveTable:
    """ITables-powered interactive table helper."""

    def test_returns_html_string(self):
        df = pl.DataFrame({"name": ["Smith", "Jones"], "score": [0.85, 0.72]})
        result = make_interactive_table(df, title="Test Table")
        assert isinstance(result, str)
        assert "<" in result

    def test_rejects_non_polars(self):
        with pytest.raises(TypeError, match="polars DataFrame"):
            make_interactive_table({"a": [1]}, title="Bad")

    def test_title_rendered_as_h4(self):
        df = pl.DataFrame({"x": [1]})
        html = make_interactive_table(df, title="My Title")
        assert "<h4>My Title</h4>" in html

    def test_no_title_omits_h4(self):
        df = pl.DataFrame({"x": [1]})
        html = make_interactive_table(df)
        assert "<h4>" not in html

    def test_column_labels_applied(self):
        df = pl.DataFrame({"slug": ["smith"], "xi_mean": [0.5]})
        html = make_interactive_table(
            df, column_labels={"slug": "Legislator", "xi_mean": "Ideal Point"}
        )
        assert "Legislator" in html
        assert "Ideal Point" in html

    def test_caption_rendered(self):
        df = pl.DataFrame({"x": [1]})
        html = make_interactive_table(df, caption="Source: test data")
        assert '<p class="caption">Source: test data</p>' in html

    def test_contains_script_tag(self):
        """connected=False should inline the DataTables JS."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        html = make_interactive_table(df)
        assert "<script" in html

    def test_number_formats_applied(self):
        df = pl.DataFrame({"value": [1.23456]})
        html = make_interactive_table(df, number_formats={"value": ".2f"})
        assert "1.23" in html


# ── ReportBuilder with Key Findings ──────────────────────────────────────────


class TestReportBuilderKeyFindings:
    """KeyFindingsSection appears before the TOC in rendered output."""

    def test_key_findings_before_toc(self):
        report = ReportBuilder(title="Test")
        report.add(KeyFindingsSection(findings=["Finding A", "Finding B"]))
        report.add(TextSection(id="s1", title="Section 1", html="<p>Content</p>"))
        html = report.render()
        kf_pos = html.index("Key Findings")
        toc_pos = html.index("Table of Contents")
        assert kf_pos < toc_pos

    def test_key_findings_not_in_toc(self):
        report = ReportBuilder(title="Test")
        report.add(KeyFindingsSection(findings=["Finding A"]))
        report.add(TextSection(id="s1", title="Section 1", html="<p>Content</p>"))
        html = report.render()
        # Key findings should NOT have a TOC entry
        assert 'href="#key-findings"' not in html

    def test_key_findings_not_numbered(self):
        report = ReportBuilder(title="Test")
        report.add(KeyFindingsSection(findings=["Finding A"]))
        report.add(TextSection(id="s1", title="Section 1", html="<p>Content</p>"))
        html = report.render()
        # Section numbering should start at 1 for the first real section
        assert '<span class="section-number">1.</span> Section 1' in html

    def test_no_key_findings_still_works(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert "Key Findings" not in html

    def test_has_sections_ignores_key_findings(self):
        report = ReportBuilder(title="Test")
        report.add(KeyFindingsSection(findings=["A"]))
        assert report.has_sections is False  # only KF, no numbered sections


# ── ReportBuilder with InteractiveTableSection ───────────────────────────────


class TestReportBuilderInteractive:
    """InteractiveTableSection is properly rendered within a report."""

    def test_interactive_table_in_report(self):
        report = ReportBuilder(title="Test")
        report.add(InteractiveTableSection(id="it1", title="Scores", html="<table>data</table>"))
        html = report.render()
        assert '<div class="interactive-table-container" id="it1">' in html
        assert '<a href="#it1">' in html

    def test_interactive_section_in_report(self):
        report = ReportBuilder(title="Test")
        report.add(InteractiveSection(id="plotly1", title="Scatter", html="<div>chart</div>"))
        html = report.render()
        assert '<div class="interactive-container" id="plotly1">' in html


# ── CSS Styles ───────────────────────────────────────────────────────────────


class TestReportCSSNewStyles:
    """New CSS classes are present in rendered output."""

    def test_interactive_table_css(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert ".interactive-table-container" in html

    def test_interactive_container_css(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert ".interactive-container" in html

    def test_key_findings_css(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert ".key-findings" in html


# ── DownloadSection ──────────────────────────────────────────────────────────


class TestDownloadSection:
    """Downloadable CSV file links."""

    def test_render_basic(self):
        section = DownloadSection(
            id="dl1",
            title="Downloads",
            files=[("data/scores.csv", "Legislator scores")],
        )
        html = section.render()
        assert '<div class="download-container" id="dl1">' in html
        assert 'href="data/scores.csv"' in html
        assert "download" in html
        assert "scores.csv" in html
        assert "Legislator scores" in html

    def test_render_multiple_files(self):
        section = DownloadSection(
            id="dl2",
            title="Downloads",
            files=[
                ("data/house.csv", "House data"),
                ("data/senate.csv", "Senate data"),
            ],
        )
        html = section.render()
        assert "house.csv" in html
        assert "senate.csv" in html
        assert "House data" in html
        assert "Senate data" in html

    def test_render_with_caption(self):
        section = DownloadSection(
            id="dl3",
            title="Downloads",
            files=[("f.csv", "File")],
            caption="All files are UTF-8 encoded",
        )
        html = section.render()
        assert '<p class="caption">All files are UTF-8 encoded</p>' in html

    def test_render_without_caption(self):
        section = DownloadSection(
            id="dl4", title="Downloads", files=[("f.csv", "File")]
        )
        html = section.render()
        assert "caption" not in html

    def test_frozen(self):
        section = DownloadSection(id="dl5", title="Downloads", files=[])
        with pytest.raises(AttributeError):
            section.id = "dl6"  # type: ignore[misc]

    def test_relative_path_preserved(self):
        section = DownloadSection(
            id="dl6",
            title="Downloads",
            files=[("data/sub/deep.csv", "Deep file")],
        )
        html = section.render()
        assert 'href="data/sub/deep.csv"' in html
        # Display shows only filename
        assert ">deep.csv</a>" in html


# ── ScrollySection ──────────────────────────────────────────────────────────


class TestScrollyStep:
    """ScrollyStep frozen dataclass."""

    def test_construction(self):
        step = ScrollyStep(narrative_html="<p>Hello</p>", visual_id="fig1")
        assert step.narrative_html == "<p>Hello</p>"
        assert step.visual_id == "fig1"

    def test_frozen(self):
        step = ScrollyStep(narrative_html="<p>A</p>", visual_id="v1")
        with pytest.raises(AttributeError):
            step.visual_id = "v2"  # type: ignore[misc]


class TestScrollySection:
    """Scrollytelling section with sticky visuals."""

    def test_render_basic(self):
        section = ScrollySection(
            id="sc1",
            title="Chapter 1",
            steps=[
                ScrollyStep(narrative_html="<p>Step 1</p>", visual_id="img1"),
                ScrollyStep(narrative_html="<p>Step 2</p>", visual_id="img2"),
            ],
            visuals={"img1": "<img src='a.png'/>", "img2": "<img src='b.png'/>"},
        )
        html = section.render()
        assert '<div class="scrolly-container" id="sc1">' in html
        assert '<div class="scrolly-visual">' in html
        assert '<div class="scrolly-narrative">' in html
        assert "Step 1" in html
        assert "Step 2" in html

    def test_first_step_active(self):
        section = ScrollySection(
            id="sc2",
            title="Chapter",
            steps=[
                ScrollyStep(narrative_html="<p>A</p>", visual_id="v1"),
                ScrollyStep(narrative_html="<p>B</p>", visual_id="v1"),
            ],
            visuals={"v1": "<div>visual</div>"},
        )
        html = section.render()
        assert 'class="scrolly-step is-active"' in html
        # Second step should not be active
        lines = html.split("\n")
        step_lines = [l for l in lines if "scrolly-step" in l and "data-visual" in l]
        assert len(step_lines) == 2
        assert "is-active" in step_lines[0]
        assert "is-active" not in step_lines[1]

    def test_visual_ids_in_output(self):
        section = ScrollySection(
            id="sc3",
            title="Chapter",
            steps=[ScrollyStep(narrative_html="<p>X</p>", visual_id="chart1")],
            visuals={"chart1": "<div>chart</div>"},
        )
        html = section.render()
        assert 'id="scrolly-fig-chart1"' in html
        assert 'data-visual="chart1"' in html

    def test_render_with_caption(self):
        section = ScrollySection(
            id="sc4",
            title="Chapter",
            steps=[ScrollyStep(narrative_html="<p>X</p>", visual_id="v")],
            visuals={"v": "<div/>"},
            caption="Scroll to explore",
        )
        html = section.render()
        assert '<p class="caption">Scroll to explore</p>' in html

    def test_frozen(self):
        section = ScrollySection(
            id="sc5",
            title="Chapter",
            steps=[],
            visuals={},
        )
        with pytest.raises(AttributeError):
            section.id = "sc6"  # type: ignore[misc]


class TestReportBuilderScrolly:
    """ScrollySection integration with ReportBuilder."""

    def test_scrolly_in_report(self):
        report = ReportBuilder(title="Test")
        report.add(
            ScrollySection(
                id="scrolly1",
                title="Scrolly Chapter",
                steps=[ScrollyStep(narrative_html="<p>A</p>", visual_id="v1")],
                visuals={"v1": "<div>visual</div>"},
            )
        )
        html = report.render()
        assert '<div class="scrolly-container" id="scrolly1">' in html
        assert '<a href="#scrolly1">' in html

    def test_scrolly_injects_js(self):
        report = ReportBuilder(title="Test")
        report.add(
            ScrollySection(
                id="scrolly2",
                title="Scrolly",
                steps=[ScrollyStep(narrative_html="<p>A</p>", visual_id="v1")],
                visuals={"v1": "<div/>"},
            )
        )
        html = report.render()
        assert "IntersectionObserver" in html

    def test_no_scrolly_no_js(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="t1", title="Plain", html="<p>No scrolly</p>"))
        html = report.render()
        assert "IntersectionObserver" not in html

    def test_scrolly_css_present(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert ".scrolly-container" in html
        assert ".scrolly-step" in html
        assert ".scrolly-visual" in html


class TestReportCSSDownloadStyles:
    """Download section CSS classes are present."""

    def test_download_css(self):
        report = ReportBuilder(title="Test")
        report.add(TextSection(id="s1", title="S", html="<p>Hi</p>"))
        html = report.render()
        assert ".download-container" in html
        assert ".download-list" in html
