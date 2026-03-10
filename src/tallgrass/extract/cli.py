"""CLI entry point for tallgrass-extract.

Extracts sections from one or more pipeline HTML reports into a standalone,
self-contained HTML file. Supports multi-report extraction for presentations.

Usage:
    # List sections in a report
    tallgrass-extract --list report.html

    # Extract from one report
    tallgrass-extract report.html --section 15

    # Extract from multiple reports
    tallgrass-extract \\
        report_a.html --section 15 --section 16 \\
        report_b.html --section 3 \\
        --title "My Presentation"

    # Custom output path
    tallgrass-extract report.html --section 15 --output my-extract.html
"""

from __future__ import annotations

import sys
from pathlib import Path

from tallgrass.extract.extractor import (
    ExtractedSection,
    default_output_dir,
    extract_sections,
    generate_slug,
    list_sections,
    parse_report_css,
    parse_report_title,
    render_extracted,
    write_extracted,
)


def _parse_args(argv: list[str]) -> tuple[list[tuple[Path, list[str]]], dict[str, str | None]]:
    """Parse CLI arguments into report groups and global options.

    Returns:
        (report_groups, global_options) where report_groups is a list of
        (report_path, [selectors]) and global_options has keys:
        'title', 'output', 'list'.

    The argument grammar:
        tallgrass-extract [REPORT --section SEL...]... [--title T] [--output O] [--list]

    A new report group starts whenever a positional argument ending in .html
    is encountered. --section flags bind to the most recent report group.
    """
    groups: list[tuple[Path, list[str]]] = []
    global_opts: dict[str, str | None] = {"title": None, "output": None, "list": None}

    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg == "--list":
            global_opts["list"] = "true"
            i += 1
        elif arg == "--title":
            if i + 1 >= len(argv):
                _error("--title requires a value")
            global_opts["title"] = argv[i + 1]
            i += 2
        elif arg == "--output":
            if i + 1 >= len(argv):
                _error("--output requires a value")
            global_opts["output"] = argv[i + 1]
            i += 2
        elif arg == "--section":
            if i + 1 >= len(argv):
                _error("--section requires a value")
            if not groups:
                _error("--section must follow a report path")
            groups[-1][1].append(argv[i + 1])
            i += 2
        elif arg.startswith("--"):
            _error(f"Unknown option: {arg}")
        else:
            # Positional argument — treat as a report path.
            path = Path(arg)
            if not path.exists():
                _error(f"Report not found: {arg}")
            groups.append((path, []))
            i += 1

    return groups, global_opts


def _error(msg: str) -> None:
    """Print error message and exit."""
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _print_sections(path: Path) -> None:
    """Print a table of sections in a report and exit."""
    html = path.read_text(encoding="utf-8")
    sections = list_sections(html)
    title = parse_report_title(html)

    print(f"\n{title}")
    print(f"{'─' * len(title)}")
    print(f"{'#':<6} {'ID':<45} {'Title'}")
    print(f"{'─' * 5}  {'─' * 44} {'─' * 30}")

    for s in sections:
        print(f"{s.number:<6} {s.id:<45} {s.title}")

    print(f"\n{len(sections)} sections found.")


def main(argv: list[str] | None = None) -> None:
    """Entry point for tallgrass-extract."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv == ["--help"] or argv == ["-h"]:
        print(__doc__)
        sys.exit(0)

    groups, global_opts = _parse_args(argv)

    # --list mode: print sections for the first (or only) report.
    if global_opts["list"]:
        if not groups:
            _error("--list requires a report path")
        _print_sections(groups[0][0])
        sys.exit(0)

    # Extraction mode: need at least one report with at least one section.
    if not groups:
        _error("No report paths provided. Use --help for usage.")

    has_sections = any(sels for _, sels in groups)
    if not has_sections:
        _error("No --section selectors provided. Use --list to see available sections.")

    # Extract sections from each report; capture CSS from first source.
    all_extracted: list[ExtractedSection] = []
    source_css = ""
    for path, selectors in groups:
        if not selectors:
            continue
        html = path.read_text(encoding="utf-8")
        if not source_css:
            source_css = parse_report_css(html)
        extracted = extract_sections(html, selectors, source_path=path)
        all_extracted.extend(extracted)

    if not all_extracted:
        _error("No sections matched.")

    # Render.
    title = global_opts["title"]
    rendered = render_extracted(all_extracted, title=title, source_css=source_css)

    # Determine output path.
    if global_opts["output"]:
        out_path = default_output_dir() / global_opts["output"]
    else:
        slug = generate_slug(title) if title else "extracted"
        out_path = default_output_dir() / f"{slug}.html"

    written = write_extracted(rendered, out_path)
    print(f"Extracted {len(all_extracted)} section(s) → {written}")
