"""Reusable run context for structured analysis output.

Every analysis script (EDA, PCA, IRT, etc.) uses RunContext to get:
  - Structured output directories under a run ID
  - Automatic console log capture (run_log.txt)
  - Run metadata (run_info.json): git hash, timestamp, parameters
  - A `latest` symlink pointing to the most recent run
  - A convenience report symlink in the session root (e.g. 01_eda_report.html → latest/01_eda/...)

Run-directory mode (all biennium sessions):
  All phases write into a grouped directory keyed by run_id:
    results/<session>/<run_id>/<analysis>/plots/ + data/
  A session-level `latest` symlink points to the run directory.
  Report convenience symlinks chain through latest/<phase>/.

  When run from a pipeline, all phases share the same run_id.
  When run individually, a run_id is auto-generated so the directory
  structure is always consistent.

Flat mode (cross-session and special sessions):
  When run_id is None and the session is not a biennium (e.g. "cross-session",
  "2024s"), each phase writes to its own date directory:
    results/<session>/<analysis>/<date>/plots/ + data/
  A phase-level `latest` symlink points to the date directory.

Usage:
    with RunContext(
        session="2025-26",
        analysis_name="01_eda",
        params=vars(args),
        primer=EDA_PRIMER,        # Markdown primer written to results/<session>/01_eda/README.md
    ) as ctx:
        # ctx.plots_dir, ctx.data_dir, ctx.run_dir are ready
        df.write_parquet(ctx.data_dir / "vote_matrix.parquet")
        save_fig(fig, ctx.plots_dir / "plot.png")
        save_manifest(manifest, ctx.run_dir)
"""

import io
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import TextIO
from zoneinfo import ZoneInfo

_CT = ZoneInfo("America/Chicago")

_LEADERSHIP_SUFFIX_RE = re.compile(r"\s*-\s+.*$")
"""Matches leadership suffixes like ' - House Minority Caucus Chair'."""


def strip_leadership_suffix(name: str) -> str:
    """Remove leadership suffix from a legislator's display name.

    The scraper stores raw names from kslegislature.gov which include
    leadership titles (e.g., "Ty Masterson - President of the Senate").
    This strips everything after " - " for clean display labels.

    Examples:
        "Ty Masterson - President of the Senate" → "Ty Masterson"
        "Tim Shallenburger - Vice President of the Senate" → "Tim Shallenburger"
        "John Alcala" → "John Alcala"  (no-op)
    """
    return _LEADERSHIP_SUFFIX_RE.sub("", name).strip()


class _TeeStream:
    """Wraps a stream to duplicate output to both the original stream and a buffer.

    All print() output goes to both the console (so the user sees progress)
    and an internal StringIO buffer (captured for run_log.txt).
    """

    def __init__(self, original: io.TextIOBase) -> None:
        self._original = original
        self._buffer = io.StringIO()

    def write(self, data: str) -> int:
        self._original.write(data)
        self._buffer.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()

    def getvalue(self) -> str:
        return self._buffer.getvalue()


def _normalize_session(session: str) -> str:
    """Convert session shorthand to biennium directory format.

    Uses the Kansas Legislature numbering scheme (e.g., 91st_2025-2026).

    Examples:
        "2025-26"  -> "91st_2025-2026"
        "2023-24"  -> "90th_2023-2024"
        "2024s"    -> "2024s"  (special sessions pass through)
        "2025_26"  -> "91st_2025-2026"
    """
    # Normalize underscores to hyphens first
    session = session.replace("_", "-")

    # Match "YYYY-YY" or "YYYY-YYYY" pattern (biennium)
    m = re.match(r"^(\d{4})-(\d{2,4})$", session)
    if m:
        try:
            from tallgrass.session import KSSession
        except ImportError:
            from session import KSSession  # type: ignore[no-redef]
        start = int(m.group(1))
        ks = KSSession.from_year(start)
        return ks.output_name

    return session


def _git_commit_hash() -> str:
    """Get the current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):  # fmt: skip
        pass
    return "unknown"


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string.

    Examples: "3.2s", "1m 45s", "1h 12m 5s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m {secs}s"


def _next_run_label(analysis_dir: Path, today: str) -> str:
    """Return a unique run label for today, starting at .1.

    First run of the day:  "260223.1"
    Second run:            "260223.2"
    Third run:             "260223.3"

    Checks for existing directories (not symlinks) under *analysis_dir*.
    """
    n = 1
    while (analysis_dir / f"{today}.{n}").exists() and not (
        analysis_dir / f"{today}.{n}"
    ).is_symlink():
        n += 1
    return f"{today}.{n}"


def generate_run_id(session: str, results_root: Path | None = None) -> str:
    """Generate a run ID for grouping pipeline phases.

    Format: {bb}-{YYMMDD}.{n} where bb is the legislature number and n starts at 1.

    Examples:
        "2025-26" → "91-260227.1"
        "2024s"   → "2024s-260227.1"
        "2025-26" (second run same day) → "91-260227.2"

    Args:
        session: Session string (e.g. "2025-26", "2024s").
        results_root: Optional results root for collision checking.
            If provided, skips existing directories to find next available.
    """
    normalized = _normalize_session(session)
    # Extract legislature prefix (e.g. "91st" from "91st_2025-2026", or full string for specials)
    prefix = normalized.split("_")[0] if "_" in normalized else normalized
    # Strip ordinal suffix (e.g. "91st" → "91", "84th" → "84")
    prefix = re.sub(r"(st|nd|rd|th)$", "", prefix)
    ts = datetime.now(_CT).strftime("%y%m%d")
    base = f"{prefix}-{ts}"

    if results_root is None:
        return f"{base}.1"

    # Find next available increment
    n = 1
    while (results_root / f"{base}.{n}").exists() and not (
        results_root / f"{base}.{n}"
    ).is_symlink():
        n += 1
    return f"{base}.{n}"


def resolve_upstream_dir(
    phase: str,
    results_root: Path,
    run_id: str | None = None,
    override: Path | None = None,
) -> Path:
    """Resolve the output directory for an upstream phase.

    Precedence:
      1. Explicit CLI override (e.g. --eda-dir /some/path)
      2. Run-directory path: results_root/{run_id}/{phase}
      3. Flat phase path: results_root/{phase}/latest (cross-session/special)
      4. Fallback: results_root/latest/{phase}

    For standalone biennium runs, run_id is auto-generated so precedence 2
    won't find upstream data from a different run.  Precedence 4 resolves
    through the session-level `latest` symlink to the most recent pipeline run.

    The caller should verify the returned path exists before reading from it.
    """
    if override is not None:
        return override
    if run_id is not None:
        return results_root / run_id / phase
    legacy = results_root / phase / "latest"
    if legacy.exists():
        return legacy
    return results_root / "latest" / phase


class RunContext:
    """Context manager that sets up structured output for an analysis run.

    Creates the directory tree, captures console output, and writes
    metadata on exit.

    Attributes:
        session: Normalized session string (e.g. "2025-2026").
        analysis_name: Name of the analysis phase (e.g. "eda", "pca").
        params: Script parameters to record in run_info.json.
        run_dir: Root of this run's output (results/<session>/<analysis>/<date>/).
        plots_dir: Directory for PNG plots.
        data_dir: Directory for parquet/intermediate data files.
    """

    def __init__(
        self,
        session: str,
        analysis_name: str,
        params: dict | None = None,
        results_root: Path | None = None,
        primer: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.session = _normalize_session(session)
        self.analysis_name = analysis_name
        self.params = params or {}

        from tallgrass.session import STATE_DIR

        root = results_root or (Path("results") / STATE_DIR)
        today = datetime.now(_CT).strftime("%y%m%d")
        self._session_root = root / self.session

        # Track whether run_id was explicitly provided (pipeline) vs auto-generated
        # (standalone). Only explicit run_ids update the `latest` symlink (ADR-0069).
        self._explicit_run_id = run_id is not None

        # Auto-generate run_id for biennium sessions (contain "_" after normalization,
        # e.g. "91st_2025-2026") so standalone phase runs use the same directory
        # structure as pipeline runs.  Non-biennium sessions (cross-session, special
        # sessions like "2024s") keep the flat date-label layout.
        if run_id is None and "_" in self.session:
            run_id = generate_run_id(session, results_root=self._session_root)

        self.run_id = run_id

        if run_id is not None:
            # Run-directory mode: results/{session}/{run_id}/{analysis}/
            self._analysis_dir = self._session_root / run_id / analysis_name
            self.run_dir = self._analysis_dir
            self._run_label = run_id
        else:
            # Flat mode: results/{session}/{analysis}/{date}/
            self._analysis_dir = self._session_root / analysis_name
            run_label = _next_run_label(self._analysis_dir, today)
            self.run_dir = self._analysis_dir / run_label
            self._run_label = run_label

        self.plots_dir = self.run_dir / "plots"
        self.data_dir = self.run_dir / "data"

        self._today = today
        self._primer = primer
        self._tee: _TeeStream | None = None
        self._original_stdout: TextIO | None = None
        self._start_time: datetime | None = None

        # Track exported CSVs for DownloadSection
        self._exported_csvs: list[tuple[str, str]] = []

        # Lazy-init report builder (avoids importing report.py at module level)
        self.report = self._init_report()

    def __enter__(self) -> RunContext:
        self.setup()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.finalize(failed=exc_type is not None)

    def _init_report(self) -> object:
        """Initialize a ReportBuilder, or None if the report module isn't available."""
        try:
            try:
                from analysis.report import ReportBuilder
            except ModuleNotFoundError:
                from report import ReportBuilder  # type: ignore[no-redef]
            return ReportBuilder(
                title=f"{self.analysis_name.upper()} Report",
                session=self.session,
            )
        except ImportError:
            return None

    def export_csv(
        self,
        df: object,
        filename: str,
        description: str,
        column_labels: dict[str, str] | None = None,
    ) -> Path:
        """Export a Polars DataFrame as CSV and register it for the download section.

        Args:
            df: A polars DataFrame to export.
            filename: Output filename (e.g. "ideal_points_house.csv").
            description: Human-readable description for the download link.
            column_labels: Optional column rename mapping for the exported CSV.

        Returns:
            Path to the written CSV file.
        """
        import polars as pl

        if not isinstance(df, pl.DataFrame):
            msg = f"export_csv expects a polars DataFrame, got {type(df).__name__}"
            raise TypeError(msg)

        out = df
        if column_labels:
            rename_map = {k: v for k, v in column_labels.items() if k in out.columns}
            if rename_map:
                out = out.rename(rename_map)

        # Drop columns with nested types (List, Struct, Array) that CSV cannot serialize
        nested = [
            c for c in out.columns
            if out[c].dtype.base_type() in (pl.List, pl.Struct, pl.Array)
        ]
        if nested:
            out = out.drop(nested)

        csv_path = self.data_dir / filename
        out.write_csv(csv_path)

        # Store relative path from report HTML location to CSV
        rel_path = f"data/{filename}"
        self._exported_csvs.append((rel_path, description))
        return csv_path

    def setup(self) -> None:
        """Create directories, write primer, and start log capture."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        # Write analysis primer
        if self._primer:
            if self.run_id is not None:
                # Run-directory mode: primer in the phase dir inside the run
                readme = self._analysis_dir / "README.md"
            else:
                # Legacy mode: primer at the analysis level (parent of date dirs)
                readme = self._analysis_dir / "README.md"
            readme.write_text(self._primer, encoding="utf-8")

        # Start capturing stdout
        self._original_stdout = sys.stdout
        self._tee = _TeeStream(sys.stdout)
        sys.stdout = self._tee  # type: ignore[assignment]
        self._start_time = datetime.now(_CT)

    def finalize(self, *, failed: bool = False) -> None:
        """Write run_info.json, run_log.txt, and update latest symlink."""
        # Restore stdout before writing metadata (so our writes aren't captured)
        log_text = ""
        if self._tee is not None:
            log_text = self._tee.getvalue()
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout  # type: ignore[assignment]

        # Write run log
        log_path = self.run_dir / "run_log.txt"
        log_path.write_text(log_text, encoding="utf-8")

        # Write run info
        end_time = datetime.now(_CT)
        elapsed_seconds = (end_time - self._start_time).total_seconds() if self._start_time else 0.0
        run_info = {
            "analysis": self.analysis_name,
            "session": self.session,
            "run_date": self._today,
            "run_label": self._run_label,
            "run_id": self.run_id,
            "timestamp_start": (self._start_time.isoformat() if self._start_time else None),
            "timestamp_end": end_time.isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 1),
            "elapsed_display": _format_elapsed(elapsed_seconds),
            "git_commit": _git_commit_hash(),
            "python_version": sys.version,
            "params": self.params,
        }
        info_path = self.run_dir / "run_info.json"
        with open(info_path, "w") as f:
            json.dump(run_info, f, indent=2, default=str)

        # Print elapsed time (after stdout is restored so it goes to console, not log)
        print(f"\n{self.analysis_name.upper()} completed in {run_info['elapsed_display']}")

        # Write HTML report if sections were added
        if self.report is not None and hasattr(self.report, "has_sections"):
            if self.report.has_sections:
                _append_download_section(self.report, self._exported_csvs)
                _append_missing_votes(self.report, self.session)
                self.report.git_hash = run_info["git_commit"]
                self.report.elapsed_display = run_info["elapsed_display"]
                report_path = self.run_dir / f"{self.analysis_name}_report.html"
                self.report.write(report_path)

                if self.run_id is not None and self._explicit_run_id:
                    # Run-directory mode: symlink chains through session-level latest
                    report_link = self._session_root / f"{self.analysis_name}_report.html"
                    if report_link.is_symlink() or report_link.exists():
                        report_link.unlink()
                    report_link.symlink_to(
                        Path("latest") / self.analysis_name / f"{self.analysis_name}_report.html"
                    )
                elif self.run_id is None:
                    # Flat mode: symlink through phase-level latest
                    session_root = self._analysis_dir.parent
                    report_link = session_root / f"{self.analysis_name}_report.html"
                    if report_link.is_symlink() or report_link.exists():
                        report_link.unlink()
                    report_link.symlink_to(
                        Path(self.analysis_name) / "latest" / f"{self.analysis_name}_report.html"
                    )

        # Update latest symlink
        # Skip symlink update on failed runs so downstream phases don't see partial results
        if not failed:
            if self.run_id is not None and self._explicit_run_id:
                # Run-directory mode: session-level `latest` → run_id (idempotent)
                latest = self._session_root / "latest"
                if latest.is_symlink() or latest.exists():
                    latest.unlink()
                latest.symlink_to(self.run_id)
            elif self.run_id is None:
                # Flat mode: phase-level `latest` → date label
                latest = self._analysis_dir / "latest"
                if latest.is_symlink() or latest.exists():
                    latest.unlink()
                latest.symlink_to(self._run_label)


def _append_download_section(report: object, exported_csvs: list[tuple[str, str]]) -> None:
    """Append a Data Downloads section if any CSVs were exported."""
    if not exported_csvs:
        return

    try:
        from analysis.report import DownloadSection
    except ModuleNotFoundError:
        from report import DownloadSection  # type: ignore[no-redef]

    report.add(  # type: ignore[union-attr]
        DownloadSection(
            id="data-downloads",
            title="Data Downloads",
            files=exported_csvs,
            caption="CSV files for use in external analysis tools.",
        )
    )


def _parse_vote_tally(vote_text: str) -> tuple[int, int, int] | None:
    """Parse 'Yea: 63 Nay: 59' into (yea, nay, margin) or None if unparseable."""
    m = re.search(r"Yea:\s*(\d+)\s*Nay:\s*(\d+)", vote_text)
    if not m:
        return None
    yea, nay = int(m.group(1)), int(m.group(2))
    return yea, nay, abs(yea - nay)


def _append_missing_votes(report: object, session: str) -> None:
    """Append a Missing Votes section to the report from the failure manifest."""
    from tallgrass.session import STATE_DIR

    try:
        from analysis.report import TextSection
    except ModuleNotFoundError:
        from report import TextSection  # type: ignore[no-redef]

    data_dir = Path("data") / STATE_DIR / session
    manifest_path = data_dir / "failure_manifest.json"
    if not manifest_path.exists():
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    failures = manifest.get("failures", [])
    if not failures:
        return

    total = manifest.get("total_vote_pages", "?")

    # Parse tallies and sort by margin (closest first), unparseable last
    rows: list[tuple[int | None, dict]] = []
    for fail in failures:
        tally = _parse_vote_tally(fail.get("vote_text", ""))
        rows.append((tally[2] if tally else None, fail))

    rows.sort(key=lambda r: (r[0] is None, r[0] or 0))

    # Build HTML table
    total_display = f"{total:,}" if isinstance(total, int) else total
    lines = [
        f"<p>{len(failures)} of {total_display} vote pages could not be fetched."
        " Close votes are bolded.</p>",
        '<table style="width:100%; border-collapse:collapse; font-size:14px; margin-top:12px;">',
        "<thead><tr>"
        '<th style="text-align:left; border-bottom:2px solid #333; padding:6px;">Bill</th>'
        '<th style="text-align:center; border-bottom:2px solid #333; padding:6px;">Tally</th>'
        '<th style="text-align:center; border-bottom:2px solid #333; padding:6px;">Margin</th>'
        '<th style="text-align:left; border-bottom:2px solid #333; padding:6px;">Error</th>'
        '<th style="text-align:left; border-bottom:2px solid #333; padding:6px;">Link</th>'
        "</tr></thead><tbody>",
    ]

    for margin, fail in rows:
        bill = fail.get("bill_number", "?")
        url = fail.get("vote_url", "")
        status = fail.get("status_code", "")
        error_type = fail.get("error_type", "")
        error_label = f"{status}" if status else error_type

        tally = _parse_vote_tally(fail.get("vote_text", ""))
        if tally:
            yea, nay, mg = tally
            tally_str = f"{yea}-{nay}"
            margin_str = f"<strong>{mg}</strong>" if mg <= 10 else str(mg)
        else:
            tally_str = fail.get("vote_text", "?")
            margin_str = "?"

        link = f'<a href="{url}">vote page</a>' if url else ""
        lines.append(
            "<tr>"
            f'<td style="padding:4px 6px; border-bottom:1px solid #eee;">{bill}</td>'
            '<td style="padding:4px 6px; border-bottom:1px solid #eee; text-align:center;">'
            f"{tally_str}</td>"
            '<td style="padding:4px 6px; border-bottom:1px solid #eee; text-align:center;">'
            f"{margin_str}</td>"
            f'<td style="padding:4px 6px; border-bottom:1px solid #eee;">{error_label}</td>'
            f'<td style="padding:4px 6px; border-bottom:1px solid #eee;">{link}</td>'
            "</tr>"
        )

    lines.append("</tbody></table>")
    lines.append(
        '<p style="margin-top:12px; font-size:13px; color:#555;">'
        "Re-running the scraper retries failed pages. For persistent 404s,"
        " check legislative journals or archives.</p>"
    )

    report.add(  # type: ignore[union-attr]
        TextSection(
            id="missing-votes",
            title="Missing Votes",
            html="\n".join(lines),
        )
    )
