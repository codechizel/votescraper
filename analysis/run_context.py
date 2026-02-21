"""Reusable run context for structured analysis output.

Every analysis script (EDA, PCA, IRT, etc.) uses RunContext to get:
  - Structured output directories: results/<session>/<analysis>/<date>/plots/ + data/
  - Automatic console log capture (run_log.txt)
  - Run metadata (run_info.json): git hash, timestamp, parameters
  - A `latest` symlink pointing to the most recent run

Usage:
    with RunContext(
        session="2025-26",
        analysis_name="eda",
        params=vars(args),
        primer=EDA_PRIMER,        # Markdown primer written to results/<session>/eda/README.md
    ) as ctx:
        # ctx.plots_dir, ctx.data_dir, ctx.run_dir are ready
        df.write_parquet(ctx.data_dir / "vote_matrix.parquet")
        save_fig(fig, ctx.plots_dir / "plot.png")
        save_manifest(manifest, ctx.run_dir)
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType


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
            from ks_vote_scraper.session import KSSession
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
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


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
    ) -> None:
        self.session = _normalize_session(session)
        self.analysis_name = analysis_name
        self.params = params or {}

        root = results_root or Path("results")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        self.run_dir = root / self.session / analysis_name / today
        self.plots_dir = self.run_dir / "plots"
        self.data_dir = self.run_dir / "data"

        # Parent of date dirs — where the `latest` symlink and primer live
        self._analysis_dir = root / self.session / analysis_name
        self._today = today
        self._primer = primer
        self._tee: _TeeStream | None = None
        self._original_stdout: io.TextIOBase | None = None
        self._start_time: datetime | None = None

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
        self.finalize()

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

    def setup(self) -> None:
        """Create directories, write primer, and start log capture."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        # Write analysis primer (lives at the analysis level, not per-run)
        if self._primer:
            readme = self._analysis_dir / "README.md"
            readme.write_text(self._primer, encoding="utf-8")

        # Start capturing stdout
        self._original_stdout = sys.stdout
        self._tee = _TeeStream(sys.stdout)
        sys.stdout = self._tee  # type: ignore[assignment]
        self._start_time = datetime.now(timezone.utc)

    def finalize(self) -> None:
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
        end_time = datetime.now(timezone.utc)
        run_info = {
            "analysis": self.analysis_name,
            "session": self.session,
            "run_date": self._today,
            "timestamp_start": (self._start_time.isoformat() if self._start_time else None),
            "timestamp_end": end_time.isoformat(),
            "git_commit": _git_commit_hash(),
            "python_version": sys.version,
            "params": self.params,
        }
        info_path = self.run_dir / "run_info.json"
        with open(info_path, "w") as f:
            json.dump(run_info, f, indent=2, default=str)

        # Write HTML report if sections were added
        if self.report is not None and hasattr(self.report, "has_sections"):
            if self.report.has_sections:
                self.report.git_hash = run_info["git_commit"]
                report_path = self.run_dir / f"{self.analysis_name}_report.html"
                self.report.write(report_path)

        # Update latest symlink (relative so it's portable)
        latest = self._analysis_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(self._today)
