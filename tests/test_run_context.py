"""
Tests for analysis run context infrastructure in analysis/run_context.py.

Covers _TeeStream output capture, session normalization, git hash retrieval,
and RunContext lifecycle (directory creation, log capture, symlinks).

Run: uv run pytest tests/test_run_context.py -v
"""

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.run_context import (
    RunContext,
    _format_elapsed,
    _git_commit_hash,
    _next_run_label,
    _normalize_session,
    _TeeStream,
    generate_run_id,
    resolve_upstream_dir,
    strip_leadership_suffix,
)

# ── _TeeStream ───────────────────────────────────────────────────────────────


class TestTeeStream:
    """Duplicates output to original stream and internal buffer."""

    def test_write_returns_length(self):
        original = io.StringIO()
        tee = _TeeStream(original)
        assert tee.write("hello") == 5

    def test_write_goes_to_original(self):
        original = io.StringIO()
        tee = _TeeStream(original)
        tee.write("hello")
        assert original.getvalue() == "hello"

    def test_write_goes_to_buffer(self):
        original = io.StringIO()
        tee = _TeeStream(original)
        tee.write("hello")
        assert tee.getvalue() == "hello"

    def test_multiple_writes_accumulate(self):
        original = io.StringIO()
        tee = _TeeStream(original)
        tee.write("hello ")
        tee.write("world")
        assert tee.getvalue() == "hello world"
        assert original.getvalue() == "hello world"

    def test_flush_does_not_raise(self):
        original = io.StringIO()
        tee = _TeeStream(original)
        tee.flush()  # should not raise

    def test_empty_write(self):
        original = io.StringIO()
        tee = _TeeStream(original)
        assert tee.write("") == 0
        assert tee.getvalue() == ""


# ── _normalize_session() ─────────────────────────────────────────────────────


class TestNormalizeSession:
    """Convert session shorthand to biennium directory format."""

    def test_dash_two_digit(self):
        """'2025-26' → '91st_2025-2026'."""
        assert _normalize_session("2025-26") == "91st_2025-2026"

    def test_dash_four_digit(self):
        """'2025-2026' → '91st_2025-2026'."""
        assert _normalize_session("2025-2026") == "91st_2025-2026"

    def test_underscore_two_digit(self):
        """'2025_26' → '91st_2025-2026' (underscore normalized to dash)."""
        assert _normalize_session("2025_26") == "91st_2025-2026"

    def test_historical_session(self):
        """'2023-24' → '90th_2023-2024'."""
        assert _normalize_session("2023-24") == "90th_2023-2024"

    def test_special_session_passthrough(self):
        """'2024s' passes through unchanged."""
        assert _normalize_session("2024s") == "2024s"

    def test_already_normalized_underscore_becomes_dash(self):
        """Underscore in biennium format is normalized to dash."""
        assert _normalize_session("91st_2025-2026") == "91st-2025-2026"

    def test_bare_year_resolves_to_biennium(self):
        """Bare year resolves to full biennium name."""
        assert _normalize_session("2025") == "91st_2025-2026"
        assert _normalize_session("2001") == "79th_2001-2002"

    def test_legislature_number_resolves_to_biennium(self):
        """Legislature number (1-3 digits) resolves to full biennium name."""
        assert _normalize_session("79") == "79th_2001-2002"
        assert _normalize_session("91") == "91st_2025-2026"
        assert _normalize_session("84") == "84th_2011-2012"


# ── _next_run_label() ────────────────────────────────────────────────────────


class TestNextRunLabel:
    """Unique run labels always include .N suffix starting at .1."""

    def test_first_run_returns_dot_1(self, tmp_path):
        assert _next_run_label(tmp_path, "260223") == "260223.1"

    def test_second_run_returns_dot_2(self, tmp_path):
        (tmp_path / "260223.1").mkdir()
        assert _next_run_label(tmp_path, "260223") == "260223.2"

    def test_third_run_returns_dot_3(self, tmp_path):
        (tmp_path / "260223.1").mkdir()
        (tmp_path / "260223.2").mkdir()
        assert _next_run_label(tmp_path, "260223") == "260223.3"

    def test_gap_fills_next_available(self, tmp_path):
        """If .1 exists but .2 doesn't, next is .2 even if .3 exists."""
        (tmp_path / "260223.1").mkdir()
        (tmp_path / "260223.3").mkdir()  # gap at .2
        assert _next_run_label(tmp_path, "260223") == "260223.2"

    def test_symlink_not_counted_as_existing(self, tmp_path):
        """A symlink doesn't count as an existing run."""
        target = tmp_path / "something"
        target.mkdir()
        (tmp_path / "260223.1").symlink_to(target)
        assert _next_run_label(tmp_path, "260223") == "260223.1"

    def test_nonexistent_analysis_dir(self, tmp_path):
        """Works even if the analysis dir doesn't exist yet."""
        nonexistent = tmp_path / "does_not_exist"
        assert _next_run_label(nonexistent, "260223") == "260223.1"


# ── _format_elapsed() ────────────────────────────────────────────────────────


class TestFormatElapsed:
    """Human-readable elapsed time formatting."""

    def test_seconds_only(self):
        assert _format_elapsed(3.2) == "3.2s"

    def test_zero(self):
        assert _format_elapsed(0.0) == "0.0s"

    def test_minutes_and_seconds(self):
        assert _format_elapsed(105) == "1m 45s"

    def test_hours_minutes_seconds(self):
        assert _format_elapsed(4325) == "1h 12m 5s"

    def test_exactly_one_minute(self):
        assert _format_elapsed(60) == "1m 0s"

    def test_just_under_one_minute(self):
        assert _format_elapsed(59.9) == "59.9s"


# ── _git_commit_hash() ───────────────────────────────────────────────────────


class TestGitCommitHash:
    """Get current git commit hash."""

    def test_returns_string(self):
        result = _git_commit_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_hex_or_unknown(self):
        """Result is either a 40-char hex hash or 'unknown'."""
        result = _git_commit_hash()
        if result != "unknown":
            assert len(result) == 40
            assert all(c in "0123456789abcdef" for c in result)


# ── strip_leadership_suffix() ─────────────────────────────────────────────────


class TestStripLeadershipSuffix:
    """Strip leadership titles from legislator display names."""

    def test_strips_president_of_senate(self):
        assert strip_leadership_suffix("Ty Masterson - President of the Senate") == "Ty Masterson"

    def test_strips_vice_president(self):
        result = strip_leadership_suffix("Tim Shallenburger - Vice President of the Senate")
        assert result == "Tim Shallenburger"

    def test_strips_house_leader(self):
        assert strip_leadership_suffix("John Alcala - House Majority Leader") == "John Alcala"

    def test_strips_speaker_pro_tem(self):
        assert strip_leadership_suffix("Alice Brown - Speaker Pro Tem") == "Alice Brown"

    def test_no_suffix_unchanged(self):
        assert strip_leadership_suffix("Mary Ware") == "Mary Ware"

    def test_empty_string(self):
        assert strip_leadership_suffix("") == ""

    def test_hyphenated_name_preserved(self):
        """Suffix pattern requires space-dash-space, not a bare hyphen."""
        assert strip_leadership_suffix("Mary Smith-Jones") == "Mary Smith-Jones"


# ── RunContext ────────────────────────────────────────────────────────────────


class TestRunContext:
    """Context manager for structured analysis output."""

    def test_setup_creates_directories(self, tmp_path):
        ctx = RunContext(
            session="2025-26",
            analysis_name="test_analysis",
            results_root=tmp_path,
        )
        ctx.setup()
        assert ctx.plots_dir.exists()
        assert ctx.data_dir.exists()
        assert ctx.run_dir.exists()
        # Restore stdout since setup replaces it
        ctx.finalize()

    def test_session_normalized(self, tmp_path):
        ctx = RunContext(
            session="2025-26",
            analysis_name="test",
            results_root=tmp_path,
        )
        assert ctx.session == "91st_2025-2026"

    def test_finalize_writes_run_info(self, tmp_path):
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            params={"flag": True},
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        info_path = ctx.run_dir / "run_info.json"
        assert info_path.exists()
        data = json.loads(info_path.read_text())
        assert data["analysis"] == "test"
        assert data["session"] == "2024s"
        assert data["params"]["flag"] is True
        assert "git_commit" in data
        assert "python_version" in data
        assert "elapsed_seconds" in data
        assert isinstance(data["elapsed_seconds"], float)
        assert data["elapsed_seconds"] >= 0
        assert "elapsed_display" in data

    def test_finalize_writes_run_log(self, tmp_path):
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx.setup()
        print("test log line")
        ctx.finalize()
        log_path = ctx.run_dir / "run_log.txt"
        assert log_path.exists()
        assert "test log line" in log_path.read_text()

    def test_finalize_creates_latest_symlink(self, tmp_path):
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        latest = tmp_path / "2024s" / "test" / "latest"
        assert latest.is_symlink()

    def test_context_manager_protocol(self, tmp_path):
        """__enter__ returns self, __exit__ calls finalize."""
        with RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        ) as ctx:
            assert isinstance(ctx, RunContext)
            assert ctx.plots_dir.exists()
        # After exit, run_info should exist
        assert (ctx.run_dir / "run_info.json").exists()

    def test_primer_written(self, tmp_path):
        primer_text = "# Test Analysis\nThis is a primer."
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            primer=primer_text,
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        readme = tmp_path / "2024s" / "test" / "README.md"
        assert readme.exists()
        assert readme.read_text() == primer_text

    def test_no_primer_no_readme(self, tmp_path):
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        readme = tmp_path / "2024s" / "test" / "README.md"
        assert not readme.exists()

    def test_stdout_restored_after_finalize(self, tmp_path):
        original = sys.stdout
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx.setup()
        assert sys.stdout is not original  # tee is active
        ctx.finalize()
        assert sys.stdout is original  # restored

    def test_default_params_empty_dict(self, tmp_path):
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        assert ctx.params == {}

    def test_consecutive_runs_get_separate_dirs(self, tmp_path):
        """Second run on same day gets .2 suffix."""
        ctx1 = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx1.setup()
        print("first run output")
        ctx1.finalize()

        first_run_dir = ctx1.run_dir
        assert first_run_dir.exists()
        assert first_run_dir.name.endswith(".1")

        ctx2 = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx2.setup()
        print("second run output")
        ctx2.finalize()

        second_run_dir = ctx2.run_dir
        assert second_run_dir.exists()
        assert first_run_dir != second_run_dir
        assert second_run_dir.name.endswith(".2")

        # First run's output preserved
        assert "first run output" in (first_run_dir / "run_log.txt").read_text()
        assert "second run output" in (second_run_dir / "run_log.txt").read_text()

        # Latest points to the second run
        latest = tmp_path / "2024s" / "test" / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == second_run_dir.resolve()

    def test_third_run_gets_suffix_3(self, tmp_path):
        """Third same-day run gets .3 suffix."""
        for _ in range(3):
            ctx = RunContext(
                session="2024s",
                analysis_name="test",
                results_root=tmp_path,
            )
            ctx.setup()
            ctx.finalize()

        # Should have .1, .2, and .3
        analysis_dir = tmp_path / "2024s" / "test"
        today = ctx._today
        assert (analysis_dir / f"{today}.1").exists()
        assert (analysis_dir / f"{today}.2").exists()
        assert (analysis_dir / f"{today}.3").exists()

    def test_run_info_includes_run_label(self, tmp_path):
        """run_info.json records the run_label for traceability."""
        ctx1 = RunContext(session="2024s", analysis_name="test", results_root=tmp_path)
        ctx1.setup()
        ctx1.finalize()

        info = json.loads((ctx1.run_dir / "run_info.json").read_text())
        assert info["run_label"].endswith(".1")


# ── generate_run_id() ──────────────────────────────────────────────────────


class TestGenerateRunId:
    """Generate run IDs for grouped pipeline output."""

    def test_format_matches_pattern(self):
        """Run ID has format {bb}-{YYMMDD}.{n}."""
        import re

        run_id = generate_run_id("2025-26")
        assert re.match(r"^91-\d{6}\.\d+$", run_id)

    def test_first_run_starts_at_dot_1(self):
        """First run always gets .1 suffix."""
        run_id = generate_run_id("2025-26")
        assert run_id.endswith(".1")

    def test_special_session_prefix(self):
        """Special sessions use the full session string as prefix."""
        import re

        run_id = generate_run_id("2024s")
        assert re.match(r"^2024s-\d{6}\.\d+$", run_id)

    def test_historical_session(self):
        """Historical sessions extract the legislature number."""
        run_id = generate_run_id("2023-24")
        assert run_id.startswith("90-")

    def test_same_day_without_root(self):
        """Without results_root, same-day calls always produce .1."""
        id1 = generate_run_id("2025-26")
        id2 = generate_run_id("2025-26")
        assert id1 == id2
        assert id1.endswith(".1")

    def test_same_day_collision_with_root(self, tmp_path):
        """With results_root, same-day runs increment: .1, .2, .3."""
        id1 = generate_run_id("2025-26", results_root=tmp_path)
        assert id1.endswith(".1")
        (tmp_path / id1).mkdir()
        id2 = generate_run_id("2025-26", results_root=tmp_path)
        assert id2.endswith(".2")
        (tmp_path / id2).mkdir()
        id3 = generate_run_id("2025-26", results_root=tmp_path)
        assert id3.endswith(".3")

    def test_no_colons_in_id(self):
        """Run IDs use hyphens, not colons, for filesystem safety."""
        run_id = generate_run_id("2025-26")
        assert ":" not in run_id

    def test_ordinal_suffix_stripped(self):
        """Legislature ordinal suffix (st/nd/rd/th) should be stripped."""
        run_id = generate_run_id("2025-26")
        assert run_id.startswith("91-")  # not "91st-"
        run_id_84 = generate_run_id("2011-12")
        assert run_id_84.startswith("84-")  # not "84th-"


# ── resolve_upstream_dir() ─────────────────────────────────────────────────


class TestResolveUpstreamDir:
    """Resolve upstream phase directories with 4-level precedence."""

    def test_explicit_override_wins(self, tmp_path):
        """Precedence 1: explicit CLI override takes priority over everything."""
        override = tmp_path / "my_custom_eda"
        result = resolve_upstream_dir("01_eda", tmp_path, run_id="run-123", override=override)
        assert result == override

    def test_run_id_path(self, tmp_path):
        """Precedence 2: run_id constructs results_root/{run_id}/{phase}."""
        result = resolve_upstream_dir("01_eda", tmp_path, run_id="91st-2026-02-27T19-30-00")
        assert result == tmp_path / "91st-2026-02-27T19-30-00" / "01_eda"

    def test_legacy_phase_latest(self, tmp_path):
        """Precedence 3: legacy path results_root/{phase}/latest when it exists."""
        legacy = tmp_path / "01_eda" / "latest"
        legacy.mkdir(parents=True)
        result = resolve_upstream_dir("01_eda", tmp_path)
        assert result == legacy

    def test_new_layout_fallback(self, tmp_path):
        """Precedence 4: results_root/latest/{phase} when legacy doesn't exist."""
        result = resolve_upstream_dir("01_eda", tmp_path)
        assert result == tmp_path / "latest" / "01_eda"

    def test_override_none_uses_run_id(self, tmp_path):
        """override=None falls through to run_id path."""
        result = resolve_upstream_dir("02_pca", tmp_path, run_id="run-1", override=None)
        assert result == tmp_path / "run-1" / "02_pca"

    def test_no_run_id_no_override_legacy_exists(self, tmp_path):
        """With both None, prefers legacy if it exists."""
        legacy = tmp_path / "05_irt" / "latest"
        legacy.mkdir(parents=True)
        result = resolve_upstream_dir("05_irt", tmp_path, run_id=None, override=None)
        assert result == legacy


# ── RunContext run-directory mode ──────────────────────────────────────────


class TestRunContextRunIdMode:
    """RunContext with run_id groups phases under a single run directory."""

    def test_run_dir_under_run_id(self, tmp_path):
        """run_dir is results/{session}/{run_id}/{analysis}/."""
        ctx = RunContext(
            session="2024s",
            analysis_name="01_eda",
            run_id="2024s-2026-02-27T19-30-00",
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        expected = tmp_path / "2024s" / "2024s-2026-02-27T19-30-00" / "01_eda"
        assert ctx.run_dir == expected
        assert ctx.run_dir.exists()

    def test_session_level_latest_symlink(self, tmp_path):
        """Session-level latest symlink points to the run_id directory."""
        run_id = "2024s-2026-02-27T19-30-00"
        ctx = RunContext(
            session="2024s",
            analysis_name="01_eda",
            run_id=run_id,
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        latest = tmp_path / "2024s" / "latest"
        assert latest.is_symlink()
        assert str(latest.readlink()) == run_id

    def test_session_level_symlink_idempotent(self, tmp_path):
        """Multiple phases in the same run write the same session-level symlink."""
        run_id = "2024s-2026-02-27T19-30-00"
        for phase in ["01_eda", "02_pca", "05_irt"]:
            ctx = RunContext(
                session="2024s",
                analysis_name=phase,
                run_id=run_id,
                results_root=tmp_path,
            )
            ctx.setup()
            ctx.finalize()
        latest = tmp_path / "2024s" / "latest"
        assert latest.is_symlink()
        assert str(latest.readlink()) == run_id

    def test_run_info_includes_run_id(self, tmp_path):
        """run_info.json includes the run_id field."""
        run_id = "2024s-2026-02-27T19-30-00"
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            run_id=run_id,
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        info = json.loads((ctx.run_dir / "run_info.json").read_text())
        assert info["run_id"] == run_id

    def test_plots_and_data_dirs_created(self, tmp_path):
        """plots/ and data/ subdirs created inside the run-directory phase dir."""
        ctx = RunContext(
            session="2024s",
            analysis_name="02_pca",
            run_id="run-1",
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        assert ctx.plots_dir.exists()
        assert ctx.data_dir.exists()
        assert ctx.plots_dir == ctx.run_dir / "plots"
        assert ctx.data_dir == ctx.run_dir / "data"

    def test_failed_run_no_symlink(self, tmp_path):
        """Failed runs don't update the session-level latest symlink."""
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            run_id="run-fail",
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize(failed=True)
        latest = tmp_path / "2024s" / "latest"
        assert not latest.exists()


class TestRunContextFlatMode:
    """RunContext flat mode for non-biennium sessions (special, cross-session)."""

    def test_flat_dir_structure(self, tmp_path):
        """Non-biennium session uses flat path: results/{session}/{analysis}/{date}/."""
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        # run_dir should be under {session}/{analysis}/{date}
        assert ctx.run_dir.parent.name == "test"
        assert ctx.run_dir.parent.parent.name == "2024s"

    def test_flat_phase_level_symlink(self, tmp_path):
        """Non-biennium session has latest symlink at the phase level."""
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        latest = tmp_path / "2024s" / "test" / "latest"
        assert latest.is_symlink()
        # Session-level latest should NOT exist
        session_latest = tmp_path / "2024s" / "latest"
        assert not session_latest.exists()

    def test_run_info_run_id_null(self, tmp_path):
        """run_info.json has run_id: null in flat mode."""
        ctx = RunContext(
            session="2024s",
            analysis_name="test",
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        info = json.loads((ctx.run_dir / "run_info.json").read_text())
        assert info["run_id"] is None


class TestRunContextAutoRunId:
    """Biennium sessions auto-generate a run_id when none is provided."""

    def test_auto_generates_run_id(self, tmp_path):
        """Biennium session without explicit run_id gets one auto-generated."""
        ctx = RunContext(
            session="2025-26",
            analysis_name="01_eda",
            results_root=tmp_path,
        )
        assert ctx.run_id is not None
        assert ctx.run_id.startswith("91-")

    def test_auto_run_id_creates_run_dir_structure(self, tmp_path):
        """Auto-generated run_id produces results/{session}/{run_id}/{analysis}/."""
        ctx = RunContext(
            session="2025-26",
            analysis_name="01_eda",
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        # run_dir should be under {session}/{run_id}/{analysis}
        assert ctx.run_dir.name == "01_eda"
        assert ctx.run_dir.parent.parent.name == "91st_2025-2026"

    def test_auto_run_id_does_not_update_latest(self, tmp_path):
        """Auto-generated run_id does NOT create a session-level latest symlink."""
        ctx = RunContext(
            session="2025-26",
            analysis_name="01_eda",
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        latest = tmp_path / "91st_2025-2026" / "latest"
        assert not latest.exists()

    def test_explicit_run_id_does_update_latest(self, tmp_path):
        """Explicit run_id DOES create a session-level latest symlink."""
        run_id = "91-260301.1"
        ctx = RunContext(
            session="2025-26",
            analysis_name="01_eda",
            run_id=run_id,
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()
        latest = tmp_path / "91st_2025-2026" / "latest"
        assert latest.is_symlink()
        assert str(latest.readlink()) == run_id

    def test_auto_run_id_preserves_existing_latest(self, tmp_path):
        """Auto-generated run_id preserves an existing latest symlink."""
        session_root = tmp_path / "91st_2025-2026"
        session_root.mkdir(parents=True)
        # Pre-set latest → a pipeline run
        original_target = "91-260301.1"
        (session_root / original_target).mkdir()
        latest = session_root / "latest"
        latest.symlink_to(original_target)

        # Run standalone phase with auto-generated run_id
        ctx = RunContext(
            session="2025-26",
            analysis_name="01_eda",
            results_root=tmp_path,
        )
        ctx.setup()
        ctx.finalize()

        # latest should still point to the original pipeline run
        assert latest.is_symlink()
        assert str(latest.readlink()) == original_target

    def test_auto_run_id_increments(self, tmp_path):
        """Successive auto-generated run_ids increment the counter."""
        ctx1 = RunContext(
            session="2025-26",
            analysis_name="01_eda",
            results_root=tmp_path,
        )
        ctx1.setup()
        ctx1.finalize()
        rid1 = ctx1.run_id

        ctx2 = RunContext(
            session="2025-26",
            analysis_name="02_pca",
            results_root=tmp_path,
        )
        ctx2.setup()
        ctx2.finalize()
        rid2 = ctx2.run_id

        # Both should be for the same day but different increments
        assert rid1 != rid2
        assert rid1.endswith(".1")
        assert rid2.endswith(".2")
