"""Tests for analysis.db — database loading for the analysis pipeline.

Unit tests mock the database connection; integration tests require PostgreSQL
(marked ``@pytest.mark.web``).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

# ── Unit Tests (mocked DB) ──────────────────────────────────────────────────


class TestSessionNameMapping:
    def test_session_name_from_data_dir(self):
        from analysis.db import _session_name_from_data_dir

        assert _session_name_from_data_dir(Path("data/kansas/91st_2025-2026")) == "91st_2025-2026"
        assert _session_name_from_data_dir(Path("data/kansas/84th_2011-2012")) == "84th_2011-2012"


class TestDbAvailable:
    def test_returns_false_when_no_db(self):
        from analysis.db import db_available

        with patch("analysis.db.get_connection", side_effect=Exception("no db")):
            assert db_available() is False

    def test_returns_true_when_connected(self):
        from analysis.db import db_available

        mock_conn = MagicMock()
        mock_conn.execute.return_value = None
        with patch("analysis.db.get_connection", return_value=mock_conn):
            assert db_available() is True


class TestCsvFallback:
    """Verify that use_csv=True skips DB entirely and reads CSV."""

    def test_load_votes_csv(self, tmp_path):
        from analysis.db import load_votes

        data_dir = tmp_path / "91st_2025-2026"
        data_dir.mkdir()
        csv = data_dir / "91st_2025-2026_votes.csv"
        csv.write_text("session,vote_id,vote\n91st (2025-2026),v1,Yea\n")

        df = load_votes(data_dir, use_csv=True)
        assert df.height == 1
        assert df["vote"][0] == "Yea"

    def test_load_rollcalls_csv(self, tmp_path):
        from analysis.db import load_rollcalls

        data_dir = tmp_path / "91st_2025-2026"
        data_dir.mkdir()
        csv = data_dir / "91st_2025-2026_rollcalls.csv"
        csv.write_text("session,vote_id,bill_number,chamber\n91st (2025-2026),v1,HB 1,House\n")

        df = load_rollcalls(data_dir, use_csv=True)
        assert df.height == 1

    def test_load_legislators_csv(self, tmp_path):
        from analysis.db import load_legislators

        data_dir = tmp_path / "91st_2025-2026"
        data_dir.mkdir()
        csv = data_dir / "91st_2025-2026_legislators.csv"
        csv.write_text(
            "session,name,full_name,slug,chamber,party,district,member_url,ocd_id\n"
            "91st (2025-2026),Smith,Smith,rep_smith,House,,1,http://example.com,\n"
        )

        df = load_legislators(data_dir, use_csv=True)
        assert df.height == 1
        # Verify cleaning: empty party -> Independent
        assert df["party"][0] == "Independent"

    def test_load_bill_texts_csv(self, tmp_path):
        from analysis.db import load_bill_texts

        data_dir = tmp_path / "91st_2025-2026"
        data_dir.mkdir()
        csv = data_dir / "91st_2025-2026_bill_texts.csv"
        csv.write_text(
            "bill_number,document_type,version,text,page_count,source_url\n"
            "HB 1,introduced,1,Some text,1,http://example.com\n"
        )

        df = load_bill_texts(data_dir, use_csv=True)
        assert df.height == 1

    def test_load_alec_csv(self, tmp_path):
        from analysis.db import load_alec

        alec_dir = tmp_path
        csv = alec_dir / "alec_model_bills.csv"
        csv.write_text(
            "title,text,category,bill_type,date,url,task_force\n"
            "Test Bill,Some text,Cat,Resolution,2024-01-01,http://example.com,TF\n"
        )

        df = load_alec(alec_dir, use_csv=True)
        assert df.height == 1


class TestDbFallbackToCSV:
    """When DB is unavailable, load_*() should fall back to CSV."""

    def test_votes_falls_back(self, tmp_path):
        from analysis.db import load_votes

        data_dir = tmp_path / "91st_2025-2026"
        data_dir.mkdir()
        csv = data_dir / "91st_2025-2026_votes.csv"
        csv.write_text("session,vote_id,vote\n91st (2025-2026),v1,Yea\n")

        with patch("analysis.db.load_votes_db", side_effect=Exception("no db")):
            df = load_votes(data_dir)
            assert df.height == 1

    def test_rollcalls_falls_back(self, tmp_path):
        from analysis.db import load_rollcalls

        data_dir = tmp_path / "91st_2025-2026"
        data_dir.mkdir()
        csv = data_dir / "91st_2025-2026_rollcalls.csv"
        csv.write_text("session,vote_id,chamber\n91st (2025-2026),v1,House\n")

        with patch("analysis.db.load_rollcalls_db", side_effect=Exception("no db")):
            df = load_rollcalls(data_dir)
            assert df.height == 1

    def test_legislators_falls_back(self, tmp_path):
        from analysis.db import load_legislators

        data_dir = tmp_path / "91st_2025-2026"
        data_dir.mkdir()
        csv = data_dir / "91st_2025-2026_legislators.csv"
        csv.write_text(
            "session,name,full_name,slug,chamber,party,district,member_url,ocd_id\n"
            "91st (2025-2026),Smith,Smith,rep_smith,House,Republican,1,http://example.com,\n"
        )

        with patch("analysis.db.load_legislators_db", side_effect=Exception("no db")):
            df = load_legislators(data_dir)
            assert df.height == 1


class TestCleanLegislatorsDf:
    def test_fills_independent(self):
        from analysis.db import _clean_legislators_df

        df = pl.DataFrame(
            {
                "full_name": ["Smith"],
                "party": [""],
                "ocd_id": [None],
            }
        )
        result = _clean_legislators_df(df)
        assert result["party"][0] == "Independent"
        assert result["ocd_id"][0] == ""

class TestTryDb:
    def test_returns_none_on_exception(self):
        from analysis.db import _try_db

        def bad_loader():
            raise RuntimeError("boom")

        assert _try_db(bad_loader) is None

    def test_returns_none_on_empty_df(self):
        from analysis.db import _try_db

        def empty_loader():
            return pl.DataFrame({"a": []})

        assert _try_db(empty_loader) is None

    def test_returns_df_on_success(self):
        from analysis.db import _try_db

        def good_loader():
            return pl.DataFrame({"a": [1]})

        result = _try_db(good_loader)
        assert result is not None
        assert result.height == 1


# ── Integration Tests (require PostgreSQL) ──────────────────────────────────


@pytest.mark.web
class TestDbLoadingIntegration:
    """Integration tests that query real PostgreSQL data.

    Requires ``just db-up`` and data loaded via ``just db-load-all``.
    """

    def test_db_available(self):
        from analysis.db import db_available

        assert db_available() is True

    def test_load_votes_db(self):
        from analysis.db import load_votes_db

        df = load_votes_db("91st_2025-2026")
        assert df.height > 0
        expected_cols = {"session", "bill_number", "vote_id", "legislator_name", "vote"}
        assert expected_cols.issubset(set(df.columns))

    def test_load_rollcalls_db(self):
        from analysis.db import load_rollcalls_db

        df = load_rollcalls_db("91st_2025-2026")
        assert df.height > 0
        assert "vote_id" in df.columns
        assert "chamber" in df.columns

    def test_load_legislators_db(self):
        from analysis.db import load_legislators_db

        df = load_legislators_db("91st_2025-2026")
        assert df.height > 0
        assert "slug" in df.columns
        assert "party" in df.columns

    def test_load_bill_texts_db(self):
        from analysis.db import load_bill_texts_db

        df = load_bill_texts_db("91st_2025-2026")
        assert df.height > 0
        assert "bill_number" in df.columns
        assert "text" in df.columns

    def test_load_alec_db(self):
        from analysis.db import load_alec_db

        df = load_alec_db()
        assert df.height > 0
        assert "title" in df.columns
        assert "text" in df.columns

    def test_routing_uses_db_by_default(self):
        """Verify that load_votes() uses DB when available."""
        from analysis.db import load_votes

        data_dir = Path("data/kansas/91st_2025-2026")
        df = load_votes(data_dir)
        assert df.height > 0

    def test_session_format_matches_csv(self):
        """Verify DB session format matches CSV convention."""
        from analysis.db import load_votes_db

        df = load_votes_db("91st_2025-2026")
        if df.height > 0:
            session_val = df["session"][0]
            assert session_val == "91st (2025-2026)"
