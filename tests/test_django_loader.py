"""Tests for the CSV-to-PostgreSQL loader management commands.

Run: just test-web
Requires: PostgreSQL running (just db-up)
"""

import os
import sys
from pathlib import Path

import pytest

django = pytest.importorskip("django")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tallgrass_web.settings.test")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "web"))
django.setup()

from io import StringIO  # noqa: E402

from django.core.management import call_command  # noqa: E402
from django.core.management.base import CommandError  # noqa: E402
from legislature.management.commands.load_session import (  # noqa: E402
    parse_bool,
    parse_date,
    parse_datetime,
    parse_int,
    parse_session_name,
    parse_vote_date,
)
from legislature.models import (  # noqa: E402
    ALECModelBill,
    BillAction,
    BillText,
    Legislator,
    RollCall,
    Session,
    Vote,
)

pytestmark = [pytest.mark.web, pytest.mark.django_db]

# -- CSV header constants (too long for inline) -----

_VOTES_HDR = (
    "session,bill_number,bill_title,vote_id,"
    "vote_datetime,vote_date,chamber,motion,"
    "legislator_name,legislator_slug,vote"
)

_RC_HDR = (
    "session,bill_number,bill_title,vote_id,"
    "vote_url,vote_datetime,vote_date,chamber,"
    "motion,vote_type,result,short_title,"
    "sponsor,sponsor_slugs,"
    "yea_count,nay_count,present_passing_count,"
    "absent_not_voting_count,"
    "not_voting_count,total_votes,passed"
)

_LEG_HDR = "name,full_name,legislator_slug,chamber,party,district,member_url,ocd_id"

_BA_HDR = (
    "session,bill_number,action_code,chamber,"
    "committee_names,occurred_datetime,"
    "session_date,status,journal_page_number"
)


# -- Unit tests: parsing helpers ------------------


class TestParseSessionName:
    """Session name parsing from directory names."""

    def test_regular_91st(self):
        r = parse_session_name("91st_2025-2026")
        assert r["legislature_number"] == 91
        assert r["start_year"] == 2025
        assert r["end_year"] == 2026
        assert r["is_special"] is False

    def test_regular_84th(self):
        r = parse_session_name("84th_2011-2012")
        assert r["legislature_number"] == 84
        assert r["start_year"] == 2011

    def test_regular_90th(self):
        r = parse_session_name("90th_2023-2024")
        assert r["legislature_number"] == 90

    def test_regular_82nd(self):
        r = parse_session_name("82nd_2007-2008")
        assert r["legislature_number"] == 82

    def test_regular_83rd(self):
        r = parse_session_name("83rd_2009-2010")
        assert r["legislature_number"] == 83

    def test_special_2024(self):
        r = parse_session_name("2024s")
        assert r["legislature_number"] == 90
        assert r["start_year"] == 2024
        assert r["end_year"] == 2024
        assert r["is_special"] is True

    def test_special_2020(self):
        r = parse_session_name("2020s")
        assert r["legislature_number"] == 88
        assert r["is_special"] is True

    def test_special_2013(self):
        r = parse_session_name("2013s")
        assert r["legislature_number"] == 85
        assert r["is_special"] is True

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_session_name("bad_name")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_session_name("")


class TestParseDateHelpers:
    """Date and type conversion helpers."""

    def test_parse_vote_date_normal(self):
        from datetime import date

        assert parse_vote_date("03/20/2025") == date(2025, 3, 20)

    def test_parse_vote_date_empty(self):
        assert parse_vote_date("") is None

    def test_parse_datetime_normal(self):
        from datetime import datetime

        r = parse_datetime("2025-03-20T20:35:13")
        assert r == datetime(2025, 3, 20, 20, 35, 13)

    def test_parse_datetime_empty(self):
        assert parse_datetime("") is None

    def test_parse_date_normal(self):
        from datetime import date

        assert parse_date("2025-03-20") == date(2025, 3, 20)

    def test_parse_date_empty(self):
        assert parse_date("") is None

    def test_parse_bool_true(self):
        assert parse_bool("True") is True

    def test_parse_bool_false(self):
        assert parse_bool("False") is False

    def test_parse_bool_empty(self):
        assert parse_bool("") is None

    def test_parse_bool_other(self):
        assert parse_bool("maybe") is None

    def test_parse_int_normal(self):
        assert parse_int("42") == 42

    def test_parse_int_empty(self):
        assert parse_int("") == 0

    def test_parse_int_zero(self):
        assert parse_int("0") == 0


# -- Helpers for integration tests ----------------


def _write_csv(path: Path, content: str):
    """Write CSV content to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _mini_leg(session_dir: Path, prefix: str):
    """Write a 1-legislator CSV."""
    _write_csv(
        session_dir / f"{prefix}_legislators.csv",
        f"{_LEG_HDR}\nSmith,John Smith,sen_smith_john_1,Senate,Republican,16,,\n",
    )


def _vid_to_iso(vid: str) -> str:
    """Convert je_YYYYMMDDHHMMSS to ISO datetime."""
    d = vid.replace("je_", "").replace("kf_", "")
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{d[8:10]}:{d[10:12]}:{d[12:14]}"


def _mini_rc(session_dir: Path, prefix: str, sess: str, vid: str, vdate: str):
    """Write a 1-rollcall CSV."""
    iso = _vid_to_iso(vid)
    _write_csv(
        session_dir / f"{prefix}_rollcalls.csv",
        f"{_RC_HDR}\n"
        f"{sess},SB 1,Test,{vid},,"
        f"{iso},{vdate},"
        "Senate,Final,Final,Passed,Test,,,1,0,0,0,0,1,True\n",
    )


def _mini_votes(session_dir: Path, prefix: str, sess: str, vid: str, vdate: str):
    """Write a 1-vote CSV."""
    iso = _vid_to_iso(vid)
    _write_csv(
        session_dir / f"{prefix}_votes.csv",
        f"{_VOTES_HDR}\n"
        f"{sess},SB 1,Test,{vid},"
        f"{iso},{vdate},"
        "Senate,Final,Smith,sen_smith_john_1,Yea\n",
    )


def _make_session(
    tmp_path: Path,
    name: str,
    sess_label: str,
    vid: str,
    vdate: str,
):
    """Create a minimal 1-legislator, 1-rollcall, 1-vote session."""
    d = tmp_path / "data" / "kansas" / name
    _mini_leg(d, name)
    _mini_rc(d, name, sess_label, vid, vdate)
    _mini_votes(d, name, sess_label, vid, vdate)
    return d


@pytest.fixture
def synthetic_session(tmp_path):
    """Create a synthetic session: 5 legislators, 2 rollcalls, 10 votes."""
    d = tmp_path / "data" / "kansas" / "91st_2025-2026"
    prefix = "91st_2025-2026"

    _write_csv(
        d / f"{prefix}_legislators.csv",
        f"{_LEG_HDR}\n"
        "Smith,John Smith,sen_smith_john_1,Senate,Republican,16,,\n"
        "Jones,Mary Jones,sen_jones_mary_1,Senate,Democrat,20,,\n"
        "Brown,Bob Brown,rep_brown_bob_1,House,Republican,42,,\n"
        "Davis,Ann Davis,rep_davis_ann_1,House,Democrat,57,,\n"
        "Wilson,Pat Wilson,rep_wilson_pat_1,House,Republican,1,,\n",
    )

    _write_csv(
        d / f"{prefix}_rollcalls.csv",
        f"{_RC_HDR}\n"
        "91st (2025-2026),SB 1,Test bill,"
        "je_20250320203513,http://example.com,"
        "2025-03-20T20:35:13,03/20/2025,Senate,"
        "Final Action,Final Action,Passed,Test,"
        "Senator Smith,sen_smith_john_1,"
        "3,2,0,0,0,5,True\n"
        "91st (2025-2026),HB 100,Another bill,"
        "je_20250321120000,http://example.com,"
        "2025-03-21T12:00:00,03/21/2025,House,"
        "Emergency Final Action,Emergency,Failed,"
        "Another,Rep Brown,,2,3,0,0,0,5,False\n",
    )

    legs = [
        ("Smith", "sen_smith_john_1"),
        ("Jones", "sen_jones_mary_1"),
        ("Brown", "rep_brown_bob_1"),
        ("Davis", "rep_davis_ann_1"),
        ("Wilson", "rep_wilson_pat_1"),
    ]
    vids = ["je_20250320203513", "je_20250321120000"]
    patterns = [
        ["Yea", "Nay", "Yea", "Nay", "Yea"],
        ["Nay", "Yea", "Nay", "Yea", "Nay"],
    ]
    rows = [_VOTES_HDR]
    for vid, pat in zip(vids, patterns):
        for (nm, sl), v in zip(legs, pat):
            rows.append(
                f"91st (2025-2026),SB 1,Test,{vid},"
                f"2025-03-20T20:35:13,03/20/2025,"
                f"Senate,Final Action,{nm},{sl},{v}"
            )
    _write_csv(d / f"{prefix}_votes.csv", "\n".join(rows) + "\n")

    _write_csv(
        d / f"{prefix}_bill_actions.csv",
        f"{_BA_HDR}\n"
        "91st_2025-2026,SB 1,intro,Senate,,"
        "2025-01-13T10:00:00,2025-01-13,Introduced,\n"
        "91st_2025-2026,HB 100,intro,House,"
        "Judiciary; Ways and Means,"
        "2025-01-14T11:00:00,2025-01-14,Introduced,42\n",
    )

    _write_csv(
        d / f"{prefix}_bill_texts.csv",
        "session,bill_number,document_type,"
        "version,text,page_count,source_url\n"
        "91st (2025-2026),SB 1,introduced,"
        '00_0000,"AN ACT concerning tests",'
        "3,http://example.com/sb1.pdf\n"
        "91st (2025-2026),HB 100,introduced,"
        '00_0000,"AN ACT concerning more tests",'
        "5,http://example.com/hb100.pdf\n",
    )

    return d


@pytest.fixture
def alec_csv(tmp_path):
    """Create a minimal ALEC CSV."""
    path = tmp_path / "alec_model_bills.csv"
    _write_csv(
        path,
        "title,category,bill_type,date,task_force,url,text\n"
        "Castle Doctrine Act,Criminal Justice,"
        "Model Policy,,Public Safety,"
        "http://alec.org/1,Section 1...\n"
        "Right to Work Act,Labor,Model Policy,,"
        "Commerce,http://alec.org/2,Section 1...\n"
        "Voter ID Act,Elections,Model Policy,,"
        "ACCE,http://alec.org/3,Section 1...\n",
    )
    return path


def _load(name, tmp_path, **kwargs):
    """Call load_session with --data-root pointing at tmp_path."""
    root = str(tmp_path / "data" / "kansas")
    call_command(
        "load_session",
        name,
        "--data-root",
        root,
        stdout=kwargs.pop("stdout", StringIO()),
        **kwargs,
    )


# -- Integration tests: load_session ---------------


class TestLoadSession:
    """Integration tests for load_session."""

    def test_load_basic(self, synthetic_session, tmp_path):
        _load("91st_2025-2026", tmp_path)
        assert Legislator.objects.count() == 5
        assert RollCall.objects.count() == 2
        assert Vote.objects.count() == 10
        assert BillAction.objects.count() == 2
        assert BillText.objects.count() == 2

    def test_session_created(self, synthetic_session, tmp_path):
        _load("91st_2025-2026", tmp_path)
        s = Session.objects.get(
            state__code="KS",
            start_year=2025,
            is_special=False,
        )
        assert s.legislature_number == 91
        assert s.end_year == 2026
        assert s.name == "91st_2025-2026"

    def test_fk_integrity(self, synthetic_session, tmp_path):
        _load("91st_2025-2026", tmp_path)
        for vote in Vote.objects.all():
            assert vote.rollcall is not None
            assert vote.legislator is not None
            sess = vote.rollcall.session
            assert sess == vote.legislator.session

    def test_rollcall_data(self, synthetic_session, tmp_path):
        _load("91st_2025-2026", tmp_path)
        rc = RollCall.objects.get(vote_id="je_20250320203513")
        assert rc.bill_number == "SB 1"
        assert rc.chamber == "Senate"
        assert rc.passed is True
        assert rc.yea_count == 3
        assert rc.nay_count == 2

        rc2 = RollCall.objects.get(vote_id="je_20250321120000")
        assert rc2.passed is False

    def test_idempotent(self, synthetic_session, tmp_path):
        _load("91st_2025-2026", tmp_path)
        _load("91st_2025-2026", tmp_path)
        assert Legislator.objects.count() == 5
        assert RollCall.objects.count() == 2
        assert Vote.objects.count() == 10

    def test_dry_run(self, synthetic_session, tmp_path):
        out = StringIO()
        root = str(tmp_path / "data" / "kansas")
        call_command(
            "load_session",
            "91st_2025-2026",
            "--data-root",
            root,
            "--dry-run",
            stdout=out,
        )
        assert "Dry run" in out.getvalue()
        assert Legislator.objects.count() == 0

    def test_skip_bill_text(self, synthetic_session, tmp_path):
        root = str(tmp_path / "data" / "kansas")
        call_command(
            "load_session",
            "91st_2025-2026",
            "--data-root",
            root,
            "--skip-bill-text",
            stdout=StringIO(),
        )
        assert Legislator.objects.count() == 5
        assert BillText.objects.count() == 0

    def test_missing_optional_csvs(self, tmp_path):
        _make_session(
            tmp_path,
            "84th_2011-2012",
            "84th (2011-2012)",
            "je_20110101120000",
            "01/01/2011",
        )
        _load("84th_2011-2012", tmp_path)
        assert Legislator.objects.count() == 1
        assert Vote.objects.count() == 1
        assert BillAction.objects.count() == 0
        assert BillText.objects.count() == 0

    def test_special_session(self, tmp_path):
        _make_session(
            tmp_path,
            "2024s",
            "2024s",
            "je_20240601120000",
            "06/01/2024",
        )
        _load("2024s", tmp_path)
        s = Session.objects.get(start_year=2024, is_special=True)
        assert s.legislature_number == 90
        assert Legislator.objects.count() == 1

    def test_missing_data_dir_raises(self, tmp_path):
        with pytest.raises(CommandError, match="Data directory"):
            call_command(
                "load_session",
                "91st_2025-2026",
                "--data-root",
                str(tmp_path / "x"),
                stdout=StringIO(),
            )

    def test_missing_required_csv_raises(self, tmp_path):
        d = tmp_path / "data" / "kansas" / "91st_2025-2026"
        d.mkdir(parents=True)
        with pytest.raises(CommandError, match="Required CSV"):
            _load("91st_2025-2026", tmp_path)

    def test_invalid_session_name_raises(self):
        with pytest.raises(CommandError, match="Cannot parse"):
            call_command(
                "load_session",
                "bad_name",
                stdout=StringIO(),
            )

    def test_missing_column_raises(self, tmp_path):
        d = tmp_path / "data" / "kansas" / "91st_2025-2026"
        p = "91st_2025-2026"
        _write_csv(
            d / f"{p}_legislators.csv",
            "name,full_name,chamber\nSmith,John,Senate\n",
        )
        _write_csv(
            d / f"{p}_rollcalls.csv",
            "bill_number,vote_id,chamber\nSB 1,v1,Senate\n",
        )
        _write_csv(
            d / f"{p}_votes.csv",
            "legislator_slug,vote_id,vote\nslug,v1,Yea\n",
        )
        with pytest.raises(CommandError, match="missing required"):
            _load("91st_2025-2026", tmp_path)

    def test_bill_action_data(self, synthetic_session, tmp_path):
        _load("91st_2025-2026", tmp_path)
        ba = BillAction.objects.get(bill_number="HB 100")
        assert ba.chamber == "House"
        assert ba.committee_names == "Judiciary; Ways and Means"
        assert ba.journal_page_number == "42"

    def test_bill_text_data(self, synthetic_session, tmp_path):
        _load("91st_2025-2026", tmp_path)
        bt = BillText.objects.get(bill_number="SB 1")
        assert bt.document_type == "introduced"
        assert bt.page_count == 3
        assert "AN ACT" in bt.text

    def test_vote_categories(self, synthetic_session, tmp_path):
        _load("91st_2025-2026", tmp_path)
        cats = set(Vote.objects.values_list("vote", flat=True))
        assert "Yea" in cats
        assert "Nay" in cats

    def test_legislator_fields(self, synthetic_session, tmp_path):
        _load("91st_2025-2026", tmp_path)
        leg = Legislator.objects.get(legislator_slug="sen_smith_john_1")
        assert leg.name == "Smith"
        assert leg.full_name == "John Smith"
        assert leg.chamber == "Senate"
        assert leg.party == "Republican"
        assert leg.district == "16"

    def test_null_passed(self, tmp_path):
        d = tmp_path / "data" / "kansas" / "91st_2025-2026"
        p = "91st_2025-2026"
        _mini_leg(d, p)
        _write_csv(
            d / f"{p}_rollcalls.csv",
            f"{_RC_HDR}\n"
            "91st (2025-2026),SB 1,Test,"
            "je_20250320203513,,"
            "2025-03-20T20:35:13,03/20/2025,"
            "Senate,Final,Final,Unknown,Test,,,"
            "1,0,0,0,0,1,\n",
        )
        _mini_votes(
            d,
            p,
            "91st (2025-2026)",
            "je_20250320203513",
            "03/20/2025",
        )
        _load("91st_2025-2026", tmp_path)
        rc = RollCall.objects.get(vote_id="je_20250320203513")
        assert rc.passed is None

    def test_orphan_vote_skipped(self, tmp_path):
        d = tmp_path / "data" / "kansas" / "91st_2025-2026"
        p = "91st_2025-2026"
        _mini_leg(d, p)
        _write_csv(
            d / f"{p}_rollcalls.csv",
            f"{_RC_HDR}\n"
            "91st (2025-2026),SB 1,Test,"
            "je_20250320203513,,"
            "2025-03-20T20:35:13,03/20/2025,"
            "Senate,Final,Final,Passed,Test,,,"
            "1,0,0,0,0,1,True\n",
        )
        # Two votes: one valid slug, one unknown
        _write_csv(
            d / f"{p}_votes.csv",
            f"{_VOTES_HDR}\n"
            "91st (2025-2026),SB 1,Test,"
            "je_20250320203513,"
            "2025-03-20T20:35:13,03/20/2025,"
            "Senate,Final,Smith,sen_smith_john_1,Yea\n"
            "91st (2025-2026),SB 1,Test,"
            "je_20250320203513,"
            "2025-03-20T20:35:13,03/20/2025,"
            "Senate,Final,Unknown,sen_unknown_1,Nay\n",
        )
        err = StringIO()
        _load("91st_2025-2026", tmp_path, stderr=err)
        assert Vote.objects.count() == 1
        assert "Skipped" in err.getvalue()


# -- Integration tests: load_alec -----------------


class TestLoadAlec:
    """Integration tests for load_alec."""

    def test_load_basic(self, alec_csv):
        call_command(
            "load_alec",
            "--csv",
            str(alec_csv),
            stdout=StringIO(),
        )
        assert ALECModelBill.objects.count() == 3

    def test_idempotent(self, alec_csv):
        call_command(
            "load_alec",
            "--csv",
            str(alec_csv),
            stdout=StringIO(),
        )
        call_command(
            "load_alec",
            "--csv",
            str(alec_csv),
            stdout=StringIO(),
        )
        assert ALECModelBill.objects.count() == 3

    def test_dry_run(self, alec_csv):
        out = StringIO()
        call_command(
            "load_alec",
            "--csv",
            str(alec_csv),
            "--dry-run",
            stdout=out,
        )
        assert "Dry run" in out.getvalue()
        assert ALECModelBill.objects.count() == 0

    def test_fields(self, alec_csv):
        call_command(
            "load_alec",
            "--csv",
            str(alec_csv),
            stdout=StringIO(),
        )
        bill = ALECModelBill.objects.get(title="Castle Doctrine Act")
        assert bill.category == "Criminal Justice"
        assert bill.task_force == "Public Safety"

    def test_missing_csv_raises(self, tmp_path):
        p = str(tmp_path / "nonexistent.csv")
        with pytest.raises(CommandError, match="ALEC CSV not found"):
            call_command(
                "load_alec",
                "--csv",
                p,
                stdout=StringIO(),
            )


# -- Integration tests: load_all -------------------


class TestLoadAll:
    """Integration tests for load_all."""

    def test_discovers_and_loads(self, tmp_path):
        root = tmp_path / "data" / "kansas"
        for name, yr in [("84th_2011-2012", 2011), ("91st_2025-2026", 2025)]:
            d = root / name
            _mini_leg(d, name)
            _write_csv(
                d / f"{name}_rollcalls.csv",
                f"{_RC_HDR}\n"
                f"session,SB 1,Test,je_{yr}0101120000,,"
                f"{yr}-01-01T12:00:00,01/01/{yr},"
                "Senate,Final,Final,Passed,Test,,,"
                "1,0,0,0,0,1,True\n",
            )
            _write_csv(
                d / f"{name}_votes.csv",
                f"{_VOTES_HDR}\n"
                f"session,SB 1,Test,je_{yr}0101120000,"
                f"{yr}-01-01T12:00:00,01/01/{yr},"
                "Senate,Final,Smith,sen_smith_john_1,Yea\n",
            )

        # ALEC is skipped (default CSV doesn't exist in tmp_path)
        call_command(
            "load_all",
            "--data-root",
            str(root),
            stdout=StringIO(),
        )
        assert Session.objects.count() == 2
        assert Legislator.objects.count() == 2
