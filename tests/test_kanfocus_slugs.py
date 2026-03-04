"""
Tests for KanFocus slug generation and cross-reference matching.

Run: uv run pytest tests/test_kanfocus_slugs.py -v
"""

import csv
from pathlib import Path

import pytest

from tallgrass.kanfocus.slugs import (
    build_slug_lookup,
    generate_slug,
    load_existing_slugs,
    match_to_existing,
    normalize_name,
)

pytestmark = pytest.mark.scraper


# ── normalize_name() ──────────────────────────────────────────────────────


class TestNormalizeName:
    """Name normalization for slug generation."""

    def test_simple_name(self):
        assert normalize_name("Steve Abrams") == "Steve Abrams"

    def test_junior_suffix(self):
        assert normalize_name("Ramon Gonzalez Jr.") == "Ramon Gonzalez"

    def test_junior_no_dot(self):
        assert normalize_name("Ramon Gonzalez Jr") == "Ramon Gonzalez"

    def test_senior_suffix(self):
        assert normalize_name("John Smith Sr.") == "John Smith"

    def test_iii_suffix(self):
        assert normalize_name("Robert Williams III") == "Robert Williams"

    def test_nickname_in_parens(self):
        assert normalize_name("Thomas C. (Tim) Owens") == "Thomas Owens"

    def test_middle_initial(self):
        assert normalize_name("Stephen R. Morris") == "Stephen Morris"

    def test_multiple_middle_initials(self):
        assert normalize_name("Louis E. Ruiz") == "Louis Ruiz"

    def test_no_change_for_clean_name(self):
        assert normalize_name("Laura Kelly") == "Laura Kelly"

    def test_multi_word_last_name(self):
        assert normalize_name("Mary Pilcher Cook") == "Mary Pilcher Cook"

    def test_hyphen_replaced_with_space(self):
        assert normalize_name("Oletha Faust-Goudeau") == "Oletha Faust Goudeau"

    def test_multiple_hyphens(self):
        assert normalize_name("Anna Jones-Smith-Brown") == "Anna Jones Smith Brown"


# ── generate_slug() ───────────────────────────────────────────────────────


class TestGenerateSlug:
    """Generate tallgrass-compatible slugs from KanFocus names."""

    def test_senate_slug(self):
        assert generate_slug("Steve Abrams", "S") == "sen_abrams_steve_1"

    def test_house_slug(self):
        assert generate_slug("Barbara Ballard", "H") == "rep_ballard_barbara_1"

    def test_multi_word_last_name(self):
        assert generate_slug("Mary Pilcher Cook", "H") == "rep_pilcher_cook_mary_1"

    def test_suffix_stripped(self):
        assert generate_slug("Ramon Gonzalez Jr.", "H") == "rep_gonzalez_ramon_1"

    def test_nickname_stripped(self):
        assert generate_slug("Thomas C. (Tim) Owens", "S") == "sen_owens_thomas_1"

    def test_middle_initial_stripped(self):
        assert generate_slug("Stephen R. Morris", "S") == "sen_morris_stephen_1"

    def test_hyphenated_name(self):
        assert generate_slug("Oletha Faust-Goudeau", "S") == "sen_faust_goudeau_oletha_1"

    def test_case_insensitive(self):
        slug = generate_slug("Steve Abrams", "S")
        assert slug == slug.lower()

    def test_three_word_name(self):
        assert generate_slug("Melody McCray Miller", "H") == "rep_mccray_miller_melody_1"

    def test_parenthetical_with_middle(self):
        """Robert (Bob) Montgomery → normalize → Robert Montgomery."""
        assert generate_slug("Robert (Bob) Montgomery", "H") == "rep_montgomery_robert_1"


# ── load_existing_slugs() ─────────────────────────────────────────────────


class TestLoadExistingSlugs:
    """Load name→slug mapping from existing CSV."""

    def test_nonexistent_csv_returns_empty(self, tmp_path: Path):
        result = load_existing_slugs(tmp_path, "nonexistent")
        assert result == {}

    def test_loads_from_csv(self, tmp_path: Path):
        csv_path = tmp_path / "test_legislators.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "full_name",
                    "slug",
                    "chamber",
                    "party",
                    "district",
                    "member_url",
                    "ocd_id",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "name": "Steve Abrams",
                    "full_name": "Steve Abrams",
                    "slug": "sen_abrams_steve_1",
                    "chamber": "Senate",
                    "party": "Republican",
                    "district": "32nd",
                    "member_url": "",
                    "ocd_id": "",
                }
            )

        result = load_existing_slugs(tmp_path, "test")
        assert result["steve abrams"] == "sen_abrams_steve_1"

    def test_case_insensitive_keys(self, tmp_path: Path):
        csv_path = tmp_path / "test_legislators.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "full_name",
                    "slug",
                    "chamber",
                    "party",
                    "district",
                    "member_url",
                    "ocd_id",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "name": "Laura Kelly",
                    "full_name": "Laura Kelly",
                    "slug": "sen_kelly_laura_1",
                    "chamber": "Senate",
                    "party": "Democrat",
                    "district": "18th",
                    "member_url": "",
                    "ocd_id": "",
                }
            )

        result = load_existing_slugs(tmp_path, "test")
        assert "laura kelly" in result

    def test_loads_both_kf_and_je_entries(self, tmp_path: Path):
        """Both KF-generated and JE entries are loaded from CSV."""
        csv_path = tmp_path / "test_legislators.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name", "full_name", "slug", "chamber",
                    "party", "district", "member_url", "ocd_id",
                ],
            )
            writer.writeheader()
            # KF-generated entry (no member_url)
            writer.writerow({
                "name": "Brad Barrett", "full_name": "Brad Barrett",
                "slug": "rep_barrett_brad_1", "chamber": "House",
                "party": "Republican", "district": "76th",
                "member_url": "", "ocd_id": "",
            })
            # JE entry (has member_url)
            writer.writerow({
                "name": "Barrett", "full_name": "Bradley Barrett",
                "slug": "rep_barrett_bradley_1", "chamber": "House",
                "party": "Republican", "district": "76",
                "member_url": "https://www.kslegislature.gov/li/b2025_26/members/rep_barrett_bradley_1/",
                "ocd_id": "ocd-person/abc123",
            })

        result = load_existing_slugs(tmp_path, "test")
        assert "brad barrett" in result
        assert "bradley barrett" in result
        assert result["bradley barrett"] == "rep_barrett_bradley_1"


# ── match_to_existing() ───────────────────────────────────────────────────


class TestMatchToExisting:
    """Cross-reference KanFocus names against existing slugs.

    Run: uv run pytest tests/test_kanfocus_slugs.py -k TestMatchToExisting -v
    """

    def test_direct_match(self):
        existing = {"steve abrams": "sen_abrams_steve_1"}
        result = match_to_existing("Steve Abrams", "S", "32nd", existing)
        assert result == "sen_abrams_steve_1"

    def test_no_match_returns_none(self):
        existing = {"steve abrams": "sen_abrams_steve_1"}
        result = match_to_existing("Unknown Person", "S", "1st", existing)
        assert result is None

    def test_empty_existing_returns_none(self):
        result = match_to_existing("Steve Abrams", "S", "32nd", {})
        assert result is None

    def test_normalized_match(self):
        """Name with middle initial matches normalized form."""
        existing = {"stephen morris": "sen_morris_stephen_1"}
        result = match_to_existing("Stephen R. Morris", "S", "39th", existing)
        assert result == "sen_morris_stephen_1"

    # -- Nickname matching --

    def test_nickname_brad_to_bradley(self):
        """KF 'Brad' matches JE 'Bradley'."""
        existing = {"bradley ralph": "rep_ralph_bradley_1"}
        result = match_to_existing("Brad Ralph", "H", "1st", existing)
        assert result == "rep_ralph_bradley_1"

    def test_nickname_bill_to_william(self):
        """KF 'Bill' matches JE 'William'."""
        existing = {"william feuerborn": "rep_feuerborn_william_1"}
        result = match_to_existing("Bill Feuerborn", "H", "10th", existing)
        assert result == "rep_feuerborn_william_1"

    def test_nickname_reverse_formal_to_nick(self):
        """JE 'Robert' matches KF 'Bob'."""
        existing = {"bob grant": "sen_grant_bob_1"}
        result = match_to_existing("Robert Grant", "S", "5th", existing)
        assert result == "sen_grant_bob_1"

    def test_nickname_mike_to_michael(self):
        existing = {"michael thompson": "rep_thompson_michael_1"}
        result = match_to_existing("Mike Thompson", "H", "1st", existing)
        assert result == "rep_thompson_michael_1"

    def test_nickname_no_false_match(self):
        """Nickname fallback doesn't match unrelated names."""
        existing = {"john smith": "sen_smith_john_1"}
        result = match_to_existing("Mike Thompson", "H", "1st", existing)
        assert result is None

    # -- Chamber validation --

    def test_chamber_mismatch_rejects_senate_slug_for_house(self):
        """Senate slug should not match for a House vote."""
        existing = {"mike thompson": "sen_thompson_mike_1"}
        result = match_to_existing("Mike Thompson", "H", "1st", existing)
        assert result is None

    def test_chamber_mismatch_rejects_house_slug_for_senate(self):
        existing = {"laura kelly": "rep_kelly_laura_1"}
        result = match_to_existing("Laura Kelly", "S", "18th", existing)
        assert result is None

    def test_chamber_match_allows_correct_prefix(self):
        existing = {"mike thompson": "rep_thompson_mike_1"}
        result = match_to_existing("Mike Thompson", "H", "1st", existing)
        assert result == "rep_thompson_mike_1"

    def test_chamber_validation_on_nickname_match(self):
        """Nickname match also enforces chamber compatibility."""
        existing = {"william feuerborn": "sen_feuerborn_william_1"}
        result = match_to_existing("Bill Feuerborn", "H", "10th", existing)
        assert result is None

    # -- Hyphen normalization --

    def test_hyphen_normalized_in_lookup(self):
        """Faust-Goudeau normalizes to 'faust goudeau' and matches."""
        existing = {"oletha faust goudeau": "sen_faust_goudeau_oletha_1"}
        result = match_to_existing("Oletha Faust-Goudeau", "S", "29th", existing)
        assert result == "sen_faust_goudeau_oletha_1"


# ── build_slug_lookup() ───────────────────────────────────────────────────


class TestBuildSlugLookup:
    """Get slug for a legislator, preferring existing matches."""

    def test_uses_existing_when_available(self):
        existing = {"steve abrams": "sen_abrams_steve_1"}
        result = build_slug_lookup("Steve Abrams", "S", "32nd", existing)
        assert result == "sen_abrams_steve_1"

    def test_generates_fresh_when_no_match(self):
        result = build_slug_lookup("New Person", "H", "1st", {})
        assert result == "rep_person_new_1"

    def test_prefers_existing_over_generated(self):
        """Even if generated slug would differ, prefer existing."""
        existing = {"steve abrams": "sen_abrams_stephen_1"}  # different first name in slug
        result = build_slug_lookup("Steve Abrams", "S", "32nd", existing)
        assert result == "sen_abrams_stephen_1"
