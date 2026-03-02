"""Tests for legislator profile data logic.

Validates target selection, scorecard building, bill-type breakdown, defection
analysis, voting neighbors, and surprising vote filtering.

Usage:
    uv run pytest tests/test_profiles.py -v
"""

from __future__ import annotations

import polars as pl
import pytest
from analysis.profiles_data import (
    MAX_PROFILE_TARGETS,
    ProfileTarget,
    build_full_voting_record,
    build_scorecard,
    compute_bill_type_breakdown,
    find_defection_bills,
    find_legislator_surprising_votes,
    find_voting_neighbors,
    gather_profile_targets,
    resolve_names,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def house_leg_df() -> pl.DataFrame:
    """Minimal legislator DataFrame for the house with 10 legislators.

    Includes all columns needed by synthesis_detect and profiles_data.
    Party split: 7 Republican, 3 Democrat.
    """
    return pl.DataFrame(
        {
            "legislator_slug": [f"rep_{chr(97 + i)}" for i in range(10)],
            "full_name": [
                "Alice Adams",
                "Bob Baker",
                "Carol Clark",
                "Dave Davis",
                "Eve Evans",
                "Frank Fisher",
                "Grace Green",
                "Hank Hill",
                "Iris Irving",
                "Jack Jones",
            ],
            "party": ["Republican"] * 7 + ["Democrat"] * 3,
            "district": [str(i + 1) for i in range(10)],
            "chamber": ["house"] * 10,
            "xi_mean": [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -2.0, -2.5, -3.0],
            "xi_sd": [0.2] * 10,
            "unity_score": [0.98, 0.95, 0.92, 0.88, 0.85, 0.80, 0.70, 0.95, 0.92, 0.98],
            "loyalty_rate": [0.95, 0.90, 0.88, 0.82, 0.78, 0.75, 0.60, 0.90, 0.85, 0.95],
            "maverick_rate": [0.02, 0.05, 0.08, 0.12, 0.15, 0.20, 0.30, 0.05, 0.08, 0.02],
            "weighted_maverick": [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.25, 0.03, 0.05, 0.01],
            "betweenness": [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.15, 0.03, 0.02, 0.01],
            "eigenvector": [0.1, 0.1, 0.1, 0.1, 0.1, 0.12, 0.15, 0.1, 0.1, 0.1],
            "accuracy": [0.95, 0.94, 0.93, 0.90, 0.88, 0.85, 0.78, 0.94, 0.92, 0.96],
            "n_defections": [1, 3, 5, 8, 10, 14, 20, 3, 5, 1],
            "loyalty_zscore": [1.0, 0.5, 0.3, -0.2, -0.5, -0.8, -1.5, 0.5, 0.3, 1.0],
            "PC1": [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -2.0, -2.5, -3.0],
            "PC2": [0.1, 0.2, 0.1, -0.1, 0.0, 0.3, -0.2, 0.1, 0.0, -0.1],
            "xi_mean_percentile": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1, 0.0],
            "betweenness_percentile": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 0.3, 0.2, 0.1],
        }
    )


@pytest.fixture
def senate_leg_df() -> pl.DataFrame:
    """Minimal senate DataFrame with 6 legislators."""
    return pl.DataFrame(
        {
            "legislator_slug": [f"sen_{chr(97 + i)}" for i in range(6)],
            "full_name": [
                "Sam Smith",
                "Tom Turner",
                "Uma Upton",
                "Vera Vance",
                "Will Walker",
                "Xena Xavier",
            ],
            "party": ["Republican"] * 4 + ["Democrat"] * 2,
            "district": [str(i + 1) for i in range(6)],
            "chamber": ["senate"] * 6,
            "xi_mean": [3.5, 2.0, 1.0, 0.5, -2.0, -3.0],
            "xi_sd": [0.3] * 6,
            "unity_score": [0.98, 0.90, 0.75, 0.82, 0.95, 0.98],
            "loyalty_rate": [0.95, 0.85, 0.60, 0.78, 0.90, 0.95],
            "maverick_rate": [0.02, 0.10, 0.25, 0.18, 0.05, 0.02],
            "weighted_maverick": [0.01, 0.07, 0.20, 0.12, 0.03, 0.01],
            "betweenness": [0.01, 0.03, 0.12, 0.06, 0.02, 0.01],
            "eigenvector": [0.1, 0.1, 0.15, 0.12, 0.1, 0.1],
            "accuracy": [0.96, 0.91, 0.80, 0.87, 0.93, 0.97],
            "n_defections": [1, 6, 15, 10, 3, 1],
            "loyalty_zscore": [1.0, 0.3, -1.2, -0.5, 0.5, 1.0],
            "PC1": [3.5, 2.0, 1.0, 0.5, -2.0, -3.0],
            "PC2": [0.0, 0.1, -0.3, 0.2, 0.0, -0.1],
            "xi_mean_percentile": [1.0, 0.8, 0.6, 0.4, 0.17, 0.0],
            "betweenness_percentile": [0.17, 0.33, 0.83, 0.67, 0.17, 0.0],
        }
    )


@pytest.fixture
def leg_dfs(house_leg_df, senate_leg_df) -> dict[str, pl.DataFrame]:
    """Combined leg_dfs dict for both chambers."""
    return {"house": house_leg_df, "senate": senate_leg_df}


@pytest.fixture
def bill_params() -> pl.DataFrame:
    """Synthetic IRT bill parameters with varying discrimination."""
    return pl.DataFrame(
        {
            "vote_id": [f"v{i}" for i in range(10)],
            "beta_mean": [2.0, 1.8, 1.6, 0.8, 0.3, 0.2, 0.1, -0.4, -1.7, -2.1],
            "alpha_mean": [0.5] * 10,
            "bill_number": [f"HB {i}" for i in range(10)],
            "short_title": [f"Bill {i}" for i in range(10)],
            "motion": ["Final Action"] * 10,
        }
    )


@pytest.fixture
def votes_long() -> pl.DataFrame:
    """Synthetic long-form votes: 5 legislators x 10 votes.

    rep_a (R) votes with party on everything.
    rep_g (R) defects frequently — the maverick.
    """
    rows = []
    for vote_idx in range(10):
        vote_id = f"v{vote_idx}"
        # Republicans mostly vote Yea
        for slug in ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e"]:
            rows.append(
                {"legislator_slug": slug, "vote_id": vote_id, "vote_binary": 1, "chamber": "house"}
            )
        # rep_g defects on half the votes
        rows.append(
            {
                "legislator_slug": "rep_g",
                "vote_id": vote_id,
                "vote_binary": 0 if vote_idx < 5 else 1,
                "chamber": "house",
            }
        )
        # Democrats mostly vote Nay
        for slug in ["rep_h", "rep_i", "rep_j"]:
            rows.append(
                {"legislator_slug": slug, "vote_id": vote_id, "vote_binary": 0, "chamber": "house"}
            )
    return pl.DataFrame(rows)


@pytest.fixture
def rollcalls() -> pl.DataFrame:
    """Synthetic rollcalls with bill metadata."""
    return pl.DataFrame(
        {
            "vote_id": [f"v{i}" for i in range(10)],
            "bill_number": [f"HB {i}" for i in range(10)],
            "short_title": [f"Bill about topic {i}" for i in range(10)],
            "motion": ["Final Action"] * 10,
            "result": ["Passed"] * 10,
        }
    )


@pytest.fixture
def surprising_votes_df() -> pl.DataFrame:
    """Synthetic surprising votes from prediction phase."""
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_g", "rep_g", "rep_g", "rep_a", "rep_b"],
            "full_name": ["Grace Green", "Grace Green", "Grace Green", "Alice Adams", "Bob Baker"],
            "party": ["Republican"] * 5,
            "vote_id": ["v1", "v2", "v3", "v4", "v5"],
            "bill_number": ["HB 1", "HB 2", "HB 3", "HB 4", "HB 5"],
            "motion": ["Final Action"] * 5,
            "actual": [0, 0, 1, 1, 0],
            "predicted": [1, 1, 0, 0, 1],
            "y_prob": [0.92, 0.88, 0.85, 0.78, 0.72],
            "confidence_error": [0.92, 0.88, 0.85, 0.78, 0.72],
        }
    )


# ── Tests: Gather Profile Targets ────────────────────────────────────────────


class TestGatherProfileTargets:
    """Tests for the gather_profile_targets() function."""

    def test_detects_from_synthesis(self, leg_dfs):
        """Should detect at least one target via detect_all()."""
        targets = gather_profile_targets(leg_dfs)
        assert len(targets) >= 1
        assert all(isinstance(t, ProfileTarget) for t in targets)

    def test_extra_slugs_added(self, leg_dfs):
        """User-specified slugs should appear with role='Requested'."""
        targets = gather_profile_targets(leg_dfs, extra_slugs=["rep_a"])
        slugs = [t.slug for t in targets]
        assert "rep_a" in slugs
        rep_a = next(t for t in targets if t.slug == "rep_a")
        assert rep_a.role == "Requested"

    def test_deduplicates(self, leg_dfs):
        """Same slug detected + requested should appear only once."""
        # First find what gets detected
        targets_auto = gather_profile_targets(leg_dfs)
        if not targets_auto:
            pytest.skip("No auto-detected targets")
        auto_slug = targets_auto[0].slug
        # Add same slug as extra — should not duplicate
        targets = gather_profile_targets(leg_dfs, extra_slugs=[auto_slug])
        slug_counts = [t.slug for t in targets].count(auto_slug)
        assert slug_counts == 1

    def test_max_eight_targets(self, leg_dfs):
        """Never returns more than MAX_PROFILE_TARGETS."""
        # Request many extra slugs
        all_slugs = []
        for df in leg_dfs.values():
            all_slugs.extend(df["legislator_slug"].to_list())
        targets = gather_profile_targets(leg_dfs, extra_slugs=all_slugs)
        assert len(targets) <= MAX_PROFILE_TARGETS


# ── Tests: Build Scorecard ───────────────────────────────────────────────────


class TestBuildScorecard:
    """Tests for the build_scorecard() function."""

    def test_returns_all_metrics(self, house_leg_df):
        """Scorecard dict should have keys for each available metric."""
        result = build_scorecard(house_leg_df, "rep_a")
        assert result is not None
        assert "xi_mean_percentile" in result
        assert "unity_score" in result
        assert "accuracy" in result

    def test_party_averages_included(self, house_leg_df):
        """Scorecard should include _party_avg keys."""
        result = build_scorecard(house_leg_df, "rep_a")
        assert result is not None
        assert "xi_mean_percentile_party_avg" in result
        assert "unity_score_party_avg" in result

    def test_returns_none_for_missing_slug(self, house_leg_df):
        """Should return None for a slug not in the DataFrame."""
        result = build_scorecard(house_leg_df, "rep_nonexistent")
        assert result is None


# ── Tests: Bill Type Breakdown ───────────────────────────────────────────────


class TestComputeBillTypeBreakdown:
    """Tests for the compute_bill_type_breakdown() function."""

    def test_correct_tier_counts(self, bill_params, votes_long):
        """High/low disc bill counts should match classification thresholds."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = compute_bill_type_breakdown(
            "rep_a", bill_params, votes_long, "Republican", party_slugs
        )
        # High disc: |beta_mean| > 1.5 → v0(2.0), v1(1.8), v2(1.6), v8(1.7), v9(2.1) = 5
        # Low disc: |beta_mean| < 0.5 → v4(0.3), v5(0.2), v6(0.1), v7(0.4) = 4
        assert result is not None
        assert result.high_disc_n == 5
        assert result.low_disc_n == 4

    def test_yea_rates_computed(self, bill_params, votes_long):
        """Yea rates should match hand-calculated values."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = compute_bill_type_breakdown(
            "rep_a", bill_params, votes_long, "Republican", party_slugs
        )
        assert result is not None
        # rep_a votes 1 (Yea) on all votes → both rates should be 1.0
        assert result.high_disc_yea_rate == pytest.approx(1.0)
        assert result.low_disc_yea_rate == pytest.approx(1.0)

    def test_returns_none_for_few_bills(self):
        """Should return None when fewer than MIN_BILLS_PER_TIER bills."""
        # Only 2 bills total, both high disc
        bp = pl.DataFrame(
            {
                "vote_id": ["v0", "v1"],
                "beta_mean": [2.0, 1.8],
            }
        )
        vl = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_a"],
                "vote_id": ["v0", "v1"],
                "vote_binary": [1, 1],
                "chamber": ["house", "house"],
            }
        )
        result = compute_bill_type_breakdown("rep_a", bp, vl, "Republican", ["rep_a"])
        assert result is None

    def test_party_averages(self, bill_params, votes_long):
        """Party average rates should differ from the maverick."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result_maverick = compute_bill_type_breakdown(
            "rep_g", bill_params, votes_long, "Republican", party_slugs
        )
        result_loyal = compute_bill_type_breakdown(
            "rep_a", bill_params, votes_long, "Republican", party_slugs
        )
        assert result_maverick is not None
        assert result_loyal is not None
        # rep_g has different Yea rate than rep_a, so party avg should be between
        assert result_maverick.party_high_disc_yea_rate == result_loyal.party_high_disc_yea_rate


# ── Tests: Find Defection Bills ──────────────────────────────────────────────


class TestFindDefectionBills:
    """Tests for the find_defection_bills() function."""

    def test_finds_defections(self, votes_long, rollcalls):
        """Should return bills where the legislator disagreed with party."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = find_defection_bills("rep_g", votes_long, rollcalls, "Republican", party_slugs)
        # rep_g votes 0 on v0-v4 while party majority votes 1
        assert result.height > 0
        assert "legislator_vote" in result.columns
        assert "party_majority_vote" in result.columns

    def test_sorted_by_closeness(self, votes_long, rollcalls):
        """Defections should be sorted with close votes first."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = find_defection_bills("rep_g", votes_long, rollcalls, "Republican", party_slugs)
        if result.height > 1:
            # party_yea_pct closer to 50% should come first
            pcts = result["party_yea_pct"].to_list()
            margins = [abs(p - 50.0) for p in pcts]
            assert margins == sorted(margins)

    def test_respects_n_limit(self, votes_long, rollcalls):
        """Should return at most n rows."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = find_defection_bills(
            "rep_g", votes_long, rollcalls, "Republican", party_slugs, n=2
        )
        assert result.height <= 2

    def test_empty_for_loyal_legislator(self, votes_long, rollcalls):
        """Should return empty DataFrame for a loyal party-line voter."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = find_defection_bills("rep_a", votes_long, rollcalls, "Republican", party_slugs)
        assert result.height == 0

    def test_handles_rollcalls_missing_columns(self, votes_long):
        """Should not crash when rollcalls lacks metadata columns."""
        # Rollcalls with only vote_id — no bill_number, short_title, or motion
        minimal_rc = pl.DataFrame(
            {
                "vote_id": [f"v{i}" for i in range(10)],
            }
        )
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = find_defection_bills("rep_g", votes_long, minimal_rc, "Republican", party_slugs)
        assert result.height > 0
        assert "bill_number" in result.columns
        assert "short_title" in result.columns
        assert "motion" in result.columns


# ── Tests: Find Voting Neighbors ─────────────────────────────────────────────


class TestFindVotingNeighbors:
    """Tests for the find_voting_neighbors() function."""

    def test_closest_correct(self, votes_long, house_leg_df):
        """Most similar should be same-voting legislators."""
        result = find_voting_neighbors("rep_a", votes_long, house_leg_df)
        assert result is not None
        closest = result["closest"]
        assert len(closest) > 0
        # rep_a always votes 1, so other always-1 voters should be closest
        closest_slugs = [c["slug"] for c in closest]
        assert "rep_b" in closest_slugs  # also always Yea

    def test_most_different_correct(self, votes_long, house_leg_df):
        """Most different should be opposite-voting legislators."""
        result = find_voting_neighbors("rep_a", votes_long, house_leg_df)
        assert result is not None
        most_diff = result["most_different"]
        assert len(most_diff) > 0
        # Democrats always vote 0, so they should be most different from rep_a (always 1)
        diff_slugs = [c["slug"] for c in most_diff]
        assert any(
            s.startswith("rep_h") or s.startswith("rep_i") or s.startswith("rep_j")
            for s in diff_slugs
        )

    def test_excludes_self(self, votes_long, house_leg_df):
        """Target slug should not appear in results."""
        result = find_voting_neighbors("rep_a", votes_long, house_leg_df)
        assert result is not None
        all_slugs = [c["slug"] for c in result["closest"]] + [
            c["slug"] for c in result["most_different"]
        ]
        assert "rep_a" not in all_slugs

    def test_agreement_range(self, votes_long, house_leg_df):
        """All agreement values should be between 0 and 1."""
        result = find_voting_neighbors("rep_a", votes_long, house_leg_df)
        assert result is not None
        for entry in result["closest"] + result["most_different"]:
            assert 0.0 <= entry["agreement"] <= 1.0


# ── Tests: Find Legislator Surprising Votes ──────────────────────────────────


class TestFindLegislatorSurprisingVotes:
    """Tests for the find_legislator_surprising_votes() function."""

    def test_filters_to_slug(self, surprising_votes_df):
        """Should only return rows for the target legislator."""
        result = find_legislator_surprising_votes("rep_g", surprising_votes_df)
        assert result is not None
        assert result.height == 3
        assert (result["legislator_slug"] == "rep_g").all()

    def test_respects_n_limit(self, surprising_votes_df):
        """Should return at most n rows."""
        result = find_legislator_surprising_votes("rep_g", surprising_votes_df, n=2)
        assert result is not None
        assert result.height == 2

    def test_returns_none_for_no_data(self, surprising_votes_df):
        """Should return None for a slug not in the data."""
        result = find_legislator_surprising_votes("rep_nonexistent", surprising_votes_df)
        assert result is None

    def test_returns_none_for_none_input(self):
        """Should return None when surprising_votes_df is None."""
        result = find_legislator_surprising_votes("rep_a", None)
        assert result is None


# ── Fixtures: Name Resolution ───────────────────────────────────────────────


def _stub_columns(n: int) -> dict:
    """Return dummy metric columns for n legislators."""
    return {
        "xi_mean": [1.0] * n,
        "xi_sd": [0.2] * n,
        "unity_score": [0.9] * n,
        "loyalty_rate": [0.9] * n,
        "maverick_rate": [0.1] * n,
        "weighted_maverick": [0.05] * n,
        "betweenness": [0.03] * n,
        "eigenvector": [0.1] * n,
        "accuracy": [0.9] * n,
        "n_defections": [2] * n,
        "loyalty_zscore": [0.5] * n,
        "PC1": [1.0] * n,
        "PC2": [0.1] * n,
        "xi_mean_percentile": [0.5] * n,
        "betweenness_percentile": [0.3] * n,
    }


@pytest.fixture
def leg_dfs_with_dupes() -> dict[str, pl.DataFrame]:
    """leg_dfs with known duplicate last names for name-resolution tests.

    House: Alice Smith, Bob Smith, Carol Jones, Dave Carpenter, Eve Carpenter.
    Senate: Frank Jones, Gina Unique.
    """
    house_data = {
        "legislator_slug": [
            "rep_smith_alice_1",
            "rep_smith_bob_1",
            "rep_jones_carol_1",
            "rep_carpenter_dave_1",
            "rep_carpenter_eve_1",
        ],
        "full_name": [
            "Alice Smith",
            "Bob Smith",
            "Carol Jones",
            "Dave Carpenter",
            "Eve Carpenter",
        ],
        "party": ["Republican", "Republican", "Democrat", "Republican", "Democrat"],
        "district": ["1", "2", "3", "4", "5"],
        "chamber": ["house"] * 5,
        **_stub_columns(5),
    }
    senate_data = {
        "legislator_slug": ["sen_jones_frank_1", "sen_unique_gina_1"],
        "full_name": ["Frank Jones", "Gina Unique"],
        "party": ["Republican", "Democrat"],
        "district": ["10", "11"],
        "chamber": ["senate"] * 2,
        **_stub_columns(2),
    }
    return {
        "house": pl.DataFrame(house_data),
        "senate": pl.DataFrame(senate_data),
    }


# ── Tests: Resolve Names ───────────────────────────────────────────────────


class TestResolveNames:
    """Tests for the resolve_names() function."""

    def test_exact_full_name_match(self, leg_dfs_with_dupes):
        """Exact full name (case-insensitive) returns ok with one match."""
        result = resolve_names(["Alice Smith"], leg_dfs_with_dupes)
        assert len(result) == 1
        assert result[0].status == "ok"
        assert result[0].matches[0]["slug"] == "rep_smith_alice_1"

    def test_exact_full_name_case_insensitive(self, leg_dfs_with_dupes):
        """Case should not matter for full-name matching."""
        result = resolve_names(["alice smith"], leg_dfs_with_dupes)
        assert len(result) == 1
        assert result[0].status == "ok"
        assert result[0].matches[0]["full_name"] == "Alice Smith"

    def test_last_name_unique_match(self, leg_dfs_with_dupes):
        """A unique last name returns ok."""
        result = resolve_names(["Unique"], leg_dfs_with_dupes)
        assert len(result) == 1
        assert result[0].status == "ok"
        assert result[0].matches[0]["slug"] == "sen_unique_gina_1"

    def test_last_name_ambiguous(self, leg_dfs_with_dupes):
        """Last name with multiple matches returns ambiguous with all matches."""
        result = resolve_names(["Smith"], leg_dfs_with_dupes)
        assert len(result) == 1
        assert result[0].status == "ambiguous"
        assert len(result[0].matches) == 2
        slugs = {m["slug"] for m in result[0].matches}
        assert slugs == {"rep_smith_alice_1", "rep_smith_bob_1"}

    def test_first_name_disambiguates(self, leg_dfs_with_dupes):
        """Full name disambiguates among same-last-name legislators."""
        result = resolve_names(["Dave Carpenter"], leg_dfs_with_dupes)
        assert len(result) == 1
        assert result[0].status == "ok"
        assert result[0].matches[0]["slug"] == "rep_carpenter_dave_1"

    def test_no_match(self, leg_dfs_with_dupes):
        """Non-existent name returns no_match."""
        result = resolve_names(["Nobody"], leg_dfs_with_dupes)
        assert len(result) == 1
        assert result[0].status == "no_match"
        assert result[0].matches == []

    def test_cross_chamber_same_last_name(self, leg_dfs_with_dupes):
        """Last name present in both chambers returns ambiguous."""
        result = resolve_names(["Jones"], leg_dfs_with_dupes)
        assert len(result) == 1
        assert result[0].status == "ambiguous"
        assert len(result[0].matches) == 2
        chambers = {m["chamber"] for m in result[0].matches}
        assert chambers == {"house", "senate"}

    def test_cross_chamber_disambiguated_by_full_name(self, leg_dfs_with_dupes):
        """Full name disambiguates across chambers."""
        result = resolve_names(["Frank Jones"], leg_dfs_with_dupes)
        assert len(result) == 1
        assert result[0].status == "ok"
        assert result[0].matches[0]["chamber"] == "senate"

    def test_multiple_queries(self, leg_dfs_with_dupes):
        """Multiple queries return one NameMatch each."""
        result = resolve_names(["Alice Smith", "Nobody", "Unique"], leg_dfs_with_dupes)
        assert len(result) == 3
        assert result[0].status == "ok"
        assert result[1].status == "no_match"
        assert result[2].status == "ok"

    def test_empty_list_returns_empty(self, leg_dfs_with_dupes):
        """Empty input returns empty output."""
        result = resolve_names([], leg_dfs_with_dupes)
        assert result == []

    def test_preserves_original_query(self, leg_dfs_with_dupes):
        """NameMatch.query should preserve the exact original input string."""
        result = resolve_names(["  Alice Smith  "], leg_dfs_with_dupes)
        assert result[0].query == "  Alice Smith  "

    def test_resolved_slugs_work_with_gather(self, leg_dfs_with_dupes):
        """Resolved slugs should be accepted by gather_profile_targets()."""
        matches = resolve_names(["Gina Unique"], leg_dfs_with_dupes)
        slugs = [m["slug"] for nm in matches for m in nm.matches]
        targets = gather_profile_targets(leg_dfs_with_dupes, extra_slugs=slugs)
        target_slugs = [t.slug for t in targets]
        assert "sen_unique_gina_1" in target_slugs


# ── Tests: Build Full Voting Record ─────────────────────────────────────────


class TestBuildFullVotingRecord:
    """Tests for the build_full_voting_record() function."""

    def test_returns_all_votes(self, votes_long, rollcalls):
        """Should return one row per vote cast by the legislator."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = build_full_voting_record(
            "rep_a", votes_long, rollcalls, "Republican", party_slugs
        )
        assert result.height == 10  # rep_a votes on all 10 rollcalls

    def test_correct_columns(self, votes_long, rollcalls):
        """Should have the expected output columns."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = build_full_voting_record(
            "rep_a", votes_long, rollcalls, "Republican", party_slugs
        )
        expected_cols = {
            "date", "bill_number", "short_title", "motion",
            "vote", "party_majority", "with_party", "passed",
        }
        assert set(result.columns) == expected_cols

    def test_with_party_flag_correct(self, votes_long, rollcalls):
        """Loyal voter should have with_party=True on all votes."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = build_full_voting_record(
            "rep_a", votes_long, rollcalls, "Republican", party_slugs
        )
        # rep_a always votes Yea, party majority is Yea (5/6 R voters = Yea)
        assert result["with_party"].all()

    def test_maverick_has_defections(self, votes_long, rollcalls):
        """Maverick (rep_g) should have some with_party=False rows."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = build_full_voting_record(
            "rep_g", votes_long, rollcalls, "Republican", party_slugs
        )
        defections = result.filter(~pl.col("with_party"))
        assert defections.height > 0

    def test_empty_for_unknown_slug(self, votes_long, rollcalls):
        """Should return empty DataFrame for nonexistent slug."""
        result = build_full_voting_record(
            "rep_nonexistent", votes_long, rollcalls, "Republican", ["rep_a"]
        )
        assert result.height == 0
        assert "vote" in result.columns  # schema preserved

    def test_vote_labels_are_yea_nay(self, votes_long, rollcalls):
        """Vote column should contain only 'Yea' and 'Nay'."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = build_full_voting_record(
            "rep_g", votes_long, rollcalls, "Republican", party_slugs
        )
        unique_votes = set(result["vote"].to_list())
        assert unique_votes <= {"Yea", "Nay"}

    def test_bill_metadata_joined(self, votes_long, rollcalls):
        """Should join bill metadata from rollcalls."""
        party_slugs = ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "rep_g"]
        result = build_full_voting_record(
            "rep_a", votes_long, rollcalls, "Republican", party_slugs
        )
        # All bill_numbers should be "HB N" format from our fixture
        bills = result["bill_number"].to_list()
        assert all(b.startswith("HB ") for b in bills)

    def test_sorted_by_date_descending(self):
        """Result should be sorted by date descending (most recent first)."""
        # Use vote_ids with the actual format: je_YYYYMMDDHHmmss
        vl = pl.DataFrame(
            {
                "legislator_slug": ["rep_a"] * 3,
                "vote_id": [
                    "je_20250115100000",
                    "je_20250320140000",
                    "je_20250210120000",
                ],
                "vote_binary": [1, 1, 0],
                "chamber": ["house"] * 3,
            }
        )
        rc = pl.DataFrame(
            {
                "vote_id": [
                    "je_20250115100000",
                    "je_20250320140000",
                    "je_20250210120000",
                ],
                "bill_number": ["HB 1", "HB 2", "HB 3"],
                "short_title": ["A", "B", "C"],
                "motion": ["Final Action"] * 3,
            }
        )
        result = build_full_voting_record(
            "rep_a", vl, rc, "Republican", ["rep_a"]
        )
        dates = result["date"].to_list()
        assert dates == ["2025-03-20", "2025-02-10", "2025-01-15"]
