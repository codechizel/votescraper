"""Tests for dynamic ideal point estimation (Phase 16).

Run:
    uv run pytest tests/test_dynamic_irt.py -v
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pymc as pm
import pytest
from analysis.dynamic_irt import (
    DEFAULT_TAU_SIGMA,
    SMALL_CHAMBER_TAU_SIGMA,
    SMALL_CHAMBER_THRESHOLD,
    build_dynamic_irt_graph,
)
from analysis.dynamic_irt_data import (
    BIENNIUM_LABELS,
    BIENNIUM_SESSIONS,
    LABEL_TO_SESSION,
    MIN_BRIDGE_OVERLAP,
    SESSION_TO_LABEL,
    build_global_roster,
    compute_adjacent_bridges,
    compute_bridge_coverage,
    normalize_name,
    prepare_emirt_csv,
    stack_bienniums,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_legislators(
    names: list[str],
    *,
    prefix: str = "rep",
    party: str = "Republican",
    chamber: str = "House",
    start_district: int = 1,
) -> pl.DataFrame:
    """Build a legislators DataFrame matching the CSV schema."""
    return pl.DataFrame(
        {
            "legislator_slug": [f"{prefix}_{n.split()[-1].lower()}" for n in names],
            "full_name": names,
            "party": [party] * len(names),
            "chamber": [chamber] * len(names),
            "district": list(range(start_district, start_district + len(names))),
        }
    )


def _make_vote_matrix(
    names: list[str],
    n_votes: int = 10,
    *,
    prefix: str = "rep",
    seed: int = 42,
) -> pl.DataFrame:
    """Build a vote matrix DataFrame (legislator_slug x vote_id, binary)."""
    rng = np.random.default_rng(seed)
    slugs = [f"{prefix}_{n.split()[-1].lower()}" for n in names]
    vote_ids = [f"v_{i:04d}" for i in range(n_votes)]
    data: dict = {"legislator_slug": slugs}
    for vid in vote_ids:
        data[vid] = rng.integers(0, 2, size=len(names)).tolist()
    return pl.DataFrame(data)


def _make_irt_data(
    names: list[str],
    n_votes: int = 10,
    *,
    prefix: str = "rep",
    seed: int = 42,
) -> dict:
    """Build an IRT data dict matching prepare_irt_data() output."""
    rng = np.random.default_rng(seed)
    n_leg = len(names)
    slugs = [f"{prefix}_{n.split()[-1].lower()}" for n in names]
    vote_ids = [f"v_{i:04d}" for i in range(n_votes)]

    # Generate random observations (some missing)
    leg_idx_list = []
    vote_idx_list = []
    y_list = []
    for i in range(n_leg):
        for j in range(n_votes):
            if rng.random() > 0.1:  # 90% participation
                leg_idx_list.append(i)
                vote_idx_list.append(j)
                y_list.append(int(rng.integers(0, 2)))

    return {
        "leg_idx": np.array(leg_idx_list, dtype=np.int64),
        "vote_idx": np.array(vote_idx_list, dtype=np.int64),
        "y": np.array(y_list, dtype=np.int64),
        "n_legislators": n_leg,
        "n_votes": n_votes,
        "n_obs": len(y_list),
        "leg_slugs": slugs,
        "vote_ids": vote_ids,
    }


def _make_multi_biennium_data(
    n_shared: int = 10,
    n_unique_per: int = 3,
    n_bienniums: int = 3,
    n_votes: int = 15,
) -> tuple[dict[int, dict], dict[int, pl.DataFrame]]:
    """Create multi-biennium IRT data with shared and unique legislators."""
    shared_names = [f"Shared Member {i}" for i in range(n_shared)]
    all_irt: dict[int, dict] = {}
    all_legs: dict[int, pl.DataFrame] = {}

    for t in range(n_bienniums):
        unique_names = [f"Unique T{t} M{i}" for i in range(n_unique_per)]
        names = shared_names + unique_names
        all_irt[t] = _make_irt_data(names, n_votes, seed=42 + t)
        all_legs[t] = _make_legislators(names)

    return all_irt, all_legs


# ── Constants Tests ──────────────────────────────────────────────────────────


class TestConstants:
    """Tests for module constants."""

    def test_session_label_lengths_match(self) -> None:
        """Session and label lists must have same length."""
        assert len(BIENNIUM_SESSIONS) == len(BIENNIUM_LABELS)

    def test_session_to_label_round_trip(self) -> None:
        """SESSION_TO_LABEL and LABEL_TO_SESSION are inverses."""
        for sess, label in SESSION_TO_LABEL.items():
            assert LABEL_TO_SESSION[label] == sess

    def test_sessions_are_ordered(self) -> None:
        """Sessions should be chronologically ordered."""
        years = [int(s.split("-")[0]) for s in BIENNIUM_SESSIONS]
        assert years == sorted(years)

    def test_labels_are_ordered(self) -> None:
        """Labels should be ordinally ordered."""
        nums = [int(lb.rstrip("stndrdth")) for lb in BIENNIUM_LABELS]
        assert nums == sorted(nums)


# ── Normalize Name Tests ─────────────────────────────────────────────────────


class TestNormalizeName:
    """Tests for legislator name normalization."""

    def test_lowercase(self) -> None:
        assert normalize_name("John Smith") == "john smith"

    def test_strips_whitespace(self) -> None:
        assert normalize_name("  John Smith  ") == "john smith"

    def test_removes_leadership_suffix(self) -> None:
        assert normalize_name("Ty Masterson - President of the Senate") == "ty masterson"

    def test_no_suffix(self) -> None:
        assert normalize_name("John Alcala") == "john alcala"

    def test_hyphenated_name_not_stripped(self) -> None:
        """Hyphenated names (not leadership suffixes) should be preserved."""
        assert normalize_name("Mary Smith-Jones") == "mary smith-jones"


# ── Global Roster Tests ──────────────────────────────────────────────────────


class TestBuildGlobalRoster:
    """Tests for build_global_roster()."""

    def test_basic_roster(self) -> None:
        """Build a roster with legislators across 2 bienniums."""
        names_t0 = ["Alice Smith", "Bob Jones", "Carol White"]
        names_t1 = ["Alice Smith", "Bob Jones", "Dave Green"]
        legs = {
            0: _make_legislators(names_t0),
            1: _make_legislators(names_t1),
        }

        roster, name_to_global = build_global_roster(legs, "House")

        # 4 unique legislators
        assert roster.height == 4
        assert len(name_to_global) == 4

        # Alice and Bob served 2 periods
        alice_row = roster.filter(pl.col("name_norm") == "alice smith")
        assert alice_row["n_periods"][0] == 2

        # Carol served only period 0
        carol_row = roster.filter(pl.col("name_norm") == "carol white")
        assert carol_row["n_periods"][0] == 1
        assert carol_row["first_period"][0] == 0
        assert carol_row["last_period"][0] == 0

        # Dave served only period 1
        dave_row = roster.filter(pl.col("name_norm") == "dave green")
        assert dave_row["n_periods"][0] == 1
        assert dave_row["first_period"][0] == 1

    def test_empty_input(self) -> None:
        """Empty input should return empty roster."""
        roster, name_to_global = build_global_roster({}, "House")
        assert roster.height == 0
        assert len(name_to_global) == 0

    def test_single_biennium(self) -> None:
        """Single biennium roster."""
        names = ["Alice Smith", "Bob Jones"]
        legs = {0: _make_legislators(names)}
        roster, name_to_global = build_global_roster(legs, "House")
        assert roster.height == 2
        for row in roster.iter_rows(named=True):
            assert row["n_periods"] == 1
            assert row["first_period"] == 0
            assert row["last_period"] == 0

    def test_chamber_filtering(self) -> None:
        """Only include legislators from requested chamber."""
        house_names = ["Alice Smith"]
        senate_names = ["Bob Jones"]
        legs_combined = pl.concat(
            [
                _make_legislators(house_names, chamber="House"),
                _make_legislators(senate_names, prefix="sen", chamber="Senate"),
            ]
        )
        roster, _ = build_global_roster({0: legs_combined}, "House")
        assert roster.height == 1
        assert roster["name_norm"][0] == "alice smith"

    def test_leadership_suffix_dedup(self) -> None:
        """Leadership suffixes should not create duplicate entries."""
        legs = {
            0: _make_legislators(["Ty Masterson"]),
            1: _make_legislators(["Ty Masterson - President of the Senate"]),
        }
        roster, _ = build_global_roster(legs, "House")
        assert roster.height == 1
        assert roster["n_periods"][0] == 2

    def test_global_indices_sequential(self) -> None:
        """Global indices should be sequential starting from 0."""
        names = ["A", "B", "C"]
        legs = {0: _make_legislators(names)}
        roster, name_to_global = build_global_roster(legs, "House")
        indices = sorted(name_to_global.values())
        assert indices == [0, 1, 2]

    def test_party_tracking(self) -> None:
        """Party should reflect the legislator's party."""
        legs = {
            0: _make_legislators(["Alice Smith"], party="Democrat"),
            1: _make_legislators(["Alice Smith"], party="Democrat"),
        }
        roster, _ = build_global_roster(legs, "House")
        assert "Democrat" in roster["parties"][0]


# ── Stack Bienniums Tests ────────────────────────────────────────────────────


class TestStackBienniums:
    """Tests for stack_bienniums()."""

    def test_basic_stacking(self) -> None:
        """Stack 2 bienniums with shared legislators."""
        all_irt, all_legs = _make_multi_biennium_data(
            n_shared=5,
            n_unique_per=2,
            n_bienniums=2,
            n_votes=10,
        )

        _, name_to_global = build_global_roster(all_legs, "House")
        stacked = stack_bienniums("House", all_irt, name_to_global, all_legs)

        assert stacked["n_time"] == 2
        assert stacked["n_legislators"] == 9  # 5 shared + 2 unique_t0 + 2 unique_t1
        assert stacked["n_obs"] > 0
        assert len(stacked["y"]) == stacked["n_obs"]
        assert len(stacked["leg_global_idx"]) == stacked["n_obs"]
        assert len(stacked["bill_idx"]) == stacked["n_obs"]
        assert len(stacked["time_idx"]) == stacked["n_obs"]

    def test_bill_indices_globally_unique(self) -> None:
        """Bill indices should be globally unique across bienniums."""
        all_irt, all_legs = _make_multi_biennium_data(
            n_shared=5,
            n_unique_per=0,
            n_bienniums=2,
            n_votes=10,
        )
        _, name_to_global = build_global_roster(all_legs, "House")
        stacked = stack_bienniums("House", all_irt, name_to_global, all_legs)

        # Total bills = sum of per-biennium bills
        assert stacked["n_bills"] == 20  # 10 + 10

        # All bill indices should be in [0, n_bills)
        assert stacked["bill_idx"].min() >= 0
        assert stacked["bill_idx"].max() < stacked["n_bills"]

    def test_time_indices_correct(self) -> None:
        """Time indices should match biennium assignment."""
        all_irt, all_legs = _make_multi_biennium_data(
            n_shared=5,
            n_unique_per=0,
            n_bienniums=3,
            n_votes=5,
        )
        _, name_to_global = build_global_roster(all_legs, "House")
        stacked = stack_bienniums("House", all_irt, name_to_global, all_legs)

        assert set(stacked["time_idx"].tolist()).issubset({0, 1, 2})
        assert stacked["n_time"] == 3

    def test_leg_periods_populated(self) -> None:
        """leg_periods should correctly track which bienniums each legislator served."""
        all_irt, all_legs = _make_multi_biennium_data(
            n_shared=5,
            n_unique_per=2,
            n_bienniums=2,
            n_votes=10,
        )
        _, name_to_global = build_global_roster(all_legs, "House")
        stacked = stack_bienniums("House", all_irt, name_to_global, all_legs)

        # Shared legislators should appear in both periods
        for name in ["shared member 0", "shared member 1"]:
            gidx = name_to_global[name]
            assert len(stacked["leg_periods"][gidx]) == 2

    def test_party_idx_assigned(self) -> None:
        """Party indices should be assigned for all legislators."""
        all_irt, all_legs = _make_multi_biennium_data(
            n_shared=3,
            n_unique_per=1,
            n_bienniums=2,
            n_votes=5,
        )
        _, name_to_global = build_global_roster(all_legs, "House")
        stacked = stack_bienniums("House", all_irt, name_to_global, all_legs)

        assert len(stacked["party_idx"]) == stacked["n_legislators"]
        assert len(stacked["party_names"]) > 0

    def test_single_biennium_stacking(self) -> None:
        """Single biennium should produce valid stacked output."""
        names = ["Alice Smith", "Bob Jones"]
        all_irt = {0: _make_irt_data(names, 10)}
        all_legs = {0: _make_legislators(names)}
        _, name_to_global = build_global_roster(all_legs, "House")
        stacked = stack_bienniums("House", all_irt, name_to_global, all_legs)

        assert stacked["n_time"] == 1
        assert stacked["n_legislators"] == 2


# ── Bridge Coverage Tests ────────────────────────────────────────────────────


class TestBridgeCoverage:
    """Tests for bridge coverage analysis."""

    def test_full_coverage(self) -> None:
        """All legislators in both periods → 100% overlap."""
        leg_periods = [[0, 1], [0, 1], [0, 1]]
        df = compute_bridge_coverage(leg_periods, 2)
        assert df.height == 1
        assert df["overlap_pct"][0] == 100.0
        assert df["shared_count"][0] == 3

    def test_no_overlap(self) -> None:
        """No shared legislators → 0% overlap."""
        leg_periods = [[0], [0], [1], [1]]
        df = compute_bridge_coverage(leg_periods, 2)
        assert df["overlap_pct"][0] == 0.0
        assert df["shared_count"][0] == 0

    def test_partial_overlap(self) -> None:
        """Some shared legislators."""
        leg_periods = [[0, 1], [0], [1]]
        df = compute_bridge_coverage(leg_periods, 2)
        assert df["shared_count"][0] == 1
        # overlap = 1/1 (min of period counts: 1 for t=0 is 2, t=1 is 2 → min=2 → 50%)
        # Actually: total_a=2, total_b=2, denom=2, shared=1 → 50%
        assert df["overlap_pct"][0] == 50.0

    def test_three_periods(self) -> None:
        """Three periods should produce 3 pairs."""
        leg_periods = [[0, 1, 2], [0, 1], [1, 2]]
        df = compute_bridge_coverage(leg_periods, 3)
        assert df.height == 3  # (0,1), (0,2), (1,2)

    def test_custom_labels(self) -> None:
        """Custom labels should appear in output."""
        leg_periods = [[0, 1]]
        labels = ["84th", "85th"]
        df = compute_bridge_coverage(leg_periods, 2, labels=labels)
        assert df["label_a"][0] == "84th"
        assert df["label_b"][0] == "85th"


class TestAdjacentBridges:
    """Tests for adjacent bridge coverage."""

    def test_adjacent_only(self) -> None:
        """Only adjacent pairs should be computed."""
        leg_periods = [[0, 1, 2], [0, 1], [1, 2]]
        df = compute_adjacent_bridges(leg_periods, 3)
        assert df.height == 2  # 0→1 and 1→2

    def test_sufficient_flag(self) -> None:
        """Sufficient flag based on MIN_BRIDGE_OVERLAP."""
        leg_periods = [[0, 1] for _ in range(MIN_BRIDGE_OVERLAP + 1)]
        df = compute_adjacent_bridges(leg_periods, 2)
        assert df["sufficient"][0] is True

    def test_insufficient_flag(self) -> None:
        """Insufficient overlap should be flagged."""
        leg_periods = [[0, 1] for _ in range(MIN_BRIDGE_OVERLAP - 1)]
        df = compute_adjacent_bridges(leg_periods, 2)
        assert df["sufficient"][0] is False

    def test_pair_labels(self) -> None:
        """Pair labels should use arrow notation."""
        leg_periods = [[0, 1]]
        labels = ["84th", "85th"]
        df = compute_adjacent_bridges(leg_periods, 2, labels=labels)
        assert df["pair"][0] == "84th→85th"


# ── emIRT CSV Tests ──────────────────────────────────────────────────────────


class TestEmirtCsv:
    """Tests for emIRT CSV preparation."""

    def test_prepare_csv(self, tmp_path) -> None:
        """CSV should have expected columns."""
        all_irt, all_legs = _make_multi_biennium_data(
            n_shared=3,
            n_unique_per=0,
            n_bienniums=2,
            n_votes=5,
        )
        _, name_to_global = build_global_roster(all_legs, "House")
        stacked = stack_bienniums("House", all_irt, name_to_global, all_legs)

        csv_path = tmp_path / "emirt_input.csv"
        prepare_emirt_csv(stacked, csv_path)

        df = pl.read_csv(csv_path)
        assert set(df.columns) == {"legislator_id", "bill_id", "vote", "time_period"}
        assert df.height == stacked["n_obs"]

    def test_vote_values_binary(self, tmp_path) -> None:
        """Votes should be 0 or 1."""
        all_irt, all_legs = _make_multi_biennium_data(
            n_shared=3,
            n_unique_per=0,
            n_bienniums=2,
            n_votes=5,
        )
        _, name_to_global = build_global_roster(all_legs, "House")
        stacked = stack_bienniums("House", all_irt, name_to_global, all_legs)

        csv_path = tmp_path / "emirt_input.csv"
        prepare_emirt_csv(stacked, csv_path)

        df = pl.read_csv(csv_path)
        assert set(df["vote"].unique().to_list()).issubset({0, 1})


# ── Model Structure Tests ────────────────────────────────────────────────────


class TestModelStructure:
    """Tests for the PyMC model graph builder."""

    @pytest.fixture
    def small_stacked_data(self) -> dict:
        """Create small stacked dataset for model testing."""
        all_irt, all_legs = _make_multi_biennium_data(
            n_shared=5,
            n_unique_per=2,
            n_bienniums=2,
            n_votes=8,
        )
        _, name_to_global = build_global_roster(all_legs, "House")
        return stack_bienniums("House", all_irt, name_to_global, all_legs)

    def test_graph_compiles(self, small_stacked_data) -> None:
        """Model graph should compile without error."""
        from analysis.dynamic_irt import build_dynamic_irt_graph

        model = build_dynamic_irt_graph(small_stacked_data, "per_party")
        assert model is not None

    def test_graph_has_xi(self, small_stacked_data) -> None:
        """Model should have xi deterministic with correct dims."""
        from analysis.dynamic_irt import build_dynamic_irt_graph

        model = build_dynamic_irt_graph(small_stacked_data, "per_party")
        var_names = [v.name for v in model.deterministics]
        assert "xi" in var_names

    def test_per_party_tau_auto_switches_for_small_chamber(self, small_stacked_data) -> None:
        """Per-party tau auto-switches to global for small chambers (ADR-0070)."""
        from analysis.dynamic_irt import build_dynamic_irt_graph

        # small_stacked_data has < SMALL_CHAMBER_THRESHOLD legislators,
        # so per_party request auto-switches to global tau
        model = build_dynamic_irt_graph(small_stacked_data, "per_party")
        tau_var = [v for v in model.free_RVs if v.name == "tau"][0]
        assert tau_var.eval().shape == ()  # scalar, not per-party

    def test_global_tau_scalar(self, small_stacked_data) -> None:
        """Global tau should be a scalar."""
        from analysis.dynamic_irt import build_dynamic_irt_graph

        model = build_dynamic_irt_graph(small_stacked_data, "global")
        tau_var = [v for v in model.free_RVs if v.name == "tau"][0]
        assert tau_var.eval().shape == ()

    def test_positive_beta(self, small_stacked_data) -> None:
        """Beta should use HalfNormal (positive values only)."""
        from analysis.dynamic_irt import build_dynamic_irt_graph

        model = build_dynamic_irt_graph(small_stacked_data, "per_party")
        beta_var = [v for v in model.free_RVs if v.name == "beta"][0]
        # HalfNormal has owner op type HalfNormal
        assert "HalfNormal" in str(type(beta_var.owner.op))

    def test_xi_init_shape(self, small_stacked_data) -> None:
        """xi_init should have n_legislators shape."""
        from analysis.dynamic_irt import build_dynamic_irt_graph

        model = build_dynamic_irt_graph(small_stacked_data, "per_party")
        xi_init = [v for v in model.free_RVs if v.name == "xi_init"][0]
        assert xi_init.eval().shape == (small_stacked_data["n_legislators"],)

    def test_xi_innovations_shape(self, small_stacked_data) -> None:
        """xi_innovations should have (T-1, n_leg) shape."""
        from analysis.dynamic_irt import build_dynamic_irt_graph

        model = build_dynamic_irt_graph(small_stacked_data, "per_party")
        innov = [v for v in model.free_RVs if v.name == "xi_innovations"][0]
        expected = (
            small_stacked_data["n_time"] - 1,
            small_stacked_data["n_legislators"],
        )
        assert innov.eval().shape == expected

    def test_alpha_shape(self, small_stacked_data) -> None:
        """Alpha should have n_bills shape."""
        from analysis.dynamic_irt import build_dynamic_irt_graph

        model = build_dynamic_irt_graph(small_stacked_data, "per_party")
        alpha = [v for v in model.free_RVs if v.name == "alpha"][0]
        assert alpha.eval().shape == (small_stacked_data["n_bills"],)

    def test_observed_data(self, small_stacked_data) -> None:
        """Model should have observed 'obs' variable."""
        from analysis.dynamic_irt import build_dynamic_irt_graph

        model = build_dynamic_irt_graph(small_stacked_data, "per_party")
        obs_names = [v.name for v in model.observed_RVs]
        assert "obs" in obs_names


# ── Post-Processing Tests ────────────────────────────────────────────────────


class TestDecomposePolarization:
    """Tests for polarization decomposition."""

    def test_zero_shift(self) -> None:
        """No movement should produce zero decomposition."""
        from analysis.dynamic_irt import decompose_polarization

        traj = pl.DataFrame(
            {
                "global_idx": [0, 0, 1, 1],
                "name_norm": ["a", "a", "b", "b"],
                "full_name": ["A", "A", "B", "B"],
                "party": ["Republican", "Republican", "Republican", "Republican"],
                "time_period": [0, 1, 0, 1],
                "biennium_label": ["84th", "85th", "84th", "85th"],
                "xi_mean": [1.0, 1.0, -1.0, -1.0],
                "xi_sd": [0.1] * 4,
                "xi_hdi_2.5": [0.9] * 4,
                "xi_hdi_97.5": [1.1] * 4,
                "served": [True] * 4,
            }
        )

        decomp = decompose_polarization(traj)
        assert decomp.height > 0
        assert abs(decomp["total_shift"][0]) < 1e-10

    def test_conversion_only(self) -> None:
        """All members return but shift → pure conversion."""
        from analysis.dynamic_irt import decompose_polarization

        traj = pl.DataFrame(
            {
                "global_idx": [0, 0, 1, 1],
                "name_norm": ["a", "a", "b", "b"],
                "full_name": ["A", "A", "B", "B"],
                "party": ["Republican", "Republican", "Republican", "Republican"],
                "time_period": [0, 1, 0, 1],
                "biennium_label": ["84th", "85th", "84th", "85th"],
                "xi_mean": [1.0, 2.0, -1.0, 0.0],
                "xi_sd": [0.1] * 4,
                "xi_hdi_2.5": [0.9] * 4,
                "xi_hdi_97.5": [1.1] * 4,
                "served": [True] * 4,
            }
        )

        decomp = decompose_polarization(traj)
        assert decomp.height > 0
        # Total shift = mean([2,0]) - mean([1,-1]) = 1 - 0 = 1
        assert abs(decomp["total_shift"][0] - 1.0) < 1e-10
        # All returning → conversion equals total
        assert abs(decomp["conversion"][0] - 1.0) < 1e-10
        assert abs(decomp["replacement"][0]) < 1e-10

    def test_empty_trajectories(self) -> None:
        """Empty input should return empty DataFrame."""
        from analysis.dynamic_irt import decompose_polarization

        traj = pl.DataFrame(
            schema={
                "global_idx": pl.Int64,
                "name_norm": pl.Utf8,
                "full_name": pl.Utf8,
                "party": pl.Utf8,
                "time_period": pl.Int64,
                "biennium_label": pl.Utf8,
                "xi_mean": pl.Float64,
                "xi_sd": pl.Float64,
                "xi_hdi_2.5": pl.Float64,
                "xi_hdi_97.5": pl.Float64,
                "served": pl.Boolean,
            }
        )
        decomp = decompose_polarization(traj)
        assert decomp.height == 0


class TestIdentifyTopMovers:
    """Tests for top mover identification."""

    def test_ranking_by_total_movement(self) -> None:
        """Top movers should be ranked by total movement (descending)."""
        from analysis.dynamic_irt import identify_top_movers

        traj = pl.DataFrame(
            {
                "global_idx": [0, 0, 1, 1, 2, 2],
                "name_norm": ["a", "a", "b", "b", "c", "c"],
                "full_name": ["A", "A", "B", "B", "C", "C"],
                "party": ["R", "R", "R", "R", "R", "R"],
                "time_period": [0, 1, 0, 1, 0, 1],
                "biennium_label": ["84th", "85th", "84th", "85th", "84th", "85th"],
                "xi_mean": [0.0, 3.0, 0.0, 1.0, 0.0, 0.5],  # A moves 3, B moves 1, C moves 0.5
                "xi_sd": [0.1] * 6,
                "xi_hdi_2.5": [0.0] * 6,
                "xi_hdi_97.5": [0.0] * 6,
                "served": [True] * 6,
            }
        )

        movers = identify_top_movers(traj, n_top=3)
        assert movers.height == 3
        assert movers["name_norm"][0] == "a"  # biggest mover first
        assert movers["total_movement"][0] == 3.0

    def test_single_period_excluded(self) -> None:
        """Legislators with only 1 period cannot be movers."""
        from analysis.dynamic_irt import identify_top_movers

        traj = pl.DataFrame(
            {
                "global_idx": [0],
                "name_norm": ["a"],
                "full_name": ["A"],
                "party": ["R"],
                "time_period": [0],
                "biennium_label": ["84th"],
                "xi_mean": [1.0],
                "xi_sd": [0.1],
                "xi_hdi_2.5": [0.9],
                "xi_hdi_97.5": [1.1],
                "served": [True],
            }
        )

        movers = identify_top_movers(traj)
        assert movers.height == 0

    def test_n_top_limits(self) -> None:
        """n_top should limit the output."""
        from analysis.dynamic_irt import identify_top_movers

        rows = []
        for i in range(30):
            for t in range(2):
                rows.append(
                    {
                        "global_idx": i,
                        "name_norm": f"m{i}",
                        "full_name": f"Member {i}",
                        "party": "R",
                        "time_period": t,
                        "biennium_label": ["84th", "85th"][t],
                        "xi_mean": float(i) * (t + 1),
                        "xi_sd": 0.1,
                        "xi_hdi_2.5": 0.0,
                        "xi_hdi_97.5": 0.0,
                        "served": True,
                    }
                )
        traj = pl.DataFrame(rows)
        movers = identify_top_movers(traj, n_top=5)
        assert movers.height == 5

    def test_net_movement_direction(self) -> None:
        """Direction should be rightward for positive net shift."""
        from analysis.dynamic_irt import identify_top_movers

        traj = pl.DataFrame(
            {
                "global_idx": [0, 0],
                "name_norm": ["a", "a"],
                "full_name": ["A", "A"],
                "party": ["R", "R"],
                "time_period": [0, 1],
                "biennium_label": ["84th", "85th"],
                "xi_mean": [-1.0, 1.0],
                "xi_sd": [0.1, 0.1],
                "xi_hdi_2.5": [-1.1, 0.9],
                "xi_hdi_97.5": [-0.9, 1.1],
                "served": [True, True],
            }
        )

        movers = identify_top_movers(traj)
        assert movers["direction"][0] == "rightward"
        assert movers["net_movement"][0] == 2.0


class TestFixPeriodSignFlips:
    """Tests for post-hoc per-period sign correction."""

    @staticmethod
    def _make_fake_idata(n_time: int, n_leg: int, xi_values: np.ndarray):
        """Build a minimal InferenceData with xi posterior.

        Args:
            n_time: Number of time periods.
            n_leg: Number of legislators.
            xi_values: Array of shape (n_time, n_leg) — broadcast to (1, 1, T, L).
        """
        import arviz as az
        import xarray as xr

        xi_4d = xi_values[np.newaxis, np.newaxis, :, :]  # (1, 1, T, L)
        ds = xr.Dataset({"xi": (["chain", "draw", "xi_dim_0", "xi_dim_1"], xi_4d)})
        return az.InferenceData(posterior=ds)

    @staticmethod
    def _make_data(n_time: int, n_leg: int, leg_names: list[str]) -> dict:
        """Build a minimal stacked data dict."""
        return {
            "leg_names": leg_names,
            "leg_periods": [[t for t in range(n_time)] for _ in range(n_leg)],
            "party_idx": np.zeros(n_leg, dtype=int),
            "party_names": ["Republican"],
        }

    @staticmethod
    def _make_roster(leg_names: list[str]) -> pl.DataFrame:
        """Build a minimal roster."""
        return pl.DataFrame(
            {
                "global_idx": list(range(len(leg_names))),
                "name_norm": leg_names,
                "full_name": [n.title() for n in leg_names],
                "parties": ["Republican"] * len(leg_names),
                "first_period": [0] * len(leg_names),
                "last_period": [0] * len(leg_names),
                "n_periods": [1] * len(leg_names),
            }
        )

    def test_no_flip_needed(self) -> None:
        """Positive correlations should return empty corrections list."""
        from analysis.dynamic_irt import fix_period_sign_flips

        names = [f"leg_{i}" for i in range(10)]
        xi = np.array([[float(i) for i in range(10)]])  # (1, 10) — one period
        idata = self._make_fake_idata(1, 10, xi)
        data = self._make_data(1, 10, names)
        roster = self._make_roster(names)

        static = {0: pl.DataFrame({"name_norm": names, "xi_mean": [float(i) for i in range(10)]})}

        result_idata, corrections = fix_period_sign_flips(idata, data, static, roster)
        assert corrections == []
        # xi should be unchanged
        np.testing.assert_array_equal(result_idata.posterior["xi"].values[0, 0, 0], xi[0])

    def test_single_period_flip(self) -> None:
        """Negative r should negate xi for that period and record correction."""
        from analysis.dynamic_irt import fix_period_sign_flips

        names = [f"leg_{i}" for i in range(10)]
        # Dynamic xi is negative of static (flipped)
        xi = np.array([[-float(i) for i in range(10)]])  # (1, 10)
        idata = self._make_fake_idata(1, 10, xi)
        data = self._make_data(1, 10, names)
        roster = self._make_roster(names)

        static = {0: pl.DataFrame({"name_norm": names, "xi_mean": [float(i) for i in range(10)]})}

        result_idata, corrections = fix_period_sign_flips(idata, data, static, roster)
        assert len(corrections) == 1
        assert corrections[0]["label"] == "84th"
        assert corrections[0]["r_before"] < 0
        assert corrections[0]["r_after"] > 0
        assert corrections[0]["n_matched"] == 10
        assert len(corrections[0]["reference_legs"]) == 3

        # xi should now be positive (negated)
        corrected_xi = result_idata.posterior["xi"].values[0, 0, 0]
        np.testing.assert_array_almost_equal(corrected_xi, [float(i) for i in range(10)])

    def test_multiple_flips(self) -> None:
        """Multiple periods with negative r should all be corrected."""
        from analysis.dynamic_irt import fix_period_sign_flips

        names = [f"leg_{i}" for i in range(10)]
        # Both periods flipped
        xi = np.array(
            [
                [-float(i) for i in range(10)],
                [-float(i) * 2 for i in range(10)],
            ]
        )  # (2, 10)
        idata = self._make_fake_idata(2, 10, xi)
        data = self._make_data(2, 10, names)
        roster = self._make_roster(names)

        static = {
            0: pl.DataFrame({"name_norm": names, "xi_mean": [float(i) for i in range(10)]}),
            1: pl.DataFrame({"name_norm": names, "xi_mean": [float(i) * 2 for i in range(10)]}),
        }

        _, corrections = fix_period_sign_flips(idata, data, static, roster)
        assert len(corrections) == 2
        assert corrections[0]["label"] == "84th"
        assert corrections[1]["label"] == "85th"

    def test_insufficient_matches_skipped(self) -> None:
        """Fewer than 5 matched legislators should be skipped."""
        from analysis.dynamic_irt import fix_period_sign_flips

        names = [f"leg_{i}" for i in range(4)]
        xi = np.array([[-float(i) for i in range(4)]])  # (1, 4) — flipped
        idata = self._make_fake_idata(1, 4, xi)
        data = self._make_data(1, 4, names)
        roster = self._make_roster(names)

        static = {0: pl.DataFrame({"name_norm": names, "xi_mean": [float(i) for i in range(4)]})}

        _, corrections = fix_period_sign_flips(idata, data, static, roster)
        assert corrections == []  # too few matches, skip

    def test_no_static_data(self) -> None:
        """Empty static dict should be a graceful no-op."""
        from analysis.dynamic_irt import fix_period_sign_flips

        names = [f"leg_{i}" for i in range(10)]
        xi = np.array([[-float(i) for i in range(10)]])
        idata = self._make_fake_idata(1, 10, xi)
        data = self._make_data(1, 10, names)
        roster = self._make_roster(names)

        _, corrections = fix_period_sign_flips(idata, data, {}, roster)
        assert corrections == []


class TestCorrelateWithStatic:
    """Tests for static IRT correlation."""

    def test_perfect_correlation(self) -> None:
        """Identical values should produce r = 1.0."""
        from analysis.dynamic_irt import correlate_with_static

        traj = pl.DataFrame(
            {
                "global_idx": [0, 1, 2, 3, 4],
                "name_norm": ["a", "b", "c", "d", "e"],
                "full_name": ["A", "B", "C", "D", "E"],
                "party": ["R"] * 5,
                "time_period": [0] * 5,
                "biennium_label": ["84th"] * 5,
                "xi_mean": [1.0, 2.0, 3.0, 4.0, 5.0],
                "xi_sd": [0.1] * 5,
                "xi_hdi_2.5": [0.9] * 5,
                "xi_hdi_97.5": [1.1] * 5,
                "served": [True] * 5,
            }
        )

        static = {
            0: pl.DataFrame(
                {
                    "name_norm": ["a", "b", "c", "d", "e"],
                    "xi_mean": [1.0, 2.0, 3.0, 4.0, 5.0],
                }
            )
        }

        corr = correlate_with_static(traj, static)
        assert corr.height == 1
        assert abs(corr["pearson_r"][0] - 1.0) < 1e-6

    def test_no_static_data(self) -> None:
        """Missing static data should return empty DataFrame."""
        from analysis.dynamic_irt import correlate_with_static

        traj = pl.DataFrame(
            {
                "global_idx": [0],
                "name_norm": ["a"],
                "full_name": ["A"],
                "party": ["R"],
                "time_period": [0],
                "biennium_label": ["84th"],
                "xi_mean": [1.0],
                "xi_sd": [0.1],
                "xi_hdi_2.5": [0.9],
                "xi_hdi_97.5": [1.1],
                "served": [True],
            }
        )

        corr = correlate_with_static(traj, {})
        assert corr.height == 0


class TestCheckConvergence:
    """Tests for convergence checking."""

    def test_convergence_dict_keys(self) -> None:
        """Check that convergence returns expected keys."""
        from analysis.dynamic_irt import ESS_THRESHOLD, RHAT_THRESHOLD

        # Test the thresholds are set correctly
        assert RHAT_THRESHOLD > 1.0
        assert ESS_THRESHOLD > 0


# ── Report Tests ─────────────────────────────────────────────────────────────


class TestReport:
    """Tests for the report builder."""

    def test_build_report_no_crash(self) -> None:
        """Report builder should not crash with minimal input."""
        from analysis.dynamic_irt_report import build_dynamic_irt_report

        from analysis.report import ReportBuilder

        report = ReportBuilder(title="Test Report")
        results = {"chambers": []}
        build_dynamic_irt_report(
            report,
            results=results,
            plots_dir=None,
            biennium_labels=["84th"],
        )
        # Should add at least overview + methodology
        assert len(report._sections) >= 2

    def test_methodology_section_present(self) -> None:
        """Methodology section should always be present."""
        from analysis.dynamic_irt_report import build_dynamic_irt_report

        from analysis.report import ReportBuilder

        report = ReportBuilder(title="Test Report")
        results = {"chambers": []}
        build_dynamic_irt_report(
            report,
            results=results,
            plots_dir=None,
            biennium_labels=["84th"],
        )
        section_ids = [s.id for _, s in report._sections]
        assert "methodology" in section_ids


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests."""

    def test_single_biennium_valid(self) -> None:
        """Single biennium should produce valid (degenerate) output."""
        names = ["Alice Smith", "Bob Jones", "Carol White"]
        all_irt = {0: _make_irt_data(names, 10)}
        all_legs = {0: _make_legislators(names)}
        _, name_to_global = build_global_roster(all_legs, "House")
        stacked = stack_bienniums("House", all_irt, name_to_global, all_legs)

        assert stacked["n_time"] == 1
        assert stacked["n_legislators"] == 3

        # Bridge coverage with single period
        bridge = compute_adjacent_bridges(stacked["leg_periods"], stacked["n_time"])
        assert bridge.height == 0  # no adjacent pairs

    def test_legislator_serving_one_period_in_multi(self) -> None:
        """Legislator serving only 1 period should still be tracked."""
        names_t0 = ["Alice Smith", "Bob Jones"]
        names_t1 = ["Alice Smith", "Carol White"]  # Bob leaves, Carol joins
        names_t2 = ["Alice Smith", "Carol White"]

        all_irt = {
            0: _make_irt_data(names_t0, 5, seed=1),
            1: _make_irt_data(names_t1, 5, seed=2),
            2: _make_irt_data(names_t2, 5, seed=3),
        }
        all_legs = {
            0: _make_legislators(names_t0),
            1: _make_legislators(names_t1),
            2: _make_legislators(names_t2),
        }
        roster, name_to_global = build_global_roster(all_legs, "House")
        stacked = stack_bienniums("House", all_irt, name_to_global, all_legs)

        # Bob only served period 0
        bob_idx = name_to_global["bob jones"]
        assert stacked["leg_periods"][bob_idx] == [0]

        # Alice served all 3
        alice_idx = name_to_global["alice smith"]
        assert stacked["leg_periods"][alice_idx] == [0, 1, 2]

    def test_no_overlap_still_works(self) -> None:
        """Completely disjoint legislator sets should still produce valid data."""
        names_t0 = ["Alice Smith", "Bob Jones"]
        names_t1 = ["Carol White", "Dave Green"]

        all_irt = {
            0: _make_irt_data(names_t0, 5, seed=1),
            1: _make_irt_data(names_t1, 5, seed=2),
        }
        all_legs = {
            0: _make_legislators(names_t0),
            1: _make_legislators(names_t1),
        }
        _, name_to_global = build_global_roster(all_legs, "House")
        stacked = stack_bienniums("House", all_irt, name_to_global, all_legs)

        assert stacked["n_legislators"] == 4
        bridge = compute_adjacent_bridges(stacked["leg_periods"], stacked["n_time"])
        assert bridge["shared_count"][0] == 0

    def test_mixed_parties(self) -> None:
        """Multiple parties should be tracked correctly."""
        dems = _make_legislators(["Dem A", "Dem B"], party="Democrat")
        reps = _make_legislators(["Rep A", "Rep B"], party="Republican", prefix="rep2")
        legs = pl.concat([dems, reps])

        all_legs = {0: legs, 1: legs}
        all_names = ["Dem A", "Dem B", "Rep A", "Rep B"]
        all_irt = {
            0: _make_irt_data(all_names, 5, seed=1),
            1: _make_irt_data(all_names, 5, seed=2),
        }

        _, name_to_global = build_global_roster(all_legs, "House")
        stacked = stack_bienniums("House", all_irt, name_to_global, all_legs)

        assert len(stacked["party_names"]) == 2
        assert "Democrat" in stacked["party_names"]
        assert "Republican" in stacked["party_names"]


# ── Graph Construction Tests (ADR-0070) ──────────────────────────────────────


def _make_small_stacked_data(n_leg: int = 10, n_time: int = 2) -> dict:
    """Build minimal stacked data dict for graph construction tests."""
    n_bills = 20
    n_obs = n_leg * n_bills
    return {
        "leg_global_idx": np.repeat(np.arange(n_leg), n_bills),
        "bill_idx": np.tile(np.arange(n_bills), n_leg),
        "time_idx": np.zeros(n_obs, dtype=int),
        "y": np.random.default_rng(42).integers(0, 2, size=n_obs),
        "n_legislators": n_leg,
        "n_bills": n_bills,
        "n_time": n_time,
        "n_obs": n_obs,
        "party_idx": np.array([0] * (n_leg // 2) + [1] * (n_leg - n_leg // 2)),
        "party_names": ["Democrat", "Republican"],
        "leg_names": [f"leg_{i}" for i in range(n_leg)],
        "bill_ids": [f"bill_{i}" for i in range(n_bills)],
        "leg_periods": [list(range(n_time)) for _ in range(n_leg)],
    }


class TestDynamicIRTGraphConstruction:
    """Graph construction tests — no MCMC, just model structure (ADR-0070)."""

    def test_uninformative_prior_default(self) -> None:
        """Default xi_init uses Normal(0, 1)."""
        data = _make_small_stacked_data(n_leg=20)
        model = build_dynamic_irt_graph(data, xi_init_mu=None)
        assert "xi_init" in [rv.name for rv in model.free_RVs]

    def test_informative_prior_uses_mu(self) -> None:
        """Provided xi_init_mu is used as prior mean."""
        data = _make_small_stacked_data(n_leg=20)
        mu = np.linspace(-1, 1, 20)
        model = build_dynamic_irt_graph(data, xi_init_mu=mu)
        assert "xi_init" in [rv.name for rv in model.free_RVs]
        assert model is not None

    def test_adaptive_tau_small_chamber(self) -> None:
        """Small chamber (n_leg < 80) auto-switches to global tau."""
        data = _make_small_stacked_data(n_leg=40)
        model = build_dynamic_irt_graph(data, evolution_structure="per_party")
        tau_rv = model["tau"]
        # Small chamber should auto-switch to global (scalar tau)
        assert tau_rv.type.ndim == 0, "Small chamber should use global (scalar) tau"

    def test_adaptive_tau_large_chamber(self) -> None:
        """Large chamber (n_leg >= 80) keeps per-party tau."""
        data = _make_small_stacked_data(n_leg=100)
        model = build_dynamic_irt_graph(data, evolution_structure="per_party")
        tau_rv = model["tau"]
        # Large chamber keeps per-party tau (vector)
        assert tau_rv.type.ndim == 1, "Large chamber should use per-party (vector) tau"

    def test_tau_sigma_override(self) -> None:
        """Explicit tau_sigma=0.3 overrides adaptive logic."""
        data = _make_small_stacked_data(n_leg=40)
        model = build_dynamic_irt_graph(data, tau_sigma=0.3)
        assert "tau" in [rv.name for rv in model.free_RVs]
        assert model is not None
