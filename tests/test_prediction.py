"""Tests for prediction analysis module.

Uses a synthetic 20-legislator × 50-vote fixture with known party-line structure.
10 Republicans (IRT positive) and 10 Democrats (IRT negative), voting along
party lines with some noise.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from analysis.prediction import (
    build_bill_features,
    build_vote_features,
    compute_per_legislator_accuracy,
    compute_shap_values,
    find_surprising_votes,
    train_passage_models,
    train_vote_models,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_LEGISLATORS = 20
N_ROLLCALLS = 50


@pytest.fixture
def synthetic_legislators() -> pl.DataFrame:
    """20 legislators: 10 R, 10 D."""
    rows = []
    for i in range(10):
        rows.append(
            {
                "name": f"R{i}",
                "full_name": f"Rep Republican{i}",
                "slug": f"rep_r{i}_1",
                "chamber": "House",
                "party": "Republican",
                "district": str(i + 1),
                "member_url": f"http://example.com/r{i}",
            }
        )
    for i in range(10):
        rows.append(
            {
                "name": f"D{i}",
                "full_name": f"Rep Democrat{i}",
                "slug": f"rep_d{i}_1",
                "chamber": "House",
                "party": "Democrat",
                "district": str(i + 11),
                "member_url": f"http://example.com/d{i}",
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture
def synthetic_votes(synthetic_legislators: pl.DataFrame) -> pl.DataFrame:
    """20 legislators × 50 rollcalls with party-line voting + noise."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    slugs = synthetic_legislators["slug"].to_list()
    parties = synthetic_legislators["party"].to_list()

    for j in range(N_ROLLCALLS):
        vote_id = f"je_202503{j:02d}120000"
        for slug, party in zip(slugs, parties):
            # Party-line with 10% noise
            if party == "Republican":
                vote = "Yea" if rng.random() > 0.10 else "Nay"
            else:
                vote = "Nay" if rng.random() > 0.20 else "Yea"

            rows.append(
                {
                    "session": "2025-26",
                    "bill_number": f"HB {j + 1}",
                    "bill_title": f"Test Bill {j + 1}",
                    "vote_id": vote_id,
                    "vote_datetime": f"2025-03-{j % 28 + 1:02d}T12:00:00",
                    "vote_date": f"2025-03-{j % 28 + 1:02d}",
                    "chamber": "House",
                    "motion": "Final Action",
                    "legislator_name": slug.replace("rep_", "").replace("_1", ""),
                    "legislator_slug": slug,
                    "vote": vote,
                }
            )

    return pl.DataFrame(rows)


@pytest.fixture
def synthetic_rollcalls() -> pl.DataFrame:
    """50 roll calls with realistic metadata."""
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for j in range(N_ROLLCALLS):
        yea = rng.integers(8, 18)
        nay = 20 - yea
        passed = yea > nay
        rows.append(
            {
                "session": "2025-26",
                "bill_number": f"HB {j + 1}",
                "bill_title": f"Test Bill {j + 1}",
                "vote_id": f"je_202503{j:02d}120000",
                "vote_url": f"http://example.com/vote/{j}",
                "vote_datetime": f"2025-03-{j % 28 + 1:02d}T12:00:00",
                "vote_date": f"2025-03-{j % 28 + 1:02d}",
                "chamber": "House",
                "motion": "Final Action",
                "vote_type": "Final Action" if j % 3 != 2 else "Emergency Final Action",
                "result": "Passed" if passed else "Failed",
                "short_title": f"Test {j}",
                "sponsor": "Test Sponsor",
                "yea_count": int(yea),
                "nay_count": int(nay),
                "present_passing_count": 0,
                "absent_not_voting_count": 0,
                "not_voting_count": 0,
                "total_votes": 20,
                "passed": passed,
            }
        )

    return pl.DataFrame(rows)


@pytest.fixture
def synthetic_ideal_points() -> pl.DataFrame:
    """IRT ideal points: Rs positive, Ds negative."""
    rows = []
    for i in range(10):
        rows.append(
            {
                "legislator_slug": f"rep_r{i}_1",
                "xi_mean": 1.5 + 0.1 * i,
                "xi_sd": 0.3,
                "xi_hdi_2.5": 1.0 + 0.1 * i,
                "xi_hdi_97.5": 2.0 + 0.1 * i,
                "full_name": f"Rep Republican{i}",
                "party": "Republican",
                "district": str(i + 1),
                "chamber": "House",
            }
        )
    for i in range(10):
        rows.append(
            {
                "legislator_slug": f"rep_d{i}_1",
                "xi_mean": -1.5 - 0.1 * i,
                "xi_sd": 0.3,
                "xi_hdi_2.5": -2.0 - 0.1 * i,
                "xi_hdi_97.5": -1.0 - 0.1 * i,
                "full_name": f"Rep Democrat{i}",
                "party": "Democrat",
                "district": str(i + 11),
                "chamber": "House",
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture
def synthetic_bill_params() -> pl.DataFrame:
    """IRT bill parameters for 50 votes."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for j in range(N_ROLLCALLS):
        rows.append(
            {
                "vote_id": f"je_202503{j:02d}120000",
                "alpha_mean": rng.normal(0, 1),
                "alpha_sd": 0.2,
                "beta_mean": rng.normal(1, 0.5),
                "beta_sd": 0.2,
                "bill_number": f"HB {j + 1}",
                "short_title": f"Test {j}",
                "motion": "Final Action",
                "vote_type": "Final Action",
                "is_veto_override": j < 3,  # First 3 are overrides
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture
def synthetic_party_loyalty() -> pl.DataFrame:
    rows = []
    for i in range(10):
        rows.append(
            {
                "legislator_slug": f"rep_r{i}_1",
                "party": "Republican",
                "n_agree": 45 - i,
                "n_contested_votes": 50,
                "loyalty_rate": (45 - i) / 50,
                "full_name": f"Rep Republican{i}",
            }
        )
    for i in range(10):
        rows.append(
            {
                "legislator_slug": f"rep_d{i}_1",
                "party": "Democrat",
                "n_agree": 40 - i,
                "n_contested_votes": 50,
                "loyalty_rate": (40 - i) / 50,
                "full_name": f"Rep Democrat{i}",
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture
def synthetic_centrality() -> pl.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for i in range(10):
        rows.append(
            {
                "legislator_slug": f"rep_r{i}_1",
                "full_name": f"Rep Republican{i}",
                "party": "Republican",
                "xi_mean": 1.5 + 0.1 * i,
                "degree": rng.uniform(0.5, 0.9),
                "weighted_degree": rng.uniform(5, 10),
                "betweenness": rng.uniform(0, 0.05),
                "eigenvector": rng.uniform(0.1, 0.2),
                "closeness": rng.uniform(0.5, 0.8),
                "pagerank": rng.uniform(0.04, 0.06),
            }
        )
    for i in range(10):
        rows.append(
            {
                "legislator_slug": f"rep_d{i}_1",
                "full_name": f"Rep Democrat{i}",
                "party": "Democrat",
                "xi_mean": -1.5 - 0.1 * i,
                "degree": rng.uniform(0.5, 0.9),
                "weighted_degree": rng.uniform(5, 10),
                "betweenness": rng.uniform(0, 0.05),
                "eigenvector": rng.uniform(0.1, 0.2),
                "closeness": rng.uniform(0.5, 0.8),
                "pagerank": rng.uniform(0.04, 0.06),
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture
def synthetic_pc_scores() -> pl.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for i in range(10):
        rows.append(
            {
                "legislator_slug": f"rep_r{i}_1",
                "PC1": 10 + rng.normal(0, 2),
                "PC2": rng.normal(0, 1),
                "PC3": rng.normal(0, 0.5),
                "PC4": rng.normal(0, 0.3),
                "PC5": rng.normal(0, 0.2),
                "full_name": f"Rep Republican{i}",
                "party": "Republican",
                "district": str(i + 1),
                "chamber": "House",
            }
        )
    for i in range(10):
        rows.append(
            {
                "legislator_slug": f"rep_d{i}_1",
                "PC1": -10 + rng.normal(0, 2),
                "PC2": rng.normal(0, 1),
                "PC3": rng.normal(0, 0.5),
                "PC4": rng.normal(0, 0.3),
                "PC5": rng.normal(0, 0.2),
                "full_name": f"Rep Democrat{i}",
                "party": "Democrat",
                "district": str(i + 11),
                "chamber": "House",
            }
        )
    return pl.DataFrame(rows)


# ── Tests: Build Vote Features ───────────────────────────────────────────────


class TestBuildVoteFeatures:
    def test_correct_shape(
        self,
        synthetic_votes,
        synthetic_rollcalls,
        synthetic_legislators,
        synthetic_ideal_points,
        synthetic_bill_params,
        synthetic_party_loyalty,
        synthetic_centrality,
        synthetic_pc_scores,
    ):
        result = build_vote_features(
            synthetic_votes,
            synthetic_rollcalls,
            synthetic_legislators,
            synthetic_ideal_points,
            synthetic_bill_params,
            synthetic_party_loyalty,
            synthetic_centrality,
            synthetic_pc_scores,
            "House",
        )
        # Should have ~1000 rows (20 legislators × 50 votes, minus any dropped)
        assert result.height > 500
        assert result.height <= N_LEGISLATORS * N_ROLLCALLS

    def test_no_nan_in_features(
        self,
        synthetic_votes,
        synthetic_rollcalls,
        synthetic_legislators,
        synthetic_ideal_points,
        synthetic_bill_params,
        synthetic_party_loyalty,
        synthetic_centrality,
        synthetic_pc_scores,
    ):
        result = build_vote_features(
            synthetic_votes,
            synthetic_rollcalls,
            synthetic_legislators,
            synthetic_ideal_points,
            synthetic_bill_params,
            synthetic_party_loyalty,
            synthetic_centrality,
            synthetic_pc_scores,
            "House",
        )
        # After drop_nulls, no NaN should remain in feature columns
        exclude = {"legislator_slug", "vote_id", "vote_binary"}
        for col in result.columns:
            if col not in exclude:
                assert result[col].null_count() == 0, f"NaN found in {col}"

    def test_correct_target_encoding(
        self,
        synthetic_votes,
        synthetic_rollcalls,
        synthetic_legislators,
        synthetic_ideal_points,
        synthetic_bill_params,
        synthetic_party_loyalty,
        synthetic_centrality,
        synthetic_pc_scores,
    ):
        result = build_vote_features(
            synthetic_votes,
            synthetic_rollcalls,
            synthetic_legislators,
            synthetic_ideal_points,
            synthetic_bill_params,
            synthetic_party_loyalty,
            synthetic_centrality,
            synthetic_pc_scores,
            "House",
        )
        unique_targets = result["vote_binary"].unique().sort().to_list()
        assert unique_targets == [0, 1]

    def test_expected_columns_present(
        self,
        synthetic_votes,
        synthetic_rollcalls,
        synthetic_legislators,
        synthetic_ideal_points,
        synthetic_bill_params,
        synthetic_party_loyalty,
        synthetic_centrality,
        synthetic_pc_scores,
    ):
        result = build_vote_features(
            synthetic_votes,
            synthetic_rollcalls,
            synthetic_legislators,
            synthetic_ideal_points,
            synthetic_bill_params,
            synthetic_party_loyalty,
            synthetic_centrality,
            synthetic_pc_scores,
            "House",
        )
        expected = {
            "party_binary",
            "xi_mean",
            "xi_sd",
            "loyalty_rate",
            "PC1",
            "PC2",
            "betweenness",
            "eigenvector",
            "pagerank",
            "alpha_mean",
            "beta_mean",
            "is_veto_override",
            "day_of_session",
            "xi_x_beta",
            "vote_binary",
            "legislator_slug",
            "vote_id",
        }
        assert expected.issubset(set(result.columns))


# ── Tests: Build Bill Features ───────────────────────────────────────────────


class TestBuildBillFeatures:
    def test_correct_shape(self, synthetic_rollcalls, synthetic_bill_params):
        result = build_bill_features(synthetic_rollcalls, synthetic_bill_params, "House")
        assert result.height > 0
        assert result.height <= N_ROLLCALLS

    def test_target_is_binary(self, synthetic_rollcalls, synthetic_bill_params):
        result = build_bill_features(synthetic_rollcalls, synthetic_bill_params, "House")
        unique_targets = result["passed_binary"].unique().sort().to_list()
        assert all(t in [0, 1] for t in unique_targets)

    def test_vote_type_one_hot_present(self, synthetic_rollcalls, synthetic_bill_params):
        result = build_bill_features(synthetic_rollcalls, synthetic_bill_params, "House")
        vt_cols = [c for c in result.columns if c.startswith("vt_")]
        assert len(vt_cols) > 0


# ── Tests: Train Vote Models ────────────────────────────────────────────────


class TestTrainVoteModels:
    def test_returns_all_models(
        self,
        synthetic_votes,
        synthetic_rollcalls,
        synthetic_legislators,
        synthetic_ideal_points,
        synthetic_bill_params,
        synthetic_party_loyalty,
        synthetic_centrality,
        synthetic_pc_scores,
    ):
        features = build_vote_features(
            synthetic_votes,
            synthetic_rollcalls,
            synthetic_legislators,
            synthetic_ideal_points,
            synthetic_bill_params,
            synthetic_party_loyalty,
            synthetic_centrality,
            synthetic_pc_scores,
            "House",
        )
        result = train_vote_models(features, "House")
        assert "Logistic Regression" in result["models"]
        assert "XGBoost" in result["models"]
        assert "Random Forest" in result["models"]

    def test_cv_results_have_expected_keys(
        self,
        synthetic_votes,
        synthetic_rollcalls,
        synthetic_legislators,
        synthetic_ideal_points,
        synthetic_bill_params,
        synthetic_party_loyalty,
        synthetic_centrality,
        synthetic_pc_scores,
    ):
        features = build_vote_features(
            synthetic_votes,
            synthetic_rollcalls,
            synthetic_legislators,
            synthetic_ideal_points,
            synthetic_bill_params,
            synthetic_party_loyalty,
            synthetic_centrality,
            synthetic_pc_scores,
            "House",
        )
        result = train_vote_models(features, "House")
        assert len(result["cv_results"]) == 5  # N_SPLITS
        fold0 = result["cv_results"][0]
        assert "XGBoost_accuracy" in fold0
        assert "XGBoost_auc" in fold0

    def test_auc_above_random(
        self,
        synthetic_votes,
        synthetic_rollcalls,
        synthetic_legislators,
        synthetic_ideal_points,
        synthetic_bill_params,
        synthetic_party_loyalty,
        synthetic_centrality,
        synthetic_pc_scores,
    ):
        features = build_vote_features(
            synthetic_votes,
            synthetic_rollcalls,
            synthetic_legislators,
            synthetic_ideal_points,
            synthetic_bill_params,
            synthetic_party_loyalty,
            synthetic_centrality,
            synthetic_pc_scores,
            "House",
        )
        result = train_vote_models(features, "House")
        cv_df = pl.DataFrame(result["cv_results"])
        xgb_auc_mean = cv_df["XGBoost_auc"].mean()
        assert xgb_auc_mean > 0.5


# ── Tests: Train Passage Models ──────────────────────────────────────────────


class TestTrainPassageModels:
    def test_returns_all_models(self, synthetic_rollcalls, synthetic_bill_params):
        features = build_bill_features(synthetic_rollcalls, synthetic_bill_params, "House")
        result = train_passage_models(features, "House")
        if result.get("skipped"):
            pytest.skip("Too few observations")
        assert "Logistic Regression" in result["models"]
        assert "XGBoost" in result["models"]
        assert "Random Forest" in result["models"]

    def test_handles_small_sample(self, synthetic_rollcalls, synthetic_bill_params):
        # Use a small subset
        small_rc = synthetic_rollcalls.head(5)
        features = build_bill_features(small_rc, synthetic_bill_params, "House")
        # Should not crash even with very few rows
        result = train_passage_models(features, "House")
        assert isinstance(result, dict)


# ── Tests: Per-Legislator Accuracy ───────────────────────────────────────────


class TestPerLegislatorAccuracy:
    def test_one_row_per_legislator(
        self,
        synthetic_votes,
        synthetic_rollcalls,
        synthetic_legislators,
        synthetic_ideal_points,
        synthetic_bill_params,
        synthetic_party_loyalty,
        synthetic_centrality,
        synthetic_pc_scores,
    ):
        features = build_vote_features(
            synthetic_votes,
            synthetic_rollcalls,
            synthetic_legislators,
            synthetic_ideal_points,
            synthetic_bill_params,
            synthetic_party_loyalty,
            synthetic_centrality,
            synthetic_pc_scores,
            "House",
        )
        result = train_vote_models(features, "House")
        feature_cols = result["feature_names"]
        xgb = result["models"]["XGBoost"]

        leg_acc = compute_per_legislator_accuracy(
            xgb, features, feature_cols, synthetic_ideal_points
        )
        # Should have one row per legislator
        assert leg_acc.height == features["legislator_slug"].n_unique()

    def test_accuracy_in_range(
        self,
        synthetic_votes,
        synthetic_rollcalls,
        synthetic_legislators,
        synthetic_ideal_points,
        synthetic_bill_params,
        synthetic_party_loyalty,
        synthetic_centrality,
        synthetic_pc_scores,
    ):
        features = build_vote_features(
            synthetic_votes,
            synthetic_rollcalls,
            synthetic_legislators,
            synthetic_ideal_points,
            synthetic_bill_params,
            synthetic_party_loyalty,
            synthetic_centrality,
            synthetic_pc_scores,
            "House",
        )
        result = train_vote_models(features, "House")
        feature_cols = result["feature_names"]
        xgb = result["models"]["XGBoost"]

        leg_acc = compute_per_legislator_accuracy(
            xgb, features, feature_cols, synthetic_ideal_points
        )
        accuracies = leg_acc["accuracy"].to_numpy()
        assert all(0 <= a <= 1 for a in accuracies)


# ── Tests: Surprising Votes ──────────────────────────────────────────────────


class TestSurprisingVotes:
    def test_returns_top_n(
        self,
        synthetic_votes,
        synthetic_rollcalls,
        synthetic_legislators,
        synthetic_ideal_points,
        synthetic_bill_params,
        synthetic_party_loyalty,
        synthetic_centrality,
        synthetic_pc_scores,
    ):
        features = build_vote_features(
            synthetic_votes,
            synthetic_rollcalls,
            synthetic_legislators,
            synthetic_ideal_points,
            synthetic_bill_params,
            synthetic_party_loyalty,
            synthetic_centrality,
            synthetic_pc_scores,
            "House",
        )
        result = train_vote_models(features, "House")
        feature_cols = result["feature_names"]
        xgb = result["models"]["XGBoost"]

        surprising = find_surprising_votes(
            xgb, features, feature_cols, synthetic_rollcalls, synthetic_ideal_points, top_n=10
        )
        assert surprising.height <= 10

    def test_has_expected_columns(
        self,
        synthetic_votes,
        synthetic_rollcalls,
        synthetic_legislators,
        synthetic_ideal_points,
        synthetic_bill_params,
        synthetic_party_loyalty,
        synthetic_centrality,
        synthetic_pc_scores,
    ):
        features = build_vote_features(
            synthetic_votes,
            synthetic_rollcalls,
            synthetic_legislators,
            synthetic_ideal_points,
            synthetic_bill_params,
            synthetic_party_loyalty,
            synthetic_centrality,
            synthetic_pc_scores,
            "House",
        )
        result = train_vote_models(features, "House")
        feature_cols = result["feature_names"]
        xgb = result["models"]["XGBoost"]

        surprising = find_surprising_votes(
            xgb, features, feature_cols, synthetic_rollcalls, synthetic_ideal_points, top_n=10
        )
        if surprising.height > 0:
            expected = {"confidence_error", "y_prob", "actual", "predicted"}
            assert expected.issubset(set(surprising.columns))


# ── Tests: SHAP ──────────────────────────────────────────────────────────────


class TestSHAP:
    def test_shap_values_shape(
        self,
        synthetic_votes,
        synthetic_rollcalls,
        synthetic_legislators,
        synthetic_ideal_points,
        synthetic_bill_params,
        synthetic_party_loyalty,
        synthetic_centrality,
        synthetic_pc_scores,
    ):
        features = build_vote_features(
            synthetic_votes,
            synthetic_rollcalls,
            synthetic_legislators,
            synthetic_ideal_points,
            synthetic_bill_params,
            synthetic_party_loyalty,
            synthetic_centrality,
            synthetic_pc_scores,
            "House",
        )
        result = train_vote_models(features, "House")
        feature_cols = result["feature_names"]
        xgb = result["models"]["XGBoost"]

        X_sample = result["X_test"][:50]
        shap_vals = compute_shap_values(xgb, X_sample, feature_cols)
        assert shap_vals.values.shape[0] == X_sample.shape[0]
        assert shap_vals.values.shape[1] == len(feature_cols)

    def test_feature_names_present(
        self,
        synthetic_votes,
        synthetic_rollcalls,
        synthetic_legislators,
        synthetic_ideal_points,
        synthetic_bill_params,
        synthetic_party_loyalty,
        synthetic_centrality,
        synthetic_pc_scores,
    ):
        features = build_vote_features(
            synthetic_votes,
            synthetic_rollcalls,
            synthetic_legislators,
            synthetic_ideal_points,
            synthetic_bill_params,
            synthetic_party_loyalty,
            synthetic_centrality,
            synthetic_pc_scores,
            "House",
        )
        result = train_vote_models(features, "House")
        feature_cols = result["feature_names"]
        xgb = result["models"]["XGBoost"]

        X_sample = result["X_test"][:50]
        shap_vals = compute_shap_values(xgb, X_sample, feature_cols)
        assert shap_vals.feature_names == feature_cols
