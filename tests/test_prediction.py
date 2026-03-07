"""Tests for prediction analysis module.

Uses a synthetic 20-legislator × 50-vote fixture with known party-line structure.
10 Republicans (IRT positive) and 10 Democrats (IRT negative), voting along
party lines with some noise.
"""

import numpy as np
import polars as pl
import pytest
from analysis.nlp_features import fit_topic_features
from analysis.prediction import (
    HARDEST_N,
    _compute_day_of_session,
    _get_feature_cols,
    _temporal_split_eval,
    build_bill_features,
    build_vote_features,
    compute_per_legislator_accuracy,
    compute_shap_values,
    compute_stratified_accuracy,
    detect_hardest_legislators,
    evaluate_holdout,
    find_surprising_bills,
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
                "legislator_slug": f"rep_r{i}_1",
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
                "legislator_slug": f"rep_d{i}_1",
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
    slugs = synthetic_legislators["legislator_slug"].to_list()
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
    """50 roll calls with realistic metadata and varied short_title text."""
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    # Three topic clusters for NLP testing
    tax_titles = [
        "income tax rate reduction act",
        "property tax exemption for veterans",
        "sales tax on digital goods",
        "tax credit for small business investment",
        "property tax assessment reform",
        "corporate income tax adjustment",
        "sales tax exemption for food items",
        "tax increment financing authorization",
        "estate tax threshold modification",
        "revenue neutral tax reform package",
        "income tax bracket indexing",
        "property tax relief for seniors",
        "sales tax holiday for school supplies",
        "tax deduction for charitable giving",
        "fuel tax rate adjustment",
        "tax compliance and enforcement",
        "tobacco tax increase for health funding",
    ]
    election_titles = [
        "elections and voting procedures reform",
        "voter registration modernization act",
        "ballot access and election security",
        "campaign finance reform and disclosure",
        "election day polling place requirements",
        "absentee ballot processing standards",
        "voter identification requirements update",
        "election audit procedures and oversight",
        "campaign contribution limits revision",
        "voting rights protection and access",
        "election equipment certification standards",
        "polling place accessibility requirements",
        "early voting expansion act",
        "election results certification process",
        "campaign advertising disclosure requirements",
        "redistricting commission establishment",
        "ranked choice voting pilot program",
    ]
    health_titles = [
        "medicaid expansion and coverage update",
        "healthcare facility licensing standards",
        "mental health services funding increase",
        "hospital transparency reporting requirements",
        "prescription drug cost reduction act",
        "school health services expansion",
        "emergency medical services funding",
        "public health emergency preparedness",
        "healthcare workforce development act",
        "telemedicine practice standards",
        "nursing home quality standards",
        "healthcare price transparency act",
        "rural hospital preservation fund",
        "substance abuse treatment funding",
        "maternal health care improvement",
        "health insurance marketplace reform",
    ]

    for j in range(N_ROLLCALLS):
        yea = rng.integers(8, 18)
        nay = 20 - yea
        passed = yea > nay
        # Cycle through topic clusters
        if j % 3 == 0:
            short_title = tax_titles[j // 3 % len(tax_titles)]
        elif j % 3 == 1:
            short_title = election_titles[j // 3 % len(election_titles)]
        else:
            short_title = health_titles[j // 3 % len(health_titles)]

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
                "short_title": short_title,
                "sponsor": (
                    "Representative Republican0" if j % 2 == 0 else "Representative Democrat0"
                ),
                "sponsor_slugs": "rep_r0_1" if j % 2 == 0 else "rep_d0_1",
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


# ── Shared derived fixture ───────────────────────────────────────────────────


@pytest.fixture
def vote_features(
    synthetic_votes,
    synthetic_rollcalls,
    synthetic_legislators,
    synthetic_ideal_points,
    synthetic_bill_params,
    synthetic_party_loyalty,
    synthetic_centrality,
    synthetic_pc_scores,
) -> pl.DataFrame:
    """Pre-built vote-level feature matrix used by most tests."""
    return build_vote_features(
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


@pytest.fixture
def trained_models(vote_features) -> dict:
    """Pre-trained vote models (all 3) used by downstream tests."""
    return train_vote_models(vote_features, "House")


# ── Tests: Build Vote Features ───────────────────────────────────────────────


class TestBuildVoteFeatures:
    def test_correct_shape(self, vote_features):
        # Should have ~1000 rows (20 legislators × 50 votes, minus any dropped)
        assert vote_features.height > 500
        assert vote_features.height <= N_LEGISLATORS * N_ROLLCALLS

    def test_no_nan_in_features(self, vote_features):
        # After drop_nulls, no NaN should remain in feature columns
        exclude = {"legislator_slug", "vote_id", "vote_binary"}
        for col in vote_features.columns:
            if col not in exclude:
                assert vote_features[col].null_count() == 0, f"NaN found in {col}"

    def test_correct_target_encoding(self, vote_features):
        unique_targets = vote_features["vote_binary"].unique().sort().to_list()
        assert unique_targets == [0, 1]

    def test_expected_columns_present(self, vote_features):
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
        assert expected.issubset(set(vote_features.columns))


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
    def test_returns_all_models(self, trained_models):
        assert "Logistic Regression" in trained_models["models"]
        assert "XGBoost" in trained_models["models"]
        assert "Random Forest" in trained_models["models"]

    def test_cv_results_have_expected_keys(self, trained_models):
        assert len(trained_models["cv_results"]) == 5  # N_SPLITS
        fold0 = trained_models["cv_results"][0]
        assert "XGBoost_accuracy" in fold0
        assert "XGBoost_auc" in fold0

    def test_auc_above_random(self, trained_models):
        cv_df = pl.DataFrame(trained_models["cv_results"])
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


# ── Tests: Surprising Bills ─────────────────────────────────────────────────


class TestSurprisingBills:
    """Run: uv run pytest tests/test_prediction.py::TestSurprisingBills -v"""

    def test_empty_result_has_schema(self, synthetic_rollcalls, synthetic_bill_params):
        """When model gets everything right, empty DataFrame should have proper schema."""
        features = build_bill_features(synthetic_rollcalls, synthetic_bill_params, "House")
        result = train_passage_models(features, "House")
        if result.get("skipped"):
            pytest.skip("Too few observations")

        xgb = result["models"]["XGBoost"]
        feature_cols = result["feature_names"]

        # Find surprising bills (may or may not be empty)
        surprising = find_surprising_bills(xgb, features, feature_cols, synthetic_rollcalls)

        # Whether empty or not, should always have the expected columns
        expected_cols = {
            "vote_id",
            "bill_number",
            "passed_binary",
            "y_prob",
            "predicted",
            "confidence_error",
            "motion",
            "vote_type",
            "yea_count",
            "nay_count",
        }
        assert expected_cols.issubset(set(surprising.columns))

    def test_empty_schema_direct(self):
        """Directly test that a perfect model returns typed empty DataFrame, not Schema()."""

        class FakeModel:
            def predict_proba(self, X):
                probs = np.zeros((len(X), 2))
                probs[:, 1] = (X[:, 0] > 0).astype(float)
                return probs

            def predict(self, X):
                return (X[:, 0] > 0).astype(int)

        features_df = pl.DataFrame(
            {
                "vote_id": ["v1", "v2", "v3"],
                "bill_number": ["HB1", "HB2", "HB3"],
                "passed_binary": [1, 1, 0],
                "feat1": [1.0, 1.0, -1.0],
            }
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["v1", "v2", "v3"],
                "motion": ["m1", "m2", "m3"],
                "vote_type": ["Final Action", "Final Action", "Final Action"],
                "yea_count": [80, 90, 10],
                "nay_count": [20, 10, 90],
            }
        )

        result = find_surprising_bills(FakeModel(), features_df, ["feat1"], rollcalls)
        assert result.height == 0
        assert len(result.columns) > 0, "Empty DataFrame should have column schema"
        assert "vote_id" in result.columns


# ── Tests: Per-Legislator Accuracy ───────────────────────────────────────────


class TestPerLegislatorAccuracy:
    def test_one_row_per_legislator(self, vote_features, trained_models, synthetic_ideal_points):
        xgb = trained_models["models"]["XGBoost"]
        feature_cols = trained_models["feature_names"]

        leg_acc = compute_per_legislator_accuracy(
            xgb, vote_features, feature_cols, synthetic_ideal_points
        )
        assert leg_acc.height == vote_features["legislator_slug"].n_unique()

    def test_accuracy_in_range(self, vote_features, trained_models, synthetic_ideal_points):
        xgb = trained_models["models"]["XGBoost"]
        feature_cols = trained_models["feature_names"]

        leg_acc = compute_per_legislator_accuracy(
            xgb, vote_features, feature_cols, synthetic_ideal_points
        )
        accuracies = leg_acc["accuracy"].to_numpy()
        assert all(0 <= a <= 1 for a in accuracies)


# ── Tests: Surprising Votes ──────────────────────────────────────────────────


class TestSurprisingVotes:
    def test_returns_top_n(
        self, vote_features, trained_models, synthetic_rollcalls, synthetic_ideal_points
    ):
        xgb = trained_models["models"]["XGBoost"]
        feature_cols = trained_models["feature_names"]

        surprising = find_surprising_votes(
            xgb, vote_features, feature_cols, synthetic_rollcalls, synthetic_ideal_points, top_n=10
        )
        assert surprising.height <= 10

    def test_has_expected_columns(
        self, vote_features, trained_models, synthetic_rollcalls, synthetic_ideal_points
    ):
        xgb = trained_models["models"]["XGBoost"]
        feature_cols = trained_models["feature_names"]

        surprising = find_surprising_votes(
            xgb, vote_features, feature_cols, synthetic_rollcalls, synthetic_ideal_points, top_n=10
        )
        if surprising.height > 0:
            expected = {"confidence_error", "y_prob", "actual", "predicted"}
            assert expected.issubset(set(surprising.columns))


# ── Tests: SHAP ──────────────────────────────────────────────────────────────


class TestSHAP:
    def test_shap_values_shape(self, trained_models):
        feature_cols = trained_models["feature_names"]
        xgb = trained_models["models"]["XGBoost"]

        X_sample = trained_models["X_test"][:50]
        shap_vals = compute_shap_values(xgb, X_sample, feature_cols)
        assert shap_vals.values.shape[0] == X_sample.shape[0]
        assert shap_vals.values.shape[1] == len(feature_cols)

    def test_feature_names_present(self, trained_models):
        feature_cols = trained_models["feature_names"]
        xgb = trained_models["models"]["XGBoost"]

        X_sample = trained_models["X_test"][:50]
        shap_vals = compute_shap_values(xgb, X_sample, feature_cols)
        assert shap_vals.feature_names == feature_cols


# ── Tests: Build Bill Features with NLP Topics ──────────────────────────────


class TestBuildBillFeaturesWithTopics:
    """Tests for build_bill_features() with NLP topic features."""

    def test_topic_columns_present(self, synthetic_rollcalls, synthetic_bill_params):
        """When topic_features provided, topic_* columns appear in output."""
        chamber_rc = synthetic_rollcalls.filter(pl.col("chamber") == "House")
        topic_df, _ = fit_topic_features(chamber_rc["short_title"])
        topic_df = topic_df.with_columns(chamber_rc["vote_id"])

        result = build_bill_features(
            synthetic_rollcalls, synthetic_bill_params, "House", topic_features=topic_df
        )
        topic_cols = [c for c in result.columns if c.startswith("topic_")]
        assert len(topic_cols) > 0

    def test_backward_compat_with_none(self, synthetic_rollcalls, synthetic_bill_params):
        """Passing topic_features=None produces the same result as before."""
        result_none = build_bill_features(
            synthetic_rollcalls, synthetic_bill_params, "House", topic_features=None
        )
        result_default = build_bill_features(synthetic_rollcalls, synthetic_bill_params, "House")
        assert result_none.columns == result_default.columns
        assert result_none.height == result_default.height

    def test_feature_count_increases(self, synthetic_rollcalls, synthetic_bill_params):
        """Adding topics should increase the feature column count."""
        result_no_topics = build_bill_features(synthetic_rollcalls, synthetic_bill_params, "House")
        chamber_rc = synthetic_rollcalls.filter(pl.col("chamber") == "House")
        topic_df, _ = fit_topic_features(chamber_rc["short_title"])
        topic_df = topic_df.with_columns(chamber_rc["vote_id"])

        result_with_topics = build_bill_features(
            synthetic_rollcalls, synthetic_bill_params, "House", topic_features=topic_df
        )
        assert result_with_topics.width > result_no_topics.width

    def test_passage_model_trains_with_topics(self, synthetic_rollcalls, synthetic_bill_params):
        """Passage models should train successfully with topic features."""
        chamber_rc = synthetic_rollcalls.filter(pl.col("chamber") == "House")
        topic_df, _ = fit_topic_features(chamber_rc["short_title"])
        topic_df = topic_df.with_columns(chamber_rc["vote_id"])

        features = build_bill_features(
            synthetic_rollcalls, synthetic_bill_params, "House", topic_features=topic_df
        )
        result = train_passage_models(features, "House")
        if result.get("skipped"):
            pytest.skip("Too few observations")
        assert "XGBoost" in result["models"]
        assert len(result["feature_names"]) > 0
        # Verify topic columns are in the feature list
        topic_feature_names = [f for f in result["feature_names"] if f.startswith("topic_")]
        assert len(topic_feature_names) > 0


# ── Tests: Detect Hardest Legislators ────────────────────────────────────────


class TestDetectHardestLegislators:
    """Tests for detect_hardest_legislators().

    Run: uv run pytest tests/test_prediction.py::TestDetectHardestLegislators -v
    """

    def test_returns_correct_count(self, vote_features, trained_models, synthetic_ideal_points):
        """Returns up to HARDEST_N legislators."""
        xgb = trained_models["models"]["XGBoost"]
        feature_cols = trained_models["feature_names"]
        leg_acc = compute_per_legislator_accuracy(
            xgb, vote_features, feature_cols, synthetic_ideal_points
        )
        hardest = detect_hardest_legislators(leg_acc)
        assert len(hardest) <= HARDEST_N
        assert len(hardest) > 0

    def test_sorted_by_accuracy_ascending(
        self, vote_features, trained_models, synthetic_ideal_points
    ):
        """Results are sorted worst-first (ascending accuracy)."""
        xgb = trained_models["models"]["XGBoost"]
        feature_cols = trained_models["feature_names"]
        leg_acc = compute_per_legislator_accuracy(
            xgb, vote_features, feature_cols, synthetic_ideal_points
        )
        hardest = detect_hardest_legislators(leg_acc)
        accuracies = [h.accuracy for h in hardest]
        assert accuracies == sorted(accuracies)

    def test_explanation_not_empty(self, vote_features, trained_models, synthetic_ideal_points):
        """Every result has a non-empty explanation string."""
        xgb = trained_models["models"]["XGBoost"]
        feature_cols = trained_models["feature_names"]
        leg_acc = compute_per_legislator_accuracy(
            xgb, vote_features, feature_cols, synthetic_ideal_points
        )
        hardest = detect_hardest_legislators(leg_acc)
        for h in hardest:
            assert isinstance(h.explanation, str)
            assert len(h.explanation) > 0

    def test_frozen_dataclass(self, vote_features, trained_models, synthetic_ideal_points):
        """HardestLegislator is immutable."""
        xgb = trained_models["models"]["XGBoost"]
        feature_cols = trained_models["feature_names"]
        leg_acc = compute_per_legislator_accuracy(
            xgb, vote_features, feature_cols, synthetic_ideal_points
        )
        hardest = detect_hardest_legislators(leg_acc)
        assert len(hardest) > 0
        with pytest.raises(AttributeError):
            hardest[0].accuracy = 0.99  # type: ignore[misc]

    def test_empty_dataframe_returns_empty(self):
        """Empty input DataFrame returns empty list."""
        empty_df = pl.DataFrame(
            {
                "legislator_slug": [],
                "full_name": [],
                "party": [],
                "xi_mean": [],
                "accuracy": [],
                "n_votes": [],
            },
            schema={
                "legislator_slug": pl.Utf8,
                "full_name": pl.Utf8,
                "party": pl.Utf8,
                "xi_mean": pl.Float64,
                "accuracy": pl.Float64,
                "n_votes": pl.Int64,
            },
        )
        hardest = detect_hardest_legislators(empty_df)
        assert hardest == []

    def test_moderate_gets_centrist_explanation(self):
        """A legislator with xi_mean near 0 (cross-party midpoint) gets moderate explanation."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["rep_moderate_1", "rep_r1_1", "rep_d1_1"],
                "full_name": ["Moderate Joe", "Strong R", "Strong D"],
                "party": ["Republican", "Republican", "Democrat"],
                "xi_mean": [0.1, 2.0, -2.0],
                "accuracy": [0.70, 0.95, 0.95],
                "n_votes": [50, 50, 50],
            }
        )
        hardest = detect_hardest_legislators(df, n=1)
        assert len(hardest) == 1
        assert "Moderate" in hardest[0].explanation or "Centrist" in hardest[0].explanation

    def test_null_full_name_uses_slug(self):
        """When full_name is null, slug is used instead (no crash)."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["rep_mystery_1", "rep_r1_1", "rep_d1_1"],
                "full_name": [None, "Strong R", "Strong D"],
                "party": ["Republican", "Republican", "Democrat"],
                "xi_mean": [1.0, 2.0, -2.0],
                "accuracy": [0.70, 0.95, 0.95],
                "n_votes": [50, 50, 50],
            }
        )
        hardest = detect_hardest_legislators(df, n=1)
        assert len(hardest) == 1
        assert hardest[0].full_name == "rep_mystery_1"
        assert isinstance(hardest[0].full_name, str)

    def test_single_party_no_crash(self):
        """Only one party present — midpoint falls back to 0.0, no crash."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["rep_r0_1", "rep_r1_1", "rep_r2_1"],
                "full_name": ["Alice", "Bob", "Carol"],
                "party": ["Republican", "Republican", "Republican"],
                "xi_mean": [1.0, 1.5, 2.0],
                "accuracy": [0.80, 0.90, 0.95],
                "n_votes": [50, 50, 50],
            }
        )
        hardest = detect_hardest_legislators(df, n=2)
        assert len(hardest) == 2
        for h in hardest:
            assert isinstance(h.explanation, str)
            assert len(h.explanation) > 0

    def test_custom_n_parameter(self):
        """Custom n limits output count."""
        df = pl.DataFrame(
            {
                "legislator_slug": [f"rep_x{i}_1" for i in range(10)],
                "full_name": [f"Legislator {i}" for i in range(10)],
                "party": ["Republican"] * 5 + ["Democrat"] * 5,
                "xi_mean": [2.0, 1.5, 1.0, 0.5, 0.1, -0.1, -0.5, -1.0, -1.5, -2.0],
                "accuracy": [0.7 + i * 0.02 for i in range(10)],
                "n_votes": [50] * 10,
            }
        )
        hardest_3 = detect_hardest_legislators(df, n=3)
        assert len(hardest_3) == 3
        hardest_1 = detect_hardest_legislators(df, n=1)
        assert len(hardest_1) == 1

    def test_strongly_conservative_explanation(self):
        """Republican with xi far above party median gets 'Strongly conservative' explanation."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["rep_extreme_1", "rep_r1_1", "rep_r2_1", "rep_d1_1"],
                "full_name": ["Extreme R", "Normal R1", "Normal R2", "Normal D"],
                "party": ["Republican", "Republican", "Republican", "Democrat"],
                "xi_mean": [4.0, 1.0, 1.2, -2.0],
                "accuracy": [0.70, 0.95, 0.95, 0.95],
                "n_votes": [50, 50, 50, 50],
            }
        )
        hardest = detect_hardest_legislators(df, n=1)
        assert len(hardest) == 1
        assert "conservative" in hardest[0].explanation.lower()

    def test_strongly_liberal_explanation(self):
        """Democrat with xi far below party median gets 'Strongly liberal' explanation."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["rep_extreme_1", "rep_d1_1", "rep_d2_1", "rep_r1_1"],
                "full_name": ["Extreme D", "Normal D1", "Normal D2", "Normal R"],
                "party": ["Democrat", "Democrat", "Democrat", "Republican"],
                "xi_mean": [-4.0, -1.0, -1.2, 2.0],
                "accuracy": [0.70, 0.95, 0.95, 0.95],
                "n_votes": [50, 50, 50, 50],
            }
        )
        hardest = detect_hardest_legislators(df, n=1)
        assert len(hardest) == 1
        assert "liberal" in hardest[0].explanation.lower()

    def test_fallback_explanation(self):
        """Legislator that doesn't match other categories gets fallback explanation."""
        # Democrat with xi slightly above party median (not extreme, not centrist, not moderate)
        df = pl.DataFrame(
            {
                "legislator_slug": ["rep_odd_1", "rep_r1_1", "rep_d1_1"],
                "full_name": ["Odd One", "Strong R", "Strong D"],
                "party": ["Democrat", "Republican", "Democrat"],
                "xi_mean": [-0.8, 2.0, -2.0],
                "accuracy": [0.70, 0.95, 0.95],
                "n_votes": [50, 50, 50],
            }
        )
        hardest = detect_hardest_legislators(df, n=1)
        assert len(hardest) == 1
        assert "doesn\u2019t fit" in hardest[0].explanation

    def test_null_xi_mean_no_crash(self):
        """When xi_mean is null, should not crash and should use 0.0 as default."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["rep_null_xi_1", "rep_r1_1", "rep_d1_1"],
                "full_name": ["Null Xi", "Strong R", "Strong D"],
                "party": ["Republican", "Republican", "Democrat"],
                "xi_mean": [None, 2.0, -2.0],
                "accuracy": [0.70, 0.95, 0.95],
                "n_votes": [50, 50, 50],
            },
            schema={
                "legislator_slug": pl.Utf8,
                "full_name": pl.Utf8,
                "party": pl.Utf8,
                "xi_mean": pl.Float64,
                "accuracy": pl.Float64,
                "n_votes": pl.Int64,
            },
        )
        hardest = detect_hardest_legislators(df, n=1)
        assert len(hardest) == 1
        assert hardest[0].xi_mean == 0.0

    def test_fields_populated_correctly(self):
        """All HardestLegislator fields match input data."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["rep_test_1", "rep_r1_1", "rep_d1_1"],
                "full_name": ["Test Person", "Strong R", "Strong D"],
                "party": ["Republican", "Republican", "Democrat"],
                "xi_mean": [1.5, 2.0, -2.0],
                "accuracy": [0.75, 0.95, 0.95],
                "n_votes": [100, 50, 50],
            }
        )
        hardest = detect_hardest_legislators(df, n=1)
        h = hardest[0]
        assert h.slug == "rep_test_1"
        assert h.full_name == "Test Person"
        assert h.party == "Republican"
        assert h.xi_mean == 1.5
        assert h.accuracy == 0.75
        assert h.n_votes == 100


# ── Tests: Evaluate Holdout ─────────────────────────────────────────────────


class TestEvaluateHoldout:
    """Tests for evaluate_holdout().

    Run: uv run pytest tests/test_prediction.py::TestEvaluateHoldout -v
    """

    def test_returns_all_models(self, trained_models):
        """Results include all three models."""
        results = evaluate_holdout(
            trained_models["models"],
            trained_models["X_test"],
            trained_models["y_test"],
        )
        assert len(results) == 3
        model_names = {r["model"] for r in results}
        assert model_names == {"Logistic Regression", "XGBoost", "Random Forest"}

    def test_metrics_in_range(self, trained_models):
        """All metrics are in valid ranges."""
        results = evaluate_holdout(
            trained_models["models"],
            trained_models["X_test"],
            trained_models["y_test"],
        )
        for r in results:
            assert 0 <= r["accuracy"] <= 1
            assert 0 <= r["auc"] <= 1
            assert 0 <= r["brier"] <= 1
            assert r["logloss"] >= 0

    def test_has_proper_scoring_rules(self, trained_models):
        """Holdout results include Brier score and log-loss."""
        results = evaluate_holdout(
            trained_models["models"],
            trained_models["X_test"],
            trained_models["y_test"],
        )
        for r in results:
            assert "brier" in r
            assert "logloss" in r


# ── Tests: Compute Day of Session ───────────────────────────────────────────


class TestComputeDayOfSession:
    """Tests for _compute_day_of_session().

    Run: uv run pytest tests/test_prediction.py::TestComputeDayOfSession -v
    """

    def test_yyyy_mm_dd_format(self):
        """Handles ISO date format (YYYY-MM-DD)."""
        dates = pl.Series(["2025-01-10", "2025-01-11", "2025-01-15"])
        result = _compute_day_of_session(dates)
        assert result.to_list() == [0, 1, 5]

    def test_mm_dd_yyyy_format(self):
        """Handles US date format (MM/DD/YYYY)."""
        dates = pl.Series(["01/10/2025", "01/11/2025", "01/15/2025"])
        result = _compute_day_of_session(dates)
        assert result.to_list() == [0, 1, 5]

    def test_single_date_is_day_zero(self):
        """A single date produces day 0."""
        dates = pl.Series(["2025-03-15"])
        result = _compute_day_of_session(dates)
        assert result.to_list() == [0]


# ── Tests: Baselines ────────────────────────────────────────────────────────


class TestBaselines:
    """Tests for baseline computations in train_vote_models().

    Run: uv run pytest tests/test_prediction.py::TestBaselines -v
    """

    def test_majority_baseline_above_50(self, trained_models):
        """Majority-class baseline should be ≥50% by definition."""
        baselines = trained_models["baselines"]
        assert baselines["majority_class_acc"] >= 0.5

    def test_party_baseline_above_majority(self, trained_models):
        """Party-only should beat majority-class on polarized data."""
        baselines = trained_models["baselines"]
        assert baselines["party_only_acc"] >= baselines["majority_class_acc"]

    def test_party_baseline_auc_above_random(self, trained_models):
        """Party-only AUC should beat random (0.5) on polarized data."""
        baselines = trained_models["baselines"]
        assert baselines["party_only_auc"] > 0.5


# ── Tests: CV Results Include Proper Scoring Rules ──────────────────────────


class TestProperScoringRules:
    """Tests that CV results include Brier score and log-loss.

    Run: uv run pytest tests/test_prediction.py::TestProperScoringRules -v
    """

    def test_cv_has_brier_and_logloss(self, trained_models):
        """Every CV fold includes Brier score and log-loss for all models."""
        for fold in trained_models["cv_results"]:
            for name in ["Logistic Regression", "XGBoost", "Random Forest"]:
                assert f"{name}_brier" in fold, f"Missing {name}_brier"
                assert f"{name}_logloss" in fold, f"Missing {name}_logloss"

    def test_brier_in_valid_range(self, trained_models):
        """Brier score should be in [0, 1]."""
        for fold in trained_models["cv_results"]:
            for name in ["Logistic Regression", "XGBoost", "Random Forest"]:
                brier = fold[f"{name}_brier"]
                assert 0 <= brier <= 1

    def test_logloss_is_positive(self, trained_models):
        """Log-loss should be non-negative."""
        for fold in trained_models["cv_results"]:
            for name in ["Logistic Regression", "XGBoost", "Random Forest"]:
                ll = fold[f"{name}_logloss"]
                assert ll >= 0


# ── Tests: Sponsor Party Feature ────────────────────────────────────────────


class TestSponsorFeature:
    """Tests for sponsor_party_R feature in build_bill_features().

    Run: uv run pytest tests/test_prediction.py::TestSponsorFeature -v
    """

    def test_slug_based_sponsor_party(
        self, synthetic_rollcalls, synthetic_bill_params, synthetic_legislators
    ):
        """When sponsor_slugs present, sponsor_party_R is derived from slug matching."""
        result = build_bill_features(
            synthetic_rollcalls,
            synthetic_bill_params,
            "House",
            legislators=synthetic_legislators,
        )
        assert "sponsor_party_R" in result.columns
        # Even-numbered bills have R sponsor (rep_r0_1), odd have D sponsor (rep_d0_1)
        vals = result["sponsor_party_R"].unique().sort().to_list()
        assert set(vals).issubset({0, 1})

    def test_text_fallback_when_no_slugs(self, synthetic_rollcalls, synthetic_bill_params):
        """When sponsor_slugs column is absent, falls back to text matching."""
        rc_no_slugs = synthetic_rollcalls.drop("sponsor_slugs")
        # Build legislators with full_name matching sponsor text format
        # Sponsor text is "Representative Republican0" / "Representative Democrat0"
        # match_sponsor_to_party extracts last name and matches against full_name
        legs = pl.DataFrame(
            {
                "legislator_slug": ["rep_r0_1", "rep_d0_1"],
                "full_name": ["Republican0", "Democrat0"],
                "chamber": ["House", "House"],
                "party": ["Republican", "Democrat"],
            }
        )
        result = build_bill_features(
            rc_no_slugs,
            synthetic_bill_params,
            "House",
            legislators=legs,
        )
        # Should still have the feature (via text fallback)
        assert "sponsor_party_R" in result.columns

    def test_committee_sponsor_is_zero(self, synthetic_bill_params, synthetic_legislators):
        """Committee-sponsored bills get sponsor_party_R=0."""
        rc = pl.DataFrame(
            {
                "session": ["2025-26"],
                "bill_number": ["HB 1"],
                "bill_title": ["Test"],
                "vote_id": ["je_20250300120000"],
                "vote_url": ["http://example.com"],
                "vote_datetime": ["2025-03-01T12:00:00"],
                "vote_date": ["2025-03-01"],
                "chamber": ["House"],
                "motion": ["Final Action"],
                "vote_type": ["Final Action"],
                "result": ["Passed"],
                "short_title": ["Tax bill"],
                "sponsor": ["Committee on Taxation"],
                "sponsor_slugs": [""],
                "yea_count": [15],
                "nay_count": [5],
                "present_passing_count": [0],
                "absent_not_voting_count": [0],
                "not_voting_count": [0],
                "total_votes": [20],
                "passed": [True],
            }
        )
        bp = synthetic_bill_params.head(1)
        result = build_bill_features(rc, bp, "House", legislators=synthetic_legislators)
        if "sponsor_party_R" in result.columns:
            assert result["sponsor_party_R"][0] == 0

    def test_backward_compat_no_legislators(self, synthetic_rollcalls, synthetic_bill_params):
        """Without legislators parameter, sponsor_party_R is absent (backward compatible)."""
        result = build_bill_features(synthetic_rollcalls, synthetic_bill_params, "House")
        assert "sponsor_party_R" not in result.columns


# ── Tests: Stratified Accuracy ──────────────────────────────────────────────


class TestStratifiedAccuracy:
    """Tests for compute_stratified_accuracy().

    Run: uv run pytest tests/test_prediction.py::TestStratifiedAccuracy -v
    """

    def test_all_prefixes_present(self):
        """Each unique prefix gets a row."""
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1])
        prefixes = ["HB", "HB", "SB", "SB", "SB"]
        result = compute_stratified_accuracy(y_true, y_pred, prefixes)
        assert result.height == 2
        assert set(result["prefix"].to_list()) == {"HB", "SB"}

    def test_accuracy_computation(self):
        """Accuracy is correct per prefix."""
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])
        prefixes = ["HB", "HB", "SB", "SB"]
        result = compute_stratified_accuracy(y_true, y_pred, prefixes)
        hb = result.filter(pl.col("prefix") == "HB")
        assert hb["accuracy"][0] == 1.0  # both correct
        sb = result.filter(pl.col("prefix") == "SB")
        assert sb["accuracy"][0] == 0.5  # 1 of 2 correct

    def test_passage_rate(self):
        """Passage rate is the mean of y_true per prefix."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 1])
        prefixes = ["HB", "HB", "SB", "SB"]
        result = compute_stratified_accuracy(y_true, y_pred, prefixes)
        for row in result.iter_rows(named=True):
            assert row["passage_rate"] == 0.5

    def test_sorted_by_count_desc(self):
        """Results sorted by count descending."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        prefixes = ["SB", "HB", "HB", "HB", "SB"]
        result = compute_stratified_accuracy(y_true, y_pred, prefixes)
        counts = result["count"].to_list()
        assert counts == sorted(counts, reverse=True)

    def test_single_prefix(self):
        """Single prefix produces one row."""
        y_true = np.array([1, 1, 0])
        y_pred = np.array([1, 0, 0])
        prefixes = ["HB", "HB", "HB"]
        result = compute_stratified_accuracy(y_true, y_pred, prefixes)
        assert result.height == 1
        assert result["prefix"][0] == "HB"
        assert result["count"][0] == 3


# ── Tests: Bill Features Metadata ──────────────────────────────────────────


class TestBillFeaturesMetadata:
    """Tests for bill_prefix in build_bill_features() metadata.

    Run: uv run pytest tests/test_prediction.py::TestBillFeaturesMetadata -v
    """

    def test_bill_prefix_in_output(self, synthetic_rollcalls, synthetic_bill_params):
        """bill_prefix should be in the output metadata columns."""
        result = build_bill_features(synthetic_rollcalls, synthetic_bill_params, "House")
        assert "bill_prefix" in result.columns


# ── Tests: Test Indices ─────────────────────────────────────────────────────


class TestTestIndices:
    """Tests that train_vote_models returns test_indices for holdout evaluation.

    Run: uv run pytest tests/test_prediction.py::TestTestIndices -v
    """

    def test_test_indices_present(self, trained_models):
        """train_vote_models returns test_indices array."""
        assert "test_indices" in trained_models
        assert len(trained_models["test_indices"]) > 0

    def test_test_indices_match_test_size(self, trained_models, vote_features):
        """test_indices length matches TEST_SIZE proportion."""
        n_test = len(trained_models["test_indices"])
        n_total = vote_features.height
        ratio = n_test / n_total
        # Should be approximately TEST_SIZE (0.20) ± 0.05
        assert 0.15 <= ratio <= 0.25


# ── Tests: Surprising Votes Empty Schema ────────────────────────────────────


class TestSurprisingVotesEmptySchema:
    """Tests that find_surprising_votes returns typed empty DataFrame.

    Run: uv run pytest tests/test_prediction.py::TestSurprisingVotesEmptySchema -v
    """

    def test_empty_schema_has_columns(self):
        """A perfect model produces empty DataFrame with proper column schema."""

        class PerfectModel:
            def predict_proba(self, X):
                probs = np.zeros((len(X), 2))
                probs[:, 1] = (X[:, 0] > 0).astype(float)
                return probs

            def predict(self, X):
                return (X[:, 0] > 0).astype(int)

        features_df = pl.DataFrame(
            {
                "vote_id": ["v1", "v2"],
                "legislator_slug": ["rep_a_1", "rep_b_1"],
                "vote_binary": [1, 0],
                "feat1": [1.0, -1.0],
            }
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["v1", "v2"],
                "bill_number": ["HB1", "HB2"],
                "motion": ["Final Action", "Final Action"],
            }
        )
        ideal_points = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_1", "rep_b_1"],
                "full_name": ["Alice", "Bob"],
                "party": ["Republican", "Democrat"],
            }
        )

        result = find_surprising_votes(
            PerfectModel(), features_df, ["feat1"], rollcalls, ideal_points
        )
        assert result.height == 0
        assert len(result.columns) > 0, "Empty DataFrame should have column schema"
        assert "vote_id" in result.columns
        assert "confidence_error" in result.columns


# ── Tests: Temporal Split ───────────────────────────────────────────────────


class TestTemporalSplit:
    """Tests for _temporal_split_eval().

    Run: uv run pytest tests/test_prediction.py::TestTemporalSplit -v
    """

    def test_returns_results(self, synthetic_rollcalls, synthetic_bill_params):
        """Temporal split returns results for all 3 models."""
        features = build_bill_features(synthetic_rollcalls, synthetic_bill_params, "House")
        feature_cols = _get_feature_cols(
            features, "passed_binary", ["vote_id", "bill_number", "vote_date", "bill_prefix"]
        )
        results = _temporal_split_eval(features, feature_cols, "House")
        if results:  # May be empty if too few rows
            assert len(results) == 3
            assert all("train_size" in r for r in results)
            assert all("test_size" in r for r in results)
            assert all("brier" in r for r in results)
            assert all("logloss" in r for r in results)
