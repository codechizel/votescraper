"""Tests for NLP topic features module.

Validates TF-IDF + NMF topic modeling on synthetic bill title text.

Usage:
    uv run pytest tests/test_nlp_features.py -v
"""

from __future__ import annotations

import polars as pl
import pytest

from analysis.nlp_features import (
    NMF_N_TOPICS,
    TopicModel,
    fit_topic_features,
    get_topic_display_names,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def topic_corpus() -> pl.Series:
    """30 bill titles with 3 distinguishable topic clusters."""
    titles = [
        # Cluster 1: elections/voting (10 docs)
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
        # Cluster 2: taxes/revenue (10 docs)
        "income tax rate reduction act",
        "property tax exemption for veterans",
        "sales tax on digital goods and services",
        "tax credit for small business investment",
        "property tax assessment reform",
        "corporate income tax rate adjustment",
        "sales tax exemption for food items",
        "tax increment financing authorization",
        "estate tax threshold modification",
        "revenue neutral tax reform package",
        # Cluster 3: healthcare/education (10 docs)
        "medicaid expansion and coverage update",
        "school funding formula revision",
        "healthcare facility licensing standards",
        "public education accountability measures",
        "mental health services funding increase",
        "student loan assistance program creation",
        "hospital transparency reporting requirements",
        "teacher compensation and benefits reform",
        "prescription drug cost reduction act",
        "early childhood education funding",
    ]
    return pl.Series("short_title", titles)


@pytest.fixture
def fitted_result(topic_corpus: pl.Series) -> tuple[pl.DataFrame, TopicModel]:
    """Pre-fitted topic features for reuse across tests."""
    return fit_topic_features(topic_corpus)


# ── Tests: Fit Topic Features ────────────────────────────────────────────────


class TestFitTopicFeatures:
    """Tests for the core fit_topic_features() function."""

    def test_returns_correct_column_count(self, fitted_result):
        """Topic DataFrame should have exactly NMF_N_TOPICS columns."""
        topic_df, _ = fitted_result
        assert topic_df.width == NMF_N_TOPICS

    def test_column_names_follow_pattern(self, fitted_result):
        """Columns should be named topic_0 through topic_{N-1}."""
        topic_df, _ = fitted_result
        expected = [f"topic_{i}" for i in range(NMF_N_TOPICS)]
        assert topic_df.columns == expected

    def test_row_count_matches_input(self, topic_corpus, fitted_result):
        """Output rows must match input document count."""
        topic_df, _ = fitted_result
        assert topic_df.height == topic_corpus.len()

    def test_values_are_non_negative(self, fitted_result):
        """NMF produces non-negative weights."""
        topic_df, _ = fitted_result
        for col in topic_df.columns:
            assert (topic_df[col] >= 0).all(), f"Negative value in {col}"

    def test_metadata_fields_populated(self, fitted_result):
        """TopicModel metadata should be fully populated."""
        _, model = fitted_result
        assert model.n_topics > 0
        assert model.n_topics <= NMF_N_TOPICS
        assert model.vocabulary_size > 0
        assert model.n_documents == 30
        assert len(model.topic_labels) == NMF_N_TOPICS
        assert len(model.topic_top_words) == NMF_N_TOPICS

    def test_top_words_are_nonempty(self, fitted_result):
        """Active topics should have non-empty top words lists."""
        _, model = fitted_result
        for i in range(model.n_topics):
            col = f"topic_{i}"
            assert len(model.topic_top_words[col]) > 0, f"Empty top words for {col}"

    def test_reproducible(self, topic_corpus):
        """Two calls with the same input should produce identical results."""
        df1, m1 = fit_topic_features(topic_corpus)
        df2, m2 = fit_topic_features(topic_corpus)
        assert df1.equals(df2)
        assert m1.topic_labels == m2.topic_labels


# ── Tests: Edge Cases ────────────────────────────────────────────────────────


class TestHandleEdgeCases:
    """Tests for defensive handling of degenerate input."""

    def test_null_values_dont_crash(self):
        """Null entries should be handled gracefully."""
        vals = ["tax reform", None, "election day", None, "school funding"]
        texts = pl.Series("short_title", vals)
        topic_df, model = fit_topic_features(texts)
        assert topic_df.height == 5
        assert model.n_documents == 5

    def test_empty_strings_dont_crash(self):
        """Empty strings should be handled gracefully."""
        texts = pl.Series("short_title", ["tax reform", "", "election day", "", "school funding"])
        topic_df, model = fit_topic_features(texts)
        assert topic_df.height == 5

    def test_minimal_corpus(self):
        """A corpus with very few documents should not crash."""
        texts = pl.Series("short_title", ["tax reform", "election day"])
        topic_df, model = fit_topic_features(texts)
        assert topic_df.height == 2
        assert model.n_topics >= 1

    def test_all_identical_texts(self):
        """All identical documents should not crash."""
        texts = pl.Series("short_title", ["same text"] * 10)
        topic_df, model = fit_topic_features(texts)
        assert topic_df.height == 10


# ── Tests: Display Names ────────────────────────────────────────────────────


class TestGetTopicDisplayNames:
    """Tests for the get_topic_display_names() function."""

    def test_returns_dict(self, fitted_result):
        _, model = fitted_result
        result = get_topic_display_names(model)
        assert isinstance(result, dict)

    def test_keys_match_column_names(self, fitted_result):
        _, model = fitted_result
        result = get_topic_display_names(model)
        expected_keys = {f"topic_{i}" for i in range(NMF_N_TOPICS)}
        assert set(result.keys()) == expected_keys

    def test_values_have_topic_prefix(self, fitted_result):
        _, model = fitted_result
        result = get_topic_display_names(model)
        for key, value in result.items():
            assert value.startswith("Topic:"), f"{key} label doesn't start with 'Topic:'"


# ── Tests: TopicModel Dataclass ──────────────────────────────────────────────


class TestTopicModel:
    """Tests for the TopicModel frozen dataclass."""

    def test_is_frozen(self, fitted_result):
        """TopicModel should be immutable."""
        _, model = fitted_result
        with pytest.raises(AttributeError):
            model.n_topics = 99

    def test_feature_names_match_columns(self, fitted_result):
        """feature_names should match the DataFrame column names."""
        topic_df, model = fitted_result
        assert model.feature_names == topic_df.columns
