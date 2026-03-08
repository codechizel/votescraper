"""Tests for Phase 18: Bill Text Analysis.

Covers data loading, preprocessing, embedding cache, topic modeling (mocked),
CAP classification (mocked), similarity, vote cross-reference, and report
builder.

Run: uv run pytest tests/test_bill_text.py -v
"""

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.bill_text import (
    MIN_CLUSTER_SIZE,
    RANDOM_SEED,
    SIMILARITY_THRESHOLD,
    compute_bill_similarity,
    compute_cap_passage_rates,
    compute_topic_party_cohesion,
    extract_topic_info,
    plot_cap_distribution,
    plot_caucus_splitting,
    plot_similarity_heatmap,
    plot_topic_distribution,
    plot_topic_party_heatmap,
    save_filtering_manifest,
)
from analysis.bill_text_classify import (
    CAP_CATEGORIES,
    CAP_CATEGORY_LABELS,
    _content_hash,
    _load_cache,
    _parse_response,
    _save_cache,
    classify_bills_cap,
)
from analysis.bill_text_data import (
    DEFAULT_EMBEDDING_MODEL,
    MAX_TOKENS_APPROX,
    _compute_cache_key,
    assign_bills_to_chambers,
    load_bill_texts,
    preprocess_for_embedding,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_bill_texts_csv(tmp_path: Path) -> Path:
    """Create a minimal bill_texts.csv for testing."""
    data_dir = tmp_path / "91st_2025-2026"
    data_dir.mkdir()
    csv_path = data_dir / "91st_2025-2026_bill_texts.csv"
    csv_path.write_text(
        textwrap.dedent("""\
        session,bill_number,document_type,version,text,page_count,source_url
        2025,HB 2001,introduced,1,An act concerning taxation and property valuation,3,http://example.com/hb2001
        2025,HB 2001,supp_note,1,This bill modifies property tax valuations for counties,1,http://example.com/hb2001_sn
        2025,HB 2002,introduced,1,An act relating to education funding and K-12 schools,5,http://example.com/hb2002
        2025,SB 100,introduced,1,An act concerning criminal justice sentencing reform,4,http://example.com/sb100
        2025,SB 101,supp_note,1,This bill establishes new healthcare regulations,2,http://example.com/sb101
    """)
    )
    return data_dir


# ── bill_text_data tests ─────────────────────────────────────────────────────


class TestLoadBillTexts:
    def test_loads_csv(self, sample_bill_texts_csv: Path) -> None:
        df = load_bill_texts(sample_bill_texts_csv, use_csv=True)
        assert len(df) > 0
        assert "bill_number" in df.columns
        assert "text" in df.columns
        assert "text_source" in df.columns

    def test_prefers_supp_note(self, sample_bill_texts_csv: Path) -> None:
        df = load_bill_texts(sample_bill_texts_csv, use_csv=True)
        hb2001 = df.filter(pl.col("bill_number") == "HB 2001")
        assert len(hb2001) == 1
        assert hb2001["text_source"][0] == "supp_note"

    def test_deduplicates_per_bill(self, sample_bill_texts_csv: Path) -> None:
        df = load_bill_texts(sample_bill_texts_csv, use_csv=True)
        assert df["bill_number"].n_unique() == len(df)

    def test_raises_on_missing_csv(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "missing_dir"
        data_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="bill_texts"):
            load_bill_texts(data_dir, use_csv=True)


class TestPreprocessForEmbedding:
    def test_strips_enacting_clause(self) -> None:
        text = "Be it enacted by the Legislature of the State of Kansas: Some actual content."
        result = preprocess_for_embedding(text)
        assert "enacted" not in result.lower()
        assert "actual content" in result

    def test_normalizes_ksa_references(self) -> None:
        text = "Amending K.S.A. 12-4516 and K.S.A. 79-32,117."
        result = preprocess_for_embedding(text)
        assert "STATUTE_REF" in result

    def test_truncates_long_text(self) -> None:
        long_text = "word " * 10000
        result = preprocess_for_embedding(long_text)
        assert len(result) <= MAX_TOKENS_APPROX

    def test_handles_empty_string(self) -> None:
        assert preprocess_for_embedding("") == ""

    def test_strips_severability(self) -> None:
        text = (
            "Main content. If any provision of this act is held invalid, "
            "the remaining provisions shall remain in effect."
        )
        result = preprocess_for_embedding(text)
        assert "Main content" in result

    def test_collapses_whitespace(self) -> None:
        text = "word1     word2      word3"
        result = preprocess_for_embedding(text)
        assert "     " not in result


class TestComputeCacheKey:
    def test_deterministic(self) -> None:
        key1 = _compute_cache_key("model", ["bill1", "bill2"], ["text1", "text2"])
        key2 = _compute_cache_key("model", ["bill1", "bill2"], ["text1", "text2"])
        assert key1 == key2

    def test_different_model_different_key(self) -> None:
        key1 = _compute_cache_key("model_a", ["bill1"], ["text1"])
        key2 = _compute_cache_key("model_b", ["bill1"], ["text1"])
        assert key1 != key2

    def test_different_text_different_key(self) -> None:
        key1 = _compute_cache_key("model", ["bill1"], ["text_v1"])
        key2 = _compute_cache_key("model", ["bill1"], ["text_v2"])
        assert key1 != key2

    def test_order_independent(self) -> None:
        key1 = _compute_cache_key("model", ["bill1", "bill2"], ["text1", "text2"])
        key2 = _compute_cache_key("model", ["bill2", "bill1"], ["text2", "text1"])
        assert key1 == key2


class TestAssignBillsToChambers:
    def test_assigns_by_rollcall_chamber(self) -> None:
        bill_texts = pl.DataFrame(
            {
                "bill_number": ["HB 1", "SB 1"],
                "text": ["house bill", "senate bill"],
            }
        )
        rollcalls = pl.DataFrame(
            {
                "bill_number": ["HB 1", "SB 1"],
                "chamber": ["House", "Senate"],
                "vote_id": ["v1", "v2"],
            }
        )
        result = assign_bills_to_chambers(bill_texts, rollcalls)
        assert "house" in result
        assert "senate" in result
        assert "all" in result
        assert len(result["house"]) == 1
        assert result["house"]["bill_number"][0] == "HB 1"

    def test_all_key_always_present(self) -> None:
        bill_texts = pl.DataFrame(
            {
                "bill_number": ["HB 1"],
                "text": ["content"],
            }
        )
        rollcalls = pl.DataFrame(
            {
                "bill_number": ["HB 1"],
                "chamber": ["House"],
                "vote_id": ["v1"],
            }
        )
        result = assign_bills_to_chambers(bill_texts, rollcalls)
        assert "all" in result
        assert len(result["all"]) == 1


class TestGetOrComputeEmbeddings:
    @patch("fastembed.TextEmbedding")
    def test_computes_and_caches(self, mock_embed_cls: MagicMock, tmp_path: Path) -> None:
        from analysis.bill_text_data import get_or_compute_embeddings

        mock_model = MagicMock()
        mock_model.embed.return_value = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        mock_embed_cls.return_value = mock_model

        result = get_or_compute_embeddings(
            ["text1", "text2"],
            ["bill1", "bill2"],
            cache_dir=tmp_path,
        )

        assert result.shape == (2, 3)
        parquet_files = list(tmp_path.glob("embeddings_*.parquet"))
        assert len(parquet_files) == 1

    @patch("fastembed.TextEmbedding")
    def test_reuses_cache(self, mock_embed_cls: MagicMock, tmp_path: Path) -> None:
        from analysis.bill_text_data import get_or_compute_embeddings

        mock_model = MagicMock()
        mock_model.embed.return_value = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        mock_embed_cls.return_value = mock_model

        get_or_compute_embeddings(
            ["text1", "text2"],
            ["bill1", "bill2"],
            cache_dir=tmp_path,
        )

        mock_model.embed.reset_mock()
        mock_embed_cls.reset_mock()

        result = get_or_compute_embeddings(
            ["text1", "text2"],
            ["bill1", "bill2"],
            cache_dir=tmp_path,
        )

        assert result.shape == (2, 2)
        mock_embed_cls.assert_not_called()


# ── bill_text_classify tests ─────────────────────────────────────────────────


class TestCAPCategories:
    def test_has_20_categories(self) -> None:
        assert len(CAP_CATEGORIES) == 20

    def test_all_categories_have_labels(self) -> None:
        for key in CAP_CATEGORIES:
            assert key in CAP_CATEGORY_LABELS


class TestContentHash:
    def test_deterministic(self) -> None:
        h1 = _content_hash("text", "model")
        h2 = _content_hash("text", "model")
        assert h1 == h2

    def test_different_model_different_hash(self) -> None:
        h1 = _content_hash("text", "model_a")
        h2 = _content_hash("text", "model_b")
        assert h1 != h2


class TestParseResponse:
    def test_parses_clean_json(self) -> None:
        result = _parse_response('{"primary_category": "education", "confidence": 4}')
        assert result["primary_category"] == "education"

    def test_strips_markdown_fences(self) -> None:
        result = _parse_response('```json\n{"primary_category": "health", "confidence": 3}\n```')
        assert result["primary_category"] == "health"

    def test_handles_malformed_json(self) -> None:
        result = _parse_response("not json at all")
        assert result["primary_category"] == "government_operations"
        assert result["confidence"] == 1


class TestClassifyBillsCap:
    def test_skips_without_api_key(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = classify_bills_cap(
                ["text1"],
                ["bill1"],
                cache_path=tmp_path / "cache.json",
            )
        assert len(result) == 0

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_uses_cache(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"

        key = _content_hash("text1", "claude-sonnet-4-5-20241022")
        cache = {
            key: {
                "primary_category": "education",
                "confidence": 5,
                "top3": ["education", "labor", "social_welfare"],
                "summary": "An education bill.",
            }
        }
        cache_path.write_text(json.dumps(cache))

        result = classify_bills_cap(
            ["text1"],
            ["bill1"],
            cache_path=cache_path,
        )
        assert len(result) == 1
        assert result["cap_category"][0] == "education"


class TestCacheOperations:
    def test_load_empty_cache(self, tmp_path: Path) -> None:
        assert _load_cache(tmp_path / "nonexistent.json") == {}

    def test_save_and_load_cache(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        _save_cache(cache_path, {"key": {"value": 42}})
        loaded = _load_cache(cache_path)
        assert loaded["key"]["value"] == 42


# ── bill_text.py tests (core functions) ──────────────────────────────────────


class TestComputeBillSimilarity:
    def test_identity_similarity(self) -> None:
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        sim_matrix, pairs = compute_bill_similarity(
            embeddings,
            ["A", "B", "C"],
            threshold=0.99,
        )
        assert sim_matrix.shape == (3, 3)
        assert any(p["bill_a"] == "A" and p["bill_b"] == "C" for p in pairs)

    def test_threshold_filtering(self) -> None:
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        _, pairs = compute_bill_similarity(
            embeddings,
            ["A", "B"],
            threshold=0.5,
        )
        assert len(pairs) == 0

    def test_top_n_limit(self) -> None:
        embeddings = np.ones((5, 3))
        _, pairs = compute_bill_similarity(
            embeddings,
            [f"B{i}" for i in range(5)],
            threshold=0.5,
            top_n=3,
        )
        assert len(pairs) <= 3


class TestComputeTopicPartyCohesion:
    def test_computes_rice_index(self) -> None:
        bill_topics = pl.DataFrame(
            {
                "bill_number": ["HB 1", "HB 2"],
                "topic_id": [0, 0],
                "topic_label": ["Topic 0: tax", "Topic 0: tax"],
            }
        )
        rollcalls = pl.DataFrame(
            {
                "bill_number": ["HB 1", "HB 2"],
                "vote_id": ["v1", "v2"],
                "chamber": ["House", "House"],
            }
        )
        votes = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_1", "rep_a_1", "rep_b_1", "rep_b_1"],
                "vote_id": ["v1", "v2", "v1", "v2"],
                "vote": ["Yea", "Yea", "Yea", "Nay"],
            }
        )
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_1", "rep_b_1"],
                "party": ["Republican", "Republican"],
            }
        )

        cohesion, splitting = compute_topic_party_cohesion(
            bill_topics,
            rollcalls,
            votes,
            legislators,
        )
        assert len(cohesion) > 0
        assert len(splitting) > 0
        assert splitting[0]["majority_rice"] < 1.0

    def test_handles_empty_merge(self) -> None:
        bill_topics = pl.DataFrame(
            {
                "bill_number": ["HB 999"],
                "topic_id": [0],
                "topic_label": ["Topic"],
            }
        )
        rollcalls = pl.DataFrame(
            {
                "bill_number": ["HB 1"],
                "vote_id": ["v1"],
            }
        )
        votes = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_1"],
                "vote_id": ["v1"],
                "vote": ["Yea"],
            }
        )
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_1"],
                "party": ["Republican"],
            }
        )

        cohesion, splitting = compute_topic_party_cohesion(
            bill_topics,
            rollcalls,
            votes,
            legislators,
        )
        assert len(splitting) == 0


class TestComputeCapPassageRates:
    def test_computes_rates(self) -> None:
        cap_df = pl.DataFrame(
            {
                "bill_number": ["HB 1", "HB 2", "SB 1"],
                "cap_label": ["Education", "Education", "Health"],
            }
        )
        rollcalls = pl.DataFrame(
            {
                "bill_number": ["HB 1", "HB 2", "SB 1"],
                "vote_id": ["v1", "v2", "v3"],
                "passed": [True, False, True],
            }
        )

        result = compute_cap_passage_rates(cap_df, rollcalls)
        assert len(result) == 2
        assert "passage_rate" in result.columns

    def test_handles_empty(self) -> None:
        result = compute_cap_passage_rates(pl.DataFrame(), pl.DataFrame())
        assert len(result) == 0


# ── Topic model tests (mocked) ──────────────────────────────────────────────


class TestExtractTopicInfo:
    def test_extracts_assignments(self) -> None:
        mock_model = MagicMock()
        import pandas as pd

        mock_model.get_topic_info.return_value = pd.DataFrame(
            {
                "Topic": [-1, 0, 1],
                "Count": [5, 30, 15],
                "Name": ["noise", "topic_0", "topic_1"],
            }
        )
        mock_model.get_topic.side_effect = lambda tid: (
            [
                ("tax", 0.5),
                ("property", 0.3),
                ("county", 0.2),
                ("valuation", 0.1),
                ("assessment", 0.05),
            ]
            if tid == 0
            else [
                ("education", 0.6),
                ("school", 0.3),
                ("k12", 0.2),
                ("funding", 0.1),
                ("teacher", 0.05),
            ]
        )

        bill_topics, topic_info = extract_topic_info(
            mock_model,
            [0, 1, -1],
            ["HB 1", "HB 2", "HB 3"],
        )

        assert len(bill_topics) == 3
        assert len(topic_info) == 3
        assert topic_info[0]["topic_id"] == -1
        assert "Noise" in topic_info[0]["topic_label"]


# ── Report builder tests ─────────────────────────────────────────────────────


class TestBuildBillTextReport:
    def test_builds_without_cap(self, tmp_path: Path) -> None:
        from analysis.bill_text_report import build_bill_text_report

        from analysis.report import ReportBuilder

        report = ReportBuilder(title="Test", session="2025-2026")
        results = {
            "n_bills_analyzed": 100,
            "n_topics": 8,
            "text_source_counts": {"supp_note": 60, "introduced": 40},
            "avg_text_length": 5000,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_dim": 384,
            "topic_info": [
                {
                    "topic_id": 0,
                    "topic_label": "Topic 0: tax",
                    "count": 30,
                    "top_words": "tax, property",
                },
                {
                    "topic_id": 1,
                    "topic_label": "Topic 1: edu",
                    "count": 20,
                    "top_words": "education, school",
                },
            ],
            "caucus_splitting": [
                {
                    "topic_id": 0,
                    "topic_label": "Topic 0: tax",
                    "majority_rice": 0.7,
                    "minority_rice": 0.9,
                    "split_score": 0.3,
                    "n_bills": 15,
                }
            ],
            "top_similar_pairs": [
                {"bill_a": "HB 1", "bill_b": "HB 2", "similarity": 0.95},
            ],
            "parameters": {"min_cluster_size": 15},
        }

        build_bill_text_report(
            report,
            results=results,
            plots_dir=tmp_path,
            has_cap=False,
        )
        assert len(report._sections) > 0

    def test_builds_with_cap(self, tmp_path: Path) -> None:
        from analysis.bill_text_report import build_bill_text_report

        from analysis.report import ReportBuilder

        report = ReportBuilder(title="Test", session="2025-2026")
        cap_df = pl.DataFrame(
            {
                "bill_number": ["HB 1", "HB 2"],
                "cap_category": ["education", "health"],
                "cap_label": ["Education", "Health"],
                "cap_confidence": [4, 3],
                "cap_top3": ["education; labor", "health; social_welfare"],
                "bill_summary": ["An education bill.", "A health bill."],
            }
        )
        results = {
            "n_bills_analyzed": 50,
            "n_topics": 5,
            "text_source_counts": {"supp_note": 30, "introduced": 20},
            "avg_text_length": 4000,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_dim": 384,
            "topic_info": [],
            "caucus_splitting": [],
            "top_similar_pairs": [],
            "cap_classifications": cap_df,
            "cap_passage_rates": None,
            "parameters": {"min_cluster_size": 15},
        }

        build_bill_text_report(
            report,
            results=results,
            plots_dir=tmp_path,
            has_cap=True,
        )
        assert len(report._sections) > 0


# ── Plotting tests (smoke tests) ────────────────────────────────────────────


class TestPlotting:
    def test_plot_topic_distribution(self, tmp_path: Path) -> None:
        topic_info = [
            {"topic_id": 0, "topic_label": "Topic 0: tax", "count": 30, "top_words": "tax"},
            {"topic_id": 1, "topic_label": "Topic 1: edu", "count": 20, "top_words": "edu"},
            {"topic_id": -1, "topic_label": "Noise", "count": 5, "top_words": ""},
        ]
        plot_topic_distribution(topic_info, tmp_path, "all")
        assert (tmp_path / "topic_distribution_all.png").exists()

    def test_plot_similarity_heatmap(self, tmp_path: Path) -> None:
        embeddings = np.random.default_rng(42).random((10, 5))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / norms
        sim = normed @ normed.T
        plot_similarity_heatmap(sim, [f"B{i}" for i in range(10)], tmp_path, top_n=8)
        assert (tmp_path / "bill_similarity_heatmap.png").exists()

    def test_plot_topic_party_heatmap(self, tmp_path: Path) -> None:
        cohesion = pl.DataFrame(
            {
                "topic_id": [0, 0, 1, 1],
                "topic_label": ["Topic 0: tax", "Topic 0: tax", "Topic 1: edu", "Topic 1: edu"],
                "party": ["Republican", "Democrat", "Republican", "Democrat"],
                "rice_index": [0.8, 0.9, 0.6, 0.95],
            }
        )
        plot_topic_party_heatmap(cohesion, tmp_path, "house")
        assert (tmp_path / "topic_party_heatmap_house.png").exists()

    def test_plot_cap_distribution(self, tmp_path: Path) -> None:
        cap_df = pl.DataFrame(
            {
                "bill_number": ["HB 1", "HB 2", "HB 3"],
                "cap_label": ["Education", "Education", "Health"],
            }
        )
        plot_cap_distribution(cap_df, tmp_path)
        assert (tmp_path / "cap_category_distribution.png").exists()

    def test_plot_caucus_splitting(self, tmp_path: Path) -> None:
        data = [
            {"topic_label": f"Topic {i}: words", "split_score": 0.5 - i * 0.05, "n_bills": 10}
            for i in range(5)
        ]
        plot_caucus_splitting(data, tmp_path, "senate")
        assert (tmp_path / "caucus_splitting_topics_senate.png").exists()


# ── Constants consistency ────────────────────────────────────────────────────


class TestConstants:
    def test_min_cluster_size_positive(self) -> None:
        assert MIN_CLUSTER_SIZE > 0

    def test_random_seed_set(self) -> None:
        assert RANDOM_SEED == 42

    def test_similarity_threshold_in_range(self) -> None:
        assert 0 < SIMILARITY_THRESHOLD < 1

    def test_default_embedding_model(self) -> None:
        assert "bge" in DEFAULT_EMBEDDING_MODEL.lower()

    def test_max_tokens_reasonable(self) -> None:
        assert 1000 < MAX_TOKENS_APPROX < 50000


# ── CLI arg parsing ──────────────────────────────────────────────────────────


class TestParseArgs:
    def test_defaults(self) -> None:
        from analysis.bill_text import parse_args

        with patch("sys.argv", ["bill_text.py"]):
            args = parse_args()
        assert args.session == "2025-26"
        assert args.classify is False
        assert args.batch is False
        assert args.min_cluster_size == 15

    def test_classify_flag(self) -> None:
        from analysis.bill_text import parse_args

        with patch("sys.argv", ["bill_text.py", "--classify"]):
            args = parse_args()
        assert args.classify is True

    def test_custom_cluster_size(self) -> None:
        from analysis.bill_text import parse_args

        with patch("sys.argv", ["bill_text.py", "--min-cluster-size", "10"]):
            args = parse_args()
        assert args.min_cluster_size == 10


# ── Filtering manifest ───────────────────────────────────────────────────────


class TestFilteringManifest:
    def test_saves_manifest(self, tmp_path: Path) -> None:
        save_filtering_manifest(
            {"phase": "20_bill_text", "n_bills": 100},
            tmp_path,
        )
        path = tmp_path / "filtering_manifest.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["phase"] == "20_bill_text"
