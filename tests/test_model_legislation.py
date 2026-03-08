"""Tests for Phase 20: Model Legislation Detection.

Covers ALEC scraper (mocked), OpenStates adapter (mocked), similarity computation,
match classification, n-gram overlap, match summary, and report builder.

Run: uv run pytest tests/test_model_legislation.py -v
"""

import csv
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── ALEC Scraper Tests ──────────────────────────────────────────────────────

pytestmark = pytest.mark.scraper


class TestALECModels:
    """ALEC model bill dataclass tests."""

    def test_frozen(self):
        from tallgrass.alec.models import ALECModelBill

        bill = ALECModelBill(
            title="Test Act",
            text="Be it enacted...",
            category="Criminal Justice",
            bill_type="Model Policy",
            date="2020-01-15",
            url="https://alec.org/model-policy/test-act/",
            task_force="Public Safety",
        )
        assert bill.title == "Test Act"
        with pytest.raises(AttributeError):
            bill.title = "Changed"  # type: ignore[misc]

    def test_fields(self):
        from tallgrass.alec.models import ALECModelBill

        bill = ALECModelBill(
            title="A",
            text="B",
            category="C",
            bill_type="D",
            date="",
            url="http://x",
            task_force="E",
        )
        assert bill.text == "B"
        assert bill.date == ""


class TestALECOutput:
    """ALEC CSV export tests."""

    def test_save_csv(self, tmp_path: Path):
        from tallgrass.alec.models import ALECModelBill
        from tallgrass.alec.output import FIELDNAMES, save_alec_bills

        bills = [
            ALECModelBill(
                title="Voter ID Act",
                text="Section 1. All voters must present...",
                category="Elections",
                bill_type="Model Policy",
                date="2009-12-04",
                url="https://alec.org/model-policy/voter-id-act/",
                task_force="Elections",
            ),
            ALECModelBill(
                title="Right to Work Act",
                text="Section 1. No person shall be required...",
                category="Labor",
                bill_type="Model Policy",
                date="",
                url="https://alec.org/model-policy/right-to-work/",
                task_force="Commerce",
            ),
        ]

        csv_path = save_alec_bills(tmp_path, bills)
        assert csv_path.exists()
        assert csv_path.name == "alec_model_bills.csv"

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["title"] == "Voter ID Act"
        assert rows[1]["category"] == "Labor"
        assert set(reader.fieldnames) == set(FIELDNAMES)  # type: ignore[arg-type]

    def test_save_empty(self, tmp_path: Path):
        from tallgrass.alec.output import save_alec_bills

        csv_path = save_alec_bills(tmp_path, [])
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 0


class TestALECScraper:
    """ALEC scraper function tests (mocked HTTP)."""

    def test_extract_bill_text_entry_content(self):
        from tallgrass.alec.scraper import _extract_bill_text

        html = """
        <html><body>
        <nav>Navigation</nav>
        <div class="entry-content">
            <p>Section 1. This model policy establishes...</p>
            <p>Section 2. The state shall implement...</p>
        </div>
        <aside>Sidebar</aside>
        </body></html>
        """
        text = _extract_bill_text(html)
        assert "Section 1" in text
        assert "Section 2" in text
        assert "Navigation" not in text
        assert "Sidebar" not in text

    def test_extract_bill_text_article_fallback(self):
        from tallgrass.alec.scraper import _extract_bill_text

        html = """
        <html><body>
        <article>
            <p>Model legislation content here.</p>
        </article>
        </body></html>
        """
        text = _extract_bill_text(html)
        assert "Model legislation content" in text

    def test_extract_bill_text_empty(self):
        from tallgrass.alec.scraper import _extract_bill_text

        text = _extract_bill_text("")
        assert text == ""

    def test_parse_listing_entry(self):
        from bs4 import BeautifulSoup

        from tallgrass.alec.scraper import _parse_listing_entry

        html = """
        <article>
            <h2><a href="https://alec.org/model-policy/test-act/">Test Act</a></h2>
            <span>Model Policy</span>
            <a href="/model-policy-category/education/">Education</a>
        </article>
        """
        soup = BeautifulSoup(html, "lxml")
        article = soup.find("article")
        result = _parse_listing_entry(article)

        assert result is not None
        assert result["url"] == "https://alec.org/model-policy/test-act/"
        assert result["title"] == "Test Act"
        assert result["category"] == "Education"

    def test_parse_listing_entry_no_link(self):
        from bs4 import BeautifulSoup

        from tallgrass.alec.scraper import _parse_listing_entry

        html = "<article><h2>No link here</h2></article>"
        soup = BeautifulSoup(html, "lxml")
        article = soup.find("article")
        result = _parse_listing_entry(article)
        assert result is None

    def test_detect_max_page(self):
        from bs4 import BeautifulSoup

        from tallgrass.alec.scraper import _detect_max_page

        html = """
        <html><body>
        <a href="/model-policy/page/2/">2</a>
        <a href="/model-policy/page/3/">3</a>
        <a href="/model-policy/page/54/">54</a>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        assert _detect_max_page(soup) == 54

    def test_detect_max_page_single(self):
        from bs4 import BeautifulSoup

        from tallgrass.alec.scraper import _detect_max_page

        soup = BeautifulSoup("<html><body></body></html>", "lxml")
        assert _detect_max_page(soup) == 1


class TestALECCli:
    """ALEC CLI argument parsing tests."""

    def test_default_args(self):
        from tallgrass.alec.cli import ALEC_DATA_DIR

        assert ALEC_DATA_DIR == Path("data/external/alec")

    @patch("tallgrass.alec.cli.scrape_alec_corpus", return_value=[])
    @patch("tallgrass.alec.cli.save_alec_bills")
    def test_main_no_bills(self, mock_save, mock_scrape):
        from tallgrass.alec.cli import main

        main(["--output-dir", "/tmp/test_alec"])
        mock_scrape.assert_called_once()
        mock_save.assert_not_called()  # exits early when no bills


# ── OpenStates Adapter Tests ────────────────────────────────────────────────


class TestOpenStatesAdapter:
    """OpenStates multi-state adapter tests."""

    def test_init_valid_state(self):
        from tallgrass.text.openstates import OpenStatesAdapter

        adapter = OpenStatesAdapter("mo")
        assert adapter.state == "mo"
        assert adapter.state_name == "missouri"

    def test_init_uppercase(self):
        from tallgrass.text.openstates import OpenStatesAdapter

        adapter = OpenStatesAdapter("OK")
        assert adapter.state == "ok"
        assert adapter.state_name == "oklahoma"

    def test_init_unknown_state(self):
        from tallgrass.text.openstates import OpenStatesAdapter

        with pytest.raises(ValueError, match="Unknown state"):
            OpenStatesAdapter("zz")

    def test_data_dir(self):
        from tallgrass.text.openstates import OpenStatesAdapter

        adapter = OpenStatesAdapter("ne")
        assert adapter.data_dir("2025") == Path("data/nebraska/2025")

    def test_cache_dir(self):
        from tallgrass.text.openstates import OpenStatesAdapter

        adapter = OpenStatesAdapter("co")
        assert adapter.cache_dir("2025") == Path("data/colorado/2025/.cache/text")

    def test_normalize_bill_number(self):
        from tallgrass.text.openstates import OpenStatesAdapter

        adapter = OpenStatesAdapter("mo")
        assert adapter._normalize_bill_number("HB 1234") == "HB 1234"
        assert adapter._normalize_bill_number("SB  567") == "SB 567"
        assert adapter._normalize_bill_number(" LB 1 ") == "LB 1"

    def test_extract_pdf_url_introduced(self):
        from tallgrass.text.openstates import OpenStatesAdapter

        adapter = OpenStatesAdapter("mo")
        versions = [
            {
                "note": "Enrolled",
                "date": "2025-06-01",
                "links": [
                    {"url": "https://example.com/enrolled.pdf", "media_type": "application/pdf"}
                ],
            },
            {
                "note": "Introduced",
                "date": "2025-01-15",
                "links": [
                    {"url": "https://example.com/intro.pdf", "media_type": "application/pdf"}
                ],
            },
        ]
        url, label = adapter._extract_pdf_url(versions)
        assert url == "https://example.com/intro.pdf"
        assert "Introduced" in label

    def test_extract_pdf_url_fallback_earliest(self):
        from tallgrass.text.openstates import OpenStatesAdapter

        adapter = OpenStatesAdapter("ok")
        versions = [
            {
                "note": "Engrossed",
                "date": "2025-03-01",
                "links": [
                    {"url": "https://example.com/engrossed.pdf", "media_type": "application/pdf"}
                ],
            },
            {
                "note": "Committee Substitute",
                "date": "2025-02-01",
                "links": [{"url": "https://example.com/cs.pdf", "media_type": "application/pdf"}],
            },
        ]
        url, label = adapter._extract_pdf_url(versions)
        assert url == "https://example.com/cs.pdf"

    def test_extract_pdf_url_empty(self):
        from tallgrass.text.openstates import OpenStatesAdapter

        adapter = OpenStatesAdapter("ne")
        url, label = adapter._extract_pdf_url([])
        assert url == ""
        assert label == ""

    def test_extract_pdf_url_prefiled(self):
        from tallgrass.text.openstates import OpenStatesAdapter

        adapter = OpenStatesAdapter("co")
        versions = [
            {
                "note": "Prefiled",
                "date": "2024-12-01",
                "links": [{"url": "https://example.com/pre.pdf", "media_type": "application/pdf"}],
            },
        ]
        url, label = adapter._extract_pdf_url(versions)
        assert url == "https://example.com/pre.pdf"

    @patch.object(
        __import__("tallgrass.text.openstates", fromlist=["OpenStatesAdapter"]).OpenStatesAdapter,
        "_api_get",
    )
    def test_discover_bills_mock(self, mock_api_get):
        from tallgrass.text.openstates import OpenStatesAdapter

        adapter = OpenStatesAdapter("mo")

        mock_api_get.return_value = {
            "results": [
                {
                    "identifier": "HB 100",
                    "versions": [
                        {
                            "note": "Introduced",
                            "date": "2025-01-10",
                            "links": [
                                {
                                    "url": "https://example.com/hb100.pdf",
                                    "media_type": "application/pdf",
                                }
                            ],
                        }
                    ],
                },
                {
                    "identifier": "SB 50",
                    "versions": [
                        {
                            "note": "Filed",
                            "date": "2025-01-11",
                            "links": [
                                {
                                    "url": "https://example.com/sb50.pdf",
                                    "media_type": "application/pdf",
                                }
                            ],
                        }
                    ],
                },
            ],
            "pagination": {"max_page": 1},
        }

        refs = adapter.discover_bills("2025")
        assert len(refs) == 2
        assert refs[0].bill_number == "HB 100"
        assert refs[0].url == "https://example.com/hb100.pdf"
        assert refs[0].session == "MO-2025"

    @patch.object(
        __import__("tallgrass.text.openstates", fromlist=["OpenStatesAdapter"]).OpenStatesAdapter,
        "_api_get",
    )
    def test_discover_bills_dedup(self, mock_api_get):
        from tallgrass.text.openstates import OpenStatesAdapter

        adapter = OpenStatesAdapter("ok")

        # Same bill appears twice
        mock_api_get.return_value = {
            "results": [
                {
                    "identifier": "HB 1",
                    "versions": [
                        {
                            "note": "Introduced",
                            "date": "2025-01-01",
                            "links": [
                                {"url": "https://x.com/1.pdf", "media_type": "application/pdf"}
                            ],
                        }
                    ],
                },
                {
                    "identifier": "HB 1",
                    "versions": [
                        {
                            "note": "Enrolled",
                            "date": "2025-06-01",
                            "links": [
                                {"url": "https://x.com/1e.pdf", "media_type": "application/pdf"}
                            ],
                        }
                    ],
                },
            ],
            "pagination": {"max_page": 1},
        }

        refs = adapter.discover_bills("2025")
        assert len(refs) == 1  # deduplicated


# ── Similarity Computation Tests ────────────────────────────────────────────


class TestSimilarityComputation:
    """Cosine similarity and cross-corpus matching tests."""

    def test_identical_embeddings(self):
        from analysis.model_legislation_data import compute_cross_similarity

        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        bills = ["HB 1", "HB 2"]

        result = compute_cross_similarity(emb, bills, emb, bills, threshold=0.5)
        # Each bill should match itself with similarity 1.0
        exact = result.filter(pl.col("similarity") > 0.99)
        assert len(exact) >= 2

    def test_orthogonal_embeddings(self):
        from analysis.model_legislation_data import compute_cross_similarity

        ks = np.array([[1.0, 0.0]])
        other = np.array([[0.0, 1.0]])

        result = compute_cross_similarity(ks, ["HB 1"], other, ["ALEC_1"], threshold=0.5)
        assert len(result) == 0  # orthogonal = similarity 0

    def test_threshold_filtering(self):
        from analysis.model_legislation_data import compute_cross_similarity

        ks = np.array([[1.0, 0.0], [0.0, 1.0]])
        other = np.array([[0.95, 0.31]])  # ~0.95 similarity with first, ~0.31 with second

        result_high = compute_cross_similarity(ks, ["HB 1", "HB 2"], other, ["A1"], threshold=0.90)
        result_low = compute_cross_similarity(ks, ["HB 1", "HB 2"], other, ["A1"], threshold=0.10)

        assert len(result_high) <= len(result_low)

    def test_empty_corpus(self):
        from analysis.model_legislation_data import compute_cross_similarity

        ks = np.array([[1.0, 0.0]])
        other = np.empty((0, 2))

        result = compute_cross_similarity(ks, ["HB 1"], other, [], threshold=0.5)
        assert len(result) == 0

    def test_result_columns(self):
        from analysis.model_legislation_data import compute_cross_similarity

        ks = np.array([[1.0, 0.0]])
        other = np.array([[0.9, 0.44]])

        result = compute_cross_similarity(ks, ["HB 1"], other, ["A1"], threshold=0.5)
        assert "ks_bill" in result.columns
        assert "other_id" in result.columns
        assert "similarity" in result.columns
        assert "rank" in result.columns


# ── Match Classification Tests ──────────────────────────────────────────────


class TestMatchClassification:
    """Threshold-based match tier classification tests."""

    def test_near_identical(self):
        from analysis.model_legislation_data import classify_match

        assert classify_match(0.98) == "near-identical"
        assert classify_match(0.95) == "near-identical"
        assert classify_match(1.00) == "near-identical"

    def test_strong_match(self):
        from analysis.model_legislation_data import classify_match

        assert classify_match(0.90) == "strong match"
        assert classify_match(0.85) == "strong match"

    def test_related(self):
        from analysis.model_legislation_data import classify_match

        assert classify_match(0.75) == "related"
        assert classify_match(0.70) == "related"

    def test_below_threshold(self):
        from analysis.model_legislation_data import classify_match

        assert classify_match(0.69) == "below threshold"
        assert classify_match(0.50) == "below threshold"
        assert classify_match(0.0) == "below threshold"

    def test_boundary_values(self):
        from analysis.model_legislation_data import classify_match

        # Exact boundaries
        assert classify_match(0.95) == "near-identical"
        assert classify_match(0.8499) == "related"
        assert classify_match(0.6999) == "below threshold"


# ── N-gram Overlap Tests ────────────────────────────────────────────────────


class TestNgramOverlap:
    """N-gram text overlap computation tests."""

    def test_identical_texts(self):
        from analysis.model_legislation_data import compute_ngram_overlap

        text = "the quick brown fox jumps over the lazy dog and runs away fast"
        overlap = compute_ngram_overlap(text, text, n=5)
        assert overlap == pytest.approx(1.0)

    def test_no_overlap(self):
        from analysis.model_legislation_data import compute_ngram_overlap

        text_a = "the quick brown fox jumps over the lazy dog"
        text_b = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        overlap = compute_ngram_overlap(text_a, text_b, n=5)
        assert overlap == pytest.approx(0.0)

    def test_partial_overlap(self):
        from analysis.model_legislation_data import compute_ngram_overlap

        text_a = "the state shall require all voters to present identification at the polls"
        text_b = "the state shall require all citizens to register before voting begins"
        overlap = compute_ngram_overlap(text_a, text_b, n=5)
        assert 0.0 < overlap < 1.0

    def test_short_text(self):
        from analysis.model_legislation_data import compute_ngram_overlap

        overlap = compute_ngram_overlap("too short", "also short", n=5)
        assert overlap == 0.0

    def test_empty_text(self):
        from analysis.model_legislation_data import compute_ngram_overlap

        assert compute_ngram_overlap("", "some text here and more words", n=5) == 0.0
        assert compute_ngram_overlap("some text here and more words", "", n=5) == 0.0

    def test_custom_n(self):
        from analysis.model_legislation_data import compute_ngram_overlap

        text = "one two three four five six seven eight nine ten"
        # With n=3, more n-grams overlap than with n=5
        overlap_3 = compute_ngram_overlap(text, text, n=3)
        overlap_5 = compute_ngram_overlap(text, text, n=5)
        assert overlap_3 == pytest.approx(1.0)
        assert overlap_5 == pytest.approx(1.0)


# ── Match Summary Tests ─────────────────────────────────────────────────────


class TestMatchSummary:
    """Unified match table building tests."""

    def test_alec_only(self):
        from analysis.model_legislation_data import build_match_summary

        alec = pl.DataFrame(
            {
                "ks_bill": ["HB 1", "HB 2"],
                "other_id": ["url1", "url2"],
                "similarity": [0.92, 0.75],
                "rank": [1, 1],
            }
        )

        result = build_match_summary(
            alec_matches=alec,
            cross_state_matches={},
            ks_metadata=pl.DataFrame({"bill_number": ["HB 1", "HB 2"]}),
        )

        assert len(result) == 2
        assert all(result["source"] == "ALEC")
        assert "ks_bill" in result.columns
        assert "match_tier" in result.columns

    def test_cross_state_matches(self):
        from analysis.model_legislation_data import build_match_summary

        alec = pl.DataFrame(
            schema={
                "ks_bill": pl.Utf8,
                "other_id": pl.Utf8,
                "similarity": pl.Float64,
                "rank": pl.Int64,
            }
        )
        cross = {
            "mo": pl.DataFrame(
                {
                    "ks_bill": ["HB 1"],
                    "other_id": ["MO HB 500"],
                    "similarity": [0.88],
                    "rank": [1],
                }
            ),
        }

        result = build_match_summary(
            alec_matches=alec,
            cross_state_matches=cross,
            ks_metadata=pl.DataFrame({"bill_number": ["HB 1"]}),
        )

        assert len(result) == 1
        assert result["source"][0] == "MO"

    def test_empty_matches(self):
        from analysis.model_legislation_data import build_match_summary

        result = build_match_summary(
            alec_matches=pl.DataFrame(
                schema={
                    "ks_bill": pl.Utf8,
                    "other_id": pl.Utf8,
                    "similarity": pl.Float64,
                    "rank": pl.Int64,
                }
            ),
            cross_state_matches={},
            ks_metadata=pl.DataFrame({"bill_number": []}),
        )

        assert len(result) == 0
        assert "ks_bill" in result.columns

    def test_sorted_by_similarity(self):
        from analysis.model_legislation_data import build_match_summary

        alec = pl.DataFrame(
            {
                "ks_bill": ["HB 1", "HB 2", "HB 3"],
                "other_id": ["a", "b", "c"],
                "similarity": [0.72, 0.98, 0.85],
                "rank": [1, 1, 1],
            }
        )

        result = build_match_summary(
            alec_matches=alec,
            cross_state_matches={},
            ks_metadata=pl.DataFrame({"bill_number": ["HB 1", "HB 2", "HB 3"]}),
        )

        sims = result["similarity"].to_list()
        assert sims == sorted(sims, reverse=True)

    def test_with_topic_assignments(self):
        from analysis.model_legislation_data import build_match_summary

        alec = pl.DataFrame(
            {
                "ks_bill": ["HB 1"],
                "other_id": ["url1"],
                "similarity": [0.80],
                "rank": [1],
            }
        )

        topics = pl.DataFrame(
            {
                "bill_number": ["HB 1"],
                "topic_label": ["Education"],
            }
        )

        result = build_match_summary(
            alec_matches=alec,
            cross_state_matches={},
            ks_metadata=pl.DataFrame({"bill_number": ["HB 1"]}),
            topic_assignments=topics,
        )

        assert result["topic"][0] == "Education"

    def test_alec_title_lookup(self):
        from analysis.model_legislation_data import build_match_summary

        alec = pl.DataFrame(
            {
                "ks_bill": ["HB 1"],
                "other_id": ["https://alec.org/model-policy/voter-id/"],
                "similarity": [0.91],
                "rank": [1],
            }
        )

        alec_meta = pl.DataFrame(
            {
                "url": ["https://alec.org/model-policy/voter-id/"],
                "title": ["Voter ID Act"],
                "category": ["Elections"],
                "bill_type": ["Model Policy"],
                "date": [""],
                "task_force": [""],
                "text": ["..."],
            }
        )

        result = build_match_summary(
            alec_matches=alec,
            cross_state_matches={},
            ks_metadata=pl.DataFrame({"bill_number": ["HB 1"]}),
            alec_metadata=alec_meta,
        )

        assert result["match_label"][0] == "Voter ID Act"


# ── Data Loading Tests ──────────────────────────────────────────────────────


class TestDataLoading:
    """ALEC corpus and cross-state text loading tests."""

    def test_load_alec_corpus(self, tmp_path: Path):
        from analysis.model_legislation_data import load_alec_corpus

        long_text_1 = (
            "Section 1 requires all voters to present valid "
            "photo identification before casting a ballot at "
            "any polling location in the state"
        )
        long_text_2 = (
            "Section 1 no person shall be required as a "
            "condition of employment to join or pay dues to "
            "any labor organization operating in the state"
        )
        csv_path = tmp_path / "alec_model_bills.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["title", "category", "bill_type", "date",
                        "task_force", "url", "text"])
            w.writerow(["Voter ID Act", "Elections", "Model Policy",
                        "2009-01-01", "TF 1", "https://alec.org/1",
                        long_text_1])
            w.writerow(["Right to Work", "Labor", "Model Policy",
                        "", "TF 2", "https://alec.org/2", long_text_2])

        texts, ids, meta = load_alec_corpus(tmp_path, use_csv=True)
        assert len(texts) == 2
        assert len(ids) == 2
        assert "alec.org/1" in ids[0] or "alec.org/2" in ids[0]
        assert len(meta) == 2

    def test_load_alec_corpus_missing(self, tmp_path: Path):
        from analysis.model_legislation_data import load_alec_corpus

        with pytest.raises(FileNotFoundError, match="ALEC corpus not found"):
            load_alec_corpus(tmp_path, use_csv=True)

    def test_load_alec_corpus_filters_empty(self, tmp_path: Path):
        from analysis.model_legislation_data import load_alec_corpus

        csv_path = tmp_path / "alec_model_bills.csv"
        long_text = (
            "Section 1 requires all voters to present valid photo "
            "identification before casting a ballot at any polling "
            "location in the state"
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["title", "category", "bill_type", "date", "task_force", "url", "text"]
            )
            writer.writerow(
                ["Good Bill", "Cat", "Type", "", "TF", "https://alec.org/1", long_text]
            )
            writer.writerow(
                ["Empty Bill", "Cat", "Type", "", "TF", "https://alec.org/2", "short"]
            )

        texts, ids, meta = load_alec_corpus(tmp_path, use_csv=True)
        assert len(texts) == 1  # "short" is < MIN_TEXT_LENGTH


# ── Constants Consistency Tests ─────────────────────────────────────────────


class TestConstants:
    """Verify threshold and parameter constants are consistent."""

    def test_threshold_ordering(self):
        from analysis.model_legislation_data import (
            THRESHOLD_NEAR_IDENTICAL,
            THRESHOLD_RELATED,
            THRESHOLD_STRONG_MATCH,
        )

        assert THRESHOLD_NEAR_IDENTICAL > THRESHOLD_STRONG_MATCH > THRESHOLD_RELATED
        assert THRESHOLD_NEAR_IDENTICAL == 0.95
        assert THRESHOLD_STRONG_MATCH == 0.85
        assert THRESHOLD_RELATED == 0.70

    def test_ngram_defaults(self):
        from analysis.model_legislation_data import NGRAM_OVERLAP_THRESHOLD, NGRAM_SIZE

        assert NGRAM_SIZE == 5
        assert NGRAM_OVERLAP_THRESHOLD == 0.20


# ── Report Builder Tests ────────────────────────────────────────────────────


class TestReportBuilder:
    """Model legislation report builder tests."""

    def test_generate_key_findings_with_matches(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "23_model_legislation"))
        from analysis.model_legislation_report import _generate_key_findings

        results = {
            "n_kansas_bills": 400,
            "n_alec_bills": 1000,
            "n_alec_matches": 15,
            "n_near_identical": 2,
            "n_strong_matches": 5,
            "n_cross_state_matches": 8,
            "states_with_matches": ["mo", "ok"],
            "top_alec_match": {
                "ks_bill": "HB 2773",
                "match_label": "Charter Schools Act",
                "similarity": 0.97,
            },
        }

        findings = _generate_key_findings(results)
        assert len(findings) >= 3
        assert any("400" in f for f in findings)
        assert any("ALEC" in f for f in findings)

    def test_generate_key_findings_no_matches(self):
        from analysis.model_legislation_report import _generate_key_findings

        results = {
            "n_kansas_bills": 400,
            "n_alec_bills": 1000,
            "n_alec_matches": 0,
            "n_near_identical": 0,
            "n_strong_matches": 0,
            "n_cross_state_matches": 0,
            "states_with_matches": [],
            "top_alec_match": None,
        }

        findings = _generate_key_findings(results)
        assert any("No Kansas bills matched" in f for f in findings)


# ── CLI Args Tests ──────────────────────────────────────────────────────────


class TestCLIArgs:
    """Phase 20 CLI argument parsing tests."""

    def test_default_args(self):
        from analysis.model_legislation import parse_args

        args = parse_args([])
        assert args.session == "2025-26"
        assert args.threshold == 0.70
        assert not args.alec_only
        assert "mo" in args.states.lower()

    def test_alec_only(self):
        from analysis.model_legislation import parse_args

        args = parse_args(["--alec-only"])
        assert args.alec_only is True

    def test_custom_threshold(self):
        from analysis.model_legislation import parse_args

        args = parse_args(["--threshold", "0.85"])
        assert args.threshold == 0.85

    def test_custom_states(self):
        from analysis.model_legislation import parse_args

        args = parse_args(["--states", "MO,NE"])
        assert "MO" in args.states
        assert "NE" in args.states
