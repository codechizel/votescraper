"""Model legislation detection — data loading, embedding, similarity computation.

Pure functions for loading ALEC corpus and cross-state bill texts, computing
cosine similarity, n-gram overlap, and building unified match summaries.

Reuses Phase 18 embedding infrastructure (``get_or_compute_embeddings`` from
``bill_text_data.py``) so all corpora share the same embedding space.
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

# ── Constants ────────────────────────────────────────────────────────────────

# Similarity thresholds (LID literature standard)
THRESHOLD_NEAR_IDENTICAL = 0.95
THRESHOLD_STRONG_MATCH = 0.85
THRESHOLD_RELATED = 0.70

# N-gram overlap settings
NGRAM_SIZE = 5
NGRAM_OVERLAP_THRESHOLD = 0.20

# Minimum text length to consider for matching
MIN_TEXT_LENGTH = 100


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_alec_corpus(alec_dir: Path) -> tuple[list[str], list[str], pl.DataFrame]:
    """Load ALEC model bills from CSV.

    Returns:
        texts: Raw bill texts (one per model bill).
        identifiers: Unique identifiers for each bill (URL-based).
        metadata_df: DataFrame with title, category, bill_type, etc.
    """
    csv_path = alec_dir / "alec_model_bills.csv"
    if not csv_path.exists():
        msg = f"ALEC corpus not found: {csv_path}. Run `just alec` first."
        raise FileNotFoundError(msg)

    df = pl.read_csv(csv_path)

    # Filter out empty texts
    df = df.filter(
        pl.col("text").is_not_null() & (pl.col("text").str.len_chars() >= MIN_TEXT_LENGTH)
    )

    texts = df["text"].to_list()
    # Use URL as unique identifier (titles may not be unique)
    identifiers = df["url"].to_list()

    return texts, identifiers, df


def load_cross_state_texts(
    states: list[str],
    session: str,
) -> dict[str, tuple[list[str], list[str], pl.DataFrame]]:
    """Load bill texts for comparison states.

    Returns:
        Dict mapping state abbreviation to (texts, bill_numbers, metadata_df).
        States with no data are silently omitted.
    """
    result: dict[str, tuple[list[str], list[str], pl.DataFrame]] = {}

    state_names = {
        "mo": "missouri",
        "ok": "oklahoma",
        "ne": "nebraska",
        "co": "colorado",
    }

    for state in states:
        state_lower = state.lower()
        state_name = state_names.get(state_lower, state_lower)
        data_dir = Path(f"data/{state_name}/{session}")

        # Look for bill_texts.csv with various naming patterns
        csv_candidates = list(data_dir.glob("*_bill_texts.csv"))
        if not csv_candidates:
            print(f"  No bill texts found for {state.upper()} ({data_dir})")
            continue

        csv_path = csv_candidates[0]  # take first match
        df = pl.read_csv(csv_path)

        # Filter empty texts
        df = df.filter(
            pl.col("text").is_not_null() & (pl.col("text").str.len_chars() >= MIN_TEXT_LENGTH)
        )

        if len(df) == 0:
            print(f"  No valid bill texts for {state.upper()} (all empty)")
            continue

        texts = df["text"].to_list()
        bill_numbers = df["bill_number"].to_list()

        result[state_lower] = (texts, bill_numbers, df)
        print(f"  Loaded {len(texts)} bill texts for {state.upper()}")

    return result


# ── Similarity Computation ───────────────────────────────────────────────────


def compute_cross_similarity(
    ks_embeddings: np.ndarray,
    ks_bills: list[str],
    other_embeddings: np.ndarray,
    other_ids: list[str],
    threshold: float = THRESHOLD_RELATED,
) -> pl.DataFrame:
    """Cosine similarity between Kansas bills and another corpus.

    Args:
        ks_embeddings: Kansas bill embeddings (n_ks, dim).
        ks_bills: Kansas bill numbers.
        other_embeddings: Other corpus embeddings (n_other, dim).
        other_ids: Other corpus identifiers.
        threshold: Minimum similarity to report.

    Returns:
        DataFrame with columns: ks_bill, other_id, similarity, rank.
    """
    # Normalize embeddings to unit vectors for cosine similarity
    ks_norm = ks_embeddings / np.linalg.norm(ks_embeddings, axis=1, keepdims=True)
    other_norm = other_embeddings / np.linalg.norm(other_embeddings, axis=1, keepdims=True)

    # Compute full similarity matrix
    sim_matrix = ks_norm @ other_norm.T  # (n_ks, n_other)

    # Extract pairs above threshold
    rows: list[dict] = []
    for i, ks_bill in enumerate(ks_bills):
        # Sort other bills by similarity (descending)
        sims = sim_matrix[i]
        sorted_idx = np.argsort(sims)[::-1]

        rank = 0
        for j in sorted_idx:
            if sims[j] < threshold:
                break
            rank += 1
            rows.append(
                {
                    "ks_bill": ks_bill,
                    "other_id": other_ids[j],
                    "similarity": float(sims[j]),
                    "rank": rank,
                }
            )

    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "ks_bill": pl.Utf8,
                "other_id": pl.Utf8,
                "similarity": pl.Float64,
                "rank": pl.Int64,
            }
        )
    )


def classify_match(similarity: float) -> str:
    """Classify a similarity score into a match tier.

    Args:
        similarity: Cosine similarity in [0, 1].

    Returns:
        Match tier label: "near-identical", "strong match", "related", or "below threshold".
    """
    if similarity >= THRESHOLD_NEAR_IDENTICAL:
        return "near-identical"
    elif similarity >= THRESHOLD_STRONG_MATCH:
        return "strong match"
    elif similarity >= THRESHOLD_RELATED:
        return "related"
    else:
        return "below threshold"


# ── N-gram Overlap ───────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization."""
    return _WORD_RE.findall(text.lower())


def compute_ngram_overlap(text_a: str, text_b: str, n: int = NGRAM_SIZE) -> float:
    """Fraction of n-grams in text_a also found in text_b.

    Used as secondary evidence for high-similarity pairs (>= 0.85)
    to confirm genuine text reuse rather than topical similarity.

    Returns value in [0, 1].  0.20+ suggests genuine text sharing.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if len(tokens_a) < n or len(tokens_b) < n:
        return 0.0

    ngrams_a = Counter(tuple(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1))
    ngrams_b = set(tuple(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1))

    if not ngrams_a:
        return 0.0

    # Count how many of text_a's n-grams appear in text_b
    shared = sum(count for ng, count in ngrams_a.items() if ng in ngrams_b)
    total = sum(ngrams_a.values())

    return shared / total


# ── Match Summary ────────────────────────────────────────────────────────────


def build_match_summary(
    alec_matches: pl.DataFrame,
    cross_state_matches: dict[str, pl.DataFrame],
    ks_metadata: pl.DataFrame,
    alec_metadata: pl.DataFrame | None = None,
    topic_assignments: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build unified match table across all sources.

    Args:
        alec_matches: ALEC similarity results (ks_bill, other_id, similarity, rank).
        cross_state_matches: Per-state similarity results.
        ks_metadata: Kansas bill metadata (bill_number, text, etc.).
        alec_metadata: ALEC bill metadata (title, category, url, etc.).
        topic_assignments: Phase 18 topic assignments (bill_number, topic_label).

    Returns:
        DataFrame with columns: ks_bill, source, match_id, match_label,
        similarity, match_tier, ngram_overlap, topic.
    """
    rows: list[dict] = []

    # ALEC matches
    if len(alec_matches) > 0:
        for row in alec_matches.iter_rows(named=True):
            label = row["other_id"]
            if alec_metadata is not None:
                # Look up title by URL
                title_rows = alec_metadata.filter(pl.col("url") == row["other_id"])
                if len(title_rows) > 0:
                    label = title_rows["title"][0]

            rows.append(
                {
                    "ks_bill": row["ks_bill"],
                    "source": "ALEC",
                    "match_id": row["other_id"],
                    "match_label": label,
                    "similarity": row["similarity"],
                    "match_tier": classify_match(row["similarity"]),
                    "ngram_overlap": None,
                    "topic": "",
                }
            )

    # Cross-state matches
    for state, matches_df in cross_state_matches.items():
        if len(matches_df) == 0:
            continue
        for row in matches_df.iter_rows(named=True):
            rows.append(
                {
                    "ks_bill": row["ks_bill"],
                    "source": state.upper(),
                    "match_id": row["other_id"],
                    "match_label": f"{state.upper()} {row['other_id']}",
                    "similarity": row["similarity"],
                    "match_tier": classify_match(row["similarity"]),
                    "ngram_overlap": None,
                    "topic": "",
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "ks_bill": pl.Utf8,
                "source": pl.Utf8,
                "match_id": pl.Utf8,
                "match_label": pl.Utf8,
                "similarity": pl.Float64,
                "match_tier": pl.Utf8,
                "ngram_overlap": pl.Float64,
                "topic": pl.Utf8,
            }
        )

    summary = pl.DataFrame(rows)

    # Add topic assignments if available
    if topic_assignments is not None and "bill_number" in topic_assignments.columns:
        topic_map = topic_assignments.select(["bill_number", "topic_label"]).unique(
            subset=["bill_number"], keep="first"
        )
        summary = summary.join(
            topic_map.rename({"bill_number": "ks_bill", "topic_label": "_topic"}),
            on="ks_bill",
            how="left",
        )
        summary = summary.with_columns(
            pl.coalesce(pl.col("_topic"), pl.col("topic")).alias("topic")
        ).drop("_topic")

    # Sort by similarity descending
    summary = summary.sort("similarity", descending=True)

    return summary
