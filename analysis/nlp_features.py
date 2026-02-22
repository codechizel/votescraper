"""
NLP topic features for bill passage prediction.

Fits TF-IDF + NMF on bill short_title text to produce topic proportion features.
Pure-logic module (no I/O, no plotting) following the synthesis_detect.py pattern.

Usage (called from prediction.py):
    from analysis.nlp_features import fit_topic_features, get_topic_display_names
    topic_df, topic_model = fit_topic_features(rollcalls["short_title"])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Constants ────────────────────────────────────────────────────────────────

NMF_N_TOPICS = 6
TFIDF_MAX_DF = 0.85
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 500
TFIDF_NGRAM_RANGE = (1, 2)
NMF_TOP_WORDS = 5
NMF_RANDOM_STATE = 42


@dataclass(frozen=True)
class TopicModel:
    """Metadata about a fitted NMF topic model."""

    n_topics: int
    topic_labels: list[str]
    topic_top_words: dict[str, list[str]]
    feature_names: list[str]
    vocabulary_size: int
    n_documents: int


def _empty_topic_result(n_docs: int) -> tuple[pl.DataFrame, TopicModel]:
    """Return zero-filled topic DataFrame and empty model for degenerate input."""
    topic_cols = {f"topic_{i}": [0.0] * n_docs for i in range(NMF_N_TOPICS)}
    topic_df = pl.DataFrame(topic_cols)
    model = TopicModel(
        n_topics=0,
        topic_labels=[f"topic_{i}" for i in range(NMF_N_TOPICS)],
        topic_top_words={f"topic_{i}": [] for i in range(NMF_N_TOPICS)},
        feature_names=[f"topic_{i}" for i in range(NMF_N_TOPICS)],
        vocabulary_size=0,
        n_documents=n_docs,
    )
    return topic_df, model


def fit_topic_features(texts: pl.Series) -> tuple[pl.DataFrame, TopicModel]:
    """Fit TF-IDF + NMF on short_title text and return topic proportion columns.

    Args:
        texts: A polars Series of bill short_title strings. May contain nulls or
            empty strings — these are replaced with empty strings before fitting.

    Returns:
        A tuple of (topic_df, topic_model) where topic_df has columns
        topic_0 .. topic_{N-1} with one row per input document, and topic_model
        holds metadata (labels, top words, vocabulary size).
    """
    # Defensively handle nulls and empty strings
    clean_texts = texts.fill_null("").cast(pl.Utf8).to_list()
    n_docs = len(clean_texts)

    # Guard: if corpus is too small or degenerate, return zero-filled DataFrame
    non_empty = [t for t in clean_texts if t.strip()]
    if len(non_empty) < 2:
        return _empty_topic_result(n_docs)

    # Fit TF-IDF — use relaxed params for small corpora
    effective_min_df = min(TFIDF_MIN_DF, max(1, len(non_empty) // 2))
    try:
        vectorizer = TfidfVectorizer(
            max_df=TFIDF_MAX_DF,
            min_df=effective_min_df,
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words="english",
        )
        tfidf_matrix = vectorizer.fit_transform(clean_texts)
    except ValueError:
        # Fallback: no min_df/max_df filtering
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 1),
            stop_words="english",
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(clean_texts)
        except ValueError:
            return _empty_topic_result(n_docs)

    vocab = vectorizer.get_feature_names_out().tolist()
    if len(vocab) == 0:
        return _empty_topic_result(n_docs)

    # Adjust n_topics if vocabulary is too small
    effective_topics = min(NMF_N_TOPICS, len(vocab), n_docs)
    if effective_topics < 1:
        return _empty_topic_result(n_docs)

    # Fit NMF
    nmf = NMF(
        n_components=effective_topics,
        random_state=NMF_RANDOM_STATE,
        max_iter=300,
    )
    W = nmf.fit_transform(tfidf_matrix)  # (n_docs, n_topics)

    # Build topic labels and top words
    feature_names_out = vectorizer.get_feature_names_out()
    topic_labels: list[str] = []
    topic_top_words: dict[str, list[str]] = {}

    for topic_idx in range(effective_topics):
        col_name = f"topic_{topic_idx}"
        top_word_indices = nmf.components_[topic_idx].argsort()[::-1][:NMF_TOP_WORDS]
        top_words = [feature_names_out[i] for i in top_word_indices]
        label = " / ".join(w.title() for w in top_words[:3])
        topic_labels.append(f"Topic: {label}")
        topic_top_words[col_name] = top_words

    # Build polars DataFrame of topic proportions
    topic_cols = {f"topic_{i}": W[:, i].tolist() for i in range(effective_topics)}
    # Pad with zeros if fewer topics than NMF_N_TOPICS
    for i in range(effective_topics, NMF_N_TOPICS):
        topic_cols[f"topic_{i}"] = [0.0] * n_docs
        topic_labels.append("Topic: (unused)")
        topic_top_words[f"topic_{i}"] = []

    topic_df = pl.DataFrame(topic_cols)

    # Build column-name list matching the DataFrame
    col_names = [f"topic_{i}" for i in range(NMF_N_TOPICS)]

    model = TopicModel(
        n_topics=effective_topics,
        topic_labels=topic_labels,
        topic_top_words=topic_top_words,
        feature_names=col_names,
        vocabulary_size=len(vocab),
        n_documents=n_docs,
    )

    return topic_df, model


def get_topic_display_names(model: TopicModel) -> dict[str, str]:
    """Map topic column names to plain-English labels for SHAP plots.

    Returns:
        Dict like {"topic_0": "Topic: Elections / Voting / Ballots", ...}
    """
    result: dict[str, str] = {}
    for i, label in enumerate(model.topic_labels):
        result[f"topic_{i}"] = label
    return result


def plot_topic_words(
    model: TopicModel,
    chamber: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Multi-panel horizontal bar chart showing top words per NMF topic."""
    n_topics = model.n_topics
    if n_topics == 0:
        return

    fig, axes = plt.subplots(
        1,
        n_topics,
        figsize=(3.5 * n_topics, 4),
        sharey=False,
    )

    if n_topics == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_topics))

    for idx, ax in enumerate(axes):
        col_name = f"topic_{idx}"
        words = model.topic_top_words.get(col_name, [])
        if not words:
            ax.set_visible(False)
            continue

        # Reverse for bottom-to-top reading
        words_display = list(reversed(words))
        y_pos = range(len(words_display))
        # Use uniform bar widths (we don't have component weights here)
        bar_widths = list(reversed(list(range(1, len(words) + 1))))

        ax.barh(
            list(y_pos),
            bar_widths,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(words_display, fontsize=9)
        ax.set_title(model.topic_labels[idx], fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    fig.suptitle(
        f"{chamber} — Bill Title Topics (NMF, K={n_topics})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
