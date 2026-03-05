"""
Kansas Legislature — Bill Text Analysis (Phase 18)

Fits BERTopic (UMAP → HDBSCAN → c-TF-IDF) on full bill text to discover
policy topics automatically.  Optionally classifies bills into the 20 CAP
major topic categories via Claude API.  Computes bill similarity from text
embeddings and cross-references topics with voting patterns (Rice index
per topic × party, caucus-splitting scores).

Embedding: FastEmbed (ONNX Runtime, ~50-100 MB) — no PyTorch dependency.
Topic modeling: BERTopic with pre-computed embeddings.
Classification: Claude Sonnet via standard or Batch API (optional, requires
ANTHROPIC_API_KEY).

Usage:
  uv run python analysis/20_bill_text/bill_text.py [--session 2025-26]
      [--classify] [--batch] [--embedding-model MODEL]
      [--min-cluster-size 15] [--run-id RUN_ID]

Outputs (in results/<session>/<run_id>/20_bill_text/):
  - data/:   Parquet files (topics, embeddings, CAP classifications, similarity)
  - plots/:  PNG visualizations (topic bars, heatmaps, word clouds, similarity)
  - filtering_manifest.json, run_info.json, run_log.txt
  - bill_text_report.html
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext
except ModuleNotFoundError:
    from run_context import RunContext

try:
    from analysis.bill_text_report import build_bill_text_report
except ModuleNotFoundError:
    from bill_text_report import build_bill_text_report  # type: ignore[no-redef]

try:
    from analysis.bill_text_data import (
        DEFAULT_EMBEDDING_MODEL,
        assign_bills_to_chambers,
        get_or_compute_embeddings,
        load_bill_texts,
        load_rollcalls,
        load_votes,
        preprocess_for_embedding,
    )
except ModuleNotFoundError:
    from bill_text_data import (  # type: ignore[no-redef]
        DEFAULT_EMBEDDING_MODEL,
        assign_bills_to_chambers,
        get_or_compute_embeddings,
        load_bill_texts,
        load_rollcalls,
        load_votes,
        preprocess_for_embedding,
    )

try:
    from analysis.bill_text_classify import (
        classify_bills_cap,
        classify_bills_cap_batch,
    )
except ModuleNotFoundError:
    from bill_text_classify import (  # type: ignore[no-redef]
        classify_bills_cap,
        classify_bills_cap_batch,
    )

try:
    from analysis.phase_utils import load_legislators, print_header, save_fig
except ImportError:
    from phase_utils import (  # type: ignore[no-redef]
        load_legislators,
        print_header,
        save_fig,
    )


# ── Constants ────────────────────────────────────────────────────────────────

MIN_CLUSTER_SIZE = 15
"""Minimum HDBSCAN cluster size.  Controls topic granularity."""

MIN_SAMPLES = 5
"""Minimum HDBSCAN samples.  Affects noise point assignment."""

UMAP_N_COMPONENTS = 10
"""UMAP output dimensions for HDBSCAN input."""

UMAP_N_NEIGHBORS = 15
"""UMAP neighborhood size.  Balances local vs global structure."""

UMAP_MIN_DIST = 0.0
"""UMAP minimum distance.  0.0 allows tighter clusters."""

RANDOM_SEED = 42
"""Reproducibility seed for all stochastic operations."""

SIMILARITY_THRESHOLD = 0.80
"""Cosine similarity threshold for reporting similar bill pairs."""

TOP_SIMILAR_PAIRS = 30
"""Number of most-similar bill pairs to include in report."""

HEATMAP_TOP_N = 50
"""Number of most-connected bills to show in similarity heatmap."""

VECTORIZER_MAX_DF = 0.85
"""Maximum document frequency for c-TF-IDF vocabulary.  Terms appearing in
>85% of bills are filtered — catches domain-ubiquitous words like 'state',
'kansas' without needing them in the stopword list (and without blocking
useful bigrams like 'state board')."""

LEGISLATIVE_STOPWORDS: frozenset[str] = frozenset({
    # Mandatory legal language (every bill)
    "shall",
    # Preprocessing artifact (normalized K.S.A. references)
    "statuteref",
    # Structural markers
    "section", "subsection", "paragraph",
    # Amendatory boilerplate
    "amendments", "amendment", "amended", "amend",
    # Archaic legal connectors
    "thereto", "thereof", "therein",
    "herein", "hereby", "hereof",
    "pursuant", "provision", "provisions",
})
"""Legislative boilerplate terms filtered from c-TF-IDF topic labels.
These appear in virtually every Kansas bill regardless of policy area.
Combined with scikit-learn's English stopwords in the BERTopic vectorizer."""

PARTY_COLORS = {
    "Republican": "#E81B23",
    "Democrat": "#0015BC",
    "Independent": "#999999",
}

# ── Primer ───────────────────────────────────────────────────────────────────

BILL_TEXT_PRIMER = """\
# Bill Text Analysis (Phase 18)

## Purpose

Discovers policy topics from full bill text and measures how those topics
interact with partisan voting patterns.  Answers: what policy areas are most
legislatively active?  Which topics split the majority caucus internally?
Which bills are near-duplicates based on textual similarity?

Optionally classifies bills into the standardized 20-category Comparative
Agendas Project (CAP) taxonomy via Claude API, enabling cross-state and
cross-session comparison.

## Method

### Topic Discovery (BERTopic)
1. **Embed** bill texts using FastEmbed (BAAI/bge-small-en-v1.5, 384-dim).
2. **Reduce** embeddings via UMAP (10D, cosine metric, random_state=42).
3. **Cluster** reduced embeddings with HDBSCAN (min_cluster_size=15).
4. **Label** clusters with c-TF-IDF top words.

### CAP Classification (Optional)
Claude Sonnet classifies each bill into one of 20 CAP categories.
Responses are cached by content hash — repeat runs skip API calls.
Requires `ANTHROPIC_API_KEY` environment variable.

### Bill Similarity
Cosine similarity on 384-dim embeddings.  Pairs above 0.80 threshold
are reported.  Clustered heatmap shows semantic structure.

### Vote Cross-Reference
Join topics to roll call votes.  Compute Rice index per topic × party.
Caucus-splitting score = 1 − Rice(majority party).  Higher scores indicate
policy areas that fracture the dominant party's voting bloc.

## Inputs

Reads from `data/kansas/{legislature}_{start}-{end}/`:
- `{name}_bill_texts.csv` — Full bill text from BT1 (`just text`)
- `{name}_rollcalls.csv` — Roll call metadata
- `{name}_votes.csv` — Individual vote records
- `{name}_legislators.csv` — Legislator metadata

## Outputs

All outputs in `results/<session>/<run_id>/20_bill_text/`:

### `data/` — Parquet intermediates

| File | Description |
|------|-------------|
| `bill_topics_all.parquet` | Topic assignments for all bills |
| `bill_topics_house.parquet` | House-only topic assignments |
| `bill_topics_senate.parquet` | Senate-only topic assignments |
| `bill_embeddings.parquet` | 384-dim embedding vectors per bill |
| `cap_classifications.parquet` | CAP categories (if --classify) |
| `topic_model_info.json` | Topic count, top words, sizes |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `topic_distribution_{chamber}.png` | Bar chart of topic sizes |
| `topic_party_heatmap_{chamber}.png` | Topic × party Rice index |
| `caucus_splitting_topics_{chamber}.png` | Topics ranked by majority dissent |
| `cap_category_distribution.png` | CAP 20-category bar chart |
| `bill_similarity_heatmap.png` | Clustered cosine similarity matrix |

## Interpretation Guide

- **Topics** are discovered automatically — their labels are the top c-TF-IDF
  words, not curated names.  Review the word lists to understand each topic.
- **Caucus-splitting score** near 1.0 = the majority party splits 50/50 on
  that topic.  Near 0.0 = near-unanimous.
- **Bill similarity** > 0.90 often indicates companion bills or
  near-identical amendments.  Check source URLs to confirm.
- **CAP confidence** 1-5 scale is Claude's self-reported certainty.
  Confidence < 3 warrants manual review.

## Caveats

1. **Supplemental notes preferred** over introduced text when both exist.
   Supp notes are shorter plain-English summaries, better for NLP.
   Some bills only have introduced text (longer, more boilerplate).
2. **Topic count is automatic** (HDBSCAN decides K).  Noise points
   (topic -1) are bills that don't cluster with any topic.
3. **CAP classification is LLM-based** — not deterministic across model
   updates.  Caching ensures within-run reproducibility.
4. **Embedding truncation** to ~2000 tokens may lose content from very
   long bills (100+ pages).
"""


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Bill Text Analysis (Phase 18)")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Run CAP classification via Claude API (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use Batch API for CAP classification (50%% cheaper, async)",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"FastEmbed model for text embeddings (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=MIN_CLUSTER_SIZE,
        help=f"HDBSCAN min_cluster_size (default: {MIN_CLUSTER_SIZE})",
    )
    parser.add_argument("--csv", action="store_true", help="Force CSV loading (skip database)")
    return parser.parse_args()


# ── Topic Modeling ───────────────────────────────────────────────────────────


def fit_topic_model(
    texts: list[str],
    embeddings: np.ndarray,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[object, list[int], np.ndarray]:
    """Fit BERTopic with reproducible settings.

    Returns (model, topics, probabilities).
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=random_state,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | LEGISLATIVE_STOPWORDS),
        ngram_range=(1, 2),
        min_df=2,
        max_df=VECTORIZER_MAX_DF,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(texts, embeddings)
    return topic_model, topics, probs


def extract_topic_info(
    topic_model: object,
    topics: list[int],
    bill_numbers: list[str],
) -> tuple[pl.DataFrame, list[dict]]:
    """Extract topic assignments and topic metadata from fitted BERTopic model.

    Returns (bill_topics_df, topic_info_list).
    """
    # Get topic info from model
    info = topic_model.get_topic_info()  # type: ignore[union-attr]

    topic_info_list: list[dict] = []
    for _, row in info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            label = "Noise (unassigned)"
            words = ""
        else:
            # Get top words for this topic
            topic_words = topic_model.get_topic(topic_id)  # type: ignore[union-attr]
            words = ", ".join([w for w, _ in topic_words[:10]])
            label = f"Topic {topic_id}: {', '.join([w for w, _ in topic_words[:5]])}"
        topic_info_list.append(
            {
                "topic_id": topic_id,
                "topic_label": label,
                "count": int(row["Count"]),
                "top_words": words,
            }
        )

    # Build bill-level topic assignments
    topic_labels = {t["topic_id"]: t["topic_label"] for t in topic_info_list}
    topic_words = {t["topic_id"]: t["top_words"] for t in topic_info_list}
    bill_topics = pl.DataFrame(
        {
            "bill_number": bill_numbers,
            "topic_id": topics,
        }
    ).with_columns(
        pl.col("topic_id")
        .map_elements(lambda tid: topic_labels.get(tid, f"Topic {tid}"), return_dtype=pl.Utf8)
        .alias("topic_label"),
        pl.col("topic_id")
        .map_elements(lambda tid: topic_words.get(tid, ""), return_dtype=pl.Utf8)
        .alias("top_words"),
    )

    return bill_topics, topic_info_list


# ── Bill Similarity ──────────────────────────────────────────────────────────


def compute_bill_similarity(
    embeddings: np.ndarray,
    bill_numbers: list[str],
    threshold: float = SIMILARITY_THRESHOLD,
    top_n: int = TOP_SIMILAR_PAIRS,
) -> tuple[np.ndarray, list[dict]]:
    """Cosine similarity matrix.  Return matrix and top pairs above threshold.

    Returns (similarity_matrix, top_pairs_list).
    """
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T

    # Extract top pairs above threshold (upper triangle only)
    pairs: list[dict] = []
    n = len(bill_numbers)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                pairs.append(
                    {
                        "bill_a": bill_numbers[i],
                        "bill_b": bill_numbers[j],
                        "similarity": float(sim_matrix[i, j]),
                    }
                )

    # Sort by similarity descending, take top N
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return sim_matrix, pairs[:top_n]


# ── Vote Cross-Reference ────────────────────────────────────────────────────


def compute_topic_party_cohesion(
    bill_topics: pl.DataFrame,
    rollcalls: pl.DataFrame,
    votes: pl.DataFrame,
    legislators: pl.DataFrame,
) -> tuple[pl.DataFrame, list[dict]]:
    """Compute Rice index per topic × party.  Returns cohesion table and
    caucus-splitting rankings.

    Rice index = |%Yea - %Nay| within a group.  Ranges 0 (split 50/50)
    to 1.0 (unanimous).

    Caucus-splitting score = 1 - Rice(majority party) for each topic.
    """
    # Ensure consistent column names
    if "slug" in votes.columns and "legislator_slug" not in votes.columns:
        votes = votes.rename({"slug": "legislator_slug"})

    # Join bill topics → rollcalls → votes → legislators
    # rollcalls has bill_number and vote_id
    rc_cols = ["bill_number", "vote_id"]
    if "chamber" in rollcalls.columns:
        rc_cols.append("chamber")
    rc_slim = rollcalls.select([c for c in rc_cols if c in rollcalls.columns])

    merged = (
        bill_topics.select(["bill_number", "topic_id", "topic_label"])
        .join(rc_slim, on="bill_number", how="inner")
        .join(
            votes.select(["vote_id", "legislator_slug", "vote"]),
            on="vote_id",
            how="inner",
        )
        .join(
            legislators.select(["legislator_slug", "party"]),
            on="legislator_slug",
            how="left",
        )
    )

    if len(merged) == 0:
        return pl.DataFrame(), []

    # Fill empty party to "Independent"
    merged = merged.with_columns(
        pl.when(pl.col("party").is_null() | (pl.col("party") == ""))
        .then(pl.lit("Independent"))
        .otherwise(pl.col("party"))
        .alias("party")
    )

    # Compute Rice index per topic × party
    # Rice = |%Yea - %Nay| where Yea = "Yea", Nay = "Nay"
    grouped = (
        merged.filter(pl.col("vote").is_in(["Yea", "Nay"]))
        .group_by(["topic_id", "topic_label", "party"])
        .agg(
            pl.col("vote").filter(pl.col("vote") == "Yea").len().alias("n_yea"),
            pl.col("vote").filter(pl.col("vote") == "Nay").len().alias("n_nay"),
            pl.col("vote").len().alias("n_total"),
            pl.col("bill_number").n_unique().alias("n_bills"),
        )
        .with_columns(
            (
                (pl.col("n_yea").cast(pl.Float64) / pl.col("n_total"))
                - (pl.col("n_nay").cast(pl.Float64) / pl.col("n_total"))
            )
            .abs()
            .alias("rice_index")
        )
        .filter(pl.col("topic_id") != -1)  # exclude noise cluster
    )

    if len(grouped) == 0:
        return pl.DataFrame(), []

    # Determine majority party (most legislators)
    party_counts = legislators.group_by("party").len().sort("len", descending=True)
    majority_party = party_counts.row(0)[0]
    if not majority_party or majority_party == "Independent":
        majority_party = "Republican"  # Kansas default

    # Determine minority party
    minority_parties = party_counts.filter(
        (pl.col("party") != majority_party) & (pl.col("party") != "Independent")
    )
    minority_party = minority_parties.row(0)[0] if len(minority_parties) > 0 else "Democrat"

    # Build caucus-splitting rankings
    caucus_splitting: list[dict] = []
    topics = grouped.select("topic_id", "topic_label").unique()

    for row in topics.iter_rows(named=True):
        tid = row["topic_id"]
        tlabel = row["topic_label"]

        maj_row = grouped.filter((pl.col("topic_id") == tid) & (pl.col("party") == majority_party))
        min_row = grouped.filter((pl.col("topic_id") == tid) & (pl.col("party") == minority_party))

        maj_rice = float(maj_row["rice_index"][0]) if len(maj_row) > 0 else 1.0
        min_rice = float(min_row["rice_index"][0]) if len(min_row) > 0 else 1.0
        n_bills = int(maj_row["n_bills"][0]) if len(maj_row) > 0 else 0

        caucus_splitting.append(
            {
                "topic_id": tid,
                "topic_label": tlabel,
                "majority_rice": maj_rice,
                "minority_rice": min_rice,
                "split_score": 1.0 - maj_rice,
                "n_bills": n_bills,
            }
        )

    # Sort by split score descending (most splitting first)
    caucus_splitting.sort(key=lambda x: x["split_score"], reverse=True)

    return grouped, caucus_splitting


def compute_cap_passage_rates(
    cap_df: pl.DataFrame,
    rollcalls: pl.DataFrame,
) -> pl.DataFrame:
    """Compute passage rate per CAP category."""
    if len(cap_df) == 0:
        return pl.DataFrame()

    # Join CAP to rollcalls on bill_number
    if "passed" not in rollcalls.columns:
        return pl.DataFrame()

    merged = cap_df.join(
        rollcalls.select(["bill_number", "vote_id", "passed"]),
        on="bill_number",
        how="inner",
    )

    if len(merged) == 0:
        return pl.DataFrame()

    return (
        merged.group_by("cap_label")
        .agg(
            pl.col("bill_number").n_unique().alias("n_bills"),
            pl.col("vote_id").n_unique().alias("n_votes"),
            (
                pl.col("passed").filter(pl.col("passed")).len().cast(pl.Float64)
                / pl.col("passed").filter(pl.col("passed").is_not_null()).len()
            ).alias("passage_rate"),
        )
        .sort("n_bills", descending=True)
    )


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_topic_distribution(
    topic_info: list[dict],
    plots_dir: Path,
    suffix: str = "all",
) -> None:
    """Bar chart of topic sizes."""
    # Exclude noise cluster for cleaner plot
    topics = [t for t in topic_info if t["topic_id"] != -1]
    if not topics:
        return

    topics.sort(key=lambda x: x["count"], reverse=True)
    labels = [t["topic_label"][:50] for t in topics]
    counts = [t["count"] for t in topics]

    fig, ax = plt.subplots(figsize=(12, max(6, len(topics) * 0.4)))
    bars = ax.barh(range(len(labels)), counts, color="#4C72B0")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Bills")
    chamber_label = {"all": "All Bills", "house": "House", "senate": "Senate"}.get(suffix, suffix)
    ax.set_title(f"BERTopic: Topic Distribution ({chamber_label})")

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    save_fig(fig, plots_dir / f"topic_distribution_{suffix}.png")


def plot_topic_party_heatmap(
    cohesion: pl.DataFrame,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Heatmap: Rice index per topic × party."""
    if len(cohesion) == 0:
        return

    # Pivot to topic × party matrix
    pivot = cohesion.pivot(on="party", index="topic_label", values="rice_index")
    if len(pivot) == 0:
        return

    party_cols = [c for c in pivot.columns if c != "topic_label"]
    if not party_cols:
        return

    labels = pivot["topic_label"].to_list()
    data = pivot.select(party_cols).to_numpy()

    fig, ax = plt.subplots(figsize=(8, max(5, len(labels) * 0.4)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(party_cols)))
    ax.set_xticklabels(party_cols)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([lbl[:50] for lbl in labels], fontsize=8)
    ax.set_title(f"Topic × Party Rice Index ({chamber.title()})")

    # Add value annotations
    for i in range(len(labels)):
        for j in range(len(party_cols)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, label="Rice Index")
    plt.tight_layout()
    save_fig(fig, plots_dir / f"topic_party_heatmap_{chamber}.png")


def plot_caucus_splitting(
    caucus_splitting: list[dict],
    plots_dir: Path,
    chamber: str,
) -> None:
    """Horizontal bar chart: topics ranked by caucus-splitting score."""
    if not caucus_splitting:
        return

    # Top 15 most splitting
    data = caucus_splitting[:15]
    labels = [d["topic_label"][:50] for d in data]
    scores = [d["split_score"] for d in data]

    fig, ax = plt.subplots(figsize=(10, max(5, len(data) * 0.4)))
    colors = ["#E81B23" if s > 0.3 else "#4C72B0" for s in scores]
    ax.barh(range(len(labels)), scores, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Caucus-Splitting Score (1 − majority Rice)")
    ax.set_title(f"Most Caucus-Splitting Topics ({chamber.title()})")
    ax.axvline(0.3, color="gray", linestyle="--", alpha=0.5, label="High dissent (>0.30)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, plots_dir / f"caucus_splitting_topics_{chamber}.png")


def plot_cap_distribution(
    cap_df: pl.DataFrame,
    plots_dir: Path,
) -> None:
    """Bar chart: CAP 20-category distribution."""
    if len(cap_df) == 0:
        return

    counts = cap_df.group_by("cap_label").len().sort("len", descending=True)
    labels = counts["cap_label"].to_list()
    values = counts["len"].to_list()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(labels)), values, color="#4C72B0")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Bills")
    ax.set_title("CAP Policy Classification (Claude Sonnet)")

    for i, v in enumerate(values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=8)

    plt.tight_layout()
    save_fig(fig, plots_dir / "cap_category_distribution.png")


def plot_cap_party_breakdown(
    cap_df: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
    plots_dir: Path,
) -> None:
    """Stacked bar: CAP categories by sponsor party."""
    if len(cap_df) == 0 or "sponsor_slugs" not in rollcalls.columns:
        return

    # Get first sponsor party per bill
    bill_party = (
        rollcalls.filter(pl.col("sponsor_slugs").is_not_null() & (pl.col("sponsor_slugs") != ""))
        .with_columns(pl.col("sponsor_slugs").str.split("; ").list.first().alias("first_sponsor"))
        .join(
            legislators.select(["legislator_slug", "party"]),
            left_on="first_sponsor",
            right_on="legislator_slug",
            how="left",
        )
        .select(["bill_number", "party"])
        .unique(subset=["bill_number"], keep="first")
    )

    merged = cap_df.join(bill_party, on="bill_number", how="inner")
    if len(merged) == 0:
        return

    # Fill empty party
    merged = merged.with_columns(
        pl.when(pl.col("party").is_null() | (pl.col("party") == ""))
        .then(pl.lit("Independent"))
        .otherwise(pl.col("party"))
        .alias("party")
    )

    pivot = merged.group_by(["cap_label", "party"]).len().sort("len", descending=True)

    # Create stacked bar
    categories = pivot.select("cap_label").unique().sort("cap_label")["cap_label"].to_list()
    parties = sorted(pivot.select("party").unique()["party"].to_list())

    fig, ax = plt.subplots(figsize=(12, 8))
    bottoms = np.zeros(len(categories))
    for party in parties:
        party_data = pivot.filter(pl.col("party") == party)
        values = []
        for cat in categories:
            row = party_data.filter(pl.col("cap_label") == cat)
            values.append(int(row["len"][0]) if len(row) > 0 else 0)
        color = PARTY_COLORS.get(party, "#999999")
        ax.barh(range(len(categories)), values, left=bottoms, color=color, label=party)
        bottoms += np.array(values)

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Bills")
    ax.set_title("CAP Categories by Sponsor Party")
    ax.legend()

    plt.tight_layout()
    save_fig(fig, plots_dir / "cap_party_breakdown.png")


def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    bill_numbers: list[str],
    plots_dir: Path,
    top_n: int = HEATMAP_TOP_N,
) -> None:
    """Clustered similarity heatmap of top-N most-connected bills."""
    n = len(bill_numbers)
    if n == 0:
        return

    # Select top-N bills by average similarity to others
    avg_sim = np.mean(sim_matrix, axis=1)
    top_idx = np.argsort(avg_sim)[-min(top_n, n) :]
    sub_matrix = sim_matrix[np.ix_(top_idx, top_idx)]
    sub_labels = [bill_numbers[i] for i in top_idx]

    # Hierarchical clustering for ordering
    dist = 1 - sub_matrix
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)  # Fix floating point issues
    condensed = squareform(dist)
    if len(condensed) > 0:
        Z = linkage(condensed, method="average")
        dn = dendrogram(Z, no_plot=True)
        order = dn["leaves"]
        sub_matrix = sub_matrix[np.ix_(order, order)]
        sub_labels = [sub_labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sub_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    if len(sub_labels) <= 30:
        ax.set_xticks(range(len(sub_labels)))
        ax.set_xticklabels(sub_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(sub_labels)))
        ax.set_yticklabels(sub_labels, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(f"Bill Similarity (top {len(sub_labels)} by avg similarity)")
    fig.colorbar(im, ax=ax, label="Cosine Similarity")

    plt.tight_layout()
    save_fig(fig, plots_dir / "bill_similarity_heatmap.png")


# ── Filtering Manifest ───────────────────────────────────────────────────────


def save_filtering_manifest(manifest: dict, out_dir: Path) -> None:
    """Save filtering/processing manifest for reproducibility."""
    path = out_dir / "filtering_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Saved: {path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from tallgrass.session import KSSession

    session = KSSession.from_session_string(args.session)
    data_dir = Path(args.data_dir) if args.data_dir else session.data_dir
    results_root = session.results_dir

    # Check bill texts exist (run `just text` first)
    bill_text_csv = data_dir / f"{session.output_name}_bill_texts.csv"
    if not bill_text_csv.exists():
        from analysis.db import db_available

        if args.csv or not db_available():
            print(
                f"[Phase 20] Skipping: bill texts not found "
                f"(run `just text {args.session}` first)"
            )
            return

    with RunContext(
        session=args.session,
        analysis_name="20_bill_text",
        params=vars(args),
        primer=BILL_TEXT_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print_header("Phase 20: Bill Text Analysis")

        # ── Load data ────────────────────────────────────────────────────
        print_header("Loading Data")

        try:
            bill_texts = load_bill_texts(data_dir, use_csv=args.csv)
        except FileNotFoundError:
            print("  No bill texts available — skipping Phase 20.")
            return
        print(f"  Bills with text: {len(bill_texts)}")

        rollcalls = load_rollcalls(data_dir, use_csv=args.csv)
        votes = load_votes(data_dir, use_csv=args.csv)
        legislators = load_legislators(data_dir, use_csv=args.csv)
        if "slug" in legislators.columns and "legislator_slug" not in legislators.columns:
            legislators = legislators.rename({"slug": "legislator_slug"})
        print(f"  Roll calls: {len(rollcalls)}")
        print(f"  Legislators: {len(legislators)}")

        # Text source breakdown
        text_source_counts: dict[str, int] = {}
        if "text_source" in bill_texts.columns:
            for row in bill_texts.group_by("text_source").len().iter_rows(named=True):
                text_source_counts[row["text_source"]] = row["len"]
        print(f"  Text sources: {text_source_counts}")

        # ── Preprocess + Embed ───────────────────────────────────────────
        print_header("Text Preprocessing & Embedding")

        bill_numbers = bill_texts["bill_number"].to_list()
        raw_texts = bill_texts["text"].to_list()
        processed_texts = [preprocess_for_embedding(t) for t in raw_texts]

        avg_text_length = sum(len(t) for t in processed_texts) / max(len(processed_texts), 1)
        print(f"  Average processed text length: {avg_text_length:,.0f} chars")

        embeddings = get_or_compute_embeddings(
            processed_texts,
            bill_numbers,
            cache_dir=ctx.data_dir,
            model_name=args.embedding_model,
        )

        # Save embeddings
        dim_cols = {f"dim_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
        emb_df = pl.DataFrame({"bill_number": bill_numbers, **dim_cols})
        emb_df.write_parquet(ctx.data_dir / "bill_embeddings.parquet")
        print(f"  Saved: bill_embeddings.parquet ({embeddings.shape[1]}-dim)")

        # ── Assign chambers ──────────────────────────────────────────────
        chamber_dfs = assign_bills_to_chambers(bill_texts, rollcalls)

        # ── Topic Modeling ───────────────────────────────────────────────
        print_header("BERTopic Topic Modeling")

        topic_model, topics, probs = fit_topic_model(
            processed_texts,
            embeddings,
            min_cluster_size=args.min_cluster_size,
        )

        bill_topics, topic_info = extract_topic_info(topic_model, topics, bill_numbers)

        n_topics = len([t for t in topic_info if t["topic_id"] != -1])
        n_noise = len([t for t in topics if t == -1])
        print(f"  Topics discovered: {n_topics}")
        print(f"  Noise points: {n_noise} ({n_noise / len(topics):.1%})")

        # Save topic assignments
        bill_topics.write_parquet(ctx.data_dir / "bill_topics_all.parquet")
        ctx.export_csv(bill_topics, "bill_topics.csv", "Topic assignments per bill")

        # Save per-chamber
        for chamber_key in ["house", "senate"]:
            if chamber_key in chamber_dfs:
                chamber_bills = set(chamber_dfs[chamber_key]["bill_number"].to_list())
                chamber_topics = bill_topics.filter(pl.col("bill_number").is_in(chamber_bills))
                if len(chamber_topics) > 0:
                    chamber_topics.write_parquet(
                        ctx.data_dir / f"bill_topics_{chamber_key}.parquet"
                    )

        # Save topic model info
        with open(ctx.data_dir / "topic_model_info.json", "w") as f:
            json.dump(topic_info, f, indent=2)
        print("  Saved: topic_model_info.json")

        # Plot topic distribution
        plot_topic_distribution(topic_info, ctx.plots_dir, "all")

        # ── CAP Classification (optional) ────────────────────────────────
        has_cap = False
        cap_df = pl.DataFrame()

        if args.classify:
            print_header("CAP Classification")

            if args.batch:
                batch_id = classify_bills_cap_batch(
                    raw_texts,
                    bill_numbers,
                    cache_path=ctx.data_dir / "cap_cache.json",
                )
                if batch_id:
                    print(f"  Batch submitted: {batch_id}")
                    print("  Retrieve results later and re-run without --batch")
            else:
                cap_df = classify_bills_cap(
                    raw_texts,
                    bill_numbers,
                    cache_path=ctx.data_dir / "cap_cache.json",
                )
                if len(cap_df) > 0:
                    has_cap = True
                    cap_df.write_parquet(ctx.data_dir / "cap_classifications.parquet")
                    ctx.export_csv(
                        cap_df,
                        "cap_classifications.csv",
                        "CAP policy classifications per bill",
                    )
                    plot_cap_distribution(cap_df, ctx.plots_dir)
                    plot_cap_party_breakdown(cap_df, rollcalls, legislators, ctx.plots_dir)

        # ── Bill Similarity ──────────────────────────────────────────────
        print_header("Bill Similarity")

        sim_matrix, top_pairs = compute_bill_similarity(embeddings, bill_numbers)
        print(f"  Pairs above {SIMILARITY_THRESHOLD}: {len(top_pairs)}")
        plot_similarity_heatmap(sim_matrix, bill_numbers, ctx.plots_dir)

        # ── Vote Cross-Reference ─────────────────────────────────────────
        print_header("Vote Cross-Reference")

        all_caucus_splitting: list[dict] = []

        for chamber_key in ["house", "senate"]:
            if chamber_key not in chamber_dfs:
                continue

            chamber_bills = set(chamber_dfs[chamber_key]["bill_number"].to_list())
            chamber_topics = bill_topics.filter(pl.col("bill_number").is_in(chamber_bills))

            if len(chamber_topics) == 0:
                continue

            # Filter votes to this chamber
            chamber_prefix = "sen_" if chamber_key == "senate" else "rep_"
            chamber_votes = votes.filter(pl.col("legislator_slug").str.starts_with(chamber_prefix))
            chamber_legislators = legislators.filter(
                pl.col("legislator_slug").str.starts_with(chamber_prefix)
            )

            cohesion, splitting = compute_topic_party_cohesion(
                chamber_topics,
                rollcalls,
                chamber_votes,
                chamber_legislators,
            )
            all_caucus_splitting.extend(splitting)

            if len(cohesion) > 0:
                plot_topic_party_heatmap(cohesion, ctx.plots_dir, chamber_key)
                plot_caucus_splitting(splitting, ctx.plots_dir, chamber_key)
                plot_topic_distribution(
                    [
                        t
                        for t in topic_info
                        if t["topic_id"] in chamber_topics["topic_id"].unique().to_list()
                    ],
                    ctx.plots_dir,
                    chamber_key,
                )

        # Sort overall caucus splitting
        all_caucus_splitting.sort(key=lambda x: x["split_score"], reverse=True)

        # Export cohesion CSV
        if all_caucus_splitting:
            cohesion_df = pl.DataFrame(all_caucus_splitting)
            ctx.export_csv(
                cohesion_df,
                "topic_party_cohesion.csv",
                "Per-topic Rice index and caucus-splitting scores",
            )

        # ── CAP passage rates ────────────────────────────────────────────
        cap_passage = pl.DataFrame()
        if has_cap:
            cap_passage = compute_cap_passage_rates(cap_df, rollcalls)

        # ── Build Report ─────────────────────────────────────────────────
        print_header("Building Report")

        results = {
            "n_bills_analyzed": len(bill_texts),
            "n_topics": n_topics,
            "text_source_counts": text_source_counts,
            "avg_text_length": avg_text_length,
            "embedding_model": args.embedding_model,
            "embedding_dim": embeddings.shape[1],
            "topic_info": topic_info,
            "caucus_splitting": all_caucus_splitting,
            "top_similar_pairs": top_pairs,
            "cap_classifications": cap_df if has_cap else None,
            "cap_passage_rates": cap_passage if has_cap and len(cap_passage) > 0 else None,
            "parameters": {
                "embedding_model": args.embedding_model,
                "min_cluster_size": args.min_cluster_size,
                "umap_n_components": UMAP_N_COMPONENTS,
                "umap_n_neighbors": UMAP_N_NEIGHBORS,
                "umap_min_dist": UMAP_MIN_DIST,
                "hdbscan_min_samples": MIN_SAMPLES,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "random_seed": RANDOM_SEED,
                "cap_classification": has_cap,
            },
        }

        build_bill_text_report(
            ctx.report,
            results=results,
            plots_dir=ctx.plots_dir,
            has_cap=has_cap,
        )

        # ── Filtering Manifest ───────────────────────────────────────────
        save_filtering_manifest(
            {
                "phase": "20_bill_text",
                "session": args.session,
                "n_bills_total": len(bill_texts),
                "n_bills_with_supp_note": text_source_counts.get("supp_note", 0),
                "n_bills_introduced_only": text_source_counts.get("introduced", 0),
                "n_topics": n_topics,
                "n_noise_bills": n_noise,
                "embedding_model": args.embedding_model,
                "embedding_dim": embeddings.shape[1],
                "min_cluster_size": args.min_cluster_size,
                "cap_classified": has_cap,
                "n_similar_pairs_above_threshold": len(top_pairs),
            },
            ctx.run_dir,
        )

        print_header("Phase 18 Complete")


if __name__ == "__main__":
    main()
