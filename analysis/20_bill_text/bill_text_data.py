"""Bill text data loading, preprocessing, and embedding cache (Phase 18).

Pure functions for text preprocessing plus FastEmbed-based embedding with
parquet caching.  No plotting, no report building — just data in, data out.

Text selection: prefers supplemental notes (shorter, plain-English summaries)
over introduced text when both exist for the same bill.

Embedding model: BAAI/bge-small-en-v1.5 (384-dim) via FastEmbed (ONNX Runtime).
No PyTorch dependency — FastEmbed uses ONNX for inference (~50-100 MB).
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import polars as pl

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
"""384-dim BGE model via FastEmbed. Good quality-to-size ratio for legislative text."""

MAX_TOKENS_APPROX = 8000
"""Approximate character limit for embedding input (~2000 tokens × 4 chars).
bge-small-en-v1.5 supports 512 tokens; we truncate to leave headroom after
tokenization.  Better to truncate than embed garbage from overflow."""

# Boilerplate patterns to strip before embedding
_ENACTING_CLAUSE = re.compile(
    r"Be it enacted by the Legislature of the State of Kansas[:\s]*",
    re.IGNORECASE,
)
_SEVERABILITY = re.compile(
    r"(?:If any provision|Severability)[^\n]*(?:\n[^\n]*){0,5}",
    re.IGNORECASE,
)
_EFFECTIVE_DATE = re.compile(
    r"This act shall take effect[^\n]*(?:\n[^\n]*){0,3}",
    re.IGNORECASE,
)
_KSA_REFERENCE = re.compile(r"K\.?S\.?A\.?\s*\d[\d\-a-z]*", re.IGNORECASE)
_SECTION_HEADER = re.compile(r"(?:Section|Sec\.)\s+\d+\.", re.IGNORECASE)
_WHITESPACE_RUNS = re.compile(r"\s{3,}")
_PAGE_NUMBERS = re.compile(r"\n\s*\d+\s*\n")


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_bill_texts(data_dir: Path) -> pl.DataFrame:
    """Load bill_texts.csv, prefer supp_note over introduced when both exist.

    Returns a DataFrame with columns: bill_number, text, document_type,
    text_source, page_count, source_url.  One row per bill (deduplicated).
    """
    prefix = data_dir.name
    csv_path = data_dir / f"{prefix}_bill_texts.csv"
    if not csv_path.exists():
        msg = f"Bill texts CSV not found: {csv_path}. Run `just text` first."
        raise FileNotFoundError(msg)

    df = pl.read_csv(csv_path)

    # Prefer supp_note over introduced for each bill
    # Sort so supp_note comes first (alphabetically "s" > "i"), then deduplicate
    df = (
        df.sort(["bill_number", "document_type"], descending=[False, True])
        .unique(subset=["bill_number"], keep="first")
        .with_columns(
            pl.when(pl.col("document_type") == "supp_note")
            .then(pl.lit("supp_note"))
            .otherwise(pl.lit("introduced"))
            .alias("text_source"),
        )
    )

    # Drop rows with empty/null text
    df = df.filter(pl.col("text").is_not_null() & (pl.col("text").str.len_chars() > 0))

    return df


def load_rollcalls(data_dir: Path) -> pl.DataFrame:
    """Load rollcalls CSV with standard columns."""
    prefix = data_dir.name
    return pl.read_csv(data_dir / f"{prefix}_rollcalls.csv")


def load_votes(data_dir: Path) -> pl.DataFrame:
    """Load votes CSV with column rename (slug -> legislator_slug)."""
    prefix = data_dir.name
    df = pl.read_csv(data_dir / f"{prefix}_votes.csv")
    if "slug" in df.columns and "legislator_slug" not in df.columns:
        df = df.rename({"slug": "legislator_slug"})
    return df


# ── Text Preprocessing ───────────────────────────────────────────────────────


def preprocess_for_embedding(text: str) -> str:
    """Strip legislative boilerplate, normalize references, truncate.

    Designed for embedding quality: reduces vocabulary noise from repeated
    statutory references and procedural language.
    """
    if not text:
        return ""

    # Strip common boilerplate
    text = _ENACTING_CLAUSE.sub("", text)
    text = _SEVERABILITY.sub("", text)
    text = _EFFECTIVE_DATE.sub("", text)

    # Normalize K.S.A. references to reduce vocabulary noise
    text = _KSA_REFERENCE.sub("STATUTE_REF", text)

    # Clean up section headers (less noisy but keep structure)
    text = _SECTION_HEADER.sub("Section:", text)

    # Remove page numbers embedded in text
    text = _PAGE_NUMBERS.sub("\n", text)

    # Collapse whitespace runs
    text = _WHITESPACE_RUNS.sub(" ", text)
    text = text.strip()

    # Truncate to approximate token limit
    if len(text) > MAX_TOKENS_APPROX:
        text = text[:MAX_TOKENS_APPROX]

    return text


# ── Embedding ────────────────────────────────────────────────────────────────


def _compute_cache_key(
    model_name: str,
    bill_numbers: list[str],
    texts: list[str],
) -> str:
    """SHA-256 hash of model + bill content for cache invalidation."""
    h = hashlib.sha256()
    h.update(model_name.encode())
    for bn, t in sorted(zip(bill_numbers, texts, strict=True)):
        h.update(bn.encode())
        h.update(t.encode())
    return h.hexdigest()[:16]


def get_or_compute_embeddings(
    texts: list[str],
    bill_numbers: list[str],
    cache_dir: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """Embed texts via FastEmbed, cache to parquet.  Reuse cache if exists.

    Args:
        texts: Preprocessed bill texts (one per bill).
        bill_numbers: Corresponding bill numbers (same order as texts).
        cache_dir: Directory for caching embeddings parquet.
        model_name: FastEmbed model identifier.

    Returns:
        NumPy array of shape (n_bills, embedding_dim).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _compute_cache_key(model_name, bill_numbers, texts)
    cache_path = cache_dir / f"embeddings_{cache_key}.parquet"

    if cache_path.exists():
        print(f"  Loading cached embeddings: {cache_path.name}")
        cached = pl.read_parquet(cache_path)
        # Verify bill numbers match
        cached_bills = cached["bill_number"].to_list()
        if cached_bills == sorted(bill_numbers):
            # Reorder to match input order
            order_map = {bn: i for i, bn in enumerate(bill_numbers)}
            cached = (
                cached.with_columns(
                    pl.col("bill_number")
                    .map_elements(lambda bn: order_map.get(bn, 0), return_dtype=pl.Int64)
                    .alias("_order")
                )
                .sort("_order")
                .drop("_order")
            )
            embedding_cols = [c for c in cached.columns if c.startswith("dim_")]
            return cached.select(embedding_cols).to_numpy()

    print(f"  Computing embeddings with {model_name}...")
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=model_name)

    # FastEmbed returns a generator — collect to array
    embeddings_list = list(model.embed(texts))
    embeddings = np.array(embeddings_list)
    print(f"  Embedding shape: {embeddings.shape}")

    # Cache to parquet (sorted by bill_number for stable cache key)
    sort_idx = sorted(range(len(bill_numbers)), key=lambda i: bill_numbers[i])
    sorted_bills = [bill_numbers[i] for i in sort_idx]
    sorted_embeddings = embeddings[sort_idx]

    dim_cols = {f"dim_{i}": sorted_embeddings[:, i] for i in range(embeddings.shape[1])}
    cache_df = pl.DataFrame({"bill_number": sorted_bills, **dim_cols})
    cache_df.write_parquet(cache_path)
    print(f"  Cached embeddings: {cache_path.name}")

    return embeddings


# ── Bill-Chamber Mapping ─────────────────────────────────────────────────────


def assign_bills_to_chambers(
    bill_texts: pl.DataFrame,
    rollcalls: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Split bill texts by chamber based on rollcall data.

    Bills that appear in both chambers are assigned to the chamber with more
    roll call votes.  Bills with no rollcall match are included in an "all"
    group (used for topic modeling on full corpus).

    Returns dict with keys "house", "senate", "all".
    """
    # Get chamber per bill from rollcalls (bill with most votes in that chamber)
    if "chamber" in rollcalls.columns:
        bill_chamber = (
            rollcalls.group_by(["bill_number", "chamber"])
            .len()
            .sort("len", descending=True)
            .unique(subset=["bill_number"], keep="first")
            .select(["bill_number", "chamber"])
        )
    else:
        # Infer chamber from vote_id prefix if chamber column missing
        bill_chamber = (
            rollcalls.with_columns(
                pl.when(pl.col("vote_id").str.starts_with("je_"))
                .then(pl.lit("Senate"))
                .otherwise(pl.lit("House"))
                .alias("chamber")
            )
            .group_by(["bill_number", "chamber"])
            .len()
            .sort("len", descending=True)
            .unique(subset=["bill_number"], keep="first")
            .select(["bill_number", "chamber"])
        )

    # Join chamber info to bill texts
    merged = bill_texts.join(bill_chamber, on="bill_number", how="left")

    result: dict[str, pl.DataFrame] = {"all": bill_texts}
    for chamber in ["House", "Senate"]:
        chamber_df = merged.filter(pl.col("chamber") == chamber).drop("chamber")
        if len(chamber_df) > 0:
            result[chamber.lower()] = chamber_df

    return result
