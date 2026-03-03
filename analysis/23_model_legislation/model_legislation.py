"""
Kansas Legislature — Model Legislation Detection (Phase 20, BT5)

Detects Kansas bills that match known ALEC model legislation (template matching)
and bills that appear in neighboring states (cross-state diffusion).

Uses the same BGE embeddings as Phase 18 for cosine similarity, with n-gram
overlap as secondary confirmation for high-similarity pairs.

Usage:
  uv run python analysis/20_model_legislation/model_legislation.py
  uv run python analysis/20_model_legislation/model_legislation.py --session 2025-26
  uv run python analysis/20_model_legislation/model_legislation.py --alec-only
  uv run python analysis/20_model_legislation/model_legislation.py --states MO,OK,NE,CO

Outputs (in results/<session>/<run_id>/20_model_legislation/):
  - data/:   Parquet files (match tables, similarity matrices)
  - plots/:  PNG histograms, heatmaps
  - 20_model_legislation_report.html
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir

try:
    from analysis.phase_utils import print_header, save_fig
except ModuleNotFoundError:
    from phase_utils import print_header, save_fig  # type: ignore[no-redef]

try:
    from analysis.model_legislation_report import build_model_legislation_report
except ModuleNotFoundError:
    from model_legislation_report import build_model_legislation_report  # type: ignore[no-redef]

try:
    from analysis.model_legislation_data import (
        NGRAM_SIZE,
        THRESHOLD_NEAR_IDENTICAL,
        THRESHOLD_RELATED,
        THRESHOLD_STRONG_MATCH,
        build_match_summary,
        compute_cross_similarity,
        compute_ngram_overlap,
        load_alec_corpus,
        load_cross_state_texts,
    )
except ModuleNotFoundError:
    from model_legislation_data import (  # type: ignore[no-redef]
        NGRAM_SIZE,
        THRESHOLD_NEAR_IDENTICAL,
        THRESHOLD_RELATED,
        THRESHOLD_STRONG_MATCH,
        build_match_summary,
        compute_cross_similarity,
        compute_ngram_overlap,
        load_alec_corpus,
        load_cross_state_texts,
    )

# Lazy imports for Phase 18 data utilities
_bill_text_data = None


def _get_bill_text_data():  # type: ignore[no-untyped-def]
    global _bill_text_data
    if _bill_text_data is None:
        try:
            from analysis import bill_text_data as btd
        except ModuleNotFoundError:
            # Direct import fallback
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "20_bill_text"))
            import bill_text_data as btd  # type: ignore[no-redef]
        _bill_text_data = btd
    return _bill_text_data


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_STATES = ["mo", "ok", "ne", "co"]
ALEC_DATA_DIR = Path("data/external/alec")
EXCERPT_LENGTH = 500  # chars for detail card excerpts

MODEL_LEGISLATION_PRIMER = """\
## Model Legislation Detection (Phase 20)

**Purpose:** Identify Kansas bills that resemble known model legislation
(ALEC templates) or appear in neighboring states (policy diffusion).

**Method:** Cosine similarity on 384-dim BGE embeddings (same model as
Phase 18 bill text analysis). N-gram overlap confirms text reuse for
high-similarity pairs.

**Thresholds:**
- >= 0.95: Near-identical (likely direct copy or adaptation)
- >= 0.85: Strong match (adapted from same source)
- >= 0.70: Related (similar policy area, warrants investigation)
- 5-gram overlap >= 20%: Confirms genuine text sharing

**Inputs:** Kansas bill texts (from BT1), ALEC model policy corpus,
neighbor state bill texts (optional).

**Interpretation:** High similarity does not prove causation — bills may
independently address the same policy problem with similar language.
Near-identical matches (>= 0.95) with high n-gram overlap are the
strongest evidence of model legislation adoption.
"""


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_similarity_distribution(
    max_similarities: np.ndarray,
    labels: list[str],
    plots_dir: Path,
) -> None:
    """Histogram of maximum similarity scores per Kansas bill."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(max_similarities, bins=50, color="#4a90d9", edgecolor="white", alpha=0.8)

    # Add threshold lines
    for thresh, label, color in [
        (THRESHOLD_RELATED, "Related (0.70)", "#FFA500"),
        (THRESHOLD_STRONG_MATCH, "Strong (0.85)", "#FF4500"),
        (THRESHOLD_NEAR_IDENTICAL, "Near-identical (0.95)", "#DC143C"),
    ]:
        ax.axvline(x=thresh, color=color, linestyle="--", linewidth=1.5, label=label)

    ax.set_xlabel("Maximum Cosine Similarity")
    ax.set_ylabel("Number of Kansas Bills")
    ax.set_title("Distribution of Best-Match Similarity Scores")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1.05)

    save_fig(fig, plots_dir / "similarity_distribution.png")
    plt.close(fig)


def plot_topic_match_heatmap(
    match_summary: pl.DataFrame,
    plots_dir: Path,
) -> None:
    """Heatmap showing match counts by topic and source."""
    if len(match_summary) == 0:
        return

    # Filter to rows with topics
    with_topics = match_summary.filter(pl.col("topic").is_not_null() & (pl.col("topic") != ""))
    if len(with_topics) == 0:
        return

    # Pivot: topic x source
    pivot = (
        with_topics.group_by(["topic", "source"])
        .len()
        .pivot(on="source", index="topic", values="len")
        .fill_null(0)
    )

    # Sort by total matches
    sources = [c for c in pivot.columns if c != "topic"]
    if not sources:
        return

    pivot = (
        pivot.with_columns(pl.sum_horizontal([pl.col(s) for s in sources]).alias("_total"))
        .sort("_total", descending=True)
        .drop("_total")
    )

    # Limit to top 15 topics for readability
    pivot = pivot.head(15)

    topics = pivot["topic"].to_list()
    data = pivot.select(sources).to_numpy()

    fig, ax = plt.subplots(figsize=(10, max(4, len(topics) * 0.4)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontsize=8)

    # Add value annotations
    for i in range(len(topics)):
        for j in range(len(sources)):
            val = int(data[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Number of Matches")
    ax.set_title("Model Legislation Matches by Policy Area")
    ax.set_xlabel("Source")

    save_fig(fig, plots_dir / "topic_match_heatmap.png")
    plt.close(fig)


def plot_match_network(
    match_summary: pl.DataFrame,
    plots_dir: Path,
) -> None:
    """Network diagram for Kansas bills matched in multiple states."""
    # Find bills matched in 2+ sources
    multi_source = (
        match_summary.group_by("ks_bill")
        .agg(pl.col("source").n_unique().alias("n_sources"))
        .filter(pl.col("n_sources") >= 2)
    )
    if len(multi_source) == 0:
        return

    multi_bills = set(multi_source["ks_bill"].to_list())
    multi_matches = match_summary.filter(pl.col("ks_bill").is_in(multi_bills))

    try:
        import networkx as nx
    except ImportError:
        return

    G = nx.Graph()

    for row in multi_matches.iter_rows(named=True):
        ks = row["ks_bill"]
        source = row["source"]
        sim = row["similarity"]

        if not G.has_node(ks):
            G.add_node(ks, node_type="kansas")
        if not G.has_node(source):
            G.add_node(source, node_type="source")

        G.add_edge(ks, source, weight=sim)

    if len(G.nodes) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, seed=42)

    # Draw source nodes (larger)
    source_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "source"]
    ks_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "kansas"]

    nx.draw_networkx_nodes(
        G, pos, nodelist=source_nodes, node_color="#FF6B35", node_size=800, ax=ax, label="Source"
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=ks_nodes, node_color="#4A90D9", node_size=400, ax=ax, label="Kansas Bill"
    )

    # Draw edges with width proportional to similarity
    edges = G.edges(data=True)
    widths = [d.get("weight", 0.7) * 3 for _, _, d in edges]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    ax.set_title("Kansas Bills Matched in Multiple Sources")
    ax.legend(loc="upper left")
    ax.axis("off")

    save_fig(fig, plots_dir / "match_network.png")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="model-legislation",
        description="Detect Kansas bills matching ALEC model legislation and neighbor state bills.",
    )
    parser.add_argument("--session", default="2025-26", help="Biennium string (default: 2025-26)")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--states",
        default=",".join(DEFAULT_STATES),
        help=f"Comma-separated neighbor state abbreviations (default: {','.join(DEFAULT_STATES)})",
    )
    parser.add_argument("--alec-only", action="store_true", help="Skip cross-state comparison")
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD_RELATED,
        help=f"Minimum similarity threshold (default: {THRESHOLD_RELATED})",
    )
    parser.add_argument(
        "--alec-dir",
        type=Path,
        default=ALEC_DATA_DIR,
        help=f"ALEC corpus directory (default: {ALEC_DATA_DIR})",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="FastEmbed model (default: same as Phase 18)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    btd = _get_bill_text_data()

    from tallgrass.session import KSSession

    session = KSSession.from_session_string(args.session)
    data_dir = session.data_dir
    results_root = session.results_dir.parent  # up from session-specific dir

    # Check prerequisites: bill texts + ALEC corpus
    bill_text_csvs = list(data_dir.glob("*_bill_texts.csv"))
    alec_csv = Path("data/external/alec/alec_model_bills.csv")
    if not bill_text_csvs and not alec_csv.exists():
        print("[Phase 23] Skipping: no bill texts or ALEC corpus (run `just text` + `just alec`)")
        return

    embedding_model = args.embedding_model or btd.DEFAULT_EMBEDDING_MODEL
    threshold = args.threshold
    states = [s.strip().lower() for s in args.states.split(",") if s.strip()]

    with RunContext(
        session=args.session,
        analysis_name="23_model_legislation",
        params=vars(args),
        results_root=results_root,
        primer=MODEL_LEGISLATION_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print_header("Model Legislation Detection (Phase 23)")

        # ── Step 1: Load Kansas bill texts + embeddings ──────────────────
        print("\n[Step 1] Loading Kansas bill texts...")
        ks_bill_texts = btd.load_bill_texts(data_dir)
        n_ks = len(ks_bill_texts)
        print(f"  Loaded {n_ks} Kansas bill texts")

        ks_bills = ks_bill_texts["bill_number"].to_list()
        ks_raw_texts = ks_bill_texts["text"].to_list()
        ks_processed = [btd.preprocess_for_embedding(t) for t in ks_raw_texts]

        print("  Computing Kansas embeddings...")
        ks_embeddings = btd.get_or_compute_embeddings(
            ks_processed, ks_bills, cache_dir=ctx.data_dir, model_name=embedding_model
        )
        print(f"  Embedding shape: {ks_embeddings.shape}")

        # Load topic assignments from Phase 18 if available
        topic_assignments = None
        try:
            upstream_dir = resolve_upstream_dir(
                session=args.session,
                analysis_name="20_bill_text",
                results_root=results_root,
                run_id=args.run_id,
            )
            topic_path = upstream_dir / "data" / "bill_topics.parquet"
            if topic_path.exists():
                topic_assignments = pl.read_parquet(topic_path)
                print(f"  Loaded {len(topic_assignments)} topic assignments from Phase 18")
        except Exception:
            print("  No Phase 18 topic assignments found (continuing without)")

        # ── Step 2: Load and embed ALEC corpus ───────────────────────────
        print("\n[Step 2] Loading ALEC corpus...")
        try:
            alec_texts, alec_ids, alec_metadata = load_alec_corpus(args.alec_dir)
            n_alec = len(alec_texts)
            print(f"  Loaded {n_alec} ALEC model bills")

            alec_processed = [btd.preprocess_for_embedding(t) for t in alec_texts]
            print("  Computing ALEC embeddings...")
            alec_embeddings = btd.get_or_compute_embeddings(
                alec_processed, alec_ids, cache_dir=ctx.data_dir, model_name=embedding_model
            )
        except FileNotFoundError as e:
            print(f"  {e}")
            alec_texts, alec_ids, alec_metadata = [], [], pl.DataFrame()
            alec_embeddings = np.empty((0, ks_embeddings.shape[1]))
            n_alec = 0

        # ── Step 3: Compute ALEC similarity ──────────────────────────────
        print("\n[Step 3] Computing ALEC similarity...")
        if n_alec > 0:
            alec_matches = compute_cross_similarity(
                ks_embeddings, ks_bills, alec_embeddings, alec_ids, threshold=threshold
            )
            print(f"  Found {len(alec_matches)} ALEC matches above {threshold}")
        else:
            alec_matches = pl.DataFrame(
                schema={
                    "ks_bill": pl.Utf8,
                    "other_id": pl.Utf8,
                    "similarity": pl.Float64,
                    "rank": pl.Int64,
                }
            )

        # ── Step 4: Cross-state comparison ───────────────────────────────
        cross_state_matches: dict[str, pl.DataFrame] = {}
        cross_state_embeddings: dict[str, np.ndarray] = {}
        cross_state_texts: dict[str, list[str]] = {}

        if not args.alec_only and states:
            print("\n[Step 4] Loading cross-state bill texts...")
            # Determine session string for other states
            # Use the start year as a simple session ID for OpenStates
            cross_session = str(session.start_year)

            state_data = load_cross_state_texts(states, cross_session)

            for state, (texts, bill_nums, meta_df) in state_data.items():
                processed = [btd.preprocess_for_embedding(t) for t in texts]
                print(f"  Computing {state.upper()} embeddings...")
                emb = btd.get_or_compute_embeddings(
                    processed,
                    bill_nums,
                    cache_dir=ctx.data_dir,
                    model_name=embedding_model,
                )
                cross_state_embeddings[state] = emb
                cross_state_texts[state] = texts

                matches = compute_cross_similarity(
                    ks_embeddings, ks_bills, emb, bill_nums, threshold=threshold
                )
                cross_state_matches[state] = matches
                print(f"  Found {len(matches)} {state.upper()} matches above {threshold}")
        else:
            print("\n[Step 4] Cross-state comparison skipped (--alec-only)")

        # ── Step 5: N-gram overlap for strong matches ────────────────────
        print("\n[Step 5] Computing n-gram overlap for strong matches...")
        n_ngram_computed = 0

        # Build text lookups
        ks_text_map = dict(zip(ks_bills, ks_raw_texts, strict=True))
        alec_text_map = dict(zip(alec_ids, alec_texts, strict=True)) if alec_texts else {}

        # Compute n-gram overlap for ALEC strong matches
        ngram_results: dict[tuple[str, str], float] = {}
        if len(alec_matches) > 0:
            strong = alec_matches.filter(pl.col("similarity") >= THRESHOLD_STRONG_MATCH)
            for row in strong.iter_rows(named=True):
                ks_text = ks_text_map.get(row["ks_bill"], "")
                other_text = alec_text_map.get(row["other_id"], "")
                if ks_text and other_text:
                    overlap = compute_ngram_overlap(ks_text, other_text, n=NGRAM_SIZE)
                    ngram_results[(row["ks_bill"], row["other_id"])] = overlap
                    n_ngram_computed += 1

        # Compute for cross-state strong matches
        state_text_maps: dict[str, dict[str, str]] = {}
        for state, (texts, bill_nums, _) in (
            load_cross_state_texts(states, str(session.start_year)).items()
            if not args.alec_only and states
            else {}
        ):
            state_text_maps[state] = dict(zip(bill_nums, texts, strict=True))

        for state, matches_df in cross_state_matches.items():
            if len(matches_df) == 0:
                continue
            strong = matches_df.filter(pl.col("similarity") >= THRESHOLD_STRONG_MATCH)
            stm = state_text_maps.get(state, {})
            for row in strong.iter_rows(named=True):
                ks_text = ks_text_map.get(row["ks_bill"], "")
                other_text = stm.get(row["other_id"], "")
                if ks_text and other_text:
                    overlap = compute_ngram_overlap(ks_text, other_text, n=NGRAM_SIZE)
                    ngram_results[(row["ks_bill"], row["other_id"])] = overlap
                    n_ngram_computed += 1

        print(f"  Computed n-gram overlap for {n_ngram_computed} strong matches")

        # ── Step 6: Build match summary ──────────────────────────────────
        print("\n[Step 6] Building match summary...")
        match_summary = build_match_summary(
            alec_matches=alec_matches,
            cross_state_matches=cross_state_matches,
            ks_metadata=ks_bill_texts,
            alec_metadata=alec_metadata if n_alec > 0 else None,
            topic_assignments=topic_assignments,
        )

        # Add n-gram overlap to summary
        if ngram_results:

            def _lookup_ngram(ks_bill: str, match_id: str) -> float | None:
                return ngram_results.get((ks_bill, match_id))

            overlap_values = [
                _lookup_ngram(row["ks_bill"], row["match_id"])
                for row in match_summary.iter_rows(named=True)
            ]
            match_summary = match_summary.with_columns(
                pl.Series("ngram_overlap", overlap_values, dtype=pl.Float64)
            )

        print(f"  Total matches: {len(match_summary)}")

        # Count by tier
        n_near_identical = len(match_summary.filter(pl.col("match_tier") == "near-identical"))
        n_strong = len(match_summary.filter(pl.col("match_tier") == "strong match"))
        n_related = len(match_summary.filter(pl.col("match_tier") == "related"))
        print(f"  Near-identical: {n_near_identical}, Strong: {n_strong}, Related: {n_related}")

        # Save match summary
        match_summary.write_parquet(ctx.data_dir / "match_summary.parquet")

        # ── Step 7: Compute max similarity per Kansas bill ───────────────
        print("\n[Step 7] Computing max similarity per Kansas bill...")
        max_sims = np.zeros(n_ks)

        if n_alec > 0:
            ks_norm = ks_embeddings / np.linalg.norm(ks_embeddings, axis=1, keepdims=True)
            alec_norm = alec_embeddings / np.linalg.norm(alec_embeddings, axis=1, keepdims=True)
            alec_sim = ks_norm @ alec_norm.T
            max_sims = np.maximum(max_sims, alec_sim.max(axis=1))

        for state, emb in cross_state_embeddings.items():
            if emb.shape[0] > 0:
                other_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                state_sim = ks_norm @ other_norm.T
                max_sims = np.maximum(max_sims, state_sim.max(axis=1))

        # ── Step 8: Build near-identical detail cards ────────────────────
        print("\n[Step 8] Preparing detail cards...")
        near_identical_details: list[dict] = []
        if n_near_identical > 0:
            near_rows = match_summary.filter(pl.col("match_tier") == "near-identical")
            for row in near_rows.iter_rows(named=True):
                ks_text = ks_text_map.get(row["ks_bill"], "")
                # Look up match text
                match_text = ""
                if row["source"] == "ALEC":
                    match_text = alec_text_map.get(row["match_id"], "")
                else:
                    stm = state_text_maps.get(row["source"].lower(), {})
                    match_text = stm.get(row["match_id"], "")

                near_identical_details.append(
                    {
                        "ks_bill": row["ks_bill"],
                        "source": row["source"],
                        "match_label": row["match_label"],
                        "similarity": row["similarity"],
                        "ngram_overlap": row.get("ngram_overlap"),
                        "ks_excerpt": ks_text[:EXCERPT_LENGTH] if ks_text else "",
                        "match_excerpt": match_text[:EXCERPT_LENGTH] if match_text else "",
                    }
                )

        # ── Step 9: Generate plots ───────────────────────────────────────
        print("\n[Step 9] Generating plots...")
        plot_similarity_distribution(max_sims, ks_bills, ctx.plots_dir)
        plot_topic_match_heatmap(match_summary, ctx.plots_dir)
        plot_match_network(match_summary, ctx.plots_dir)

        # ── Step 10: Build report ────────────────────────────────────────
        print("\n[Step 10] Building report...")

        # Separate ALEC and cross-state summaries for report
        alec_summary = match_summary.filter(pl.col("source") == "ALEC")
        cross_summary = match_summary.filter(pl.col("source") != "ALEC")

        # Top ALEC match for key findings
        top_alec_match = None
        if len(alec_summary) > 0:
            top_row = alec_summary.sort("similarity", descending=True).row(0, named=True)
            top_alec_match = {
                "ks_bill": top_row["ks_bill"],
                "match_label": top_row["match_label"],
                "similarity": top_row["similarity"],
            }

        # States with matches
        states_with_matches = []
        for state, matches_df in cross_state_matches.items():
            if len(matches_df) > 0:
                states_with_matches.append(state)

        results_dict = {
            "n_kansas_bills": n_ks,
            "n_alec_bills": n_alec,
            "n_alec_matches": len(alec_summary),
            "n_near_identical": n_near_identical,
            "n_strong_matches": n_strong,
            "n_cross_state_matches": len(cross_summary),
            "states_with_matches": states_with_matches,
            "top_alec_match": top_alec_match,
            "embedding_model": embedding_model,
            "embedding_dim": ks_embeddings.shape[1],
            "threshold": threshold,
            "ngram_size": NGRAM_SIZE,
            "cross_states": states if not args.alec_only else [],
            "alec_match_summary": alec_summary,
            "cross_state_match_summary": cross_summary,
            "near_identical_details": near_identical_details,
            "parameters": {
                "session": args.session,
                "threshold": threshold,
                "states": args.states if not args.alec_only else "none",
                "alec_only": args.alec_only,
                "embedding_model": embedding_model,
                "ngram_size": NGRAM_SIZE,
            },
        }

        # Add per-state bill counts
        for state in states:
            if not args.alec_only:
                n_state = cross_state_embeddings.get(state, np.empty((0,))).shape[0]
                results_dict[f"n_{state}_bills"] = n_state

        build_model_legislation_report(
            report=ctx.report,
            results=results_dict,
            plots_dir=ctx.plots_dir,
        )

        # Save filtering manifest
        manifest = {
            "n_kansas_bills": n_ks,
            "n_alec_bills": n_alec,
            "similarity_threshold": threshold,
            "n_alec_matches": len(alec_summary),
            "n_cross_state_matches": len(cross_summary),
            "n_near_identical": n_near_identical,
            "n_strong_matches": n_strong,
            "n_related": n_related,
            "ngram_size": NGRAM_SIZE,
            "states": states if not args.alec_only else [],
        }
        with open(ctx.run_dir / "filtering_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\nPhase 20 complete: {len(match_summary)} total matches")


if __name__ == "__main__":
    main()
