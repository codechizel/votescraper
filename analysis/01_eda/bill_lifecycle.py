"""Bill lifecycle classification and Sankey visualization.

Maps KLISS API HISTORY status strings to canonical legislative stages and
builds a Plotly Sankey diagram showing bill flow through the legislature.

Usage (called from eda.py):
    from analysis.bill_lifecycle import classify_action, plot_bill_lifecycle_sankey
"""

import polars as pl

# ── Lifecycle Stage Classification ──────────────────────────────────────────

# Map canonical stage names to keyword lists for matching against status text.
# Order matters: first match wins. More specific patterns come first.
LIFECYCLE_STAGES: dict[str, list[str]] = {
    "Introduced": ["introduced"],
    "Committee Referral": ["referred to committee"],
    "Hearing": ["hearing", "scheduled for hearing"],
    "Committee Report": ["committee report recommending"],
    "Committee of the Whole": ["committee of the whole"],
    "Floor Vote": ["final action", "emergency final action", "roll call"],
    "Cross-Chamber": ["received by", "transmitted to"],
    "Signed into Law": ["approved by governor"],
    "Vetoed": ["vetoed by governor", "line item veto"],
    "Governor": ["enrolled and presented"],
}

# Ordered stage list for Sankey layout (left to right)
STAGE_ORDER = [
    "Introduced",
    "Committee Referral",
    "Hearing",
    "Committee Report",
    "Committee of the Whole",
    "Floor Vote",
    "Cross-Chamber",
    "Governor",
    "Signed into Law",
    "Vetoed",
    "Died",
    "Other",
]

# Colors per stage for the Sankey nodes
STAGE_COLORS = {
    "Introduced": "#3498db",
    "Committee Referral": "#2980b9",
    "Hearing": "#1abc9c",
    "Committee Report": "#27ae60",
    "Committee of the Whole": "#2ecc71",
    "Floor Vote": "#f39c12",
    "Cross-Chamber": "#e67e22",
    "Governor": "#9b59b6",
    "Signed into Law": "#27ae60",
    "Vetoed": "#c0392b",
    "Died": "#95a5a6",
    "Other": "#bdc3c7",
}


def classify_action(status_text: str) -> str:
    """Map a KLISS status string to a canonical lifecycle stage.

    Scans the status text (case-insensitive) for keywords matching each stage.
    Returns the first matching stage, or "Other" if no match found.
    """
    status_lower = status_text.lower()
    for stage, keywords in LIFECYCLE_STAGES.items():
        if any(kw in status_lower for kw in keywords):
            return stage
    return "Other"


# ── Transition Computation ──────────────────────────────────────────────────


def compute_bill_transitions(actions_df: pl.DataFrame) -> pl.DataFrame:
    """Compute bill stage transitions from classified actions.

    For each bill, sorts actions by timestamp, classifies each action into a
    canonical stage, deduplicates consecutive same-stage entries, then counts
    transitions between stage pairs.

    Parameters
    ----------
    actions_df : pl.DataFrame
        Bill actions with at least ``bill_number``, ``occurred_datetime``,
        and ``status`` columns.

    Returns
    -------
    pl.DataFrame
        Columns: ``source``, ``target``, ``value`` (count of bills making
        that transition).
    """
    if actions_df.is_empty():
        return pl.DataFrame({"source": [], "target": [], "value": []}).cast(
            {"source": pl.Utf8, "target": pl.Utf8, "value": pl.Int64}
        )

    # Classify each action
    classified = actions_df.with_columns(
        pl.col("status").map_elements(classify_action, return_dtype=pl.Utf8).alias("stage")
    )

    # Sort by bill + timestamp, get ordered stage list per bill
    sorted_df = classified.sort("bill_number", "occurred_datetime")

    # Group by bill and collect stage sequences
    bill_stages = sorted_df.group_by("bill_number").agg(pl.col("stage"))

    # Count transitions: iterate stage lists, deduplicate consecutive stages
    transition_counts: dict[tuple[str, str], int] = {}
    terminal_stages = {"Floor Vote", "Cross-Chamber", "Governor", "Signed into Law", "Vetoed"}

    for row in bill_stages.iter_rows(named=True):
        stages = row["stage"]
        if not stages:
            continue

        # Deduplicate consecutive same-stage entries
        deduped: list[str] = [stages[0]]
        for s in stages[1:]:
            if s != deduped[-1]:
                deduped.append(s)

        # Count transitions between consecutive stages
        for i in range(len(deduped) - 1):
            pair = (deduped[i], deduped[i + 1])
            transition_counts[pair] = transition_counts.get(pair, 0) + 1

        # Infer "Died" for bills that stall before reaching a terminal stage
        last_stage = deduped[-1]
        if last_stage not in terminal_stages:
            pair = (last_stage, "Died")
            transition_counts[pair] = transition_counts.get(pair, 0) + 1

    if not transition_counts:
        return pl.DataFrame({"source": [], "target": [], "value": []}).cast(
            {"source": pl.Utf8, "target": pl.Utf8, "value": pl.Int64}
        )

    sources, targets, values = [], [], []
    for (src, tgt), count in sorted(transition_counts.items()):
        sources.append(src)
        targets.append(tgt)
        values.append(count)

    return pl.DataFrame({"source": sources, "target": targets, "value": values})


# ── Sankey Visualization ────────────────────────────────────────────────────


def plot_bill_lifecycle_sankey(
    actions_df: pl.DataFrame,
    title: str = "Bill Lifecycle Flow",
) -> "object | None":
    """Sankey diagram showing bill flow through legislative stages.

    Each bill traces a path from introduction through committee, floor vote,
    and potentially to the governor. Link thickness = number of bills flowing
    between stages.

    Parameters
    ----------
    actions_df : pl.DataFrame
        Bill actions with ``bill_number``, ``occurred_datetime``, ``status``.
    title : str
        Chart title.

    Returns
    -------
    go.Figure or None
        Plotly figure, or None if no data or plotly not available.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not installed — skipping Sankey diagram")
        return None

    if actions_df.is_empty():
        return None

    transitions = compute_bill_transitions(actions_df)
    if transitions.is_empty():
        return None

    # Build label list from stages actually present in transitions
    stages_present = set(transitions["source"].to_list() + transitions["target"].to_list())
    labels = [s for s in STAGE_ORDER if s in stages_present]
    label_idx = {label: i for i, label in enumerate(labels)}

    sources, targets, values = [], [], []
    for row in transitions.iter_rows(named=True):
        src, tgt, val = row["source"], row["target"], row["value"]
        if src in label_idx and tgt in label_idx:
            sources.append(label_idx[src])
            targets.append(label_idx[tgt])
            values.append(val)

    if not values:
        return None

    node_colors = [STAGE_COLORS.get(label, "#bdc3c7") for label in labels]

    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 15,
                    "thickness": 20,
                    "label": labels,
                    "color": node_colors,
                },
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "color": "rgba(200,200,200,0.4)",
                },
            )
        ]
    )
    fig.update_layout(
        title=title,
        font={"size": 12},
        width=900,
        height=600,
    )
    return fig
