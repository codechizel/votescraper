"""Shared visualization helpers for legislative analysis.

Provides hemicycle (semicircle parliament) charts for visualizing chamber
composition by party or by vote category on specific roll calls.
"""

import math

import plotly.graph_objects as go

try:
    from analysis.tuning import PARTY_COLORS  # noqa: F401 (re-exported to eda_report.py)
except ModuleNotFoundError:
    from tuning import PARTY_COLORS  # type: ignore[no-redef]  # noqa: F401

# ── Color constants ──────────────────────────────────────────────────────────

VOTE_COLORS: dict[str, str] = {
    "Yea": "#2ecc71",
    "Nay": "#e74c3c",
    "Absent and Not Voting": "#95a5a6",
    "Present and Passing": "#f39c12",
    "Not Voting": "#bdc3c7",
}


# ── Hemicycle layout ─────────────────────────────────────────────────────────


def _compute_seat_positions(
    total_seats: int, n_rows: int | None = None
) -> list[tuple[float, float]]:
    """Compute (x, y) positions for seats in a hemicycle layout.

    Distributes *total_seats* across concentric semicircular arcs. Outer rows
    hold proportionally more seats (proportional to arc circumference).

    Returns positions ordered left-to-right, inner-to-outer.
    """
    if total_seats <= 0:
        return []

    if n_rows is None:
        n_rows = max(2, round(math.sqrt(total_seats / math.pi)))

    # Row radii and proportional seat distribution
    row_radii = [1.0 + i * 0.4 for i in range(n_rows)]
    row_weights = row_radii  # proportional to circumference
    total_weight = sum(row_weights)
    row_counts = [max(1, round(total_seats * w / total_weight)) for w in row_weights]

    # Adjust to match exact total (add/remove from outermost rows first)
    diff = total_seats - sum(row_counts)
    for i in range(abs(diff)):
        idx = -(i + 1) % n_rows
        row_counts[idx] += 1 if diff > 0 else -1

    positions: list[tuple[float, float]] = []
    for radius, count in zip(row_radii, row_counts):
        for j in range(count):
            theta = math.pi * (j + 0.5) / count
            x = -radius * math.cos(theta)  # left-to-right
            y = radius * math.sin(theta)
            positions.append((x, y))

    return positions


# ── Chart builders ───────────────────────────────────────────────────────────


def make_hemicycle_chart(
    seats: list[dict[str, str | int]],
    title: str,
    *,
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """Create a hemicycle (semicircle parliament) chart.

    Args:
        seats: List of dicts with keys ``"label"`` (str), ``"color"`` (str),
            ``"count"`` (int). Each entry represents a group (e.g. party) with
            its seat count and color.
        title: Chart title.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with hemicycle layout.
    """
    total_seats = sum(int(g["count"]) for g in seats)
    if total_seats == 0:
        fig = go.Figure()
        fig.update_layout(title=title, width=width, height=height)
        return fig

    positions = _compute_seat_positions(total_seats)

    # Build per-seat color and hover label lists
    colors: list[str] = []
    labels: list[str] = []
    for group in seats:
        count = int(group["count"])
        colors.extend([str(group["color"])] * count)
        labels.extend([str(group["label"])] * count)

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    # Marker size: scale inversely with seat count for readability
    marker_size = max(6, min(14, 600 // total_seats))

    fig = go.Figure(
        data=go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker={"size": marker_size, "color": colors, "line": {"width": 0}},
            text=labels,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Legend: one invisible trace per group for a clean legend
    for group in seats:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": str(group["color"])},
                name=f"{group['label']} ({group['count']})",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5},
        xaxis={"visible": False, "scaleanchor": "y"},
        yaxis={"visible": False},
        width=width,
        height=height,
        margin={"l": 20, "r": 20, "t": 50, "b": 60},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def make_vote_hemicycle(
    vote_counts: list[dict[str, str | int]],
    title: str,
    *,
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """Create a hemicycle colored by vote category for a specific roll call.

    Args:
        vote_counts: List of dicts with keys ``"label"`` (vote category),
            ``"color"`` (str), ``"count"`` (int).
        title: Chart title.

    Returns:
        Plotly Figure with hemicycle layout colored by vote.
    """
    return make_hemicycle_chart(vote_counts, title, width=width, height=height)
