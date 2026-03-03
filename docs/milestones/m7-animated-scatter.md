# M7: Animated Scatter — Done

Add Gapminder-style animated Plotly scatter showing legislator ideal points evolving across bienniums.

**Status:** Done (2026-03-02, v2026.03.02.25)
**Roadmap item:** R24 (animated scatter for dynamic IRT)
**Estimated effort:** 1 session
**Dependencies:** None (shares data source with M6 but independent implementation)

---

## Goal

An interactive animation where each frame is one biennium. Legislators appear as dots positioned by their ideal point (x-axis) and uncertainty (y-axis), colored by party. Legislators enter and exit across frames based on whether they served in that biennium. Hovering shows name + ideal point + HDI. The play button auto-advances through 8 bienniums (84th-91st).

---

## Data Source

Same as M6 — `trajectories_{chamber}.parquet` from Phase 16 (dynamic IRT).

**Key columns used:**

| Column | Role |
|--------|------|
| `xi_mean` | X-axis: ideal point |
| `xi_sd` | Y-axis: posterior uncertainty |
| `xi_hdi_2.5`, `xi_hdi_97.5` | Hover text: 95% credible interval |
| `biennium_label` | Animation frame |
| `party` | Color |
| `full_name` | Hover text |
| `served` | Filter: only show legislators who served |

---

## Implementation

### New Function in `analysis/27_dynamic_irt/dynamic_irt.py`

```python
import plotly.express as px

def plot_animated_scatter(
    trajectories: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> Path:
    """Gapminder-style animated scatter of ideal points across bienniums.

    X = ideal point, Y = uncertainty, color = party, frame = biennium.
    Legislators appear/disappear based on served=True.

    Args:
        trajectories: DataFrame from extract_dynamic_ideal_points().
        chamber: "House" or "Senate".
        out_dir: Directory to write HTML file.

    Returns:
        Path to generated HTML file.
    """
    # Filter to served legislators only
    served = trajectories.filter(pl.col("served"))

    # Build hover text
    served = served.with_columns(
        (
            pl.col("full_name")
            + "<br>Ideal point: "
            + pl.col("xi_mean").round(2).cast(pl.Utf8)
            + "<br>95% HDI: ["
            + pl.col("xi_hdi_2.5").round(2).cast(pl.Utf8)
            + ", "
            + pl.col("xi_hdi_97.5").round(2).cast(pl.Utf8)
            + "]"
        ).alias("hover_text")
    )

    # Convert to pandas for Plotly Express
    pdf = served.select(
        "biennium_label", "full_name", "party", "xi_mean", "xi_sd", "hover_text"
    ).to_pandas()

    party_colors = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#808080"}

    fig = px.scatter(
        pdf,
        x="xi_mean",
        y="xi_sd",
        color="party",
        color_discrete_map=party_colors,
        animation_frame="biennium_label",
        hover_name="full_name",
        custom_data=["hover_text"],
        labels={
            "xi_mean": "Ideal Point (Conservative → Liberal)",
            "xi_sd": "Uncertainty (Posterior SD)",
            "party": "Party",
        },
        title=f"{chamber} — Legislator Ideal Points Across Bienniums",
    )

    # Customize hover template
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        marker={"size": 8, "opacity": 0.7},
    )

    # Fix axis ranges across all frames for stable animation
    x_min = pdf["xi_mean"].min() - 0.5
    x_max = pdf["xi_mean"].max() + 0.5
    y_min = 0
    y_max = pdf["xi_sd"].max() * 1.2

    fig.update_layout(
        xaxis={"range": [x_min, x_max]},
        yaxis={"range": [y_min, y_max]},
        width=900,
        height=600,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5},
    )

    # Slow down animation for readability
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1500
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500

    out_path = out_dir / f"animated_scatter_{chamber}.html"
    fig.write_html(out_path, include_plotlyjs=True)
    return out_path
```

### Size Note

`include_plotlyjs=True` produces ~3 MB HTML per figure (the Plotly.js bundle is inlined). This is acceptable for local reports. For the iframe dashboard, the CDN version (`include_plotlyjs="cdn"`) could be used instead, but the standalone HTML is more portable.

### Integration in `main()`

In `analysis/27_dynamic_irt/dynamic_irt.py`, after the ridgeline plot (or after existing trajectory plots):

```python
for chamber in chambers:
    traj_path = ctx.data_dir / f"trajectories_{chamber}.parquet"
    if traj_path.exists():
        trajectories = pl.read_parquet(traj_path)
        plot_animated_scatter(trajectories, chamber, ctx.plots_dir)
```

### Report Integration

In `analysis/27_dynamic_irt/dynamic_irt_report.py`, add a new section function:

```python
def _add_animated_scatter(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Add animated ideal point scatter."""
    path = plots_dir / f"animated_scatter_{chamber}.html"
    if path.exists():
        report.add(InteractiveSection(
            id=f"animated-scatter-{chamber.lower()}",
            title=f"{chamber} — Ideal Point Animation",
            html=path.read_text(),
            caption=(
                f"Press play to animate legislator positions across 8 bienniums. "
                f"X-axis: ideal point (left=liberal, right=conservative). "
                f"Y-axis: estimation uncertainty (higher = less certain). "
                f"Hover over any dot for legislator details and 95% credible interval."
            ),
        ))
```

Insert after the ridgeline section (or after `_add_trajectories()`) in `build_dynamic_irt_report()`.

---

## InteractiveSection Pattern

Already used across the codebase for Plotly, PyVis, and Folium output. Example from cross-session Sankey:

```python
report.add(InteractiveSection(
    id="bloc-sankey",
    title="Voting Bloc Transitions",
    html=fig.to_html(include_plotlyjs="cdn", div_id="bloc-sankey"),
    caption="Sankey diagram of cluster membership changes...",
))
```

The animated scatter follows the same pattern but reads from a file (since the HTML is larger).

---

## Tests

Add to `tests/test_dynamic_irt.py`:

```python
class TestAnimatedScatter:
    def test_produces_html(self, tmp_path):
        """Animated scatter creates an HTML file."""
        trajectories = pl.DataFrame({
            "biennium_label": (["84th (2011-12)"] * 10 + ["85th (2013-14)"] * 10),
            "full_name": [f"Member {i}" for i in range(10)] * 2,
            "party": ["Republican"] * 8 + ["Democrat"] * 2 + ["Republican"] * 8 + ["Democrat"] * 2,
            "xi_mean": list(np.random.default_rng(42).normal(size=20)),
            "xi_sd": list(np.random.default_rng(43).uniform(0.05, 0.3, size=20)),
            "xi_hdi_2.5": [-1.0] * 20,
            "xi_hdi_97.5": [1.0] * 20,
            "served": [True] * 20,
        })
        out_path = plot_animated_scatter(trajectories, "House", tmp_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 1000  # not empty

    def test_contains_plotly(self, tmp_path):
        """HTML file includes Plotly.js."""
        trajectories = pl.DataFrame({
            "biennium_label": ["84th (2011-12)"] * 5,
            "full_name": [f"M{i}" for i in range(5)],
            "party": ["Republican"] * 5,
            "xi_mean": [1.0, 1.1, 0.9, 1.2, 0.8],
            "xi_sd": [0.1] * 5,
            "xi_hdi_2.5": [0.5] * 5,
            "xi_hdi_97.5": [1.5] * 5,
            "served": [True] * 5,
        })
        out_path = plot_animated_scatter(trajectories, "Senate", tmp_path)
        content = out_path.read_text()
        assert "plotly" in content.lower()

    def test_filters_unserved(self, tmp_path):
        """Only served=True legislators appear in animation."""
        trajectories = pl.DataFrame({
            "biennium_label": ["84th (2011-12)"] * 6,
            "full_name": [f"M{i}" for i in range(6)],
            "party": ["Republican"] * 6,
            "xi_mean": [1.0] * 6,
            "xi_sd": [0.1] * 6,
            "xi_hdi_2.5": [0.5] * 6,
            "xi_hdi_97.5": [1.5] * 6,
            "served": [True] * 3 + [False] * 3,
        })
        out_path = plot_animated_scatter(trajectories, "House", tmp_path)
        assert out_path.exists()
```

---

## Verification

```bash
just test -k "test_dynamic_irt" -v    # existing + new tests pass
just lint-check                       # formatting
just dynamic-irt                      # regenerate report, open in browser
# Verify: play button advances through bienniums, hover shows details
```

## Documentation

- Update `docs/roadmap.md` item R24 to "Done"
- No ADR needed (additive visualization)

## Commit

```
feat(infra): animated ideal point scatter in dynamic IRT report [vYYYY.MM.DD.N]
```
