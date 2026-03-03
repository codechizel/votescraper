# M6: Ridgeline Plots — COMPLETED

Add ridgeline (joy) plots showing temporal ideology distributions across 8 bienniums in the dynamic IRT report.

**Roadmap item:** R23 (ridgeline plots for temporal ideology distributions)
**Status:** Completed 2026-03-02
**Dependencies:** None (uses existing `trajectories_{chamber}.parquet` data)

---

## Goal

Ridgeline plots stack density curves vertically by time period, providing a compact view of how the ideological distribution evolves. Each biennium gets one row with overlapping Republican (red) and Democrat (blue) KDE curves. More space-efficient than 8 separate density plots and immediately shows polarization trends.

---

## Data Source

Dynamic IRT Phase 16 already produces the trajectories data:

**File:** `results/kansas/{session}/{run_id}/16_dynamic_irt/trajectories_{chamber}.parquet`

**Columns from `extract_dynamic_ideal_points()` (dynamic_irt.py:503-551):**

| Column | Type | Description |
|--------|------|-------------|
| `global_idx` | int | Legislator global index |
| `name_norm` | str | Normalized name |
| `full_name` | str | From roster |
| `party` | str | Republican/Democrat/Independent |
| `time_period` | int | Biennium index (0-7) |
| `biennium_label` | str | e.g., "84th (2011-12)" |
| `xi_mean` | float | Posterior mean ideal point |
| `xi_sd` | float | Posterior SD |
| `xi_hdi_2.5` | float | 95% HDI lower bound |
| `xi_hdi_97.5` | float | 95% HDI upper bound |
| `served` | bool | Legislator active in this period |

**Key filter:** Only use rows where `served == True` — non-served rows have interpolated posteriors that should not be included in density estimation.

---

## Implementation

### New Function in `analysis/27_dynamic_irt/dynamic_irt.py`

```python
from scipy.stats import gaussian_kde

def plot_ridgeline_ideology(
    trajectories: pl.DataFrame,
    chamber: str,
    out_path: Path,
    *,
    x_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 200,
) -> None:
    """Ridgeline plot of ideological distributions across bienniums.

    One row per biennium (84th-91st), stacked vertically with slight overlap.
    Republicans in red, Democrats in blue, with filled KDE curves.

    Args:
        trajectories: DataFrame with columns biennium_label, party, xi_mean, served.
        chamber: "House" or "Senate" (for title).
        out_path: Path to save the PNG figure.
        x_range: Ideal point range for x-axis.
        n_points: Number of points for KDE evaluation.
    """
    import matplotlib.pyplot as plt

    served = trajectories.filter(pl.col("served"))
    biennium_labels = served["biennium_label"].unique().sort().to_list()
    n_bienniums = len(biennium_labels)

    fig, ax = plt.subplots(figsize=(10, 1.5 * n_bienniums))
    x_grid = np.linspace(x_range[0], x_range[1], n_points)

    party_colors = {"Republican": "#E81B23", "Democrat": "#0015BC"}
    vertical_spacing = 1.0
    overlap_factor = 0.7  # how much curves can overlap

    for i, label in enumerate(reversed(biennium_labels)):
        y_offset = i * vertical_spacing
        biennium_data = served.filter(pl.col("biennium_label") == label)

        for party, color in party_colors.items():
            party_data = biennium_data.filter(pl.col("party") == party)
            values = party_data["xi_mean"].to_numpy()

            if len(values) < 3:
                continue

            kde = gaussian_kde(values, bw_method="silverman")
            density = kde(x_grid)
            # Scale density to fit within vertical spacing
            density = density / density.max() * overlap_factor

            ax.fill_between(x_grid, y_offset, y_offset + density,
                            alpha=0.5, color=color, linewidth=0)
            ax.plot(x_grid, y_offset + density, color=color, linewidth=1.0)

        # Label on left
        ax.text(x_range[0] - 0.3, y_offset + 0.15, label,
                ha="right", va="center", fontsize=9)

    ax.set_xlim(x_range[0] - 2.5, x_range[1] + 0.5)
    ax.set_ylim(-0.3, n_bienniums * vertical_spacing + 0.5)
    ax.set_xlabel("Ideal Point (Conservative → Liberal)", fontsize=11)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{chamber} — Ideological Distribution Over Time", fontsize=13, pad=15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#E81B23", alpha=0.5, label="Republican"),
                       Patch(facecolor="#0015BC", alpha=0.5, label="Democrat")]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

### Integration in `main()`

In `analysis/27_dynamic_irt/dynamic_irt.py`, after the existing trajectory plots (around the call to `plot_trajectories()`):

```python
# After existing trajectory plotting:
for chamber in chambers:
    traj_path = ctx.data_dir / f"trajectories_{chamber}.parquet"
    if traj_path.exists():
        trajectories = pl.read_parquet(traj_path)
        plot_ridgeline_ideology(
            trajectories, chamber,
            ctx.plots_dir / f"ridgeline_{chamber}.png",
        )
```

### Report Integration

In `analysis/27_dynamic_irt/dynamic_irt_report.py`, add after `_add_polarization_trend()` (line 292) and before `_add_tau_posterior()` (line 336):

```python
def _add_ridgeline(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Add ridgeline ideology distribution plot."""
    path = plots_dir / f"ridgeline_{chamber}.png"
    if path.exists():
        report.add(FigureSection.from_file(
            id=f"ridgeline-{chamber.lower()}",
            title=f"{chamber} — Ideological Ridgeline",
            path=path,
            caption=(
                f"Kernel density estimates of ideal point distributions for each biennium. "
                f"Republicans (red) and Democrats (blue) shown separately. "
                f"Wider peaks indicate more within-party ideological variation. "
                f"Increasing separation between party peaks indicates growing polarization."
            ),
        ))
```

---

## Tests

Add to `tests/test_dynamic_irt.py`:

```python
class TestRidgelinePlot:
    def test_produces_png(self, tmp_path):
        """Ridgeline plot creates a PNG file."""
        names = [f"Member {i}" for i in range(30)]
        trajectories = pl.DataFrame({
            "biennium_label": ["84th (2011-12)"] * 20 + ["85th (2013-14)"] * 20 + ...,
            "party": ["Republican"] * 15 + ["Democrat"] * 5 + ...,
            "xi_mean": np.random.default_rng(42).normal(size=...),
            "served": [True] * ...,
        })
        out_path = tmp_path / "ridgeline_House.png"
        plot_ridgeline_ideology(trajectories, "House", out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_skips_small_parties(self, tmp_path):
        """Parties with < 3 members are skipped (KDE needs 3+ points)."""
        trajectories = pl.DataFrame({
            "biennium_label": ["84th (2011-12)"] * 5,
            "party": ["Republican"] * 4 + ["Independent"],
            "xi_mean": [1.0, 1.1, 0.9, 1.2, 0.0],
            "served": [True] * 5,
        })
        out_path = tmp_path / "ridgeline_House.png"
        plot_ridgeline_ideology(trajectories, "House", out_path)
        assert out_path.exists()

    def test_filters_unserved(self, tmp_path):
        """Only served=True legislators included in density."""
        trajectories = pl.DataFrame({
            "biennium_label": ["84th (2011-12)"] * 10,
            "party": ["Republican"] * 10,
            "xi_mean": list(range(10)),
            "served": [True] * 5 + [False] * 5,
        })
        # Would need to inspect KDE to verify, but at minimum no crash
        out_path = tmp_path / "ridgeline_House.png"
        plot_ridgeline_ideology(trajectories, "House", out_path)
        assert out_path.exists()
```

---

## Verification

```bash
just test -k "test_dynamic_irt" -v    # existing + new tests pass
just lint-check                       # formatting
just dynamic-irt                      # regenerate report, inspect ridgeline
```

## Documentation

- Update `docs/roadmap.md` item R23 to "Done"
- No ADR needed (additive visualization)

## Commit

```
feat(infra): ridgeline ideology plots in dynamic IRT report [vYYYY.MM.DD.N]
```
