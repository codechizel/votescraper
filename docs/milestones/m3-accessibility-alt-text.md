# M3: Accessibility — Descriptive Alt-Text

**Status:** Complete (2026-03-02, ADR-0079)

Add descriptive alt-text to all figure and interactive sections across the report system, targeting WCAG 2.1 AA compliance.

**Roadmap item:** R25 (descriptive alt-text for all figures)
**Actual effort:** 1 session (Phase A + Phase B combined)

---

## Current State

`FigureSection` in `analysis/report.py:58-102` already uses `title` as HTML `alt` attribute (line 98):

```python
parts.append(f'<img src="data:image/png;base64,{self.image_data}" alt="{self.title}" />')
```

**Problem:** Titles like "Party Ideal Points" or "ROC Curve" are labels, not descriptions. WCAG 2.1 AA requires alt-text that conveys the **information or function** of the image — not just its label. A screen reader user should understand what the chart shows without seeing it.

`InteractiveSection` in `analysis/report.py:141-156` has no `aria-label` at all — the wrapping `<div>` is unlabeled.

---

## Phase A: Infrastructure (Non-Breaking)

### Changes to `analysis/report.py`

#### `FigureSection` (line 58)

Add an `alt_text` field:

```python
@dataclass(frozen=True)
class FigureSection:
    id: str
    title: str
    image_data: str
    caption: str | None = None
    alt_text: str | None = None  # NEW: descriptive alt-text for screen readers
```

Update `render()` (line 98) to prefer `alt_text` over `title`:

```python
alt = self.alt_text or self.title
parts.append(f'<img src="data:image/png;base64,{self.image_data}" alt="{alt}" />')
```

This is **non-breaking** — all existing usages continue to use `title` as fallback.

#### `FigureSection.from_file()` class method

Also pass through `alt_text`:

```python
@classmethod
def from_file(cls, id: str, title: str, path: Path, caption: str | None = None,
              alt_text: str | None = None) -> "FigureSection":
    ...
    return cls(id=id, title=title, image_data=encoded, caption=caption, alt_text=alt_text)
```

#### `InteractiveSection` (line 141)

Add an `aria_label` field:

```python
@dataclass(frozen=True)
class InteractiveSection:
    id: str
    title: str
    html: str
    caption: str | None = None
    aria_label: str | None = None  # NEW: accessible label for interactive content
```

Update `render()` to include the label on the container div:

```python
aria = f' aria-label="{self.aria_label}"' if self.aria_label else ""
parts.append(f'<div class="interactive-container" id="{self.id}"{aria}>')
```

### Test Updates

Add tests in `tests/test_report.py`:

```python
class TestAccessibility:
    def test_figure_alt_text_override(self):
        section = FigureSection(id="test", title="My Chart",
                                image_data="abc", alt_text="Bar chart showing...")
        html = section.render()
        assert 'alt="Bar chart showing..."' in html

    def test_figure_alt_text_falls_back_to_title(self):
        section = FigureSection(id="test", title="My Chart", image_data="abc")
        html = section.render()
        assert 'alt="My Chart"' in html

    def test_interactive_aria_label(self):
        section = InteractiveSection(id="test", title="Network",
                                     html="<div></div>",
                                     aria_label="Interactive network graph...")
        html = section.render()
        assert 'aria-label="Interactive network graph..."' in html
```

---

## Phase B: Rollout (Incremental)

Update all report builders to provide descriptive `alt_text` strings. Prioritize by report usage frequency and analytical importance.

### Total Scope

- **162 `FigureSection` usages** across 24 report builder files
- **~20 `InteractiveSection` usages** (Plotly, PyVis, Folium)
- **~15 `InteractiveTableSection` usages** (ITables) — these are already accessible via HTML table semantics

### Alt-Text Writing Guidelines

Good alt-text for data visualizations follows this pattern:

```
[Chart type] showing [what data] for [context]. [Key finding or pattern].
```

Examples:

| Current Title | Descriptive Alt-Text |
|---------------|---------------------|
| "Party Ideal Points" | "Density plot of IRT ideal points by party. Republicans cluster around +1.0, Democrats around -1.5, with no overlap." |
| "ROC Curve — House" | "ROC curve for House vote prediction model. AUC = 0.98, well above the diagonal baseline." |
| "Network Graph" | "Co-voting network graph with 125 legislators as nodes. Two disconnected clusters correspond to party affiliation." |
| "Shrinkage Scatter" | "Scatter plot comparing flat IRT vs hierarchical IRT ideal points. Points near diagonal show minimal shrinkage; outliers are pulled toward party means." |

### Rollout Batches

**Batch 1 (highest priority):** EDA + Synthesis + Profiles — most-read reports

| File | `FigureSection` Count | Priority |
|------|----------------------|----------|
| `analysis/01_eda/eda_report.py` | 8 | First impressions for every user |
| `analysis/11_synthesis/synthesis_report.py` | 15 | Narrative summary — most shared |
| `analysis/12_profiles/profiles_report.py` | 6 | Per-legislator deep dives |

**Batch 2:** IRT + Clustering + Network — core analytical visualizations

| File | `FigureSection` Count |
|------|----------------------|
| `analysis/04_irt/irt_report.py` | 12 |
| `analysis/05_clustering/clustering_report.py` | 12 |
| `analysis/06_network/network_report.py` | 13 |

**Batch 3:** Prediction + Indices + Cross-Session

| File | `FigureSection` Count |
|------|----------------------|
| `analysis/08_prediction/prediction_report.py` | 11 |
| `analysis/07_indices/indices_report.py` | 12 |
| `analysis/13_cross_session/cross_session_report.py` | 8 |

**Batch 4:** Remaining phases

| File | `FigureSection` Count |
|------|----------------------|
| `analysis/15_tsa/tsa_report.py` | 10 |
| `analysis/16_dynamic_irt/dynamic_irt_report.py` | 8 |
| `analysis/06b_network_bipartite/bipartite_report.py` | 8 |
| `analysis/10_hierarchical/hierarchical_report.py` | 7 |
| `analysis/02_pca/pca_report.py` | 6 |
| `analysis/02b_umap/umap_report.py` | 5 |
| `analysis/02c_mca/mca_report.py` | 5 |
| `analysis/04b_irt_2d/irt_2d_report.py` | 4 |
| `analysis/04c_ppc/ppc_report.py` | 6 |
| `analysis/05b_lca/lca_report.py` | 5 |
| `analysis/07b_beta_binomial/beta_binomial_report.py` | 4 |
| `analysis/14_external_validation/external_validation_report.py` | 4 |
| `analysis/14b_external_validation_dime/external_validation_dime_report.py` | 4 |
| `analysis/17_wnominate/wnominate_report.py` | 4 |

### InteractiveSection Updates

Add `aria_label` to all Plotly/PyVis/Folium interactive sections:

| Phase | Type | Current |
|-------|------|---------|
| Phase 04 (IRT) | Plotly scatter | No label |
| Phase 06 (Network) | PyVis graph | No label |
| Phase 07 (Indices) | Plotly scatter | No label |
| Phase 13 (Cross-Session) | Plotly Sankey | No label |
| Phase 01 (EDA) | Folium choropleth | No label |

---

## Verification

```bash
# Phase A:
just test -k "test_report" -v        # report tests pass
just lint-check                       # formatting

# Phase B (per batch):
just pipeline 2025-26                 # regenerate reports
# Manually inspect HTML output for alt attributes
grep -r 'alt="' results/kansas/91st_2025-2026/latest/*/report.html | head -20
```

## Documentation

- Update `docs/roadmap.md` item R25 to "Done" (after all batches complete)
- Consider an ADR if the alt-text pattern becomes a convention other projects might follow

## Commits

```
# Phase A:
feat(infra): add alt_text/aria_label to report section types [vYYYY.MM.DD.N]

# Phase B (one per batch):
docs(infra): add descriptive alt-text to EDA, Synthesis, Profiles reports [vYYYY.MM.DD.N]
docs(infra): add descriptive alt-text to IRT, Clustering, Network reports [vYYYY.MM.DD.N]
docs(infra): add descriptive alt-text to Prediction, Indices, Cross-Session reports [vYYYY.MM.DD.N]
docs(infra): add descriptive alt-text to remaining phase reports [vYYYY.MM.DD.N]
```
