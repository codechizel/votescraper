# ADR-0079: WCAG 2.1 AA Accessibility — Alt-Text and ARIA Labels

**Date:** 2026-03-02
**Status:** Accepted

## Context

The HTML report system produces ~162 `FigureSection` images and ~20 `InteractiveSection` embeds across 23 report builder files. Before this change:

- `FigureSection.render()` used the section `title` as the HTML `alt` attribute. Titles like "Party Ideal Points" or "ROC Curve" are labels, not descriptions — WCAG 2.1 AA requires alt-text that conveys the **information or function** of the image.
- `InteractiveSection.render()` had no `aria-label` at all — the container `<div>` was unlabeled for assistive technology.

This was identified as roadmap item R25 and scoped in milestone M3.

## Decision

### Infrastructure (Phase A)

Added accessibility fields to two frozen dataclasses in `analysis/report.py`:

1. **`FigureSection`** gains `alt_text: str | None = None`. The `render()` method uses `self.alt_text or self.title` for the `<img alt="...">` attribute — backward-compatible fallback to title when alt_text is not set. Both `from_file()` and `from_figure()` classmethods pass through the new parameter.

2. **`InteractiveSection`** gains `aria_label: str | None = None`. The `render()` method includes `aria-label="..."` on the container `<div>` when set, omits it when `None`.

Seven tests added in `tests/test_report.py::TestAccessibility` covering both override and fallback behavior.

### Rollout (Phase B)

All 23 report builder files updated with descriptive alt-text following the WCAG pattern:

```
[Chart type] showing [what data] for [context]. [Key finding or pattern].
```

Coverage:
- **132 `FigureSection` calls** received `alt_text=` parameter
- **8 `InteractiveSection` calls** received `aria_label=` parameter
- **~15 `InteractiveTableSection` calls** left unchanged (HTML table semantics already accessible)

Rollout was done in 4 batches by priority:
1. EDA + Synthesis + Profiles (highest-traffic reports)
2. IRT + Clustering + Network (core analytical)
3. Prediction + Indices + Cross-Session
4. All remaining phases (TSA, Dynamic IRT, Bipartite, Hierarchical, PCA, UMAP, MCA, 2D IRT, PPC, LCA, Beta-Binomial, External Validation, DIME, W-NOMINATE)

## Consequences

**Positive:**
- Screen reader users can understand data visualizations without seeing them
- Alt-text describes chart type, data content, and key findings — not just labels
- ARIA labels on interactive sections (Plotly, PyVis, Folium) make them discoverable
- All existing report builder calls continue to work (optional parameters with fallback)

**Negative:**
- Alt-text strings are static — they describe the chart's purpose, not dynamic data values. For per-legislator profiles, alt-text includes the legislator name via f-strings.
- Maintenance burden: new `FigureSection` calls should include `alt_text=` (though the fallback to `title` means forgetting is non-breaking)

**Trade-offs:**
- Chose static descriptive alt-text over dynamically computed descriptions. Dynamic alt-text (e.g., "Republicans cluster at +1.2") would require each report builder to compute summary statistics and pass them into the alt string — high effort for marginal accessibility gain. Static descriptions that explain *what the chart shows* satisfy WCAG 2.1 AA Level A (Success Criterion 1.1.1).
