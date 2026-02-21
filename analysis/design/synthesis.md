# Synthesis Design Choices

## Assumptions

1. **All 7 upstream phases have been run.** Synthesis reads parquets from IRT, Indices, Network, Clustering, Prediction, PCA, and EDA. Missing phases produce warnings but the report still generates with available data.

2. **IRT ideal points are the base table.** The unified legislator DataFrame is built by LEFT JOINing all other phase outputs onto IRT ideal points. If IRT hasn't been run, synthesis fails.

3. **Notable legislators are session-independent concepts.** The *roles* (maverick, bridge-builder, metric paradox) recur across sessions. The specific legislators who fill those roles are detected from data.

## Parameters & Constants

| Parameter | Value | Justification | Location |
|-----------|-------|---------------|----------|
| Unity skip threshold | 0.95 | If all majority-party members have unity > 0.95, there is no meaningful maverick to profile. Avoids highlighting trivial differences. | `synthesis_detect.py:detect_chamber_maverick` |
| Paradox rank gap threshold | 0.50 | Requires the legislator to be in the top half on one metric and bottom half on the other. Prevents weak paradoxes from being reported. | `synthesis_detect.py:detect_metric_paradox` |
| Bridge midpoint tolerance | 1 SD | Bridge-builder must have IRT ideal point within 1 standard deviation of the cross-party midpoint. Ensures they are genuinely near the center. | `synthesis_detect.py:detect_bridge_builder` |
| Min party size for paradox | 5 | Need at least 5 members in the majority party to compute meaningful percentile ranks. | `synthesis_detect.py:detect_metric_paradox` |
| Max annotation slugs | 3 | Cap on annotated legislators per chamber in dashboard scatters. Prevents visual clutter. | `synthesis_detect.py:detect_annotation_slugs` |

## Methodological Choices

### 1. Detection over hardcoding

**Decision:** Replace hardcoded legislator slugs with algorithmic detection.

**Alternatives:** (a) Keep hardcoded names and update per session. (b) Use a configuration file listing slugs. (c) Detect from data.

**Impact:** Option (c) eliminates per-session maintenance and ensures the report works on any biennium. The detection thresholds may need occasional tuning but the concepts themselves are stable.

### 2. Maverick = lowest unity in majority party

**Decision:** Use `unity_score` (CQ-standard party unity) as the primary maverick indicator, with `weighted_maverick` as tiebreaker.

**Alternatives:** Could use `maverick_rate`, `loyalty_rate`, or a composite. Unity score was chosen because it is the standard political science metric and is most interpretable for nontechnical audiences.

### 3. Bridge-builder = betweenness near midpoint

**Decision:** Require bridge candidates to be near the cross-party midpoint, not just have high betweenness.

**Alternatives:** Highest betweenness regardless of position. This was rejected because at Kappa threshold 0.40, betweenness is computed within the party's disconnected component — a high-betweenness legislator at the extreme of their party is not a "bridge" in any meaningful sense.

### 4. Paradox = IRT rank vs loyalty rank divergence

**Decision:** Measure the gap between a legislator's IRT ideology percentile rank and clustering loyalty percentile rank within the majority party. The 0.5 threshold ensures the paradox is dramatic enough to be narratively compelling.

**Alternatives:** Could use raw score differences instead of percentile ranks. Percentile ranks were chosen because they normalize across sessions with different score scales.

### 5. Graceful degradation

**Decision:** Every detection function returns `None` when no suitable candidate is found. Report sections handle `None` by showing brief explanatory text ("no significant metric paradox detected") or by skipping the section entirely (profile cards). The report's section count varies from 27-30 depending on what is detected.

## Downstream Implications

- **Narrative quality:** Template-based narratives are adequate but less polished than hand-crafted prose. For publication-quality reports, the generated text should be reviewed and edited.
- **Plot filenames:** Profile cards use `profile_{lastname}.png` (extracted from slug). Paradox plots use `metric_paradox_{chamber}.png`. These names change across sessions.
- **Report section IDs:** The `metric-paradox` section ID replaces the old `tyson-paradox` ID. Any external links or bookmarks to the old ID will break.
