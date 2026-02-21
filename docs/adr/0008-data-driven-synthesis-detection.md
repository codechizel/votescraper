# ADR-0008: Data-Driven Synthesis Detection

**Date:** 2026-02-21
**Status:** Accepted

## Context

The synthesis report (`synthesis.py` + `synthesis_report.py`) originally hardcoded 5 legislator slugs and names — Schreiber, Dietrich, Tyson, Helgerson, Thompson — with hand-written narrative paragraphs about each. This worked for the 2025-2026 session but would break when applied to any other biennium: the slugs wouldn't exist, profiles would be empty, and narrative sections would silently fail.

The *concepts* these legislators illustrated (maverick, bridge-builder, metric paradox) are analytically valuable and session-independent. The specific legislators who embody those concepts should be detected from data, not hardcoded.

## Decision

Extract all legislator detection logic into a new module `analysis/synthesis_detect.py` with three detectors:

1. **Maverick detector** (`detect_chamber_maverick`): Lowest `unity_score` in the majority party, ties broken by highest `weighted_maverick`. Returns `None` if all unity > 0.95 (no maverick to report).

2. **Bridge-builder detector** (`detect_bridge_builder`): Highest `betweenness` centrality among legislators whose IRT ideal point is within 1 SD of the cross-party midpoint. Falls back to highest betweenness regardless if no one is near the midpoint. Returns `None` if betweenness column is missing.

3. **Metric paradox detector** (`detect_metric_paradox`): Within the majority party, finds the legislator with the largest gap between IRT percentile rank and clustering loyalty percentile rank. Must exceed a 0.5 gap (top half on one metric, bottom half on the other). Returns `None` if no paradox is significant enough.

All detectors return frozen dataclasses (`NotableLegislator`, `ParadoxCase`) with pre-formatted titles, subtitles, and machine-readable reason codes. The synthesis report templates these into narrative sections.

Annotation slugs for dashboard scatter plots are also computed dynamically via `detect_annotation_slugs`, which collects detected notable slugs plus the most extreme legislator per party.

## Consequences

**Positive:**
- Synthesis report works on any biennium without code changes.
- Detection logic is testable in isolation (pure data functions, no plotting or I/O).
- Report sections gracefully degrade: if no maverick is found (all unity > 0.95), the mavericks section says "party discipline is exceptionally uniform" instead of crashing.
- New notable legislators are discovered automatically (e.g., the 2025-2026 run now also profiles Borjon as house bridge-builder, Hill as senate bridge-builder, and Anderson as house paradox case — none of which were in the hardcoded list).

**Negative:**
- Detection thresholds (unity > 0.95 skip, rank gap > 0.5 for paradox) are somewhat arbitrary. They produce reasonable results on the current data but may need tuning for sessions with very different voting patterns.
- The bridge-builder detector requires betweenness centrality from the network phase; if network analysis hasn't been run, it returns `None`.
- Narrative quality depends on the template strings being sufficiently generic. Hand-crafted prose about a specific legislator will always read better than a template — but the templates are a reasonable tradeoff for session-independence.

**Files changed:**
- `analysis/synthesis_detect.py` — New, ~270 lines. Pure detection logic.
- `analysis/synthesis.py` — Removed `PROFILE_LEGISLATORS` and `ANNOTATE_SLUGS` constants. Renamed `plot_tyson_paradox` to `plot_metric_paradox`. Updated `main()` to use `detect_all()`.
- `analysis/synthesis_report.py` — All narrative sections now take a `notables` dict and template from detected legislators. Renamed `_add_tyson_*` functions to `_add_paradox_*`. Added `session` parameter to remove hardcoded session strings.
