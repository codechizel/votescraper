# ADR-0114: Horseshoe-aware report system

**Date:** 2026-03-12
**Status:** Accepted

## Context

The 79th Legislature (2001-02, 30R/10D Senate) is the canonical horseshoe biennium — the standard 1D IRT model folds minority-party members back toward the majority due to extreme party-size imbalance. Despite having canonical ideal point routing (ADR-0110, ADR-0111, ADR-0112) that automatically detects and corrects for the horseshoe at the data level, no HTML report mentioned the horseshoe effect. Users reading reports for horseshoe-affected sessions had no warning that IRT-based interpretations might be distorted.

A full audit of all 20 reports for the 79th session uncovered 27 issues: 5 critical data bugs, 6 horseshoe awareness gaps, 4 template parameterization issues, 3 KanFocus graceful degradation issues, and 10 analytical improvements.

## Decision

1. **Horseshoe warning banners** — Add `load_horseshoe_status()` and `horseshoe_warning_html()` to `phase_utils.py`. These read the `routing_manifest.json` from the canonical ideal point routing phase and generate a styled HTML warning banner explaining the horseshoe effect and what the routing decided.

2. **Propagate warnings to 8 report builders** — IRT, 2D IRT, Hierarchical, Clustering, Network, TSA, Synthesis, and Profiles all accept an optional `horseshoe_status` parameter. When present and horseshoe is detected, a warning banner is inserted after key findings.

3. **Data-driven captions** — Replace hardcoded captions and interpretations throughout the report system with data-driven logic that adapts to the actual values (eigenvalue ratios, correlation coefficients, Rice indices, Yea base rates).

4. **Party mean inversion warning** — When 1D IRT Republican mean < Democrat mean (a signature of horseshoe distortion), add an explicit warning to key findings.

5. **Horseshoe-aware cluster labels** — When horseshoe is detected, use party-composition labels (`"R-dominated (14R/0D)"`) instead of xi_mean-based labels that are unreliable under distortion.

## Consequences

- Reports for horseshoe-affected sessions now prominently warn readers
- Reports for balanced sessions show no warnings (graceful no-op)
- Downstream consumers (journalists, policymakers) are alerted to interpretive caveats
- The `horseshoe_status` parameter is optional with `None` default — no pipeline changes required; callers pass it when the routing manifest is available

## Related

- ADR-0110: Tiered convergence quality gate
- ADR-0111: Canonical init strategy
- ADR-0112: 2D IRT supermajority tuning
- `docs/79th-report-audit.md`: Full audit record
- `docs/canonical-ideal-points.md`: Canonical routing documentation
