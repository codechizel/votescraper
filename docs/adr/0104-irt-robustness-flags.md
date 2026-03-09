# ADR-0104: IRT Robustness Flags

**Date:** 2026-03-08
**Status:** Accepted

## Context

The 1D IRT model (Phase 05) produces misleading results in supermajority chambers due to the horseshoe effect — ultra-conservative rebels who vote Nay (opposing the establishment) are placed near Democrats who also vote Nay (opposing conservative policy), because 1D cannot distinguish the two motivations. ADR-0103 addressed identification strategy selection but did not provide runtime diagnostics or alternative analyses for quantifying and mitigating the horseshoe distortion.

Previously, investigating horseshoe effects required modifying code directly — toggling contested-vote refitting, enabling horseshoe metrics, or cross-referencing 2D results. Each investigation was ad hoc and unreproducible because the exact configuration wasn't recorded in the output.

Users requested CLI flags that enable/disable robustness analyses without code changes, with all flag states always visible in the HTML report regardless of whether they are active.

## Decision

### Extensible flag registry

A `RobustnessFlag` frozen dataclass holds `(name, enabled, label, description)`. The `RobustnessFlags` registry class provides:

- `ALL_FLAGS` — canonical list of flag names
- `LABELS` / `DESCRIPTIONS` — human-readable metadata
- `build_flags(args)` — classmethod that reads an argparse `Namespace` and returns `list[RobustnessFlag]`

Adding a new flag requires: (1) add name to `ALL_FLAGS`, (2) add label/description, (3) add CLI arg, (4) add logic block in `main()`.

### Four initial flags

| Flag | CLI arg | Default | What it does |
|------|---------|---------|--------------|
| `contested_only` | `--contested-only` | off | Re-fits IRT using only cross-party contested votes (both parties must split ≥10% per side). Strips intra-party rebel dynamics. Requires ≥ `MIN_CONTESTED_FOR_REFIT` (50) contested votes. |
| `horseshoe_diagnostic` | `--horseshoe-diagnostic` | off | Computes 6 quantitative horseshoe metrics: Democrat wrong-side fraction, party overlap, eigenvalue ratio, R/D ideal point means, and whether any R is more liberal than the D mean. |
| `promote_2d` | `--promote-2d` | off | Cross-references 1D rankings with 2D IRT Dimension 1 rankings (from Phase 04b). Flags legislators whose rank shifts by ≥ `PROMOTE_2D_RANK_SHIFT` (10) positions. |
| `irt_2d_dir` | `--irt-2d-dir` | auto | Directory containing 2D IRT results. Auto-resolved from `results_dir/../04b_irt_2d` when `--promote-2d` is active. |

### Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `MIN_CONTESTED_FOR_REFIT` | 50 | Minimum contested votes to attempt contested-only refit |
| `HORSESHOE_DEM_WRONG_SIDE_FRAC` | 0.20 | Threshold: if >20% of Democrats are on the wrong side, flag horseshoe |
| `PROMOTE_2D_RANK_SHIFT` | 10 | Minimum rank-position shift between 1D and 2D to flag a legislator |

### Always-visible report sections

1. **Identification Summary** — always shown. Reports: strategy name, reference (literature citation), anchor method, conservative/liberal anchor slugs, sign flip status, auto-detection rationale. This was an oversight in ADR-0103 — anchor identity and method were not previously reported.

2. **Robustness Flags Table** — always shown. Lists all flags with ON/OFF status and description, regardless of which are active. Users see at a glance what diagnostic options exist and which were used for this run.

3. **Conditional sections** — shown only when the corresponding flag is ON:
   - Horseshoe Diagnostic: 6-metric table with pass/fail verdict
   - Contested-Only Refit: summary (vote counts, correlation with full model) + top movers table
   - 2D Cross-Reference: summary (match count, mean shift) + flagged legislators table

### CLI

```bash
just irt --horseshoe-diagnostic                    # enable horseshoe metrics
just irt --contested-only --horseshoe-diagnostic   # both flags
just irt --promote-2d                              # cross-reference 2D results
just irt --promote-2d --irt-2d-dir /path/to/2d     # explicit 2D results path
```

## Consequences

**Benefits:**
- Robustness analyses are reproducible — exact flag state is recorded in both the manifest and HTML report
- No code changes needed to toggle diagnostics — CLI flags only
- Extensible — new flags follow the same pattern (add to registry, add CLI arg, add logic block)
- Identification summary closes the reporting oversight from ADR-0103

**Costs:**
- Contested-only refit runs a full second MCMC (~5-10 min per chamber) when enabled
- 2D cross-reference requires Phase 04b to have been run previously

**Not addressed:**
- Automatic horseshoe correction (e.g., auto-switching to 2D when horseshoe is detected) — intentionally left manual. The user should review diagnostics and decide whether to re-run with different settings. A supermajority audit (78th–91st) found only 3/28 chambers trigger horseshoe detection, but 5 sessions show problematic 1D-2D disagreement.
- Multi-dimensional IRT as a production replacement for 1D — remains experimental (Phase 04b). Interactive Plotly plots in Phase 04b now serve as visual horseshoe diagnostics (hover over Dim 1 vs PC1 to identify misplaced legislators).

**Related documentation:**
- `docs/horseshoe-effect-and-solutions.md` — General-audience explanation of the horseshoe effect and six approaches to addressing it
- `docs/79th-horseshoe-robustness-analysis.md` — Empirical validation of all three robustness flags on the 79th biennium
- `docs/irt-identification-strategies.md` — Identification strategy system documentation
- `results/experimental_lab/2026-03-08_supermajority-auto-promote/` — Experiment testing auto-promotion of 2D results when horseshoe is detected
