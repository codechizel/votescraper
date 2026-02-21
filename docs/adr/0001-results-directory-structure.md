# ADR-0001: Results directory structure

**Date:** 2026-02-19
**Status:** Accepted

## Context

EDA Phase 1 outputs 8 PNGs, 7 parquets, and 1 JSON manifest, all dumped flat into `data/91st_2025-2026/analysis/`. As we add more analysis phases (PCA, IRT, clustering), this flat structure becomes unmanageable:

- No separation between analysis types (EDA vs PCA vs IRT outputs mixed together)
- No run history (each run overwrites the previous)
- No metadata tracking (which git commit produced these results?)
- No way for downstream scripts to reliably find the latest output

Additionally, `data/` is semantically wrong for analysis outputs — it's the scraper's domain.

## Decision

Analysis outputs go in a structured `results/` directory:

```
results/
  91st_2025-2026/
    eda/
      2026-02-19/
        plots/                  <- PNGs
        data/                   <- Parquet intermediates
        filtering_manifest.json <- What was filtered and why
        run_info.json           <- Git hash, timestamp, parameters
        run_log.txt             <- Captured console output
      latest -> 2026-02-19/    <- Symlink to most recent run
    pca/
      ...
```

Key design choices:

- **Biennium naming** (`91st_2025-2026` not `2025-26`): uses the Kansas Legislature's numbered biennium system for clarity
- **Date-based run directories**: each run is immutable; re-running creates a new date directory (or overwrites same-day)
- **`latest` symlink**: relative path (`latest -> 2026-02-19`) so the tree is portable across machines
- **`run_info.json`**: captures git hash, timestamp, Python version, and script parameters for full reproducibility
- **`run_log.txt`**: captures all console output (print statements) via a TeeStream wrapper
- **Reusable `RunContext` class**: any analysis script uses the same context manager to get structured output

The `results/` directory is gitignored (like `data/`).

## Consequences

- **Good**: Each run is reproducible and traceable to a git commit
- **Good**: Downstream scripts can always read from `results/<session>/<analysis>/latest/`
- **Good**: Old results are preserved for comparison
- **Good**: Single `RunContext` class means all future analysis scripts get this for free
- **Trade-off**: More disk usage (old runs accumulate) — acceptable given parquet compression and small dataset size
- **Trade-off**: Analysis scripts must import `RunContext` instead of choosing their own output path — minimal overhead
