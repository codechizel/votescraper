# ADR-0002: Polars over pandas

**Date:** 2026-02-19
**Status:** Accepted

## Context

The analysis pipeline reads CSVs (~68K vote rows, ~500 rollcalls, ~172 legislators), builds vote matrices, computes agreement scores, and produces filtered outputs. We needed to choose a DataFrame library for all data manipulation.

## Decision

Use **polars** as the primary DataFrame library. Pandas is acceptable only when a downstream library strictly requires a pandas DataFrame (e.g., seaborn's `clustermap`).

Rationale:

- **Null handling**: Polars distinguishes between `null` (missing) and `0` natively. Our vote matrix uses `null` for "did not vote" and `0` for "Nay" — pandas conflates these without careful `dtype` management.
- **Type safety**: Polars enforces column types at creation, catching data bugs early. Pandas silently coerces types.
- **Expression API**: Polars' lazy expressions (`pl.col(...).filter(...)`) compose cleanly without the indexing footguns of pandas (`df[df["col"] > x]` vs `df.loc[...]` vs `df.iloc[...]`).
- **Performance**: Polars is faster for our workloads (CSV reads, pivots, group-bys), though at our data size (~68K rows) the difference is seconds, not minutes.

## Consequences

- **Good**: Cleaner null semantics throughout the analysis pipeline
- **Good**: More readable code via the expression API
- **Trade-off**: Seaborn and some scipy functions require pandas input — we convert at the boundary (`df.to_pandas()`) rather than using pandas throughout
- **Trade-off**: Polars is less widely known than pandas — contributors may need to learn the API
