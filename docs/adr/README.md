# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the KS Vote Scraper project.

ADRs document significant technical decisions so future contributors understand *why* things are the way they are, not just *what* they are.

## Format

Each ADR follows a lightweight MADR-style template:

```markdown
# ADR-NNNN: Title

**Date:** YYYY-MM-DD
**Status:** Accepted | Superseded | Deprecated

## Context
What prompted this decision?

## Decision
What did we decide?

## Consequences
What are the trade-offs?
```

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-results-directory-structure.md) | Results directory structure | Accepted | 2026-02-19 |
| [0002](0002-polars-over-pandas.md) | Polars over pandas | Accepted | 2026-02-19 |
| [0003](0003-two-phase-fetch-parse.md) | Two-phase fetch/parse pattern | Accepted | 2026-02-19 |
| [0004](0004-html-report-system.md) | HTML report system | Accepted | 2026-02-19 |
| [0005](0005-pca-implementation-choices.md) | PCA implementation choices | Accepted | 2026-02-19 |

## Creating a New ADR

1. Copy the template above
2. Use the next sequential number (`NNNN`)
3. Add an entry to the index table
