# Commit Workflow

## Version Tag Format
`[vYYYY.MM.DD.N]` where N is the sequential number for that day (starting at 1).

## Commit Message Format
```
type(scope): description [vYYYY.MM.DD.N]
```

**No Co-Authored-By lines.** Never append co-author trailers to commits.

### Types
- feat: New feature
- fix: Bug fix
- refactor: Code restructuring
- docs: Documentation only
- test: Adding/updating tests
- chore: Maintenance, deps, config
- style: Formatting, no logic change

### Scopes
- scraper: Core scraping pipeline
- models: Data models
- output: CSV export
- session: Session/biennium URL logic
- config: Settings/config changes
- cli: Command-line interface
- docs: Documentation files
- infra: CI, hooks, tooling

## Pre-Commit Checklist
1. Run `just lint` — must pass
2. Stage relevant files (NOT `data/`, `.env`, or secrets)
3. Commit with conventional message and version tag

## Pushing
**Never push without explicit permission.** Commit as often as useful; push only when asked.
