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
- web: Django project, database models, admin

## Post-Feature Workflow

After completing a feature or fix, **always update documentation before committing:**

1. **Update existing docs** — check CLAUDE.md, relevant ADRs, roadmap, and design docs for sections that reference the changed code.
2. **Create a new ADR if warranted** — non-obvious architectural decisions or trade-offs. Not every feature needs one.
3. **Include doc changes in the same commit** — code and its documentation ship together.

## Documentation Standards

- **All project documents must be Markdown** (`.md`). ADRs, method docs, analysis primers, prose in `docs/`.
- ATX-style headings (`#`, `##`, `###`). Fenced code blocks with language tags.
- Prefer tables over bullet lists for structured multi-column data.
- Analysis primers: auto-written by `RunContext` from the `primer` parameter. Structure: Purpose, Method, Inputs, Outputs, Interpretation Guide, Caveats.
- ADRs: one decision per file in `docs/adr/`. Follow template in `docs/adr/README.md`. Update both the topic section and chronological index.

## Pre-Commit Checklist
1. Run `just lint` — must pass
2. Stage relevant files (NOT `data/`, `.env`, or secrets)
3. Commit with conventional message and version tag

## Pushing
**Never push without explicit permission.** Commit as often as useful; push only when asked.
