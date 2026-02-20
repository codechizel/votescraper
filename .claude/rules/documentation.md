# Documentation Rules

## Format
- **All project documents must be Markdown** (`.md`). This includes ADRs, method docs, analysis primers, and any prose in `docs/`.
- Use ATX-style headings (`#`, `##`, `###`).
- Use fenced code blocks with language tags (` ```python `, ` ```bash `, etc.).
- Prefer tables over bullet lists when presenting structured data with multiple columns.

## Analysis Primers
- Every analysis phase (EDA, PCA, IRT, clustering, etc.) must include a `README.md` in its results directory (`results/<session>/<analysis>/README.md`).
- The primer explains **what the analysis does, why, and how to interpret the outputs** — written for a reader who hasn't seen the code.
- Structure: Purpose, Method, Inputs, Outputs (with file descriptions), Interpretation Guide, Caveats.
- The primer is written once and updated when the analysis changes, not on every run.
- `RunContext` writes the primer automatically from the `primer` parameter.

## ADRs
- One decision per file in `docs/adr/`.
- Follow the template in `docs/adr/README.md`.
- Update the index table when adding new ADRs.
