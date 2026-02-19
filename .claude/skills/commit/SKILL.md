# Commit Skill

Wrap-up macro: lint, version bump, and commit.

## Steps

1. Run `just lint` — fix any issues
2. Determine today's version tag:
   - Format: `[vYYYY.MM.DD.N]`
   - Check git log for existing tags today, increment N
3. Stage all relevant files (NOT `data/`, `.env`, or secrets)
4. Commit with message: `type(scope): description [vYYYY.MM.DD.N]`
   - **No Co-Authored-By lines**
5. Report the commit hash and version tag

## Usage
User says `/commit` → execute this skill.

## Prompt Template
```
Analyze staged and unstaged changes. Determine the commit type and scope.
Generate version tag for today. Create the commit following conventional
commit format. Never include Co-Authored-By trailers.
```
