# Tallgrass — Command Runner (Just: https://github.com/casey/just)
#
# Thin aliases over `uv run` commands. The main value-adds:
#   1. `just check` sequences lint + typecheck + tests as a single quality gate
#   2. OMP/OPENBLAS thread caps below prevent E-core spillover on Apple Silicon
#   3. `just --list` documents every runnable command in the project
#
# All recipes pass *args through, so `just profiles --names "Masterson"`
# is equivalent to `uv run python analysis/12_profiles/profiles.py --names "Masterson"`.
#
# Cap thread pools to P-core count (6) to prevent E-core spillover on Apple Silicon.
# See ADR-0022 and results/experimental_lab/2026-02-23_parallel-chains-performance/.
export OMP_NUM_THREADS := "6"
export OPENBLAS_NUM_THREADS := "6"

# Ensure /usr/bin is on PATH so PyTensor can find clang++/g++ for C compilation.
# Without this, background processes or stripped shells fall back to pure Python (~18x slower).
export PATH := "/usr/bin:/bin:" + env("PATH")

# Default: show available commands
default:
    @just --list

# Scrape current session (cached)
scrape *args:
    uv run tallgrass {{args}}

# Merge special session data into parent biennium
merge-special *args:
    uv run tallgrass --merge-special {{args}}

# Scrape with fresh cache
scrape-fresh *args:
    uv run tallgrass --clear-cache {{args}}

# Lint and format
lint:
    uv run ruff check --fix src/
    uv run ruff format src/

# Lint check only (no fix)
lint-check:
    uv run ruff check src/
    uv run ruff format --check src/

# Install dependencies
install:
    uv sync

# List available sessions
sessions:
    uv run tallgrass --list-sessions

# Run EDA analysis
eda *args:
    uv run python analysis/01_eda/eda.py {{args}}

# Run PCA analysis
pca *args:
    uv run python analysis/02_pca/pca.py {{args}}

# Run MCA analysis
mca *args:
    uv run python analysis/02c_mca/mca.py {{args}}

# Run Bayesian IRT analysis
irt *args:
    uv run python analysis/04_irt/irt.py {{args}}

# Run 2D Bayesian IRT analysis (experimental)
irt-2d *args:
    uv run python analysis/04b_irt_2d/irt_2d.py {{args}}

# Run posterior predictive checks + LOO-CV model comparison
ppc *args:
    uv run python analysis/04c_ppc/ppc.py {{args}}

# Run clustering analysis
clustering *args:
    uv run python analysis/05_clustering/clustering.py {{args}}

# Run latent class analysis
lca *args:
    uv run python analysis/05b_lca/lca.py {{args}}

# Run network analysis
network *args:
    uv run python analysis/06_network/network.py {{args}}

# Run bipartite bill-legislator network analysis
bipartite *args:
    uv run python analysis/06b_network_bipartite/bipartite.py {{args}}

# Run prediction analysis
prediction *args:
    uv run python analysis/08_prediction/prediction.py {{args}}

# Run UMAP analysis
umap *args:
    uv run python analysis/03_umap/umap_viz.py {{args}}

# Run classical indices analysis
indices *args:
    uv run python analysis/07_indices/indices.py {{args}}

# Run Beta-Binomial Bayesian party loyalty
betabinom *args:
    uv run python analysis/09_beta_binomial/beta_binomial.py {{args}}

# Run hierarchical Bayesian IRT
hierarchical *args:
    uv run python analysis/10_hierarchical/hierarchical.py {{args}}

# Run synthesis report
synthesis *args:
    uv run python analysis/11_synthesis/synthesis.py {{args}}

# Run cross-session validation
cross-session *args:
    uv run python analysis/13_cross_session/cross_session.py {{args}}

# Run legislator profiles
profiles *args:
    uv run python analysis/12_profiles/profiles.py {{args}}

# Run external validation against Shor-McCarty scores
external-validation *args:
    uv run python analysis/14_external_validation/external_validation.py {{args}}

# Run external validation against DIME/CFscores
dime *args:
    uv run python analysis/14b_external_validation_dime/external_validation_dime.py {{args}}

# Run time series analysis (drift + changepoints)
tsa *args:
    uv run python analysis/15_tsa/tsa.py {{args}}

# Run dynamic ideal point estimation (state-space IRT)
dynamic-irt *args:
    uv run python analysis/16_dynamic_irt/dynamic_irt.py {{args}}

# Run W-NOMINATE + Optimal Classification validation
wnominate *args:
    uv run python analysis/17_wnominate/wnominate.py {{args}}

# Run full analysis pipeline for a session (all phases grouped under one run ID)
pipeline session="2025-26" *args:
    #!/usr/bin/env bash
    set -euo pipefail
    RUN_ID=$(uv run python -c "from tallgrass.session import KSSession; from analysis.run_context import generate_run_id; ks = KSSession.from_session_string('{{session}}'); print(generate_run_id('{{session}}', results_root=ks.results_dir))")
    echo "Pipeline run: $RUN_ID"
    echo "Session:      {{session}}"
    echo ""
    just eda      --session {{session}} --run-id "$RUN_ID" {{args}}
    just pca      --session {{session}} --run-id "$RUN_ID" {{args}}
    just mca      --session {{session}} --run-id "$RUN_ID" {{args}}
    just irt      --session {{session}} --run-id "$RUN_ID" {{args}}
    just umap     --session {{session}} --run-id "$RUN_ID" {{args}}
    just clustering --session {{session}} --run-id "$RUN_ID" {{args}}
    just lca      --session {{session}} --run-id "$RUN_ID" {{args}}
    just network  --session {{session}} --run-id "$RUN_ID" {{args}}
    just bipartite --session {{session}} --run-id "$RUN_ID" {{args}}
    just indices  --session {{session}} --run-id "$RUN_ID" {{args}}
    just prediction --session {{session}} --run-id "$RUN_ID" {{args}}
    just betabinom --session {{session}} --run-id "$RUN_ID" {{args}}
    just hierarchical --session {{session}} --run-id "$RUN_ID" {{args}}
    just synthesis --session {{session}} --run-id "$RUN_ID" {{args}}
    just profiles --session {{session}} --run-id "$RUN_ID" {{args}}
    just tsa       --session {{session}} --run-id "$RUN_ID" {{args}}
    just dashboard {{session}} --run-id "$RUN_ID"
    echo ""
    echo "Pipeline complete: $RUN_ID"

# Generate dashboard index for a session
dashboard session="2025-26" *args:
    uv run python analysis/dashboard.py {{session}} {{args}}

# --- Worktree lifecycle ---

# Path to the primary (non-worktree) repo root
_main_root := `git worktree list --porcelain | head -1 | sed 's/^worktree //'`

# Create a new worktree with branch worktree-<name>
wt-new name:
    #!/usr/bin/env bash
    set -euo pipefail
    MAIN_ROOT="{{_main_root}}"
    WT_PATH="$MAIN_ROOT/.claude/worktrees/{{name}}"
    if [ -d "$WT_PATH" ]; then
        echo "Error: worktree '{{name}}' already exists at $WT_PATH"
        exit 1
    fi
    git worktree add -b "worktree-{{name}}" "$WT_PATH" main
    echo ""
    echo "Worktree ready:"
    echo "  Path:   $WT_PATH"
    echo "  Branch: worktree-{{name}}"
    echo ""
    echo "To enter: cd $WT_PATH"

# Merge worktree branch to main, remove worktree, delete branch.
# Usage: just wt-done              (from inside worktree — auto-detects path/branch)
#        just wt-done feature-name  (from main repo — prevents CWD death in Claude Code)
# Delegates to _wt-done-impl in the MAIN repo's Justfile so that worktrees
# created before a Justfile fix still pick up the latest logic.
wt-done *name:
    #!/usr/bin/env bash
    set -euo pipefail
    MAIN_ROOT="{{_main_root}}"
    NAME="{{name}}"
    if [ -n "$NAME" ]; then
        # Called with worktree name — resolve path and branch from name
        WT_PATH="$MAIN_ROOT/.claude/worktrees/$NAME"
        BRANCH="worktree-$NAME"
        if [ ! -d "$WT_PATH" ]; then
            echo "Error: worktree not found at $WT_PATH"
            exit 1
        fi
    else
        # Called from inside worktree — auto-detect
        WT_PATH="$(pwd)"
        BRANCH="$(git branch --show-current)"
    fi
    just --justfile "$MAIN_ROOT/Justfile" _wt-done-impl "$WT_PATH" "$BRANCH"

# Implementation: called by wt-done with worktree path and branch name.
# Lives in the main repo's Justfile so fixes are always picked up.
[private]
_wt-done-impl wt_path branch:
    #!/usr/bin/env bash
    set -euo pipefail
    BRANCH="{{branch}}"
    WT_PATH="{{wt_path}}"
    MAIN_ROOT="{{_main_root}}"
    if [ -z "$BRANCH" ] || [ "$BRANCH" = "main" ]; then
        echo "Error: not on a worktree branch (current: ${BRANCH:-detached})"
        exit 1
    fi
    # Verify clean working tree (check from the worktree)
    if ! git -C "$WT_PATH" diff --quiet || ! git -C "$WT_PATH" diff --cached --quiet; then
        echo "Error: uncommitted changes. Commit or stash first."
        exit 1
    fi
    # Fast-forward main to the worktree branch
    # (git fetch . branch:main is blocked by Git 2.35+ when main is checked out
    # in the primary repo, so we use merge-base check + update-ref instead)
    if ! git -C "$WT_PATH" merge-base --is-ancestor main HEAD; then
        echo ""
        echo "Error: cannot fast-forward main to $BRANCH."
        echo "Main has diverged. Rebase first: git rebase main"
        exit 1
    fi
    git -C "$WT_PATH" update-ref refs/heads/main HEAD
    # Remove worktree + branch (already running from main, no CWD death risk)
    git worktree remove "$WT_PATH" 2>/dev/null || git worktree remove --force "$WT_PATH"
    git branch -d "$BRANCH" 2>/dev/null || git branch -D "$BRANCH" 2>/dev/null
    git worktree prune
    echo ""
    echo "Done: merged $BRANCH -> main, removed worktree."
    echo "Run 'git push origin main' when ready."

# Run all tests
test *args:
    uv run pytest tests/ {{args}} -v

# Run scraper tests only
test-scraper *args:
    uv run pytest tests/ -m scraper {{args}} -v

# Run fast tests (skip slow/integration)
test-fast *args:
    uv run pytest tests/ -m "not slow" {{args}} -v

# Type check with ty (scraper must pass clean, analysis warnings-only)
typecheck:
    uvx ty check src/
    uvx ty check analysis/

# Check if an experiment is running (nutpie shows its own progress bar in the terminal)
monitor:
    @if [ -f /tmp/tallgrass/experiment.pid ] && kill -0 "$(cat /tmp/tallgrass/experiment.pid)" 2>/dev/null; then echo "Experiment running (PID $(cat /tmp/tallgrass/experiment.pid))"; ps -p "$(cat /tmp/tallgrass/experiment.pid)" -o pid,etime,command | tail -1; else echo "No experiment running"; fi

# Full check (lint + typecheck + tests)
check:
    just lint-check
    just typecheck
    just test
