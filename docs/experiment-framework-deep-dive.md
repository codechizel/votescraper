# Experiment Framework: Design Patterns, Ecosystem Survey, and Implementation Plan

*February 2026* — Updated 2026-02-28: All production MCMC now uses nutpie (ADR-0051, ADR-0053). PyMC `callback` and `cores` references in the monitoring sections are historical.

This document analyzes how the Tallgrass project currently manages experiments, surveys the Python ecosystem for experiment management tools and design patterns, and proposes a lightweight framework that eliminates code duplication, enforces platform constraints, and keeps experiments cleanly separated from production code.

## The Problem

Tallgrass has a growing experimental practice. Six experiments have been run since 2026-02-23, testing PCA-informed initialization, 4-chain MCMC, 2D IRT, beta prior variants, and positive-beta constraints. The experiments have produced valuable results — PCA init alone dropped House R-hat from 1.0102 to 1.0026 — but the process has three structural problems.

### 1. Code Duplication Creates Drift Risk

Each experiment reimplements the production model-building logic because the production functions don't accept the experimental parameter. The positive-beta experiment (`run_experiment.py`, 789 lines) contains two functions — `build_model_with_variant()` and `build_joint_model_with_variant()` — that are near-verbatim copies of production's `build_per_chamber_model()` and `build_joint_model()` in `hierarchical.py`. The only difference: an `if variant == ...` block that swaps the beta prior.

This means:
- If a bug is fixed in production's `build_per_chamber_model()`, the experiment's copy silently drifts
- If the experiment discovers a better prior and gets promoted to production, the integration is a manual diff-and-merge
- Re-running an old experiment after production changes may produce different results without anyone noticing

### 2. Platform Constraints Are Implicit

The Apple Silicon M3 Pro has hard rules for MCMC work (ADR-0022, `docs/apple-silicon-mcmc-tuning.md`): cap thread pools at 6, run bienniums sequentially, never schedule MCMC on efficiency cores. These rules live in documentation and the Justfile's `OMP_NUM_THREADS=6` export — but experiment scripts bypass the Justfile (they're run with `uv run python results/experimental_lab/.../run_experiment.py`). An experimenter who forgets to set `OMP_NUM_THREADS` or launches two experiments in parallel will get skewed timing data from E-core scheduling, and may not realize it.

The PCA-init experiment discovered this empirically: jitter + 4 chains causes mode-splitting (ADR-0045). The 4-chain experiment documented that first-run PyTensor compilation adds 10-20 minutes. These are platform-specific gotchas that each experiment rediscovers independently.

### 3. No Standard Experiment Lifecycle

The `TEMPLATE.md` provides excellent documentation structure, but there's no code-level standard for:
- How experiment parameters relate to production constants
- How to load data and run diagnostics identically to production
- How to compare results across variants in a standardized way
- How to promote a successful experiment to production

Each experiment reinvents this scaffolding. The beta-prior experiment (`irt_beta_experiment.py`) uses relaxed convergence thresholds (`RHAT_THRESHOLD = 1.05` vs production's `1.01`) — a reasonable choice for exploration, but one that's invisible unless you read the source.

## Ecosystem Survey

We evaluated nine experiment management frameworks, the Bayesian community's established practices, and four categories of design patterns. The full landscape:

### Frameworks Evaluated

| Framework | Approach | Dependencies | Verdict |
|-----------|----------|-------------|---------|
| **Sacred** (IDSIA) | Decorator-based config + MongoDB logging | Heavy (MongoDB for full features) | Elegant but opinionated; changes how you write functions |
| **Hydra** (Meta) | YAML config composition + CLI overrides | Heavy (antlr4, omegaconf) | Powerful but designed for ML hyperparameter sweeps |
| **MLflow** | Full ML lifecycle platform | Very heavy (Flask, SQLAlchemy, protobuf) | Team-oriented; overkill for single-developer |
| **Weights & Biases** | Cloud-hosted tracking + visualization | Cloud dependency | Designed for deep learning; cloud coupling disqualifies |
| **DVC** | Git-like data versioning + pipeline DAGs | Moderate | Oriented toward changing data; our CSVs are stable inputs |
| **Optuna** | Bayesian hyperparameter optimization | Light (SQLite) | Hyperparameter search, not model comparison |
| **MLXP** (INRIA, 2024) | Lightweight Hydra wrapper + job versioning | Heavy (inherits Hydra) | Most thoughtful for research; still too heavy |
| **epyc** (St Andrews) | Parameter-sweep simulation management | Minimal (pure Python) | Oriented toward grid sweeps, not structural model variants |
| **Pyrallis/Draccus** | Frozen dataclass ↔ YAML ↔ CLI mapping | Tiny | Lightest option with YAML support; no tracking |

### What the Bayesian Community Actually Does

The Bayesian statistics community has converged on a **no-framework approach**. The canonical reference — Gelman et al.'s "Bayesian Workflow" (2020) — describes an iterative process: build a simple model, check diagnostics, expand, compare. The "experiment record" is the ArviZ `InferenceData` object (serialized to NetCDF), and model comparison uses `az.compare()` with LOO-CV or WAIC.

Key references:
- **PyMC examples gallery**: Each model variant is a `with pm.Model():` block. Comparison is inline via `az.compare()`. No framework.
- **Bambi**: Model variants expressed as formula strings. Comparison via ArviZ.
- **CmdStanPy**: Models are separate `.stan` files. No built-in experiment management.
- **Bayesian Computation book, Ch. 9**: Multiple models fit and compared in a single notebook. The notebook IS the experiment record.
- **DrWatson.jl** (Julia): The most aligned with scientific single-developer work — function-based, filesystem-only, deterministic naming from config dicts, automatic Git tagging. No Python equivalent exists.

**The consensus**: version-controlled scripts + ArviZ InferenceData persistence + informal naming conventions. No dominant framework. The reason is structural: Bayesian model variants differ in *model topology* (different priors, likelihoods, parameterizations), not just in numeric hyperparameters. YAML config files can express `n_samples: 4000` but not "replace `pm.Normal('beta', ...)` with `pm.LogNormal('beta', ...)`."

### Design Patterns Evaluated

#### OOP Patterns

| Pattern | What It Does | Verdict for Tallgrass |
|---------|-------------|----------------------|
| **Strategy** (Protocol + swap) | Each variant is a class implementing `build()` | Overkill — variants differ by 3-5 lines inside a `with pm.Model()` block |
| **Template Method** (base class pipeline) | Base defines fit/predict/save, subclasses override model spec | Too heavy (7 abstract methods in PyMC-Marketing's `ModelBuilder`) |
| **Factory** (config → model) | Dict/dataclass drives model construction | Reasonable at 5+ variants; premature below that |
| **Subclassing** | Inherit from a base model class | Introduces accidental complexity; PyMC's context manager doesn't compose well with inheritance |

The research literature is clear: when model variants share 80-95% of their code and differ only in data (which prior, which constraint), **functional approaches beat behavioral (OOP) approaches**. The variation between experiments is data — different prior distributions, different hyperparameters — not different algorithms. A `beta_prior: str` parameter handles this more naturally than a class hierarchy.

#### Functional Patterns

| Pattern | What It Does | Verdict |
|---------|-------------|---------|
| **Higher-order functions** | Pipeline takes a `ModelBuilder` callable | Natural fit — matches existing codebase conventions |
| **Partial application** | `functools.partial` for parameter sweeps | Dead simple for numeric sweeps; loses type info |
| **Config dataclass + builder** | Frozen dataclass bundles all params; builder reads it | Best balance of type safety and flexibility |

#### Configuration Approaches

| Approach | When It Wins | When It Loses |
|----------|-------------|---------------|
| **YAML/TOML files** (Hydra) | Many numeric hyperparameters, team collaboration | Structural model variants (different priors/likelihoods) |
| **Frozen dataclasses** | Type-safe, immutable, no dependencies | Need to define the schema upfront |
| **Pydantic** | Runtime validation, rich error messages | 6.5x slower than dataclasses; overkill for internal configs |
| **Code-as-config** | Small number of structural variants | Doesn't scale to 20+ parameter combinations |

**Winner for Tallgrass**: Frozen dataclasses. Already the project convention for data models (`IndividualVote`, `RollCall`). No new dependencies. Type-checked by ty. Immutable, hashable, serializable via `dataclasses.asdict()`.

## Current Architecture Assessment

| Aspect | Current State | Risk |
|--------|--------------|------|
| Separation from production | Experiments in `experimental/` and `results/experimental_lab/` | Low |
| Code duplication | `run_experiment.py` rewrites model-building logic | **High** |
| Test coverage | Zero tests for experimental code | Medium |
| Platform awareness | Thread caps, core scheduling not enforced by experiment code | **High** |
| Documentation | Excellent (`TEMPLATE.md`, `experiment.md`) | Low |
| Integration path | Manual diff-and-merge | Medium |
| Reproducibility | Deterministic seed, exact parameter tracking | Low |
| Parameter drift | Relaxed thresholds in experimental models | Medium |

## Proposed Design: Experiment Variants via Composition

The core insight: experiments don't need a framework. They need **production functions that accept the variation as a parameter**, plus a thin experiment runner that enforces platform constraints and standardizes output.

### Architecture Overview

```
analysis/
    10_hierarchical/
        hierarchical.py          ← production pipeline (unchanged API)
        model_spec.py            ← NEW: model specification as data
    experiment_runner.py         ← NEW: shared experiment infrastructure
    experimental/
        irt_2d_experiment.py     ← structural variants (different model topology)
        irt_beta_experiment.py   ← prior variants (parameterizable)

results/experimental_lab/
    TEMPLATE.md
    2026-02-27_positive-beta/
        experiment.md
        run_experiment.py        ← thin: config + dispatch only
```

### Component 1: Model Specification as Data (`model_spec.py`)

Factor the part that varies — the beta prior — into a frozen dataclass that both production and experiments consume:

```python
@dataclass(frozen=True)
class BetaPriorSpec:
    """Specification for the bill discrimination (beta) prior."""
    distribution: Literal["normal", "lognormal", "halfnormal"]
    params: dict[str, float]  # e.g. {"mu": 0, "sigma": 1}

    def build(self, n_votes: int, dims: str = "vote") -> pm.Distribution:
        """Instantiate the PyMC distribution inside an active model context."""
        match self.distribution:
            case "normal":
                return pm.Normal("beta", shape=n_votes, dims=dims, **self.params)
            case "lognormal":
                return pm.LogNormal("beta", shape=n_votes, dims=dims, **self.params)
            case "halfnormal":
                return pm.HalfNormal("beta", shape=n_votes, dims=dims, **self.params)

# Production default
PRODUCTION_BETA = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})
```

Production's `build_per_chamber_model()` gains one new parameter with a default:

```python
def build_per_chamber_model(
    data: dict,
    n_samples: int,
    n_tune: int,
    n_chains: int,
    cores: int | None = None,
    target_accept: float = HIER_TARGET_ACCEPT,
    xi_offset_initvals: np.ndarray | None = None,
    beta_prior: BetaPriorSpec = PRODUCTION_BETA,  # ← NEW, defaults to current behavior
) -> tuple[az.InferenceData, float]:
```

Inside the model block, the hardcoded `beta = pm.Normal("beta", mu=0, sigma=1, ...)` becomes `beta = beta_prior.build(n_votes)`. **Zero behavior change for production callers** — the default is the current prior.

Experiments pass a different spec:

```python
from analysis.hierarchical import build_per_chamber_model
from analysis.hierarchical.model_spec import BetaPriorSpec

result = build_per_chamber_model(
    data,
    n_samples=2000,
    n_tune=1500,
    n_chains=4,
    beta_prior=BetaPriorSpec("lognormal", {"mu": 0, "sigma": 0.5}),
)
```

No duplication. No drift. One source of truth for the model-building logic.

### Component 2: Experiment Configuration Dataclass

Bundle all experiment parameters into a single typed, immutable object:

```python
@dataclass(frozen=True)
class ExperimentConfig:
    """Complete specification for an experiment run."""
    name: str
    description: str
    session: str = "2025-26"
    beta_prior: BetaPriorSpec = PRODUCTION_BETA
    n_samples: int = HIER_N_SAMPLES
    n_tune: int = HIER_N_TUNE
    n_chains: int = HIER_N_CHAINS
    target_accept: float = HIER_TARGET_ACCEPT
    include_joint: bool = False
    # Convergence thresholds — always match production
    rhat_threshold: float = RHAT_THRESHOLD
    ess_threshold: float = ESS_THRESHOLD
    max_divergences: int = MAX_DIVERGENCES

    def output_dir(self, base: Path) -> Path:
        """Deterministic output path from config fields."""
        return base / f"{self.name}"
```

This eliminates scattered constants and makes experiment configuration inspectable, serializable, and diffable. The convergence thresholds default to production values — experiments that relax them must do so explicitly.

### Component 3: Platform-Aware Experiment Runner (`experiment_runner.py`)

This is the critical piece that encodes platform knowledge. The runner validates the environment before any MCMC work begins:

```python
@dataclass(frozen=True)
class PlatformCheck:
    """Validates Apple Silicon MCMC constraints before sampling."""
    omp_threads: int
    openblas_threads: int
    cpu_count: int
    active_mcmc_processes: int
    pytensor_cxx: str  # Empty string = no compiler = pure Python fallback

    @classmethod
    def current(cls) -> PlatformCheck:
        """Snapshot the current platform state."""
        import os
        omp = int(os.environ.get("OMP_NUM_THREADS", "0"))
        openblas = int(os.environ.get("OPENBLAS_NUM_THREADS", "0"))
        # Check for other MCMC processes already running
        active = _count_active_mcmc_processes()
        return cls(
            omp_threads=omp,
            openblas_threads=openblas,
            cpu_count=os.cpu_count() or 1,
            active_mcmc_processes=active,
        )

    def validate(self, n_chains: int) -> list[str]:
        """Return list of warnings. Empty = all clear."""
        warnings = []
        if not self.pytensor_cxx:
            warnings.append(
                "FATAL: PyTensor has no C++ compiler (config.cxx is empty). "
                "Sampling will be ~18x slower in pure Python mode. "
                "Common causes: (1) Xcode updated but license not accepted — "
                "open Xcode.app and agree, or run `sudo xcodebuild -license accept`. "
                "(2) /usr/bin not on PATH — run via `just` or set PATH=/usr/bin:$PATH."
            )
        if self.omp_threads == 0 or self.omp_threads > 6:
            warnings.append(
                f"OMP_NUM_THREADS={self.omp_threads} (expected ≤6 for M3 Pro). "
                f"Set OMP_NUM_THREADS=6 to prevent E-core scheduling. "
                f"See docs/apple-silicon-mcmc-tuning.md"
            )
        if self.openblas_threads == 0 or self.openblas_threads > 6:
            warnings.append(
                f"OPENBLAS_NUM_THREADS={self.openblas_threads} (expected ≤6). "
                f"See ADR-0022."
            )
        if self.active_mcmc_processes > 0:
            warnings.append(
                f"{self.active_mcmc_processes} other MCMC process(es) detected. "
                f"Concurrent MCMC jobs force E-core scheduling (2.5x slowdown). "
                f"Run bienniums sequentially."
            )
        if n_chains > 6:
            warnings.append(
                f"n_chains={n_chains} exceeds P-core count (6). "
                f"Chains will spill to E-cores."
            )
        return warnings
```

The experiment runner calls `PlatformCheck.current().validate(config.n_chains)` before sampling begins. Warnings are printed prominently and logged to `metrics.json`. If thread pools aren't capped, the runner can either warn-and-continue or refuse-to-start (configurable).

The runner also handles the standard experiment lifecycle:

```python
def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Standard experiment lifecycle with platform guardrails."""
    # 1. Platform validation
    platform = PlatformCheck.current()
    warnings = platform.validate(config.n_chains)
    if warnings:
        for w in warnings:
            print(f"  ⚠ PLATFORM WARNING: {w}")

    # 2. Load data (identical to production)
    ks = KSSession.from_session_string(config.session)
    house_matrix, senate_matrix, _ = load_eda_matrices(ks.results_dir / "01_eda" / "latest")
    house_pca, senate_pca = load_pca_scores(ks.results_dir / "02_pca" / "latest")
    rollcalls, legislators = load_metadata(ks.data_dir)

    # 3. Run per-chamber models (production function, experimental config)
    for chamber, matrix, pca in [("House", house_matrix, house_pca), ...]:
        data = prepare_hierarchical_data(matrix, legislators, chamber)
        xi_init = compute_pca_initvals(pca, data)
        idata, time_s = build_per_chamber_model(
            data,
            n_samples=config.n_samples,
            n_tune=config.n_tune,
            n_chains=config.n_chains,
            beta_prior=config.beta_prior,
            xi_offset_initvals=xi_init,
        )
        # 4. Convergence diagnostics (production thresholds)
        convergence = check_hierarchical_convergence(idata, chamber)
        # ...

    # 5. Save standardized output
    # ...
```

### Component 4: Thin Experiment Scripts

With the runner doing the heavy lifting, experiment scripts become configuration-only:

```python
"""Positive beta constraint experiment."""
from analysis.experiment_runner import ExperimentConfig, run_experiment
from analysis.hierarchical.model_spec import BetaPriorSpec

VARIANTS = [
    ExperimentConfig(
        name="run_01_baseline",
        description="beta ~ Normal(0, 1) — current production model",
        beta_prior=BetaPriorSpec("normal", {"mu": 0, "sigma": 1}),
    ),
    ExperimentConfig(
        name="run_02_lognormal",
        description="beta ~ LogNormal(0, 0.5) — soft positive",
        beta_prior=BetaPriorSpec("lognormal", {"mu": 0, "sigma": 0.5}),
    ),
    ExperimentConfig(
        name="run_03_halfnormal",
        description="beta ~ HalfNormal(1) — hard zero floor",
        beta_prior=BetaPriorSpec("halfnormal", {"sigma": 1}),
    ),
]

if __name__ == "__main__":
    for variant in VARIANTS:
        run_experiment(variant)
```

**789 lines → ~25 lines.** The experiment script is pure configuration. All model-building, sampling, diagnostics, and reporting logic lives in production code (tested, maintained) or the shared runner (tested once, used everywhere).

### What Changes for Each Experiment Type

**Prior/hyperparameter variants** (beta prior, alpha prior, target_accept): Use the `ExperimentConfig` + `BetaPriorSpec` pattern. The model-building function accepts the variation as a parameter. No code duplication.

**Structural variants** (2D IRT, 1PL Rasch, mixed centering): These genuinely need different model-building code. They stay in `analysis/experimental/` as standalone scripts, but they import shared data-loading and diagnostic functions from production. The `experiment_runner.py` provides `PlatformCheck` and standardized output even for structural variants.

**Sampling/convergence variants** (more draws, different n_chains, ADVI init): Pure configuration changes — handled entirely by `ExperimentConfig` fields.

### What Doesn't Change

- The existing `TEMPLATE.md` and `experiment.md` documentation standard
- The `results/experimental_lab/YYYY-MM-DD_description/` directory convention
- The append-only results policy
- The 91st-biennium default session rationale
- The promotion path (experiment → ADR → production commit)

## Experiment Monitoring and Process Visibility

> **Status (2026-03-02):** The callback-based monitoring described below (Layers 1-2)
> was designed for PyMC's `pm.sample(callback=...)` API. After migrating to nutpie
> (ADR-0051, ADR-0053), which uses Rust threads instead of Python multiprocessing,
> the callback infrastructure became dead code and was removed (ADR-0080). nutpie
> provides its own terminal progress bar (step size, divergences, gradients/draw per
> chain). Layer 3 (PID lock + process group) remains in production as
> `ExperimentLifecycle`. `just monitor` now checks the PID file.

### The Opacity Problem

When a hierarchical IRT experiment runs, `ps aux` shows something like:

```
josephclaeys  12345  98.0  4.2  ...  python
josephclaeys  12346  99.1  3.8  ...  python
josephclaeys  12347  98.5  3.9  ...  python
josephclaeys  12348  97.8  3.7  ...  python
josephclaeys  12349   0.2  1.1  ...  python
```

Five Python processes. Which is the parent? Which are chains? Are they from the current experiment or orphans from a crashed run? Are they tuning or drawing? How far along are they? There's no way to tell without digging into `/proc` or guessing from CPU usage patterns.

This opacity has caused real problems. Orphan PyMC workers from a killed experiment coexist with a new run's workers, creating 8-process CPU saturation that forces E-core scheduling and skews timing data by 2.5x (documented in `docs/hierarchical-4-chain-experiment.md`). The current mitigation is manual: `ps aux | grep run_experiment | grep -v grep` before every run. That's fragile.

### How PyMC Manages Processes (What We're Working With)

Reading the actual PyMC 5.27.1 source (`pymc/sampling/parallel.py`), the architecture is:

1. **`ParallelSampler`** (parent process) orchestrates multiple chains
2. **`ProcessAdapter`** creates one child per chain via `multiprocessing.Process(daemon=True, name=f"worker_chain_{chain}", target=_run_process, ...)`
3. **`_Process`** (child process) runs the sampling loop, communicates via `multiprocessing.Pipe` + shared memory (`RawArray`)

Key facts for our design:
- On macOS ARM64, PyMC defaults to `fork` (not `spawn`). Forked children inherit imports and environment.
- Processes are named `worker_chain_0`, `worker_chain_1`, etc. — but this is Python-internal only. `ps` shows "python".
- The `daemon=True` flag means children are killed when the parent exits normally. But SIGKILL on the parent leaves orphans.
- **PyMC supports a `callback` parameter** on `pm.sample()` that fires in the parent process after each draw from any chain. It receives a `Draw` namedtuple with `chain`, `draw_idx`, `tuning`, `stats`, and `point`. This is our primary hook.
- Shutdown (`ProcessAdapter.terminate_all`) sends "abort" via pipe, waits 2 seconds, then sends SIGTERM. If the parent is killed before this runs, children hang on `pipe.recv()` forever.

### Solution: Three Layers, No Library Modifications

The monitoring design uses three complementary layers. None require modifying PyMC, PyTensor, or any other dependency.

#### Layer 1: Process Titles via `setproctitle` (~15 lines)

The [`setproctitle`](https://github.com/dvarrazzo/py-setproctitle) package changes the OS-level process title visible to `ps`, `top`, and Activity Monitor. It's used in production by Dask, Ray, Celery, and PostgreSQL. The implementation is a C extension that overwrites `argv` memory — nanosecond-scale, zero measurable overhead.

**macOS ARM64 status**: Works with `ps` and Activity Monitor when using `uv`/`venv` Python (a Conda-specific binary compatibility issue was fixed in 1.3.x and doesn't affect us).

**Parent process**: Set the title at experiment start, update at phase transitions.

```python
from setproctitle import setproctitle

setproctitle("tallgrass:hierarchical:house:sampling 0/2000")
```

**Dynamic updates via PyMC callback**: Update the parent's title with progress from the callback:

```python
def progress_callback(trace, draw):
    phase = "tuning" if draw.tuning else "sampling"
    setproctitle(
        f"tallgrass:hierarchical:house:{phase} "
        f"chain={draw.chain} draw={draw.draw_idx}/{total_draws}"
    )

pm.sample(..., callback=progress_callback)
```

**Child processes**: Since PyMC uses `fork` on ARM macOS, children inherit the imported `setproctitle` module. We can inject a title-setting call by wrapping the model's logp function or — more practically — by accepting that child titles won't update per-draw (they don't have a callback hook) and instead setting them once at fork time. This is a limitation we accept: the parent's title shows aggregate progress, and `ps` shows all five processes with the `tallgrass:` prefix so they're identifiable.

**What `ps` looks like after**:

```
josephclaeys  12345  0.2  1.1  tallgrass:hierarchical:house:sampling chain=2 draw=1500/2000
josephclaeys  12346 98.0  4.2  tallgrass:worker_chain_0
josephclaeys  12347 99.1  3.8  tallgrass:worker_chain_1
josephclaeys  12348 98.5  3.9  tallgrass:worker_chain_2
josephclaeys  12349 97.8  3.7  tallgrass:worker_chain_3
```

Now `ps aux | grep tallgrass` finds everything. `pgrep -f tallgrass` counts them. Orphans are obvious — they'll still say `tallgrass:worker_chain_N` after the parent exits.

For the child process titles, we use a `fork`-safe approach: set the parent's title to include `tallgrass:` before calling `pm.sample()`. Since `fork` copies the parent's memory, each child inherits the `setproctitle` module state. We then use `multiprocessing`'s initializer pattern (or a one-time setup in `_run_process` if we choose to monkeypatch — see below) to set distinct titles per chain. However, the simplest approach that avoids touching PyMC internals is to set the parent's title and rely on the `daemon=True` flag plus process group management for cleanup.

#### Layer 2: Status File with Atomic Writes (~30 lines)

A JSON status file written atomically (write-to-temp-then-`os.replace`) every N draws. This is the same pattern used by Dask's Nanny for worker health monitoring. The PyMC callback handles the writes:

```python
import json, os, tempfile
from datetime import datetime, timezone

STATUS_PATH = Path("/tmp/tallgrass/experiment.status.json")

def write_status(status: dict) -> None:
    """Atomic write: temp file + os.replace (POSIX atomic rename)."""
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = tempfile.NamedTemporaryFile(
        mode="w", dir=STATUS_PATH.parent, delete=False, suffix=".tmp",
    )
    try:
        json.dump(status, fd)
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        os.replace(fd.name, STATUS_PATH)
    except BaseException:
        fd.close()
        os.unlink(fd.name)
        raise

# In the callback (fires in parent process, every draw from every chain):
def monitoring_callback(trace, draw):
    if draw.draw_idx % 50 == 0:  # Update every 50 draws (~2-5 seconds)
        write_status({
            "pid": os.getpid(),
            "experiment": "positive-beta",
            "variant": "lognormal",
            "chamber": "house",
            "phase": "tuning" if draw.tuning else "sampling",
            "chain": draw.chain,
            "draw": draw.draw_idx,
            "total_draws": total_draws,
            "divergences": sum(
                1 for s in draw.stats if s.get("diverging", False)
            ),
            "elapsed_s": round(time.time() - start_time, 1),
            "heartbeat": datetime.now(timezone.utc).isoformat(),
        })
```

**Reading the status file** from another terminal:

```bash
# One-shot
cat /tmp/tallgrass/experiment.status.json | python -m json.tool

# Live watch
watch -n 5 'cat /tmp/tallgrass/experiment.status.json | python -m json.tool'
```

Or a Justfile recipe:

```just
monitor:
    @watch -n 5 'cat /tmp/tallgrass/experiment.status.json 2>/dev/null | python -m json.tool || echo "No experiment running"'
```

#### Layer 3: Process Group + PID Lock (~20 lines)

Two mechanisms for lifecycle management:

**Process group isolation** — `os.setpgrp()` at experiment start creates a new process group. All PyMC workers inherit it (both `fork` and `spawn` modes). This enables single-command cleanup:

```python
os.setpgrp()  # This process becomes group leader
# ... all pm.sample() workers inherit the group ...

# From another terminal, to kill everything:
# kill -TERM -$(cat /tmp/tallgrass/experiment.pid)
```

**PID file with fcntl locking** — prevents concurrent experiments and detects orphans. The `pid` library (5 lines) uses advisory file locks that the OS releases automatically even after SIGKILL:

```python
from pid import PidFile, PidFileAlreadyLockedError

try:
    with PidFile("tallgrass-experiment", piddir="/tmp/tallgrass"):
        run_experiment(config)
except PidFileAlreadyLockedError:
    print("Another experiment is already running. Kill it first or wait.")
    sys.exit(1)
```

If a previous experiment was killed with SIGKILL, the lock is released by the OS, and the next run detects the stale PID file and proceeds. No manual cleanup needed.

**SIGTERM handler** — converts SIGTERM to a clean exit so `atexit` runs:

```python
import signal
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
```

This ensures the PID file, status file, and any other cleanup runs when you `kill` the process group.

### Combined: What the Monitoring Callback Looks Like

```python
def create_monitoring_callback(
    experiment_name: str,
    variant: str,
    chamber: str,
    total_draws: int,
    status_path: Path,
) -> Callable:
    """Create a PyMC callback that updates process title + status file."""
    start_time = time.time()

    def callback(trace, draw):
        phase = "tuning" if draw.tuning else "sampling"

        # Layer 1: Process title (every draw, nanosecond cost)
        setproctitle(
            f"tallgrass:{experiment_name}:{chamber}:{phase} "
            f"draw={draw.draw_idx}/{total_draws}"
        )

        # Layer 2: Status file (every 50 draws, ~2-5s interval)
        if draw.draw_idx % 50 == 0:
            write_status(status_path, {
                "pid": os.getpid(),
                "experiment": experiment_name,
                "variant": variant,
                "chamber": chamber,
                "phase": phase,
                "chain": draw.chain,
                "draw": draw.draw_idx,
                "total_draws": total_draws,
                "elapsed_s": round(time.time() - start_time, 1),
                "heartbeat": datetime.now(timezone.utc).isoformat(),
            })

    return callback
```

### Why Not Modify PyMC?

Tinkering with the open source code is tempting — inject `setproctitle` into `_Process.run()` so child processes show `tallgrass:worker_chain_0:draw_1500`. But:

1. **Maintenance burden**: Every PyMC update requires re-applying the patch. PyMC releases frequently (5.27.1 is current; 5.28 is likely months away).
2. **The callback already exists**: PyMC's `callback` parameter gives us per-draw visibility in the parent process. The parent is what we actually need to monitor — it's the one that reports progress, saves results, and manages the lifecycle.
3. **Child process titles are secondary**: Knowing that 4 `tallgrass:worker_chain_N` processes exist is sufficient for orphan detection. We don't need per-draw progress from each child — the parent's callback aggregates this.
4. **Process group management handles cleanup**: `kill -TERM -$PGID` reaches all children regardless of their titles.

The one scenario where modifying PyMC would help is if children hang on `pipe.recv()` after the parent dies (the "abort message never sent" case). But process group management solves this too — SIGTERM reaches every process in the group.

### Production: nutpie (Implemented)

All MCMC models now use nutpie (ADR-0051, ADR-0053), which eliminates the multiprocessing problem entirely:

- **Single process**: All chains run as Rust threads within one Python process. No child processes, no orphans.
- **Built-in progress bar**: nutpie's Rust `indicatif` crate shows per-chain step size, divergences, and gradients/draw directly in the terminal.
- **One PID to monitor**: `setproctitle` on the single process shows everything. `ps` shows one process, not five.
- **Non-blocking API**: `sampler.inspect()` can grab partial posterior at any time without interrupting.

This made the PyMC callback infrastructure (Layers 1-2 above) unnecessary, and it was removed in ADR-0080.

### New Dependencies

| Package | Purpose | Size | macOS ARM64 |
|---------|---------|------|-------------|
| `setproctitle` | OS-level process titles | ~20 KB wheel | Pre-built wheel available |
| `pid` | PID file with fcntl locking | ~10 KB pure Python | Works everywhere |
| `psutil` (optional) | Process tree inspection for `PlatformCheck` | ~500 KB wheel | Pre-built wheel available |

`setproctitle` and `pid` are tiny. `psutil` is only needed if `PlatformCheck` inspects the process tree; a `subprocess + ps` fallback avoids the dependency.

## Implementation Plan

### Phase 1: Model Specification Extraction (No Behavior Change)

1. Create `analysis/07_hierarchical/model_spec.py` with `BetaPriorSpec` and `PRODUCTION_BETA`
2. Add `beta_prior: BetaPriorSpec = PRODUCTION_BETA` parameter to `build_per_chamber_model()` and `build_joint_model()`
3. Replace the hardcoded `beta = pm.Normal(...)` line with `beta = beta_prior.build(n_votes)`
4. Verify all 1,172 tests pass — this must be a zero-behavior-change refactor
5. Run `just hierarchical --session 2025-26` and diff output against previous run

**Risk:** Low. The default parameter preserves current behavior. Existing callers are unaffected.

### Phase 2: Process Monitoring Infrastructure

1. Add `setproctitle` to `pyproject.toml` dependencies
2. Create `analysis/experiment_monitor.py` with:
   - `PlatformCheck` dataclass with `validate()` encoding M3 Pro rules from ADR-0022
   - `write_status()` — atomic JSON status file writer
   - `ExperimentLifecycle` context manager — wraps PID file, process group, SIGTERM handler, cleanup
3. Add `_count_active_mcmc_processes()` using `pgrep -f tallgrass` (avoids `psutil` dependency)
4. Add Justfile recipe: `just monitor` → check PID file (nutpie shows its own progress bar)
5. Add tests: mock environment variables, verify warnings fire; verify atomic write; verify PID lock conflict detection

**Risk:** Low. Pure addition. Does not modify production code.

### Phase 3: Experiment Configuration Dataclass

1. Add `ExperimentConfig` to `experiment_runner.py`
2. Add `output_dir()` method for deterministic naming
3. Add `to_json()` / `from_json()` for serialization (via `dataclasses.asdict`)
4. Add Justfile recipe: `just experiment *args` → `uv run python -m analysis.experiment_runner {{args}}`

**Risk:** Low. Pure addition.

### Phase 4: Shared Experiment Runner

1. Add `run_experiment()` function that orchestrates: `ExperimentLifecycle` → platform check → data load → model build → sample → diagnose → save → report
2. Factor out the data-loading, PCA-init, and reporting logic currently duplicated across experiment scripts
3. Support both per-chamber and joint model runs via config flags
4. Produce standardized `metrics.json` with platform info, config dump, convergence results, and timing

**Risk:** Medium. This is the largest change. Test by running the existing positive-beta experiment through the new runner and comparing output.

### Phase 5: Retrofit Existing Experiments (Optional)

1. Rewrite `results/experimental_lab/2026-02-27_positive-beta/run_experiment.py` to use the new runner
2. Verify identical output (metrics, plots, report)
3. Leave older experiments as-is (they're completed; no value in rewriting)

**Risk:** Low. Old experiments are archival. New experiments use the new pattern going forward.

### Scope Boundaries

**In scope:**
- `BetaPriorSpec` for the hierarchical IRT beta prior (the active experiment axis)
- `PlatformCheck` for Apple Silicon MCMC constraints
- `ExperimentConfig` for parameter bundles
- Shared experiment runner for the hierarchical IRT pipeline

**Out of scope (until needed):**
- AlphaPriorSpec, ParamSpec for other priors (add when an experiment needs them)
- Registry pattern for model variants (add at 5+ structural variants)
- YAML config loading (add if collaborators join who prefer config files)
- Experiment comparison dashboard (add if comparing 10+ variants becomes routine)
- nutpie sampler migration (evaluate after the monitoring framework is stable; eliminates multiprocessing entirely)
- Modifying PyMC internals for child process titles (the callback approach gives sufficient parent-side visibility)

## Why Not OOP?

This is a common question. The short answer: **the variation between experiments is data (which prior, which hyperparameters), not behavior (which algorithm)**. When variation is data, functional approaches (parameterized functions, config dataclasses) beat behavioral approaches (Strategy classes, Template Methods) every time.

Specifically:
- **PyMC's context manager** (`with pm.Model():`) doesn't compose well with class inheritance. You can't split a `with` block across a base class `setup()` and a subclass `build()`.
- **Model variants share 90-95% of their code.** The difference between the baseline and LogNormal experiments is literally one line: `pm.Normal("beta", ...)` vs `pm.LogNormal("beta", ...)`. A class hierarchy wrapping one line of variation is pure overhead.
- **The iteration cycle is fast.** Write variant → sample → look at diagnostics → decide. Classes add indirection without adding clarity.
- **Python functions are first-class.** The Strategy pattern exists to pass behavior as a parameter in languages without first-class functions. Python has `Callable`.

The Template Method pattern (PyMC-Marketing's `ModelBuilder`) is justified for *deployment* — when you need `save()`, `load()`, `predict()` on new data. Tallgrass experiments are exploratory: build, sample, diagnose, compare, decide. The lifecycle is simpler.

## What the Framework Encodes

The key insight is that an experiment framework for Tallgrass isn't about tracking or configuration management — those problems are already solved by `experiment.md` documents and `metrics.json` files. The framework's job is to encode **hard-won operational knowledge** so that each new experiment doesn't have to rediscover it:

1. **Platform constraints** (ADR-0022): Thread pool caps, no concurrent MCMC, no E-core scheduling. The `PlatformCheck` module makes these violations visible before hours of sampling begin.

2. **Model specification as data** (`BetaPriorSpec`): Production functions accept experimental variation as a parameter, eliminating the need to copy-paste and modify model-building code.

3. **Convergence standards**: Production thresholds (R-hat < 1.01, ESS > 400) are the default. Experiments that relax them must do so explicitly in the config, making the deviation visible in the `metrics.json` dump.

4. **Initialization rules** (ADR-0044, ADR-0045): PCA init with `adapt_diag` (no jitter) when using 4+ chains. The runner applies these rules automatically based on config.

5. **Output standardization**: Every experiment produces the same directory structure, metrics format, and report. All hierarchical experiments call `build_hierarchical_report()` — the same 18-22 section production report (party posteriors, ICC, variance decomposition, dispersion, shrinkage scatter/table, forest plots, convergence diagnostics, cross-chamber comparison, flat vs hier comparison, analysis parameters). Comparison across variants is mechanical, not manual.

## References

- Gelman, A. et al. (2020). Bayesian Workflow. *arXiv:2011.01808*.
- Martin, O.A., Kumar, R., & Lao, J. (2022). *Bayesian Modeling and Computation in Python*. Ch. 9.
- Papaspiliopoulos, O., Roberts, G.O., & Sköld, M. (2007). A general framework for the parametrization of hierarchical models. *Statistical Science*, 22(1), 59–73.
- Kieffer, A. et al. (2024). MLXP: A Framework for Conducting Replicable Experiments in Python. *arXiv:2402.13831*.
- Datseris, G. & Isensee, J. (2023). DrWatson: the perfect sidekick for your scientific inquiries. *JOSS*, 5(54), 2673.
- Caron, L.P. et al. (2022). epyc: a Python library for managing computational experiments. *JOSS*, 7(73), 3764.

## Related Documents

- [Apple Silicon MCMC Tuning](apple-silicon-mcmc-tuning.md) — P/E core scheduling, thread pool caps, parallel chain rules
- [Hierarchical Convergence Improvement](hierarchical-convergence-improvement.md) — the 9-priority plan this framework supports
- [Hierarchical 4-Chain Experiment](hierarchical-4-chain-experiment.md) — jitter mode-splitting discovery
- [Hierarchical PCA Init Experiment](hierarchical-pca-init-experiment.md) — R-hat fix, ESS threshold analysis
- [2D IRT Deep Dive](2d-irt-deep-dive.md) — structural variant (different model topology)
- [ADR-0022](adr/0022-analysis-parallelism-and-timing.md) — Analysis parallelism and runtime timing
- [ADR-0044](adr/0044-hierarchical-pca-informed-init.md) — PCA-informed initialization
- [ADR-0045](adr/0045-4-chain-hierarchical-irt.md) — 4-chain with adapt_diag
