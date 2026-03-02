"""Experiment monitoring infrastructure for MCMC sampling.

Two components:
  1. PlatformCheck — validates Apple Silicon MCMC constraints before sampling
  2. ExperimentLifecycle — context manager for PID lock, process group, cleanup

nutpie provides its own terminal progress bar (step size, divergences,
gradients/draw per chain via Rust's indicatif crate). Use `just monitor`
to check whether an experiment process is alive.

See docs/experiment-framework-deep-dive.md for design rationale.
"""

import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

STATUS_DIR = Path("/tmp/tallgrass")
STATUS_PATH = STATUS_DIR / "experiment.status.json"
PID_PATH = STATUS_DIR / "experiment.pid"


# ── PlatformCheck ────────────────────────────────────────────────────────────


def _count_active_mcmc_processes() -> int:
    """Count other tallgrass MCMC processes using pgrep."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "tallgrass"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return 0
        pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        # Exclude our own PID
        own_pid = str(os.getpid())
        other_pids = [p for p in pids if p != own_pid]
        return len(other_pids)
    except FileNotFoundError:
        return 0


@dataclass(frozen=True)
class PlatformCheck:
    """Validates Apple Silicon MCMC constraints before sampling.

    Encodes the hard-won rules from ADR-0022 and docs/apple-silicon-mcmc-tuning.md:
    - Thread pools must be capped at 6 (P-core count on M3 Pro)
    - No concurrent MCMC jobs (E-core scheduling causes 2.5x slowdown)
    - PyTensor needs a C++ compiler (otherwise 18x slower in pure Python)
    """

    omp_threads: int
    openblas_threads: int
    cpu_count: int
    active_mcmc_processes: int
    pytensor_cxx: str

    @classmethod
    def current(cls) -> "PlatformCheck":
        """Snapshot the current platform state."""
        omp = int(os.environ.get("OMP_NUM_THREADS", "0"))
        openblas = int(os.environ.get("OPENBLAS_NUM_THREADS", "0"))

        # Get PyTensor compiler status
        pytensor_cxx = ""
        try:
            import pytensor

            pytensor_cxx = str(getattr(pytensor.config, "cxx", ""))
        except ImportError:
            pass

        return cls(
            omp_threads=omp,
            openblas_threads=openblas,
            cpu_count=os.cpu_count() or 1,
            active_mcmc_processes=_count_active_mcmc_processes(),
            pytensor_cxx=pytensor_cxx,
        )

    def validate(self, n_chains: int) -> list[str]:
        """Return list of warnings. Empty = all clear.

        Warnings prefixed with 'FATAL:' indicate conditions that will produce
        invalid or severely degraded results.
        """
        warnings: list[str] = []
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
                f"OMP_NUM_THREADS={self.omp_threads} (expected <=6 for M3 Pro). "
                f"Set OMP_NUM_THREADS=6 to prevent E-core scheduling. "
                f"See docs/apple-silicon-mcmc-tuning.md"
            )
        if self.openblas_threads == 0 or self.openblas_threads > 6:
            warnings.append(
                f"OPENBLAS_NUM_THREADS={self.openblas_threads} (expected <=6). See ADR-0022."
            )
        if self.active_mcmc_processes > 0:
            warnings.append(
                f"{self.active_mcmc_processes} other MCMC process(es) detected. "
                f"Concurrent MCMC jobs force E-core scheduling (2.5x slowdown). "
                f"Run bienniums sequentially."
            )
        if n_chains > 6:
            warnings.append(
                f"n_chains={n_chains} exceeds P-core count (6). Chains will spill to E-cores."
            )
        return warnings


# ── Status File (Atomic Writes) ─────────────────────────────────────────────


def write_status(status: dict, path: Path = STATUS_PATH) -> None:
    """Atomic write: temp file + os.replace (POSIX atomic rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    )
    try:
        json.dump(status, fd, default=str)
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        os.replace(fd.name, path)
    except BaseException:
        fd.close()
        try:
            os.unlink(fd.name)
        except OSError:
            pass
        raise


# ── Experiment Lifecycle ────────────────────────────────────────────────────


class ExperimentLifecycle:
    """Context manager for experiment process lifecycle.

    On enter:
      - Creates /tmp/tallgrass/ directory
      - Writes PID file with advisory fcntl lock (prevents concurrent experiments)
      - Sets process group (os.setpgrp) for clean cleanup
      - Installs SIGTERM handler for graceful exit
      - Sets initial process title

    On exit:
      - Releases lock and removes PID file
      - Removes status file
      - Restores original process title
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self._pid_fd: int | None = None
        self._original_title: str | None = None
        self._original_sigterm = None

    def __enter__(self) -> "ExperimentLifecycle":
        STATUS_DIR.mkdir(parents=True, exist_ok=True)

        # PID file with advisory lock
        try:
            self._pid_fd = os.open(str(PID_PATH), os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
        except OSError as e:
            msg = f"Failed to create PID file {PID_PATH}: {e}"
            raise RuntimeError(msg) from e
        try:
            fcntl.flock(self._pid_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self._pid_fd)
            self._pid_fd = None
            msg = (
                "Another experiment is already running (PID lock held). "
                "Kill it first or wait for it to finish."
            )
            raise RuntimeError(msg)

        os.write(self._pid_fd, str(os.getpid()).encode())
        os.fsync(self._pid_fd)

        # Process group for clean cleanup
        try:
            os.setpgrp()
        except OSError:
            pass  # May fail if already group leader

        # SIGTERM → clean exit (so atexit/finally runs)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

        # Process title
        try:
            from setproctitle import getproctitle, setproctitle

            self._original_title = getproctitle()
            setproctitle(f"tallgrass:{self.experiment_name}:starting")
        except ImportError:
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Release PID lock
        if self._pid_fd is not None:
            try:
                fcntl.flock(self._pid_fd, fcntl.LOCK_UN)
                os.close(self._pid_fd)
            except OSError:
                pass
            try:
                PID_PATH.unlink(missing_ok=True)
            except OSError:
                pass

        # Remove status file
        try:
            STATUS_PATH.unlink(missing_ok=True)
        except OSError:
            pass

        # Restore SIGTERM handler
        if self._original_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            except OSError, ValueError:
                pass

        # Restore process title
        if self._original_title is not None:
            try:
                from setproctitle import setproctitle

                setproctitle(self._original_title)
            except ImportError:
                pass
