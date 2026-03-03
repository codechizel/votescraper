"""Shared experiment runner for hierarchical IRT experiments.

Eliminates the 200+ lines of duplicated model-building code in experiment scripts.
An experiment becomes a ~25-line config that passes a BetaPriorSpec to the production
model-building functions.

Components:
  - ExperimentConfig: frozen dataclass bundling all experiment parameters
  - compute_pca_initvals: PCA-informed initialization for xi_offset
  - run_experiment: full experiment lifecycle (platform check → data load → sample →
    diagnose → extract → plot → report → metrics)

See docs/experiment-framework-deep-dive.md for design rationale.

Usage:
    from analysis.experiment_runner import ExperimentConfig, run_experiment
    from analysis.model_spec import BetaPriorSpec

    config = ExperimentConfig(
        name="run_01_baseline",
        description="beta ~ Normal(0, 1)",
        beta_prior=BetaPriorSpec("normal", {"mu": 0, "sigma": 1}),
    )
    metrics = run_experiment(config, output_base=Path("results/experimental_lab/my-exp"))
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from analysis.model_spec import PRODUCTION_BETA, BetaPriorSpec
except ModuleNotFoundError:
    from model_spec import PRODUCTION_BETA, BetaPriorSpec  # type: ignore[no-redef]

try:
    from analysis.experiment_monitor import ExperimentLifecycle, PlatformCheck
except ModuleNotFoundError:
    from experiment_monitor import ExperimentLifecycle, PlatformCheck  # type: ignore[no-redef]

try:
    from analysis.hierarchical import (
        build_joint_model,
        build_per_chamber_model,
        check_hierarchical_convergence,
        compute_flat_hier_correlation,
        compute_variance_decomposition,
        extract_group_params,
        extract_hierarchical_ideal_points,
        fix_joint_sign_convention,
        plot_dispersion,
        plot_icc,
        plot_joint_party_spread,
        plot_party_posteriors,
        plot_shrinkage_scatter,
        prepare_hierarchical_data,
        print_header,
    )
except ModuleNotFoundError:
    from hierarchical import (  # type: ignore[no-redef]
        build_joint_model,
        build_per_chamber_model,
        check_hierarchical_convergence,
        compute_flat_hier_correlation,
        compute_variance_decomposition,
        extract_group_params,
        extract_hierarchical_ideal_points,
        fix_joint_sign_convention,
        plot_dispersion,
        plot_icc,
        plot_joint_party_spread,
        plot_party_posteriors,
        plot_shrinkage_scatter,
        prepare_hierarchical_data,
        print_header,
    )

try:
    from analysis.hierarchical_report import build_hierarchical_report
except ModuleNotFoundError:
    from hierarchical_report import build_hierarchical_report  # type: ignore[no-redef]

try:
    from analysis.irt import (
        RANDOM_SEED,
        load_eda_matrices,
        load_metadata,
        load_pca_scores,
        plot_forest,
    )
except ModuleNotFoundError:
    from irt import (  # type: ignore[no-redef]
        RANDOM_SEED,
        load_eda_matrices,
        load_metadata,
        load_pca_scores,
        plot_forest,
    )

try:
    from analysis.report import ReportBuilder
except ModuleNotFoundError:
    from report import ReportBuilder  # type: ignore[no-redef]


# ── Constants ────────────────────────────────────────────────────────────────

# Import production defaults from hierarchical.py
HIER_N_SAMPLES = 2000
HIER_N_TUNE = 1500
HIER_N_CHAINS = 4
HIER_TARGET_ACCEPT = 0.95


# ── ExperimentConfig ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete specification for a hierarchical IRT experiment run.

    All parameters have production defaults. Experiments override only what they
    vary. The config is serializable via dataclasses.asdict() and lands in
    metrics.json for reproducibility.
    """

    name: str
    description: str
    session: str = "2025-26"
    beta_prior: BetaPriorSpec = PRODUCTION_BETA
    n_samples: int = HIER_N_SAMPLES
    n_tune: int = HIER_N_TUNE
    n_chains: int = HIER_N_CHAINS
    target_accept: float = HIER_TARGET_ACCEPT
    include_joint: bool = False
    chambers: tuple[str, ...] = ("House", "Senate")


# ── PCA Initialization ──────────────────────────────────────────────────────


def compute_pca_initvals(pca_scores: pl.DataFrame, data: dict) -> np.ndarray:
    """Compute xi_offset initvals from PCA PC1 scores.

    Standardizes PC1 to unit normal for use as xi_offset starting values.
    This prevents reflection mode-splitting — see ADR-0044.
    """
    slug_order = {s: i for i, s in enumerate(data["leg_slugs"])}
    pc1_vals = (
        pca_scores.filter(pl.col("legislator_slug").is_in(data["leg_slugs"]))
        .sort(pl.col("legislator_slug").replace_strict(slug_order))["PC1"]
        .to_numpy()
    )
    pc1_std = (pc1_vals - pc1_vals.mean()) / (pc1_vals.std() + 1e-8)
    return pc1_std.astype(np.float64)


# ── Per-Chamber Run ──────────────────────────────────────────────────────────


def _run_per_chamber(
    chamber: str,
    config: ExperimentConfig,
    matrix: pl.DataFrame,
    pca_scores: pl.DataFrame,
    legislators: pl.DataFrame,
    flat_ip: pl.DataFrame | None,
    out_dir: Path,
) -> dict:
    """Run one chamber with the specified experiment config. Full production output."""
    ch = chamber.lower()
    data = prepare_hierarchical_data(matrix, legislators, chamber)

    print_header(f"HIERARCHICAL IRT — {chamber}")
    print(f"  {data['n_legislators']} legislators x {data['n_votes']} votes")
    print(
        f"  Observed cells: {data['n_obs']:,} / {data['n_legislators'] * data['n_votes']:,} "
        f"({data['n_obs'] / (data['n_legislators'] * data['n_votes']):.1%})"
    )
    print(f"  Yea rate: {data['y'].mean():.3f}")
    for p in range(data["n_parties"]):
        n = int((data["party_idx"] == p).sum())
        print(f"  {data['party_names'][p]}: {n} legislators")

    # PCA init
    xi_init = compute_pca_initvals(pca_scores, data)
    print(f"  PCA init: {len(xi_init)} params, range [{xi_init.min():.2f}, {xi_init.max():.2f}]")

    # Monitoring callback
    # Sample using production function with experimental beta_prior
    print_header(f"SAMPLING — {chamber}")
    idata, sampling_time = build_per_chamber_model(
        data,
        n_samples=config.n_samples,
        n_tune=config.n_tune,
        n_chains=config.n_chains,
        target_accept=config.target_accept,
        xi_offset_initvals=xi_init,
        beta_prior=config.beta_prior,
    )

    # Convergence
    convergence = check_hierarchical_convergence(idata, chamber)

    # Extract results
    print_header(f"EXTRACTING RESULTS — {chamber}")
    ideal_points = extract_hierarchical_ideal_points(
        idata,
        data,
        legislators,
        flat_ip=flat_ip,
    )
    group_params = extract_group_params(idata, data)
    icc_df = compute_variance_decomposition(idata, data)

    flat_corr = float("nan")
    if flat_ip is not None:
        flat_corr = compute_flat_hier_correlation(ideal_points, flat_ip, chamber)

    # Print group params
    print("\n  Group-level parameters:")
    for row in group_params.iter_rows(named=True):
        print(
            f"    {row['party']}: mu={row['mu_mean']:+.3f} "
            f"[{row['mu_hdi_2.5']:+.3f}, {row['mu_hdi_97.5']:+.3f}], "
            f"sigma={row['sigma_within_mean']:.3f}"
        )

    # Save data
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ideal_points.write_parquet(data_dir / f"hierarchical_ideal_points_{ch}.parquet")
    group_params.write_parquet(data_dir / f"group_params_{ch}.parquet")
    icc_df.write_parquet(data_dir / f"variance_decomposition_{ch}.parquet")
    idata.to_netcdf(str(data_dir / f"idata_{ch}.nc"))

    # Plots
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print_header(f"PLOTS — {chamber}")
    plot_party_posteriors(idata, data, chamber, plots_dir)
    plot_icc(icc_df, chamber, plots_dir)
    plot_shrinkage_scatter(ideal_points, chamber, plots_dir)
    plot_forest(ideal_points, chamber, plots_dir)
    plot_dispersion(idata, data, chamber, plots_dir)

    return {
        "data": data,
        "idata": idata,
        "ideal_points": ideal_points,
        "group_params": group_params,
        "icc_df": icc_df,
        "convergence": convergence,
        "sampling_time": sampling_time,
        "flat_corr": flat_corr,
    }


# ── Joint Model Run ─────────────────────────────────────────────────────────


def _run_joint(
    config: ExperimentConfig,
    per_chamber_results: dict,
    legislators: pl.DataFrame,
    rollcalls: pl.DataFrame,
    out_dir: Path,
) -> dict | None:
    """Run the joint cross-chamber model with experiment config."""
    print_header("JOINT CROSS-CHAMBER MODEL")

    house_data = per_chamber_results["House"]["data"]
    senate_data = per_chamber_results["Senate"]["data"]

    joint_idata, combined_data, joint_time = build_joint_model(
        house_data,
        senate_data,
        n_samples=config.n_samples,
        n_tune=config.n_tune,
        n_chains=config.n_chains,
        rollcalls=rollcalls,
        target_accept=config.target_accept,
        beta_prior=config.beta_prior,
    )

    joint_convergence = check_hierarchical_convergence(joint_idata, "Joint")

    # Fix sign indeterminacy
    joint_idata, flipped_chambers = fix_joint_sign_convention(
        joint_idata,
        combined_data,
        per_chamber_results,
    )

    # Extract
    joint_ip = extract_hierarchical_ideal_points(
        joint_idata,
        combined_data,
        legislators,
        flipped_chambers=flipped_chambers,
    )

    # Save
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    joint_ip.write_parquet(data_dir / "hierarchical_ideal_points_joint.parquet")
    joint_idata.to_netcdf(str(data_dir / "idata_joint.nc"))

    # Joint plots
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print_header("JOINT PLOTS")
    plot_joint_party_spread(joint_idata, combined_data, plots_dir)
    plot_forest(joint_ip, "Joint", plots_dir)

    return {
        "idata": joint_idata,
        "combined_data": combined_data,
        "ideal_points": joint_ip,
        "convergence": joint_convergence,
        "sampling_time": joint_time,
    }


# ── Main Runner ──────────────────────────────────────────────────────────────


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed time for report headers."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m {secs}s"


def run_experiment(config: ExperimentConfig, output_base: Path) -> dict:
    """Run a complete hierarchical IRT experiment with platform guardrails.

    This is the primary entry point for experiments. It:
      1. Validates platform constraints (thread caps, compiler, concurrent jobs)
      2. Wraps the run in ExperimentLifecycle (PID lock, process title, cleanup)
      3. Loads data identically to production
      4. Runs per-chamber models with the experimental beta_prior
      5. Optionally runs the joint cross-chamber model
      6. Generates HTML report via production report builder
      7. Saves metrics.json with config, convergence, and timing

    Args:
        config: Experiment configuration (beta_prior, sampling params, etc.).
        output_base: Base directory for experiment output (e.g.
            results/experimental_lab/2026-02-27_positive-beta).

    Returns:
        Metrics dict with config dump, convergence results, and timing.
    """
    from tallgrass.session import KSSession

    out_dir = output_base / config.name

    print(f"Experiment: {config.name}")
    print(f"  {config.description}")
    print(f"  Beta prior: {config.beta_prior.describe()}")
    print(f"  Session: {config.session}")
    print(f"  Output: {out_dir}")
    print()

    # Platform validation
    platform = PlatformCheck.current()
    warnings = platform.validate(config.n_chains)
    if warnings:
        for w in warnings:
            print(f"  PLATFORM WARNING: {w}")
        # Abort on FATAL warnings
        fatal = [w for w in warnings if w.startswith("FATAL")]
        if fatal:
            print("\n  Aborting due to FATAL platform warnings.")
            sys.exit(1)
    else:
        print("  Platform checks: OK")
    print()

    with ExperimentLifecycle(config.name):
        # Load data
        ks = KSSession.from_session_string(config.session)
        eda_dir = ks.results_dir / "01_eda" / "latest"
        pca_dir = ks.results_dir / "02_pca" / "latest"
        irt_dir = ks.results_dir / "05_irt" / "latest"

        house_matrix, senate_matrix, _ = load_eda_matrices(eda_dir)
        house_pca, senate_pca = load_pca_scores(pca_dir)
        rollcalls, legislators = load_metadata(ks.data_dir)

        # Load flat IRT for comparison
        flat_ip: dict[str, pl.DataFrame | None] = {}
        for ch in ("house", "senate"):
            flat_path = irt_dir / "data" / f"ideal_points_{ch}.parquet"
            flat_ip[ch] = pl.read_parquet(flat_path) if flat_path.exists() else None

        t_total = time.time()
        per_chamber_results: dict[str, dict] = {}

        # Per-chamber runs
        pca_map = {"House": house_pca, "Senate": senate_pca}
        matrix_map = {"House": house_matrix, "Senate": senate_matrix}

        for chamber in config.chambers:
            ch = chamber.lower()
            per_chamber_results[chamber] = _run_per_chamber(
                chamber,
                config,
                matrix_map[chamber],
                pca_map[chamber],
                legislators,
                flat_ip[ch],
                out_dir,
            )

        # Joint model (optional)
        joint_results = None
        if config.include_joint:
            try:
                joint_results = _run_joint(
                    config,
                    per_chamber_results,
                    legislators,
                    rollcalls,
                    out_dir,
                )
            except Exception as e:
                print(f"\n  WARNING: Joint model failed: {e}")
                print("  Continuing with per-chamber results only.")

        elapsed = time.time() - t_total

        # HTML Report
        print_header("HTML REPORT")
        report = ReportBuilder(
            title=(
                f"Kansas Legislature {config.session} — Hierarchical Bayesian IRT "
                f"[{config.beta_prior.describe()}]"
            ),
            session=config.session,
            git_hash="experiment",
            elapsed_display=_fmt_elapsed(elapsed),
        )
        plots_dir = out_dir / "plots"
        build_hierarchical_report(
            report,
            chamber_results=per_chamber_results,
            joint_results=joint_results,
            plots_dir=plots_dir,
        )
        report_path = out_dir / "hierarchical_report.html"
        report.write(report_path)
        print(f"  Saved: {report_path}")

        # Build metrics
        print_header("EXPERIMENT SUMMARY")
        metrics: dict = {
            "config": {
                "name": config.name,
                "description": config.description,
                "session": config.session,
                "beta_prior": {
                    "distribution": config.beta_prior.distribution,
                    "params": config.beta_prior.params,
                    "describe": config.beta_prior.describe(),
                },
                "n_samples": config.n_samples,
                "n_tune": config.n_tune,
                "n_chains": config.n_chains,
                "target_accept": config.target_accept,
                "include_joint": config.include_joint,
                "random_seed": RANDOM_SEED,
            },
            "elapsed_s": round(elapsed, 1),
            "chambers": {},
        }

        for chamber in config.chambers:
            if chamber not in per_chamber_results:
                continue
            res = per_chamber_results[chamber]
            ch_metrics = {
                "n_legislators": res["ideal_points"].height,
                "sampling_time_s": round(res["sampling_time"], 1),
                "convergence_ok": res["convergence"]["all_ok"],
                "rhat_xi_max": res["convergence"].get("xi_rhat_max"),
                "rhat_mu_party_max": res["convergence"].get("mu_party_rhat_max"),
                "rhat_sigma_within_max": res["convergence"].get("sigma_within_rhat_max"),
                "ess_xi_min": res["convergence"].get("xi_ess_min"),
                "ess_mu_party_min": res["convergence"].get("mu_party_ess_min"),
                "divergences": res["convergence"].get("divergences", -1),
                "flat_corr": round(res["flat_corr"], 4),
                "icc_mean": round(float(res["icc_df"]["icc_mean"][0]), 4),
            }
            metrics["chambers"][chamber] = ch_metrics
            print(f"\n  {chamber}:")
            print(f"    Convergence: {'PASS' if ch_metrics['convergence_ok'] else 'FAIL'}")
            print(f"    R-hat(xi) max: {ch_metrics['rhat_xi_max']:.4f}")
            print(f"    ESS(xi) min: {ch_metrics['ess_xi_min']:.0f}")
            print(f"    Divergences: {ch_metrics['divergences']}")
            print(f"    Flat IRT r: {ch_metrics['flat_corr']:.4f}")
            print(f"    ICC: {ch_metrics['icc_mean']:.4f}")
            print(f"    Time: {ch_metrics['sampling_time_s']:.0f}s")

        if joint_results is not None:
            jm = {
                "n_legislators": joint_results["ideal_points"].height,
                "sampling_time_s": round(joint_results["sampling_time"], 1),
                "convergence_ok": joint_results["convergence"]["all_ok"],
                "rhat_xi_max": joint_results["convergence"].get("xi_rhat_max"),
                "ess_xi_min": joint_results["convergence"].get("xi_ess_min"),
                "divergences": joint_results["convergence"].get("divergences", -1),
                "n_shared_bills": joint_results["combined_data"]["n_shared_bills"],
            }
            metrics["joint"] = jm
            print("\n  Joint:")
            print(f"    Convergence: {'PASS' if jm['convergence_ok'] else 'FAIL'}")
            print(f"    R-hat(xi) max: {jm['rhat_xi_max']:.4f}")
            print(f"    ESS(xi) min: {jm['ess_xi_min']:.0f}")
            print(f"    Divergences: {jm['divergences']}")
            print(f"    Time: {jm['sampling_time_s']:.0f}s")

        # Save metrics
        metrics_path = out_dir / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"\n{'=' * 80}")
        print(f"  DONE — {config.name} in {_fmt_elapsed(elapsed)}")
        print(f"  Metrics: {metrics_path}")
        print(f"  Report: {report_path}")
        print(f"{'=' * 80}")

    return metrics
