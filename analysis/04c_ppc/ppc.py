"""Kansas Legislature — Posterior Predictive Checks + LOO-CV Model Comparison (Phase 4c)

Standalone validation phase that loads InferenceData from upstream IRT phases
(flat 1D, 2D experimental, hierarchical) and produces a unified PPC battery
plus LOO-CV model comparison.

Usage:
  uv run python analysis/04c_ppc/ppc.py [--session 2025-26] [--run-id ...]
      [--skip-loo] [--n-reps 500] [--skip-q3]

Outputs (in results/<session>/<run_id>/04c_ppc/):
  - data/:   JSON summaries, Pareto k parquet, Q3 matrices
  - plots/:  PNG visualizations (calibration, item/person fit, LOO, Pareto k)
  - ppc_report.html
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir

try:
    from analysis.phase_utils import print_header, save_fig
except ModuleNotFoundError:
    from phase_utils import print_header, save_fig

try:
    from analysis.ppc_data import (
        add_log_likelihood_to_idata,
        compare_models,
        compute_item_ppc,
        compute_log_likelihood_1d,
        compute_log_likelihood_2d,
        compute_log_likelihood_hierarchical,
        compute_loo,
        compute_person_ppc,
        compute_vote_margin_ppc,
        compute_yens_q3,
        run_ppc_battery,
        summarize_pareto_k,
    )
except ModuleNotFoundError:
    from ppc_data import (  # type: ignore[no-redef]
        add_log_likelihood_to_idata,
        compare_models,
        compute_item_ppc,
        compute_log_likelihood_1d,
        compute_log_likelihood_2d,
        compute_log_likelihood_hierarchical,
        compute_loo,
        compute_person_ppc,
        compute_vote_margin_ppc,
        compute_yens_q3,
        run_ppc_battery,
        summarize_pareto_k,
    )

try:
    from analysis.ppc_report import build_ppc_report
except ModuleNotFoundError:
    from ppc_report import build_ppc_report  # type: ignore[no-redef]

try:
    from analysis.irt import prepare_irt_data
except ModuleNotFoundError:
    from irt import prepare_irt_data  # type: ignore[no-redef]


# ── Primer ──────────────────────────────────────────────────────────────────

PPC_PRIMER = """\
# Posterior Predictive Checks + LOO-CV Model Comparison

## Purpose

This phase answers two questions: (1) Does each IRT model reproduce the data
it was fit to? (2) Which model fits best after accounting for complexity?

Posterior predictive checks (PPCs) simulate replicated datasets from the fitted
model and compare summary statistics to the observed data. If the model is
well-calibrated, observed statistics should fall within the replicated
distribution.

LOO-CV (leave-one-out cross-validation via PSIS) estimates each model's
out-of-sample predictive accuracy without refitting, enabling principled
model comparison.

## Method

### PPC Battery
For each model, we draw 500 posterior samples and simulate replicated vote
outcomes. Summary statistics compared:
- **Yea rate**: Overall fraction of Yea votes (calibration check)
- **Classification accuracy**: Fraction of correctly predicted votes
- **GMP** (Geometric Mean Probability): exp(mean(log(p_correct))) — penalizes
  confident wrong predictions more than accuracy
- **APRE** (Aggregate Proportional Reduction in Error): improvement over
  modal-category baseline (controls for 82% Yea base rate)

### Item and Person Fit
- **Item endorsement rates**: Per-roll-call observed vs replicated Yea fraction
- **Person total scores**: Per-legislator observed vs replicated Yea count
- Items/persons flagged when observed value falls outside 95% replicated interval

### Yen's Q3 Local Dependence
Residual correlations between items after conditioning on ability. |Q3| > 0.2
indicates local dependence (unmodeled structure). Key for assessing whether a
second dimension is needed.

### LOO-CV Model Comparison
PSIS-LOO (Vehtari et al. 2017) estimates out-of-sample ELPD. ArviZ `compare()`
ranks models by ELPD with stacking weights. Pareto k diagnostics flag
observations where importance sampling is unreliable.

## Inputs
- EDA vote matrices: `results/<session>/<run_id>/01_eda/data/vote_matrix_{chamber}_filtered.parquet`
- Flat IRT: `results/<session>/<run_id>/04_irt/data/idata_{chamber}.nc`
- 2D IRT: `results/<session>/<run_id>/04b_irt_2d/data/idata_{chamber}.nc`
- Hierarchical IRT: `results/<session>/<run_id>/10_hierarchical/data/idata_{chamber}.nc`

## Outputs

### `data/` — JSON + Parquet intermediates
| File | Description |
|------|-------------|
| `ppc_summary_{chamber}.json` | PPC battery results per model |
| `item_fit_{chamber}.json` | Item-level misfit counts per model |
| `person_fit_{chamber}.json` | Person-level misfit counts per model |
| `q3_summary_{chamber}.json` | Q3 local dependence statistics |
| `loo_comparison_{chamber}.json` | LOO-CV comparison table |
| `pareto_k_{model}_{chamber}.parquet` | Pointwise Pareto k values |

### `plots/` — PNG visualizations
| File | Description |
|------|-------------|
| `calibration_{chamber}.png` | Yea rate: observed vs replicated per model |
| `item_fit_{chamber}.png` | Item endorsement: observed vs predicted |
| `person_fit_{chamber}.png` | Person totals: observed vs predicted |
| `margins_{chamber}.png` | Vote margin distributions |
| `pareto_k_{chamber}.png` | Pareto k diagnostic scatter |
| `loo_comparison_{chamber}.png` | ELPD forest plot |

## Interpretation Guide
- **Bayesian p-value in [0.1, 0.9]**: Model is well-calibrated for that statistic
- **GMP > 0.7**: Good probabilistic predictions
- **APRE > 0.3**: Meaningful improvement over baseline
- **< 5% misfitting items/persons**: Acceptable fit
- **|Q3| > 0.2 violations < 5%**: No major local dependence
- **LOO ELPD**: Higher = better. SE-based comparison; delta > 2*SE is meaningful

## Caveats
- LOO-CV requires log-likelihood computation (~30s per model per chamber)
- Q3 is O(n_votes^2) — can be slow for large vote matrices
- 2D model may not be available (experimental phase)
- Hierarchical model excludes Independents (different observation set)
"""

# ── Constants ───────────────────────────────────────────────────────────────

PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}
MODEL_COLORS = {"Flat 1D": "#1f77b4", "2D IRT": "#ff7f0e", "Hierarchical": "#2ca02c"}
DEFAULT_N_REPS = 500
DEFAULT_Q3_DRAWS = 100
CHAMBERS = ["House", "Senate"]


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KS Legislature PPC + LOO-CV Model Comparison (Phase 4c)"
    )
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override flat IRT results directory")
    parser.add_argument("--irt-2d-dir", default=None, help="Override 2D IRT results directory")
    parser.add_argument(
        "--hierarchical-dir", default=None, help="Override hierarchical IRT results directory"
    )
    parser.add_argument("--skip-loo", action="store_true", help="Skip LOO-CV computation")
    parser.add_argument("--skip-q3", action="store_true", help="Skip Q3 local dependence")
    parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS, help="PPC replications")
    return parser.parse_args()


# ── Helpers ─────────────────────────────────────────────────────────────────


def _load_idata_safe(nc_path: Path, model_name: str) -> az.InferenceData | None:
    """Load InferenceData from NetCDF, returning None with warning if missing."""
    if not nc_path.exists():
        print(f"  WARNING: {model_name} NetCDF not found: {nc_path}")
        return None
    try:
        return az.from_netcdf(str(nc_path))
    except Exception as e:
        print(f"  WARNING: Failed to load {model_name}: {e}")
        return None


# ── Plots ───────────────────────────────────────────────────────────────────


def plot_calibration(
    ppc_results: dict[str, dict],
    chamber: str,
    plots_dir: Path,
) -> None:
    """Yea rate calibration: observed line + replicated histogram per model."""
    n_models = len(ppc_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)

    for ax, (model_name, ppc) in zip(axes[0], ppc_results.items()):
        color = MODEL_COLORS.get(model_name, "#333333")
        ax.hist(ppc["replicated_yea_rates"], bins=30, alpha=0.7, color=color, edgecolor="white")
        ax.axvline(ppc["observed_yea_rate"], color="red", linewidth=2, label="Observed")
        ax.set_title(f"{model_name}\np = {ppc['bayesian_p_yea_rate']:.3f}")
        ax.set_xlabel("Yea Rate")
        ax.set_ylabel("Replications")
        ax.legend(fontsize=8)

    fig.suptitle(f"{chamber} — Yea Rate Calibration", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, plots_dir / f"calibration_{chamber.lower()}.png")


def plot_item_fit(
    item_results: dict[str, dict],
    chamber: str,
    plots_dir: Path,
) -> None:
    """Item endorsement: observed vs predicted scatter per model."""
    n_models = len(item_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)

    for ax, (model_name, item) in zip(axes[0], item_results.items()):
        obs = item["observed_rates"]
        pred_mean = item["replicated_rates"].mean(axis=0)
        color = MODEL_COLORS.get(model_name, "#333333")

        ax.scatter(obs, pred_mean, alpha=0.4, s=10, color=color)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)

        # Highlight misfitting
        misfit_idx = item["misfitting_items"]
        if len(misfit_idx) > 0:
            ax.scatter(
                obs[misfit_idx],
                pred_mean[misfit_idx],
                color="red",
                s=20,
                marker="x",
                label=f"{len(misfit_idx)} misfitting",
            )
            ax.legend(fontsize=8)

        ax.set_title(f"{model_name}")
        ax.set_xlabel("Observed Rate")
        ax.set_ylabel("Predicted Rate")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(f"{chamber} — Item Endorsement Rates", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, plots_dir / f"item_fit_{chamber.lower()}.png")


def plot_person_fit(
    person_results: dict[str, dict],
    chamber: str,
    plots_dir: Path,
) -> None:
    """Person total scores: observed vs predicted scatter per model."""
    n_models = len(person_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)

    for ax, (model_name, person) in zip(axes[0], person_results.items()):
        obs = person["observed_totals"]
        pred_mean = person["replicated_totals"].mean(axis=0)
        color = MODEL_COLORS.get(model_name, "#333333")

        ax.scatter(obs, pred_mean, alpha=0.4, s=15, color=color)
        lim = max(obs.max(), pred_mean.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1)

        misfit_idx = person["misfitting_persons"]
        if len(misfit_idx) > 0:
            ax.scatter(
                obs[misfit_idx],
                pred_mean[misfit_idx],
                color="red",
                s=25,
                marker="x",
                label=f"{len(misfit_idx)} misfitting",
            )
            ax.legend(fontsize=8)

        ax.set_title(f"{model_name}")
        ax.set_xlabel("Observed Yea Count")
        ax.set_ylabel("Predicted Yea Count")

    fig.suptitle(f"{chamber} — Person Total Scores", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, plots_dir / f"person_fit_{chamber.lower()}.png")


def plot_vote_margins(
    margin_results: dict[str, dict],
    chamber: str,
    plots_dir: Path,
) -> None:
    """Vote margin distributions: observed vs replicated envelope."""
    n_models = len(margin_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)

    for ax, (model_name, margins) in zip(axes[0], margin_results.items()):
        obs = margins["observed_margins"]
        rep = margins["replicated_margins"]
        color = MODEL_COLORS.get(model_name, "#333333")

        ax.hist(obs, bins=25, alpha=0.6, color="gray", edgecolor="white", label="Observed")
        rep_mean = rep.mean(axis=0)
        ax.hist(
            rep_mean,
            bins=25,
            alpha=0.4,
            color=color,
            edgecolor="white",
            label="Replicated mean",
        )
        ax.set_title(model_name)
        ax.set_xlabel("Vote Margin")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle(f"{chamber} — Vote Margin Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, plots_dir / f"margins_{chamber.lower()}.png")


def plot_pareto_k(
    pareto_results: dict[str, dict],
    chamber: str,
    plots_dir: Path,
) -> None:
    """Pareto k diagnostic scatter per model."""
    n_models = len(pareto_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)

    thresholds = [0.5, 0.7, 1.0]
    threshold_colors = ["#4CAF50", "#FFC107", "#FF5722", "#B71C1C"]

    for ax, (model_name, k_vals) in zip(axes[0], pareto_results.items()):
        n = len(k_vals)
        x = np.arange(n)

        # Color by category
        colors = np.where(
            k_vals < 0.5,
            threshold_colors[0],
            np.where(
                k_vals < 0.7,
                threshold_colors[1],
                np.where(k_vals < 1.0, threshold_colors[2], threshold_colors[3]),
            ),
        )
        ax.scatter(x, k_vals, c=colors, s=3, alpha=0.5)

        for t, c in zip(thresholds, threshold_colors[1:]):
            ax.axhline(t, color=c, linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_title(model_name)
        ax.set_xlabel("Observation Index")
        ax.set_ylabel("Pareto k")
        ax.set_ylim(min(-0.1, k_vals.min() - 0.1), max(1.5, k_vals.max() + 0.1))

    fig.suptitle(f"{chamber} — Pareto k Diagnostics", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, plots_dir / f"pareto_k_{chamber.lower()}.png")


def plot_loo_comparison(
    comparison_df: object,
    chamber: str,
    plots_dir: Path,
) -> None:
    """ELPD forest plot from az.compare() output."""
    fig, ax = plt.subplots(figsize=(8, 4))
    az.plot_compare(comparison_df, ax=ax)
    ax.set_title(f"{chamber} — LOO-CV Model Comparison")
    fig.tight_layout()
    save_fig(fig, plots_dir / f"loo_comparison_{chamber.lower()}.png")


# ── Per-Chamber Processing ──────────────────────────────────────────────────


def process_chamber(
    chamber: str,
    matrix: pl.DataFrame,
    available_models: dict[str, tuple[az.InferenceData, str]],
    *,
    n_reps: int,
    skip_loo: bool,
    skip_q3: bool,
    ctx: RunContext,
) -> dict | None:
    """Run PPC battery for all available models in one chamber.

    available_models: {model_name: (idata, model_type)}
    Returns results dict or None if no models available.
    """
    ch = chamber.lower()

    if not available_models:
        print(f"\n  {chamber}: No models available — skipping")
        return None

    print_header(f"PPC — {chamber}")
    print(f"  Models available: {', '.join(available_models.keys())}")

    chamber_results: dict[str, dict] = {}
    ppc_results: dict[str, dict] = {}
    item_results: dict[str, dict] = {}
    person_results: dict[str, dict] = {}
    margin_results: dict[str, dict] = {}
    q3_results: dict[str, dict] = {}
    loo_models: dict[str, az.InferenceData] = {}
    pareto_results: dict[str, np.ndarray] = {}

    for model_name, (idata, model_type) in available_models.items():
        print(f"\n  ── {model_name} ──")

        # Prepare data dict for this model
        # For hierarchical: idata may have different legislators (no Independents)
        if model_type == "hierarchical":
            # Filter matrix to match hierarchical model's legislator set
            if "legislator" in idata.posterior.coords:
                hier_slugs = list(idata.posterior.coords["legislator"].values)
                matrix_filtered = matrix.filter(pl.col("legislator_slug").is_in(hier_slugs))
            else:
                matrix_filtered = matrix
            data = prepare_irt_data(matrix_filtered, chamber)
        else:
            data = prepare_irt_data(matrix, chamber)

        # PPC battery
        print(f"    Running PPC battery ({n_reps} replications)...")
        ppc = run_ppc_battery(idata, data, n_reps=n_reps, model_type=model_type)
        ppc_results[model_name] = ppc

        print(f"    Observed Yea rate: {ppc['observed_yea_rate']:.3f}")
        print(
            f"    Replicated Yea rate: {ppc['replicated_yea_rate_mean']:.3f} "
            f"+/- {ppc['replicated_yea_rate_sd']:.3f}"
        )
        print(f"    Bayesian p-value: {ppc['bayesian_p_yea_rate']:.3f}")
        print(f"    Accuracy: {ppc['mean_accuracy']:.3f}")
        print(f"    GMP: {ppc['mean_gmp']:.3f}")
        print(f"    APRE: {ppc['apre']:.3f}")

        # Item-level PPC
        print("    Running item-level checks...")
        item = compute_item_ppc(idata, data, n_reps=n_reps, model_type=model_type)
        item_results[model_name] = item
        print(
            f"    Misfitting items: {item['n_misfitting']}/{item['n_votes']} "
            f"({100 * item['n_misfitting'] / max(item['n_votes'], 1):.1f}%)"
        )

        # Person-level PPC
        print("    Running person-level checks...")
        person = compute_person_ppc(idata, data, n_reps=n_reps, model_type=model_type)
        person_results[model_name] = person
        print(
            f"    Misfitting persons: {person['n_misfitting']}/{person['n_legislators']} "
            f"({100 * person['n_misfitting'] / max(person['n_legislators'], 1):.1f}%)"
        )

        # Vote margins
        print("    Running vote margin checks...")
        margins = compute_vote_margin_ppc(idata, data, n_reps=n_reps, model_type=model_type)
        margin_results[model_name] = margins

        # Q3 (skip for 2D — it's the reference for dimensionality)
        if not skip_q3:
            print(f"    Computing Yen's Q3 ({DEFAULT_Q3_DRAWS} draws)...")
            q3 = compute_yens_q3(
                idata, data, n_draws_sample=DEFAULT_Q3_DRAWS, model_type=model_type
            )
            q3_results[model_name] = q3
            print(
                f"    Q3 violations (|Q3|>0.2): {q3['n_violations']}/{q3['n_pairs']} "
                f"({100 * q3['violation_rate']:.1f}%)"
            )
            print(f"    Max |Q3|: {q3['max_abs_q3']:.3f}, Mean |Q3|: {q3['mean_abs_q3']:.3f}")

        # Log-likelihood for LOO
        if not skip_loo:
            print("    Computing log-likelihood...")
            if model_type == "2d":
                log_lik = compute_log_likelihood_2d(idata, data)
            elif model_type == "hierarchical":
                log_lik = compute_log_likelihood_hierarchical(idata, data)
            else:
                log_lik = compute_log_likelihood_1d(idata, data)

            idata_with_ll = add_log_likelihood_to_idata(idata, log_lik)
            loo_models[model_name] = idata_with_ll

        # Accumulate per-model results
        chamber_results[model_name] = {
            "model_type": model_type,
            "ppc": {
                k: v
                for k, v in ppc.items()
                if k not in ("replicated_yea_rates", "replicated_accuracies")
            },
            "item_fit": {
                "n_misfitting": item["n_misfitting"],
                "n_votes": item["n_votes"],
                "misfit_pct": 100 * item["n_misfitting"] / max(item["n_votes"], 1),
            },
            "person_fit": {
                "n_misfitting": person["n_misfitting"],
                "n_legislators": person["n_legislators"],
                "misfit_pct": 100 * person["n_misfitting"] / max(person["n_legislators"], 1),
            },
        }
        if model_name in q3_results:
            chamber_results[model_name]["q3"] = {
                k: v for k, v in q3_results[model_name].items() if k != "q3_matrix"
            }

    # ── LOO-CV Comparison ──
    loo_comparison = None
    loo_data = {}
    if not skip_loo and len(loo_models) >= 1:
        print_header(f"LOO-CV — {chamber}")

        if len(loo_models) >= 2:
            print(f"  Comparing {len(loo_models)} models...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loo_comparison, loo_individual = compare_models(loo_models)
            print(f"\n{loo_comparison}")
        else:
            # Single model — just compute LOO
            loo_individual = {}
            for name, idata_ll in loo_models.items():
                print(f"  Computing LOO for {name}...")
                loo_individual[name] = compute_loo(idata_ll)

        for name, loo_result in loo_individual.items():
            pareto_summary = summarize_pareto_k(loo_result)
            pareto_results[name] = loo_result.pareto_k.values
            chamber_results[name]["loo"] = {
                "elpd_loo": float(loo_result.elpd_loo),
                "se": float(loo_result.se),
                "p_loo": float(loo_result.p_loo),
                "pareto_k": pareto_summary,
            }
            print(
                f"  {name}: ELPD = {loo_result.elpd_loo:.1f} (SE = {loo_result.se:.1f}), "
                f"p_loo = {loo_result.p_loo:.1f}"
            )
            print(
                f"    Pareto k: {pareto_summary['good']} good, {pareto_summary['ok']} ok, "
                f"{pareto_summary['bad']} bad, {pareto_summary['very_bad']} very bad"
            )
            loo_data[name] = {
                "elpd_loo": float(loo_result.elpd_loo),
                "se": float(loo_result.se),
                "p_loo": float(loo_result.p_loo),
                "pareto_k": pareto_summary,
            }

    # ── Save Data ──
    print_header(f"SAVING — {chamber}")

    # PPC summaries
    ppc_summary = {k: v for k, v in chamber_results.items()}
    with open(ctx.data_dir / f"ppc_summary_{ch}.json", "w") as f:
        json.dump(ppc_summary, f, indent=2, default=str)
    print(f"  Saved: ppc_summary_{ch}.json")

    # LOO comparison
    if loo_data:
        with open(ctx.data_dir / f"loo_comparison_{ch}.json", "w") as f:
            json.dump(loo_data, f, indent=2, default=str)
        print(f"  Saved: loo_comparison_{ch}.json")

    # Pareto k parquets
    for name, k_vals in pareto_results.items():
        model_key = name.lower().replace(" ", "_")
        pk_df = pl.DataFrame({"obs_idx": np.arange(len(k_vals)), "pareto_k": k_vals})
        pk_df.write_parquet(ctx.data_dir / f"pareto_k_{model_key}_{ch}.parquet")
        print(f"  Saved: pareto_k_{model_key}_{ch}.parquet")

    # Q3 matrices as parquet
    for name, q3 in q3_results.items():
        model_key = name.lower().replace(" ", "_")
        q3_df = pl.DataFrame(
            {f"item_{j}": q3["q3_matrix"][:, j] for j in range(q3["q3_matrix"].shape[1])}
        )
        q3_df.write_parquet(ctx.data_dir / f"q3_matrix_{model_key}_{ch}.parquet")
        print(f"  Saved: q3_matrix_{model_key}_{ch}.parquet")

    # ── Plots ──
    if ppc_results:
        plot_calibration(ppc_results, chamber, ctx.plots_dir)
    if item_results:
        plot_item_fit(item_results, chamber, ctx.plots_dir)
    if person_results:
        plot_person_fit(person_results, chamber, ctx.plots_dir)
    if margin_results:
        plot_vote_margins(margin_results, chamber, ctx.plots_dir)
    if pareto_results:
        plot_pareto_k(pareto_results, chamber, ctx.plots_dir)
    if loo_comparison is not None:
        plot_loo_comparison(loo_comparison, chamber, ctx.plots_dir)

    return {
        "chamber": chamber,
        "models": chamber_results,
        "loo_comparison": loo_comparison,
        "ppc_results": ppc_results,
        "item_results": item_results,
        "person_results": person_results,
        "margin_results": margin_results,
        "q3_results": q3_results,
        "pareto_results": pareto_results,
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    with RunContext(
        session=args.session,
        analysis_name="04c_ppc",
        params=vars(args),
        primer=PPC_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"Posterior Predictive Checks — Session {args.session}")

        # ── Resolve upstream directories ──
        from tallgrass.session import KSSession

        ks = KSSession.from_session_string(args.session)
        results_root = ks.results_dir

        eda_dir = resolve_upstream_dir(
            "01_eda",
            results_root,
            args.run_id,
            override=Path(args.eda_dir) if args.eda_dir else None,
        )
        irt_dir = resolve_upstream_dir(
            "04_irt",
            results_root,
            args.run_id,
            override=Path(args.irt_dir) if args.irt_dir else None,
        )
        irt_2d_dir = resolve_upstream_dir(
            "04b_irt_2d",
            results_root,
            args.run_id,
            override=Path(args.irt_2d_dir) if args.irt_2d_dir else None,
        )
        hier_dir = resolve_upstream_dir(
            "10_hierarchical",
            results_root,
            args.run_id,
            override=Path(args.hierarchical_dir) if args.hierarchical_dir else None,
        )

        print(f"  EDA dir: {eda_dir}")
        print(f"  IRT dir: {irt_dir}")
        print(f"  IRT 2D dir: {irt_2d_dir}")
        print(f"  Hierarchical dir: {hier_dir}")

        # ── Load EDA vote matrices ──
        print_header("LOADING DATA")
        house_path = eda_dir / "data" / "vote_matrix_house_filtered.parquet"
        senate_path = eda_dir / "data" / "vote_matrix_senate_filtered.parquet"
        if not house_path.exists() and not senate_path.exists():
            print("Phase 04c (PPC): skipping — no EDA vote matrices available")
            return
        house_matrix = pl.read_parquet(house_path) if house_path.exists() else None
        senate_matrix = pl.read_parquet(senate_path) if senate_path.exists() else None
        if house_matrix is not None:
            print(
                f"  House matrix: {house_matrix.height} legislators "
                f"x {house_matrix.width - 1} votes"
            )
        if senate_matrix is not None:
            print(
                f"  Senate matrix: {senate_matrix.height} legislators "
                f"x {senate_matrix.width - 1} votes"
            )

        # ── Process each chamber ──
        all_results: dict[str, dict] = {}

        for chamber, matrix in [("House", house_matrix), ("Senate", senate_matrix)]:
            if matrix is None:
                print(f"\n  {chamber}: EDA matrix not available — skipping")
                continue
            ch = chamber.lower()

            # Discover available models
            available: dict[str, tuple[az.InferenceData, str]] = {}

            # Flat 1D IRT
            idata_1d = _load_idata_safe(irt_dir / "data" / f"idata_{ch}.nc", "Flat 1D")
            if idata_1d is not None:
                available["Flat 1D"] = (idata_1d, "1d")

            # 2D IRT (experimental)
            idata_2d = _load_idata_safe(irt_2d_dir / "data" / f"idata_{ch}.nc", "2D IRT")
            if idata_2d is not None:
                available["2D IRT"] = (idata_2d, "2d")

            # Hierarchical IRT
            idata_hier = _load_idata_safe(hier_dir / "data" / f"idata_{ch}.nc", "Hierarchical")
            if idata_hier is not None:
                available["Hierarchical"] = (idata_hier, "hierarchical")

            result = process_chamber(
                chamber,
                matrix,
                available,
                n_reps=args.n_reps,
                skip_loo=args.skip_loo,
                skip_q3=args.skip_q3,
                ctx=ctx,
            )
            if result is not None:
                all_results[chamber] = result

        if not all_results:
            print("\n  No models found for any chamber — exiting")
            return

        # ── HTML Report ──
        print_header("BUILDING REPORT")
        build_ppc_report(
            ctx.report,
            all_results=all_results,
            session=args.session,
            skip_loo=args.skip_loo,
            skip_q3=args.skip_q3,
            plots_dir=ctx.plots_dir,
        )

        print(f"\n  Report: {ctx.run_dir / 'ppc_report.html'}")


if __name__ == "__main__":
    main()
