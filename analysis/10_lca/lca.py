"""
Kansas Legislature — Latent Class Analysis (Phase 10)

Fits Bernoulli mixture models (Latent Class Analysis) directly on the binary
vote matrix to test for discrete factions. BIC selects the optimal number of
latent classes. If K=2, this provides model-based confirmation that discrete
factions don't exist beyond the party split. If K>2, the Salsa effect test
checks whether extra classes represent qualitative distinctions or just
quantitative grading along the same ideological continuum.

Extends Phase 5 (Clustering) with the statistically principled model for
binary data — LCA's Bernoulli likelihood is the correct generative model
for Yea/Nay votes, unlike GMM which assumes continuous Gaussian data.

Library: StepMix (Python, MIT, JSS 2025). scikit-learn API, native
binary/NaN support, BIC/AIC built in.

Usage:
  uv run python analysis/10_lca/lca.py [--session 2025-26] [--k-max 8]
      [--skip-within-party] [--run-id RUN_ID]

Outputs (in results/<session>/<run_id>/10_lca/):
  - data/:   Parquet files (class assignments, enumeration results, profiles)
  - plots/:  PNG visualizations (BIC elbow, profile heatmap, membership, IRT boxplot)
  - filtering_manifest.json, run_info.json, run_log.txt
  - lca_report.html
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
from stepmix import StepMix

# ── StepMix / scikit-learn compatibility shim ────────────────────────────────
# StepMix 2.2.1 uses sklearn's private `_validate_data` method and the deprecated
# `force_all_finite` kwarg. Both were removed in scikit-learn 1.6+/1.8.
# This shim restores compatibility until StepMix ships a fixed release.
if not hasattr(StepMix, "_validate_data"):
    from sklearn.utils.validation import validate_data as _sklearn_validate_data

    def _validate_data_compat(self, X, **kwargs):  # type: ignore[no-untyped-def]
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _sklearn_validate_data(self, X, **kwargs)

    StepMix._validate_data = _validate_data_compat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir

try:
    from analysis.lca_report import build_lca_report
except ModuleNotFoundError:
    from lca_report import build_lca_report  # type: ignore[no-redef]

try:
    from analysis.phase_utils import load_legislators, print_header, save_fig
except ImportError:
    from phase_utils import load_legislators, print_header, save_fig

# ── Constants ────────────────────────────────────────────────────────────────

K_MAX = 8
"""Maximum number of latent classes to test."""

N_INIT = 50
"""Random starts per K (Nylund-Gibson recommendation for EM stability)."""

MAX_ITER = 1000
"""Maximum EM iterations per fit."""

RANDOM_SEED = 42
"""Reproducibility seed for all stochastic operations."""

MIN_VOTES = 20
"""Minimum votes per legislator (same as EDA/IRT)."""

MINORITY_THRESHOLD = 0.025
"""Filter near-unanimous votes (minority < 2.5%, same as EDA)."""

MIN_CLASS_FRACTION = 0.05
"""Flag classes smaller than 5% of sample as potentially spurious."""

SALSA_THRESHOLD = 0.80
"""Spearman r above this between class profiles = Salsa effect (quantitative grading)."""

TOP_DISCRIMINATING_BILLS = 30
"""Number of most-discriminating bills to show in profile heatmap."""

PARTY_COLORS = {
    "Republican": "#E81B23",
    "Democrat": "#0015BC",
    "Independent": "#999999",
}

CLASS_CMAP = "Set2"
"""Colormap for latent class assignments."""

# ── Primer ───────────────────────────────────────────────────────────────────

LCA_PRIMER = """\
# Latent Class Analysis (Phase 10)

## Purpose

Tests whether Kansas legislators form **discrete factions** (latent classes)
based on their voting patterns. Phase 5 (Clustering) found k=2 optimal using
distance-based, centroid-based, density-based, and Gaussian model-based methods.
LCA provides the statistically principled confirmation: a Bernoulli mixture
model operating directly on the binary vote matrix with BIC-selected K.

If BIC selects K=2, we have model-based evidence that discrete factions don't
exist beyond the two-party split. If K>2, the Salsa effect test checks whether
additional classes represent qualitatively distinct voting patterns or merely
quantitative grading along the same ideological continuum.

## Method

### Latent Class Analysis (Bernoulli Mixture)

LCA models each legislator as belonging to one of K unobserved classes.
Within each class, the probability of voting Yea on each bill is an
independent Bernoulli draw (local independence assumption). The model is:

    P(vote_ij = 1 | class k) = π_jk

where π_jk is the class-specific item response probability for bill j.

Parameters are estimated via Expectation-Maximization (EM) with 50 random
starts per K to avoid local optima.

### Model Selection

- **Primary:** BIC (Bayesian Information Criterion) — penalizes complexity
- **Secondary:** AIC (less conservative, for reference)
- **Descriptive:** Entropy (classification certainty, >0.8 = good separation)
- **Strategy:** Fit K=1 to K=8; select K at BIC minimum

### Salsa Effect Detection

When K>2, pairwise Spearman correlations between class P(Yea) profiles
detect whether classes represent qualitative distinctions (distinct voting
on different issues) or just quantitative grading (same pattern at different
intensities). Correlation > 0.80 = "Salsa effect" (mild/medium/hot, not
different flavors).

### Cross-Validation

- **IRT comparison:** Mean/median IRT ideal point per class; monotonicity check
- **Phase 5 agreement:** Adjusted Rand Index vs hierarchical, k-means, GMM, spectral

### Within-Party Analysis

Separate LCA on Republican-only and Democrat-only subsets to test for
intra-party factions (e.g., moderates vs. conservatives within the R caucus).

## Inputs

Reads from `results/<session>/<run_id>/01_eda/data/`:
- `vote_matrix_house_filtered.parquet` — Filtered binary vote matrices
- `vote_matrix_senate_filtered.parquet`

Reads from `results/<session>/<run_id>/05_irt/data/`:
- `ideal_points_house.parquet` — IRT ideal points for cross-validation
- `ideal_points_senate.parquet`

Reads from `results/<session>/<run_id>/09_clustering/data/` (optional):
- `hierarchical_assignments_house.parquet` — For ARI comparison
- `hierarchical_assignments_senate.parquet`
- `kmeans_assignments_house.parquet`
- `kmeans_assignments_senate.parquet`

Reads from `data/kansas/{legislature}_{start}-{end}/`:
- `{name}_legislators.csv` — Legislator metadata (party, name)

## Outputs

All outputs land in `results/<session>/<run_id>/10_lca/`:

### `data/` — Parquet intermediates

| File | Description |
|------|-------------|
| `enumeration_{chamber}.parquet` | BIC/AIC/entropy for K=1..K_max |
| `class_assignments_{chamber}.parquet` | Legislator class labels + probabilities |
| `class_profiles_{chamber}.parquet` | P(Yea | class) for each bill |
| `within_party_{party}_{chamber}.parquet` | Intra-party class assignments (if K>1) |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `bic_elbow_{chamber}.png` | BIC/AIC by K (model selection) |
| `profile_heatmap_{chamber}.png` | P(Yea | class) for top discriminating bills |
| `membership_hist_{chamber}.png` | Max membership probability distribution |
| `irt_boxplot_{chamber}.png` | IRT ideal points by LCA class |
| `salsa_matrix_{chamber}.png` | Profile correlation matrix (if K>2) |

## Interpretation Guide

- **K=2 with BIC minimum:** Confirms party split is the only discrete structure.
  Within-party variation is continuous (consistent with Phase 5 finding).
- **K>2 with Salsa effect:** Extra classes are quantitative (mild/medium/hot
  Republicans) — not qualitatively distinct factions.
- **K>2 without Salsa effect:** Genuine multi-dimensional structure — some bills
  create cross-cutting coalitions that IRT's single dimension misses.
- **Entropy > 0.8:** Clean class separation; class assignments are confident.
- **Entropy < 0.6:** Fuzzy boundaries; classes overlap substantially.

## Caveats

1. **Local independence assumption:** Violated if bill outcomes are correlated
   within a class beyond what class membership explains. Not testable with
   BIC alone (would need bivariate residuals or Q3-style checks).
2. **Lubke & Neale (2006) impossibility:** Cannot empirically distinguish
   categorical from continuous latent structure — both models fit equally well.
   LCA finding K=2 does NOT prove classes exist; it means the data is consistent
   with a 2-class model (which a 1D continuum also produces).
3. **StepMix FIML:** Missing votes handled via Full Information Maximum
   Likelihood — no imputation needed, but high missingness (>30%) can
   destabilize EM convergence.
4. **N_INIT=50:** Should be sufficient for problems with <200 legislators and
   <500 votes, but pathological multimodality is always possible.
"""

# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Latent Class Analysis (Phase 10)")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override IRT results directory")
    parser.add_argument(
        "--clustering-dir", default=None, help="Override clustering results directory"
    )
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument("--k-max", type=int, default=K_MAX, help=f"Max classes (default {K_MAX})")
    parser.add_argument(
        "--skip-within-party",
        action="store_true",
        help="Skip within-party LCA (faster)",
    )
    return parser.parse_args()


# ── Utilities ────────────────────────────────────────────────────────────────


def save_filtering_manifest(manifest: dict, out_dir: Path) -> None:
    path = out_dir / "filtering_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Saved: {path.name}")


# ── Phase 1: Load Data ──────────────────────────────────────────────────────


def load_vote_matrices(eda_dir: Path) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load filtered binary vote matrices from EDA. Returns None if unavailable."""
    results: list[pl.DataFrame | None] = []
    for ch in ("house", "senate"):
        path = eda_dir / "data" / f"vote_matrix_{ch}_filtered.parquet"
        results.append(pl.read_parquet(path) if path.exists() else None)
    return results[0], results[1]


def load_irt_ideal_points(
    irt_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load IRT ideal points for both chambers. Returns None if unavailable."""
    results: list[pl.DataFrame | None] = []
    for ch in ("house", "senate"):
        path = irt_dir / "data" / f"ideal_points_{ch}.parquet"
        results.append(pl.read_parquet(path) if path.exists() else None)
    return results[0], results[1]


def load_clustering_labels(clustering_dir: Path, chamber: str) -> dict[str, np.ndarray] | None:
    """Load Phase 5 cluster labels for ARI comparison.

    Returns dict mapping method name to label array, or None if unavailable.
    """
    ch = chamber.lower()
    labels: dict[str, np.ndarray] = {}

    for method, filename in [
        ("hierarchical", f"hierarchical_assignments_{ch}.parquet"),
        ("kmeans", f"kmeans_assignments_{ch}.parquet"),
        ("gmm", f"gmm_assignments_{ch}.parquet"),
        ("spectral", f"spectral_assignments_{ch}.parquet"),
    ]:
        path = clustering_dir / "data" / filename
        if path.exists():
            df = pl.read_parquet(path)
            # Use first cluster column (the optimal-k one)
            cluster_cols = [c for c in df.columns if c.startswith("cluster_")]
            if cluster_cols:
                labels[method] = df[cluster_cols[0]].to_numpy()

    return labels if labels else None


# ── Phase 2: Vote Matrix Construction ────────────────────────────────────────


def build_vote_matrix(
    vm: pl.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert Polars vote matrix to numpy array for StepMix.

    Args:
        vm: Polars DataFrame with legislator_slug as first column,
            remaining columns are vote_ids with values 1/0/null.

    Returns:
        vote_array: (n_legislators, n_votes) float array with NaN for missing
        slugs: list of legislator slugs (row labels)
        vote_ids: list of vote_id strings (column labels)
    """
    slug_col = vm.columns[0]
    slugs = vm[slug_col].to_list()
    vote_ids = [c for c in vm.columns if c != slug_col]
    vote_array = vm.select(vote_ids).to_numpy().astype(float)
    return vote_array, slugs, vote_ids


# ── Phase 3: Class Enumeration ───────────────────────────────────────────────


def enumerate_classes(
    vote_array: np.ndarray,
    k_max: int = K_MAX,
    n_init: int = N_INIT,
) -> list[dict]:
    """Fit LCA models for K=1..k_max and collect fit statistics.

    Args:
        vote_array: (n_legislators, n_votes) binary array with NaN for missing
        k_max: maximum number of classes to test
        n_init: random restarts per K

    Returns:
        List of dicts with keys: k, bic, aic, log_likelihood, entropy, converged
    """
    results: list[dict] = []

    for k in range(1, k_max + 1):
        print(f"  Fitting K={k}...", end=" ", flush=True)
        model = StepMix(
            n_components=k,
            measurement="binary_nan",
            n_init=n_init,
            max_iter=MAX_ITER,
            random_state=RANDOM_SEED,
            verbose=0,
        )
        model.fit(vote_array)

        bic = model.bic(vote_array)
        aic = model.aic(vote_array)
        ll = model.score(vote_array) * vote_array.shape[0]  # score returns mean LL

        # Compute entropy (classification certainty)
        if k == 1:
            entropy = 1.0  # Perfect certainty with 1 class
        else:
            posteriors = model.predict_proba(vote_array)
            # Relative entropy: 1 - (observed entropy / max entropy)
            log_post = np.log(np.clip(posteriors, 1e-15, 1.0))
            h = -np.sum(posteriors * log_post) / vote_array.shape[0]
            h_max = np.log(k)
            entropy = 1.0 - (h / h_max) if h_max > 0 else 1.0

        converged = model.n_iter_ < MAX_ITER

        print(
            f"BIC={bic:.1f}, AIC={aic:.1f}, entropy={entropy:.3f}"
            f"{'' if converged else ' [NOT CONVERGED]'}"
        )

        results.append(
            {
                "k": k,
                "bic": bic,
                "aic": aic,
                "log_likelihood": ll,
                "entropy": entropy,
                "converged": converged,
            }
        )

    return results


def select_optimal_k(enumeration_results: list[dict]) -> tuple[int, str]:
    """Select optimal K via BIC minimum.

    Returns:
        (optimal_k, rationale_text)
    """
    bics = [(r["k"], r["bic"]) for r in enumeration_results]
    best_k, best_bic = min(bics, key=lambda x: x[1])

    # Build rationale
    parts = [f"BIC-minimum at K={best_k} (BIC={best_bic:.1f})."]
    best_result = next(r for r in enumeration_results if r["k"] == best_k)
    parts.append(f"Entropy={best_result['entropy']:.3f}.")

    if not best_result["converged"]:
        parts.append("WARNING: model did not converge at optimal K.")

    # Check if K=2 is very close (within 2% of best)
    k2_result = next((r for r in enumeration_results if r["k"] == 2), None)
    if k2_result and best_k != 2:
        pct_diff = (k2_result["bic"] - best_bic) / abs(best_bic) * 100
        if pct_diff < 2.0:
            parts.append(
                f"Note: K=2 BIC is within {pct_diff:.1f}% of optimal — parsimony may favor K=2."
            )

    return best_k, " ".join(parts)


# ── Phase 4: Final Model Fit ────────────────────────────────────────────────


def fit_final_model(
    vote_array: np.ndarray,
    k: int,
) -> tuple[StepMix, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the BIC-selected model and extract results.

    Returns:
        model: fitted StepMix model
        labels: (n_legislators,) hard class assignments
        probabilities: (n_legislators, k) posterior membership probabilities
        profiles: (k, n_votes) class-specific P(Yea) profiles
    """
    model = StepMix(
        n_components=k,
        measurement="binary_nan",
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    model.fit(vote_array)

    labels = model.predict(vote_array)
    probabilities = model.predict_proba(vote_array)

    # Extract class profiles: P(Yea | class) for each bill
    # StepMix stores parameters in _mm.parameters; shape depends on version
    params = model.get_parameters()
    # params is a dict; measurement params contain the probabilities
    # For binary measurement, look for the pis (item response probabilities)
    measurement_params = params.get("measurement", params)
    if isinstance(measurement_params, dict) and "pis" in measurement_params:
        profiles = measurement_params["pis"]
    elif isinstance(measurement_params, dict):
        # Try first value that's an ndarray with right shape
        for v in measurement_params.values():
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == k:
                profiles = v
                break
        else:
            profiles = _extract_profiles_from_posteriors(model, vote_array, k)
    else:
        profiles = _extract_profiles_from_posteriors(model, vote_array, k)

    return model, labels, probabilities, profiles


def _extract_profiles_from_posteriors(model: StepMix, vote_array: np.ndarray, k: int) -> np.ndarray:
    """Fallback: estimate class profiles from posterior-weighted vote means."""
    posteriors = model.predict_proba(vote_array)
    profiles = np.zeros((k, vote_array.shape[1]))
    for c in range(k):
        weights = posteriors[:, c]
        total_weight = weights.sum()
        if total_weight > 0:
            for j in range(vote_array.shape[1]):
                valid = ~np.isnan(vote_array[:, j])
                if valid.any():
                    profiles[c, j] = np.average(vote_array[valid, j], weights=weights[valid])
    return profiles


# ── Phase 5: Salsa Effect Detection ─────────────────────────────────────────


def detect_salsa_effect(
    profiles: np.ndarray,
) -> dict:
    """Check whether class profiles are qualitatively distinct or quantitatively graded.

    Computes pairwise Spearman correlations between class P(Yea) profiles.
    High correlations (> SALSA_THRESHOLD) indicate the "Salsa effect":
    classes are mild/medium/hot versions of the same pattern, not genuinely
    different voting coalitions.

    Returns:
        dict with keys: is_salsa, mean_correlation, min_correlation,
                        correlation_matrix, verdict
    """
    k = profiles.shape[0]

    if k <= 1:
        return {
            "is_salsa": False,
            "mean_correlation": None,
            "min_correlation": None,
            "correlation_matrix": None,
            "verdict": "Only 1 class — Salsa test not applicable.",
        }

    corr_matrix = np.ones((k, k))
    correlations = []

    for i in range(k):
        for j in range(i + 1, k):
            r, _ = spearmanr(profiles[i], profiles[j])
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r
            correlations.append(r)

    mean_corr = float(np.mean(correlations))
    min_corr = float(np.min(correlations))
    is_salsa = min_corr > SALSA_THRESHOLD

    if is_salsa:
        verdict = (
            f"SALSA EFFECT: All {len(correlations)} profile pairs have Spearman r > "
            f"{SALSA_THRESHOLD:.2f} (min={min_corr:.3f}, mean={mean_corr:.3f}). "
            "Classes represent quantitative grading (mild/medium/hot) along the "
            "same ideological dimension, not qualitatively distinct factions."
        )
    elif mean_corr > SALSA_THRESHOLD:
        verdict = (
            f"MIXED: Mean Spearman r={mean_corr:.3f} > {SALSA_THRESHOLD:.2f} but "
            f"min={min_corr:.3f} < threshold. Most class pairs are quantitatively "
            "similar, but at least one pair shows qualitative distinction."
        )
    else:
        verdict = (
            f"DISTINCT: Mean Spearman r={mean_corr:.3f} < {SALSA_THRESHOLD:.2f}. "
            "Classes represent qualitatively different voting patterns — "
            "genuine multi-dimensional structure detected."
        )

    return {
        "is_salsa": is_salsa,
        "mean_correlation": mean_corr,
        "min_correlation": min_corr,
        "correlation_matrix": corr_matrix,
        "verdict": verdict,
    }


# ── Phase 6: IRT Cross-Validation ───────────────────────────────────────────


def cross_validate_irt(
    labels: np.ndarray,
    probabilities: np.ndarray,
    slugs: list[str],
    irt_ideal_points: pl.DataFrame,
) -> dict:
    """Cross-validate LCA classes against IRT ideal points.

    For each class, compute mean/median IRT xi. Check monotonicity
    (classes ordered by IRT position). Report overlap.

    Returns:
        dict with keys: class_stats (list of dicts), is_monotonic,
                        straddlers (legislators with max P(class) < 0.7)
    """
    # Build slug → IRT mapping
    irt_map = dict(
        zip(
            irt_ideal_points["legislator_slug"].to_list(),
            irt_ideal_points["xi_mean"].to_list(),
        )
    )

    k = probabilities.shape[1]
    class_stats = []

    for c in range(k):
        members = [i for i in range(len(slugs)) if labels[i] == c]
        xi_values = [irt_map[slugs[i]] for i in members if slugs[i] in irt_map]

        if xi_values:
            class_stats.append(
                {
                    "class": c,
                    "n": len(members),
                    "mean_xi": float(np.mean(xi_values)),
                    "median_xi": float(np.median(xi_values)),
                    "sd_xi": float(np.std(xi_values)),
                    "min_xi": float(np.min(xi_values)),
                    "max_xi": float(np.max(xi_values)),
                }
            )
        else:
            class_stats.append(
                {
                    "class": c,
                    "n": len(members),
                    "mean_xi": None,
                    "median_xi": None,
                    "sd_xi": None,
                    "min_xi": None,
                    "max_xi": None,
                }
            )

    # Sort by mean IRT and check monotonicity
    valid_stats = [s for s in class_stats if s["mean_xi"] is not None]
    sorted_stats = sorted(valid_stats, key=lambda s: s["mean_xi"])
    means = [s["mean_xi"] for s in sorted_stats]
    is_monotonic = all(means[i] <= means[i + 1] for i in range(len(means) - 1))

    # Identify straddlers: max P(class) < 0.7
    max_probs = probabilities.max(axis=1)
    straddler_mask = max_probs < 0.7
    straddlers = [
        {
            "slug": slugs[i],
            "max_prob": float(max_probs[i]),
            "assigned_class": int(labels[i]),
            "xi": irt_map.get(slugs[i]),
        }
        for i in range(len(slugs))
        if straddler_mask[i]
    ]

    return {
        "class_stats": class_stats,
        "is_monotonic": is_monotonic,
        "n_straddlers": len(straddlers),
        "straddlers": sorted(straddlers, key=lambda s: s["max_prob"]),
    }


# ── Phase 7: Clustering Agreement ───────────────────────────────────────────


def compare_with_clustering(
    lca_labels: np.ndarray,
    lca_slugs: list[str],
    clustering_labels: dict[str, np.ndarray] | None,
    clustering_slugs: list[str] | None = None,
) -> dict[str, float]:
    """Compute ARI between LCA and Phase 5 cluster labels.

    If clustering labels are not available, returns empty dict.
    """
    if clustering_labels is None:
        return {}

    ari_scores: dict[str, float] = {}

    for method, cl_labels in clustering_labels.items():
        # If clustering has same length, assume same order
        if len(cl_labels) == len(lca_labels):
            ari = adjusted_rand_score(lca_labels, cl_labels)
            ari_scores[f"lca_vs_{method}"] = float(ari)
        else:
            # Try to align by slug order (if clustering_slugs provided)
            if clustering_slugs:
                cl_map = dict(zip(clustering_slugs, cl_labels))
                aligned_lca = []
                aligned_cl = []
                for i, slug in enumerate(lca_slugs):
                    if slug in cl_map:
                        aligned_lca.append(lca_labels[i])
                        aligned_cl.append(cl_map[slug])
                if len(aligned_lca) > 5:
                    ari = adjusted_rand_score(aligned_lca, aligned_cl)
                    ari_scores[f"lca_vs_{method}"] = float(ari)

    return ari_scores


# ── Phase 8: Within-Party LCA ───────────────────────────────────────────────


def run_within_party_lca(
    vote_array: np.ndarray,
    slugs: list[str],
    legislators: pl.DataFrame,
    chamber: str,
    k_max: int = 4,
) -> dict[str, dict]:
    """Run separate LCA on R-only and D-only subsets.

    Returns dict keyed by party with enumeration results and optimal K.
    """
    # Build slug → party mapping
    party_map = dict(
        zip(
            legislators["legislator_slug"].to_list(),
            legislators["party"].to_list(),
        )
    )

    results: dict[str, dict] = {}

    for party in ["Republican", "Democrat"]:
        indices = [i for i, s in enumerate(slugs) if party_map.get(s) == party]

        if len(indices) < 10:
            print(f"  {party}: skipping (only {len(indices)} legislators)")
            results[party] = {
                "skipped": True,
                "n_legislators": len(indices),
                "reason": f"Too few legislators ({len(indices)} < 10)",
            }
            continue

        print(f"\n  {party} ({len(indices)} legislators):")
        party_array = vote_array[indices]
        party_slugs = [slugs[i] for i in indices]

        # Drop columns with all NaN or no variance in this subset
        col_mask = np.ones(party_array.shape[1], dtype=bool)
        for j in range(party_array.shape[1]):
            col = party_array[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) < 3 or np.std(valid) < 0.01:
                col_mask[j] = False

        party_array_filtered = party_array[:, col_mask]
        print(f"    {party_array_filtered.shape[1]} informative votes (of {party_array.shape[1]})")

        if party_array_filtered.shape[1] < 5:
            print("    Skipping: too few informative votes")
            results[party] = {
                "skipped": True,
                "n_legislators": len(indices),
                "reason": "Too few informative votes within party",
            }
            continue

        # Run enumeration with smaller K range
        enum_results = enumerate_classes(party_array_filtered, k_max=k_max, n_init=N_INIT)
        optimal_k, rationale = select_optimal_k(enum_results)

        party_result: dict = {
            "skipped": False,
            "n_legislators": len(indices),
            "n_votes": party_array_filtered.shape[1],
            "enumeration": enum_results,
            "optimal_k": optimal_k,
            "rationale": rationale,
            "slugs": party_slugs,
        }

        # If K>1, fit final model
        if optimal_k > 1:
            _, labels, probs, profiles = fit_final_model(party_array_filtered, optimal_k)
            party_result["labels"] = labels
            party_result["probabilities"] = probs
            party_result["profiles"] = profiles

            salsa = detect_salsa_effect(profiles)
            party_result["salsa"] = salsa
            print(f"    {salsa['verdict']}")

        results[party] = party_result

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_bic_elbow(
    enumeration_results: list[dict],
    chamber: str,
    plots_dir: Path,
) -> None:
    """BIC/AIC elbow plot for model selection."""
    ks = [r["k"] for r in enumeration_results]
    bics = [r["bic"] for r in enumeration_results]
    aics = [r["aic"] for r in enumeration_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(ks, bics, "o-", color="#2c3e50", linewidth=2, markersize=8, label="BIC")
    ax1.plot(ks, aics, "s--", color="#7f8c8d", linewidth=1.5, markersize=6, label="AIC")

    # Mark BIC minimum
    best_k = ks[np.argmin(bics)]
    best_bic = min(bics)
    ax1.axvline(best_k, color="#e74c3c", linestyle=":", alpha=0.7, linewidth=1.5)
    ax1.annotate(
        f"BIC minimum: K={best_k}",
        xy=(best_k, best_bic),
        xytext=(best_k + 0.5, best_bic + (max(bics) - min(bics)) * 0.1),
        fontsize=10,
        color="#e74c3c",
        arrowprops={"arrowstyle": "->", "color": "#e74c3c"},
    )

    ax1.set_xlabel("Number of Latent Classes (K)", fontsize=12)
    ax1.set_ylabel("Information Criterion", fontsize=12)
    ax1.set_title(f"{chamber} — LCA Model Selection", fontsize=14, fontweight="bold")
    ax1.set_xticks(ks)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Secondary axis for entropy
    ax2 = ax1.twinx()
    entropies = [r["entropy"] for r in enumeration_results]
    ax2.plot(ks, entropies, "^-", color="#27ae60", linewidth=1, markersize=5, alpha=0.7)
    ax2.set_ylabel("Entropy", fontsize=10, color="#27ae60")
    ax2.tick_params(axis="y", labelcolor="#27ae60")
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    save_fig(fig, plots_dir / f"bic_elbow_{chamber.lower()}.png")


def plot_profile_heatmap(
    profiles: np.ndarray,
    vote_ids: list[str],
    chamber: str,
    plots_dir: Path,
    n_top: int = TOP_DISCRIMINATING_BILLS,
) -> None:
    """Heatmap of P(Yea | class) for the most discriminating bills."""
    k = profiles.shape[0]

    # Find most discriminating bills (max range across classes)
    ranges = np.ptp(profiles, axis=0)
    top_indices = np.argsort(ranges)[-n_top:][::-1]

    if len(top_indices) == 0:
        return

    top_profiles = profiles[:, top_indices]
    top_labels = [vote_ids[i] if i < len(vote_ids) else f"item_{i}" for i in top_indices]

    fig, ax = plt.subplots(figsize=(max(10, n_top * 0.4), max(3, k * 1.2)))
    im = ax.imshow(top_profiles, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)

    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Class {c + 1}" for c in range(k)], fontsize=10)
    ax.set_xticks(range(len(top_labels)))
    ax.set_xticklabels(top_labels, rotation=90, fontsize=7, ha="center")
    ax.set_title(
        f"{chamber} — Class Profiles: Top {len(top_indices)} Discriminating Votes",
        fontsize=13,
        fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("P(Yea | Class)", fontsize=10)

    fig.tight_layout()
    save_fig(fig, plots_dir / f"profile_heatmap_{chamber.lower()}.png")


def plot_membership_histogram(
    probabilities: np.ndarray,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Histogram of maximum membership probabilities."""
    max_probs = probabilities.max(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(max_probs, bins=20, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(0.7, color="#e74c3c", linestyle="--", linewidth=1.5, label="Straddler (0.7)")
    ax.set_xlabel("Maximum Class Membership Probability", fontsize=12)
    ax.set_ylabel("Number of Legislators", fontsize=12)
    ax.set_title(
        f"{chamber} — Classification Certainty",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    n_straddlers = (max_probs < 0.7).sum()
    ax.annotate(
        f"{n_straddlers} straddler{'s' if n_straddlers != 1 else ''} (max P < 0.7)",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        color="#e74c3c",
        verticalalignment="top",
    )

    fig.tight_layout()
    save_fig(fig, plots_dir / f"membership_hist_{chamber.lower()}.png")


def plot_irt_boxplot(
    labels: np.ndarray,
    slugs: list[str],
    irt_ideal_points: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Boxplot of IRT ideal points by LCA class."""
    irt_map = dict(
        zip(
            irt_ideal_points["legislator_slug"].to_list(),
            irt_ideal_points["xi_mean"].to_list(),
        )
    )
    party_map = dict(
        zip(
            irt_ideal_points["legislator_slug"].to_list(),
            irt_ideal_points["party"].to_list(),
        )
    )

    k = int(labels.max()) + 1
    class_data = []
    for c in range(k):
        members = [i for i in range(len(slugs)) if labels[i] == c]
        xi_vals = [irt_map[slugs[i]] for i in members if slugs[i] in irt_map]
        class_data.append(xi_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(class_data, tick_labels=[f"Class {c + 1}" for c in range(k)], patch_artist=True)

    colors = plt.get_cmap(CLASS_CMAP)(np.linspace(0, 1, k))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points colored by party
    for c in range(k):
        members = [i for i in range(len(slugs)) if labels[i] == c]
        for i in members:
            if slugs[i] in irt_map:
                party = party_map.get(slugs[i], "Independent")
                ax.scatter(
                    c + 1 + np.random.uniform(-0.15, 0.15),
                    irt_map[slugs[i]],
                    color=PARTY_COLORS.get(party, "#999999"),
                    s=15,
                    alpha=0.6,
                    zorder=3,
                )

    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=c, label=p) for p, c in PARTY_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_xlabel("Latent Class", fontsize=12)
    ax.set_ylabel("IRT Ideal Point (xi)", fontsize=12)
    ax.set_title(
        f"{chamber} — IRT Ideal Points by LCA Class",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_fig(fig, plots_dir / f"irt_boxplot_{chamber.lower()}.png")


def plot_salsa_matrix(
    corr_matrix: np.ndarray,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Heatmap of pairwise Spearman correlations between class profiles."""
    k = corr_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([f"C{c + 1}" for c in range(k)])
    ax.set_yticklabels([f"C{c + 1}" for c in range(k)])

    # Annotate cells
    for i in range(k):
        for j in range(k):
            ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
            )

    ax.set_title(
        f"{chamber} — Profile Correlations (Salsa Test)",
        fontsize=12,
        fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman r", fontsize=10)

    fig.tight_layout()
    save_fig(fig, plots_dir / f"salsa_matrix_{chamber.lower()}.png")


# ── Discriminating Bills ─────────────────────────────────────────────────────


def find_discriminating_bills(
    profiles: np.ndarray,
    vote_ids: list[str],
    n_top: int = TOP_DISCRIMINATING_BILLS,
) -> list[dict]:
    """Find bills with the largest range in P(Yea) across classes."""
    ranges = np.ptp(profiles, axis=0)
    top_indices = np.argsort(ranges)[-n_top:][::-1]

    bills = []
    for idx in top_indices:
        if idx < len(vote_ids):
            bills.append(
                {
                    "vote_id": vote_ids[idx],
                    "range": float(ranges[idx]),
                    "profiles": profiles[:, idx].tolist(),
                }
            )
    return bills


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from tallgrass.session import KSSession

    ks = KSSession.from_session_string(args.session)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = ks.data_dir

    results_root = ks.results_dir

    eda_dir = resolve_upstream_dir(
        "01_eda",
        results_root,
        args.run_id,
        Path(args.eda_dir) if args.eda_dir else None,
    )
    irt_dir = resolve_upstream_dir(
        "05_irt",
        results_root,
        args.run_id,
        Path(args.irt_dir) if args.irt_dir else None,
    )

    # Clustering dir is optional (for ARI comparison)
    clustering_dir = None
    if args.clustering_dir:
        clustering_dir = Path(args.clustering_dir)
    else:
        try:
            clustering_dir = resolve_upstream_dir("09_clustering", results_root, args.run_id, None)
        except FileNotFoundError:
            print("  Note: Phase 5 clustering not found — skipping ARI comparison")

    with RunContext(
        session=args.session,
        analysis_name="10_lca",
        params=vars(args),
        primer=LCA_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"KS Legislature Latent Class Analysis — Session {args.session}")
        print(f"Data:       {data_dir}")
        print(f"EDA:        {eda_dir}")
        print(f"IRT:        {irt_dir}")
        print(f"Clustering: {clustering_dir or 'not available'}")
        print(f"Output:     {ctx.run_dir}")

        # ── Phase 1: Load data ──
        print_header("PHASE 1: LOADING DATA")
        vm_house, vm_senate = load_vote_matrices(eda_dir)
        irt_house, irt_senate = load_irt_ideal_points(irt_dir)
        legislators = load_legislators(data_dir)

        if vm_house is None and vm_senate is None:
            print("Phase 10 (LCA): skipping — no EDA vote matrices available")
            return

        for label, df in [("Vote matrix House", vm_house), ("Vote matrix Senate", vm_senate)]:
            info = f"{df.height} x {len(df.columns) - 1}" if df is not None else "not available"
            print(f"  {label}:  {info}")
        for label, df in [("IRT House", irt_house), ("IRT Senate", irt_senate)]:
            info = f"{df.height} legislators" if df is not None else "not available"
            print(f"  {label}:  {info}")
        print(f"  Legislators: {legislators.height}")

        chamber_configs = [
            ("House", vm_house, irt_house),
            ("Senate", vm_senate, irt_senate),
        ]

        results: dict[str, dict] = {}

        for chamber, vm, irt_ip in chamber_configs:
            if vm is None or vm.height < 5:
                n = vm.height if vm is not None else 0
                print(f"\n  Skipping {chamber}: too few legislators ({n})")
                continue
            if irt_ip is None:
                print(f"\n  Skipping {chamber}: no IRT ideal points available")
                continue

            chamber_results: dict = {}

            # ── Phase 2: Build vote matrix ──
            print_header(f"PHASE 2: VOTE MATRIX — {chamber}")
            vote_array, slugs, vote_ids = build_vote_matrix(vm)
            n_missing = np.isnan(vote_array).sum()
            n_total = vote_array.size
            print(f"  Shape: {vote_array.shape[0]} legislators x {vote_array.shape[1]} votes")
            print(f"  Missing: {n_missing} / {n_total} ({100 * n_missing / n_total:.1f}%)")

            chamber_results["n_legislators"] = vote_array.shape[0]
            chamber_results["n_votes"] = vote_array.shape[1]
            chamber_results["pct_missing"] = 100 * n_missing / n_total

            # ── Phase 3: Class enumeration ──
            print_header(f"PHASE 3: CLASS ENUMERATION — {chamber}")
            enum_results = enumerate_classes(vote_array, k_max=args.k_max)
            optimal_k, rationale = select_optimal_k(enum_results)
            print(f"\n  Optimal K: {optimal_k}")
            print(f"  {rationale}")

            chamber_results["enumeration"] = enum_results
            chamber_results["optimal_k"] = optimal_k
            chamber_results["rationale"] = rationale

            # Save enumeration results
            enum_df = pl.DataFrame(enum_results)
            enum_df.write_parquet(ctx.data_dir / f"enumeration_{chamber.lower()}.parquet")
            print(f"  Saved: enumeration_{chamber.lower()}.parquet")

            # Plot BIC elbow
            plot_bic_elbow(enum_results, chamber, ctx.plots_dir)

            # ── Phase 4: Fit final model ──
            print_header(f"PHASE 4: FINAL MODEL (K={optimal_k}) — {chamber}")
            model, labels, probabilities, profiles = fit_final_model(vote_array, optimal_k)

            # Class sizes
            for c in range(optimal_k):
                n_c = (labels == c).sum()
                frac = n_c / len(labels)
                flag = " [SMALL]" if frac < MIN_CLASS_FRACTION else ""
                print(f"  Class {c + 1}: {n_c} legislators ({100 * frac:.1f}%){flag}")

            chamber_results["labels"] = labels
            chamber_results["probabilities"] = probabilities
            chamber_results["profiles"] = profiles
            chamber_results["slugs"] = slugs
            chamber_results["vote_ids"] = vote_ids

            # Save class assignments
            assign_df = pl.DataFrame(
                {
                    "legislator_slug": slugs,
                    "class": labels.tolist(),
                    "max_probability": probabilities.max(axis=1).tolist(),
                }
            )
            # Add per-class probabilities
            for c in range(optimal_k):
                assign_df = assign_df.with_columns(
                    pl.Series(f"prob_class_{c + 1}", probabilities[:, c].tolist())
                )
            assign_df.write_parquet(ctx.data_dir / f"class_assignments_{chamber.lower()}.parquet")
            print(f"  Saved: class_assignments_{chamber.lower()}.parquet")

            # Save class profiles
            profile_df = pl.DataFrame(
                {
                    "vote_id": vote_ids,
                    **{f"class_{c + 1}_p_yea": profiles[c].tolist() for c in range(optimal_k)},
                }
            )
            profile_df.write_parquet(ctx.data_dir / f"class_profiles_{chamber.lower()}.parquet")
            print(f"  Saved: class_profiles_{chamber.lower()}.parquet")

            # Plot profiles and membership
            plot_profile_heatmap(profiles, vote_ids, chamber, ctx.plots_dir)
            plot_membership_histogram(probabilities, chamber, ctx.plots_dir)

            # ── Phase 5: Salsa effect ──
            print_header(f"PHASE 5: SALSA EFFECT TEST — {chamber}")
            salsa = detect_salsa_effect(profiles)
            print(f"  {salsa['verdict']}")
            chamber_results["salsa"] = salsa

            if salsa["correlation_matrix"] is not None and optimal_k > 2:
                plot_salsa_matrix(salsa["correlation_matrix"], chamber, ctx.plots_dir)

            # ── Phase 6: IRT cross-validation ──
            print_header(f"PHASE 6: IRT CROSS-VALIDATION — {chamber}")
            irt_cv = cross_validate_irt(labels, probabilities, slugs, irt_ip)
            print(f"  Classes monotonic in IRT space: {irt_cv['is_monotonic']}")
            print(f"  Straddlers (max P < 0.7): {irt_cv['n_straddlers']}")
            for cs in irt_cv["class_stats"]:
                if cs["mean_xi"] is not None:
                    print(
                        f"  Class {cs['class'] + 1}: mean xi={cs['mean_xi']:.3f}, "
                        f"sd={cs['sd_xi']:.3f}, n={cs['n']}"
                    )
            chamber_results["irt_cv"] = irt_cv

            # Plot IRT boxplot
            plot_irt_boxplot(labels, slugs, irt_ip, chamber, ctx.plots_dir)

            # ── Phase 7: Clustering agreement ──
            print_header(f"PHASE 7: CLUSTERING AGREEMENT — {chamber}")
            if clustering_dir is not None:
                cl_labels = load_clustering_labels(clustering_dir, chamber)
                ari_scores = compare_with_clustering(labels, slugs, cl_labels)
                for pair, ari in ari_scores.items():
                    print(f"  {pair}: ARI = {ari:.3f}")
                chamber_results["ari_scores"] = ari_scores
            else:
                print("  Skipped (Phase 5 clustering not available)")
                chamber_results["ari_scores"] = {}

            # ── Phase 8: Within-party LCA ──
            if not args.skip_within_party:
                print_header(f"PHASE 8: WITHIN-PARTY LCA — {chamber}")
                # Filter legislators to this chamber
                chamber_prefix = "rep_" if chamber == "House" else "sen_"
                chamber_legs = legislators.filter(
                    pl.col("legislator_slug").str.starts_with(chamber_prefix)
                )
                within_party = run_within_party_lca(
                    vote_array, slugs, chamber_legs, chamber, k_max=4
                )
                chamber_results["within_party"] = within_party

                # Save within-party results
                for party, wp_result in within_party.items():
                    if not wp_result.get("skipped", True) and "labels" in wp_result:
                        wp_df = pl.DataFrame(
                            {
                                "legislator_slug": wp_result["slugs"],
                                "class": wp_result["labels"].tolist(),
                            }
                        )
                        fname = f"within_party_{party.lower()}_{chamber.lower()}.parquet"
                        wp_df.write_parquet(ctx.data_dir / fname)
                        print(f"  Saved: {fname}")
            else:
                print("\n  Skipping within-party LCA (--skip-within-party)")
                chamber_results["within_party"] = {}

            # Discriminating bills
            disc_bills = find_discriminating_bills(profiles, vote_ids)
            chamber_results["discriminating_bills"] = disc_bills

            # Class composition by party
            party_map = dict(
                zip(
                    irt_ip["legislator_slug"].to_list(),
                    irt_ip["party"].to_list(),
                )
            )
            composition: list[dict] = []
            for c in range(optimal_k):
                members = [slugs[i] for i in range(len(slugs)) if labels[i] == c]
                parties = [party_map.get(s, "Unknown") for s in members]
                comp = {"class": c + 1, "n": len(members)}
                for party in ["Republican", "Democrat", "Independent"]:
                    comp[party] = parties.count(party)
                composition.append(comp)
            chamber_results["composition"] = composition

            # Class membership list (for report)
            xi_map = dict(zip(irt_ip["legislator_slug"].to_list(), irt_ip["xi_mean"].to_list()))
            name_map = dict(
                zip(
                    legislators["legislator_slug"].to_list(),
                    legislators["full_name"].to_list(),
                )
            )
            membership_rows: list[dict] = []
            for i, slug in enumerate(slugs):
                membership_rows.append(
                    {
                        "Name": name_map.get(slug, slug),
                        "Party": party_map.get(slug, "Unknown"),
                        "Class": int(labels[i]) + 1,
                        "IRT xi": xi_map.get(slug),
                        "Max P": float(probabilities[i].max()),
                    }
                )
            chamber_results["membership"] = sorted(
                membership_rows, key=lambda r: (r["Class"], r.get("IRT xi") or 0)
            )

            results[chamber] = chamber_results

        # ── Phase 9: Filtering Manifest ──
        print_header("FILTERING MANIFEST")
        manifest: dict = {
            "analysis": "lca",
            "constants": {
                "K_MAX": args.k_max,
                "N_INIT": N_INIT,
                "MAX_ITER": MAX_ITER,
                "RANDOM_SEED": RANDOM_SEED,
                "MIN_VOTES": MIN_VOTES,
                "MINORITY_THRESHOLD": MINORITY_THRESHOLD,
                "MIN_CLASS_FRACTION": MIN_CLASS_FRACTION,
                "SALSA_THRESHOLD": SALSA_THRESHOLD,
            },
            "skip_within_party": args.skip_within_party,
        }
        for chamber, result in results.items():
            ch = chamber.lower()
            manifest[f"{ch}_n_legislators"] = result["n_legislators"]
            manifest[f"{ch}_n_votes"] = result["n_votes"]
            manifest[f"{ch}_optimal_k"] = result["optimal_k"]
            manifest[f"{ch}_rationale"] = result["rationale"]
            if "salsa" in result:
                manifest[f"{ch}_salsa_verdict"] = result["salsa"]["verdict"]
            if "irt_cv" in result:
                manifest[f"{ch}_irt_monotonic"] = result["irt_cv"]["is_monotonic"]
                manifest[f"{ch}_n_straddlers"] = result["irt_cv"]["n_straddlers"]

        if not results:
            print("Phase 10 (LCA): skipping — no chambers had sufficient data")
            return

        save_filtering_manifest(manifest, ctx.run_dir)

        # ── Phase 10: HTML Report ──
        print_header("HTML REPORT")
        build_lca_report(
            ctx.report,
            results=results,
            plots_dir=ctx.plots_dir,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
