"""Shared MCMC initialization strategy for IRT ideal points.

Configures how legislator ideal point chains are initialized before sampling.
Used by Phase 06 (2D IRT), Phase 07 (Hierarchical IRT), and future models.

Design follows the IdentificationStrategy pattern (ADR-0103): string constants,
parallel metadata dicts, auto-detection with full rationale logging.

Eventually exposed as a Django CharField(choices=...) for pipeline configuration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


class InitStrategy:
    """MCMC chain initialization strategies for IRT ideal points.

    Each strategy determines the source of starting values for legislator
    ideal point parameters (xi) before MCMC sampling begins. Good initialization
    reduces tuning time and improves convergence, especially for multidimensional
    models where the posterior has rotational ambiguity.
    """

    # ── Strategy constants (kebab-case, matches CLI choices) ──
    IRT_INFORMED = "irt-informed"
    PCA_INFORMED = "pca-informed"
    AUTO = "auto"

    # Registry (excludes AUTO — same pattern as IdentificationStrategy)
    ALL_STRATEGIES = [IRT_INFORMED, PCA_INFORMED]

    DESCRIPTIONS: dict[str, str] = {
        IRT_INFORMED: (
            "1D IRT ideal points (xi_mean) — directly measures ideology. "
            "Strongest initialization: converged posterior means from the canonical "
            "1D model provide the best starting position for the ideology dimension."
        ),
        PCA_INFORMED: (
            "PCA PC1 scores — proxy for ideology via variance decomposition. "
            "Fast and always available when Phase 02 has run, but less precise "
            "than converged IRT estimates."
        ),
    }

    REFERENCES: dict[str, str] = {
        IRT_INFORMED: "Phase 05 canonical 1D Bayesian IRT (ADR-0103)",
        PCA_INFORMED: "Phase 02 PCA on binary vote matrix",
    }

    # Django-ready choices tuple: (db_value, display_label)
    CHOICES = [
        (AUTO, "Auto (prefer IRT, fall back to PCA)"),
        (IRT_INFORMED, "1D IRT ideal points"),
        (PCA_INFORMED, "PCA PC1 scores"),
    ]


def resolve_init_source(
    strategy: str,
    slugs: list[str],
    irt_scores: pl.DataFrame | None = None,
    pca_scores: pl.DataFrame | None = None,
    pca_column: str = "PC1",
) -> tuple[np.ndarray, str, str]:
    """Resolve initialization values for IRT ideal points.

    Loads upstream scores, matches to the model's legislator ordering,
    and standardizes to zero-mean unit-variance (appropriate for Normal(0,1) priors).

    Args:
        strategy: One of InitStrategy constants ("auto", "irt-informed", "pca-informed").
        slugs: Legislator slugs in model order.
        irt_scores: 1D IRT ideal points DataFrame (columns: legislator_slug, xi_mean).
        pca_scores: PCA scores DataFrame (columns: legislator_slug, PC1, PC2, ...).
        pca_column: Which PCA column to use (default "PC1"; use "PC2" for Dim 2).

    Returns:
        (values, resolved_strategy, source_label):
        - values: np.ndarray of shape (n_legislators,), standardized.
        - resolved_strategy: the actual strategy used (differs from input when auto).
        - source_label: human-readable description for logs/reports.

    Raises:
        ValueError: If the requested strategy's upstream data is unavailable
            (and strategy is not "auto").
    """
    IS = InitStrategy

    # ── Auto-detection ──
    if strategy == IS.AUTO:
        if irt_scores is not None and pca_column == "PC1":
            strategy = IS.IRT_INFORMED
        elif pca_scores is not None:
            strategy = IS.PCA_INFORMED
        else:
            return np.zeros(len(slugs)), "none", "zeros (no upstream data available)"

    # ── Resolve values ──
    if strategy == IS.IRT_INFORMED:
        if irt_scores is None:
            raise ValueError(
                "irt-informed strategy requires 1D IRT results (Phase 05). "
                "Run `just irt` first or use --init-strategy pca-informed."
            )
        score_map = {
            row["legislator_slug"]: row["xi_mean"]
            for row in irt_scores.iter_rows(named=True)
        }
        vals = np.array([score_map.get(s, 0.0) for s in slugs])
        matched = sum(1 for s in slugs if s in score_map)
        source = f"1D IRT xi_mean ({matched}/{len(slugs)} matched)"

    elif strategy == IS.PCA_INFORMED:
        if pca_scores is None:
            raise ValueError(
                "pca-informed strategy requires PCA results (Phase 02). "
                "Run `just pca` first or use --init-strategy irt-informed."
            )
        score_map = {
            row["legislator_slug"]: row[pca_column]
            for row in pca_scores.iter_rows(named=True)
        }
        vals = np.array([score_map.get(s, 0.0) for s in slugs])
        matched = sum(1 for s in slugs if s in score_map)
        source = f"PCA {pca_column} ({matched}/{len(slugs)} matched)"

    else:
        raise ValueError(
            f"Unknown init strategy: {strategy!r}. "
            f"Valid: {IS.ALL_STRATEGIES + [IS.AUTO]}"
        )

    # ── Standardize to unit scale (matches Normal(0,1) prior) ──
    std = vals.std()
    if std > 0:
        vals = (vals - vals.mean()) / std

    return vals, strategy, source


def build_init_rationale(
    irt_available: bool,
    pca_available: bool,
    selected: str,
    auto: bool = False,
) -> dict[str, str]:
    """Build rationale dict explaining why each strategy was/wasn't selected.

    Same pattern as IdentificationStrategy's rationale — every strategy gets
    an explanation, prefixed with "SELECTED" or "Not selected".

    Args:
        irt_available: Whether 1D IRT results exist.
        pca_available: Whether PCA results exist.
        selected: The strategy that was actually chosen.
        auto: Whether auto-detection was used.

    Returns:
        Dict mapping strategy name → rationale string.
    """
    IS = InitStrategy
    rationale: dict[str, str] = {}

    # IRT-informed rationale
    if irt_available:
        rationale[IS.IRT_INFORMED] = (
            "1D IRT ideal points available. Converged posterior means provide "
            "the strongest initialization for the ideology dimension."
        )
    else:
        rationale[IS.IRT_INFORMED] = (
            "1D IRT results not found. Run Phase 05 first to enable this strategy."
        )

    # PCA-informed rationale
    if pca_available:
        rationale[IS.PCA_INFORMED] = (
            "PCA scores available. PC1 provides a reasonable proxy for ideology, "
            "though less precise than converged IRT estimates."
        )
    else:
        rationale[IS.PCA_INFORMED] = (
            "PCA results not found. Run Phase 02 first to enable this strategy."
        )

    # Mark selected/not-selected
    prefix_type = "auto" if auto else "user override"
    for s in IS.ALL_STRATEGIES:
        if s == selected:
            rationale[s] = f"SELECTED ({prefix_type}). " + rationale[s]
        else:
            rationale[s] = "Not selected. " + rationale[s]

    return rationale


def load_irt_scores(irt_data_dir: Path | str, chamber: str) -> pl.DataFrame | None:
    """Load 1D IRT ideal points for a chamber from Phase 05 output.

    Args:
        irt_data_dir: Path to the IRT phase data directory (e.g., .../05_irt/data/).
        chamber: "house" or "senate" (lowercase).

    Returns:
        DataFrame with legislator_slug and xi_mean columns, or None if not found.
    """
    path = Path(irt_data_dir) / f"ideal_points_{chamber}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)
