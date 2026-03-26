"""Shared MCMC initialization strategy for IRT ideal points.

Configures how legislator ideal point chains are initialized before sampling.
Used by Phase 06 (2D IRT), Phase 07 (Hierarchical IRT), and future models.

Design follows the IdentificationStrategy pattern (ADR-0103): string constants,
parallel metadata dicts, auto-detection with full rationale logging.

Party-aware PC selection: when strategy is pca-informed, auto-detects which PC
has the strongest party correlation and uses that instead of always PC1. This
fixes the axis instability problem in supermajority Senate sessions where PC1
captures intra-R factionalism. See docs/pca-ideology-axis-instability.md.

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
    IRT_2D_DIM1 = "2d-dim1"
    CANONICAL = "canonical"
    AUTO = "auto"

    # Registry (excludes AUTO — same pattern as IdentificationStrategy)
    ALL_STRATEGIES = [IRT_INFORMED, PCA_INFORMED, IRT_2D_DIM1, CANONICAL]

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
        IRT_2D_DIM1: (
            "2D IRT Dimension 1 (xi_dim1_mean) — ideology axis from the 2D model. "
            "Use for iterative refinement: run the pipeline normally, then re-run "
            "the 1D model with 2D Dim 1 to separate ideology from establishment."
        ),
        CANONICAL: (
            "Canonical routing output (ADR-0109/0111) — horseshoe-corrected ideal "
            "points. For horseshoe-affected chambers, uses 2D Dim 1; for balanced "
            "chambers, uses 1D IRT. Best for hierarchical model initialization."
        ),
    }

    REFERENCES: dict[str, str] = {
        IRT_INFORMED: "Phase 05 canonical 1D Bayesian IRT (ADR-0103)",
        PCA_INFORMED: "Phase 02 PCA on binary vote matrix",
        IRT_2D_DIM1: "Phase 06 experimental 2D IRT (ADR-0054)",
        CANONICAL: "Phase 06 canonical routing output (ADR-0109, ADR-0111)",
    }

    # Django-ready choices tuple: (db_value, display_label)
    CHOICES = [
        (AUTO, "Auto (prefer canonical, IRT, then PCA)"),
        (CANONICAL, "Canonical routing output (horseshoe-corrected)"),
        (IRT_INFORMED, "1D IRT ideal points"),
        (PCA_INFORMED, "PCA PC1 scores"),
        (IRT_2D_DIM1, "2D IRT Dimension 1 (ideology)"),
    ]


# Minimum party correlation to accept a PC swap (avoids swapping on noise)
PC_SWAP_MIN_PARTY_CORR = 0.30


def detect_ideology_pc(
    pca_scores: pl.DataFrame,
    candidates: list[str] | None = None,
) -> tuple[str, float, dict[str, float]]:
    """Detect which PC has the strongest party correlation.

    Computes point-biserial correlation between each candidate PC and a binary
    party indicator (Republican=1, Democrat=0). Returns the PC with the strongest
    absolute correlation.

    In supermajority Kansas Senate sessions (78th-83rd, 88th), PC1 captures
    intra-Republican factionalism while PC2 captures the party divide. This
    function detects the swap and returns the correct PC for ideology init.

    Args:
        pca_scores: DataFrame with legislator_slug, party, PC1, PC2, ... columns.
        candidates: PC column names to check (default: ["PC1", "PC2"]).

    Returns:
        (best_pc, best_corr, all_corrs):
        - best_pc: column name with strongest |correlation| with party
        - best_corr: the correlation value (signed)
        - all_corrs: dict mapping each candidate → correlation
    """
    if candidates is None:
        candidates = ["PC1", "PC2"]

    # Filter to R/D only (Independents excluded from correlation)
    rd = pca_scores.filter(pl.col("party").is_in(["Republican", "Democrat"]))
    if rd.height < 5:
        return candidates[0], 0.0, {c: 0.0 for c in candidates}

    # Binary indicator: Republican=1, Democrat=0
    party_binary = (rd["party"] == "Republican").cast(pl.Float64).to_numpy()

    all_corrs: dict[str, float] = {}
    for pc in candidates:
        if pc not in rd.columns:
            all_corrs[pc] = 0.0
            continue
        pc_vals = rd[pc].to_numpy().astype(np.float64)
        # Point-biserial correlation = Pearson correlation with binary variable
        valid = ~np.isnan(pc_vals)
        if valid.sum() < 5:
            all_corrs[pc] = 0.0
            continue
        r = float(np.corrcoef(pc_vals[valid], party_binary[valid])[0, 1])
        all_corrs[pc] = r

    # Select the PC with strongest absolute correlation
    best_pc = max(candidates, key=lambda c: abs(all_corrs.get(c, 0.0)))
    best_corr = all_corrs.get(best_pc, 0.0)

    return best_pc, best_corr, all_corrs


def load_pca_override(session: str | None, chamber: str | None) -> str | None:
    """Load a manual PCA dimension override for a session+chamber.

    Reads ``analysis/pca_overrides.yaml`` and returns the ideology PC column name
    (e.g., ``"PC2"``) if an override exists, or ``None`` to fall through to
    automated detection.

    Args:
        session: Canonical session string (e.g., ``"79th_2001-2002"``).
        chamber: Chamber name (e.g., ``"senate"`` or ``"Senate"``).

    Returns:
        PC column name (e.g., ``"PC2"``) or ``None``.
    """
    if session is None or chamber is None:
        return None

    override_path = Path(__file__).parent / "pca_overrides.yaml"
    if not override_path.exists():
        return None

    import yaml

    with open(override_path) as f:
        overrides = yaml.safe_load(f)
    if not isinstance(overrides, dict):
        return None

    # Extract legislature name from session string: "79th_2001-2002" → "79th"
    legislature_name = session.split("_")[0]
    key = f"{legislature_name}_{chamber.title()}"

    entry = overrides.get(key)
    if isinstance(entry, dict) and "ideology_pc" in entry:
        return entry["ideology_pc"]
    return None


def resolve_init_source(
    strategy: str,
    slugs: list[str],
    irt_scores: pl.DataFrame | None = None,
    pca_scores: pl.DataFrame | None = None,
    irt_2d_scores: pl.DataFrame | None = None,
    canonical_scores: pl.DataFrame | None = None,
    pca_column: str = "PC1",
    session: str | None = None,
    chamber: str | None = None,
) -> tuple[np.ndarray, str, str]:
    """Resolve initialization values for IRT ideal points.

    Loads upstream scores, matches to the model's legislator ordering,
    and standardizes to zero-mean unit-variance (appropriate for Normal(0,1) priors).

    Args:
        strategy: One of InitStrategy constants ("auto", "irt-informed",
            "pca-informed", "2d-dim1", "canonical").
        slugs: Legislator slugs in model order.
        irt_scores: 1D IRT ideal points DataFrame (columns: legislator_slug, xi_mean).
        pca_scores: PCA scores DataFrame (columns: legislator_slug, PC1, PC2, ...).
        irt_2d_scores: 2D IRT ideal points DataFrame
            (columns: legislator_slug, xi_dim1_mean).
        canonical_scores: Canonical routing output DataFrame
            (columns: legislator_slug, xi_mean). From Phase 06 canonical_irt/.
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
        if canonical_scores is not None and pca_column == "PC1":
            strategy = IS.CANONICAL
        elif irt_scores is not None and pca_column == "PC1":
            strategy = IS.IRT_INFORMED
        elif pca_scores is not None:
            strategy = IS.PCA_INFORMED
        else:
            return np.zeros(len(slugs)), "none", "zeros (no upstream data available)"

    # ── Resolve values ──
    if strategy == IS.CANONICAL:
        if canonical_scores is None:
            raise ValueError(
                "canonical strategy requires canonical routing output (Phase 06). "
                "Run `just irt-2d` first or use --init-strategy irt-informed."
            )
        score_map = {
            row["legislator_slug"]: row["xi_mean"] for row in canonical_scores.iter_rows(named=True)
        }
        vals = np.array([score_map.get(s, 0.0) for s in slugs])
        matched = sum(1 for s in slugs if s in score_map)
        source_col = "source"
        source_type = "canonical"
        if source_col in canonical_scores.columns:
            sources = canonical_scores[source_col].unique().to_list()
            source_type = f"canonical ({'/'.join(sources)})"
        source = f"{source_type} xi_mean ({matched}/{len(slugs)} matched)"

    elif strategy == IS.IRT_INFORMED:
        if irt_scores is None:
            raise ValueError(
                "irt-informed strategy requires 1D IRT results (Phase 05). "
                "Run `just irt` first or use --init-strategy pca-informed."
            )
        score_map = {
            row["legislator_slug"]: row["xi_mean"] for row in irt_scores.iter_rows(named=True)
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
        # Auto-detect best PC for ideology when caller requests default PC1.
        # When pca_column is explicitly set to something other than PC1 (e.g., "PC2"
        # for 2D IRT Dim 2 init), respect the caller's choice.
        actual_column = pca_column
        # 1. Check manual override first (pca_overrides.yaml)
        override_pc = load_pca_override(session, chamber) if pca_column == "PC1" else None
        if override_pc is not None and override_pc in pca_scores.columns:
            actual_column = override_pc
            print(
                f"  PCA override: using {override_pc} for {session} {chamber} "
                f"(from pca_overrides.yaml)"
            )
        # 2. Fall back to automated party-correlation detection
        elif pca_column == "PC1" and "party" in pca_scores.columns:
            best_pc, best_corr, all_corrs = detect_ideology_pc(pca_scores)
            pc1_corr = all_corrs.get("PC1", 0.0)
            if (
                best_pc != "PC1"
                and abs(best_corr) > PC_SWAP_MIN_PARTY_CORR
                and abs(best_corr) > abs(pc1_corr)
                and best_pc in pca_scores.columns
            ):
                actual_column = best_pc
                print(
                    f"  PC swap detected: {best_pc} has stronger party correlation "
                    f"(r={best_corr:.3f}) than PC1 (r={pc1_corr:.3f}) — "
                    f"using {best_pc} for ideology init"
                )
        score_map = {
            row["legislator_slug"]: row[actual_column] for row in pca_scores.iter_rows(named=True)
        }
        vals = np.array([score_map.get(s, 0.0) for s in slugs])
        matched = sum(1 for s in slugs if s in score_map)
        pc_note = f" (swapped from {pca_column})" if actual_column != pca_column else ""
        source = f"PCA {actual_column}{pc_note} ({matched}/{len(slugs)} matched)"

    elif strategy == IS.IRT_2D_DIM1:
        if irt_2d_scores is None:
            raise ValueError(
                "2d-dim1 strategy requires 2D IRT results (Phase 06). "
                "Run `just irt-2d` first or use --init-strategy pca-informed."
            )
        score_map = {
            row["legislator_slug"]: row["xi_dim1_mean"]
            for row in irt_2d_scores.iter_rows(named=True)
        }
        vals = np.array([score_map.get(s, 0.0) for s in slugs])
        matched = sum(1 for s in slugs if s in score_map)
        source = f"2D IRT Dim 1 ({matched}/{len(slugs)} matched)"

    else:
        raise ValueError(
            f"Unknown init strategy: {strategy!r}. Valid: {IS.ALL_STRATEGIES + [IS.AUTO]}"
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
    irt_2d_available: bool = False,
    canonical_available: bool = False,
) -> dict[str, str]:
    """Build rationale dict explaining why each strategy was/wasn't selected.

    Same pattern as IdentificationStrategy's rationale — every strategy gets
    an explanation, prefixed with "SELECTED" or "Not selected".

    Args:
        irt_available: Whether 1D IRT results exist.
        pca_available: Whether PCA results exist.
        selected: The strategy that was actually chosen.
        auto: Whether auto-detection was used.
        irt_2d_available: Whether 2D IRT results exist.

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

    # 2D Dim 1 rationale
    if irt_2d_available:
        rationale[IS.IRT_2D_DIM1] = (
            "2D IRT Dim 1 available. Ideology axis from the 2D model separates "
            "ideology from establishment — use for iterative refinement."
        )
    else:
        rationale[IS.IRT_2D_DIM1] = (
            "2D IRT results not found. Run Phase 06 first to enable this strategy."
        )

    # Canonical rationale
    if canonical_available:
        rationale[IS.CANONICAL] = (
            "Canonical routing output available. Horseshoe-corrected ideal points "
            "(2D Dim 1 for supermajority chambers, 1D IRT for balanced chambers)."
        )
    else:
        rationale[IS.CANONICAL] = (
            "Canonical routing output not found. Run Phase 06 with canonical routing "
            "to enable this strategy."
        )

    # Mark selected/not-selected
    prefix_type = "auto" if auto else "user override"
    for s in IS.ALL_STRATEGIES:
        if s == selected:
            rationale[s] = f"SELECTED ({prefix_type}). " + rationale[s]
        else:
            rationale[s] = "Not selected. " + rationale[s]

    return rationale


def load_canonical_scores(canonical_dir: Path | str, chamber: str) -> pl.DataFrame | None:
    """Load canonical routing ideal points for a chamber.

    Args:
        canonical_dir: Path to the canonical_irt directory
            (e.g., .../06_irt_2d/canonical_irt/).
        chamber: "house" or "senate" (lowercase).

    Returns:
        DataFrame with legislator_slug, xi_mean, and source columns, or None.
    """
    path = Path(canonical_dir) / f"canonical_ideal_points_{chamber}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def load_2d_scores(irt_2d_data_dir: Path | str, chamber: str) -> pl.DataFrame | None:
    """Load 2D IRT ideal points for a chamber from Phase 06 output.

    Args:
        irt_2d_data_dir: Path to the 2D IRT data directory (e.g., .../06_irt_2d/data/).
        chamber: "house" or "senate" (lowercase).

    Returns:
        DataFrame with legislator_slug and xi_dim1_mean columns, or None if not found.
    """
    path = Path(irt_2d_data_dir) / f"ideal_points_2d_{chamber}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


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
