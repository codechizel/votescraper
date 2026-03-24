"""Common Space Ideal Points — pure data logic.

Simultaneous affine alignment of canonical ideal points across bienniums
using bridge legislators. No I/O — all functions take and return DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

REFERENCE_SESSION: str = "91st_2025-2026"
"""Default reference biennium (most recent, best convergence)."""

MIN_BRIDGES: int = 20
"""Minimum bridge legislators for a reliable pairwise link."""

TRIM_PCT: int = 10
"""Trim top/bottom N% residuals from alignment (genuine movers)."""

N_BOOTSTRAP: int = 1000
"""Bootstrap iterations for uncertainty quantification."""

BOOTSTRAP_SEED: int = 42
"""Reproducibility seed for bootstrap resampling."""

CORRELATION_WARN: float = 0.70
"""Warn if any pairwise bridge correlation falls below this."""

PARTY_D_MIN: float = 1.5
"""Minimum party separation (Cohen's d) on aligned scale."""

MIN_SERVED_FOR_TRAJECTORY: int = 3
"""Minimum bienniums served to appear in career trajectory plots."""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinkingCoefficients:
    """Affine transformation parameters for one session."""

    session: str
    A: float
    B: float
    A_lo: float
    A_hi: float
    B_lo: float
    B_hi: float


@dataclass(frozen=True)
class BootstrapStats:
    """Per-session bootstrap statistics for uncertainty propagation.

    Stores Var(A), Var(B), Cov(A,B) needed for the delta method:
      Var(A*xi + B) = xi^2 * var_A + var_B + 2*xi*cov_AB
    """

    session: str
    var_A: float
    var_B: float
    cov_AB: float
    A_lo: float
    A_hi: float
    B_lo: float
    B_hi: float


@dataclass(frozen=True)
class QualityGate:
    """Quality gate result for one chamber-session."""

    session: str
    chamber: str
    party_d: float
    bridge_r: float | None
    sign_ok: bool
    passed: bool


# ---------------------------------------------------------------------------
# Global roster
# ---------------------------------------------------------------------------


def build_global_roster(
    all_scores: dict[str, dict[str, pl.DataFrame]],
    normalize_fn: callable,
) -> pl.DataFrame:
    """Build a unified roster of all legislators across all bienniums.

    Parameters
    ----------
    all_scores
        Nested dict: session -> chamber -> DataFrame with columns
        legislator_slug, full_name, party, xi_mean (canonical ideal point).
    normalize_fn
        Name normalization function (e.g., phase_utils.normalize_name).

    Returns
    -------
    DataFrame with columns: name_norm, legislator_slug, full_name, party,
    session, chamber, xi_canonical, xi_sd.
    """
    rows: list[dict] = []
    for session, chambers in sorted(all_scores.items()):
        for chamber, df in chambers.items():
            for row in df.iter_rows(named=True):
                rows.append(
                    {
                        "name_norm": normalize_fn(row["full_name"]),
                        "legislator_slug": row["legislator_slug"],
                        "full_name": row["full_name"],
                        "party": row.get("party", ""),
                        "session": session,
                        "chamber": chamber,
                        "xi_canonical": row["xi_mean"],
                        "xi_sd": row.get("xi_sd", 0.0),
                    }
                )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bridge matrix
# ---------------------------------------------------------------------------


def compute_bridge_matrix(
    roster: pl.DataFrame,
    sessions: list[str],
) -> pl.DataFrame:
    """Compute pairwise bridge counts between all session pairs.

    Returns a long-form DataFrame with columns: session_a, session_b, n_bridges.
    """
    rows: list[dict] = []
    for i, sa in enumerate(sessions):
        names_a = set(roster.filter(pl.col("session") == sa)["name_norm"].to_list())
        for j, sb in enumerate(sessions):
            if j <= i:
                continue
            names_b = set(roster.filter(pl.col("session") == sb)["name_norm"].to_list())
            shared = len(names_a & names_b)
            rows.append({"session_a": sa, "session_b": sb, "n_bridges": shared})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simultaneous affine alignment
# ---------------------------------------------------------------------------


def _build_bridge_observations(
    roster: pl.DataFrame,
    sessions: list[str],
    chamber: str,
) -> pl.DataFrame:
    """Extract all bridge observations for one chamber.

    A bridge observation is a legislator who served in two different sessions.
    Returns DataFrame: name_norm, session_s, session_t, xi_s, xi_t.
    """
    chamber_roster = roster.filter(pl.col("chamber") == chamber)

    # Pivot to wide: one row per (name_norm, session) with xi_canonical
    wide = chamber_roster.select("name_norm", "session", "xi_canonical").unique(
        subset=["name_norm", "session"]
    )

    obs_rows: list[dict] = []
    session_set = set(sessions)

    # Group by legislator, find all session pairs
    for name, group in wide.group_by("name_norm"):
        name_val = name[0]
        leg_sessions = [
            (r["session"], r["xi_canonical"])
            for r in group.iter_rows(named=True)
            if r["session"] in session_set
        ]
        for i in range(len(leg_sessions)):
            for j in range(i + 1, len(leg_sessions)):
                s_sess, s_xi = leg_sessions[i]
                t_sess, t_xi = leg_sessions[j]
                obs_rows.append(
                    {
                        "name_norm": name_val,
                        "session_s": s_sess,
                        "session_t": t_sess,
                        "xi_s": s_xi,
                        "xi_t": t_xi,
                    }
                )
    return (
        pl.DataFrame(obs_rows)
        if obs_rows
        else pl.DataFrame(
            schema={
                "name_norm": pl.Utf8,
                "session_s": pl.Utf8,
                "session_t": pl.Utf8,
                "xi_s": pl.Float64,
                "xi_t": pl.Float64,
            }
        )
    )


def _solve_pairwise_link(
    roster: pl.DataFrame,
    session_a: str,
    session_b: str,
    chamber: str,
    trim_pct: int = TRIM_PCT,
) -> tuple[float, float]:
    """Estimate affine (A, B) mapping session_a scores to session_b scale.

    xi_b[i] = A * xi_a[i] + B  for bridge legislators i.

    Returns (A, B). Falls back to (1.0, 0.0) if insufficient bridges.
    """
    chamber_roster = roster.filter(pl.col("chamber") == chamber)
    a_scores = (
        chamber_roster.filter(pl.col("session") == session_a)
        .select("name_norm", "xi_canonical")
        .rename({"xi_canonical": "xi_a"})
    )
    b_scores = (
        chamber_roster.filter(pl.col("session") == session_b)
        .select("name_norm", "xi_canonical")
        .rename({"xi_canonical": "xi_b"})
    )
    bridges = a_scores.join(b_scores, on="name_norm", how="inner")

    if bridges.height < MIN_BRIDGES:
        return (1.0, 0.0)

    xi_a = bridges["xi_a"].to_numpy()
    xi_b = bridges["xi_b"].to_numpy()

    # OLS: xi_b = A * xi_a + B
    X = np.column_stack([xi_a, np.ones(len(xi_a))])
    params, _, _, _ = np.linalg.lstsq(X, xi_b, rcond=None)

    # Trimmed re-fit
    if trim_pct > 0 and len(xi_a) > 10:
        residuals = X @ params - xi_b
        abs_resid = np.abs(residuals)
        threshold = np.percentile(abs_resid, 100 - trim_pct)
        keep = abs_resid <= threshold
        if np.sum(keep) >= 5:
            X_trim = X[keep]
            y_trim = xi_b[keep]
            params, _, _, _ = np.linalg.lstsq(X_trim, y_trim, rcond=None)

    return (float(params[0]), float(params[1]))


def solve_simultaneous_alignment(
    roster: pl.DataFrame,
    sessions: list[str],
    chamber: str,
    reference: str,
    trim_pct: int = TRIM_PCT,
) -> dict[str, tuple[float, float]]:
    """Chained pairwise affine alignment (Battauz 2023, GLS 1999).

    For each adjacent pair, estimates (A, B) via trimmed OLS on bridge
    legislators. Chains the transformations to map every session onto the
    reference scale.

    Despite the function name (kept for API compatibility), this uses
    pairwise chaining, not simultaneous least-squares. The simultaneous
    approach produced degenerate coefficients — see ADR-0120.

    Parameters
    ----------
    roster
        Global roster DataFrame.
    sessions
        Ordered list of session names (chronological).
    chamber
        "House" or "Senate".
    reference
        Reference session name (A=1, B=0).
    trim_pct
        Percentage of extreme residuals to trim per link.

    Returns
    -------
    Dict mapping session -> (A_total, B_total) to reach the reference scale.
    """
    if len(sessions) < 2:
        return {sessions[0]: (1.0, 0.0)} if sessions else {}

    # Step 1: Compute pairwise links between adjacent sessions
    # Link maps session[i] → session[i+1]
    pairwise: dict[tuple[str, str], tuple[float, float]] = {}
    for i in range(len(sessions) - 1):
        sa, sb = sessions[i], sessions[i + 1]
        A, B = _solve_pairwise_link(roster, sa, sb, chamber, trim_pct)
        pairwise[(sa, sb)] = (A, B)

    # Step 2: Find the reference session index
    ref_idx = sessions.index(reference) if reference in sessions else len(sessions) - 1

    # Step 3: Chain forward from each session to the reference
    coefficients: dict[str, tuple[float, float]] = {reference: (1.0, 0.0)}

    # Sessions before reference: chain forward (t → t+1 → ... → ref)
    for i in range(ref_idx - 1, -1, -1):
        # Compose: map session[i] → session[i+1], then session[i+1] → ref
        A_link, B_link = pairwise[(sessions[i], sessions[i + 1])]
        A_next, B_next = coefficients[sessions[i + 1]]
        # Composition: f(g(x)) = A_next * (A_link * x + B_link) + B_next
        A_total = A_next * A_link
        B_total = A_next * B_link + B_next
        coefficients[sessions[i]] = (A_total, B_total)

    # Sessions after reference: chain backward (ref → ... → t)
    # We need the inverse: map session[i+1] → session[i] is the link,
    # but we need session[i+1] → ref. Since ref → session[i+1] means
    # we need to invert the link direction.
    for i in range(ref_idx, len(sessions) - 1):
        # Link maps session[i] → session[i+1], so to go from
        # session[i+1] to session[i]'s scale: invert the link.
        A_link, B_link = pairwise[(sessions[i], sessions[i + 1])]
        if abs(A_link) < 1e-10:
            coefficients[sessions[i + 1]] = (1.0, 0.0)
            continue
        # Inverse: x = (y - B_link) / A_link → A_inv = 1/A_link, B_inv = -B_link/A_link
        A_inv = 1.0 / A_link
        B_inv = -B_link / A_link
        # Compose with session[i] → ref
        A_prev, B_prev = coefficients[sessions[i]]
        A_total = A_prev * A_inv
        B_total = A_prev * B_inv + B_prev
        coefficients[sessions[i + 1]] = (A_total, B_total)

    return coefficients


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_alignment_direct(
    roster: pl.DataFrame,
    sessions: list[str],
    chamber: str,
    reference: str,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    trim_pct: int = TRIM_PCT,
) -> dict[str, BootstrapStats]:
    """Bootstrap the chained alignment by resampling bridge legislators.

    For each bootstrap iteration, resamples bridge legislators at each
    pairwise link, re-estimates the chain, and records the total (A, B)
    per session. Returns Var(A), Var(B), Cov(A,B) for delta-method
    uncertainty propagation.
    """
    rng = np.random.default_rng(seed)
    non_ref = [s for s in sessions if s != reference]
    ref_idx = sessions.index(reference) if reference in sessions else len(sessions) - 1

    # Pre-extract bridge pairs for each adjacent link
    chamber_roster = roster.filter(pl.col("chamber") == chamber)
    link_data: dict[tuple[str, str], pl.DataFrame] = {}
    for i in range(len(sessions) - 1):
        sa, sb = sessions[i], sessions[i + 1]
        a_scores = (
            chamber_roster.filter(pl.col("session") == sa)
            .select("name_norm", "xi_canonical")
            .rename({"xi_canonical": "xi_a"})
        )
        b_scores = (
            chamber_roster.filter(pl.col("session") == sb)
            .select("name_norm", "xi_canonical")
            .rename({"xi_canonical": "xi_b"})
        )
        bridges = a_scores.join(b_scores, on="name_norm", how="inner")
        link_data[(sa, sb)] = bridges

    # Collect bootstrap (A_total, B_total) per session
    boot_results: dict[str, list[tuple[float, float]]] = {s: [] for s in non_ref}

    for _ in range(n_bootstrap):
        # Resample each pairwise link independently
        pairwise_boot: dict[tuple[str, str], tuple[float, float]] = {}
        for key, bridges in link_data.items():
            n = bridges.height
            if n < MIN_BRIDGES:
                pairwise_boot[key] = (1.0, 0.0)
                continue
            idx = rng.choice(n, size=n, replace=True)
            boot_bridges = bridges[idx.tolist()]
            xi_a = boot_bridges["xi_a"].to_numpy()
            xi_b = boot_bridges["xi_b"].to_numpy()
            X = np.column_stack([xi_a, np.ones(len(xi_a))])
            try:
                p, _, _, _ = np.linalg.lstsq(X, xi_b, rcond=None)
                pairwise_boot[key] = (float(p[0]), float(p[1]))
            except np.linalg.LinAlgError:
                pairwise_boot[key] = (1.0, 0.0)

        # Chain composition (same logic as solve_simultaneous_alignment)
        coefs: dict[str, tuple[float, float]] = {reference: (1.0, 0.0)}
        for i in range(ref_idx - 1, -1, -1):
            A_link, B_link = pairwise_boot.get((sessions[i], sessions[i + 1]), (1.0, 0.0))
            A_next, B_next = coefs[sessions[i + 1]]
            coefs[sessions[i]] = (A_next * A_link, A_next * B_link + B_next)
        for i in range(ref_idx, len(sessions) - 1):
            A_link, B_link = pairwise_boot.get((sessions[i], sessions[i + 1]), (1.0, 0.0))
            if abs(A_link) < 1e-10:
                coefs[sessions[i + 1]] = (1.0, 0.0)
                continue
            A_inv, B_inv = 1.0 / A_link, -B_link / A_link
            A_prev, B_prev = coefs[sessions[i]]
            coefs[sessions[i + 1]] = (A_prev * A_inv, A_prev * B_inv + B_prev)

        for s in non_ref:
            boot_results[s].append(coefs.get(s, (1.0, 0.0)))

    # Compute Var, Cov from bootstrap samples
    result: dict[str, BootstrapStats] = {
        reference: BootstrapStats(reference, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    }
    for s in non_ref:
        samples = boot_results[s]
        if samples:
            As = np.array([p[0] for p in samples])
            Bs = np.array([p[1] for p in samples])
            result[s] = BootstrapStats(
                session=s,
                var_A=float(np.var(As, ddof=1)),
                var_B=float(np.var(Bs, ddof=1)),
                cov_AB=float(np.cov(As, Bs)[0, 1]) if len(As) > 1 else 0.0,
                A_lo=float(np.percentile(As, 2.5)),
                A_hi=float(np.percentile(As, 97.5)),
                B_lo=float(np.percentile(Bs, 2.5)),
                B_hi=float(np.percentile(Bs, 97.5)),
            )
        else:
            result[s] = BootstrapStats(s, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    return result


# ---------------------------------------------------------------------------
# Score transformation
# ---------------------------------------------------------------------------


def transform_scores(
    roster: pl.DataFrame,
    coefficients: dict[str, tuple[float, float]],
    bootstrap_stats: dict[str, BootstrapStats] | None = None,
) -> pl.DataFrame:
    """Apply affine transformation with proper uncertainty propagation.

    Combines two independent sources of uncertainty via the delta method:

    1. **IRT estimation uncertainty** (xi_sd from the per-biennium posterior):
       sd_irt = |A| * xi_sd

    2. **Alignment uncertainty** (Var(A), Var(B), Cov(A,B) from bootstrap):
       sd_align = sqrt(xi^2 * Var(A) + Var(B) + 2*xi*Cov(A,B))

    3. **Combined** (independent sources, quadrature sum):
       sd_total = sqrt(sd_irt^2 + sd_align^2)

    For the reference session (A=1, B=0, Var=0), sd_total = xi_sd exactly.
    For distant sessions, alignment uncertainty dominates.

    Parameters
    ----------
    roster
        Global roster with xi_canonical and xi_sd columns.
    coefficients
        Dict mapping session -> (A, B).
    bootstrap_stats
        Optional dict mapping session -> BootstrapStats.

    Returns
    -------
    DataFrame with additional columns: xi_common, xi_common_sd, xi_common_lo,
    xi_common_hi (95% CI from sd_total).
    """
    rows: list[dict] = []
    for row in roster.iter_rows(named=True):
        session = row["session"]
        xi = row["xi_canonical"]
        xi_sd_irt = row.get("xi_sd", 0.0) or 0.0
        A, B = coefficients.get(session, (1.0, 0.0))
        xi_common = A * xi + B

        # IRT uncertainty on common scale
        sd_irt = abs(A) * xi_sd_irt

        # Alignment uncertainty via delta method
        sd_align = 0.0
        if bootstrap_stats and session in bootstrap_stats:
            bs = bootstrap_stats[session]
            var_common = xi**2 * bs.var_A + bs.var_B + 2 * xi * bs.cov_AB
            sd_align = float(np.sqrt(max(var_common, 0.0)))

        # Combined uncertainty (quadrature)
        sd_total = float(np.sqrt(sd_irt**2 + sd_align**2))

        # 95% CI
        xi_lo = xi_common - 1.96 * sd_total
        xi_hi = xi_common + 1.96 * sd_total

        rows.append(
            {
                **row,
                "xi_common": xi_common,
                "xi_common_sd": sd_total,
                "xi_common_lo": xi_lo,
                "xi_common_hi": xi_hi,
            }
        )

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------


def compute_quality_gates(
    transformed: pl.DataFrame,
    sessions: list[str],
    chamber: str,
    party_d_min: float = PARTY_D_MIN,
    correlation_warn: float = CORRELATION_WARN,
) -> list[QualityGate]:
    """Assess quality of the alignment for each session.

    Checks:
    1. Party separation (Cohen's d) >= party_d_min
    2. Sign consistency (R mean > D mean)
    """
    gates: list[QualityGate] = []
    chamber_data = transformed.filter(pl.col("chamber") == chamber)

    for session in sessions:
        session_data = chamber_data.filter(pl.col("session") == session)
        if session_data.height == 0:
            gates.append(
                QualityGate(
                    session=session,
                    chamber=chamber,
                    party_d=0.0,
                    bridge_r=None,
                    sign_ok=False,
                    passed=False,
                )
            )
            continue

        r_scores = session_data.filter(pl.col("party") == "Republican")["xi_common"].to_numpy()
        d_scores = session_data.filter(pl.col("party") == "Democrat")["xi_common"].to_numpy()

        if len(r_scores) < 2 or len(d_scores) < 2:
            gates.append(
                QualityGate(
                    session=session,
                    chamber=chamber,
                    party_d=0.0,
                    bridge_r=None,
                    sign_ok=False,
                    passed=False,
                )
            )
            continue

        # Cohen's d
        r_mean, d_mean = float(np.mean(r_scores)), float(np.mean(d_scores))
        pooled_sd = float(
            np.sqrt(
                (
                    (len(r_scores) - 1) * np.var(r_scores, ddof=1)
                    + (len(d_scores) - 1) * np.var(d_scores, ddof=1)
                )
                / (len(r_scores) + len(d_scores) - 2)
            )
        )
        party_d = abs(r_mean - d_mean) / pooled_sd if pooled_sd > 0 else 0.0

        # Sign: Republicans should be more positive than Democrats
        sign_ok = r_mean > d_mean

        passed = party_d >= party_d_min and sign_ok

        gates.append(
            QualityGate(
                session=session,
                chamber=chamber,
                party_d=round(party_d, 3),
                bridge_r=None,
                sign_ok=sign_ok,
                passed=passed,
            )
        )

    return gates


# ---------------------------------------------------------------------------
# Polarization trajectory
# ---------------------------------------------------------------------------


def compute_polarization_trajectory(
    transformed: pl.DataFrame,
    sessions: list[str],
    chamber: str,
) -> pl.DataFrame:
    """Compute party means and separation per session on the common scale.

    Returns DataFrame: session, r_mean, d_mean, party_gap, n_r, n_d.
    """
    chamber_data = transformed.filter(pl.col("chamber") == chamber)
    rows: list[dict] = []

    for session in sessions:
        sdata = chamber_data.filter(pl.col("session") == session)
        r = sdata.filter(pl.col("party") == "Republican")["xi_common"]
        d = sdata.filter(pl.col("party") == "Democrat")["xi_common"]

        r_mean = float(r.mean()) if len(r) > 0 else float("nan")
        d_mean = float(d.mean()) if len(d) > 0 else float("nan")

        rows.append(
            {
                "session": session,
                "r_mean": r_mean,
                "d_mean": d_mean,
                "party_gap": r_mean - d_mean,
                "n_r": len(r),
                "n_d": len(d),
            }
        )

    return pl.DataFrame(rows)
