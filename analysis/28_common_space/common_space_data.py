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
# Person identity resolution
# ---------------------------------------------------------------------------


def _slug_to_person_key(slug: str) -> str:
    """Extract person identity key from legislator slug.

    Strips the chamber prefix (rep_/sen_) so the same person in different
    chambers gets the same key: rep_smith_greg_1 → smith_greg_1.
    """
    parts = slug.split("_", 1)
    return parts[1] if len(parts) > 1 else slug


def load_slug_to_ocd() -> dict[str, str]:
    """Load the slug→OCD person ID mapping from the OpenStates cache.

    Returns empty dict if cache is missing (graceful fallback).
    """
    from pathlib import Path

    cache_path = Path("data/external/openstates/ks_slug_to_ocd.json")
    if not cache_path.exists():
        return {}
    import json

    return json.load(cache_path.open())


# OCD-level overrides for cases where OpenStates itself splits one person
# into multiple OCD IDs.  Format: {variant_ocd_id: canonical_ocd_id}
_OCD_OVERRIDES: dict[str, str] = {
    # J.R. Claeys: OpenStates has 3 OCD IDs for 1 person (different slug
    # encodings: rep_claeys_j_r_1, sen_claeys_jr_1). Editorially confirmed.
    "ocd-person/2bbf4d79-672b-46f7-8745-ee91db3169e3": (
        "ocd-person/26607b1c-8e09-4c41-a1cc-aed9bf5216d5"
    ),
    # Dan Goddard: Senate (87th-88th) then House (90th-91st). Two OCD IDs.
    "ocd-person/2d7cf122-fd6c-406c-a158-de6c61ff0afa": (
        "ocd-person/804bbf3c-f3b3-4666-a50a-d55f3716f741"
    ),
    # Ronald Ryckman Sr.: House (84th-86th) then Senate (89th-91st). Two OCD IDs.
    # (Not Ron Ryckman Jr., who is rep_ryckman_jr_ron_1, a different person.)
    "ocd-person/43caafd7-9177-40f4-b515-0681ef5930f3": (
        "ocd-person/ae978391-0baa-4dff-af69-f9d95106bb98"
    ),
}

# Slug-level overrides for sessions without OCD coverage (pre-2011 KanFocus).
# Format: {variant_person_key: canonical_person_key}
_SLUG_OVERRIDES: dict[str, str] = {
    "claeys_jr_1": "claeys_j_r_1",  # J.R. Claeys: sen slug vs rep slug
    "crum_david_1": "crum_dave_1",  # Dave Crum
    "clifford_william_1": "clifford_bill_1",  # Bill Clifford
    "pauls_janice_1": "pauls_jan_1",  # Jan Pauls
    "roth_charles_1": "roth_charlie_1",  # Charlie Roth
    "prescott_willie_1": "prescott_william_1",  # William Prescott
    "bloom_lewis_1": "bloom_bill_1",  # Lewis "Bill" Bloom
    "kelsey_richard_1": "kelsey_dick_1",  # Dick Kelsey
}


# Slug roots where different person_keys are expected — genuinely different
# people who share the same name component.  The quality gate skips these.
_SAME_NAME_DIFFERENT_PERSON: frozenset[str] = frozenset(
    {
        "thompson_mike_1",  # Rep. Mike Thompson (House) and Sen. Mike Thompson (Senate)
    }
)

_CHAMBER_PREFIXES: tuple[str, str] = ("rep_", "sen_")
"""Chamber prefixes used in Kansas Legislature slugs."""


def _other_chamber_slug(slug: str) -> str | None:
    """Return the other-chamber variant of a slug, or None if not prefixed."""
    if slug.startswith("rep_"):
        return "sen_" + slug[4:]
    if slug.startswith("sen_"):
        return "rep_" + slug[4:]
    return None


def build_person_key_lookup(slug_to_ocd: dict[str, str] | None = None) -> dict[str, str]:
    """Build a slug→person_key lookup using OCD IDs as primary identity.

    Strategy:
    1. If slug has an OCD mapping, use the OCD ID as person_key (applying
       _OCD_OVERRIDES for known OpenStates errors).
    2. Expand cross-chamber variants: if ``sen_tyson_caryn_1`` maps to an
       OCD ID, also map ``rep_tyson_caryn_1`` to the same OCD ID — unless
       that slug already has its own mapping (different person, like two
       Mike Thompsons).
    3. If no OCD mapping (pre-2011 KanFocus), fall back to slug-based key
       with _SLUG_OVERRIDES for known encoding variants.

    This correctly separates same-name legislators (e.g., two Mike Thompsons)
    who have different OCD IDs, while merging same-person slug variants
    across chamber switches.
    """
    if slug_to_ocd is None:
        slug_to_ocd = load_slug_to_ocd()

    lookup: dict[str, str] = {}
    for slug, ocd_id in slug_to_ocd.items():
        canonical_ocd = _OCD_OVERRIDES.get(ocd_id, ocd_id)
        lookup[slug] = canonical_ocd

    # Expand cross-chamber variants: if only one chamber's slug is in the
    # OCD mapping, derive the other so chamber-switchers get the same
    # person_key regardless of which slug appears in the data.
    expansions: dict[str, str] = {}
    for slug, person_key in lookup.items():
        alt = _other_chamber_slug(slug)
        if alt is not None and alt not in lookup and alt not in expansions:
            expansions[alt] = person_key
    lookup.update(expansions)

    return lookup


# Module-level cache, populated by build_global_roster()
_person_key_lookup: dict[str, str] = {}


def resolve_person_key(slug: str) -> str:
    """Resolve a legislator slug to a stable person identity key.

    Uses OCD person ID when available (correctly separates same-name
    legislators like two Mike Thompsons). Falls back to slug-based key
    with overrides for pre-2011 sessions without OCD coverage.
    """
    # Try OCD-based lookup first
    if _person_key_lookup and slug in _person_key_lookup:
        return _person_key_lookup[slug]

    # Fallback: slug-based key with overrides
    raw = _slug_to_person_key(slug)
    return _SLUG_OVERRIDES.get(raw, raw)


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
    DataFrame with columns: person_key, name_norm, legislator_slug, full_name,
    party, session, chamber, xi_canonical, xi_sd.

    Identity resolution uses OpenStates OCD person IDs as primary key (correctly
    separates same-name legislators like two Mike Thompsons). Falls back to
    slug-based keys with overrides for pre-2011 sessions without OCD coverage.
    """
    global _person_key_lookup  # noqa: PLW0603
    slug_to_ocd = load_slug_to_ocd()
    _person_key_lookup = build_person_key_lookup(slug_to_ocd)

    rows: list[dict] = []
    for session, chambers in sorted(all_scores.items()):
        for chamber, df in chambers.items():
            for row in df.iter_rows(named=True):
                slug = row["legislator_slug"]
                rows.append(
                    {
                        "person_key": resolve_person_key(slug),
                        "name_norm": normalize_fn(row["full_name"]),
                        "legislator_slug": slug,
                        "full_name": row["full_name"],
                        "party": row.get("party", ""),
                        "session": session,
                        "chamber": chamber,
                        "xi_canonical": row["xi_mean"],
                        "xi_sd": row.get("xi_sd", 0.0),
                    }
                )
    return pl.DataFrame(rows)


def detect_potential_duplicates(roster: pl.DataFrame) -> pl.DataFrame:
    """Detect person_keys that likely refer to the same legislator.

    Checks for different person_keys that share the same slug root (the part
    after stripping ``rep_``/``sen_``). This catches chamber-switch orphans
    that slipped past OCD expansion — e.g., one person_key is an OCD ID
    (from a mapped slug) and another is a slug-based fallback key (from an
    unmapped variant).

    Slug roots listed in :data:`_SAME_NAME_DIFFERENT_PERSON` are excluded
    (genuinely different people who happen to share a slug root).

    Returns a DataFrame with columns: ``slug_root``, ``person_keys``,
    ``full_names``, ``sessions``.  Empty if no duplicates found.
    """
    enriched = roster.with_columns(
        pl.col("legislator_slug")
        .map_elements(_slug_to_person_key, return_dtype=pl.Utf8)
        .alias("slug_root"),
    )

    # Group by slug_root, collect distinct person_keys
    grouped = (
        enriched.select("slug_root", "person_key", "full_name", "session")
        .unique(subset=["slug_root", "person_key"])
        .group_by("slug_root")
        .agg(
            pl.col("person_key").n_unique().alias("n_keys"),
            pl.col("person_key").unique().alias("person_keys"),
            pl.col("full_name").unique().alias("full_names"),
            pl.col("session").unique().alias("sessions"),
        )
        .filter(pl.col("n_keys") > 1)
        .filter(~pl.col("slug_root").is_in(list(_SAME_NAME_DIFFERENT_PERSON)))
        .sort("slug_root")
    )

    return grouped


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
        names_a = set(roster.filter(pl.col("session") == sa)["person_key"].to_list())
        for j, sb in enumerate(sessions):
            if j <= i:
                continue
            names_b = set(roster.filter(pl.col("session") == sb)["person_key"].to_list())
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
    Returns DataFrame: person_key, session_s, session_t, xi_s, xi_t.
    """
    chamber_roster = roster.filter(pl.col("chamber") == chamber)

    # Pivot to wide: one row per (person_key, session) with xi_canonical
    wide = chamber_roster.select("person_key", "session", "xi_canonical").unique(
        subset=["person_key", "session"]
    )

    obs_rows: list[dict] = []
    session_set = set(sessions)

    # Group by legislator, find all session pairs
    for name, group in wide.group_by("person_key"):
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
                        "person_key": name_val,
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
                "person_key": pl.Utf8,
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
        .select("person_key", "xi_canonical")
        .rename({"xi_canonical": "xi_a"})
    )
    b_scores = (
        chamber_roster.filter(pl.col("session") == session_b)
        .select("person_key", "xi_canonical")
        .rename({"xi_canonical": "xi_b"})
    )
    bridges = a_scores.join(b_scores, on="person_key", how="inner")

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


# ---------------------------------------------------------------------------
# Career scores (random-effects meta-analysis)
# ---------------------------------------------------------------------------

I_SQUARED_STABLE: float = 0.25
"""I² below this → legislator classified as 'stable'."""

I_SQUARED_MOVER: float = 0.75
"""I² above this → legislator classified as 'mover'."""


def compute_career_scores(
    transformed: pl.DataFrame,
    chamber: str,
) -> pl.DataFrame:
    """Compute one career-fixed score per legislator via RE meta-analysis.

    Uses DerSimonian-Laird random-effects pooling of per-session
    common-space scores. Each session's xi_common_sd provides the
    within-session variance; tau² captures genuine between-session
    ideological movement.

    Returns one row per legislator with career_score, career_se,
    i_squared, tau_squared, and movement_flag.
    """
    chamber_data = transformed.filter(pl.col("chamber") == chamber)
    rows: list[dict] = []

    for person_key, group in chamber_data.group_by("person_key"):
        name_val = person_key[0]
        group_sorted = group.sort("session")
        T = group_sorted.height

        last_row = group_sorted.row(-1, named=True)
        first_row = group_sorted.row(0, named=True)

        base = {
            "person_key": name_val,
            "full_name": last_row["full_name"],
            "party": last_row["party"],
            "chamber": chamber,
            "n_sessions": T,
            "first_session": first_row["session"],
            "last_session": last_row["session"],
            "most_recent_score": last_row["xi_common"],
            "most_recent_se": last_row.get("xi_common_sd", 0.0),
        }

        if T == 1:
            rows.append(
                {
                    **base,
                    "career_score": last_row["xi_common"],
                    "career_se": last_row.get("xi_common_sd", 0.0) or 0.0,
                    "i_squared": None,
                    "tau_squared": None,
                    "movement_flag": None,
                }
            )
            continue

        x = group_sorted["xi_common"].to_numpy()
        sd = group_sorted["xi_common_sd"].to_numpy()
        sd = np.maximum(sd, 1e-6)
        var = sd**2

        # Fixed-effect weights and pooled mean
        w = 1.0 / var
        w_sum = np.sum(w)
        mu_fe = np.sum(w * x) / w_sum

        # Cochran's Q
        Q = float(np.sum(w * (x - mu_fe) ** 2))
        df = T - 1

        # I²
        i_sq = max(0.0, (Q - df) / Q) if Q > 0 else 0.0

        # DerSimonian-Laird tau²
        c = w_sum - np.sum(w**2) / w_sum
        tau_sq = max(0.0, (Q - df) / c) if c > 0 else 0.0

        # Random-effects weights and pooled mean
        w_re = 1.0 / (var + tau_sq)
        w_re_sum = np.sum(w_re)
        mu_re = float(np.sum(w_re * x) / w_re_sum)
        se_re = float(np.sqrt(1.0 / w_re_sum))

        if i_sq < I_SQUARED_STABLE:
            flag = "stable"
        elif i_sq > I_SQUARED_MOVER:
            flag = "mover"
        else:
            flag = "moderate"

        rows.append(
            {
                **base,
                "career_score": mu_re,
                "career_se": se_re,
                "i_squared": round(i_sq, 4),
                "tau_squared": round(tau_sq, 4),
                "movement_flag": flag,
            }
        )

    if not rows:
        return pl.DataFrame()

    result = pl.DataFrame(rows)
    return result.with_columns(
        (pl.col("career_score") - 1.96 * pl.col("career_se")).alias("career_lo"),
        (pl.col("career_score") + 1.96 * pl.col("career_se")).alias("career_hi"),
    ).sort("career_score", descending=True)


# ---------------------------------------------------------------------------
# Cross-chamber unification
# ---------------------------------------------------------------------------


def link_chambers(
    house_transformed: pl.DataFrame,
    senate_transformed: pl.DataFrame,
    trim_pct: int = TRIM_PCT,
) -> tuple[pl.DataFrame, float, float]:
    """Link House and Senate common-space scales via chamber-switchers.

    Uses the same affine approach as cross-session linking: regress Senate
    scores on House scores for legislators who served in both chambers.
    Senate scores are mapped onto the House scale.

    Returns (unified_df, A, B) where xi_unified = A * xi_senate + B for
    Senate legislators, and xi_unified = xi_common for House legislators.
    """
    # Find chamber-switchers: legislators with scores in both chambers
    house_by_name = house_transformed.group_by("person_key").agg(
        pl.col("xi_common").mean().alias("xi_house_mean")
    )
    senate_by_name = senate_transformed.group_by("person_key").agg(
        pl.col("xi_common").mean().alias("xi_senate_mean")
    )
    bridges = house_by_name.join(senate_by_name, on="person_key")

    if bridges.height < 5:
        # Not enough bridges — return identity transform
        unified = pl.concat(
            [
                house_transformed.with_columns(pl.col("xi_common").alias("xi_unified")),
                senate_transformed.with_columns(pl.col("xi_common").alias("xi_unified")),
            ]
        )
        return unified, 1.0, 0.0

    x_senate = bridges["xi_senate_mean"].to_numpy()
    y_house = bridges["xi_house_mean"].to_numpy()

    # Trimmed regression (same as cross-session linking)
    residuals = y_house - x_senate
    lower = np.percentile(residuals, trim_pct)
    upper = np.percentile(residuals, 100 - trim_pct)
    mask = (residuals >= lower) & (residuals <= upper)
    x_trim = x_senate[mask]
    y_trim = y_house[mask]

    if len(x_trim) < 3:
        x_trim, y_trim = x_senate, y_house

    # OLS: y_house = A * x_senate + B
    X = np.column_stack([x_trim, np.ones(len(x_trim))])
    coeffs, _, _, _ = np.linalg.lstsq(X, y_trim, rcond=None)
    A, B = float(coeffs[0]), float(coeffs[1])

    # Transform Senate scores onto House scale
    senate_unified = senate_transformed.with_columns(
        (pl.col("xi_common") * A + B).alias("xi_unified"),
        (pl.col("xi_common_sd") * abs(A)).alias("xi_unified_sd"),
    )
    house_unified = house_transformed.with_columns(
        pl.col("xi_common").alias("xi_unified"),
        pl.col("xi_common_sd").alias("xi_unified_sd"),
    )

    unified = pl.concat([house_unified, senate_unified])
    return unified, A, B


def compute_unified_career_scores(
    unified: pl.DataFrame,
) -> pl.DataFrame:
    """Compute one career score per legislator, pooling across both chambers.

    Same DerSimonian-Laird RE meta-analysis as per-chamber career scores,
    but using xi_unified (cross-chamber-linked) scores.
    """
    rows: list[dict] = []

    for person_key, group in unified.group_by("person_key"):
        name_val = person_key[0]
        group_sorted = group.sort("session")
        T = group_sorted.height

        last_row = group_sorted.row(-1, named=True)
        first_row = group_sorted.row(0, named=True)

        chambers_served = sorted(group_sorted["chamber"].unique().to_list())
        chamber_str = " & ".join(chambers_served)

        base = {
            "person_key": name_val,
            "full_name": last_row["full_name"],
            "party": last_row["party"],
            "chambers": chamber_str,
            "n_sessions": T,
            "first_session": first_row["session"],
            "last_session": last_row["session"],
            "most_recent_score": last_row["xi_unified"],
            "most_recent_chamber": last_row["chamber"],
        }

        if T == 1:
            rows.append(
                {
                    **base,
                    "career_score": last_row["xi_unified"],
                    "career_se": last_row.get("xi_unified_sd", 0.0) or 0.0,
                    "i_squared": None,
                    "tau_squared": None,
                    "movement_flag": None,
                }
            )
            continue

        x = group_sorted["xi_unified"].to_numpy()
        sd = group_sorted["xi_unified_sd"].to_numpy()
        sd = np.maximum(sd, 1e-6)
        var = sd**2

        w = 1.0 / var
        w_sum = np.sum(w)
        mu_fe = np.sum(w * x) / w_sum

        Q = float(np.sum(w * (x - mu_fe) ** 2))
        df = T - 1

        i_sq = max(0.0, (Q - df) / Q) if Q > 0 else 0.0

        c = w_sum - np.sum(w**2) / w_sum
        tau_sq = max(0.0, (Q - df) / c) if c > 0 else 0.0

        w_re = 1.0 / (var + tau_sq)
        w_re_sum = np.sum(w_re)
        mu_re = float(np.sum(w_re * x) / w_re_sum)
        se_re = float(np.sqrt(1.0 / w_re_sum))

        if i_sq < I_SQUARED_STABLE:
            flag = "stable"
        elif i_sq > I_SQUARED_MOVER:
            flag = "mover"
        else:
            flag = "moderate"

        rows.append(
            {
                **base,
                "career_score": mu_re,
                "career_se": se_re,
                "i_squared": round(i_sq, 4),
                "tau_squared": round(tau_sq, 4),
                "movement_flag": flag,
            }
        )

    if not rows:
        return pl.DataFrame()

    result = pl.DataFrame(rows)
    return result.with_columns(
        (pl.col("career_score") - 1.96 * pl.col("career_se")).alias("career_lo"),
        (pl.col("career_score") + 1.96 * pl.col("career_se")).alias("career_hi"),
    ).sort("career_score", descending=True)
