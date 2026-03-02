"""IRT scale linking for cross-chamber ideal point alignment.

Implements Stocking-Lord, Haebara, Mean-Sigma, and Mean-Mean linking
to place per-chamber hierarchical IRT ideal points on a common scale
using shared (anchor) bills.

The linking approach is an alternative to concurrent calibration (the joint
model). It uses well-converged per-chamber posterior estimates and a simple
2-parameter optimization to align scales, sidestepping the MCMC convergence
problems that plague the joint model.

See docs/joint-model-deep-dive.md for the full analysis.
"""

import numpy as np
from scipy.optimize import minimize


def icc_2pl(theta: float, a: float, b: float) -> float:
    """2PL item characteristic curve: P(Yea | theta, a, b).

    Using our parameterization: logit(P) = a * theta - b
    where a = discrimination (beta), b = difficulty (alpha).
    """
    return 1.0 / (1.0 + np.exp(-(a * theta - b)))


def extract_anchor_params(
    house_idata,
    senate_idata,
    matched_bills: list[dict],
    house_vote_ids: list[str],
    senate_vote_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract posterior mean alpha/beta for shared bills from each chamber.

    Our per-chamber models use beta ~ Normal(0, 1), so discrimination can be
    negative. Standard IRT linking assumes positive discrimination. We resolve
    this by:
      1. Taking |beta| as discrimination (flipping alpha sign when beta < 0
         to preserve the ICC: a*theta - b is invariant under (a, b) -> (-a, -b)).
      2. Filtering to items where both chambers agree on sign (same bill should
         discriminate in the same direction across chambers).

    Args:
        house_idata: House per-chamber InferenceData.
        senate_idata: Senate per-chamber InferenceData.
        matched_bills: List of dicts with bill_number, house_vote_id, senate_vote_id.
        house_vote_ids: Full list of House vote_ids (for index lookup).
        senate_vote_ids: Full list of Senate vote_ids (for index lookup).

    Returns:
        (a_house, b_house, a_senate, b_senate, n_dropped) where arrays have
        length <= n_shared (items with sign disagreement are excluded) and
        n_dropped is the count of excluded items.
    """
    house_vid_to_idx = {v: i for i, v in enumerate(house_vote_ids)}
    senate_vid_to_idx = {v: i for i, v in enumerate(senate_vote_ids)}

    a_house, b_house = [], []
    a_senate, b_senate = [], []
    n_dropped = 0

    for m in matched_bills:
        hvid, svid = m["house_vote_id"], m["senate_vote_id"]
        if hvid not in house_vid_to_idx or svid not in senate_vid_to_idx:
            n_dropped += 1
            continue
        h_idx = house_vid_to_idx[hvid]
        s_idx = senate_vid_to_idx[svid]

        # Posterior means of bill parameters (raw, may be negative)
        a_h = float(house_idata.posterior["beta"].isel(vote=h_idx).mean())
        b_h = float(house_idata.posterior["alpha"].isel(vote=h_idx).mean())
        a_s = float(senate_idata.posterior["beta"].isel(vote=s_idx).mean())
        b_s = float(senate_idata.posterior["alpha"].isel(vote=s_idx).mean())

        # Filter: both chambers must agree on sign direction
        if (a_h > 0) != (a_s > 0):
            n_dropped += 1
            continue

        # Normalize to positive discrimination (|beta|, flip alpha if needed)
        if a_h < 0:
            a_h, b_h = -a_h, -b_h
            a_s, b_s = -a_s, -b_s

        a_house.append(a_h)
        b_house.append(b_h)
        a_senate.append(a_s)
        b_senate.append(b_s)

    return (
        np.array(a_house),
        np.array(b_house),
        np.array(a_senate),
        np.array(b_senate),
        n_dropped,
    )


def link_mean_mean(
    a_ref: np.ndarray,
    b_ref: np.ndarray,
    a_target: np.ndarray,
    b_target: np.ndarray,
) -> tuple[float, float]:
    """Mean-Mean linking coefficients.

    Transforms target scale -> reference scale.
    A = mean(a_ref) / mean(a_target)
    B = mean(b_ref) - A * mean(b_target)
    """
    denom = np.mean(a_target)
    if denom == 0:
        msg = "Mean-Mean linking: mean(a_target) is zero (degenerate anchor items)"
        raise ValueError(msg)
    A = np.mean(a_ref) / denom
    B = np.mean(b_ref) - A * np.mean(b_target)
    return float(A), float(B)


def link_mean_sigma(
    a_ref: np.ndarray,
    b_ref: np.ndarray,
    a_target: np.ndarray,
    b_target: np.ndarray,
) -> tuple[float, float]:
    """Mean-Sigma linking coefficients.

    Uses SD of difficulty (b) parameters only.
    A = sd(b_ref) / sd(b_target)
    B = mean(b_ref) - A * mean(b_target)
    """
    denom = np.std(b_target, ddof=1)
    if denom == 0:
        msg = "Mean-Sigma linking: sd(b_target) is zero (degenerate anchor items)"
        raise ValueError(msg)
    A = np.std(b_ref, ddof=1) / denom
    B = np.mean(b_ref) - A * np.mean(b_target)
    return float(A), float(B)


def _stocking_lord_loss(
    params: np.ndarray,
    a_ref: np.ndarray,
    b_ref: np.ndarray,
    a_target: np.ndarray,
    b_target: np.ndarray,
    theta_grid: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Stocking-Lord criterion: squared diff of test characteristic curves."""
    A, B = params

    # Transform target params to reference scale
    a_trans = a_target / A
    b_trans = A * b_target + B

    loss = 0.0
    for theta, w in zip(theta_grid, weights):
        tcc_ref = sum(icc_2pl(theta, a_ref[j], b_ref[j]) for j in range(len(a_ref)))
        tcc_trans = sum(icc_2pl(theta, a_trans[j], b_trans[j]) for j in range(len(a_trans)))
        loss += w * (tcc_ref - tcc_trans) ** 2

    return loss


def link_stocking_lord(
    a_ref: np.ndarray,
    b_ref: np.ndarray,
    a_target: np.ndarray,
    b_target: np.ndarray,
    n_points: int = 81,
    theta_range: tuple[float, float] = (-4.0, 4.0),
) -> tuple[float, float]:
    """Stocking-Lord linking via TCC matching.

    Minimizes squared difference between reference and transformed
    test characteristic curves over a grid of theta values weighted
    by a standard normal density.
    """
    theta_grid = np.linspace(theta_range[0], theta_range[1], n_points)
    weights = np.exp(-0.5 * theta_grid**2)
    weights /= weights.sum()

    # Initialize with Mean-Sigma
    A0, B0 = link_mean_sigma(a_ref, b_ref, a_target, b_target)

    result = minimize(
        _stocking_lord_loss,
        x0=[A0, B0],
        args=(a_ref, b_ref, a_target, b_target, theta_grid, weights),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 10000},
    )

    return float(result.x[0]), float(result.x[1])


def _haebara_loss(
    params: np.ndarray,
    a_ref: np.ndarray,
    b_ref: np.ndarray,
    a_target: np.ndarray,
    b_target: np.ndarray,
    theta_grid: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Haebara criterion: sum of squared ICC differences per item."""
    A, B = params

    a_trans = a_target / A
    b_trans = A * b_target + B

    loss = 0.0
    for j in range(len(a_ref)):
        for theta, w in zip(theta_grid, weights):
            p_ref = icc_2pl(theta, a_ref[j], b_ref[j])
            p_trans = icc_2pl(theta, a_trans[j], b_trans[j])
            loss += w * (p_ref - p_trans) ** 2

    return loss


def link_haebara(
    a_ref: np.ndarray,
    b_ref: np.ndarray,
    a_target: np.ndarray,
    b_target: np.ndarray,
    n_points: int = 81,
    theta_range: tuple[float, float] = (-4.0, 4.0),
) -> tuple[float, float]:
    """Haebara linking via ICC matching.

    Same as Stocking-Lord but operates at the item level:
    sums squared differences per item, not on aggregated TCC.
    """
    theta_grid = np.linspace(theta_range[0], theta_range[1], n_points)
    weights = np.exp(-0.5 * theta_grid**2)
    weights /= weights.sum()

    A0, B0 = link_mean_sigma(a_ref, b_ref, a_target, b_target)

    result = minimize(
        _haebara_loss,
        x0=[A0, B0],
        args=(a_ref, b_ref, a_target, b_target, theta_grid, weights),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 10000},
    )

    return float(result.x[0]), float(result.x[1])


LINKING_METHODS = {
    "stocking_lord": link_stocking_lord,
    "haebara": link_haebara,
    "mean_sigma": link_mean_sigma,
    "mean_mean": link_mean_mean,
}


def link_chambers(
    house_idata,
    senate_idata,
    matched_bills: list[dict],
    house_vote_ids: list[str],
    senate_vote_ids: list[str],
    method: str = "stocking_lord",
    reference: str = "House",
) -> dict:
    """Compute linking coefficients and transform ideal points to a common scale.

    Uses anchor items (shared bills) to find the affine transformation
    xi_linked = A * xi_target + B that places the target chamber's ideal
    points on the reference chamber's scale.

    Args:
        house_idata: House per-chamber InferenceData.
        senate_idata: Senate per-chamber InferenceData.
        matched_bills: Shared bills from _match_bills_across_chambers().
        house_vote_ids: Full list of House vote_ids.
        senate_vote_ids: Full list of Senate vote_ids.
        method: Linking method. One of: stocking_lord, haebara, mean_sigma, mean_mean.
        reference: Which chamber is the reference scale ("House" or "Senate").

    Returns dict with:
        A, B: Linking coefficients.
        method: Method used.
        reference: Reference chamber.
        n_anchors: Number of anchor items used.
        xi_house_linked: House xi posterior means on common scale.
        xi_senate_linked: Senate xi posterior means on common scale.
        xi_house_sd: House xi posterior SDs.
        xi_senate_sd: Senate xi posterior SDs (transformed).
    """
    link_fn = LINKING_METHODS[method]

    a_house, b_house, a_senate, b_senate, n_dropped = extract_anchor_params(
        house_idata, senate_idata, matched_bills, house_vote_ids, senate_vote_ids
    )

    if reference == "House":
        A, B = link_fn(a_house, b_house, a_senate, b_senate)
        # Senate -> House scale
        xi_house = house_idata.posterior["xi"].mean(dim=("chain", "draw")).values
        xi_senate_raw = senate_idata.posterior["xi"].mean(dim=("chain", "draw")).values
        xi_senate_linked = A * xi_senate_raw + B

        xi_house_sd = house_idata.posterior["xi"].std(dim=("chain", "draw")).values
        xi_senate_sd = np.abs(A) * senate_idata.posterior["xi"].std(dim=("chain", "draw")).values
    else:
        A, B = link_fn(a_senate, b_senate, a_house, b_house)
        # House -> Senate scale
        xi_senate = senate_idata.posterior["xi"].mean(dim=("chain", "draw")).values
        xi_house_raw = house_idata.posterior["xi"].mean(dim=("chain", "draw")).values
        xi_house_linked = A * xi_house_raw + B
        xi_senate_linked = xi_senate

        xi_senate_sd = senate_idata.posterior["xi"].std(dim=("chain", "draw")).values
        xi_house_sd = np.abs(A) * house_idata.posterior["xi"].std(dim=("chain", "draw")).values
        xi_house = xi_house_linked

    return {
        "A": A,
        "B": B,
        "method": method,
        "reference": reference,
        "n_anchors": len(matched_bills),
        "n_usable": len(a_house),
        "n_sign_disagreement": n_dropped,
        "xi_house_linked": xi_house if reference == "House" else xi_house_linked,
        "xi_senate_linked": xi_senate_linked,
        "xi_house_sd": xi_house_sd,
        "xi_senate_sd": xi_senate_sd,
    }


def compare_linking_methods(
    house_idata,
    senate_idata,
    matched_bills: list[dict],
    house_vote_ids: list[str],
    senate_vote_ids: list[str],
    reference: str = "House",
) -> dict[str, dict]:
    """Run all four linking methods and return results for comparison.

    This is a sensitivity check: if all methods agree on (A, B), the
    linking is robust. Large discrepancies signal DIF problems.
    """
    results = {}
    for method_name in LINKING_METHODS:
        results[method_name] = link_chambers(
            house_idata,
            senate_idata,
            matched_bills,
            house_vote_ids,
            senate_vote_ids,
            method=method_name,
            reference=reference,
        )
    return results
