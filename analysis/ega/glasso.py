"""GLASSO network estimation with EBIC model selection.

Estimates a sparse partial correlation network from a correlation matrix
using the Graphical LASSO (L1-penalized precision matrix estimation).
Model selection via Extended Bayesian Information Criterion (EBIC) with
hyperparameter gamma controlling sparsity preference.

Uses sklearn's ``graphical_lasso`` function which accepts an empirical
covariance matrix directly — no synthetic data generation needed.

References:
    Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse
    covariance estimation with the graphical lasso. Biostatistics, 9(3).

    Foygel, R., & Drton, M. (2010). Extended Bayesian information criteria
    for Gaussian graphical models. NeurIPS 23.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import graphical_lasso


@dataclass(frozen=True)
class GLASSOResult:
    """Result of GLASSO network estimation.

    Attributes:
        partial_corr: p × p sparse partial correlation matrix (edges).
        precision: p × p estimated precision (inverse covariance) matrix.
        selected_lambda: Regularization parameter selected by EBIC.
        ebic_curve: Array of (lambda, EBIC) values from the sweep.
        n_edges: Number of non-zero edges in the selected network.
    """

    partial_corr: NDArray[np.float64]
    precision: NDArray[np.float64]
    selected_lambda: float
    ebic_curve: NDArray[np.float64]
    n_edges: int


def _precision_to_partial_corr(precision: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert precision matrix to partial correlation matrix.

    partial_corr[i,j] = -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
    Diagonal is set to zero (no self-loops).
    """
    diag = np.sqrt(np.diag(precision))
    # Avoid division by zero
    diag = np.where(diag == 0, 1.0, diag)
    partial = -precision / np.outer(diag, diag)
    np.fill_diagonal(partial, 0.0)
    return partial


def _compute_ebic(
    precision: NDArray[np.float64],
    emp_cov: NDArray[np.float64],
    n: int,
    gamma: float = 0.5,
) -> float:
    """Compute Extended BIC for a Gaussian graphical model.

    EBIC = -2 * log_likelihood + k * log(n) + 4 * k * gamma * log(p)

    where k = number of non-zero off-diagonal entries (edges) / 2,
    and log_likelihood = (n/2) * (log|precision| - trace(emp_cov @ precision)).
    """
    p = precision.shape[0]
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        return np.inf

    log_lik = 0.5 * n * (logdet - np.trace(emp_cov @ precision))

    # Count non-zero edges (upper triangle only)
    upper = np.triu(precision, k=1)
    k = int(np.sum(np.abs(upper) > 1e-10))

    ebic = -2.0 * log_lik + k * np.log(n) + 4.0 * k * gamma * np.log(p)
    return float(ebic)


def glasso_ebic(
    corr_matrix: NDArray[np.float64],
    n_obs: int,
    gamma: float = 0.5,
    n_lambdas: int = 100,
) -> GLASSOResult:
    """Estimate sparse network via GLASSO with EBIC model selection.

    Uses sklearn's ``graphical_lasso`` function which accepts an empirical
    covariance matrix directly.  Previous implementation generated synthetic
    data with ``max(n_obs, p + 1)`` rows, which inflated the effective
    sample size when p > n (e.g., Senate: 226 bills from 40 legislators).
    The direct-covariance approach avoids this distortion (ADR-0126).

    Parameters:
        corr_matrix: p × p symmetric correlation matrix (tetrachoric or Pearson).
        n_obs: Number of observations (legislators) used to compute the
            correlation matrix. Needed for BIC/EBIC.
        gamma: EBIC hyperparameter ∈ [0, 1]. Higher = sparser networks.
            Default 0.5 (Golino's EGA default).
        n_lambdas: Number of lambda values in the sweep.

    Returns:
        GLASSOResult with the optimal sparse partial correlation network.
    """
    p = corr_matrix.shape[0]

    # Lambda range: from lambda_max (all edges zero) down to lambda_max/100
    # lambda_max = max off-diagonal absolute value in correlation matrix
    off_diag = np.abs(corr_matrix.copy())
    np.fill_diagonal(off_diag, 0.0)
    lambda_max = float(np.max(off_diag))
    if lambda_max < 1e-10:
        lambda_max = 1.0
    lambda_min = lambda_max / 100.0

    lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), num=n_lambdas)

    # Use correlation matrix as empirical covariance for GLASSO
    # (valid when input is a standardized correlation matrix)
    emp_cov = corr_matrix.copy()

    best_ebic = np.inf
    best_precision = np.eye(p)
    best_lambda = lambdas[0]
    ebic_values: list[tuple[float, float]] = []

    for lam in lambdas:
        try:
            # Use sklearn's graphical_lasso function directly with the
            # empirical covariance matrix — no synthetic data needed.
            _, precision = graphical_lasso(
                emp_cov,
                alpha=float(lam),
                max_iter=200,
                tol=1e-4,
            )
        except Exception:
            ebic_values.append((float(lam), np.inf))
            continue

        ebic = _compute_ebic(precision, emp_cov, n_obs, gamma)
        ebic_values.append((float(lam), ebic))

        if ebic < best_ebic:
            best_ebic = ebic
            best_precision = precision
            best_lambda = float(lam)

    partial_corr = _precision_to_partial_corr(best_precision)

    # Count edges
    upper = np.triu(partial_corr, k=1)
    n_edges = int(np.sum(np.abs(upper) > 1e-10))

    return GLASSOResult(
        partial_corr=partial_corr,
        precision=best_precision,
        selected_lambda=best_lambda,
        ebic_curve=np.array(ebic_values),
        n_edges=n_edges,
    )
