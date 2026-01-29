"""Statistical inference utilities."""

import math
from math import lgamma
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def ln_params_from_mean_sigma(mu: float, sigma: float) -> Tuple[float, float]:
    """Convert mean and sigma to lognormal parameters.

    Args:
        mu: Mean
        sigma: Standard deviation

    Returns:
        Tuple of (m, s) lognormal parameters
    """
    mu = float(mu)
    sigma = float(sigma)

    if not np.isfinite(mu) or not np.isfinite(sigma) or mu <= 0 or sigma < 0:
        return -np.inf, 0.0

    tau2 = np.log1p((sigma / mu) ** 2)
    m = np.log(mu) - 0.5 * tau2
    s = np.sqrt(tau2)
    return m, s


def poisson_sf_stable(
    n: int, lam: float, tol: float = 1e-14, max_terms: int = 100000
) -> float:
    """Compute Poisson survival function in a numerically stable way.

    Args:
        n: Count threshold
        lam: Poisson mean
        tol: Convergence tolerance
        max_terms: Maximum number of terms

    Returns:
        P(X >= n) where X ~ Poisson(lam)
    """
    n = int(n)
    lam = float(lam)

    if n <= 0:
        return 1.0
    if lam <= 0.0:
        return 0.0

    logpk = n * np.log(lam) - lam - lgamma(n + 1.0)
    pk = float(np.exp(logpk))
    s = pk
    k = n

    for _ in range(max_terms):
        k += 1
        pk *= lam / k
        s += pk
        if pk < tol * s:
            break

    return float(s)


def p0_lognormal_poisson(
    n_obs: int, mu: float, sigma: float, n_quad: int = 64
) -> float:
    """Compute p-value using lognormal-Poisson convolution.

    Args:
        n_obs: Observed count
        mu: Background mean
        sigma: Background uncertainty
        n_quad: Number of quadrature points

    Returns:
        p-value
    """
    if not np.isfinite(mu) or mu <= 0:
        return poisson_sf_stable(n_obs, max(mu, 0.0))
    if not np.isfinite(sigma) or sigma <= 0:
        return poisson_sf_stable(n_obs, mu)

    m, s = ln_params_from_mean_sigma(mu, sigma)
    if not np.isfinite(m) or not np.isfinite(s) or s <= 0:
        return poisson_sf_stable(n_obs, mu)

    xi, wi = np.polynomial.hermite.hermgauss(int(n_quad))
    lam = np.exp(m + np.sqrt(2.0) * s * xi)
    sf = np.array([poisson_sf_stable(n_obs, float(l)) for l in lam])
    return float((wi * sf).sum() / np.sqrt(np.pi))


def p0_from_blind_vectors(
    obs_vec: np.ndarray, mu_vec: np.ndarray, cov: Optional[np.ndarray]
) -> Tuple[float, float]:
    """Compute p0 and Z from blind window vectors.

    Args:
        obs_vec: Observed counts in blind window
        mu_vec: Background mean in blind window
        cov: Covariance matrix

    Returns:
        Tuple of (p0, Z significance)
    """
    n = int(np.sum(obs_vec))
    mu_sum = float(np.sum(mu_vec))

    if cov is None:
        sigma_sum = math.sqrt(max(mu_sum, 0.0))
    else:
        sigma_sum = math.sqrt(max(float(np.sum(cov)), 0.0))

    p0 = p0_lognormal_poisson(n, mu_sum, max(sigma_sum, 1e-12))
    p0 = min(max(float(p0), 0.0), 1.0)

    if 0.0 < p0 < 1.0:
        Z = float(norm.isf(p0))
    elif p0 == 0.0:
        Z = math.inf
    else:
        Z = 0.0

    return p0, Z


def _chol_with_jitter(C: np.ndarray, jitter0: float = 1e-10) -> np.ndarray:
    """Compute Cholesky decomposition with jitter for numerical stability."""
    C = np.asarray(C, float)
    B = C.shape[0]
    jitter = float(jitter0)

    for _ in range(8):
        try:
            return np.linalg.cholesky(C + jitter * np.eye(B))
        except np.linalg.LinAlgError:
            jitter *= 10.0

    # Fall back to eigendecomposition
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


def fit_A_profiled_gaussian(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
    allow_negative: bool = True,
) -> Dict[str, float]:
    """Fit signal amplitude with profiled Gaussian prior on background.

    Args:
        n_obs: Observed counts
        b_mean: Background mean prediction
        b_cov: Background covariance matrix
        template: Normalized signal template
        allow_negative: Whether to allow negative A values

    Returns:
        Dictionary with fit results
    """
    n = np.asarray(n_obs, int)
    b = np.asarray(b_mean, float)
    C = np.asarray(b_cov, float)
    w = np.asarray(template, float)

    B = b.size
    if C.shape != (B, B):
        raise ValueError("Cov shape mismatch")

    L = _chol_with_jitter(C)

    def nll_and_grad(x):
        A = float(x[0])
        th = np.asarray(x[1:], float)

        lam = b + L @ th + A * w
        if np.any(lam <= 0.0) or not np.all(np.isfinite(lam)):
            return 1e30, np.zeros_like(x)

        ll = np.sum(n * np.log(lam) - lam) - 0.5 * np.dot(th, th)
        nll = -ll

        r = (n / lam) - 1.0
        gA = -np.dot(w, r)
        gth = -(L.T @ r - th)
        g = np.concatenate(([gA], gth))
        return nll, g

    bounds = None
    if not allow_negative:
        bounds = [(0.0, None)] + [(None, None)] * B

    x0 = np.concatenate(([0.0], np.zeros(B)))
    res = minimize(
        fun=lambda x: nll_and_grad(x)[0],
        x0=x0,
        jac=lambda x: nll_and_grad(x)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=200, ftol=1e-9),
    )

    Ahat = float(res.x[0])
    thhat = np.asarray(res.x[1:], float)
    lamhat = b + L @ thhat + Ahat * w
    lamhat = np.clip(lamhat, 1e-12, None)

    # Observed information blocks
    W = n / (lamhat**2)
    I_AA = float(np.sum(W * (w**2)))
    I_Ath = (w * W) @ L
    I_thth = (L.T * W) @ L + np.eye(B)

    try:
        sol = np.linalg.solve(I_thth, I_Ath.reshape(-1, 1)).reshape(-1)
        I_prof = I_AA - float(I_Ath @ sol)
        varA = 1.0 / max(I_prof, 1e-18)
        sigA = float(np.sqrt(varA))
    except Exception:
        sigA = float("nan")

    return dict(
        A_hat=Ahat,
        sigma_A=sigA,
        success=bool(res.success),
        status=int(res.status),
        nll=float(res.fun),
        message=str(res.message),
    )
