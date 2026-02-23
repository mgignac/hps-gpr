"""Statistical inference utilities."""

import math
from math import lgamma
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Legacy analytic p0 helpers (kept for back-compatibility)
# ---------------------------------------------------------------------------

def ln_params_from_mean_sigma(mu: float, sigma: float) -> Tuple[float, float]:
    """Convert mean and sigma to lognormal parameters."""
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
    """Compute Poisson survival function P(X >= n) in a numerically stable way."""
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
    """Compute p-value using lognormal-Poisson convolution."""
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
    """Compute p0 and Z from blind window vectors using lognormal-Poisson model."""
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


# ---------------------------------------------------------------------------
# Cholesky helpers
# ---------------------------------------------------------------------------

def _chol_with_jitter_fallback(C: np.ndarray, jitter0: float = 1e-10) -> np.ndarray:
    """Square-root factor with numerical regularization (fallback, no main-cell dep.)."""
    C = np.asarray(C, float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Covariance must be square; got shape={C.shape}")
    if not np.all(np.isfinite(C)):
        raise ValueError("Covariance contains non-finite entries (NaN/inf).")
    C = 0.5 * (C + C.T)
    B = C.shape[0]
    if B == 0:
        return np.zeros((0, 0), float)
    diag = np.diag(C)
    scale = float(np.max(np.abs(diag))) if diag.size else 1.0
    scale = max(scale, 1.0)
    I = np.eye(B)
    jitter = float(jitter0) * scale
    for _ in range(8):
        try:
            return np.linalg.cholesky(C + jitter * I)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    # Eigenvalue-clipped symmetric square root
    w, V = np.linalg.eigh(C)
    floor = max(1e-12 * scale, float(jitter0) * scale)
    w = np.clip(w, floor, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


def _chol_with_jitter(C: np.ndarray, jitter0: float = 1e-10, max_tries: int = 8) -> np.ndarray:
    """Return a numerically safe square-root factor of a (nearly) covariance matrix.

    Attempts Cholesky with progressively larger diagonal jitter, then falls back to
    an eigenvalue-clipped symmetric square root. The returned L satisfies L L^T ≈ C.
    """
    C = np.asarray(C, float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Covariance must be square; got shape={C.shape}")
    if not np.all(np.isfinite(C)):
        raise ValueError("Covariance contains non-finite entries (NaN/inf).")
    C = 0.5 * (C + C.T)
    B = C.shape[0]
    if B == 0:
        return np.zeros((0, 0), float)
    diag = np.diag(C)
    scale = float(np.max(np.abs(diag))) if diag.size else 1.0
    scale = max(scale, 1.0)
    I = np.eye(B)
    jitter = float(jitter0) * scale
    for _ in range(int(max_tries)):
        try:
            return np.linalg.cholesky(C + jitter * I)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    try:
        w, V = np.linalg.eigh(C)
        floor = max(1e-12 * scale, float(jitter0) * scale)
        w = np.clip(w, floor, None)
        return V @ np.diag(np.sqrt(w)) @ V.T
    except np.linalg.LinAlgError:
        d = np.clip(np.diag(C), 1e-12 * scale, None)
        return np.diag(np.sqrt(d))


def _get_chol(C: np.ndarray) -> np.ndarray:
    """Route to the best available Cholesky helper."""
    return _chol_with_jitter(C)


# ---------------------------------------------------------------------------
# Profiled likelihood — background nuisance profiling
# ---------------------------------------------------------------------------

def _profile_theta_given_A(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
    A_fixed: float,
    th0: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Profile over θ with A fixed (used internally by p0_profiled_gaussian_LRT).

    Model: n_i ~ Poisson(λ_i),  λ = b_mean + L θ + A_fixed * w,  θ ~ N(0, I).
    """
    n = np.clip(np.asarray(n_obs, float), 0.0, None)
    b = np.clip(np.asarray(b_mean, float), 1e-12, None)
    C = np.asarray(b_cov, float)
    w = np.asarray(template, float)
    B = b.size
    if C.shape != (B, B):
        raise ValueError(f"Cov shape mismatch: {C.shape} vs {(B, B)}")
    if w.shape != (B,):
        raise ValueError(f"Template shape mismatch: {w.shape} vs {(B,)}")
    L = _get_chol(C)
    eps = 1e-9 * max(1.0, float(np.median(b)))
    th0 = np.zeros(B, float) if th0 is None else np.asarray(th0, float).reshape(B)
    A_fixed = float(A_fixed)

    def nll_and_grad(th):
        lam = b + L @ th + A_fixed * w
        lam_eff = np.maximum(lam, eps)
        ll = np.sum(n * np.log(lam_eff) - lam_eff) - 0.5 * float(np.dot(th, th))
        r = (n / lam_eff) - 1.0
        return -float(ll), -(L.T @ r) + th

    res = minimize(
        fun=lambda th: nll_and_grad(th)[0],
        x0=th0,
        jac=lambda th: nll_and_grad(th)[1],
        method="L-BFGS-B",
        options=dict(maxiter=400, ftol=1e-10),
    )
    return dict(
        theta_hat=np.asarray(res.x, float),
        nll=float(res.fun) if np.isfinite(getattr(res, "fun", np.nan)) else float("nan"),
        success=bool(res.success),
        status=int(getattr(res, "status", -1)),
        message=str(getattr(res, "message", "")),
    )


def profile_theta_given_A(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
    *,
    A_fixed: float,
    lam_floor: float = 1e-12,
) -> Dict[str, object]:
    """Profile the Gaussian nuisance parameters θ for a *fixed* signal amplitude A.

    Model: λ = b_mean + L θ + A · template,  θ ~ N(0, I).

    Returns dict with: theta_hat, delta_b_hat, lambda_hat, b_fit, success, nll.
    """
    n = np.clip(np.asarray(n_obs, float), 0.0, None)
    b = np.clip(np.asarray(b_mean, float), 1e-12, None)
    C = np.asarray(b_cov, float)
    w = np.asarray(template, float)
    B = b.size
    if B == 0:
        return dict(theta_hat=np.array([]), delta_b_hat=np.array([]),
                    lambda_hat=np.array([]), b_fit=np.array([]),
                    success=False, nll=float("nan"))
    L = _chol_with_jitter(C)
    eps = max(float(lam_floor), 1e-9 * max(1.0, float(np.median(b))))
    A = float(A_fixed)

    def nll_and_grad(theta: np.ndarray):
        th = np.asarray(theta, float)
        delta_b = (L @ th).astype(float)
        lam = (b + delta_b + A * w).astype(float)
        lam_eff = np.maximum(lam, eps)
        r = (n / lam_eff) - 1.0
        nll = -float(np.sum(n * np.log(lam_eff) - lam_eff)) + 0.5 * float(th @ th)
        gth = -(L.T @ r - th)
        bad = lam < eps
        if np.any(bad):
            delta = eps - lam[bad]
            k = 1e6
            nll += float(k * np.dot(delta, delta))
            gth += L[bad].T @ (-2.0 * k * delta)
        return nll, np.asarray(gth, float)

    res = minimize(
        fun=lambda th: nll_and_grad(th)[0],
        x0=np.zeros(B, float),
        jac=lambda th: nll_and_grad(th)[1],
        method="L-BFGS-B",
        options=dict(maxiter=500, ftol=1e-10),
    )
    thhat = np.asarray(res.x, float)
    delta_b = (L @ thhat).astype(float)
    lamhat = (b + delta_b + A * w).astype(float)
    return dict(
        theta_hat=thhat,
        delta_b_hat=delta_b,
        lambda_hat=lamhat,
        b_fit=(b + delta_b).astype(float),
        success=bool(getattr(res, "success", False)),
        nll=float(getattr(res, "fun", float("nan"))),
    )


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def fit_A_profiled_gaussian_details(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
    *,
    allow_negative: bool = True,
    lam_floor: float = 1e-12,
) -> Dict[str, object]:
    """Profile-likelihood fit for A with Gaussian nuisance parameters for the background.

    Returns dict with: A_hat, sigma_A, theta_hat, delta_b_hat, b_fit, lambda_hat, success, nll.
    """
    n = np.clip(np.asarray(n_obs, float), 0.0, None)
    b = np.clip(np.asarray(b_mean, float), 1e-12, None)
    C = np.asarray(b_cov, float)
    w = np.asarray(template, float)
    B = b.size
    if B == 0:
        return dict(A_hat=float("nan"), sigma_A=float("nan"), theta_hat=np.array([]),
                    delta_b_hat=np.array([]), b_fit=np.array([]), lambda_hat=np.array([]),
                    success=False, nll=float("nan"))
    L = _chol_with_jitter(C)
    eps = max(float(lam_floor), 1e-9 * max(1.0, float(np.median(b))))

    # GLS initialisation for A
    def _gls_start():
        V = C + np.diag(np.clip(b, 1.0, None)) + (eps * 100.0) * np.eye(B)
        try:
            Vinv_w = np.linalg.solve(V, w)
            denom = float(np.dot(w, Vinv_w))
            if not np.isfinite(denom) or denom <= 0:
                raise ValueError
            A0 = float(np.dot(w, np.linalg.solve(V, n - b))) / denom
            sig0 = float(np.sqrt(1.0 / denom))
        except Exception:
            A0 = float(np.sum(n - b))
            sig0 = float(np.sqrt(np.sum(np.clip(b, 1.0, None))))
        return float(A0), float(sig0)

    A0, sig0 = _gls_start()
    if not allow_negative:
        A0 = max(0.0, A0)

    def nll_and_grad(x: np.ndarray):
        A = float(x[0])
        th = np.asarray(x[1:], float)
        lam = b + L @ th + A * w
        lam_eff = np.maximum(lam, eps)
        ll = np.sum(n * np.log(lam_eff) - lam_eff) - 0.5 * float(np.dot(th, th))
        r = (n / lam_eff) - 1.0
        gA = -float(np.dot(w, r))
        gth = -(L.T @ r - th)
        bad = lam < eps
        if np.any(bad):
            delta = eps - lam[bad]
            k = 1e6
            penalty = float(k * np.dot(delta, delta))
            dpen = -2.0 * k * delta
            gA += float(np.dot(w[bad], dpen))
            gth += L[bad].T @ dpen
            return -float(ll) + penalty, np.concatenate(([gA], np.asarray(gth, float)))
        return -float(ll), np.concatenate(([gA], np.asarray(gth, float)))

    bounds = None if allow_negative else [(0.0, None)] + [(None, None)] * B
    res = minimize(
        fun=lambda x: nll_and_grad(x)[0],
        x0=np.concatenate(([A0], np.zeros(B))),
        jac=lambda x: nll_and_grad(x)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=500, ftol=1e-10),
    )

    Ahat = float(res.x[0])
    thhat = np.asarray(res.x[1:], float)
    delta_b = (L @ thhat).astype(float)
    lamhat = (b + delta_b + Ahat * w).astype(float)
    lam_eff = np.maximum(lamhat, eps)

    # Profile information (Schur complement) for σ_A
    W = n / (lam_eff ** 2)
    I_AA = float(np.sum(W * (w ** 2)))
    I_Ath = (w * W) @ L
    I_thth = (L.T * W) @ L + np.eye(B)
    sigA = float("nan")
    try:
        sol = np.linalg.solve(I_thth, I_Ath.reshape(-1, 1)).reshape(-1)
        I_prof = I_AA - float(I_Ath @ sol)
        sigA = float(np.sqrt(1.0 / max(I_prof, 1e-18)))
    except Exception:
        sigA = float(sig0)

    if not allow_negative and Ahat < 0:
        Ahat = 0.0

    return dict(
        A_hat=float(Ahat),
        sigma_A=float(sigA),
        theta_hat=thhat,
        delta_b_hat=delta_b,
        b_fit=(b + delta_b).astype(float),
        lambda_hat=lamhat,
        success=bool(getattr(res, "success", False)),
        nll=float(getattr(res, "fun", float("nan"))),
    )


def fit_A_profiled_gaussian(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
    allow_negative: bool = True,
) -> Dict[str, float]:
    """Fit signal amplitude with profiled Gaussian prior on background.

    Thin wrapper around fit_A_profiled_gaussian_details returning only the essentials.
    """
    d = fit_A_profiled_gaussian_details(
        n_obs, b_mean, b_cov, template, allow_negative=allow_negative
    )
    return dict(
        A_hat=float(d["A_hat"]),
        sigma_A=float(d["sigma_A"]),
        success=bool(d["success"]),
        nll=float(d.get("nll", np.nan)),
    )


# ---------------------------------------------------------------------------
# Profiled likelihood ratio test p0
# ---------------------------------------------------------------------------

def p0_profiled_gaussian_LRT(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
) -> Tuple[float, float, float, Dict[str, object]]:
    """Exact profiled LRT p0 for A >= 0 vs A = 0.

    Returns (p0, Z, q0, info) where q0 = -2 ln Λ and Z = sqrt(q0).

    Uses the same Poisson + Gaussian-prior additive model as fit_A_profiled_gaussian.
    The null is maximized over θ with A=0; the alternative over (A>=0, θ).
    """
    alt = fit_A_profiled_gaussian(
        n_obs=np.asarray(n_obs, int),
        b_mean=b_mean,
        b_cov=b_cov,
        template=template,
        allow_negative=False,
    )
    nll_alt = float(alt.get("nll", float("nan")))
    A_hat = float(alt.get("A_hat", float("nan")))
    sigma_A = float(alt.get("sigma_A", float("nan")))
    ok_alt = bool(alt.get("success", False))

    null = _profile_theta_given_A(
        n_obs=n_obs, b_mean=b_mean, b_cov=b_cov,
        template=template, A_fixed=0.0,
    )
    nll0 = float(null.get("nll", float("nan")))
    ok_null = bool(null.get("success", False))

    q0 = 0.0
    if np.isfinite(nll0) and np.isfinite(nll_alt):
        q0 = max(0.0, 2.0 * (nll0 - nll_alt))

    Z = float(np.sqrt(q0)) if q0 > 0 else 0.0
    p0 = min(max(float(norm.sf(Z)), 0.0), 1.0)

    info = dict(
        q0=float(q0), Z=float(Z), p0=float(p0),
        A_hat=float(A_hat), sigma_A=float(sigma_A),
        nll_alt=float(nll_alt), nll0=float(nll0),
        ok_alt=bool(ok_alt), ok_null=bool(ok_null),
        ok=bool(ok_alt and ok_null),
    )
    return float(p0), float(Z), float(q0), info


# ---------------------------------------------------------------------------
# Look-elsewhere effect (LEE)
# ---------------------------------------------------------------------------

def _z_from_p_one_sided(p: np.ndarray) -> np.ndarray:
    """One-sided Gaussian Z corresponding to p: Z = Φ⁻¹(1 - p)."""
    p = np.clip(np.asarray(p, float), 1e-300, 1.0)
    return norm.isf(p)


def _p_from_z_one_sided(z: float) -> float:
    """One-sided p-value from Z: p = 1 - Φ(Z)."""
    return float(norm.sf(float(z)))


def _p_global_from_local(
    p_local: np.ndarray, *, Neff: float, method: str = "sidak"
) -> np.ndarray:
    """Convert local one-sided p-values to global p-values via a trials factor.

    method: 'sidak' (default) or 'bonferroni'.
    """
    p = np.clip(np.asarray(p_local, float), 0.0, 1.0)
    N = max(float(Neff), 1.0)
    if str(method).lower().strip() == "bonferroni":
        return np.clip(N * p, 0.0, 1.0)
    # Šidák: p_global = 1 - (1 - p)^N, stable via log1p
    return np.clip(-np.expm1(N * np.log1p(-p)), 0.0, 1.0)


def draw_bkg_mvn_nonneg(
    mean: np.ndarray,
    cov: Optional[np.ndarray],
    size: int,
    rng: np.random.Generator,
    *,
    method: str = "reject_then_clip",
    max_tries: int = 80,
) -> np.ndarray:
    """Draw non-negative MVN background toys.

    method:
        'clip'             — draw MVN then clip negatives (fast, distorts correlations near 0)
        'reject'           — rejection-sample until all non-negative (best fidelity, may be slow)
        'reject_then_clip' — rejection-sample with a clip fallback if acceptance is too low
    """
    m = np.asarray(mean, dtype=float).reshape(-1)
    size = int(max(1, size))
    if cov is None:
        return np.tile(m, (size, 1))

    C = np.asarray(cov, dtype=float)

    def _propose(n: int) -> np.ndarray:
        try:
            return rng.multivariate_normal(m, C, size=n, check_valid="ignore", tol=1e-8)
        except Exception:
            diag = np.clip(np.diag(C), 0.0, None)
            return rng.normal(loc=m, scale=np.sqrt(diag), size=(n, m.size))

    if method == "clip":
        return np.clip(_propose(size), 0.0, None)

    out = np.empty((size, m.size), dtype=float)
    filled = 0
    tries = 0
    while filled < size and tries < int(max_tries):
        n_prop = max(1, int(1.3 * (size - filled)))
        prop = _propose(n_prop)
        acc = prop[np.all(prop >= 0.0, axis=1)]
        if acc.shape[0] > 0:
            n_take = min(acc.shape[0], size - filled)
            out[filled: filled + n_take] = acc[:n_take]
            filled += n_take
        tries += 1

    if filled < size:
        if method == "reject_then_clip":
            out[filled:] = np.clip(_propose(size - filled), 0.0, None)
        else:
            out[filled:] = np.tile(m, (size - filled, 1))

    return out


def _lee_trials_from_grid(
    masses: np.ndarray,
    ds_keys: List[str],
    *,
    datasets: dict,
    indep_width_sigma: float = 2.355,
    combo_sigma_method: str = "min",
) -> float:
    """Approximate effective trials factor N_eff for the look-elsewhere effect.

    Estimates ∫ dm / (W · σ_eff(m)) where W = indep_width_sigma and σ_eff is the
    mass resolution. N_eff is capped at the number of tested mass points.
    """
    m = np.sort(np.asarray(masses, float)[np.isfinite(np.asarray(masses, float))])
    if m.size < 2:
        return 1.0
    W = max(float(indep_width_sigma), 1e-6)

    def sigma_eff(mi: float) -> float:
        sigs = []
        for k in ds_keys:
            try:
                sigs.append(float(datasets[k].sigma(float(mi))))
            except Exception:
                continue
        sigs = [s for s in sigs if np.isfinite(s) and s > 0]
        if not sigs:
            return float("nan")
        meth = str(combo_sigma_method).lower().strip()
        if meth == "mean":
            return float(np.mean(sigs))
        if meth == "harmonic":
            return float(len(sigs) / np.sum(1.0 / np.asarray(sigs)))
        return float(np.min(sigs))  # default: "min"

    Neff = 0.0
    for dmi, mi in zip(np.diff(m), 0.5 * (m[:-1] + m[1:])):
        sig = sigma_eff(float(mi))
        if np.isfinite(sig) and sig > 0:
            Neff += float(dmi) / (W * float(sig))

    return float(np.clip(max(Neff, 1.0), 1.0, float(m.size)))
