"""Gaussian Process Regression preprocessing and fitting."""

from typing import Tuple, TYPE_CHECKING

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

if TYPE_CHECKING:
    from .config import Config


def alpha_var_log_from_counts(
    y: np.ndarray, config: "Config"
) -> np.ndarray:
    """Compute alpha (noise variance) for log-space GPR.

    Args:
        y: Count values
        config: Global configuration

    Returns:
        Alpha values for each bin
    """
    y = np.asarray(y, dtype=float)
    alpha = np.full_like(y, config.pre_zero_alpha, dtype=float)
    pos = y > 0.0

    if config.alpha_model == "1/y":
        alpha[pos] = 1.0 / np.clip(y[pos], 1e-12, None)
    else:
        alpha[pos] = 1.0 / np.clip(y[pos], 1.0, None)

    if config.pre_alpha_first_n > 0:
        k = min(config.pre_alpha_first_n, alpha.size)
        alpha[:k] *= config.pre_alpha_first_factor

    return alpha


def preprocess_xy_for_gpr(
    X: np.ndarray, y: np.ndarray, config: "Config"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess X, y arrays for GPR fitting.

    Optionally transforms to log space and computes alpha values.

    Args:
        X: Mass values
        y: Count values
        config: Global configuration

    Returns:
        Tuple of (X_in, y_in, alpha)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_in = np.log(np.clip(X, 1e-12, None)) if config.pre_log else X.copy()
    y_in = np.zeros_like(y, dtype=float)

    pos = y > 0.0
    if config.pre_log:
        y_in[pos] = np.log(y[pos])
        alpha = alpha_var_log_from_counts(y, config)
    else:
        y_in = y.copy()
        alpha = np.clip(y, 1.0, None)

    return X_in, y_in, alpha


def fit_gpr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: "Config",
    restarts: int = None,
) -> GaussianProcessRegressor:
    """Fit a Gaussian Process Regressor.

    Args:
        X_train: Training mass values
        y_train: Training count values
        config: Global configuration
        restarts: Number of optimizer restarts (defaults to config.n_restarts)

    Returns:
        Fitted GaussianProcessRegressor
    """
    if restarts is None:
        restarts = config.n_restarts

    X_in, y_in, alpha = preprocess_xy_for_gpr(X_train, y_train, config)

    gpr = GaussianProcessRegressor(
        kernel=config.get_kernel(),
        alpha=alpha,
        n_restarts_optimizer=int(restarts),
        normalize_y=False,
    )

    return gpr.fit(X_in.reshape(-1, 1), y_in)


def predict_counts_from_log_gpr(
    gpr: GaussianProcessRegressor,
    X_query: np.ndarray,
    config: "Config",
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict counts from a fitted GPR model.

    Applies lognormal moment transform if pre_log is enabled.

    Args:
        gpr: Fitted GaussianProcessRegressor
        X_query: Query mass values
        config: Global configuration

    Returns:
        Tuple of (mean counts, covariance matrix)
    """
    Xq = np.asarray(X_query, dtype=float)
    Xq_in = np.log(np.clip(Xq, 1e-12, None)) if config.pre_log else Xq.copy()

    mu, cov = gpr.predict(Xq_in.reshape(-1, 1), return_cov=True)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)

    if not config.pre_log:
        return mu, cov

    # Lognormal moment transform
    diag = np.clip(np.diag(cov), 0.0, None)
    E = np.exp(mu + 0.5 * diag)

    C = np.clip(cov, -40.0, 40.0)
    EyEj = np.outer(E, E)
    cov_y = EyEj * (np.exp(C) - 1.0)

    return E, cov_y
