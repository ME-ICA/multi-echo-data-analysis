"""Utility functions for the JupyterBook."""
import numpy as np


def predict_bold_signal(echo_times, s0, t2s):
    """Predict multi-echo signal according to monoexponential decay model.

    Parameters
    ----------
    echo_times : numpy.ndarray of shape (tes,)
        Echo times for which to predict data, in milliseconds.
    s0 : numpy.ndarray of shape (time,)
        S0 time series.
    t2s : numpy.ndarray of shpae (time,)
        T2* time series.

    Returns
    -------
    data : numpy.ndarray of shape (tes, time)
        Predicted BOLD signal from each of the echo times.

    Notes
    -----
    This is meant to be a sort of inverse to the code used
    in tedana.decay.fit_decay
    """
    if not isinstance(t2s, np.ndarray):
        t2s = np.array([t2s])

    if not isinstance(s0, np.ndarray):
        s0 = np.array([s0])

    neg_tes = (-1 * echo_times)[None, :]
    r2s = (1 / t2s)[:, None]
    intercept = np.log(s0)[:, None]
    log_data = np.dot(r2s, neg_tes) + intercept
    # Removed -1 from outside exp because it messes up dt_sig2
    data = np.exp(log_data).T
    return data


def predict_loglinear(data, echo_times):
    """Predict log-linear-transformed data.

    Parameters
    ----------
    data
    echo_times

    Returns
    -------
    log_data
    """
    log_data = np.log(np.abs(data) + 1)
    return log_data


def compute_te_dependence_statistics(data, B, tes):
    """Calculate TE-(in)dependence model statistics.

    Parameters
    ----------
    data
    B
    tes

    Returns
    -------
    F_S0
    F_R2
    pred_S0
    pred_R2
    """
    tes = tes[:, None]
    data = data[None, ...]
    B = B[:, None]
    n_echos = len(tes)
    alpha = (np.abs(B) ** 2).sum(axis=0)
    mu = np.mean(data, axis=-1)
    X1 = mu.T  # Model 1
    X2 = np.tile(tes, (1, 1)) * mu.T  # Model 2

    # S0 Model
    # (S,) model coefficient map
    coeffs_S0 = (B * X1).sum(axis=0) / (X1 ** 2).sum(axis=0)
    pred_S0 = X1 * np.tile(coeffs_S0, (n_echos, 1))
    SSE_S0 = (B - pred_S0) ** 2
    SSE_S0 = SSE_S0.sum(axis=0)  # (S,) prediction error map
    F_S0 = (alpha - SSE_S0) * (n_echos - 1) / (SSE_S0)

    # R2 Model
    coeffs_R2 = (B * X2).sum(axis=0) / (X2 ** 2).sum(axis=0)
    pred_R2 = X2 * np.tile(coeffs_R2, (n_echos, 1))
    SSE_R2 = (B - pred_R2) ** 2
    SSE_R2 = SSE_R2.sum(axis=0)
    F_R2 = (alpha - SSE_R2) * (n_echos - 1) / (SSE_R2)

    return F_S0, F_R2, pred_S0, pred_R2
