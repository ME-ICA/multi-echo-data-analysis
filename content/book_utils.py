"""Utility functions for the JupyterBook."""
import numpy as np


def predict_parameters(timeseries, echo_time, *, s0=None, t2s=None):
    """Infer the S0 or T2* time series that produces the single-echo timeseries.

    Parameters
    ----------
    timeseries : numpy.ndarray of shape (n_timepoints,)
    echo_time : float
    s0
        Mutually exclusive with t2s.
    t2s
        Mutually exclusive with s0.

    Returns
    -------
    s0 : numpy.ndarray of shape (n_timepoints,)
    t2s : numpy.ndarray of shape (n_timepoints,)
    """
    # Check that only one of s0 or t2s is provided
    assert (s0 is None) or (t2s is None)
    assert (s0 is not None) or (t2s is not None)

    # Convert data for log-linear regression
    neg_te = np.array([-1 * echo_time])[:, None]
    log_timeseries = np.log(timeseries)[:, None]

    if s0 is None:
        print("Predicting S0")
        r2s = 1 / t2s
        r2s = np.atleast_2d(r2s).T
        intercept = log_timeseries - np.dot(r2s, neg_te)
        s0 = np.exp(intercept)
    else:
        print("Predicting T2*")
        intercept = np.log(s0)
        intercept = np.atleast_2d(intercept).T
        # need to solve for r2s
        # log_timeseries = np.dot(r2s, neg_te) + intercept
        temp = log_timeseries - intercept
        r2s = np.linalg.lstsq(neg_te.T, temp.T, rcond=None)[0].T
        t2s = 1 / r2s

    t2s = np.squeeze(t2s)
    s0 = np.squeeze(s0)
    return t2s, s0


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
    This is meant to be a sort of inverse to the code used in tedana.decay.fit_decay
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
