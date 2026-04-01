"""Utility functions for the JupyterBook."""

import io
import json
import os
from typing import Any

import numpy as np
from IPython.display import Image
from myst_nb import glue as myst_nb_glue
from nilearn import image, masking


def load_pafin(data_path):
    func_dir = os.path.join(data_path, "ds006185/sub-24053/ses-1/func/")
    prefix = "sub-24053_ses-1_task-rat_dir-PA_run-01"
    data_files = [
        os.path.join(func_dir, f"{prefix}_echo-1_part-mag_desc-preproc_bold.nii.gz"),
        os.path.join(func_dir, f"{prefix}_echo-2_part-mag_desc-preproc_bold.nii.gz"),
        os.path.join(func_dir, f"{prefix}_echo-3_part-mag_desc-preproc_bold.nii.gz"),
        os.path.join(func_dir, f"{prefix}_echo-4_part-mag_desc-preproc_bold.nii.gz"),
        os.path.join(func_dir, f"{prefix}_echo-5_part-mag_desc-preproc_bold.nii.gz"),
    ]
    echo_times = []
    for f in data_files:
        json_file = f.replace('.nii.gz', '.json')
        with open(json_file, 'r') as fo:
            metadata = json.load(fo)
        echo_times.append(metadata['EchoTime'])
    mask_file = os.path.join(func_dir, f"{prefix}_part-mag_desc-brain_mask.nii.gz")
    confounds_file = os.path.join(func_dir, f"{prefix}_part-mag_desc-confounds_timeseries.tsv")
    return {
        'echo_files': data_files,
        'mask': mask_file,
        'confounds': confounds_file,
        'echo_times': echo_times,
    }


def glue_figure(name: str, fig: Any, display: bool = False, dpi: int = 150, **savefig_kw: Any) -> None:
    """Glue a matplotlib Figure for myst_nb.

    Newer matplotlib releases do not register ``image/png`` on ``Figure`` for
    IPython's display pipeline, so :func:`myst_nb.glue` would only capture the
    ``<Figure ...>`` text repr. This saves the figure to PNG and glues an
    :class:`~IPython.display.Image` instead.
    """
    buf = io.BytesIO()
    kwargs: dict[str, Any] = {"format": "png", "bbox_inches": "tight", "dpi": dpi}
    kwargs.update(savefig_kw)
    fig.savefig(buf, **kwargs)
    buf.seek(0)
    myst_nb_glue(name, Image(data=buf.getvalue()), display=display)


def regress_one_image_out_of_another(data_img, nuis_img, mask_img):
    """Do what it says on the tin."""
    # First, mean-center each image over time
    mean_data_img = image.mean_img(data_img)
    mean_nuis_img = image.mean_img(nuis_img)

    data_img_mc = image.math_img(
        "img - avg_img[..., None]",
        img=data_img,
        avg_img=mean_data_img,
    )
    nuis_img_mc = image.math_img(
        "img - avg_img[..., None]",
        img=nuis_img,
        avg_img=mean_nuis_img,
    )

    # Now get the masked data in 2D format
    data_mc = masking.apply_mask(data_img_mc, mask_img)
    nuis_mc = masking.apply_mask(nuis_img_mc, mask_img)
    # nuis_mean = masking.apply_mask(mean_nuis_img, mask_img)
    data_mean = masking.apply_mask(mean_data_img, mask_img)

    # Build beta map by performing regression on each voxel
    betas = np.zeros(data_mc.shape[1])
    for i_voxel in range(data_mc.shape[1]):
        temp_data = np.stack((data_mc[:, i_voxel], np.ones(data_mc.shape[0])), -1)
        betas = np.linalg.lstsq(temp_data, nuis_mc[:, i_voxel], rcond=None)[0][0]

    # Construct denoised time series
    scaled_nuis = nuis_mc * betas
    errorts = (data_mc - scaled_nuis) + data_mean
    errorts_img = masking.unmask(errorts, mask_img)

    return errorts_img


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
    t2s : numpy.ndarray of shape (time,)
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
