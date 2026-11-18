"""Utility functions for the JupyterBook."""

import glob
import json
import os
import shutil
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
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


def load_diva(data_path, task="localizerDetection"):
    """Collect the files of a DIVA localizer run.

    The runs are ``sub-Blossom/ses-02``'s localizer runs from the Dense Investigation
    of Variability of Affect dataset (ds002278).
    ``localizerDetection`` (the default) is the block-design run used in the task and
    external regressor pages, and ``localizerEstimation`` is the event-related run used
    in the deconvolution page.
    The raw dataset supplies the events, the physiological recording and the echo times.
    The fMRIPrep derivatives supply the preprocessed echoes (in the space of the run's
    BOLD reference), the brain mask, the confounds and the transforms.
    """
    prefix = f"sub-Blossom_ses-02_task-{task}_run-1"
    raw_dir = os.path.join(data_path, "ds002278/sub-Blossom/ses-02/func")
    func_dir = os.path.join(data_path, "ds002278-fmriprep/sub-Blossom/ses-02/func")
    anat_dir = os.path.join(data_path, "ds002278-fmriprep/sub-Blossom/ses-02/anat")
    anat_prefix = "sub-Blossom_ses-02_run-1"

    echo_files = sorted(
        glob.glob(os.path.join(func_dir, f"{prefix}_echo-*_part-mag_desc-preproc_bold.nii.gz"))
    )
    echo_times = []
    for echo_file in echo_files:
        # The echo times are in the raw sidecars
        raw_json = os.path.join(
            raw_dir,
            os.path.basename(echo_file).replace("_desc-preproc_bold.nii.gz", "_bold.json"),
        )
        with open(raw_json, "r") as fo:
            echo_times.append(json.load(fo)["EchoTime"])

    return {
        "prefix": prefix,
        "echo_files": echo_files,
        "echo_times": echo_times,
        # fMRIPrep's optimal combination of the echoes
        "optcom": os.path.join(func_dir, f"{prefix}_part-mag_desc-preproc_bold.nii.gz"),
        "optcom_json": os.path.join(func_dir, f"{prefix}_part-mag_desc-preproc_bold.json"),
        "boldref": os.path.join(func_dir, f"{prefix}_part-mag_space-run_boldref.nii.gz"),
        "mask": os.path.join(func_dir, f"{prefix}_part-mag_space-run_desc-brain_mask.nii.gz"),
        "confounds": os.path.join(func_dir, f"{prefix}_part-mag_desc-confounds_timeseries.tsv"),
        "events": os.path.join(raw_dir, f"{prefix}_events.tsv"),
        "physio": os.path.join(raw_dir, f"{prefix}_physio.tsv.gz"),
        "physio_json": os.path.join(raw_dir, f"{prefix}_physio.json"),
        "run_to_t1w": os.path.join(func_dir, f"{prefix}_from-run_to-T1w_mode-image_desc-coreg_xfm.txt"),
        "mni_to_t1w": os.path.join(anat_dir, f"{anat_prefix}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"),
        "t1w": os.path.join(anat_dir, f"{anat_prefix}_desc-preproc_T1w.nii.gz"),
    }


def atlas_mask(atlas, labels):
    """Binary image of the atlas regions whose label is in ``labels``.

    ``atlas`` is a nilearn atlas bunch with ``maps`` and ``labels``.
    """
    atlas_img = image.load_img(atlas.maps)
    values = [atlas.labels.index(label) for label in labels]
    return image.new_img_like(atlas_img, np.isin(atlas_img.get_fdata(), values).astype(np.uint8))


def sphere_mask(reference_img, center, radius):
    """Binary image of the voxels within ``radius`` mm of ``center`` (in world coordinates)."""
    ijk = np.indices(reference_img.shape[:3]).reshape(3, -1).T
    xyz = nb.affines.apply_affine(reference_img.affine, ijk)
    inside = np.linalg.norm(xyz - np.array(center), axis=1) <= radius
    return image.new_img_like(reference_img, inside.reshape(reference_img.shape[:3]).astype(np.uint8))


def warp_mni_to_run(img, data):
    """Warp a mask or atlas from MNI152NLin2009cAsym space to the space of a run's BOLD reference.

    ``data`` is the dictionary returned by :func:`load_diva`.
    NiTransforms maps each voxel of the reference (the BOLD reference) to the moving image,
    so the chain goes from the BOLD reference to the T1w image
    (the inverse of fMRIPrep's coregistration) and then from the T1w image to the template.
    Nearest-neighbor interpolation keeps the values of a mask or an atlas intact.
    """
    import nitransforms as nit

    run_to_t1w = nit.linear.load(data["run_to_t1w"], fmt="itk")
    mni_to_t1w = nit.manip.load(data["mni_to_t1w"], fmt="itk")
    run_to_mni = nit.manip.TransformChain([~run_to_t1w, *mni_to_t1w.transforms])
    return nit.resampling.apply(run_to_mni, img, reference=data["boldref"], order=0)


def warp_mni_roi_to_run(roi_img, data):
    """Warp a binary ROI from MNI152NLin2009cAsym space to the space of a run's BOLD reference.

    ``data`` is the dictionary returned by :func:`load_diva`.
    See :func:`warp_mni_to_run` for the transforms.
    The warped ROI is limited to the brain mask.
    """
    roi_run = warp_mni_to_run(roi_img, data)
    return image.math_img("(roi > 0.5) & (mask > 0)", roi=roi_run, mask=data["mask"])


def glue_figure(name: str, fig: Any, display: bool = False, dpi: int = 150, **savefig_kw: Any) -> None:
    """Save a matplotlib Figure as a named PNG for use in Jupyter Book.

    Figures are written to ``content/figures/<name>.png``.  The Sphinx
    ``{figure}`` directive references these files directly, so they are copied
    to ``_build/html/_images/<name>.png`` without any content-hash renaming.
    This keeps figure filenames stable across rebuilds, which produces a clean
    git history for the built ``docs/`` tree.

    When *display* is ``False`` (the default) the figure is closed after saving
    so it does not also appear as inline cell output.
    """
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    kwargs: dict[str, Any] = {"format": "png", "bbox_inches": "tight", "dpi": dpi}
    kwargs.update(savefig_kw)
    fig.savefig(os.path.join(figures_dir, f"{name}.png"), **kwargs)
    if not display:
        plt.close(fig)


def export_html_report(report_file: str, name: str) -> str:
    """Copy a standalone HTML report into the book so a page can embed it in an iframe.

    A report such as tedana's is a complete HTML document with its own stylesheets and
    scripts, and it loads its images from a ``figures`` folder next to it.  Displaying
    its source inside a notebook cell breaks both the report (the images are not part of
    the built site) and the page around it (the report's CSS overrides the book theme).

    This copies the report to ``content/figures/html_extra/<name>/index.html``, with its
    ``figures`` folder beside it.  ``content/figures/html_extra`` is listed under
    ``html_extra_path`` in ``_config.yml``, so the report is served at ``<name>/index.html``
    in the built site, and it travels to CI with the rest of ``content/figures``.

    Parameters
    ----------
    report_file : str
        Path to the HTML report.  Its ``figures`` folder is expected in the same directory.
    name : str
        Name of the folder the report is served from.  Letters, digits, ``-`` and ``_`` only.

    Returns
    -------
    str
        URL of the report relative to the root of the built site.
    """
    if not name or not all(char.isalnum() or char in "-_" for char in name):
        raise ValueError(f"Invalid report name: {name!r}")

    extra_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "html_extra")
    report_dir = os.path.join(extra_dir, name)
    if os.path.isdir(report_dir):
        shutil.rmtree(report_dir)

    os.makedirs(report_dir)
    shutil.copyfile(report_file, os.path.join(report_dir, "index.html"))
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(report_file)), "figures")
    if os.path.isdir(figures_dir):
        shutil.copytree(figures_dir, os.path.join(report_dir, "figures"))

    return f"{name}/index.html"


def regress_one_image_out_of_another(data_img, nuis_img, mask_img):
    """Regress one image's time series out of another's, voxel by voxel.

    For each voxel, the mean-centered nuisance time series is fit to the
    mean-centered data time series by ordinary least squares, and the fitted
    nuisance signal is subtracted from the data.
    The data's voxelwise mean is then added back.

    Parameters
    ----------
    data_img : str or niimg
        4D image to denoise (e.g., the later echo).
    nuis_img : str or niimg
        4D nuisance image to regress out (e.g., the short echo).
    mask_img : str or niimg
        Brain mask.

    Returns
    -------
    errorts_img : niimg
        4D denoised image.
    """
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
    data_mean = masking.apply_mask(mean_data_img, mask_img)

    # Voxelwise OLS fit of the nuisance time series to the data time series.
    # Both are mean-centered, so no intercept is needed:
    # beta = sum(nuis * data) / sum(nuis ** 2).
    nuis_ss = np.sum(nuis_mc**2, axis=0)
    betas = np.divide(
        np.sum(nuis_mc * data_mc, axis=0),
        nuis_ss,
        out=np.zeros(data_mc.shape[1]),
        where=nuis_ss > 0,
    )

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
    return np.round(data, decimals=5)


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
    return np.round(log_data, decimals=5)


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

    return np.round(F_S0, decimals=5), np.round(F_R2, decimals=5), np.round(pred_S0, decimals=5), np.round(pred_R2, decimals=5)
