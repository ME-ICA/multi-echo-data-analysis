---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Cerebrovascular Reactivity Mapping with `phys2cvr`

```{admonition} Abbreviations
- CVR = Cerebrovascular Reactivity
- ME = Multi-Echo
- IC = Independent Component
- ICA = Independent Component Analysis
- $P_{ET}CO_2$ = Pressure of End-Tidal Carbon Dioxide
```

```{warning}
This tutorial assumes you ran:
- [Optimal combination with `t2smap`](01_Optimal_Combination_with_t2smap.md)
- [Multi-Echo Denoising with `tedana`](03_Denoising_with_tedana.md)
- [Manual Classification with `rica`](https://me-ica.github.io/rica/)
```

```{important}
The dataset used in this book has no $CO_2$ recording, and its task is not a respiratory challenge.
So that every step below can run, **the $CO_2$ trace in this tutorial is simulated**,
and the component classification comes from `tedana`'s automatic decision tree rather than from a manual revision.
The maps derived from the simulated trace show what `phys2cvr` produces, but they are not measurements of CVR.
The simulated trace will be replaced once a suitable public dataset is available.
```

Cerebrovascular Reactivity (CVR) Mapping can greatly benefit from multi-echo (ME) data acquisition and independent component analysis (ICA) denoising, especially when it is based on breath-holds, cued deep breaths, or resting-state.

While ME-ICA will greatly reduce motion-related noise (see {cite:t}`MOIA2021117914`), because CVR is an increase in blood flow triggered by vessels dilation, there is a component of signal that is proton-density related. For this reason, the default decision tree in tedana may categorise certain components that should (or at least could) be kept as noise, so it is _strongly_ suggested to perform a manual revision of the automatic categorisation and/or adapt the decision tree to accept a better suited set of components.

On top of that, in the context of breath-holds and cued deep breaths, attention should be made in the denoising stage, as an aggressive approach will remove all sources of signal that inform CVR and the associated haemodynamic lag. Instead, a conservative denoising approach, where noise IC timeseries are orthogonalised to the (temporally shifted) $P_{ET}CO_2$ and the non-noise IC timeseries is to be preferred ({cite:t}`MOIA2021117914`).

Assuming that you are following the other tutorials in this book, once you obtained your [optimally combined signal](01_Optimal_Combination_with_t2smap.md) and your [ICA decomposition](03_Denoising_with_tedana.md) that has been [manually inspected (and corrected if needed)](https://me-ica.github.io/rica/), you will need to make sure you have:
  - a whole brain mask in functional space that will confine `phys2cvr` operations within it - the `desc-adaptiveGoodSignal_mask.nii.gz` output of `tedana` can work fine, but if you have a different mask that is also ok;
  - if you want, a ROI mask in functional space which average signal will be used as reference to align the regressor (e.g. the $CO_2$ trace) and which average haemodynamic lag will be set as 0-lag - grey matter or cerebellum are commonly used as reference ROI;
  - if you want to run ICA denoising, the timeseries of rejected and accepted components from tedana in two different files.

## Preparing the inputs

We start from the outputs of the [`tedana` tutorial](03_Denoising_with_tedana.md).

```{code-cell} ipython3
import os

import matplotlib.pyplot as plt
import nibabel as nb
import nitransforms as nit
import numpy as np
import pandas as pd
from book_utils import glue_figure
from nilearn import image, plotting

data_path = os.path.abspath('../data')
prefix = "sub-24053_ses-1_task-rat_dir-PA_run-01"
func_dir = os.path.join(data_path, "ds006185/sub-24053/ses-1/func")
anat_dir = os.path.join(data_path, "ds006185/sub-24053/ses-1/anat")
ted_dir = os.path.join(data_path, "tedana")
out_dir = os.path.join(data_path, "cvr")
os.makedirs(out_dir, exist_ok=True)

# The optimally combined data and the brain mask written by tedana
optcom_file = os.path.join(ted_dir, f"{prefix}_desc-optcom_bold.nii.gz")
mask_file = os.path.join(ted_dir, f"{prefix}_desc-adaptiveGoodSignal_mask.nii.gz")
```

### A reference ROI

We use grey matter as the reference ROI.
fMRIPrep provides a tissue segmentation in the space of the T1w image (`1` is grey matter),
so we resample it onto the grid of the functional data and keep the grey matter voxels that are inside the brain mask.
In addition to the files listed in [Download Data](00_Download_Data.md), this needs the segmentation:

```bash
datalad get sub-24053/ses-1/anat/sub-24053_ses-1_dseg.nii.gz
datalad get sub-24053/ses-1/func/sub-24053_ses-1_task-rat_dir-PA_run-01_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt
```

```{code-cell} ipython3
# The transform moves images from the BOLD reference to the T1w image,
# so its inverse brings the segmentation into the space of the BOLD data.
xfm = nit.linear.load(
    os.path.join(func_dir, f"{prefix}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"),
    fmt="itk",
)
dseg_img = nit.resampling.apply(
    ~xfm,
    os.path.join(anat_dir, "sub-24053_ses-1_dseg.nii.gz"),
    reference=mask_file,
    order=0,
)
mask_img = nb.load(mask_file)
roi = (dseg_img.get_fdata() == 1) & (mask_img.get_fdata() > 0)

roi_file = os.path.join(out_dir, "gm_mask.nii.gz")
nb.Nifti1Image(roi.astype(np.uint8), mask_img.affine).to_filename(roi_file)
print(f"{roi.sum()} grey matter voxels out of {(mask_img.get_fdata() > 0).sum()} voxels in the mask")
```

### Accepted and rejected component timeseries

The easiest way to obtain these is using the mixing matrix from `tedana` and a component classification with the help of `pandas`.
If you revised the classification with `rica`, read the `manual_classification.tsv` file you downloaded from it instead of the metrics table.
It has the same `Component` and `classification` columns.

```{code-cell} ipython3
# Read the component classification.
# Here it is the automatic one from tedana; with rica, read manual_classification.tsv instead.
classification = pd.read_table(os.path.join(ted_dir, f"{prefix}_desc-tedana_metrics.tsv"))

# Read the ICA mixing matrix
ica_mix = pd.read_table(os.path.join(ted_dir, f"{prefix}_desc-ICA_mixing.tsv"))

# Extract the list of rejected and accepted components
rej_comp = classification.loc[classification["classification"] == "rejected", "Component"].tolist()
acc_comp = classification.loc[classification["classification"] == "accepted", "Component"].tolist()
print(f"{len(acc_comp)} accepted and {len(rej_comp)} rejected components")

# Extract rejected vs accepted timeseries and save them
accepted_file = os.path.join(out_dir, "accepted_ic_timeseries.csv")
rejected_file = os.path.join(out_dir, "rejected_ic_timeseries.csv")
ica_mix[acc_comp].to_csv(accepted_file, index=False, header=False)
ica_mix[rej_comp].to_csv(rejected_file, index=False, header=False)
```

### Other nuisance regressors

Any other source of noise can be added as well.
Here we take the motion parameters and their temporal derivatives from the fMRIPrep confounds.

```{code-cell} ipython3
confounds = pd.read_table(
    os.path.join(func_dir, f"{prefix}_part-mag_desc-confounds_timeseries.tsv")
)
motion_columns = [
    f"{kind}_{axis}{suffix}"
    for kind in ("trans", "rot")
    for axis in ("x", "y", "z")
    for suffix in ("", "_derivative1")
]
motion_file = os.path.join(out_dir, "motion_params.csv")
# The first value of each derivative is undefined
confounds[motion_columns].fillna(0).to_csv(motion_file, index=False, header=False)
```

## The `phys2cvr` workflow

After [installing `phys2cvr`](https://phys2cvr.readthedocs.io/en/latest/usage/installation.html#basic-installation), potentially with [extra dependencies](https://phys2cvr.readthedocs.io/en/latest/usage/installation.html#richer-installation), we can import the main workflow of `phys2cvr` and call its help to see all available parameters (many).

```{code-cell} ipython3
:tags: [output_scroll]

from phys2cvr.workflows import phys2cvr as p2c


help(p2c)
```

You may notice many and one potential parameters. `phys2cvr` is meant to be very versatile, but fear not! Only few of them are needed in different situations. For more info, check the [workflow API page](https://phys2cvr.readthedocs.io/en/latest/generated/phys2cvr.workflows.phys2cvr.html#phys2cvr.workflows.phys2cvr)

As default, `phys2cvr` takes anything given to it and creates lagged regressors of the interpolated end-tidal of a given trace (that is, the $P_{ET}CO_2hrf$) after aligning it with a reference signal and convolving it with a canonical HRF. However, you can run a lag regression internally.
Generally speaking, you may need these parameters - although only `fname_func` is mandatory:

```python
  fname_func=optcom_file,             # The optimally combined data
  fname_co2=co2_file,                 # The CO2 trace with peaks
  fname_mask=mask_file,               # A mask to limit the analysis to
  fname_roi=roi_file,                 # A ROI to use as signal reference
  outdir=os.path.join(out_dir, "co2"),  # Output folder
  run_regression=True,                # Run the analysis!
  lag_max=9,                          # Maximum lag to consider
  lag_step=0.3,                       # Lag step to consider
  l_degree=2,                         # Legendre polynomials degrees
```

You can also run a simple regression instead of a lagged one - in that case, remove `lag_max` and `lag_step`.

In terms of regressor for your CVR analysis, many options are supported:
  1. $CO_2$ traces (and their peak indices)
  2. pre-made $P_{ET}CO_2$ traces
  3. task designs (timeseries)
  4. BOLD signals averages from a ROI mask.


## $CO_2$ traces (and their peak indices)

You can get $CO_2$ traces (and their peak indices) with whatever program you want - simply run a very light lowpass filter (and maybe a downsampling), followed by peak detection.

### A simulated $CO_2$ recording

Normally you would start from the file exported by your gas analyser.
As explained at the top of this page, we have no such file for these data, so we simulate one:
a breathing rhythm of about 15 breaths per minute, in which each exhalation rises to an end-tidal value that drifts slowly around 38 mmHg.
The recording starts 30 seconds before the functional run and ends 30 seconds after it.

```{code-cell} ipython3
:tags: [hide-input]

from scipy import signal


def simulate_co2(duration, samp_freq, seed=0):
    """Simulate a CO2 trace, in mmHg, as recorded by a gas analyser during free breathing."""
    rng = np.random.default_rng(seed)
    n_samples = int(duration * samp_freq)

    def slow_fluctuation(cutoff, std):
        """Low-pass filtered noise with the requested standard deviation."""
        sos = signal.butter(2, cutoff, fs=samp_freq, output="sos")
        noise = signal.sosfiltfilt(sos, rng.standard_normal(n_samples))
        return std * noise / noise.std()

    # Breathing rate (Hz) and end-tidal CO2 (mmHg) both drift slowly over time
    breathing_rate = 0.25 + slow_fluctuation(0.02, 0.02)
    petco2 = 38 + slow_fluctuation(0.03, 1.5)

    # CO2 is close to zero during inhalation and climbs to its end-tidal value
    # at the end of each exhalation.
    phase = 2 * np.pi * np.cumsum(breathing_rate) / samp_freq
    exhalation = 1 / (1 + np.exp(-8 * np.sin(phase)))
    plateau = 0.85 + 0.3 * ((phase / (2 * np.pi)) % 1)
    co2 = petco2 * exhalation * np.clip(plateau, 0, 1)
    return co2 + rng.normal(0, 0.3, n_samples)


optcom_img = nb.load(optcom_file)
n_vols = optcom_img.shape[3]
t_r = float(optcom_img.header.get_zooms()[3])

# Set the sampling frequency of the simulated recording
samp_freq = 100
co2_path = os.path.join(out_dir, "co2data.tsv")
np.savetxt(co2_path, simulate_co2(n_vols * t_r + 60, samp_freq, seed=2026), fmt="%.4f")
```

### Peak detection

Here's an example with [Physiopy's `peakdet`](https://peakdet.readthedocs.io/en/latest/).

```{code-cell} ipython3
import peakdet as pk

data = np.genfromtxt(co2_path)
```

```{code-cell} ipython3
co2 = pk.Physio(data, fs=samp_freq)

# Interpolation (downsampling), then filtering.
co2 = pk.operations.interpolate_physio(co2, 40.0, kind='linear')
co2 = pk.operations.filter_physio(co2, 2, method='lowpass', order=7)

# Automatic peak detection, then manual editing
# If you see that not enough/too many peaks get selected,
# you might tune the threshold and distance.
co2 = pk.operations.peakfind_physio(co2, thresh=0.5, dist=120)
# edit_physio opens an interactive window to add or remove peaks by hand.
# It cannot run while this book is built, so it is commented out here.
# co2 = pk.operations.edit_physio(co2)

co2_file = pk.save_physio(os.path.join(out_dir, "co2.phys"), co2)
print(f"{co2.peaks.size} peaks detected")
```

If your file has several columns, select the one containing the $CO_2$ data when creating the `Physio` object (e.g., `data[:, channel]`).

```{code-cell} ipython3
:tags: [hide-input]

time = np.arange(co2.data.size) / co2.fs
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(time, co2.data, color="gray", linewidth=1, label="$CO_2$ trace")
ax.plot(time[co2.peaks], co2.data[co2.peaks], "o-", color="tab:red", markersize=3, label="End-tidal peaks")
ax.set_xlim(0, 120)
ax.set_xlabel("Time (s)", fontsize=14)
ax.set_ylabel("$CO_2$ (mmHg)", fontsize=14)
ax.legend(loc="lower right", fontsize=12)
fig.tight_layout()
glue_figure("figure_cvr_co2_trace", fig, display=False)
```

```{figure} figures/figure_cvr_co2_trace.png
:name: figure_cvr_co2_trace
:align: center

The first two minutes of the simulated $CO_2$ trace, after filtering, with the end-tidal peaks detected by `peakdet`.
```

With this last output, we can run `phys2cvr`!

```{code-cell} ipython3
:tags: [hide-output]

p2c(
    fname_func=optcom_file,             # The optimally combined data
    fname_co2=co2_file,                 # The CO2 trace with peaks
    fname_mask=mask_file,               # A mask to limit the analysis to
    fname_roi=roi_file,                 # A ROI to use as signal reference
    outdir=os.path.join(out_dir, "co2"),  # Output folder
    run_regression=True,                # Run the analysis!
    lag_max=9,                          # Maximum lag to consider
    lag_step=0.3,                       # Lag step to consider
    l_degree=2,                         # Legendre polynomials degrees
    denoise_matrix_file=[motion_file],  # All the noise you want to add
    orthogonalised_matrix_file=[rejected_file],
    extra_matrix_file=[accepted_file],
    scale_factor=None,                  # If CO2 trace is not in mmHg, you can scale the final map accordingly
)
```

If you didn't use `peakdet` for your $CO_2$ processing and peak detection, add the parameter `fname_pidx='../co2_peak_indices'` to the list.

`phys2cvr` will compute the $P_{ET}CO_2hrf$ trace by linearly interpolating the end-tidal peaks of your $CO_2$ trace and convolving the former with a canonical HRF. You can select another response function (as well as skipping the interpolation) using the parameter `response_function`.

Among other files, `phys2cvr` writes a CVR map, a lag map, and a t-statistic map to the output folder.

```{code-cell} ipython3
print("\n".join(sorted(f for f in os.listdir(os.path.join(out_dir, "co2")) if f.endswith(".nii.gz"))))
```

```{code-cell} ipython3
:tags: [hide-input]

def plot_cvr_maps(cvr_dir, name):
    """Plot the CVR map, and the lag map if there is one, that phys2cvr wrote to a folder."""
    bg_img = image.mean_img(optcom_file)
    out_prefix = os.path.join(cvr_dir, f"{prefix}_desc-optcom_bold")
    cvr_file = f"{out_prefix}_cvr.nii.gz"
    if not os.path.isfile(cvr_file):
        # Without the lagged regression, phys2cvr only writes the map at the bulk shift
        cvr_file = f"{out_prefix}_cvr_simple.nii.gz"

    cvr_data = nb.load(cvr_file).get_fdata()
    vmax = np.percentile(np.abs(cvr_data[roi]), 98)

    lag_file = f"{out_prefix}_lag_mkrel.nii.gz"
    has_lag = os.path.isfile(lag_file)
    n_rows = 2 if has_lag else 1
    fig, axes = plt.subplots(nrows=n_rows, figsize=(12, 3.5 * n_rows), squeeze=False)
    axes = axes.ravel()
    plotting.plot_stat_map(
        cvr_file,
        bg_img=bg_img,
        vmax=vmax,
        display_mode="z",
        cut_coords=6,
        title="CVR",
        figure=fig,
        axes=axes[0],
    )
    if has_lag:
        plotting.plot_stat_map(
            lag_file,
            bg_img=bg_img,
            vmax=9,
            display_mode="z",
            cut_coords=6,
            title="Lag (s)",
            figure=fig,
            axes=axes[1],
        )

    glue_figure(name, fig, display=False)


plot_cvr_maps(os.path.join(out_dir, "co2"), "figure_cvr_co2_maps")
```

```{figure} figures/figure_cvr_co2_maps.png
:name: figure_cvr_co2_maps
:align: center

CVR and lag maps estimated from the simulated $CO_2$ trace.
Because the trace is simulated, it is unrelated to the BOLD signal, and these maps only illustrate the outputs.
```


## Pre-made $P_{ET}CO_2$ traces

If you have a pre-made $P_{ET}CO_2$ trace, simply skip the steps to make one.

To have such a trace for this example, we interpolate the end-tidal peaks detected above and save the result as a text file.

```{code-cell} ipython3
samples = np.arange(co2.data.size)
petco2 = np.interp(samples, co2.peaks, co2.data[co2.peaks])
petco2_file = os.path.join(out_dir, "petco2_trace.1D")
np.savetxt(petco2_file, petco2, fmt="%.4f")
```

Because a text file does not store its sampling frequency, we also need to declare it with `freq`.

```{code-cell} ipython3
:tags: [hide-output]

p2c(
    fname_func=optcom_file,             # The optimally combined data
    fname_co2=petco2_file,              # The pre-made PetCO2 trace
    freq=co2.fs,                        # The sampling frequency of the PetCO2 trace
    fname_mask=mask_file,               # A mask to limit the analysis to
    fname_roi=roi_file,                 # A ROI to use as signal reference
    outdir=os.path.join(out_dir, "petco2"),  # Output folder
    run_regression=True,                # Run the analysis!
    lag_max=9,                          # Maximum lag to consider
    lag_step=0.3,                       # Lag step to consider
    l_degree=2,                         # Legendre polynomials degrees
    denoise_matrix_file=[motion_file],  # All the noise you want to add
    orthogonalised_matrix_file=[rejected_file],
    extra_matrix_file=[accepted_file],
    scale_factor=None,                  # If CO2 trace is not in mmHg, you can scale the final map accordingly
    comp_endtidal=False,                # DO NOT compute PetCO2 internally
    response_function=None,             # DO NOT interpolate the PetCO2 trace.
)
```


## Task designs (timeseries)

Simply use your task design as $CO_2$ trace. You can interpolate it with an HRF or a respiratory response function internally, or do this beforehand.

The data used in this book were not acquired during a respiratory task, so this call is shown but not run.

```python
p2c(
    fname_func=optcom_file,             # The optimally combined data
    fname_co2='../task_design.1D',      # The task design
    freq=40,                            # The sampling frequency of the task design
    fname_mask=mask_file,               # A mask to limit the analysis to
    fname_roi=roi_file,                 # A ROI to use as signal reference
    outdir=os.path.join(out_dir, "task"),  # Output folder
    run_regression=True,                # Run the analysis!
    lag_max=9,                          # Maximum lag to consider
    lag_step=0.3,                       # Lag step to consider
    l_degree=2,                         # Legendre polynomials degrees
    denoise_matrix_file=[motion_file],  # All the noise you want to add
    orthogonalised_matrix_file=[rejected_file],
    extra_matrix_file=[accepted_file],
    scale_factor=None,                  # If CO2 trace is not in mmHg, you can scale the final map accordingly
    comp_endtidal=False,                # DO NOT compute end-tidal trace
    response_function='rrf',            # Interpolate with RRF
)
```


## BOLD signals averages from a ROI mask.

This is the simplest way to run `phys2cvr`, using the average BOLD signal from a desired ROI.
It needs no $CO_2$ recording at all, so here nothing is simulated.

```{note}
The lagged regression is switched off in this call.
When the reference signal is the ROI average, its bulk shift relative to itself is zero,
and `phys2cvr` (at least up to version 0.28.4) then fails to build the lagged regressors.
This call therefore only produces the CVR map at the bulk shift.
```

```{code-cell} ipython3
:tags: [hide-output]

p2c(
    fname_func=optcom_file,             # The optimally combined data
    fname_mask=mask_file,               # A mask to limit the analysis to
    fname_roi=roi_file,                 # A ROI to use as signal reference
    outdir=os.path.join(out_dir, "roi"),  # Output folder
    run_regression=True,                # Run the analysis!
    lagged_regression=False,            # Only estimate CVR at the bulk shift (see the note above)
    l_degree=2,                         # Legendre polynomials degrees
    denoise_matrix_file=[motion_file],  # All the noise you want to add
    orthogonalised_matrix_file=[rejected_file],
    extra_matrix_file=[accepted_file],
    comp_endtidal=False,                # DO NOT compute end-tidal trace
    response_function=None,             # DO NOT interpolate the trace
    apply_filter=True,                  # Filter the average BOLD signal (optional)
    highcut=0.04,                       # Lowpass frequency
    lowcut=0.02,                        # Highpass frequency
    butter_order=9,                     # Butterworth filter order
)
```

```{code-cell} ipython3
:tags: [hide-input]

plot_cvr_maps(os.path.join(out_dir, "roi"), "figure_cvr_roi_maps")
```

```{figure} figures/figure_cvr_roi_maps.png
:name: figure_cvr_roi_maps
:align: center

Relative CVR map estimated from the average gray matter signal.
```


## Final remarks

Remember to cite the right papers and resources depending on the desired pipeline!

For more info about `phys2cvr`, as well as knowing what to cite in which case, check out [its documentation](https://phys2cvr.readthedocs.io/en/latest/index.html).

Not a python person? Prefer to run bash CLI commands? Not a problem.
`phys2cvr` is in fact primarily [designed to be used via bash CLI](https://phys2cvr.readthedocs.io/en/latest/usage/cli.html)

```bash
phys2cvr --help
```

For more info about ME CVR, check out {cite:t}`MOIA2021117914`, {cite:t}`cohen2019improving`, and {cite:t}`Cohen2025RS`.

If you want to contribute to `phys2cvr`, have requests, comments, or questions, feel free to [open an issue](https://github.com/smoia/phys2cvr/issues)!
