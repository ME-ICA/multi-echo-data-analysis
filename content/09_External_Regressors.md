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

# External Regressors in `tedana` Decision Trees

```{admonition} Abbreviations
- CSF = CerebroSpinal Fluid
- FD = Framewise Displacement
- IC = Independent Component
- ROI = Region of Interest
- RETROICOR = RETROspective Image CORrection
- RV = Respiration Variation
- HR = Heart Rate
```

`tedana` classifies ICA components mostly from how their signal depends on echo time (TE).
Sometimes we also know what a noise source or the signal of interest looks like over time:
a recording of the participant's breathing, the head motion estimated during preprocessing,
the average signal in a region, or the timing of the task.
Decision trees can compare each component's time series to such **external regressors**,
and use the fit to reject components that look like noise or to keep components that look like the task.

This page builds the four kinds of external regressors listed in
[ME-ICA/multi-echo-data-analysis#94](https://github.com/ME-ICA/multi-echo-data-analysis/issues/94):

1. physiological regressors computed from cardiac and respiratory recordings,
2. average time series from ROIs (see [ME-ICA/tedana#1175](https://github.com/ME-ICA/tedana/issues/1175)),
3. confounds estimated by fMRIPrep, and
4. task regressors.

It then runs `tedana` with a decision tree that uses them, and compares the result with a tree that does not.

## The data

We use one run of the localizer task from the
[Dense Investigation of Variability of Affect (DIVA) dataset](https://openneuro.org/datasets/ds002278)
(see the [preprint](https://doi.org/10.1101/2024.04.16.589598)).
The participant alternated between 14-second blocks of finger tapping (motor) and of watching a flickering checkerboard (visual),
separated by 14 seconds of rest.
In half of the blocks, music played at the same time (auditory).
The run was acquired with four echoes, and cardiac (ECG) and respiratory (belt) signals were recorded during the scan.
The echoes were preprocessed with fMRIPrep and stay in the space of the run's BOLD reference image.
See [Download Data](00_Download_Data.md) for the list of files.

```{code-cell} ipython3
import json
import os
import shutil

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
from book_utils import atlas_mask, glue_figure, load_diva, warp_mni_roi_to_run
from nilearn import datasets, masking, plotting
from nilearn.glm.first_level import make_first_level_design_matrix
from scipy import signal
from tedana import workflows

data_path = os.path.abspath("../data")
data = load_diva(data_path)
prefix = data["prefix"]
out_dir = os.path.join(data_path, "external-regressors")
os.makedirs(out_dir, exist_ok=True)
```

Every regressor needs one value per volume, so we start with the timing of the volumes.
fMRIPrep corrected the data for slice timing, and its sidecar records, in `StartTime`,
the time within each volume that the corrected data represent.
That is when we sample every regressor.
fMRIPrep also flags the first volumes, acquired before the signal reached a steady state, as non-steady-state outliers.
We will drop them in `tedana` with `dummy_scans`, which drops the same rows from the external regressors.

```{code-cell} ipython3
with open(data["optcom_json"], "r") as fo:
    bold_metadata = json.load(fo)

t_r = bold_metadata["RepetitionTime"]
n_vols = nb.load(data["optcom"]).shape[3]
frame_times = np.arange(n_vols) * t_r + bold_metadata["StartTime"]

confounds = pd.read_table(data["confounds"])
dummy_scans = confounds.filter(like="non_steady_state_outlier").any(axis=1).to_numpy()
n_dummy = int(np.argmin(dummy_scans)) if dummy_scans.any() else 0
# dummy_scans only removes volumes at the start of the run
assert dummy_scans[:n_dummy].all() and not dummy_scans[n_dummy:].any()
# The times of the volumes that tedana keeps
volume_times = frame_times[n_dummy:]
print(f"{n_vols} volumes, TR = {t_r} s, {n_dummy} non-steady-state volume(s)")
```

## 1. Physiological regressors

The physiological recording is a BIDS `_physio.tsv.gz` file, with its sampling frequency, columns and start time in its sidecar.

```{code-cell} ipython3
:tags: [hide-output]

with open(data["physio_json"], "r") as fo:
    physio_metadata = json.load(fo)

print(json.dumps(physio_metadata, indent=4))
physio = pd.read_table(data["physio"], header=None, names=physio_metadata["Columns"])
samp_freq = physio_metadata["SamplingFrequency"]
```

### Aligning the recording with the scan

According to BIDS, `StartTime` is the time of the first sample relative to the first volume,
so a positive value means that the recording started *after* the scan.
That is not the case here: the recording starts before the scan, and the sign of `StartTime` is flipped.
The `trigger` channel settles the question.
It is high while the scanner acquires volumes, so its first high sample marks the start of the first volume,
and the length of the high period should match the duration of the run.

```{code-cell} ipython3
scan_on = physio["trigger"].to_numpy() > (physio["trigger"].max() / 2)
start_time = -np.argmax(scan_on) / samp_freq
physio_times = np.arange(physio.shape[0]) / samp_freq + start_time

print(f"StartTime in the sidecar: {physio_metadata['StartTime']} s")
print(f"StartTime from the trigger: {start_time:.3f} s")
print(f"The trigger is high for {scan_on.sum() / samp_freq:.2f} s; the run lasts {n_vols * t_r:.2f} s")
```

From now on, `physio_times` holds the time of each sample relative to the start of the scan.

### Cardiac and respiratory peaks

The ECG was recorded during the scan, so it carries a gradient artifact at the rate at which the scanner acquires slices.
With 48 slices acquired with a multiband factor of three, there are 16 shots per 1.5-second volume,
so the artifact sits at 10.7 Hz and its harmonics.
A band-pass filter below that frequency leaves the R peaks, which `scipy` can find.
The respiratory belt is slow, so a low-pass filter is enough.

```{code-cell} ipython3
sos = signal.butter(4, [0.5, 8], btype="bandpass", fs=samp_freq, output="sos")
cardiac = signal.sosfiltfilt(sos, physio["cardiac"].to_numpy())
# R peaks are at least 0.6 s apart (heart rate < 100 bpm) and stand out from the rest of the trace
card_peaks, _ = signal.find_peaks(cardiac, distance=int(0.6 * samp_freq), prominence=np.std(cardiac))
# Drop the occasional T wave that passes the first criteria
card_peaks = card_peaks[cardiac[card_peaks] > 0.5 * np.median(cardiac[card_peaks])]
beat_times = physio_times[card_peaks][1:]
heart_rate = 60 / np.diff(physio_times[card_peaks])

sos = signal.butter(4, 1, btype="lowpass", fs=samp_freq, output="sos")
respiration = signal.sosfiltfilt(sos, physio["respiration"].to_numpy())
resp_peaks, _ = signal.find_peaks(respiration, distance=int(2 * samp_freq), prominence=0.5 * np.std(respiration))
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(nrows=3, figsize=(14, 9))
t_start, t_end = 100, 115
window = (physio_times >= t_start) & (physio_times <= t_end)
axes[0].plot(physio_times[window], physio["cardiac"].to_numpy()[window], color="gray", linewidth=0.5, label="Raw ECG")
axes[0].plot(physio_times[window], cardiac[window], color="black", linewidth=1, label="Filtered ECG")
peaks_in_window = card_peaks[window[card_peaks]]
axes[0].plot(physio_times[peaks_in_window], cardiac[peaks_in_window], "o", color="tab:red", label="R peaks")
axes[0].set_xlim(t_start, t_end)
axes[0].set_ylabel("ECG (a.u.)", fontsize=14)
axes[0].legend(loc="upper right", ncols=3)

axes[1].plot(beat_times, heart_rate, color="tab:red")
axes[1].set_xlim(frame_times[0], frame_times[-1])
axes[1].set_ylabel("Heart rate (bpm)", fontsize=14)

t_start, t_end = 100, 160
window = (physio_times >= t_start) & (physio_times <= t_end)
axes[2].plot(physio_times[window], respiration[window], color="black")
peaks_in_window = resp_peaks[window[resp_peaks]]
axes[2].plot(physio_times[peaks_in_window], respiration[peaks_in_window], "o", color="tab:blue")
axes[2].set_xlim(t_start, t_end)
axes[2].set_ylabel("Respiration (a.u.)", fontsize=14)
axes[2].set_xlabel("Time since the first volume (s)", fontsize=14)
fig.tight_layout()
glue_figure("figure_extreg_physio", fig)
```

```{figure} figures/figure_extreg_physio.png
:name: figure_extreg_physio
:align: center

Top: 15 seconds of the ECG before and after filtering, with the detected R peaks.
The sharp spikes in the raw trace are the gradient artifact.
Middle: heart rate over the whole recording.
Bottom: one minute of the filtered respiratory trace, with the detected peaks.
```

### RETROICOR

RETROICOR {cite:p}`glover2000image` models the effect of the cardiac and respiratory cycles on the signal
as a Fourier series of their phase at the time each volume was acquired.
The cardiac phase goes from 0 to $2\pi$ between consecutive R peaks.
The respiratory phase also accounts for the depth of the breath:
its magnitude comes from the amplitude of the respiratory trace relative to its histogram,
and its sign from whether the participant is breathing in or out.
The functions below compute both phases and their Fourier series.

```{code-cell} ipython3
:tags: [hide-input]

def cardiac_phase(times, peak_times):
    """Phase of the cardiac cycle, between 0 and 2 pi, at the requested times."""
    phase = np.full(times.shape, np.nan)
    idx = np.searchsorted(peak_times, times, side="right") - 1
    valid = (idx >= 0) & (idx < peak_times.size - 1)
    previous_peak = peak_times[idx[valid]]
    next_peak = peak_times[idx[valid] + 1]
    phase[valid] = 2 * np.pi * (times[valid] - previous_peak) / (next_peak - previous_peak)
    return phase


def respiratory_phase(resp, n_bins=100):
    """Phase of the respiratory cycle at each sample (Glover et al., 2000)."""
    resp = resp - resp.min()
    hist, edges = np.histogram(resp, bins=n_bins)
    cumulative = np.cumsum(hist) / hist.sum()
    bin_idx = np.clip(np.digitize(resp, edges[1:-1]), 0, n_bins - 1)
    # The trace is low-pass filtered, so noise does not flip the sign of its derivative
    direction = np.sign(np.gradient(resp))
    return np.pi * cumulative[bin_idx] * direction


def fourier_expansion(phase, order, name):
    """Sine and cosine of each multiple of the phase, up to ``order``."""
    columns = {}
    for m in range(1, order + 1):
        columns[f"{name}_cos{m}"] = np.cos(m * phase)
        columns[f"{name}_sin{m}"] = np.sin(m * phase)

    return pd.DataFrame(columns)
```

We sample each phase at the volume times and expand it to the second order, which gives eight regressors.
The respiratory phase jumps from $\pi$ to $-\pi$ at the top of each breath,
so we take the value of the nearest sample instead of interpolating across the jump.

```{code-cell} ipython3
card_phase = cardiac_phase(frame_times, physio_times[card_peaks])
resp_phase_all = respiratory_phase(respiration)
nearest_sample = np.clip(np.round((frame_times - physio_times[0]) * samp_freq).astype(int), 0, physio_times.size - 1)
resp_phase = resp_phase_all[nearest_sample]

retroicor = pd.concat(
    [
        fourier_expansion(card_phase, 2, "physio_card"),
        fourier_expansion(resp_phase, 2, "physio_resp"),
    ],
    axis=1,
)
```

```{code-cell} ipython3
:tags: [hide-input]

fig = plt.figure(figsize=(14, 12))
grid_spec = fig.add_gridspec(nrows=4, height_ratios=[1, 1, 1.5, 1.5], hspace=0.5)
t_start, t_end = 100, 130
window = (physio_times >= t_start) & (physio_times <= t_end)
volumes_in_window = (frame_times >= t_start) & (frame_times <= t_end)
for i_row, (label, phase_trace, phase_at_volumes, color) in enumerate(
    [
        ("Cardiac", cardiac_phase(physio_times[window], physio_times[card_peaks]), card_phase, "tab:red"),
        ("Respiratory", resp_phase_all[window], resp_phase, "tab:blue"),
    ]
):
    ax = fig.add_subplot(grid_spec[i_row])
    ax.plot(physio_times[window], phase_trace, color=color, linewidth=1, label="Phase")
    ax.plot(frame_times[volumes_in_window], phase_at_volumes[volumes_in_window], "o", color="black", markersize=4, label="At each volume")
    ax.set_xlim(t_start, t_end)
    ax.set_ylabel(f"{label}\nphase (rad)", fontsize=12)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2, frameon=False)

ax.set_xlabel("Time since the first volume (s)", fontsize=12)

for i_row, kind in enumerate(["card", "resp"]):
    ax = fig.add_subplot(grid_spec[2 + i_row])
    columns = [column for column in retroicor.columns if f"_{kind}_" in column]
    for j_col, column in enumerate(columns):
        ax.plot(frame_times, retroicor[column] - 2.5 * j_col, linewidth=0.8)

    ax.set_yticks(-2.5 * np.arange(len(columns)))
    ax.set_yticklabels(columns)
    ax.set_xlim(frame_times[0], frame_times[-1])

ax.set_xlabel("Time since the first volume (s)", fontsize=12)
glue_figure("figure_extreg_retroicor", fig)
```

```{figure} figures/figure_extreg_retroicor.png
:name: figure_extreg_retroicor
:align: center

Top: 30 seconds of the cardiac and respiratory phases, with their values at the volume times (black).
Bottom: the eight RETROICOR regressors over the whole run.
The heart beats faster than the volumes are acquired, so the cardiac regressors are aliased and look like noise;
that is expected, since the cardiac noise in the data is aliased in the same way.
```

### Respiration variation and heart rate

Slower changes in the depth of breathing and in heart rate change arterial $CO_2$ and blood flow,
and they affect the BOLD signal over several seconds.
Following {cite:t}`chang2009influence`, we compute the respiration variation (RV), the standard deviation of the respiratory trace in a 6-second window,
and the heart rate averaged over the same window.
They are then convolved with the respiration response function {cite:p}`birn2008respiration`
and the cardiac response function {cite:p}`chang2009influence`, respectively.
We do this on a fine time grid that starts with the recording, so that the seconds recorded before the scan contribute to the first volumes,
and then sample the result at the volume times.

```{code-cell} ipython3
:tags: [hide-input]

def respiration_response_function(t):
    """Birn et al. (2008)."""
    return 0.6 * t**2.1 * np.exp(-t / 1.6) - 0.0023 * t**3.54 * np.exp(-t / 4.25)


def cardiac_response_function(t):
    """Chang et al. (2009)."""
    return 0.6 * t**2.7 * np.exp(-t / 1.6) - 16 / np.sqrt(2 * np.pi * 9) * np.exp(-0.5 * (t - 12) ** 2 / 9)
```

```{code-cell} ipython3
grid_step = 0.1
grid = np.arange(physio_times[0], physio_times[-1], grid_step)
window_length = round(6 / grid_step) + 1

rv = pd.Series(np.interp(grid, physio_times, respiration)).rolling(window_length, center=True, min_periods=1).std().to_numpy()
hr = pd.Series(np.interp(grid, beat_times, heart_rate)).rolling(window_length, center=True, min_periods=1).mean().to_numpy()

kernel_times = np.arange(0, 40, grid_step)
rrf = respiration_response_function(kernel_times)
crf = cardiac_response_function(kernel_times)
rv_conv = np.convolve(rv - rv.mean(), rrf)[: grid.size]
hr_conv = np.convolve(hr - hr.mean(), crf)[: grid.size]

physio_slow = pd.DataFrame(
    {
        "physio_rv": np.interp(frame_times, grid, rv_conv),
        "physio_hr": np.interp(frame_times, grid, hr_conv),
    }
)
physio_regressors = pd.concat([retroicor, physio_slow], axis=1)
```

```{code-cell} ipython3
:tags: [hide-input]

fig = plt.figure(figsize=(14, 7))
grid_spec = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1, 4], wspace=0.15, hspace=0.45)
for i_row, (name, raw, kernel, kernel_name, regressor, color) in enumerate(
    [
        ("RV", rv, rrf, "Respiration response function", physio_slow["physio_rv"], "tab:blue"),
        ("HR", hr, crf, "Cardiac response function", physio_slow["physio_hr"], "tab:red"),
    ]
):
    ax = fig.add_subplot(grid_spec[i_row, 0])
    ax.plot(kernel_times, kernel, color=color)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlim(kernel_times[0], kernel_times[-1])
    ax.set_yticks([])
    ax.set_title(kernel_name, fontsize=11)
    ax.set_xlabel("Time (s)", fontsize=11)

    ax = fig.add_subplot(grid_spec[i_row, 1])
    ax.plot(grid, (raw - raw.mean()) / raw.std(), color="gray", linewidth=1, label=f"{name} in 6-second windows")
    ax.plot(frame_times, (regressor - regressor.mean()) / regressor.std(), color=color, linewidth=2, label=f"Convolved {name}, at each volume")
    ax.set_xlim(frame_times[0], frame_times[-1])
    ax.set_ylabel(f"{name} (z)", fontsize=12)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2, frameon=False)

ax.set_xlabel("Time since the first volume (s)", fontsize=12)
glue_figure("figure_extreg_rv_hr", fig)
```

```{figure} figures/figure_extreg_rv_hr.png
:name: figure_extreg_rv_hr
:align: center

Left: the response functions.
Right: RV and HR in 6-second windows (gray), and the regressors after convolution, sampled at the volume times (color), each scaled to unit variance.
The recording starts about 9 seconds before the first volume, so the convolution already has a history at the first volume; the plot shows the run only.
```

## 2. ROI time series

`tedana` takes a table of time series, so the average signal in an ROI has to be computed before `tedana` runs
([ME-ICA/tedana#1175](https://github.com/ME-ICA/tedana/issues/1175)).
As an example, we use the lateral ventricles from the Harvard-Oxford subcortical atlas, as a CSF ROI.

```{important}
This ROI is here to show *how* an ROI time series can be used in a decision tree, not to recommend this particular ROI.
fMRIPrep already writes a similar regressor, the `csf` column of its confounds file,
which is the average signal in an eroded CSF mask from the participant's own segmentation.
An ROI that fMRIPrep does not provide, such as a draining vein or a region with a known artifact, is where this approach is most useful.
```

The ROI is in a standard space, so we warp it into the space of the BOLD data with the transforms that fMRIPrep wrote:
`_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5` (a nonlinear transform between the template and the participant's T1w image)
and `_from-run_to-T1w_mode-image_desc-coreg_xfm.txt` (a rigid transform between the run's BOLD reference and the T1w image).
`warp_mni_roi_to_run` chains them with [NiTransforms](https://github.com/nipy/nitransforms) and resamples the ROI onto the BOLD reference.

```{code-cell} ipython3
:tags: [hide-output]

harvard_oxford = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
ventricles_mni = atlas_mask(harvard_oxford, ["Left Lateral Ventricle", "Right Lateral Ventricle"])
ventricles = warp_mni_roi_to_run(ventricles_mni, data)
ventricles_file = os.path.join(out_dir, f"{prefix}_space-run_desc-ventricles_mask.nii.gz")
ventricles.to_filename(ventricles_file)
```

```{note}
The atlases distributed with FSL (and fetched by nilearn) are in the `MNI152NLin6Asym` template,
while fMRIPrep's transform starts from `MNI152NLin2009cAsym`.
The two templates differ by a few millimeters, which matters little for large ROIs.
For precise work, add the transform between the two templates from [TemplateFlow](https://www.templateflow.org) to the chain.
```

```{code-cell} ipython3
:tags: [hide-input]

fig = plt.figure(figsize=(14, 4.5), facecolor="black")
plotting.plot_roi(
    ventricles,
    bg_img=data["boldref"],
    display_mode="ortho",
    cut_coords=plotting.find_xyz_cut_coords(ventricles),
    cmap="autumn",
    colorbar=False,
    draw_cross=False,
    black_bg=True,
    dim=0,
    figure=fig,
)
glue_figure("figure_extreg_ventricles", fig)
```

```{figure} figures/figure_extreg_ventricles.png
:name: figure_extreg_ventricles
:align: center

The lateral ventricle ROI after warping it to the space of the BOLD data, on top of the BOLD reference image.
```

We average the optimally combined echoes that fMRIPrep writes (`_desc-preproc_bold.nii.gz` without an `echo` entity) in the ROI.
`tedana`'s own optimal combination would be slightly different, but this is close enough to describe a noise source.

```{code-cell} ipython3
ventricle_timeseries = masking.apply_mask(data["optcom"], ventricles_file).mean(axis=1)
```

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(14, 3.5))
for values, label, color in [(ventricle_timeseries, "Ventricle ROI", "black"), (confounds["csf"].to_numpy(), "fMRIPrep's csf column", "tab:blue")]:
    values = values[n_dummy:]
    ax.plot(volume_times, (values - values.mean()) / values.std(), color=color, linewidth=1.2, label=label)

ax.set_xlim(volume_times[0], volume_times[-1])
ax.set_xlabel("Time since the first volume (s)", fontsize=12)
ax.set_ylabel("Signal (z)", fontsize=12)
ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2, frameon=False)
glue_figure("figure_extreg_ventricle_timeseries", fig)
```

```{figure} figures/figure_extreg_ventricle_timeseries.png
:name: figure_extreg_ventricle_timeseries
:align: center

The average signal in the ventricle ROI and fMRIPrep's `csf` regressor, after the non-steady-state volume, each scaled to unit variance.
They are similar but not identical, because the masks differ.
```

## 3. fMRIPrep confounds

fMRIPrep writes many candidate confounds.
Here we use the six head motion parameters and their temporal derivatives.

```{code-cell} ipython3
motion_columns = [f"{kind}_{axis}{suffix}" for kind in ("trans", "rot") for axis in ("x", "y", "z") for suffix in ("", "_derivative1")]
# The first value of each derivative is undefined
motion = confounds[motion_columns].fillna(0)
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(nrows=3, figsize=(14, 8), sharex=True)
for axis, color in zip("xyz", ["tab:blue", "tab:orange", "tab:green"]):
    axes[0].plot(frame_times, confounds[f"trans_{axis}"], color=color, label=axis)
    axes[1].plot(frame_times, np.rad2deg(confounds[f"rot_{axis}"]), color=color, label=axis)

axes[0].set_ylabel("Translation (mm)", fontsize=12)
axes[0].legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=3, frameon=False)
axes[1].set_ylabel("Rotation (degrees)", fontsize=12)
axes[2].plot(frame_times, confounds["framewise_displacement"], color="black")
axes[2].set_ylabel("FD (mm)", fontsize=12)
axes[2].set_xlim(frame_times[0], frame_times[-1])
axes[2].set_xlabel("Time since the first volume (s)", fontsize=12)
fig.tight_layout()
glue_figure("figure_extreg_motion", fig)
```

```{figure} figures/figure_extreg_motion.png
:name: figure_extreg_motion
:align: center

The translation and rotation parameters estimated by fMRIPrep, and the framewise displacement that summarizes them.
The regressors are the six parameters and their derivatives; framewise displacement is shown only for reference.
```

## 4. Task regressors

The localizer has three conditions.
A block that combines tapping and music counts as both a motor and an auditory block,
so we split each `trial_type` into its parts and model each condition separately.
nilearn convolves the blocks with a hemodynamic response function and samples them at the volume times.

```{code-cell} ipython3
events = pd.read_table(data["events"])
events = events.assign(trial_type=events["trial_type"].str.split("/")).explode("trial_type")
events = events[["onset", "duration", "trial_type"]].sort_values("onset").reset_index(drop=True)

task_design = make_first_level_design_matrix(frame_times, events, hrf_model="spm", drift_model=None)
task_regressors = task_design[["motor", "visual", "auditory"]].reset_index(drop=True)
```

```{code-cell} ipython3
:tags: [hide-input]

condition_colors = {"motor": "tab:orange", "auditory": "tab:green", "visual": "tab:purple"}
fig, axes = plt.subplots(nrows=3, figsize=(14, 6), sharex=True)
for ax, (condition, color) in zip(axes, condition_colors.items()):
    for _, event in events[events["trial_type"] == condition].iterrows():
        ax.axvspan(event["onset"], event["onset"] + event["duration"], color=color, alpha=0.2)

    ax.plot(frame_times, task_regressors[condition], color=color, linewidth=2)
    ax.set_ylabel(condition, fontsize=12)
    ax.set_yticks([])

axes[-1].set_xlim(frame_times[0], frame_times[-1])
axes[-1].set_xlabel("Time since the first volume (s)", fontsize=12)
fig.tight_layout()
glue_figure("figure_extreg_task", fig)
```

```{figure} figures/figure_extreg_task.png
:name: figure_extreg_task
:align: center

The blocks of each condition (shaded) and the task regressors.
```

## Putting the regressors together

`tedana` reads external regressors from a tab-separated file with one column per regressor and one row per volume, *including* the dummy scans.
The names of the columns are how the decision tree refers to the regressors, so we give each kind a prefix.

```{code-cell} ipython3
external = pd.concat(
    [
        physio_regressors,
        pd.DataFrame({"csf_ventricles": ventricle_timeseries}),
        motion.add_prefix("mot_"),
        task_regressors.add_prefix("task_"),
    ],
    axis=1,
)
assert external.shape[0] == n_vols
assert not external.isna().any().any()
external_file = os.path.join(out_dir, f"{prefix}_desc-external_regressors.tsv")
external.to_csv(external_file, sep="\t", index=False)
```

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(14, 8))
zscored = (external - external.mean()) / external.std()
ax.imshow(
    zscored.T,
    aspect="auto",
    cmap="gray",
    vmin=-3,
    vmax=3,
    extent=[frame_times[0], frame_times[-1], external.shape[1] - 0.5, -0.5],
    interpolation="none",
)
ax.set_yticks(range(external.shape[1]))
ax.set_yticklabels(external.columns)
ax.set_xlabel("Time since the first volume (s)", fontsize=14)
fig.tight_layout()
glue_figure("figure_extreg_regressors", fig)
```

```{figure} figures/figure_extreg_regressors.png
:name: figure_extreg_regressors
:align: center

All external regressors, each scaled to unit variance.
```

## A decision tree that uses external regressors

A decision tree is a JSON file.
It declares how to fit the external regressors in its `external_regressor_config`,
and then uses the resulting metrics in its nodes.
`tedana` comes with demonstration trees that do this.
Our tree, [`decision_trees/external_regressors_tree.json`](https://github.com/ME-ICA/multi-echo-data-analysis/blob/main/content/decision_trees/external_regressors_tree.json),
is `tedana`'s `demo_external_regressors_motion_task_models` tree, edited in three places:
the regular expressions that select the regressors match our column names,
the nuisance model has a partial model for the physiological regressors,
and a node tags the components that the physiological regressors fit.

Here is its `external_regressor_config`:

```{literalinclude} decision_trees/external_regressors_tree.json
:language: json
:start-at: '"external_regressor_config"'
:end-before: '"nodes"'
:dedent: 4
```

It fits two models to each component's time series:

- a **nuisance** model, an F-test of all the nuisance regressors together, with **partial** models that test subsets of them;
- a **task** model, an F-test of the task regressors.

`"detrend": true` adds Legendre polynomials to both models, to absorb slow drifts.
The F-tests compare each model with a model that has only the polynomials,
and $R^2$ is the fraction of the variance left after detrending that the regressors explain.
A partial model is tested by comparing the full model with the full model minus that subset of regressors.
`tedana` selects the regressors by matching the column names to the regular expressions.

The nodes that use these models are near the end of the tree.
First, a component is provisionally rejected if the nuisance model fits it significantly (p < 0.05) and explains more than half of its variance ($R^2$ > 0.5):

```{literalinclude} decision_trees/external_regressors_tree.json
:language: json
:lines: 183-201
:dedent: 8
```

Three nodes that change nothing then tag the rejected components whose fit is driven by motion, CSF or physiological regressors.
This is the one we added for the physiological regressors:

```{literalinclude} decision_trees/external_regressors_tree.json
:language: json
:lines: 244-264
:dedent: 8
```

Finally, a component that was going to be rejected is accepted instead if the task model fits it in the same way
and it has enough TE-dependence (kappa above the kappa elbow):

```{literalinclude} decision_trees/external_regressors_tree.json
:language: json
:lines: 265-285
:dedent: 8
```

:::{dropdown} The whole tree
```{literalinclude} decision_trees/external_regressors_tree.json
:language: json
```
:::

## Running `tedana`

To see what the external regressors change, we need a tree that does everything else the same way.
The demonstration tree is not simply the `minimal` tree with extra nodes:
its early nodes only provisionally reject components, and it can rescue components with high kappa.
So the baseline, [`decision_trees/baseline_tree.json`](https://github.com/ME-ICA/multi-echo-data-analysis/blob/main/content/decision_trees/baseline_tree.json),
is our tree without the `external_regressor_config`, the five nodes that use it, and the metrics and tags that only those nodes need.

:::{dropdown} The baseline tree
```{literalinclude} decision_trees/baseline_tree.json
:language: json
```
:::

We run `tedana` with the baseline tree first.
`tedana` keeps the logs and reports of earlier runs in its output folder, so we start each run from an empty folder.

```{code-cell} ipython3
:tags: [hide-output]

baseline_dir = os.path.join(out_dir, "tedana-baseline")
if os.path.isdir(baseline_dir):
    shutil.rmtree(baseline_dir)

workflows.tedana_workflow(
    data["echo_files"],
    data["echo_times"],
    out_dir=baseline_dir,
    mask=data["mask"],
    prefix=prefix,
    dummy_scans=n_dummy,
    tedpca="mdl",
    tree=os.path.abspath("decision_trees/baseline_tree.json"),
    fixed_seed=42,
    overwrite=True,
)
```

Then we run it again with our tree and the external regressors.
To compare the two classifications, we reuse the ICA decomposition of the first run with `mixing_file`, so both trees classify the same components.

```{code-cell} ipython3
:tags: [hide-output]

external_dir = os.path.join(out_dir, "tedana-external")
if os.path.isdir(external_dir):
    shutil.rmtree(external_dir)

workflows.tedana_workflow(
    data["echo_files"],
    data["echo_times"],
    out_dir=external_dir,
    mask=data["mask"],
    prefix=prefix,
    dummy_scans=n_dummy,
    tree=os.path.abspath("decision_trees/external_regressors_tree.json"),
    external_regressors=external_file,
    mixing_file=os.path.join(baseline_dir, f"{prefix}_desc-ICA_mixing.tsv"),
    fixed_seed=42,
    overwrite=True,
)
```

## Comparing the classifications

The metrics table now has a column for each statistic of each model.

```{code-cell} ipython3
metrics_baseline = pd.read_table(os.path.join(baseline_dir, f"{prefix}_desc-tedana_metrics.tsv"), index_col="Component")
metrics_external = pd.read_table(os.path.join(external_dir, f"{prefix}_desc-tedana_metrics.tsv"), index_col="Component")
print("\n".join(c for c in metrics_external.columns if c not in metrics_baseline.columns))
```

```{code-cell} ipython3
:tags: [hide-input]

summary = pd.DataFrame(
    {
        tree_name: [
            (metrics["classification"] == "accepted").sum(),
            (metrics["classification"] == "rejected").sum(),
            metrics.loc[metrics["classification"] == "rejected", "variance explained"].sum(),
        ]
        for tree_name, metrics in [("without external regressors", metrics_baseline), ("external regressors", metrics_external)]
    },
    index=["accepted components", "rejected components", "variance explained by rejected components (%)"],
)
summary.round(1)
```

These components changed classification:

```{code-cell} ipython3
:tags: [hide-input]

changed = metrics_baseline["classification"] != metrics_external["classification"]
comparison = pd.DataFrame(
    {
        "without external regressors": metrics_baseline["classification"],
        "external regressors": metrics_external["classification"],
        "tags": metrics_external["classification_tags"],
        "kappa": metrics_external["kappa"],
        "rho": metrics_external["rho"],
        "variance explained": metrics_external["variance explained"],
        "R2 nuisance": metrics_external["R2stat nuisance model"],
        "R2 task": metrics_external["R2stat task model"],
    }
)
comparison[changed].round(3)
```

The tags show why the external regressors mattered for each rejected component:

```{code-cell} ipython3
:tags: [hide-input]

tags = metrics_external["classification_tags"].str.split(",").explode()
tags.groupby(tags).size().sort_values(ascending=False).rename("components")
```

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(8, 7))
colors = metrics_external["classification"].map({"accepted": "tab:green", "rejected": "tab:red"})
ax.scatter(
    metrics_external["R2stat nuisance model"],
    metrics_external["R2stat task model"],
    s=metrics_external["variance explained"] * 40,
    c=colors,
    edgecolor="black",
    linewidths=np.where(changed, 2, 0),
    alpha=0.7,
)
for comp in metrics_external.index[changed]:
    ax.annotate(comp.replace("ICA_", ""), (metrics_external.loc[comp, "R2stat nuisance model"], metrics_external.loc[comp, "R2stat task model"]))

ax.axvline(0.5, color="gray", linestyle="--")
ax.axhline(0.5, color="gray", linestyle="--")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("$R^2$ of the nuisance model", fontsize=14)
ax.set_ylabel("$R^2$ of the task model", fontsize=14)
ax.set_title("Green: accepted, red: rejected, outlined: changed by the external regressors", fontsize=11)
fig.tight_layout()
glue_figure("figure_extreg_r2_scatter", fig)
```

```{figure} figures/figure_extreg_r2_scatter.png
:name: figure_extreg_r2_scatter
:align: center

Each component's fit to the nuisance and task models.
The size of each point is the variance the component explains, and the dashed lines are the $R^2$ thresholds used by the tree.
```

Finally, we look at the time series of some components next to the regressors that fit them:
the components that the nuisance model changed, and, for comparison, the three accepted components that the task model fits best.
To match what `tedana` tests, we remove the Legendre polynomials from both the component and the regressors before fitting.

```{code-cell} ipython3
:tags: [hide-input]

from tedana.utils import create_legendre_polynomial_basis_set

mixing = pd.read_table(os.path.join(external_dir, f"{prefix}_desc-ICA_mixing.tsv"))
external_used = external.iloc[n_dummy:].reset_index(drop=True)
models = {
    "nuisance": external_used.filter(regex="^(mot|csf|physio)_").to_numpy(),
    "task": external_used.filter(regex="^task_").to_numpy(),
}
# The same detrending regressors tedana uses ("detrend": true)
legendre = create_legendre_polynomial_basis_set(volume_times.size, dtrank=None)


def detrend(x):
    """Remove the Legendre polynomials from each column of x."""
    return x - legendre @ np.linalg.lstsq(legendre, x, rcond=None)[0]


def detrended_fit(y, regressors):
    """Detrend y and the regressors as tedana's model does, and fit the regressors to y."""
    y_detrended = detrend(y)
    regressors_detrended = detrend(regressors)
    betas = np.linalg.lstsq(regressors_detrended, y_detrended, rcond=None)[0]
    return y_detrended, regressors_detrended @ betas


accepted = metrics_external[metrics_external["classification"] == "accepted"]
task_comps = accepted["R2stat task model"].sort_values(ascending=False).index[:3].tolist()
to_plot = [(comp, "nuisance") for comp in metrics_external.index[changed]] + [(comp, "task") for comp in task_comps]

fig, axes = plt.subplots(nrows=len(to_plot), figsize=(14, 2.4 * len(to_plot)), sharex=True)
for ax, (comp, model_name) in zip(np.atleast_1d(axes), to_plot):
    y = mixing[comp].to_numpy()
    y_detrended, y_fit = detrended_fit((y - y.mean()) / y.std(), models[model_name])
    color = "tab:red" if model_name == "nuisance" else "tab:green"
    ax.plot(volume_times, y_detrended, color="black", linewidth=1, label=f"{comp}, detrended")
    ax.plot(volume_times, y_fit, color=color, linewidth=1.5, label=f"Fit of the {model_name} model")
    r2 = metrics_external.loc[comp, f"R2stat {model_name} model"]
    if changed[comp]:
        status = f"{metrics_baseline.loc[comp, 'classification']} without and {metrics_external.loc[comp, 'classification']} with external regressors"
    else:
        status = f"{metrics_external.loc[comp, 'classification']} by both trees"

    ax.set_title(f"{comp}: {status}, $R^2$ = {r2:.2f}", fontsize=12, loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2, frameon=False)
    ax.set_yticks([])

ax.set_xlim(volume_times[0], volume_times[-1])
ax.set_xlabel("Time since the first volume (s)", fontsize=14)
fig.tight_layout()
glue_figure("figure_extreg_component_fits", fig)
```

```{figure} figures/figure_extreg_component_fits.png
:name: figure_extreg_component_fits
:align: center

Time series of the components whose classification changed (with the fit of the nuisance model, red)
and of the three accepted components that the task model fits best (with the fit of the task model, green),
after removing the Legendre polynomials.
```

## Final remarks

- Each external regressor adds a degree of freedom to the F-tests, and a model with many regressors fits noise by chance.
  The $R^2$ threshold in the tree protects against this, but keep the models small and the regressors meaningful.
- The partial-model tags only require p < 0.05 for a subset of regressors, on top of a full model that already fits well.
  With ten physiological regressors, that is an easy bar to clear, so read the tags as hints rather than diagnoses.
- Slow regressors, like the motion parameters themselves, can fit slow fluctuations that remain after detrending.
  Using only the derivatives of the motion parameters, or a higher detrending order (an integer for `"detrend"`), makes the nuisance model more specific.
- Regressors that are correlated with the task, like head motion during tapping, will make the nuisance model fit task-related components.
  The task model in this tree is there to rescue those components, as long as they are TE-dependent.
- The thresholds in the demonstration trees are not validated. Look at the components the external regressors change, as we did above,
  before you rely on a tree.

For more about decision trees and external regressors, see the
[`tedana` documentation on building decision trees](https://tedana.readthedocs.io/en/stable/building_decision_trees.html).

The denoised data from the second run are used in [Task Analysis of Multi-Echo Data](10_Task_Analysis.md).
