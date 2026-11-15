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
- IC = Independent Component
- ROI = Region of Interest
- RETROICOR = RETROspective Image CORrection
- RV = Respiration Variation
- HR = Heart Rate
```

`tedana` classifies ICA components mostly from how their signal depends on echo time (TE).
Sometimes we also know what a noise source or the signal of interest looks like over time:
a recording of the participant's breathing, the head motion estimated during preprocessing,
the average signal in the ventricles, or the timing of the task.
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

import matplotlib.pyplot as plt
import nibabel as nb
import nitransforms as nit
import numpy as np
import pandas as pd
from book_utils import glue_figure, load_diva
from nilearn import datasets, image, masking, plotting
from nilearn.glm.first_level import make_first_level_design_matrix
from scipy import signal

data_path = os.path.abspath("../data")
data = load_diva(data_path)
prefix = data["prefix"]
out_dir = os.path.join(data_path, "external-regressors")
os.makedirs(out_dir, exist_ok=True)

print("\n".join(os.path.basename(f) for f in data["echo_files"]))
print(f"Echo times (s): {data['echo_times']}")
```

Every regressor has to have one value per volume, so we start with the timing of the volumes.
fMRIPrep corrected the data for slice timing, and its sidecar file records the time (relative to the start of each volume)
that the corrected data now represent in `StartTime`.
That is the time at which we sample every regressor.

```{code-cell} ipython3
with open(data["optcom_json"], "r") as fo:
    bold_metadata = json.load(fo)

t_r = bold_metadata["RepetitionTime"]
slice_time_ref = bold_metadata["StartTime"]
n_vols = nb.load(data["optcom"]).shape[3]
frame_times = np.arange(n_vols) * t_r + slice_time_ref
print(f"{n_vols} volumes, TR = {t_r} s, regressors are sampled {slice_time_ref} s into each volume")
```

fMRIPrep also flags the first volumes, acquired before the signal reached a steady state, as "non-steady-state outliers".
We will drop them in `tedana` with `dummy_scans`.
`tedana` drops the same rows from the external regressors, so the regressors still cover the whole run.

```{code-cell} ipython3
confounds = pd.read_table(data["confounds"])
dummy_scans = confounds.filter(like="non_steady_state_outlier").any(axis=1).to_numpy()
# The outliers must be the first volumes for dummy_scans to remove them
n_dummy = int(np.argmin(dummy_scans)) if dummy_scans.any() else 0
assert dummy_scans[:n_dummy].all() and not dummy_scans[n_dummy:].any()
print(f"{n_dummy} non-steady-state volume(s)")
```

## 1. Physiological regressors

The physiological recording is a BIDS `_physio.tsv.gz` file, with its sampling frequency, columns and start time in the sidecar.

```{code-cell} ipython3
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
It is high while the scanner acquires volumes, so the first high sample is the start of the first volume,
and the length of the high period should match the duration of the run.

```{code-cell} ipython3
scan_on = physio["trigger"].to_numpy() > (physio["trigger"].max() / 2)
first_sample = np.argmax(scan_on)
start_time = -first_sample / samp_freq
physio_times = np.arange(physio.shape[0]) / samp_freq + start_time

print(f"StartTime in the sidecar: {physio_metadata['StartTime']} s")
print(f"StartTime from the trigger: {start_time:.3f} s")
print(f"Trigger is high for {scan_on.sum() / samp_freq:.2f} s; the run lasts {n_vols * t_r:.2f} s")
```

The recording starts about 9 seconds before the first volume, as the time column of the file also says.
From now on, `physio_times` holds the time of each sample relative to the start of the scan.

### Cardiac peaks

The ECG was recorded during the scan, so it carries a gradient artifact at the rate at which the scanner acquires slices.
With 48 slices acquired with a multiband factor of three, there are 16 shots per 1.5-second volume, and the artifact sits at 10.7 Hz and its harmonics.
A band-pass filter below that frequency leaves the R peaks, which `scipy` can find.

```{code-cell} ipython3
sos = signal.butter(4, [0.5, 8], btype="bandpass", fs=samp_freq, output="sos")
cardiac = signal.sosfiltfilt(sos, physio["cardiac"].to_numpy())

# R peaks are at least 0.6 s apart (heart rate < 100 bpm) and stand out from the rest of the trace.
card_peaks, _ = signal.find_peaks(cardiac, distance=int(0.6 * samp_freq), prominence=np.std(cardiac))
# Drop the occasional T wave that passes the first criteria.
card_peaks = card_peaks[cardiac[card_peaks] > 0.5 * np.median(cardiac[card_peaks])]
heart_rate = 60 / np.diff(physio_times[card_peaks])
print(f"{card_peaks.size} beats, heart rate between {heart_rate.min():.1f} and {heart_rate.max():.1f} bpm")
```

### Respiratory peaks

The respiratory belt is slow, so a low-pass filter is enough.

```{code-cell} ipython3
sos = signal.butter(4, 1, btype="lowpass", fs=samp_freq, output="sos")
respiration = signal.sosfiltfilt(sos, physio["respiration"].to_numpy())
resp_peaks, _ = signal.find_peaks(respiration, distance=int(2 * samp_freq), prominence=0.5 * np.std(respiration))
breathing_rate = 60 / np.median(np.diff(physio_times[resp_peaks]))
print(f"{resp_peaks.size} breaths, {breathing_rate:.1f} breaths per minute")
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(nrows=3, figsize=(14, 9))
window = (physio_times > 100) & (physio_times < 115)
axes[0].plot(physio_times[window], physio["cardiac"].to_numpy()[window], color="gray", linewidth=0.5, label="Raw ECG")
axes[0].plot(physio_times[window], cardiac[window], color="black", linewidth=1, label="Filtered ECG")
peaks_in_window = card_peaks[window[card_peaks]]
axes[0].plot(physio_times[peaks_in_window], cardiac[peaks_in_window], "o", color="tab:red", label="R peaks")
axes[0].set_ylabel("ECG (a.u.)", fontsize=14)
axes[0].legend(loc="upper right", ncols=3)

axes[1].plot(physio_times[card_peaks][1:], heart_rate, color="tab:red")
axes[1].set_ylabel("Heart rate (bpm)", fontsize=14)

window = (physio_times > 100) & (physio_times < 160)
axes[2].plot(physio_times[window], respiration[window], color="black")
peaks_in_window = resp_peaks[window[resp_peaks]]
axes[2].plot(physio_times[peaks_in_window], respiration[peaks_in_window], "o", color="tab:blue")
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

RETROICOR {cite:p}`glover2000image` models the effect of the cardiac and respiratory cycles on the signal as a Fourier series of their phase at the time each volume was acquired.
The cardiac phase goes from 0 to $2\pi$ between consecutive R peaks.
The respiratory phase also accounts for the depth of the breath:
it is given by the amplitude of the respiratory trace relative to its histogram, and its sign by whether the participant is breathing in or out.

```{code-cell} ipython3
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


card_phase = cardiac_phase(frame_times, physio_times[card_peaks])
resp_phase_all = respiratory_phase(respiration)
# The phase wraps from pi to -pi at the top of each breath, so we take the nearest sample instead of interpolating
nearest_sample = np.clip(np.round((frame_times - physio_times[0]) * samp_freq).astype(int), 0, physio_times.size - 1)
resp_phase = resp_phase_all[nearest_sample]

retroicor = pd.concat(
    [
        fourier_expansion(card_phase, 2, "physio_card"),
        fourier_expansion(resp_phase, 2, "physio_resp"),
    ],
    axis=1,
)
retroicor.head()
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
def respiration_response_function(t):
    """Birn et al. (2008)."""
    return 0.6 * t**2.1 * np.exp(-t / 1.6) - 0.0023 * t**3.54 * np.exp(-t / 4.25)


def cardiac_response_function(t):
    """Chang et al. (2009)."""
    return 0.6 * t**2.7 * np.exp(-t / 1.6) - 16 / np.sqrt(2 * np.pi * 9) * np.exp(-0.5 * (t - 12) ** 2 / 9)


grid_step = 0.1
grid = np.arange(physio_times[0], physio_times[-1], grid_step)
half_window = int(3 / grid_step)

# Respiration variation: the standard deviation of the respiratory trace in 6-second windows
resp_on_grid = np.interp(grid, physio_times, respiration)
rv = pd.Series(resp_on_grid).rolling(2 * half_window + 1, center=True, min_periods=1).std().to_numpy()

# Heart rate: the instantaneous heart rate, averaged in 6-second windows
beat_times = physio_times[card_peaks][1:]
hr = np.interp(grid, beat_times, heart_rate)
hr = pd.Series(hr).rolling(2 * half_window + 1, center=True, min_periods=1).mean().to_numpy()

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

## 2. ROI time series

`tedana` takes a table of time series, so the average signal in an ROI has to be computed before `tedana` runs
([ME-ICA/tedana#1175](https://github.com/ME-ICA/tedana/issues/1175)).
The obvious data to average are the optimally combined echoes, which fMRIPrep writes when it processes multi-echo data
(`_desc-preproc_bold.nii.gz` without an `echo` entity).
`tedana`'s own optimal combination would be slightly different, but this is close enough to describe a noise source.

The ROIs often come from an atlas in a standard space.
Here we take the lateral ventricles from the Harvard-Oxford subcortical atlas, as a CSF ROI,
and the primary motor, auditory and visual cortices from the Juelich histological atlas, to check the task responses.
The participant tapped with their left hand, so the motor ROI is the part of the right primary motor cortex
within 15 mm of the hand knob.

```{code-cell} ipython3
:tags: [hide-output]

harvard_oxford = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
juelich = datasets.fetch_atlas_juelich("maxprob-thr25-2mm")


def atlas_mask(atlas, labels):
    """Binary image of the atlas regions whose label is in ``labels``."""
    atlas_img = image.load_img(atlas.maps)
    values = [atlas.labels.index(label) for label in labels]
    return image.new_img_like(atlas_img, np.isin(atlas_img.get_fdata(), values).astype(np.uint8))


def sphere_mask(reference_img, center, radius):
    """Binary image of the voxels within ``radius`` mm of ``center`` (in world coordinates)."""
    ijk = np.indices(reference_img.shape[:3]).reshape(3, -1).T
    xyz = nb.affines.apply_affine(reference_img.affine, ijk)
    inside = np.linalg.norm(xyz - np.array(center), axis=1) <= radius
    return image.new_img_like(reference_img, inside.reshape(reference_img.shape[:3]).astype(np.uint8))


m1 = atlas_mask(juelich, ["GM Primary motor cortex BA4a", "GM Primary motor cortex BA4p"])
right_hand_knob = sphere_mask(m1, center=(38, -22, 54), radius=15)

atlas_rois = {
    "ventricles": atlas_mask(harvard_oxford, ["Left Lateral Ventricle", "Right Lateral Ventricle"]),
    "M1": image.math_img("m1 * sphere", m1=m1, sphere=right_hand_knob),
    "A1": atlas_mask(
        juelich,
        [
            "GM Primary auditory cortex TE1.0",
            "GM Primary auditory cortex TE1.1",
            "GM Primary auditory cortex TE1.2",
        ],
    ),
    "V1": atlas_mask(juelich, ["GM Visual cortex V1 BA17"]),
}
```

### Warping the ROIs into the space of the BOLD data

fMRIPrep writes the transforms we need:
`_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5` (a nonlinear transform between the template and the participant's T1w image)
and `_from-run_to-T1w_mode-image_desc-coreg_xfm.txt` (a rigid transform between the run's BOLD reference and the T1w image).
[NiTransforms](https://github.com/nipy/nitransforms) can chain them.
To resample an image, NiTransforms maps each voxel of the *reference* (here, the BOLD reference image)
to a location in the *moving* image (here, the atlas).
So the chain goes from the BOLD reference to the T1w image (the inverse of the coregistration), then from the T1w image to the template.

```{code-cell} ipython3
:tags: [hide-output]

run_to_t1w = nit.linear.load(data["run_to_t1w"], fmt="itk")
mni_to_t1w = nit.manip.load(data["mni_to_t1w"], fmt="itk")
run_to_mni = nit.manip.TransformChain([~run_to_t1w, *mni_to_t1w.transforms])

mask_img = nb.load(data["mask"])
roi_files = {}
for name, roi_img in atlas_rois.items():
    roi_run = nit.resampling.apply(run_to_mni, roi_img, reference=data["boldref"], order=0)
    roi_run = image.math_img("(roi > 0.5) & (mask > 0)", roi=roi_run, mask=mask_img)
    roi_files[name] = os.path.join(out_dir, f"{prefix}_space-run_desc-{name}_mask.nii.gz")
    roi_run.to_filename(roi_files[name])
```

```{note}
The Juelich and Harvard-Oxford atlases distributed with FSL (and fetched by nilearn) are in the `MNI152NLin6Asym` template,
while fMRIPrep normalized the data to `MNI152NLin2009cAsym`.
The two templates differ by a few millimeters, which matters little for these large ROIs.
For precise work, add the transform between the two templates from [TemplateFlow](https://www.templateflow.org) to the chain.
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 7))
for ax, (name, roi_file) in zip(axes.ravel(), roi_files.items()):
    plotting.plot_roi(
        roi_file,
        bg_img=data["boldref"],
        display_mode="ortho",
        cut_coords=plotting.find_xyz_cut_coords(roi_file),
        title=f"{name} ({int(nb.load(roi_file).get_fdata().sum())} voxels)",
        draw_cross=False,
        figure=fig,
        axes=ax,
    )

glue_figure("figure_extreg_rois", fig)
```

```{figure} figures/figure_extreg_rois.png
:name: figure_extreg_rois
:align: center

The atlas ROIs after warping them to the space of the BOLD data, on top of the BOLD reference image.
```

Now we average the optimally combined data in each ROI.

```{code-cell} ipython3
roi_timeseries = pd.DataFrame(
    {name: masking.apply_mask(data["optcom"], roi_file).mean(axis=1) for name, roi_file in roi_files.items()}
)
```

## 3. fMRIPrep confounds

fMRIPrep writes many candidate confounds.
Here we use the six head motion parameters and their temporal derivatives.
fMRIPrep's `csf` column is also an average signal in the CSF, but in an eroded CSF mask from the participant's own segmentation;
we use the ventricle ROI from above instead, to show how ROI time series are prepared.
The two time series need not be identical, because the masks are different
(the atlas ROI, for example, includes voxels at the edges of the ventricles that partly contain tissue).

```{code-cell} ipython3
motion_columns = [f"{kind}_{axis}{suffix}" for kind in ("trans", "rot") for axis in ("x", "y", "z") for suffix in ("", "_derivative1")]
# The first value of each derivative is undefined
motion = confounds[motion_columns].fillna(0)
print(f"Correlation between the ventricle ROI and fMRIPrep's csf column: {np.corrcoef(roi_timeseries['ventricles'], confounds['csf'])[0, 1]:.2f}")
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
print(events["trial_type"].value_counts())

task_design = make_first_level_design_matrix(
    frame_times,
    events,
    hrf_model="spm",
    drift_model=None,
)
task_regressors = task_design[["motor", "visual", "auditory"]].reset_index(drop=True)
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(nrows=3, figsize=(14, 8), sharex=True)
for ax, (roi, condition, color) in zip(axes, [("M1", "motor", "tab:orange"), ("A1", "auditory", "tab:green"), ("V1", "visual", "tab:purple")]):
    roi_ts = roi_timeseries[roi].to_numpy()
    # Percent signal change, without the slow drift
    roi_ts = signal.detrend(100 * (roi_ts / roi_ts.mean() - 1))
    ax.plot(frame_times, roi_ts, color="black", linewidth=1, label=f"{roi} ROI")
    ax2 = ax.twinx()
    ax2.plot(frame_times, task_regressors[condition], color=color, linewidth=2, label=f"{condition} regressor")
    ax2.set_yticks([])
    for _, event in events[events["trial_type"] == condition].iterrows():
        ax.axvspan(event["onset"], event["onset"] + event["duration"], color=color, alpha=0.15)

    ax.set_ylabel("Signal change (%)", fontsize=12)
    ax.set_title(f"{roi} and the {condition} blocks", fontsize=14)
    ax.set_xlim(0, frame_times[-1])

axes[-1].set_xlabel("Time (s)", fontsize=14)
fig.tight_layout()
glue_figure("figure_extreg_roi_task", fig)
```

```{figure} figures/figure_extreg_roi_task.png
:name: figure_extreg_roi_task
:align: center

Average signal in the motor, auditory and visual ROIs, after removing a linear trend (black), with the blocks of the matching condition (shaded) and the task regressor (colored).
```

## Putting the regressors together

`tedana` reads external regressors from a tab-separated file with one column per regressor and one row per volume, *including* the dummy scans.
The names of the columns are how the decision tree refers to the regressors, so we give each kind a prefix.

```{code-cell} ipython3
external = pd.concat(
    [
        physio_regressors,
        roi_timeseries[["ventricles"]].add_prefix("csf_"),
        motion.add_prefix("mot_"),
        task_regressors.add_prefix("task_"),
    ],
    axis=1,
)
assert external.shape[0] == n_vols
assert not external.isna().any().any()
external_file = os.path.join(out_dir, f"{prefix}_desc-external_regressors.tsv")
external.to_csv(external_file, sep="\t", index=False)
print("\n".join(external.columns))
```

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(14, 8))
zscored = (external - external.mean()) / external.std()
ax.imshow(zscored.T, aspect="auto", cmap="gray", vmin=-3, vmax=3, extent=[frame_times[0], frame_times[-1], external.shape[1] - 0.5, -0.5], interpolation="none")
ax.set_yticks(range(external.shape[1]))
ax.set_yticklabels(external.columns)
ax.set_xlabel("Time (s)", fontsize=14)
fig.tight_layout()
glue_figure("figure_extreg_regressors", fig)
```

```{figure} figures/figure_extreg_regressors.png
:name: figure_extreg_regressors
:align: center

All external regressors, each scaled to unit variance.
```

## A decision tree that uses external regressors

A decision tree declares how to fit the external regressors in its `external_regressor_config`,
and then uses the resulting metrics in its nodes.
`tedana` comes with demonstration trees that do this;
we start from `demo_external_regressors_motion_task_models`, which builds on the `minimal` tree.

```{code-cell} ipython3
import tedana
from tedana import workflows

demo_tree_file = os.path.join(
    os.path.dirname(tedana.__file__),
    "resources",
    "decision_trees",
    "demo_external_regressors_motion_task_models.json",
)
with open(demo_tree_file, "r") as fo:
    tree = json.load(fo)

print(json.dumps(tree["external_regressor_config"], indent=4))
```

The tree fits two models to each component's time series:

- a **nuisance** model, an F-test of all the nuisance regressors together, with **partial** models that test subsets of them;
- a **task** model, an F-test of the task regressors.

Both include Legendre polynomials (`"detrend": true`) to absorb slow drifts.
The F-tests compare each model with a model that has only the polynomials,
and $R^2$ is the fraction of the variance left after detrending that the regressors explain.
A partial model is tested by comparing the full model with the full model minus that subset of regressors.
The regressors are selected by name with regular expressions.
We adapt the configuration to our column names and add a partial model for the physiological regressors.

```{code-cell} ipython3
tree["tree_id"] = "external_regressors_physio_motion_csf_task"
tree["info"] = (
    "The demo_external_regressors_motion_task_models tree, "
    "with a partial model for physiological regressors."
)
tree["external_regressor_config"][0]["regressors"] = ["^mot_.*$", "^csf_.*$", "^physio_.*$"]
tree["external_regressor_config"][0]["partial_models"] = {
    "Motion": ["^mot_.*$"],
    "CSF": ["^csf_.*$"],
    "Physio": ["^physio_.*$"],
}
tree["external_regressor_config"][1]["regressors"] = ["^task_.*$"]
```

The nodes that use these models are near the end of the tree.
A component is rejected if the nuisance model fits it significantly (p < 0.05) and explains more than half of its variance ($R^2$ > 0.5).
Two nodes that change nothing then tag the rejected components whose fit is driven by motion or by CSF.
Later, a component that was going to be rejected is accepted instead if the task model fits it in the same way and it has enough TE-dependence (kappa above the kappa elbow).

```{code-cell} ipython3
for node in tree["nodes"]:
    if node["parameters"].get("left") in ("pval nuisance model", "pval task model"):
        print(json.dumps(node, indent=4))
```

We add a matching tagging node for the physiological partial model, and its metric and tag to the lists the tree declares.

```{code-cell} ipython3
csf_node_idx = next(
    i
    for i, node in enumerate(tree["nodes"])
    if node.get("kwargs", {}).get("tag_if_true") == "Fits CSF external regressors"
)
physio_node = json.loads(json.dumps(tree["nodes"][csf_node_idx]))
physio_node["kwargs"]["left3"] = "pval nuisance Physio partial model"
physio_node["kwargs"]["tag_if_true"] = "Fits physiological external regressors"
physio_node["_comment"] = "Not changing classifications, but tagging if fits to physiological regressors"
tree["nodes"].insert(csf_node_idx + 1, physio_node)

tree["necessary_metrics"].append("pval nuisance Physio partial model")
tree["classification_tags"].append("Fits physiological external regressors")

tree_file = os.path.join(out_dir, "external_regressors_tree.json")
with open(tree_file, "w") as fo:
    json.dump(tree, fo, indent=4)
```

## Running `tedana`

To see what the external regressors change, we need a tree that does everything else the same way.
The demonstration tree is not simply the `minimal` tree with extra nodes:
its early nodes only provisionally reject components, and it can rescue components with high kappa.
So we build the baseline from our tree itself, by removing the external regressor models and every node that uses them.

```{code-cell} ipython3
def uses_external_metrics(node):
    """Whether a node refers to a metric computed from the external regressors."""
    values = list(node.get("parameters", {}).values()) + list(node.get("kwargs", {}).values())
    return any(isinstance(value, str) and value.endswith(" model") for value in values)


baseline_tree = json.loads(json.dumps(tree))
baseline_tree["tree_id"] = "external_regressors_physio_motion_csf_task_without_external"
baseline_tree["info"] = "The external regressor tree, without the nodes that use external regressors."
baseline_tree.pop("external_regressor_config")
baseline_tree["nodes"] = [node for node in tree["nodes"] if not uses_external_metrics(node)]
baseline_tree["necessary_metrics"] = [m for m in tree["necessary_metrics"] if not m.endswith(" model")]
baseline_tree["classification_tags"] = [t for t in tree["classification_tags"] if not t.startswith(("External", "Fits"))]
print(f"Removed {len(tree['nodes']) - len(baseline_tree['nodes'])} of {len(tree['nodes'])} nodes")

baseline_tree_file = os.path.join(out_dir, "baseline_tree.json")
with open(baseline_tree_file, "w") as fo:
    json.dump(baseline_tree, fo, indent=4)
```

```{code-cell} ipython3
:tags: [hide-output]

baseline_dir = os.path.join(out_dir, "tedana-baseline")
workflows.tedana_workflow(
    data["echo_files"],
    data["echo_times"],
    out_dir=baseline_dir,
    mask=data["mask"],
    prefix=prefix,
    dummy_scans=n_dummy,
    tedpca="mdl",
    tree=baseline_tree_file,
    fixed_seed=42,
    overwrite=True,
)
```

Then we run it again with the full tree and the external regressors.
To compare the two classifications, we reuse the ICA decomposition of the first run with `mixing_file`, so both trees classify the same components.

```{code-cell} ipython3
:tags: [hide-output]

external_dir = os.path.join(out_dir, "tedana-external")
workflows.tedana_workflow(
    data["echo_files"],
    data["echo_times"],
    out_dir=external_dir,
    mask=data["mask"],
    prefix=prefix,
    dummy_scans=n_dummy,
    tree=tree_file,
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
tags = metrics_external["classification_tags"].str.split(",").explode()
tags.groupby(tags).size().sort_values(ascending=False)
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

Finally, we can look at the time series of the components that changed, next to the regressors that fit them.

```{code-cell} ipython3
:tags: [hide-input]

mixing = pd.read_table(os.path.join(external_dir, f"{prefix}_desc-ICA_mixing.tsv"))
external_used = external.iloc[n_dummy:].reset_index(drop=True)
volume_times = frame_times[n_dummy:]
models = {
    "nuisance": external_used.filter(regex="^(mot|csf|physio)_"),
    "task": external_used.filter(regex="^task_"),
}


from tedana.utils import create_legendre_polynomial_basis_set

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


changed_comps = metrics_external.index[changed].tolist()
if changed_comps:
    fig, axes = plt.subplots(nrows=len(changed_comps), figsize=(14, 2.5 * len(changed_comps)), squeeze=False)
    for ax, comp in zip(axes[:, 0], changed_comps):
        y = mixing[comp].to_numpy()
        y = (y - y.mean()) / y.std()
        model_name = "task" if metrics_external.loc[comp, "classification"] == "accepted" else "nuisance"
        y_detrended, y_fit = detrended_fit(y, models[model_name].to_numpy())
        ax.plot(volume_times, y_detrended, color="black", linewidth=1, label=f"{comp}, detrended")
        ax.plot(volume_times, y_fit, color="tab:red", linewidth=1.5, label=f"Fit of the {model_name} model")
        r2 = metrics_external.loc[comp, f"R2stat {model_name} model"]
        ax.set_title(
            f"{comp}: {metrics_baseline.loc[comp, 'classification']} without and "
            f"{metrics_external.loc[comp, 'classification']} with external regressors "
            f"($R^2$ = {r2:.2f})",
            fontsize=12,
        )
        ax.legend(loc="upper right")
        ax.set_xlim(volume_times[0], volume_times[-1])

    axes[-1, 0].set_xlabel("Time (s)", fontsize=14)
    fig.tight_layout()
    glue_figure("figure_extreg_changed_components", fig)
```

```{figure} figures/figure_extreg_changed_components.png
:name: figure_extreg_changed_components
:align: center

Time series of the components whose classification changed, after removing the Legendre polynomials (black),
with the fit of the model that changed it (red).
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
