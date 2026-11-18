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

# Model-Free Deconvolution with `pySPFM`

```{admonition} Abbreviations
- GLM = General Linear Model
- HRF = Hemodynamic Response Function
- ROI = Region of Interest
- SPFM = Sparse Paradigm Free Mapping
```

A task analysis needs to know when the task happened.
Sparse Paradigm Free Mapping (SPFM) does not:
it deconvolves the hemodynamic response from the BOLD signal and estimates a sparse **activity-inducing signal**,
a guess at *when* each region became active, without a model of the task.
The multi-echo version of SPFM {cite:p}`caballerogaudes2019deconvolution` uses the fact that BOLD changes
scale with the echo time (see [TE Dependence](TE_Dependence.md)):
it models the echoes together as changes in $R_{2}^{*}$, so fluctuations that do not follow the echo time fit the model poorly.
We use the implementation in [`pySPFM`](https://pyspfm.readthedocs.io).

Because SPFM does not need the task, it is meant for data without one, such as resting state or spontaneous events.
To see what it finds, though, it helps to use data where we *do* know what happened.
Here we use the event-related localizer run from the DIVA dataset:
sixty short trials of finger tapping, music and visual stimuli.
`pySPFM` never sees the events.
We only use them afterward, to check its estimates against the task and against a standard GLM.

```{code-cell} ipython3
import json
import logging
import os

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
from book_utils import (
    atlas_mask,
    glue_figure,
    load_diva,
    sphere_mask,
    warp_mni_roi_to_run,
    warp_mni_to_run,
)
from joblib import Parallel, delayed
from nilearn import datasets, image, plotting
from nilearn.glm import compute_contrast
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.maskers import NiftiLabelsMasker
from nilearn.signal import clean
from pySPFM import SparseDeconvolution
from threadpoolctl import threadpool_limits

# pySPFM logs a warning about computing clusters at every fit; we run everything locally
logging.getLogger("GENERAL").setLevel(logging.ERROR)

data_path = os.path.abspath("../data")
data = load_diva(data_path, task="localizerEstimation")
prefix = data["prefix"]
out_dir = os.path.join(data_path, "pySPFM")
os.makedirs(out_dir, exist_ok=True)
```

## The data

We need the repetition time and the echo times (which `pySPFM` takes in milliseconds).
We also need the time of each volume, so that we can line up the estimates with the events.
fMRIPrep corrected the data for slice timing, so each volume's reference time is shifted by the `StartTime` in the metadata.

```{code-cell} ipython3
with open(data["optcom_json"], "r") as fo:
    bold_metadata = json.load(fo)

t_r = bold_metadata["RepetitionTime"]
echo_times = [te * 1000 for te in data["echo_times"]]

# Drop any non-steady-state volumes that fMRIPrep flagged at the start of the run
confounds = pd.read_table(data["confounds"])
n_dummy = int(confounds.filter(like="non_steady_state_outlier").any(axis=1).sum())
n_vols = nb.load(data["echo_files"][0]).shape[3] - n_dummy
frame_times = (np.arange(n_vols) + n_dummy) * t_r + bold_metadata["StartTime"]

print(f"TR = {t_r} s | TEs = {[round(te, 1) for te in echo_times]} ms")
print(f"{n_vols} volumes after dropping {n_dummy} dummy scan(s)")
```

## The task

Each trial shows a visual stimulus or asks the participant to tap their fingers (with their left hand),
and some trials also play music.
As in [Task Analysis of Multi-Echo Data](10_Task_Analysis.md), a trial with two conditions
(tapping while music plays, for example) counts as a trial of each.
Unlike that page's block design, these trials are short and jittered,
which is the kind of signal SPFM is built for: brief events, separated in time.

```{code-cell} ipython3
events = pd.read_table(data["events"])
print(events["trial_type"].value_counts().to_string())
print(f"Trial durations: {events['duration'].min():.1f}-{events['duration'].max():.1f} s")
onset_gaps = np.diff(events["onset"])
print(f"Time between trial onsets: {onset_gaps.min():.1f}-{onset_gaps.max():.1f} s")

events = events.assign(trial_type=events["trial_type"].str.split("/")).explode("trial_type")
events = events[["onset", "duration", "trial_type"]].sort_values("onset").reset_index(drop=True)
conditions = ["motor", "auditory", "visual"]
condition_colors = {"motor": "tab:orange", "auditory": "tab:green", "visual": "tab:purple"}
```

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(14, 2.5))
for i_cond, condition in enumerate(conditions):
    trials = events[events["trial_type"] == condition]
    ax.broken_barh(
        list(zip(trials["onset"], trials["duration"])),
        (i_cond - 0.4, 0.8),
        color=condition_colors[condition],
    )

ax.set_yticks(range(len(conditions)), conditions)
ax.set_xlim(0, frame_times[-1] + t_r)
ax.invert_yaxis()
ax.set_xlabel("Time (s)")
glue_figure("figure_spfm_events", fig)
```

```{figure} figures/figure_spfm_events.png
:name: figure_spfm_events
:align: center

The trials of each condition.
`pySPFM` does not see these; we use them only to check its estimates.
```

## Regions

We deconvolve two sets of regional time series.
Averaging the voxels within a region boosts the signal-to-noise ratio and keeps the problem small.

The first set is the three *a priori* ROIs from [Task Analysis of Multi-Echo Data](10_Task_Analysis.md):
the right primary motor cortex (M1) near the hand knob, for tapping with the left hand,
the primary auditory cortex (A1) for the music, and the primary visual cortex (V1) for the visual stimuli.
The second is the 400-region Schaefer atlas, for a whole-brain view.
Both come from MNI space, so we warp them into the space of the BOLD data with fMRIPrep's transforms.

```{code-cell} ipython3
:tags: [hide-output]

juelich = datasets.fetch_atlas_juelich("maxprob-thr25-2mm")
m1 = atlas_mask(juelich, ["GM Primary motor cortex BA4a", "GM Primary motor cortex BA4p"])
rois_mni = {
    "M1": image.math_img("m1 * sphere", m1=m1, sphere=sphere_mask(m1, center=(38, -22, 54), radius=15)),
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
roi_names = list(rois_mni)
rois = {"motor": "M1", "auditory": "A1", "visual": "V1"}

boldref = nb.load(data["boldref"])
roi_labels = np.zeros(boldref.shape, dtype=np.int16)
for i_roi, roi_img in enumerate(rois_mni.values(), start=1):
    roi_labels[warp_mni_roi_to_run(roi_img, data).get_fdata() > 0] = i_roi

roi_labels_img = nb.Nifti1Image(roi_labels, boldref.affine)

schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, data_dir=os.path.join(data_path, "nilearn"))
in_brain = nb.load(data["mask"]).get_fdata() > 0
parcel_labels = np.round(warp_mni_to_run(nb.load(schaefer.maps), data).get_fdata()).astype(np.int16)
parcel_labels[~in_brain] = 0
parcel_labels_img = nb.Nifti1Image(parcel_labels, boldref.affine)
parcel_ids = np.unique(parcel_labels[parcel_labels > 0])
```

```{code-cell} ipython3
print({roi: int((roi_labels == i_roi).sum()) for i_roi, roi in enumerate(roi_names, start=1)}, "voxels")
print(f"{parcel_ids.size} Schaefer parcels inside the brain mask")
```

## Prepare the time series

For each echo, we average the voxels within each region.
We then remove slow drifts (a linear trend and fluctuations below 0.01 Hz) and divide by the region's mean,
so the data are fractional signal changes, which is what the multi-echo model expects:
raw signal intensities make the deconvolution collapse to zero.

We do not low-pass filter the data.
The trials start every 7 s or so, which puts much of the task's signal near 0.14 Hz,
above the 0.1 Hz cutoff that is common for resting-state data.
The deconvolution does its own smoothing, since the HRF is itself a low-pass filter.

`pySPFM` expects the echoes stacked in time: an array with one column per region,
holding the first echo's time series, then the second's, and so on.

```{code-cell} ipython3
def extract_echoes(labels_img):
    """Normalized time series of each region, with the echoes stacked in time."""
    masker = NiftiLabelsMasker(labels_img=labels_img).fit()
    stacked = []
    for echo_file in data["echo_files"]:
        signal = masker.transform(echo_file)[n_dummy:]
        cleaned = clean(signal, detrend=True, high_pass=0.01, t_r=t_r, standardize=False)
        stacked.append(cleaned / signal.mean(axis=0))

    return np.vstack(stacked), masker


X_roi, roi_masker = extract_echoes(roi_labels_img)
X_parcels, parcel_masker = extract_echoes(parcel_labels_img)
print(f"ROIs: {X_roi.shape} | parcels: {X_parcels.shape}  (echoes x volumes, regions)")
```

## Deconvolve the ROIs

`pySPFM` chooses the amount of regularization, and thus how sparse the estimate is, for each region separately.
We compare the two information criteria for the multi-echo model:
the Bayesian information criterion (BIC), with a strong penalty on the number of events,
and the Akaike information criterion (AIC), with a lighter one.

The multi-echo model estimates changes in $R_{2}^{*}$, so its HRF is scaled by $-TE$:
an increase in the BOLD signal is a *decrease* in $R_{2}^{*}$.
We flip the sign of the estimates so that positive values mean an increase in the BOLD signal.

```{code-cell} ipython3
roi_models = {}
for criterion in ["bic", "aic"]:
    roi_models[criterion] = SparseDeconvolution(tr=t_r, te=echo_times, criterion=criterion).fit(X_roi)

# The first column of the HRF matrix holds a complete response, for an event at the first volume
assert roi_models["bic"].hrf_matrix_[:, 0].sum() < 0, "Expected an HRF scaled by -TE"
roi_activity = {criterion: -model.coef_ for criterion, model in roi_models.items()}

for criterion, activity in roi_activity.items():
    n_active = (activity != 0).sum(axis=0)
    print(
        f"{criterion.upper()}: volumes with nonzero activity:",
        dict(zip(roi_names, n_active.tolist())),
        f"of {n_vols}",
    )
```

```{code-cell} ipython3
:tags: [hide-input]

fitted_bic = roi_models["bic"].get_fitted_signal()
i_echo = 2  # the third echo, the closest to the gray matter T2*
echo_slice = slice(i_echo * n_vols, (i_echo + 1) * n_vols)

fig, axes = plt.subplots(nrows=6, figsize=(14, 15), sharex=True, gridspec_kw={"hspace": 0.15})
for i_roi, (condition, roi) in enumerate(rois.items()):
    ax_bold, ax_activity = axes[2 * i_roi], axes[2 * i_roi + 1]
    trials = events[events["trial_type"] == condition]
    for ax in (ax_bold, ax_activity):
        for onset, duration in zip(trials["onset"], trials["duration"]):
            ax.axvspan(onset, onset + duration, color=condition_colors[condition], alpha=0.25, lw=0)

    ax_bold.plot(frame_times, X_roi[echo_slice, i_roi] * 100, color="0.5", lw=1, label=f"Echo {i_echo + 1}")
    ax_bold.plot(frame_times, fitted_bic[echo_slice, i_roi] * 100, color="black", lw=1.2, label="BIC fit")
    ax_bold.set_ylabel(f"{roi}\nsignal (%)")
    ax_bold.legend(loc="upper right", ncol=2)
    for criterion, color in [("aic", "tab:blue"), ("bic", "black")]:
        activity = roi_activity[criterion][:, i_roi]
        nonzero = activity != 0
        ax_activity.vlines(
            frame_times[nonzero],
            0,
            activity[nonzero],
            color=color,
            lw=1.5 if criterion == "bic" else 3,
            alpha=1 if criterion == "bic" else 0.4,
            label=criterion.upper(),
        )

    ax_activity.axhline(0, color="0.7", lw=0.6)
    ax_activity.set_ylabel(f"{roi}\nactivity (a.u.)")
    ax_activity.legend(loc="upper right", ncol=2)

axes[-1].set_xlabel("Time (s)")
axes[-1].set_xlim(0, frame_times[-1] + t_r)
glue_figure("figure_spfm_roi_timeseries", fig)
```

```{figure} figures/figure_spfm_roi_timeseries.png
:name: figure_spfm_roi_timeseries
:align: center

The deconvolution of each ROI.
In each pair of rows, the top row shows the data from the third echo and the BIC model's fit to it,
and the bottom row shows the activity-inducing signals that BIC and AIC estimated.
The shaded spans are the trials of the ROI's condition: motor trials for M1, music for A1, visual stimuli for V1.
```

Neither estimate is sparse.
Both criteria assume that the noise is independent from one sample to the next,
and in these data it is not: the echoes share much of their noise, and the noise in each echo is autocorrelated.
The criteria then mistake noise for signal and keep events throughout the run, many of them negative.
Individual trials do not stand out from the rest, so we cannot read this run's estimates trial by trial.

Not everything that SPFM finds is noise, though, nor is it all neuronal.
The largest events are the dips near 50 s and 295 s, which appear in all three ROIs at once.
A change that is shared across the brain like this is more likely to be physiological (a deep breath, for example)
than neuronal.
Such changes are still changes in $R_{2}^{*}$, so the multi-echo model cannot tell them apart from neuronal activity.

## Event-locked averages

Averaging over trials can still reveal what the estimates have in common with the task.
For each ROI and condition, we average the activity-inducing signal around the onsets of that condition's trials.
If the estimates carry information about the task, each ROI's average should peak around the onsets of its own condition,
and stay flat for the others.

```{code-cell} ipython3
lags = np.arange(-3, 7)


def event_locked_average(activity, condition):
    """Mean activity around the onsets of a condition's trials, for lags (in volumes) from the onset."""
    onsets = events.loc[events["trial_type"] == condition, "onset"].to_numpy()
    onset_vols = np.argmin(np.abs(frame_times[None, :] - onsets[:, None]), axis=1)
    onset_vols = onset_vols[(onset_vols + lags.min() >= 0) & (onset_vols + lags.max() < n_vols)]
    return np.stack([activity[i_vol + lags] for i_vol in onset_vols]).mean(axis=0)
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(ncols=3, figsize=(15, 4), sharey=True)
for i_roi, (ax, roi) in enumerate(zip(axes, roi_names)):
    for condition in conditions:
        average = event_locked_average(roi_activity["bic"], condition)[:, i_roi]
        ax.plot(lags * t_r, average * 1000, marker="o", color=condition_colors[condition], label=condition)

    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6, ls="--")
    ax.set_title(roi, fontsize=14)
    ax.set_xlabel("Time from trial onset (s)")

axes[0].set_ylabel("Mean BIC activity (a.u. x 1000)")
axes[-1].legend()
glue_figure("figure_spfm_event_locked", fig)
```

```{figure} figures/figure_spfm_event_locked.png
:name: figure_spfm_event_locked
:align: center

The BIC activity-inducing signal of each ROI, averaged around the trial onsets of each condition.
```

Each ROI's average is highest for its own condition:
M1 for the motor trials, A1 for the music and V1 for the visual stimuli.
So, although no single estimate can be trusted, the estimates as a whole carry information about the task.

The peaks fall at the onset or one volume (1.5 s) before it.
An estimate that comes before the trial means that the deconvolution placed the event too early.
This is what happens when the participant's responses rise sooner than the canonical HRF that `pySPFM` assumes,
and the rounding of each onset to the nearest volume adds up to half a volume of error.
Further from the onset, the averages pick up the neighboring trials, which start 3–10 s away.
The music overlaps the other conditions completely, since every trial with music is also a motor or visual trial.
Its average thus includes some of their activity, which is why A1's average also rises for the motor trials.

To compare with a model-based analysis, we fit a GLM to fMRIPrep's optimal combination of the echoes,
averaged within the same regions.
The GLM uses the same canonical HRF as `pySPFM` and knows the timing of every trial,
so it is the best case that SPFM's model-free estimates can be compared with.

```{code-cell} ipython3
design = make_first_level_design_matrix(
    frame_times,
    events,
    hrf_model="spm",
    drift_model="cosine",
    high_pass=0.01,
)


def glm_z_scores(masker):
    """z-score of each condition (versus baseline) in each region, from a GLM on the optimal combination."""
    signal = masker.transform(data["optcom"])[n_dummy:]
    labels, results = run_glm(signal / signal.mean(axis=0), design.to_numpy(), noise_model="ar1")
    return {
        condition: compute_contrast(labels, results, (design.columns == condition).astype(float)).z_score()
        for condition in conditions
    }


roi_z = glm_z_scores(roi_masker)
pd.DataFrame(roi_z, index=roi_names).round(1).rename_axis("GLM z-score")
```

The GLM finds the same pattern: a strong response in each ROI to its own condition, and little to the others.
It also separates the music from the other conditions, because it models all three together.
The difference is that the GLM needs to know when each trial happened, while SPFM found this pattern without that information.

## Whole-brain maps

Now we deconvolve the 400 Schaefer parcels.
`pySPFM` 2.0.2 fits one region at a time, whatever the value of its `n_jobs` parameter,
so we split the parcels into chunks and fit the chunks in parallel ourselves.
Each worker uses a single thread, so that the workers do not compete for the CPU.

```{code-cell} ipython3
def fit_chunk(X_chunk):
    # Each worker is a new process, so it needs the logging level set again
    logging.getLogger("GENERAL").setLevel(logging.ERROR)
    with threadpool_limits(1):
        model = SparseDeconvolution(tr=t_r, te=echo_times, criterion="bic").fit(X_chunk)

    return -model.coef_


n_jobs = os.cpu_count()
chunks = np.array_split(np.arange(X_parcels.shape[1]), 2 * n_jobs)
chunk_activity = Parallel(n_jobs=n_jobs)(delayed(fit_chunk)(X_parcels[:, chunk]) for chunk in chunks)
parcel_activity = np.hstack(chunk_activity)
np.save(os.path.join(out_dir, f"{prefix}_desc-parcelsBIC_activity.npy"), parcel_activity)
print(f"Mean number of volumes with nonzero activity per parcel: {(parcel_activity != 0).sum(axis=0).mean():.0f} of {n_vols}")
```

For each condition, we summarize each parcel by its event-locked average around the trial onsets,
over the volumes where the ROI averages peaked: the volume closest to the onset and the one before it.
We put these values next to the GLM's z-scores for the same parcels.
We show the 10% of parcels with the highest values in each map.

```{code-cell} ipython3
peak_lags = np.isin(lags, [-1, 0])
parcel_z = glm_z_scores(parcel_masker)
parcel_locked = {
    condition: event_locked_average(parcel_activity, condition)[peak_lags].mean(axis=0)
    for condition in conditions
}


def parcels_to_img(values):
    """Paint one value per parcel back onto the parcels, in the space of the BOLD data."""
    painted = np.zeros(parcel_labels.shape, dtype=np.float32)
    for value, label in zip(values, parcel_ids):
        painted[parcel_labels == label] = value

    return nb.Nifti1Image(painted, boldref.affine)


for condition in conditions:
    r = np.corrcoef(parcel_locked[condition], parcel_z[condition])[0, 1]
    print(f"{condition}: correlation across parcels between SPFM and GLM = {r:.2f}")
```

```{code-cell} ipython3
:tags: [hide-input]

def plot_condition_maps(condition):
    maps = {
        "SPFM (event-locked BIC activity)": parcel_locked[condition],
        "GLM (z-score)": parcel_z[condition],
    }
    cut_coords = plotting.find_xyz_cut_coords(parcels_to_img(parcel_z[condition]))
    fig = plt.figure(figsize=(14, 8), facecolor="black")
    for i_row, (title, values) in enumerate(maps.items()):
        # Show the top 10% of parcels, which have positive values in both maps
        threshold = np.percentile(values, 90)
        assert threshold > 0
        ax = fig.add_axes([0.0, 0.5 - 0.5 * i_row, 1.0, 0.45])
        plotting.plot_stat_map(
            parcels_to_img(np.maximum(values, 0)),
            bg_img=boldref,
            display_mode="ortho",
            cut_coords=cut_coords,
            threshold=float(threshold),
            vmax=float(values.max()),
            cmap="YlOrRd",
            symmetric_cbar=False,
            black_bg=True,
            dim=0,
            draw_cross=False,
            title=title,
            figure=fig,
            axes=ax,
        )

    fig.suptitle(condition, color="white", fontsize=18)
    return fig


for condition in conditions:
    glue_figure(f"figure_spfm_maps_{condition}", plot_condition_maps(condition))
```

```{figure} figures/figure_spfm_maps_motor.png
:name: figure_spfm_maps_motor
:align: center

The event-locked BIC activity (top) and the GLM z-scores (bottom) of each parcel for the motor trials,
showing the 10% of parcels with the highest values in each map.
```

```{figure} figures/figure_spfm_maps_auditory.png
:name: figure_spfm_maps_auditory
:align: center

The event-locked BIC activity (top) and the GLM z-scores (bottom) of each parcel for the music,
showing the 10% of parcels with the highest values in each map.
```

```{figure} figures/figure_spfm_maps_visual.png
:name: figure_spfm_maps_visual
:align: center

The event-locked BIC activity (top) and the GLM z-scores (bottom) of each parcel for the visual trials,
showing the 10% of parcels with the highest values in each map.
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(ncols=3, figsize=(15, 4.5))
for ax, condition in zip(axes, conditions):
    r = np.corrcoef(parcel_locked[condition], parcel_z[condition])[0, 1]
    ax.scatter(parcel_z[condition], parcel_locked[condition] * 1000, s=10, color=condition_colors[condition], alpha=0.6)
    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6)
    ax.set_title(f"{condition} (r = {r:.2f})", fontsize=14)
    ax.set_xlabel("GLM z-score")

axes[0].set_ylabel("Event-locked BIC activity (a.u. x 1000)")
glue_figure("figure_spfm_parcel_scatter", fig)
```

```{figure} figures/figure_spfm_parcel_scatter.png
:name: figure_spfm_parcel_scatter
:align: center

Each parcel's event-locked SPFM activity against its GLM z-score, for each condition.
```

Each parcel's estimate is sparser than the ROIs' estimates were.
The parcels are smaller than the ROIs, so their time series are noisier,
and BIC keeps fewer events when the noise is larger relative to the signal.

For the motor and visual trials, the SPFM and GLM maps agree.
Both pick out the right sensorimotor cortex around the hand knob for tapping with the left hand,
and the occipital cortex for the visual stimuli,
and across parcels the two are correlated at about 0.6.
The agreement for the music is weaker.
Both maps include the left superior temporal cortex,
but the SPFM map also includes parcels that the GLM does not,
partly because, as in A1, the music's event-locked average mixes in the motor and visual trials.

## Summary

- Multi-echo SPFM estimates when each region became active without a model of the task.
  Here it did so in parcels averaged across voxels, which keeps the problem small and improves the signal-to-noise ratio.
- In this run, the estimates are too dense to read trial by trial.
  BIC and AIC assume independent noise, which multi-echo fMRI data do not have, so they keep many noise events.
  Changes in $R_{2}^{*}$ that are not neuronal, such as those from breathing, pass the multi-echo model as well.
- On average, though, the estimates follow the task.
  Averaged around the trial onsets, they peak in the regions and for the conditions that a GLM finds,
  and SPFM did this without being told when the trials happened.
- That is the main use of SPFM: finding events in data whose timing we do not know,
  such as resting state or spontaneous activity during a task.
  The checks on this page show what to expect from it in data where we cannot check it.
