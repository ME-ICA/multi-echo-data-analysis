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

# Task Analysis of Multi-Echo Data

```{admonition} Abbreviations
- GLM = General Linear Model
- IC = Independent Component
- ROI = Region of Interest
```

```{warning}
This tutorial uses the outputs of [External Regressors in `tedana` Decision Trees](09_External_Regressors.md).
```

Once the echoes are combined and denoised, a task analysis of multi-echo data is the same as one of single-echo data:
a GLM fit to each voxel's time series.
The choices that are specific to multi-echo data are which data to fit, and how to account for the ICA components that `tedana` rejected.

Here we fit the same single-subject GLM to four versions of the localizer run from the DIVA dataset:

1. the second echo alone (TE = 28 ms), as a stand-in for a single-echo acquisition,
2. the optimally combined data,
3. the optimally combined data, with the rejected components, orthogonalized to the accepted ones, added to the GLM as nuisance regressors, and
4. the data denoised by `tedana`.

```{code-cell} ipython3
import json
import os

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
from book_utils import atlas_mask, glue_figure, load_diva, sphere_mask, warp_mni_roi_to_run
from matplotlib import cm, colors
from nilearn import datasets, image, masking, plotting
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

data_path = os.path.abspath("../data")
data = load_diva(data_path)
prefix = data["prefix"]
out_dir = os.path.join(data_path, "task-analysis")
os.makedirs(out_dir, exist_ok=True)
ted_dir = os.path.join(data_path, "external-regressors", "tedana-external")
```

## The data

`tedana` dropped the non-steady-state volume at the start of the run, so we drop it from the second echo as well.
All four versions of the data then have the same volumes.

```{code-cell} ipython3
with open(data["optcom_json"], "r") as fo:
    bold_metadata = json.load(fo)

t_r = bold_metadata["RepetitionTime"]
confounds = pd.read_table(data["confounds"])
n_dummy = int(confounds.filter(like="non_steady_state_outlier").any(axis=1).sum())

echo2_img = image.index_img(data["echo_files"][1], slice(n_dummy, None))
optcom_img = nb.load(os.path.join(ted_dir, f"{prefix}_desc-optcom_bold.nii.gz"))
denoised_img = nb.load(os.path.join(ted_dir, f"{prefix}_desc-denoised_bold.nii.gz"))
n_vols = optcom_img.shape[3]
assert echo2_img.shape[3] == n_vols == denoised_img.shape[3]
print(f"{n_vols} volumes after dropping {n_dummy} dummy scan(s)")
```

## *A priori* regions of interest

To compare the four analyses, we need regions where we know in advance that each condition should evoke a response.
We take them from the Juelich histological atlas:
the primary motor cortex (M1) for the motor condition, the primary auditory cortex (A1) for the auditory condition
and the primary visual cortex (V1) for the visual condition.
The participant tapped with their left hand, so the motor ROI is the part of the right primary motor cortex within 15 mm of the hand knob.
We warp them into the space of the BOLD data as in the previous page.

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

roi_files = {}
for roi, roi_img in rois_mni.items():
    roi_files[roi] = os.path.join(out_dir, f"{prefix}_space-run_desc-{roi}_mask.nii.gz")
    warp_mni_roi_to_run(roi_img, data).to_filename(roi_files[roi])
```

```{code-cell} ipython3
:tags: [hide-input]

roi_colors = {"M1": "tab:orange", "A1": "tab:green", "V1": "tab:purple"}
fig = plt.figure(figsize=(14, 13), facecolor="black")
for i_row, (roi, roi_file) in enumerate(roi_files.items()):
    bottom = 1 - (i_row + 1) / 3
    ax = fig.add_axes([0.05, bottom, 0.95, 1 / 3 - 0.01])
    fig.text(
        0.02,
        bottom + 1 / 6,
        f"{roi} ({int(nb.load(roi_file).get_fdata().sum())} voxels)",
        color="white",
        fontsize=16,
        rotation=90,
        ha="center",
        va="center",
    )
    plotting.plot_roi(
        roi_file,
        bg_img=data["boldref"],
        display_mode="ortho",
        cut_coords=plotting.find_xyz_cut_coords(roi_file),
        cmap=colors.ListedColormap([roi_colors[roi]]),
        colorbar=False,
        draw_cross=False,
        black_bg=True,
        dim=0,
        figure=fig,
        axes=ax,
    )

glue_figure("figure_task_rois", fig)
```

```{figure} figures/figure_task_rois.png
:name: figure_task_rois
:align: center

The *a priori* ROIs after warping them to the space of the BOLD data, on top of the BOLD reference image.
```

## The design matrix

As in the previous page, a block with two conditions (for example, tapping while music plays)
counts as a block of each, so we split the trial types and model the motor, auditory and visual conditions separately.
The volume times account for the dropped volume and for the slice timing reference that fMRIPrep used.
A cosine basis removes slow drifts.

We do not add the motion parameters.
The decision tree in the previous page already used them to reject the components that they fit,
and we want the four models to differ only in how they handle the ICA components.

```{code-cell} ipython3
events = pd.read_table(data["events"])
events = events.assign(trial_type=events["trial_type"].str.split("/")).explode("trial_type")
events = events[["onset", "duration", "trial_type"]].sort_values("onset").reset_index(drop=True)

frame_times = (np.arange(n_vols) + n_dummy) * t_r + bold_metadata["StartTime"]
design = make_first_level_design_matrix(
    frame_times,
    events,
    hrf_model="spm",
    drift_model="cosine",
    high_pass=1 / 128,
)
```

Every block is either a motor or a visual block, so contrasting either condition with the rest periods
also picks up everything the two have in common (attending to the screen, for example).
To isolate them, we contrast the motor and visual conditions with each other.
The auditory condition is present in only some of the blocks, so it can be contrasted with everything else.

```{code-cell} ipython3
contrasts = {
    "motor": "motor - visual",
    "auditory": "auditory",
    "visual": "visual - motor",
}
conditions = list(contrasts)
rois = {"motor": "M1", "auditory": "A1", "visual": "V1"}
```

For the third model, we add the time series of the components that `tedana` rejected.
The rejected components share some variance with the accepted ones,
and in a GLM that shared variance would be removed along with the noise.
`tedana`'s denoising does not remove it, because it fits all components together and subtracts only the unique contribution of the rejected ones
(a "non-aggressive" approach; see [Denoising Data with ICA](08_ICA_Based_Denoising.md)).
To do the same in the GLM, we first regress the accepted components out of the rejected ones.
`tedana` can write these orthogonalized components itself, with the `tedort` option.

```{code-cell} ipython3
metrics = pd.read_table(os.path.join(ted_dir, f"{prefix}_desc-tedana_metrics.tsv"))
mixing = pd.read_table(os.path.join(ted_dir, f"{prefix}_desc-ICA_mixing.tsv"))
rejected = metrics.loc[metrics["classification"] == "rejected", "Component"].tolist()
accepted = metrics.loc[metrics["classification"] == "accepted", "Component"].tolist()

accepted_ts = mixing[accepted].to_numpy()
rejected_ts = mixing[rejected].to_numpy()
betas = np.linalg.lstsq(accepted_ts, rejected_ts, rcond=None)[0]
rejected_orth = pd.DataFrame(rejected_ts - accepted_ts @ betas, columns=rejected, index=design.index)

design_rejected = pd.concat([design, rejected_orth], axis=1)
print(f"{len(rejected)} orthogonalized rejected components added to the design matrix")
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(
    ncols=2,
    figsize=(14, 6),
    gridspec_kw={"width_ratios": [design.shape[1], design_rejected.shape[1]]},
)
plotting.plot_design_matrix(design, axes=axes[0])
axes[0].set_title("Standard design", fontsize=14)
plotting.plot_design_matrix(design_rejected, axes=axes[1])
axes[1].set_title("With the orthogonalized rejected components", fontsize=14)
fig.tight_layout()
glue_figure("figure_task_design_matrices", fig)
```

```{figure} figures/figure_task_design_matrices.png
:name: figure_task_design_matrices
:align: center

The design matrices: the three conditions, the cosine drift terms and the constant,
and on the right the same with the time series of the orthogonalized rejected components.
```

```{note}
Even with orthogonalized components, the two approaches are not identical.
In the GLM, the task regressors and the rejected components are fit together,
so variance that a task regressor shares with a rejected component is attributed to neither.
`tedana` ignores the task when it denoises, so that shared variance is removed with the rejected components.
The GLM also counts the rejected components in its degrees of freedom, while a GLM fit to the denoised data does not know that some degrees of freedom were already spent.
```

## Fitting the models

The four models share everything except the data and, for the third, the design matrix.
We smooth with a 5 mm kernel, scale each voxel's signal to percent signal change, and use an AR(1) noise model.

```{code-cell} ipython3
:tags: [hide-output]

datasets_to_fit = {
    "Echo 2": (echo2_img, design),
    "Optcom": (optcom_img, design),
    "Optcom + rejected ICs": (optcom_img, design_rejected),
    "Denoised": (denoised_img, design),
}

z_maps = {}
for name, (img, design_matrix) in datasets_to_fit.items():
    model = FirstLevelModel(
        t_r=t_r,
        mask_img=data["mask"],
        smoothing_fwhm=5,
        signal_scaling=0,
        noise_model="ar1",
        minimize_memory=True,
    )
    model.fit(img, design_matrices=design_matrix)
    z_maps[name] = {
        condition: model.compute_contrast(contrast, output_type="z_score")
        for condition, contrast in contrasts.items()
    }
```

## Results

For each contrast, we show the four z-statistic maps at the same location:
the strongest response in the optimally combined data within the contrast's ROI.
That location is only used to choose the slices to display.

```{code-cell} ipython3
:tags: [hide-input]

threshold = 3.1
vmax = 10


def plot_contrast_maps(condition):
    """Plot the z-statistic maps of one contrast for the four versions of the data."""
    reference_map = image.math_img("z * roi", z=z_maps["Optcom"][condition], roi=roi_files[rois[condition]])
    cut_coords = plotting.find_xyz_cut_coords(reference_map)
    n_maps = len(z_maps)
    fig = plt.figure(figsize=(14, 4.2 * n_maps), facecolor="black")
    for i_map, (name, maps) in enumerate(z_maps.items()):
        bottom = 1 - (i_map + 1) / n_maps
        ax = fig.add_axes([0.05, bottom, 0.87, 1 / n_maps - 0.01])
        fig.text(
            0.02,
            bottom + 0.5 / n_maps,
            name,
            color="white",
            fontsize=16,
            rotation=90,
            ha="center",
            va="center",
        )
        plotting.plot_stat_map(
            maps[condition],
            bg_img=data["boldref"],
            threshold=threshold,
            vmax=vmax,
            cut_coords=cut_coords,
            display_mode="ortho",
            colorbar=False,
            draw_cross=False,
            black_bg=True,
            dim=0,
            annotate=i_map == 0,
            figure=fig,
            axes=ax,
        )

    cax = fig.add_axes([0.94, 0.3, 0.012, 0.4])
    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(vmin=-vmax, vmax=vmax), cmap=plotting.cm.cold_hot),
        cax=cax,
    )
    colorbar.set_label("z", color="white", fontsize=14)
    colorbar.ax.tick_params(colors="white")
    fig.suptitle(contrasts[condition], color="white", fontsize=18, y=1.02)
    return fig


for condition in conditions:
    glue_figure(f"figure_task_zmaps_{condition}", plot_contrast_maps(condition))
```

```{figure} figures/figure_task_zmaps_motor.png
:name: figure_task_zmaps_motor
:align: center

Z-statistic maps of the motor - visual contrast for each version of the data, thresholded at |z| > 3.1.
```

```{figure} figures/figure_task_zmaps_auditory.png
:name: figure_task_zmaps_auditory
:align: center

Z-statistic maps of the auditory contrast for each version of the data, thresholded at |z| > 3.1.
```

```{figure} figures/figure_task_zmaps_visual.png
:name: figure_task_zmaps_visual
:align: center

Z-statistic maps of the visual - motor contrast for each version of the data, thresholded at |z| > 3.1.
```

To summarize the maps, we average the z-statistics in the *a priori* ROI of each contrast.
These ROIs were defined from the anatomy, before looking at the data.
Averaging within the clusters that pass the threshold instead would be circular ("double dipping"):
the clusters are selected for their high z-statistics, so their average would be inflated,
and by different amounts for each version of the data.

```{code-cell} ipython3
:tags: [hide-input]

rows = []
for name, maps in z_maps.items():
    for condition in conditions:
        z_in_mask = masking.apply_mask(maps[condition], data["mask"])
        z_in_roi = masking.apply_mask(maps[condition], roi_files[rois[condition]])
        rows.append(
            {
                "data": name,
                "condition": condition,
                "voxels with z > 3.1": int((z_in_mask > threshold).sum()),
                "mean z in ROI": z_in_roi.mean(),
            }
        )

summary = pd.DataFrame(rows)
summary.pivot(index="data", columns="condition", values="mean z in ROI").loc[list(z_maps)].round(2)
```

The number of voxels above the threshold in the whole brain is shown as well:

```{code-cell} ipython3
:tags: [hide-input]

summary.pivot(index="data", columns="condition", values="voxels with z > 3.1").loc[list(z_maps)]
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(nrows=len(conditions), figsize=(12, 4.5 * len(conditions)), sharex=True)
data_colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green"]
for ax, condition in zip(axes, conditions):
    values = [masking.apply_mask(z_maps[name][condition], roi_files[rois[condition]]) for name in z_maps]
    parts = ax.violinplot(values, orientation="horizontal", showmedians=True)
    for body, color in zip(parts["bodies"], data_colors):
        body.set_facecolor(color)
        body.set_alpha(0.6)

    for line in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[line].set_color("black")

    ax.set_yticks(range(1, len(z_maps) + 1))
    ax.set_yticklabels(list(z_maps), fontsize=12)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(f"{contrasts[condition]} in {rois[condition]}", fontsize=14)

axes[-1].set_xlabel("z", fontsize=14)
fig.tight_layout()
glue_figure("figure_task_roi_z", fig)
```

```{figure} figures/figure_task_roi_z.png
:name: figure_task_roi_z
:align: center

Distribution of the z-statistics in the *a priori* ROI of each contrast, for each version of the data.
The line in each violin is the median.
```

## Final remarks

- Optimal combination weights each echo by its expected BOLD sensitivity in each voxel, so it helps most where a single echo time is far from the local $T_2^*$,
  such as near air-tissue boundaries. In large regions with typical $T_2^*$, like the ROIs here, the gain over a well-chosen single echo is smaller.
- A count of suprathreshold voxels is not a measure of quality on its own: removing noise that is correlated with the task can shrink a map as well as sharpen it.
- How much ICA denoising changes the results depends on how much structured noise the run contains and on how the components are classified.
  Check the classification (for example, with [rica](https://me-ica.github.io/rica/)) before you trust either.
- Statistics from denoised data overstate the degrees of freedom, because the GLM does not know that components were removed.
  Adding the orthogonalized rejected components to the design matrix avoids this, at the cost of attributing variance shared with the task to neither.
- In a group analysis, the maps of each participant would be warped to a template, for example with the transforms used to warp the ROIs.
