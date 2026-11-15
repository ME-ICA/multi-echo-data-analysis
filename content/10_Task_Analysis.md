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
- tSNR = temporal Signal-to-Noise Ratio
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
3. the optimally combined data, with the time series of the rejected components added to the GLM as nuisance regressors, and
4. the data denoised by `tedana`.

```{code-cell} ipython3
import json
import os

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
from book_utils import glue_figure, load_diva
from nilearn import image, masking, plotting
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

data_path = os.path.abspath("../data")
data = load_diva(data_path)
prefix = data["prefix"]
ext_dir = os.path.join(data_path, "external-regressors")
ted_dir = os.path.join(ext_dir, "tedana-external")
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

## The design matrix

As in the previous page, a block with two conditions (for example, tapping while music plays)
counts as a block of each, so we split the trial types and model the motor, auditory and visual conditions separately.
The volume times account for the dropped volumes and for the slice timing reference that fMRIPrep used.

```{code-cell} ipython3
events = pd.read_table(data["events"])
events = events.assign(trial_type=events["trial_type"].str.split("/")).explode("trial_type")
events = events[["onset", "duration", "trial_type"]].sort_values("onset").reset_index(drop=True)

frame_times = (np.arange(n_vols) + n_dummy) * t_r + bold_metadata["StartTime"]
motion_columns = [f"{kind}_{axis}" for kind in ("trans", "rot") for axis in ("x", "y", "z")]
motion = confounds[motion_columns].iloc[n_dummy:].to_numpy()

design = make_first_level_design_matrix(
    frame_times,
    events,
    hrf_model="spm",
    drift_model="cosine",
    high_pass=1 / 128,
    add_regs=motion,
    add_reg_names=motion_columns,
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
```

For the third model, we add the time series of the components that `tedana` rejected.

```{code-cell} ipython3
metrics = pd.read_table(os.path.join(ted_dir, f"{prefix}_desc-tedana_metrics.tsv"))
mixing = pd.read_table(os.path.join(ted_dir, f"{prefix}_desc-ICA_mixing.tsv"))
rejected = metrics.loc[metrics["classification"] == "rejected", "Component"].tolist()

design_rejected = pd.concat(
    [design, mixing[rejected].set_index(design.index)],
    axis=1,
)
print(f"{len(rejected)} rejected components added to the design matrix")
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(ncols=2, figsize=(14, 6), gridspec_kw={"width_ratios": [design.shape[1], design_rejected.shape[1]]})
plotting.plot_design_matrix(design, axes=axes[0])
axes[0].set_title("Standard design", fontsize=14)
plotting.plot_design_matrix(design_rejected, axes=axes[1])
axes[1].set_title("With the rejected components", fontsize=14)
fig.tight_layout()
glue_figure("figure_task_design_matrices", fig)
```

```{figure} figures/figure_task_design_matrices.png
:name: figure_task_design_matrices
:align: center

The design matrices: the three conditions, the six motion parameters, the cosine drift terms and the constant,
and on the right the same with the time series of the rejected components.
```

```{note}
The denoised data and the rejected components handle the overlap between the task and the noise differently.
To make the denoised data, `tedana` fits all components together and subtracts the fitted rejected ones
(a "non-aggressive" approach; see [Denoising Data with ICA](08_ICA_Based_Denoising.md)).
Because the coefficients are estimated jointly, each rejected component is scaled by its fit to the data over and above the accepted components,
but the task itself plays no part in that fit.
In the GLM, the task regressors and the rejected components are fit together,
so variance that a task regressor shares with a rejected component is attributed to neither.
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

We look at each contrast at the location of its strongest response in the optimally combined data,
inside the matching ROI from the previous page.

```{code-cell} ipython3
rois = {
    "motor": "M1",
    "auditory": "A1",
    "visual": "V1",
}
roi_files = {roi: os.path.join(ext_dir, f"{prefix}_space-run_desc-{roi}_mask.nii.gz") for roi in rois.values()}
```

```{code-cell} ipython3
:tags: [hide-input]

threshold = 3.1
fig, axes = plt.subplots(nrows=len(conditions), ncols=len(z_maps), figsize=(5 * len(z_maps), 4 * len(conditions)))
for i_row, condition in enumerate(conditions):
    roi_file = roi_files[rois[condition]]
    reference_map = image.math_img("z * roi", z=z_maps["Optcom"][condition], roi=roi_file)
    cut_coords = plotting.find_xyz_cut_coords(reference_map)
    for j_col, (name, maps) in enumerate(z_maps.items()):
        plotting.plot_stat_map(
            maps[condition],
            bg_img=data["boldref"],
            threshold=threshold,
            vmax=12,
            cut_coords=cut_coords,
            display_mode="ortho",
            colorbar=j_col == len(z_maps) - 1,
            title=f"{name}: {contrasts[condition]}" if i_row == 0 else contrasts[condition],
            draw_cross=False,
            figure=fig,
            axes=axes[i_row, j_col],
        )

glue_figure("figure_task_zmaps", fig)
```

```{figure} figures/figure_task_zmaps.png
:name: figure_task_zmaps
:align: center

Z-statistic maps of each contrast (rows) for each version of the data (columns), thresholded at z = 3.1.
```

To summarize the maps, we average the z-statistics in the ROI that matches each contrast,
and count the voxels above the threshold in the whole brain.

```{code-cell} ipython3
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

```{code-cell} ipython3
summary.pivot(index="data", columns="condition", values="voxels with z > 3.1").loc[list(z_maps)]
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(ncols=len(conditions), figsize=(15, 5), sharey=True)
colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green"]
for ax, condition in zip(axes, conditions):
    values = []
    for name in z_maps:
        values.append(masking.apply_mask(z_maps[name][condition], roi_files[rois[condition]]))

    parts = ax.violinplot(values, showmedians=True)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.6)

    ax.set_xticks(range(1, len(z_maps) + 1))
    ax.set_xticklabels([name.replace(" + ", "\n+ ") for name in z_maps], rotation=45, ha="right")
    ax.set_title(f"{contrasts[condition]} in {rois[condition]}", fontsize=14)
    ax.axhline(0, color="black", linewidth=0.5)

axes[0].set_ylabel("z", fontsize=14)
fig.tight_layout()
glue_figure("figure_task_roi_z", fig)
```

```{figure} figures/figure_task_roi_z.png
:name: figure_task_roi_z
:align: center

Distribution of the z-statistics in the ROI of each condition, for each version of the data.
```

## Final remarks

- Optimal combination weights each echo by its expected BOLD sensitivity in each voxel, so it helps most where a single echo time is far from the local $T_2^*$,
  such as near air-tissue boundaries. In large regions with typical $T_2^*$, like the ROIs here, the gain over a well-chosen single echo is smaller.
- A count of suprathreshold voxels is not a measure of quality on its own: removing noise that is correlated with the task can shrink a map as well as sharpen it.
- How much ICA denoising changes the results depends on how much structured noise the run contains and on how the components are classified.
  Check the classification (for example, with [rica](https://me-ica.github.io/rica/)) before you trust either.
- Statistics from denoised data overstate the degrees of freedom, because the GLM does not know that components were removed.
  Adding the rejected components to the design matrix avoids this, at the cost of attributing shared variance to neither the task nor the noise.
- In a group analysis, the maps of each participant would be warped to a template, for example with the transforms used in the previous page.
