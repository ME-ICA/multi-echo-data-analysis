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

# Optimal combination with `t2smap`

Use `t2smap` {cite:p}`DuPre2021` to combine data.

```{code-cell} ipython3
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from book_utils import glue_figure, load_pafin
from nilearn import image, masking, plotting
from tedana import workflows

data_path = os.path.abspath('../data')
```

```{code-cell} ipython3
data = load_pafin(data_path)
out_dir = os.path.join(data_path, "t2smap")
```

```{code-cell} ipython3
:tags: [hide-output]
workflows.t2smap_workflow(
    data['echo_files'],
    data['echo_times'],
    out_dir=out_dir,
    mask=data['mask'],
    prefix="sub-24053_ses-1_task-rat_dir-PA_run-01",
    fittype="curvefit",
    overwrite=True,
)
```

```{code-cell} ipython3
out_files = sorted(glob(os.path.join(out_dir, "*")))
out_files = [os.path.basename(f) for f in out_files]
print("\n".join(out_files))
```

```{code-cell} ipython3
:tags: [hide-output]
fig, ax = plt.subplots(figsize=(16, 8))
in_file = os.path.join(out_dir, "sub-24053_ses-1_task-rat_dir-PA_run-01_T2starmap.nii.gz")
arr = masking.apply_mask(in_file, data['mask'])
thresh = np.nanpercentile(arr, 98)
plotting.plot_stat_map(
    in_file,
    draw_cross=False,
    bg_img=None,
    figure=fig,
    axes=ax,
    cmap="Reds",
    vmin=0,
    vmax=thresh,
)
glue_figure("figure_t2starmap", fig, display=False)
```

```{figure} figures/figure_t2starmap.png
:name: "figure_t2starmap"
:align: center

T2* map estimated from multi-echo data using tedana's {py:func}`~tedana.workflows.t2smap_workflow`.
```

```{code-cell} ipython3
:tags: [hide-output]
fig, ax = plt.subplots(figsize=(16, 8))
in_file = os.path.join(out_dir, "sub-24053_ses-1_task-rat_dir-PA_run-01_S0map.nii.gz")
arr = masking.apply_mask(in_file, data['mask'])
thresh = np.nanpercentile(arr, 98)
plotting.plot_stat_map(
    in_file,
    draw_cross=False,
    bg_img=None,
    figure=fig,
    axes=ax,
    cmap="Reds",
    vmin=0,
    vmax=thresh,
)
glue_figure("figure_s0map", fig, display=False)
```

```{figure} figures/figure_s0map.png
:name: "figure_s0map"
:align: center

S0 map estimated from multi-echo data using tedana's {py:func}`~tedana.workflows.t2smap_workflow`.
```

```{code-cell} ipython3
:tags: [hide-output]
fig, axes = plt.subplots(figsize=(16, 15), nrows=5)
plotting.plot_epi(
    image.mean_img(data['echo_files'][0]),
    draw_cross=False,
    bg_img=None,
    cut_coords=[-10, 0, 10, 20, 30, 40, 50, 60, 70],
    display_mode="z",
    figure=fig,
    axes=axes[0],
)
plotting.plot_epi(
    image.mean_img(data['echo_files'][1]),
    draw_cross=False,
    bg_img=None,
    cut_coords=[-10, 0, 10, 20, 30, 40, 50, 60, 70],
    display_mode="z",
    figure=fig,
    axes=axes[1],
)
plotting.plot_epi(
    image.mean_img(data['echo_files'][2]),
    draw_cross=False,
    bg_img=None,
    cut_coords=[-10, 0, 10, 20, 30, 40, 50, 60, 70],
    display_mode="z",
    figure=fig,
    axes=axes[2],
)
plotting.plot_epi(
    image.mean_img(data['echo_files'][3]),
    draw_cross=False,
    bg_img=None,
    cut_coords=[-10, 0, 10, 20, 30, 40, 50, 60, 70],
    display_mode="z",
    figure=fig,
    axes=axes[3],
)
plotting.plot_epi(
    image.mean_img(os.path.join(out_dir, "sub-24053_ses-1_task-rat_dir-PA_run-01_desc-optcom_bold.nii.gz")),
    draw_cross=False,
    bg_img=None,
    cut_coords=[-10, 0, 10, 20, 30, 40, 50, 60, 70],
    display_mode="z",
    figure=fig,
    axes=axes[4],
)
glue_figure("figure_t2smap_epi_plots", fig, display=False)
```

```{figure} figures/figure_t2smap_epi_plots.png
:name: "figure_t2smap_epi_plots"
:align: center

Mean map of each of the echoes in the original data, along with the mean map of the optimally combined data.
```

```{code-cell} ipython3
:tags: [hide-output]
te30_tsnr = image.math_img(
    "(np.nanmean(img, axis=3) / np.nanstd(img, axis=3)) * mask",
    img=data['echo_files'][1],
    mask=data['mask'],
)
oc_tsnr = image.math_img(
    "(np.nanmean(img, axis=3) / np.nanstd(img, axis=3)) * mask",
    img=os.path.join(
        out_dir, "sub-24053_ses-1_task-rat_dir-PA_run-01_desc-optcom_bold.nii.gz"
    ),
    mask=data['mask'],
)
arr = masking.apply_mask(te30_tsnr, data['mask'])
thresh = np.nanpercentile(arr, 98)
arr = masking.apply_mask(oc_tsnr, data['mask'])
thresh = np.maximum(thresh, np.nanpercentile(arr, 98))

fig, axes = plt.subplots(figsize=(10, 8), nrows=2)
plotting.plot_stat_map(
    te30_tsnr,
    draw_cross=False,
    bg_img=None,
    threshold=0.1,
    cut_coords=[0, 10, 10],
    symmetric_cbar=False,
    figure=fig,
    axes=axes[0],
    cmap="Reds",
    vmin=0,
    vmax=thresh,
)
axes[0].set_title(f"Echo 2 (TE = {data['echo_times'][1] * 1000:.1f} ms) TSNR", fontsize=16)
plotting.plot_stat_map(
    oc_tsnr,
    draw_cross=False,
    bg_img=None,
    threshold=0.1,
    cut_coords=[0, 10, 10],
    symmetric_cbar=False,
    figure=fig,
    axes=axes[1],
    cmap="Reds",
    vmin=0,
    vmax=thresh,
)
axes[1].set_title("Optimal Combination TSNR", fontsize=16)
glue_figure("figure_t2smap_t2snr", fig, display=False)
```

```{figure} figures/figure_t2smap_t2snr.png
:name: "figure_t2smap_t2snr"
:align: center

TSNR map of the second echo in the original data, along with the TSNR map of the optimally combined data.
```

```{code-cell} ipython3
:tags: [hide-output]
fig, ax = plt.subplots(figsize=(16, 8))
plotting.plot_carpet(
    data['echo_files'][1],
    figure=fig,
    axes=ax,
)
glue_figure("figure_echo2_carpet", fig, display=False)
```

```{figure} figures/figure_echo2_carpet.png
:name: "figure_echo2_carpet"
:align: center

Carpet plot of the second echo's data.
```

```{code-cell} ipython3
:tags: [hide-output]
fig, ax = plt.subplots(figsize=(16, 8))
plotting.plot_carpet(
    os.path.join(out_dir, "sub-24053_ses-1_task-rat_dir-PA_run-01_desc-optcom_bold.nii.gz"),
    axes=ax,
)
glue_figure("figure_optcom_carpet", fig, display=False)
```

```{figure} figures/figure_optcom_carpet.png
:name: "figure_optcom_carpet"
:align: center

Carpet plot of the optimally combined data.
```

## Recovering signal in dropout regions

One advantage of multi-echo fMRI is that it can recover signal in regions that suffer from
signal dropout in single-echo fMRI {cite:p}`posse1999enhancement-176,poser2006bold-bda`.
Near air-tissue boundaries, such as the sinuses and the ear canals,
differences in magnetic susceptibility distort the local magnetic field.
Spins within a voxel then dephase faster, so $T_{2}^*$ is much shorter there than in the rest of the brain.
The most affected regions are orbitofrontal cortex, ventromedial prefrontal cortex,
and the inferior and anterior temporal lobes.

The signal at each echo time decays as $S(TE) = S_{0} e^{-TE / T_{2}^*}$.
Single-echo fMRI uses one echo time, usually chosen to be close to the average $T_{2}^*$ of gray matter
(roughly 30-40 ms at 3T), because that is where the BOLD contrast is largest for most of the brain.
In voxels with a short $T_{2}^*$, most of the signal has already decayed by that echo time,
which shows up as dark, "missing" patches in ventral frontal and temporal cortex.

In this dataset, the second echo (TE = 38.9 ms) is the closest to a typical single-echo acquisition.
The first echo still has signal in the ventral frontal and temporal lobes,
but that signal fades with each later echo.

```{code-cell} ipython3
:tags: [hide-output]
sagittal_cuts = [-30, -5, 30]
echo_idx = [0, 1, 4]
fig, axes = plt.subplots(figsize=(12, 9), nrows=len(echo_idx))
for ax, i_echo in zip(axes, echo_idx):
    plotting.plot_epi(
        image.mean_img(data['echo_files'][i_echo], copy_header=True),
        draw_cross=False,
        bg_img=None,
        cut_coords=sagittal_cuts,
        display_mode="x",
        cmap="gray",
        figure=fig,
        axes=ax,
        title=f"Echo {i_echo + 1} (TE = {data['echo_times'][i_echo] * 1000:.1f} ms)",
    )

glue_figure("figure_dropout_echoes", fig, display=False)
```

```{figure} figures/figure_dropout_echoes.png
:name: "figure_dropout_echoes"
:align: center

Sagittal slices of the mean image of the first, second, and last echoes.
Signal in orbitofrontal cortex and the inferior temporal lobes is visible in the first echo,
is weaker in the second echo, and is gone by the last echo.
Each row has its own intensity scale.
```

### How optimal combination adapts to $T_{2}^*$

The optimal combination is a weighted average of the echoes,
where the weight of echo $n$ in a voxel is

$$
w_{n} = \frac{TE_{n} e^{-TE_{n} / T_{2}^*}}{\sum_{i} TE_{i} e^{-TE_{i} / T_{2}^*}}
$$

The term $TE_{n} e^{-TE_{n} / T_{2}^*}$ is proportional to the BOLD sensitivity of echo $n$,
and it is largest when $TE_{n} = T_{2}^*$.
Because the weights depend on each voxel's own $T_{2}^*$,
the optimal combination favors the early echoes in voxels with a short $T_{2}^*$
and the later echoes in voxels with a long $T_{2}^*$.

```{code-cell} ipython3
:tags: [hide-output]
echo_times_ms = np.array(data['echo_times']) * 1000
t2s_range = np.linspace(5, 100, 500)
sensitivity = echo_times_ms[None, :] * np.exp(-echo_times_ms[None, :] / t2s_range[:, None])
weights = sensitivity / sensitivity.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 5))
for i_echo, echo_time in enumerate(echo_times_ms):
    ax.plot(t2s_range, weights[:, i_echo], linewidth=2, label=f"Echo {i_echo + 1} ({echo_time:.1f} ms)")

ax.set_xlabel("$T_{2}^*$ (ms)", fontsize=14)
ax.set_ylabel("Weight", fontsize=14)
ax.set_xlim(t2s_range[0], t2s_range[-1])
ax.set_ylim(0, 1)
ax.legend(fontsize=12)
fig.tight_layout()
glue_figure("figure_optcom_weights", fig, display=False)
```

```{figure} figures/figure_optcom_weights.png
:name: "figure_optcom_weights"
:align: center

Weight of each echo in the optimal combination, as a function of a voxel's $T_{2}^*$,
for this dataset's echo times.
In voxels with a short $T_{2}^*$, most of the weight goes to the first echo.
```

One way to see this in the brain is the *effective echo time* of each voxel,
which is the average of the echo times, weighted by the optimal combination weights.
`t2smap` only combines the echoes with good signal in each voxel, according to the adaptive mask
(and the first two echoes when only one echo has good signal), so we do the same here.

```{code-cell} ipython3
:tags: [hide-output]
t2s_file = os.path.join(out_dir, "sub-24053_ses-1_task-rat_dir-PA_run-01_T2starmap.nii.gz")
adaptive_mask_file = os.path.join(
    out_dir,
    "sub-24053_ses-1_task-rat_dir-PA_run-01_desc-adaptiveGoodSignal_mask.nii.gz",
)

# The T2* map is in seconds
t2s = masking.apply_mask(t2s_file, data['mask']) * 1000
adaptive_mask = masking.apply_mask(adaptive_mask_file, data['mask']).astype(int)

# t2smap combines the first two echoes in voxels with one or two good echoes,
# and the first n echoes in voxels with n > 2 good echoes.
# Voxels without any good echoes are not combined.
n_combined = np.where(adaptive_mask > 0, np.maximum(adaptive_mask, 2), 0)
combined = np.arange(len(echo_times_ms))[None, :] < n_combined[:, None]
good_voxels = (n_combined > 0) & (t2s > 0)

voxel_sensitivity = (
    echo_times_ms[None, :]
    * np.exp(-echo_times_ms[None, :] / t2s[good_voxels, None])
    * combined[good_voxels]
)
voxel_weights = voxel_sensitivity / voxel_sensitivity.sum(axis=1, keepdims=True)
effective_te = np.zeros(t2s.shape)
effective_te[good_voxels] = voxel_weights @ echo_times_ms
effective_te_img = masking.unmask(effective_te, data['mask'])

fig, ax = plt.subplots(figsize=(12, 4))
plotting.plot_stat_map(
    effective_te_img,
    bg_img=image.mean_img(data['echo_files'][0], copy_header=True),
    draw_cross=False,
    cut_coords=sagittal_cuts,
    display_mode="x",
    threshold=0.1,
    cmap="viridis",
    symmetric_cbar=False,
    vmin=echo_times_ms[0],
    vmax=echo_times_ms[-1],
    figure=fig,
    axes=ax,
)
glue_figure("figure_effective_te", fig, display=False)
```

```{figure} figures/figure_effective_te.png
:name: "figure_effective_te"
:align: center

Effective echo time (in ms) of the optimally combined data, on the mean image of the first echo.
In most of the brain, the optimal combination is dominated by the middle echoes,
but in orbitofrontal cortex and the inferior temporal lobes it is dominated by the first echo,
which still has signal there.
```

### Temporal signal-to-noise ratio in dropout regions

Since the optimal combination draws on the early echoes where $T_{2}^*$ is short,
the improvement in TSNR over the single-echo data should be largest in those voxels.

```{code-cell} ipython3
:tags: [hide-output]
fig, axes = plt.subplots(figsize=(12, 7), nrows=2)
for ax, tsnr_img, title in zip(
    axes,
    [te30_tsnr, oc_tsnr],
    [f"Echo 2 (TE = {echo_times_ms[1]:.1f} ms) TSNR", "Optimal Combination TSNR"],
):
    plotting.plot_stat_map(
        tsnr_img,
        draw_cross=False,
        bg_img=None,
        threshold=0.1,
        cut_coords=sagittal_cuts,
        display_mode="x",
        symmetric_cbar=False,
        cmap="Reds",
        vmin=0,
        vmax=thresh,
        figure=fig,
        axes=ax,
        title=title,
    )

glue_figure("figure_dropout_tsnr", fig, display=False)
```

```{figure} figures/figure_dropout_tsnr.png
:name: "figure_dropout_tsnr"
:align: center

Sagittal slices of the TSNR of the second echo and of the optimally combined data,
on the same color scale.
The optimally combined data have higher TSNR in most of the brain,
including the ventral frontal and temporal regions where the second echo has little signal.
```

To summarize this across all voxels with good signal, we can group voxels by their $T_{2}^*$
and compare the TSNR of the optimally combined data with that of the second echo.

```{code-cell} ipython3
:tags: [hide-output]
echo2_tsnr_arr = masking.apply_mask(te30_tsnr, data['mask'])[good_voxels]
oc_tsnr_arr = masking.apply_mask(oc_tsnr, data['mask'])[good_voxels]
valid = np.isfinite(echo2_tsnr_arr) & np.isfinite(oc_tsnr_arr) & (echo2_tsnr_arr > 0)
tsnr_ratio = oc_tsnr_arr[valid] / echo2_tsnr_arr[valid]
voxel_t2s = t2s[good_voxels][valid]

# 5-ms bins from 10 to 80 ms, plus open-ended bins below 10 ms and at or above 80 ms,
# so that every voxel is counted
inner_edges = np.arange(10, 85, 5)
bin_labels = (
    ["<10"]
    + [f"{low}-{high}" for low, high in zip(inner_edges[:-1], inner_edges[1:])]
    + [f"≥{inner_edges[-1]}"]
)
bin_idx = np.digitize(voxel_t2s, inner_edges)
bin_positions = np.arange(len(bin_labels))
ratio_percentiles = np.full((len(bin_labels), 3), np.nan)
n_voxels = np.zeros(len(bin_labels), dtype=int)
for i_bin in range(len(bin_labels)):
    in_bin = bin_idx == i_bin
    n_voxels[i_bin] = in_bin.sum()
    if n_voxels[i_bin] > 0:
        ratio_percentiles[i_bin] = np.percentile(tsnr_ratio[in_bin], [25, 50, 75])

fig, axes = plt.subplots(
    figsize=(12, 7),
    nrows=2,
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)
axes[0].fill_between(
    bin_positions,
    ratio_percentiles[:, 0],
    ratio_percentiles[:, 2],
    alpha=0.3,
    label="Interquartile range",
)
axes[0].plot(bin_positions, ratio_percentiles[:, 1], marker="o", linewidth=2, label="Median")
axes[0].axhline(1, color="black", linestyle="--", linewidth=1)
axes[0].set_ylabel("TSNR ratio\n(optimal combination / echo 2)", fontsize=14)
axes[0].legend(fontsize=12)
axes[1].bar(bin_positions, n_voxels, width=0.8, color="gray")
axes[1].set_ylabel("Voxels", fontsize=14)
axes[1].set_xlabel("$T_{2}^*$ (ms)", fontsize=14)
axes[1].set_xticks(bin_positions)
axes[1].set_xticklabels(bin_labels, rotation=45, ha="right")
fig.tight_layout()
glue_figure("figure_tsnr_ratio_by_t2s", fig, display=False)
```

```{figure} figures/figure_tsnr_ratio_by_t2s.png
:name: "figure_tsnr_ratio_by_t2s"
:align: center

Ratio of the TSNR of the optimally combined data to the TSNR of the second echo,
for voxels grouped by $T_{2}^*$.
The first and last groups are open-ended, so every voxel in the analysis is counted.
The dashed line marks equal TSNR.
The bottom panel shows the number of voxels in each group.
```

In voxels with a typical $T_{2}^*$ (40-60 ms), the optimally combined data have roughly 1.3-1.5 times the TSNR of the second echo.
In voxels with a short $T_{2}^*$ (below 25 ms), which are the voxels affected by dropout,
the gain is two times or more, and it keeps growing as $T_{2}^*$ gets shorter.
The pattern reverses in voxels with a very long $T_{2}^*$ (above about 75 ms),
where the second echo has slightly higher TSNR than the optimal combination.
Those voxels include cerebrospinal fluid and voxels where the $T_{2}^*$ fit is unreliable,
with some estimates of several hundred milliseconds.

There are a few caveats.
First, TSNR is not the same as BOLD sensitivity.
The first echo has the highest TSNR of all of the echoes,
but it has little BOLD contrast, because BOLD contrast grows with the echo time.
The optimal combination weights balance the two,
and {cite:t}`poser2006bold-bda` found that the gains in BOLD sensitivity from combining echoes
were largest in regions affected by susceptibility artifacts.
Second, the optimal combination can only recover signal that was acquired.
If the signal in a voxel has already decayed by the first echo,
or has been displaced by distortion, no combination of the echoes will bring it back.
This is one reason to keep the first echo time short
(see [Acquiring Multi-Echo Data](Acquiring_Multi_Echo_Data.md)).
