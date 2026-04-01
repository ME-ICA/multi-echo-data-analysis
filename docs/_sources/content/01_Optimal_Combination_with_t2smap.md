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

```{glue:figure} figure_t2starmap
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
)
glue_figure("figure_s0map", fig, display=False)
```

```{glue:figure} figure_s0map
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

```{glue:figure} figure_t2smap_epi_plots
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
axes[0].set_title("TE30 TSNR", fontsize=16)
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

```{glue:figure} figure_t2smap_t2snr
:name: "figure_t2smap_t2snr"
:align: center

TSNR map of each of the echoes in the original data, along with the TSNR map of the optimally combined data.
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

```{glue:figure} figure_echo2_carpet
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

```{glue:figure} figure_optcom_carpet
:name: "figure_optcom_carpet"
:align: center

Carpet plot of the optimally combined data.
```
