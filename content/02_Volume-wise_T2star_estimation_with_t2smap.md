---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Volume-wise T2*/S0 estimation with `t2smap`

Use {py:func}`tedana.workflows.t2smap_workflow` {cite:p}`DuPre2021` to calculate volume-wise T2*/S0,
as in {cite:t}`power2018ridding` and {cite:t}`heunis2021effects`.

```{code-cell} ipython3
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from myst_nb import glue
from nilearn import image, plotting
from repo2data.repo2data import Repo2Data
from tedana import workflows

# Install the data if running locally, or point to cached data if running on neurolibre
DATA_REQ_FILE = os.path.join("../binder/data_requirement.json")

# Download data
repo2data = Repo2Data(DATA_REQ_FILE)
data_path = repo2data.install()
data_path = os.path.abspath(data_path[0])
```

```{code-cell} ipython3
func_dir = os.path.join(data_path, "func/")
data_files = [
    os.path.join(
        func_dir,
        "sub-04570_task-rest_echo-1_space-scanner_desc-partialPreproc_bold.nii.gz",
    ),
    os.path.join(
        func_dir,
        "sub-04570_task-rest_echo-2_space-scanner_desc-partialPreproc_bold.nii.gz",
    ),
    os.path.join(
        func_dir,
        "sub-04570_task-rest_echo-3_space-scanner_desc-partialPreproc_bold.nii.gz",
    ),
    os.path.join(
        func_dir,
        "sub-04570_task-rest_echo-4_space-scanner_desc-partialPreproc_bold.nii.gz",
    ),
]
echo_times = [12.0, 28.0, 44.0, 60.0]
mask_file = os.path.join(
    func_dir, "sub-04570_task-rest_space-scanner_desc-brain_mask.nii.gz"
)
confounds_file = os.path.join(
    func_dir, "sub-04570_task-rest_desc-confounds_timeseries.tsv"
)

out_dir = os.path.join(data_path, "fit")
```

```{code-cell} ipython3
workflows.t2smap_workflow(
    data_files,
    echo_times,
    out_dir=out_dir,
    mask=mask_file,
    prefix="sub-04570_task-rest_space-scanner",
    fittype="loglin",
    fitmode="ts",
)
```

```{code-cell} ipython3
out_files = sorted(glob(os.path.join(out_dir, "*")))
out_files = [os.path.basename(f) for f in out_files]
print("\n".join(out_files))
```

```{code-cell} ipython3
fig, axes = plt.subplots(figsize=(16, 16), nrows=3)

plotting.plot_carpet(
    os.path.join(out_dir, "sub-04570_task-rest_space-scanner_desc-optcom_bold.nii.gz"),
    axes=axes[0],
    figure=fig,
)
axes[0].set_title("Optimally Combined Data", fontsize=20)
plotting.plot_carpet(
    os.path.join(out_dir, "sub-04570_task-rest_space-scanner_T2starmap.nii.gz"),
    axes=axes[1],
    figure=fig,
)
axes[1].set_title("T2* Estimates", fontsize=20)
plotting.plot_carpet(
    os.path.join(out_dir, "sub-04570_task-rest_space-scanner_S0map.nii.gz"),
    axes=axes[2],
    figure=fig,
)
axes[2].set_title("S0 Estimates", fontsize=20)
axes[0].xaxis.set_visible(False)
axes[1].xaxis.set_visible(False)
axes[0].spines["bottom"].set_visible(False)
axes[1].spines["bottom"].set_visible(False)
fig.tight_layout()

glue("figure_volumewise_t2ss0_carpets", fig, display=False)
```

```{glue:figure} figure_volumewise_t2ss0_carpets
:name: "figure_volumewise_t2ss0_carpets"
:align: center

Carpet plots of optimally combined data, along with volume-wise T2* and S0 estimates.
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(16, 8))
plotting.plot_stat_map(
    image.mean_img(
        os.path.join(out_dir, "sub-04570_task-rest_space-scanner_T2starmap.nii.gz")
    ),
    vmax=0.6,
    draw_cross=False,
    bg_img=None,
    figure=fig,
    axes=ax,
)
glue("figure_mean_volumewise_t2s", fig, display=False)
```

```{glue:figure} figure_mean_volumewise_t2s
:name: "figure_mean_volumewise_t2s"
:align: center

Mean map from the volume-wise T2* estimation.
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(16, 8))
plotting.plot_stat_map(
    image.mean_img(
        os.path.join(out_dir, "sub-04570_task-rest_space-scanner_S0map.nii.gz")
    ),
    vmax=8000,
    draw_cross=False,
    bg_img=None,
    figure=fig,
    axes=ax,
)
glue("figure_mean_volumewise_s0", fig, display=False)
```

```{glue:figure} figure_mean_volumewise_s0
:name: "figure_mean_volumewise_s0"
:align: center

Mean map from the volume-wise S0 estimation.
```

```{code-cell} ipython3
fig, axes = plt.subplots(figsize=(16, 15), nrows=5)
plotting.plot_epi(
    image.mean_img(data_files[0]),
    draw_cross=False,
    bg_img=None,
    cut_coords=[-10, 0, 10, 20, 30, 40, 50, 60, 70],
    display_mode="z",
    axes=axes[0],
)
plotting.plot_epi(
    image.mean_img(data_files[1]),
    draw_cross=False,
    bg_img=None,
    cut_coords=[-10, 0, 10, 20, 30, 40, 50, 60, 70],
    display_mode="z",
    axes=axes[1],
)
plotting.plot_epi(
    image.mean_img(data_files[2]),
    draw_cross=False,
    bg_img=None,
    cut_coords=[-10, 0, 10, 20, 30, 40, 50, 60, 70],
    display_mode="z",
    axes=axes[2],
)
plotting.plot_epi(
    image.mean_img(data_files[3]),
    draw_cross=False,
    bg_img=None,
    cut_coords=[-10, 0, 10, 20, 30, 40, 50, 60, 70],
    display_mode="z",
    axes=axes[3],
)
plotting.plot_epi(
    image.mean_img(
        os.path.join(
            out_dir, "sub-04570_task-rest_space-scanner_desc-optcom_bold.nii.gz"
        )
    ),
    draw_cross=False,
    bg_img=None,
    cut_coords=[-10, 0, 10, 20, 30, 40, 50, 60, 70],
    display_mode="z",
    axes=axes[4],
)
glue("figure_mean_echos_and_optcom", fig, display=False)
```

```{glue:figure} figure_mean_echos_and_optcom
:name: "figure_mean_echos_and_optcom"
:align: center

Mean images of the echo-wise data and the optimally combined data.
```

```{code-cell} ipython3
te30_tsnr = image.math_img(
    "(np.nanmean(img, axis=3) / np.nanstd(img, axis=3)) * mask",
    img=data_files[1],
    mask=mask_file,
)
oc_tsnr = image.math_img(
    "(np.nanmean(img, axis=3) / np.nanstd(img, axis=3)) * mask",
    img=os.path.join(
        out_dir, "sub-04570_task-rest_space-scanner_desc-optcom_bold.nii.gz"
    ),
    mask=mask_file,
)
vmax = np.nanmax(np.abs(oc_tsnr.get_fdata()))

fig, axes = plt.subplots(figsize=(10, 8), nrows=2)
plotting.plot_stat_map(
    te30_tsnr,
    draw_cross=False,
    bg_img=None,
    threshold=0.1,
    cut_coords=[0, 10, 10],
    vmax=vmax,
    symmetric_cbar=False,
    axes=axes[0],
)
axes[0].set_title("TE30 TSNR", fontsize=16)
plotting.plot_stat_map(
    oc_tsnr,
    draw_cross=False,
    bg_img=None,
    threshold=0.1,
    cut_coords=[0, 10, 10],
    vmax=vmax,
    symmetric_cbar=False,
    axes=axes[1],
)
axes[1].set_title("Optimal Combination TSNR", fontsize=16)
glue("figure_t2snr_te30_and_optcom", fig, display=False)
```

```{glue:figure} figure_t2snr_te30_and_optcom
:name: "figure_t2snr_te30_and_optcom"
:align: center

TSNR map of the second echo (TE30) and the optimally combined data.
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(16, 8))
plotting.plot_carpet(
    data_files[1],
    axes=ax,
)
glue("figure_echo3_carpet", fig, display=False)
```

```{glue:figure} figure_echo3_carpet
:name: "figure_echo3_carpet"
:align: center

Carpet plot of the third echo.
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(16, 8))
plotting.plot_carpet(
    os.path.join(out_dir, "sub-04570_task-rest_space-scanner_desc-optcom_bold.nii.gz"),
    axes=ax,
)
glue("figure_carpet_optcom", fig, display=False)
```

```{glue:figure} figure_carpet_optcom
:name: "figure_carpet_optcom"
:align: center

Carpet plot of the optimally combined data.
```
