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

# Dual-Echo Denoising with `nilearn`

Dual-echo fMRI leverages one of the same principles motivating multi-echo fMRI;
namely, that BOLD contrast increases with echo time, so earlier echoes tend to be more affected by non-BOLD noise than later ones.
At an early enough echo time (<5ms for 3T scanners), the signal is almost entirely driven by non-BOLD noise.
When it comes to denoising, this means that, if you acquire data with both an early echo time and a more typical echo time (~30ms for 3T),
you can simply regress the earlier echo's time series out of the later echo's time series, which will remove a lot of non-BOLD noise.

Additionally, dual-echo fMRI comes at no real cost in terms of temporal or spatial resolution, unlike multi-echo fMRI.
For multi-echo denoising to work, you need to have at least one echo time that is _later_ than the typical echo time,
which means decreasing your temporal resolution, all else remaining equal. In the case of dual-echo fMRI,
you only need a _shorter_ echo time, which occurs in what is essentially "dead time" in the pulse sequence.

Dual-echo denoising was originally proposed in [Bright & Murphy (2013)](https://dx.doi.org/10.1016%2Fj.neuroimage.2012.09.043).


```{code-cell} ipython3
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import nibabel as nb
from book_utils import regress_one_image_out_of_another
from myst_nb import glue
from nilearn import plotting

data_path = os.path.abspath('../DATA')
```

```{code-cell} ipython3
te1_img = os.path.join(
    data_path,
    "sub-04570/func/sub-04570_task-rest_echo-1_space-scanner_desc-partialPreproc_bold.nii.gz",
)
te2_img = os.path.join(
    data_path,
    "sub-04570/func/sub-04570_task-rest_echo-2_space-scanner_desc-partialPreproc_bold.nii.gz",
)
mask_img = os.path.join(
    data_path, "sub-04570/func/sub-24053_ses-1_task-rat_rec-nordic_dir-PA_run-01_desc-brain_mask.nii.gz"
)
denoised_img = regress_one_image_out_of_another(te2_img, te1_img, mask_img)
```

```{code-cell} ipython3
fig, axes = plt.subplots(figsize=(16, 16), nrows=3)

plotting.plot_carpet(te2_img, axes=axes[0], figure=fig)
axes[0].set_title("First Echo (BAD)", fontsize=20)
plotting.plot_carpet(te1_img, axes=axes[1], figure=fig)
axes[1].set_title("Second Echo (GOOD)", fontsize=20)
plotting.plot_carpet(denoised_img, axes=axes[2], figure=fig)
axes[2].set_title("Denoised Data (GREAT)", fontsize=20)
axes[0].xaxis.set_visible(False)
axes[1].xaxis.set_visible(False)
axes[0].spines["bottom"].set_visible(False)
axes[1].spines["bottom"].set_visible(False)
fig.tight_layout()
glue("figure_dual_echo_results", fig, display=False)
```

```{glue:figure} figure_dual_echo_results
:name: "figure_dual_echo_results"
:align: center

Results of dual-echo regression.
```
