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

Dual-echo fMRI builds on one of the principles behind multi-echo fMRI:
BOLD-related signal changes scale with echo time, while many non-BOLD artifacts do not.
The fractional BOLD signal change is roughly proportional to TE,
so the BOLD contribution approaches zero as TE approaches zero.
Fluctuations in the signal amplitude ($S_0$), such as those caused by head motion and inflow,
affect all echoes, including very short ones.
At a short enough echo time (a few milliseconds; [Bright & Murphy (2013)](https://dx.doi.org/10.1016%2Fj.neuroimage.2012.09.043) used 3.3 ms at 3T),
the signal is therefore dominated by these non-BOLD fluctuations.

When it comes to denoising, this means that, if you acquire data with both a very short echo time and a more typical echo time (~30 ms at 3T),
you can regress the short echo's time series out of the later echo's time series, voxel by voxel,
which removes much of the non-BOLD noise.
The short echo is not completely free of BOLD signal, so a small amount of true signal is removed too.

Dual-echo fMRI also needs less extra acquisition time than multi-echo fMRI with several long echoes.
Multi-echo denoising methods such as ME-ICA work best with three or more echoes spread around $T_2^*$,
which usually means acquiring echoes later than the typical single-echo TE and so lengthening the minimum TR, all else remaining equal.
Dual-echo fMRI instead adds an echo _before_ the typical echo time.
With spiral-out readouts, this echo can fit into what is otherwise "dead time" in the pulse sequence.
With standard echo-planar imaging, the center of k-space is reached in the middle of the readout,
so a very short first echo requires techniques such as partial Fourier or in-plane acceleration,
and adding the extra readout may still lengthen the minimum TR.

Dual-echo denoising was originally proposed in [Bright & Murphy (2013)](https://dx.doi.org/10.1016%2Fj.neuroimage.2012.09.043).

```{warning}
The tutorial dataset was not acquired as dual-echo data.
Its first echo time is around 14 ms, not the few milliseconds the method calls for,
so the first echo still carries substantial BOLD signal.
Regressing it out of the second echo removes some real BOLD signal along with the noise.
Treat the example below as a demonstration of the procedure, not of the results you would get with true dual-echo data.
```

```{code-cell} ipython3
import json
import os

import matplotlib.pyplot as plt
from book_utils import glue_figure, regress_one_image_out_of_another
from nilearn import plotting

data_path = os.path.abspath('../data')
```

```{code-cell} ipython3
:tags: [hide-output]
func_dir = os.path.join(data_path, "ds006185/sub-24053/ses-1/func/")
prefix = "sub-24053_ses-1_task-rat_dir-PA_run-01"
te1_img = os.path.join(func_dir, f"{prefix}_echo-1_part-mag_desc-preproc_bold.nii.gz")
te2_img = os.path.join(func_dir, f"{prefix}_echo-2_part-mag_desc-preproc_bold.nii.gz")
mask_img = os.path.join(func_dir, f"{prefix}_part-mag_desc-brain_mask.nii.gz")

echo_times = []
for img in (te1_img, te2_img):
    with open(img.replace(".nii.gz", ".json")) as fobj:
        echo_times.append(json.load(fobj)["EchoTime"] * 1000)

print(f"Echo times: {echo_times[0]:.1f} ms and {echo_times[1]:.1f} ms")

# Regress the first (shorter) echo out of the second (later) echo, voxel by voxel.
denoised_img = regress_one_image_out_of_another(te2_img, te1_img, mask_img)
```

```{code-cell} ipython3
:tags: [hide-output]
fig, axes = plt.subplots(figsize=(16, 16), nrows=3)

plotting.plot_carpet(te1_img, mask_img=mask_img, axes=axes[0], figure=fig)
axes[0].set_title(f"First echo (TE = {echo_times[0]:.1f} ms)", fontsize=20)
plotting.plot_carpet(te2_img, mask_img=mask_img, axes=axes[1], figure=fig)
axes[1].set_title(f"Second echo (TE = {echo_times[1]:.1f} ms)", fontsize=20)
plotting.plot_carpet(denoised_img, mask_img=mask_img, axes=axes[2], figure=fig)
axes[2].set_title("Second echo with the first echo regressed out", fontsize=20)
axes[0].xaxis.set_visible(False)
axes[1].xaxis.set_visible(False)
axes[0].spines["bottom"].set_visible(False)
axes[1].spines["bottom"].set_visible(False)
fig.tight_layout()
glue_figure("figure_dual_echo_results", fig, display=False)
```

```{figure} figures/figure_dual_echo_results.png
:name: "figure_dual_echo_results"
:align: center

Carpet plots of the first echo, the second echo,
and the second echo after voxelwise regression of the first echo.
```
