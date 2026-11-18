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

# Signal Decay

In this chapter, we illustrate how BOLD signal decays and how we can glean useful information from that decay rate.

Throughout the chapter, we describe the signal with the monoexponential decay model:

```{math}
:label: monoexponential_decay_signal
S(TE) = S_{0} e^{-TE / T_{2}^{*}} = S_{0} e^{-TE \cdot R_{2}^{*}}
```

$S_{0}$ is the signal we would measure at $TE = 0$, before any decay has happened.
$T_{2}^{*}$ is the time constant of the decay: the longer $T_{2}^{*}$ is, the more slowly the signal fades.
$R_{2}^{*} = 1 / T_{2}^{*}$ is the same quantity expressed as a decay *rate*.
The BOLD effect works through $T_{2}^{*}$:
when local blood oxygenation increases, there is less deoxygenated hemoglobin to disturb the magnetic field,
so $T_{2}^{*}$ gets longer and the signal decays more slowly.
Many sources of noise, such as head motion and inflow, instead change $S_{0}$.

The monoexponential model is an approximation.
A voxel contains several pools of water, such as water in tissue, in blood, in cerebrospinal fluid, and trapped between layers of myelin,
and each pool decays at its own rate.
The decay of the whole voxel is therefore better described by a sum of exponentials,
such as a biexponential model with two pools.
However, each extra pool adds parameters to estimate,
and multi-echo fMRI usually has only three to five echoes per volume, each with its own noise,
which is not enough to estimate those extra parameters reliably.
For that reason, multi-echo fMRI methods, including all of the ones in this book, use the monoexponential model.
[](T2star_S0_Variability.md) shows what happens when a monoexponential model is fit to decay that is not monoexponential.

To illustrate this, we will use a combination of simulated and real multi-echo data.

```{code-cell} ipython3
:tags: [hide-cell]
import json
import os
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import display
from matplotlib.animation import FuncAnimation
from myst_nb import glue
from nilearn import image, masking, plotting
from nilearn.glm import first_level
from scipy import signal

from book_utils import glue_figure, load_pafin, predict_bold_signal

sns.set_style("whitegrid")

data_path = os.path.abspath("../data")

out_dir = os.path.join(data_path, "signal-decay")
os.makedirs(out_dir, exist_ok=True)
```

```{code-cell} ipython3
:tags: [hide-cell]
data = load_pafin(data_path)
ted_dir = os.path.join(data_path, "tedana")
n_echoes = len(data['echo_times'])
# Exact echo times in milliseconds, for model fitting
tes_ms = np.array(data['echo_times']) * 1000
# Echo times rounded to whole milliseconds, for indexing the simulated curves below
data['echo_times'] = [int(np.round(te * 1000)) for te in data['echo_times']]

# Drop the last cubehelix color, which is too light to see on a white background
pal = sns.color_palette("cubehelix", n_echoes + 1)[:n_echoes]

img = image.index_img(data['echo_files'][0], 0)
arr = img.get_fdata()
vmax = np.max(arr)
idx = np.vstack(np.where(arr > 1000))

min_x, min_y, min_z = np.min(idx, axis=1)
max_x, max_y, max_z = np.max(idx, axis=1)

imgs = []
for f in data['echo_files']:
    img = image.index_img(f, 0)
    arr = img.get_fdata()
    arr = arr[min_x:max_x, min_y:max_y, min_z:max_z]
    img = nb.Nifti1Image(arr, img.affine, img.header)
    imgs.append(img)

mask = nb.load(data['mask'])

# Component parameter estimates
betas_file = os.path.join(
    ted_dir, "sub-24053_ses-1_task-rat_dir-PA_run-01_desc-ICA_components.nii.gz"
)
beta_maps = masking.apply_mask(betas_file, mask)

# Component table
comp_tbl = pd.read_table(
    os.path.join(ted_dir, "sub-24053_ses-1_task-rat_dir-PA_run-01_desc-tedana_metrics.tsv"),
    index_col="Component",
)

# Get voxel index for voxel most related to component with highest kappa value
acc_comp_tbl = comp_tbl.loc[comp_tbl["classification"] == "accepted"]
high_kappa_comp = acc_comp_tbl.sort_values(by="kappa", ascending=False).index.values[0]
high_kappa_comp_val = int(high_kappa_comp.split("_")[1])
voxel_idx = np.where(
    beta_maps[high_kappa_comp_val, :] == np.max(beta_maps[high_kappa_comp_val, :])
)[0][0]

# Extract the time series for the high-kappa voxel
# ts is rounded to make it more likely the figures are consistent when generated on different computers
#  This hopefully avoids unnecessary new files in the git history.
arr = [masking.apply_mask(f, mask) for f in data['echo_files']]
ts = np.round([d[:, voxel_idx] for d in arr], decimals=5)
ts_1d = np.hstack(ts)

n_trs = ts[0].shape[0]
```

## Signal decays as echo time increases

To start, let us look at how BOLD signal changes with increasing echo time in real data.
The data we use is a single resting-state run, with five echo times.

```{code-cell} ipython3
:tags: [hide-cell, hide-output]
plt.style.use("dark_background")

fig, axes = plt.subplots(figsize=(16, 15), nrows=len(data['echo_files']))
fig.subplots_adjust(left=0, right=0.92, top=0.97, bottom=0.01, hspace=0.15)
# Use the same intensity range for every echo, so that the decay is visible
vmax = np.max(imgs[0].get_fdata())
for i_echo, img in enumerate(imgs):
    plotting.plot_epi(
        img,
        cut_coords=[-15, 0, 15, 30, 45],
        display_mode="z",
        annotate=False,
        draw_cross=False,
        colorbar=False,
        axes=axes[i_echo],
        figure=fig,
        vmin=0,
        vmax=vmax,
        cmap="gray",
    )
    axes[i_echo].set_title(f"TE = {tes_ms[i_echo]:.1f} ms", fontsize=18, pad=0)

# One colorbar for all of the echoes
cax = fig.add_axes([0.93, 0.25, 0.012, 0.5])
cbar = fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, vmax), cmap="gray"),
    cax=cax,
)
cbar.set_label("Signal", fontsize=16)
cbar.ax.tick_params(labelsize=12)

glue_figure("fig_brain_decay", fig, display=False)

# Reset the style
plt.style.use("default")
```

```{figure} figures/fig_brain_decay.png
:name: fig_brain_decay
:align: center

The first volume of each echo, shown as axial slices from the bottom (left) to the top (right) of the brain,
with the same intensity scale for every echo.
```

As the echo time increases, the whole brain gets darker, because the signal has had more time to decay.
This means that later echoes have a lower signal-to-noise ratio (SNR) than earlier ones.
The decay is not the same everywhere, though.
Near the sinuses and ear canals (left two columns), the signal is almost gone by the third echo,
because $T_{2}^{*}$ is very short there.
The ventricles, on the other hand, become *brighter* than the surrounding tissue at later echoes,
because the cerebrospinal fluid that fills them has a long $T_{2}^{*}$ and decays slowly.
This difference in brightness between tissues is $T_{2}^{*}$-weighted contrast,
which reflects differences in the *average* $T_{2}^{*}$ of each tissue.
It is not the same thing as BOLD contrast, which comes from small *changes* in $T_{2}^{*}$ over time.
[](T2star_S0_Variability.md) describes why the average $T_{2}^{*}$ varies across the brain.

### Echo-specific data and echo time

```{code-cell} ipython3
:tags: [hide-cell]
fig, ax = plt.subplots(figsize=(14, 6))
for i_echo in range(n_echoes):
    ax.scatter(np.full(n_trs, tes_ms[i_echo]), ts[i_echo], alpha=0.05, color=pal[i_echo])

ax.set_ylabel("Signal", fontsize=16)
ax.set_xlabel("Echo Time (ms)", fontsize=16)
ax.set_xticks(np.round(tes_ms, 1))
ax.tick_params(axis="both", which="major", labelsize=14)
ax.set_xlim(0, 120)
ax.set_ylim(0, 25000)
fig.tight_layout()
glue_figure("fig_echo_scatter", fig, display=False)
```

```{figure} figures/fig_echo_scatter.png
:name: fig_echo_scatter
:align: center

The signal of a single voxel at each echo time.
Each point is one volume of the run.
```

Here, we plot the signal of a single voxel in every volume of the run, at each of the five echo times.
As expected, the signal decreases as the echo time increases.
Notice also that the spread of the points, which reflects how much the signal fluctuates over time,
is smallest at the first echo, even though the signal is largest there.
This voxel was chosen because it has strong BOLD fluctuations,
and BOLD fluctuations have little effect at short echo times.
We will see why later in this chapter.

### The signal itself changes with echo time as well

Let us visualize the time series of the same voxel at each echo time.
While the overall scale of the signal decreases with echo time, the pattern of the signal changes as well.

```{code-cell} ipython3
:tags: [hide-cell]
fig, axes = plt.subplots(n_echoes, sharex=True, sharey=False, figsize=(14, 6))
for i_echo in range(n_echoes):
    axes[i_echo].plot(ts[i_echo], color=pal[i_echo])
    axes[i_echo].set_ylabel(
        f"{tes_ms[i_echo]:.1f} ms",
        rotation=0,
        va="center",
        ha="right",
        fontsize=14,
    )
    axes[i_echo].set_yticks([])

axes[-1].set_xlabel("Volume", fontsize=16)
axes[-1].set_xlim(0, n_trs - 1)
fig.tight_layout()
glue_figure("fig_echo_timeseries", fig, display=False)
```

```{figure} figures/fig_echo_timeseries.png
:name: fig_echo_timeseries
:align: center

The time series of a single voxel at each echo time.
Each time series is scaled separately, so the y-axes are not comparable.
```

The later echoes share a slow, wave-like pattern, which is the BOLD signal in this voxel.
The same pattern is much weaker at the first echo, which is dominated by small, fast fluctuations,
along with a spike in the very first volume.
Early in a run, before the magnetization has reached a steady state, the signal is brighter at every echo.
That change affects $S_{0}$ rather than $T_{2}^{*}$, so it stands out most at the first echo,
where the BOLD fluctuations are weakest.

## Estimating $S_{0}$ and $T_{2}^{*}$ from the decay

With several echoes, we can estimate the two parameters of {eq}`monoexponential_decay_signal` for each voxel.
Fitting an exponential curve directly is possible, but there is a simpler way.
Taking the natural logarithm of both sides of {eq}`monoexponential_decay_signal` turns the curve into a straight line:

```{math}
:label: log_linear_model
\ln S(TE) = \ln S_{0} - R_{2}^{*} \cdot TE
```

So if we plot $\ln S$ against $TE$, the intercept of the line is $\ln S_{0}$ and its slope is $-R_{2}^{*}$,
and both can be estimated with ordinary least squares.
This "log-linear" fit is fast, and it is the default in `tedana` (`fittype="loglin"`).
`tedana` also offers a nonlinear fit of the exponential curve (`fittype="curvefit"`), which is slower.

Below, we fit the model to the voxel from {numref}`fig_echo_scatter`.
We fit all of the volumes at once, so we get a single $S_{0}$ and $T_{2}^{*}$ that describe the voxel's average decay.

```{code-cell} ipython3
:tags: [hide-cell]
# Log-transform the data. Like tedana, we add 1 to avoid taking the log of zero.
# With signal values in the thousands, this makes no practical difference.
log_ts = np.log(np.abs(ts_1d) + 1)

# One row per volume per echo: an intercept and the negative echo time
design = np.column_stack([np.ones(n_echoes), -tes_ms])
design = np.repeat(design, n_trs, axis=0)
intercept, r2s_fit = np.linalg.lstsq(design, log_ts, rcond=None)[0]
s0_fit = np.exp(intercept)
t2s_fit = 1 / r2s_fit

glue("s0_fit", float(s0_fit), display=False)
glue("t2s_fit", float(t2s_fit), display=False)

fit_tes = np.linspace(0, 120, 241)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: log-transformed data and the fitted line
for i_echo in range(n_echoes):
    axes[0].scatter(
        np.full(n_trs, tes_ms[i_echo]),
        np.log(np.abs(ts[i_echo]) + 1),
        alpha=0.05,
        color=pal[i_echo],
    )

axes[0].plot(fit_tes, intercept - r2s_fit * fit_tes, color="black", linewidth=2)
axes[0].annotate(
    (
        f"intercept = $\\ln S_0$ = {intercept:.2f}\n"
        f"slope = $-R_2^*$ = {-r2s_fit:.4f} ms$^{{-1}}$"
    ),
    xy=(0.04, 0.06),
    xycoords="axes fraction",
    fontsize=14,
    bbox=dict(fc="white", ec="gray", lw=1),
)
axes[0].set_title("A. Log-transformed signal is linear in TE", fontsize=18, loc="left")
axes[0].set_ylabel("ln(signal)", fontsize=16)

# Panel B: original data and the fitted exponential curve
for i_echo in range(n_echoes):
    axes[1].scatter(
        np.full(n_trs, tes_ms[i_echo]),
        ts[i_echo],
        alpha=0.05,
        color=pal[i_echo],
    )

axes[1].plot(fit_tes, s0_fit * np.exp(-fit_tes / t2s_fit), color="black", linewidth=2)
axes[1].scatter(0, s0_fit, marker="*", s=400, color="red", clip_on=False, zorder=3)
axes[1].annotate(
    f"$S_0$ = {s0_fit:.0f}",
    xy=(0, s0_fit),
    xytext=(8, s0_fit),
    fontsize=14,
    va="center",
)
axes[1].plot(
    [t2s_fit, t2s_fit, 0],
    [0, s0_fit / np.e, s0_fit / np.e],
    color="blue",
    linestyle="--",
)
axes[1].annotate(
    (
        f"At TE = $T_2^*$ = {t2s_fit:.1f} ms,\n"
        "the signal has decayed to\n"
        f"1/e (37%) of $S_0$"
    ),
    xy=(t2s_fit, s0_fit / np.e),
    xytext=(15, s0_fit * 0.12),
    fontsize=14,
    arrowprops=dict(arrowstyle="->", color="blue"),
)
axes[1].set_title("B. The same fit as an exponential curve", fontsize=18, loc="left")
axes[1].set_ylabel("Signal", fontsize=16)
axes[1].set_ylim(0, np.ceil(s0_fit / 1000) * 1000 + 1000)

for ax in axes:
    ax.set_xlabel("Echo Time (ms)", fontsize=16)
    ax.set_xlim(0, 120)
    ax.set_xticks(np.round(tes_ms, 1))
    ax.tick_params(axis="both", which="major", labelsize=12)

fig.tight_layout()
glue_figure("fig_decay_fit", fig, display=False)
```

```{figure} figures/fig_decay_fit.png
:name: fig_decay_fit
:align: center

The monoexponential model fit to one voxel.
(A) After a log transform, the signal falls on a straight line,
whose intercept and slope give $S_{0}$ and $R_{2}^{*}$.
(B) The same fit, shown on the original scale.
The red star marks $S_{0}$, and the dashed blue lines mark $T_{2}^{*}$,
the echo time at which the signal has decayed to 37% of $S_{0}$.
```

For this voxel, the fit gives $S_{0}$ ≈ {glue:text}`s0_fit:.0f` and $T_{2}^{*}$ ≈ {glue:text}`t2s_fit:.1f` ms.
$T_{2}^{*}$ has a simple interpretation:
it is the echo time at which the signal has decayed to $1/e$, or about 37%, of $S_{0}$.
This voxel's $T_{2}^{*}$ is longer than is typical for gray matter at 3T (roughly 40-55 ms).
We chose the voxel because it has strong BOLD fluctuations, not because its decay is typical,
and [](T2star_S0_Variability.md) describes why $T_{2}^{*}$ varies so much from voxel to voxel.

### Noise across echoes

Every measurement also contains thermal noise, whose amplitude is roughly the same at every echo time.
Because the signal decays, the same amount of noise is a larger fraction of the signal at later echoes,
so later echoes have a lower signal-to-noise ratio.
This has a few consequences:

- In panel A of {numref}`fig_decay_fit`, the log-transformed points spread out more at later echoes.
  Part of that spread is BOLD fluctuation, which also grows with echo time as a fraction of the signal (as we will see below),
  but part of it is noise.
- MRI magnitude images cannot be negative, so once the signal has decayed to the level of the noise,
  the measured values level off instead of continuing to fall.
  Echoes at this "noise floor" make the decay look slower than it really is.
- Fitting the decay to a single volume, with only a handful of noisy measurements,
  gives much noisier estimates than fitting all of the volumes at once, as we did here.

There are two more things to keep in mind.
First, the fit is only as good as the echoes that go into it.
In regions with strong signal dropout, such as orbitofrontal cortex,
the later echoes contain almost nothing but noise,
so `tedana` builds an *adaptive mask* that limits the fit in each voxel to the echoes with good signal
(see [](TE_Dependence.md)).
Second, the values we estimated here describe the voxel's *average* decay.
In the next section, we look at how $S_{0}$ and $T_{2}^{*}$ change from one volume to the next,
and why we need multiple echoes to tell those changes apart.

## The impact of $T_{2}^{*}$ and $S_{0}$ fluctuations on BOLD signal

In this section, we use simulations to investigate the two factors that drive changes in the signal decay pattern:
$T_{2}^{*}$ and $S_{0}$.
We simulate a voxel with an average $S_{0}$ of 16000 and an average $T_{2}^{*}$ of 45 ms,
a typical value for gray matter at 3T,
and a single-echo acquisition with an echo time of 30 ms, which is common for single-echo fMRI at 3T.

```{code-cell} ipython3
# The echo time of a typical single-echo scan at 3T
SINGLEECHO_TE = np.array([30])

# Echo times for a nice, smooth curve
FULLCURVE_TES = np.round(np.arange(0, 121, 1), decimals=5)

MEAN_T2S = 45
MEAN_S0 = 16000
```

### Simulate $T_{2}^{*}$ and $S_{0}$ fluctuations

Next, we simulate a short run of 21 volumes, in which $T_{2}^{*}$ and $S_{0}$ fluctuate over time.
To make the fluctuations easy to see, they are much larger (10%) than real fluctuations, which are a few percent or less.

```{code-cell} ipython3
# We'll convolve with HRF just for smoothness
hrf = first_level.spm_hrf(1, oversampling=1)

N_VOLS = 21

SCALING_FRACTION = 0.1  # used to scale standard deviation

# Simulate the relative T2*/S0 fluctuations.
# The original time series is random noise from a normal distribution,
# convolved with the HRF.
rng = np.random.default_rng(42)
ts = rng.normal(loc=0, scale=1, size=(N_VOLS + 20,))
ts = signal.convolve(ts, hrf)[20 : N_VOLS + 20]
ts *= SCALING_FRACTION / np.std(ts)
ts -= np.round(ts[0], decimals=5)

t2s_ts = (ts * MEAN_T2S) + MEAN_T2S
s0_ts = (ts * MEAN_S0) + MEAN_S0
```

#### BOLD signal decay for a standard single-echo scan

```{code-cell} ipython3
:tags: [hide-cell]
fullcurve_signal = predict_bold_signal(FULLCURVE_TES, [MEAN_S0], [MEAN_T2S])

fig, ax = plt.subplots(figsize=(14, 7.5))
ax.plot(
    FULLCURVE_TES,
    fullcurve_signal[:, 0],
    alpha=0.5,
    color="black",
)
# Example for TE=30ms
x_cross = SINGLEECHO_TE[0]
y_cross = float(np.squeeze(predict_bold_signal(np.array([x_cross]), [MEAN_S0], [MEAN_T2S])))
glue("singleecho_value", y_cross, display=False)
ax.plot(x_cross, y_cross, marker="x", color="red", markersize=10, label="TE = 30 ms")
ax.plot([x_cross, x_cross, 0], [0, y_cross, y_cross], color="red", linestyle="--", alpha=0.5)

ax.set_ylabel("Signal", fontsize=24)
ax.set_xlabel("Echo Time (ms)", fontsize=24)
ax.set_ylim(0, np.ceil(np.max(fullcurve_signal) / 1000) * 1000)
ax.set_xlim(0, np.max(FULLCURVE_TES))
ax.tick_params(axis="both", which="major", labelsize=14)
fig.tight_layout()
glue_figure("fig_signal_decay_single-echo", fig, display=False)
```

```{figure} figures/fig_signal_decay_single-echo.png
:name: fig_signal_decay_single-echo
:align: center

The signal decay curve for a voxel with $T_{2}^{*}$ = 45 ms (black), and the single value a single-echo scan with TE = 30 ms measures (red cross).
```

In every volume of an fMRI scan, each voxel's signal follows a decay curve like the black one,
but a single-echo scan only measures one point on that curve.
For example, if a single-echo scan is acquired at TE = 30 ms,
the signal is approximately {glue:text}`singleecho_value:.0f` (red cross and dashed lines).

#### BOLD signal decay and BOLD contrast

```{code-cell} ipython3
:tags: [hide-cell]
# T2* is exaggerated in both directions, so that the difference is easy to see
T2S_LOW_ACTIVITY = 40
T2S_HIGH_ACTIVITY = 50
curve_active = np.squeeze(predict_bold_signal(FULLCURVE_TES, [MEAN_S0], [T2S_HIGH_ACTIVITY]))
curve_inactive = np.squeeze(predict_bold_signal(FULLCURVE_TES, [MEAN_S0], [T2S_LOW_ACTIVITY]))

# The echo time at which the difference between the curves is largest
max_contrast_te = (
    np.log(T2S_HIGH_ACTIVITY / T2S_LOW_ACTIVITY)
    / (1 / T2S_LOW_ACTIVITY - 1 / T2S_HIGH_ACTIVITY)
)
glue("max_contrast_te", float(max_contrast_te), display=False)

fig, ax = plt.subplots(figsize=(14, 7.5))
ax.plot(
    FULLCURVE_TES,
    curve_active,
    color="red",
    label=f"High activity ($T_2^*$ = {T2S_HIGH_ACTIVITY} ms)",
)
ax.plot(
    FULLCURVE_TES,
    curve_inactive,
    color="blue",
    label=f"Low activity ($T_2^*$ = {T2S_LOW_ACTIVITY} ms)",
)

# Vertical lines show the difference between the curves at a few echo times
for te in [15, int(np.round(max_contrast_te)), 90]:
    ax.plot(
        [te, te],
        [curve_inactive[te], curve_active[te]],
        color="green",
        linewidth=3,
    )
    ax.annotate(
        f"{te} ms",
        xy=(te, curve_active[te]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=14,
        color="green",
    )

ax.set_ylabel("Signal", fontsize=24)
ax.set_xlabel("Echo Time (ms)", fontsize=24)
ax.set_ylim(0, np.ceil(np.max(fullcurve_signal) / 1000) * 1000)
ax.set_xlim(0, np.max(FULLCURVE_TES))
ax.legend(fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=14)
fig.tight_layout()
glue_figure("fig_signal_decay_contrast", fig, display=False)
```

```{figure} figures/fig_signal_decay_contrast.png
:name: fig_signal_decay_contrast
:align: center

Signal decay curves for a voxel at low activity (blue) and high activity (red).
The green lines show the difference between the curves at three echo times.
```

This plot shows the signal decay for two levels of activity.
When activity increases, blood oxygenation increases, so $T_{2}^{*}$ gets longer and the signal decays more slowly.
The difference in $T_{2}^{*}$ here (40 vs. 50 ms) is exaggerated to make it visible;
real BOLD changes in $T_{2}^{*}$ are a few percent.

The BOLD contrast is the difference between the two curves (green lines).
It is zero at TE = 0, grows as the echo time increases,
peaks at an echo time of about {glue:text}`max_contrast_te:.0f` ms, close to $T_{2}^{*}$,
and then shrinks again as both signals decay toward zero.
This is why single-echo fMRI uses an echo time that is close to the typical $T_{2}^{*}$ of gray matter.
Note that this is the *absolute* difference between the curves.
As a *fraction* of the signal, the difference keeps growing with echo time,
which becomes important in [](TE_Dependence.md).

### Single-echo data over time

Now let us add the fluctuations, starting with a single-echo acquisition at TE = 30 ms.
In the animation below, the bottom panel shows the signal measured at TE = 30 ms in each volume,
and the top panel shows the resulting time series.
This is what a single-echo fMRI time series is: one measurement per volume, at a single echo time.

```{code-cell} ipython3
:tags: [hide-cell]
fullcurve_signal = predict_bold_signal(FULLCURVE_TES, s0_ts, t2s_ts)
singleecho_signal = fullcurve_signal[SINGLEECHO_TE, :]

# gif code based on https://www.geeksforgeeks.org/create-an-animated-gif-using-python-matplotlib/
fig, axes = plt.subplots(
    nrows=2, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 3]}
)

# Set constant params for figure
axes[0].set_ylabel("Signal", fontsize=24)
axes[0].set_xlabel("Volume", fontsize=24)
axes[0].set_xlim(0, N_VOLS - 1)
axes[0].tick_params(axis="both", which="major", labelsize=14)
axes[1].set_ylabel("Signal", fontsize=24)
axes[1].set_xlabel("Echo Time (ms)", fontsize=24)
axes[1].set_xticks([0, 30, 60, 90, 120])
axes[1].set_ylim(0, np.ceil(np.max(fullcurve_signal) / 1000) * 1000)
axes[1].set_xlim(0, np.max(FULLCURVE_TES))
axes[1].tick_params(axis="both", which="major", labelsize=14)
fig.tight_layout()

ax0_line_plot = axes[0].plot(singleecho_signal[0, :], color="black", zorder=0)
ax0_scatter_plot = axes[0].scatter(
    0,
    singleecho_signal[:, 0],
    color="orange",
    s=150,
    label="Single-Echo Signal",
    zorder=1,
)
ax1_scatter_plot = axes[1].scatter(
    SINGLEECHO_TE,
    singleecho_signal[:, 0],
    color="orange",
    s=150,
    alpha=1.0,
    label="Single-Echo Signal",
    zorder=1,
)


def AnimationFunction(frame):
    """Function takes frame as an input."""
    ax0_scatter_plot.set_offsets(
        np.column_stack(
            (
                frame,
                singleecho_signal[:, frame],
            )
        )
    )

    ax1_scatter_plot.set_offsets(
        np.column_stack(
            (
                SINGLEECHO_TE,
                singleecho_signal[:, frame],
            )
        )
    )


anim_created = FuncAnimation(fig, AnimationFunction, frames=N_VOLS, interval=100)
html = display.HTML(anim_created.to_jshtml())
glue("fig_single-echo", html, display=False)
```

```{glue:figure} fig_single-echo
:name: fig_single-echo
:align: center

Single-echo data from a simulated voxel whose $S_{0}$ and $T_{2}^{*}$ fluctuate. Top: the time series measured at TE = 30 ms. Bottom: the measurement in the current volume.
```

### Each single-echo measurement comes from a decay curve

Next, we add the full decay curve for each volume.
Each single-echo measurement is just one point on that volume's decay curve,
and the curve itself changes from volume to volume as $S_{0}$ and $T_{2}^{*}$ fluctuate.

```{code-cell} ipython3
:tags: [hide-cell]
fullcurve_signal = predict_bold_signal(FULLCURVE_TES, s0_ts, t2s_ts)
singleecho_signal = fullcurve_signal[SINGLEECHO_TE, :]

# gif code based on https://www.geeksforgeeks.org/create-an-animated-gif-using-python-matplotlib/
fig, axes = plt.subplots(
    nrows=2, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 3]}
)

# Set constant params for figure
axes[0].set_ylabel("Signal", fontsize=24)
axes[0].set_xlabel("Volume", fontsize=24)
axes[0].set_xlim(0, N_VOLS - 1)
axes[0].tick_params(axis="both", which="major", labelsize=14)
axes[1].set_ylabel("Signal", fontsize=24)
axes[1].set_xlabel("Echo Time (ms)", fontsize=24)
axes[1].set_xticks([0, 30, 60, 90, 120])
axes[1].set_ylim(0, np.ceil(np.max(fullcurve_signal) / 1000) * 1000)
axes[1].set_xlim(0, np.max(FULLCURVE_TES))
axes[1].tick_params(axis="both", which="major", labelsize=14)
fig.tight_layout()

ax0_line_plot = axes[0].plot(singleecho_signal[0, :], color="black", zorder=0)
ax0_scatter_plot = axes[0].scatter(
    0,
    singleecho_signal[:, 0],
    color="orange",
    s=150,
    label="Single-Echo Signal",
    zorder=1,
)
ax1_line_plot = axes[1].plot(
    FULLCURVE_TES,
    fullcurve_signal[:, 0],
    alpha=0.5,
    color="black",
    zorder=0,
)[0]
ax1_scatter_plot = axes[1].scatter(
    SINGLEECHO_TE,
    singleecho_signal[:, 0],
    color="orange",
    s=150,
    alpha=1.0,
    label="Single-Echo Signal",
    zorder=1,
)


def AnimationFunction(frame):
    """Function takes frame as an input."""
    ax0_scatter_plot.set_offsets(
        np.column_stack(
            (
                frame,
                singleecho_signal[:, frame],
            )
        )
    )

    ax1_line_plot.set_data(
        (
            FULLCURVE_TES,
            fullcurve_signal[:, frame],
        )
    )
    ax1_scatter_plot.set_offsets(
        np.column_stack(
            (
                SINGLEECHO_TE,
                singleecho_signal[:, frame],
            )
        )
    )


anim_created = FuncAnimation(fig, AnimationFunction, frames=N_VOLS, interval=100)
html = display.HTML(anim_created.to_jshtml())
glue("fig_signal_decay", html, display=False)
```

```{glue:figure} fig_signal_decay
:name: fig_signal_decay
:align: center

The same single-echo data, with the full decay curve of the current volume (gray). The single-echo measurement is one point on that curve.
```

### Where $S_{0}$ and $T_{2}^{*}$ sit on the curve

We are still looking at a single-echo acquisition with TE = 30 ms.
The decay model from the start of this chapter, written for volume $t$ and with noise added, is:

```{math}
:label: monoexponential_decay_undeveloped_single_echo
S(t, TE) = S_0(t) \times e^{\frac{-TE}{T_2^{*}(t)}} + noise
```

Both $S_{0}$ and $T_{2}^{*}$ can change from volume to volume, for different reasons,
and a change in either one changes the measured signal.
In the animation below, the red star marks $S_{0}$, where the curve meets $TE = 0$,
and the blue star marks the point on the curve at $TE = T_{2}^{*}$.
Both move as the fluctuations change the curve.

```{code-cell} ipython3
:tags: [hide-cell]
fullcurve_signal = predict_bold_signal(FULLCURVE_TES, s0_ts, t2s_ts)
singleecho_signal = fullcurve_signal[SINGLEECHO_TE, :]

fig, axes = plt.subplots(
    nrows=2, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 3]}
)

t2s_value = predict_bold_signal(
    np.array([t2s_ts[0]]), np.array([s0_ts[0]]), np.array([t2s_ts[0]])
)[0]

ax0_line_plot = axes[0].plot(singleecho_signal[0, :], color="black", zorder=0)
ax0_scatter_plot = axes[0].scatter(
    0,
    singleecho_signal[:, 0],
    color="orange",
    s=150,
    label="Single-Echo Signal",
    zorder=1,
)
ax1_line_plot = axes[1].plot(
    FULLCURVE_TES,
    fullcurve_signal[:, 0],
    alpha=0.5,
    color="black",
    zorder=0,
)[0]
ax1_scatter_plot = axes[1].scatter(
    SINGLEECHO_TE,
    singleecho_signal[:, 0],
    color="orange",
    s=150,
    alpha=1.0,
    label="Single-Echo Signal",
    zorder=1,
)
ax1_t2s_scatter_plot = axes[1].scatter(
    t2s_ts[0],
    t2s_value,
    marker="*",
    color="blue",
    s=500,
    alpha=0.5,
    label="$T_{2}^{*}$",
    zorder=1,
)
ax1_s0_scatter_plot = axes[1].scatter(
    0,
    s0_ts[0],
    marker="*",
    color="red",
    s=500,
    alpha=0.5,
    label="$S_{0}$",
    clip_on=False,
    zorder=2,
)

# Set constant params for figure
axes[0].set_ylabel("Signal", fontsize=24)
axes[0].set_xlabel("Volume", fontsize=24)
axes[0].set_xlim(0, N_VOLS - 1)
axes[0].tick_params(axis="both", which="major", labelsize=14)
axes[1].legend(loc="upper right", fontsize=20)
axes[1].set_ylabel("Signal", fontsize=24)
axes[1].set_xlabel("Echo Time (ms)", fontsize=24)
axes[1].set_xticks([0, 30, 60, 90, 120])
axes[1].set_ylim(0, np.ceil(np.max(fullcurve_signal) / 1000) * 1000)
axes[1].set_xlim(0, np.max(FULLCURVE_TES))
axes[1].tick_params(axis="both", which="major", labelsize=14)
fig.tight_layout()


def AnimationFunction(frame):
    """Function takes frame as an input."""
    t2s_value = predict_bold_signal(
        np.array([t2s_ts[frame]]), np.array([s0_ts[frame]]), np.array([t2s_ts[frame]])
    )[0]

    ax0_scatter_plot.set_offsets(
        np.column_stack(
            (
                frame,
                singleecho_signal[:, frame],
            )
        )
    )

    ax1_line_plot.set_data(
        (
            FULLCURVE_TES,
            fullcurve_signal[:, frame],
        )
    )
    ax1_scatter_plot.set_offsets(
        np.column_stack(
            (
                SINGLEECHO_TE,
                singleecho_signal[:, frame],
            )
        )
    )
    ax1_t2s_scatter_plot.set_offsets(
        np.column_stack(
            (
                t2s_ts[frame],
                t2s_value,
            )
        )
    )
    ax1_s0_scatter_plot.set_offsets(
        np.column_stack(
            (
                0,
                s0_ts[frame],
            )
        )
    )


anim_created = FuncAnimation(fig, AnimationFunction, frames=N_VOLS, interval=100)
html = display.HTML(anim_created.to_jshtml())
glue("fig_signal_decay2", html, display=False)
```

```{glue:figure} fig_signal_decay2
:name: fig_signal_decay2
:align: center

The same data, with $S_{0}$ (red star) and the point on the curve at $TE = T_{2}^{*}$ (blue star) marked in each volume.
```

### $S_{0}$ and $T_{2}^{*}$ fluctuations change the curve differently

To see the two effects separately, we now let only one of them fluctuate at a time, using the same relative fluctuations for both.
The top panel shows the relative fluctuations over time.
In the bottom panel, the red curve is the decay curve when only $S_{0}$ fluctuates (with $T_{2}^{*}$ fixed),
and the blue curve is the decay curve when only $T_{2}^{*}$ fluctuates (with $S_{0}$ fixed).
A change in $S_{0}$ scales the whole curve up or down, including its starting point at $TE = 0$.
A change in $T_{2}^{*}$ leaves the starting point alone and changes how quickly the curve falls.

```{code-cell} ipython3
:tags: [hide-cell]
s0based_fullcurve_signal = predict_bold_signal(
    FULLCURVE_TES, s0_ts, np.full(N_VOLS, MEAN_T2S)
)
t2sbased_fullcurve_signal = predict_bold_signal(
    FULLCURVE_TES, np.full(N_VOLS, MEAN_S0), t2s_ts
)
unperturbed_fullcurve_signal = np.round(np.mean(
    (s0based_fullcurve_signal[:, 0], t2sbased_fullcurve_signal[:, 0]),
    axis=0,
), decimals=5)

fig, axes = plt.subplots(
    nrows=2, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 3]}
)

t2s_value = predict_bold_signal(
    np.array([t2s_ts[0]]), np.array([s0_ts[0]]), np.array([t2s_ts[0]])
)[0]

ax0_line_plot = axes[0].plot(ts, color="black", zorder=0)
ax0_scatter_plot = axes[0].scatter(
    0,
    ts[0],
    color="purple",
    s=150,
    label="Single-Echo Signal",
    zorder=1,
)
ax1_orig_line_plot = axes[1].plot(
    FULLCURVE_TES,
    unperturbed_fullcurve_signal,
    alpha=0.5,
    color="black",
    label="Original Average Decay Curve",
    zorder=-1,
)[0]
ax1_s0_line_plot = axes[1].plot(
    FULLCURVE_TES,
    s0based_fullcurve_signal[:, 0],
    alpha=0.5,
    color="red",
    label="$S_0$-Driven Decay Curve",
)[0]
ax1_t2s_line_plot = axes[1].plot(
    FULLCURVE_TES,
    t2sbased_fullcurve_signal[:, 0],
    alpha=0.5,
    color="blue",
    label="$T_{2}^{*}$-Driven Decay Curve",
)[0]

# Set constant params for figure
axes[0].set_ylabel("Relative change\nin $S_0$ or $T_2^*$", fontsize=18)
axes[0].set_xlabel("Volume", fontsize=24)
axes[0].set_xlim(0, N_VOLS - 1)
axes[0].tick_params(axis="both", which="major", labelsize=14)
axes[1].legend(loc="upper right", fontsize=20)
axes[1].set_ylabel("Signal", fontsize=24)
axes[1].set_xlabel("Echo Time (ms)", fontsize=24)
axes[1].set_xticks([0, 30, 60, 90, 120])
axes[1].set_ylim(0, np.ceil(np.max(s0based_fullcurve_signal) / 1000) * 1000)
axes[1].set_xlim(0, np.max(FULLCURVE_TES))
axes[1].tick_params(axis="both", which="major", labelsize=14)
fig.tight_layout()


def AnimationFunction(frame):
    """Function takes frame as an input."""
    ax0_scatter_plot.set_offsets(
        np.column_stack(
            (
                frame,
                ts[frame],
            )
        )
    )

    ax1_t2s_line_plot.set_data(
        (
            FULLCURVE_TES,
            t2sbased_fullcurve_signal[:, frame],
        )
    )
    ax1_s0_line_plot.set_data(
        (
            FULLCURVE_TES,
            s0based_fullcurve_signal[:, frame],
        )
    )


anim_created = FuncAnimation(fig, AnimationFunction, frames=N_VOLS, interval=100)
html = display.HTML(anim_created.to_jshtml())
glue("fig_signal_decay3", html, display=False)
```

```{glue:figure} fig_signal_decay3
:name: fig_signal_decay3
:align: center

Decay curves when only $S_{0}$ fluctuates (red) or only $T_{2}^{*}$ fluctuates (blue), with the same relative fluctuations (top). The black curve is the average decay curve.
```

### Single-echo data cannot tell the two apart

Now let us measure both curves with a single-echo acquisition at TE = 30 ms (red and blue points).
The two points move up and down with the same timing,
even though they come from very different changes in the decay curve.
With only one echo, we see that the signal went up or down,
but we cannot tell whether $S_{0}$ or $T_{2}^{*}$ changed.

```{code-cell} ipython3
:tags: [hide-cell]
s0based_fullcurve_signal = predict_bold_signal(
    FULLCURVE_TES, s0_ts, np.full(N_VOLS, MEAN_T2S)
)
t2sbased_fullcurve_signal = predict_bold_signal(
    FULLCURVE_TES, np.full(N_VOLS, MEAN_S0), t2s_ts
)
s0based_singleecho_signal = s0based_fullcurve_signal[SINGLEECHO_TE, :]
t2sbased_singleecho_signal = t2sbased_fullcurve_signal[SINGLEECHO_TE, :]

fig, axes = plt.subplots(
    nrows=2, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 3]}
)

ax0_line_plot = axes[0].plot(ts, color="black", zorder=0)
ax0_scatter_plot = axes[0].scatter(
    0,
    ts[0],
    color="purple",
    s=150,
    zorder=1,
)
ax1_s0_line_plot = axes[1].plot(
    FULLCURVE_TES,
    s0based_fullcurve_signal[:, 0],
    color="red",
    alpha=0.5,
    label="$S_0$-Driven Decay Curve",
    zorder=0,
)[0]
ax1_t2s_line_plot = axes[1].plot(
    FULLCURVE_TES,
    t2sbased_fullcurve_signal[:, 0],
    color="blue",
    alpha=0.5,
    label="$T_{2}^{*}$-Driven Decay Curve",
    zorder=1,
)[0]
ax1_s0_scatter_plot = axes[1].scatter(
    SINGLEECHO_TE,
    s0based_singleecho_signal[:, 0],
    color="red",
    s=150,
    alpha=1.0,
    label="$S_0$-Driven Single-Echo Signal",
    zorder=2,
)
ax1_t2s_scatter_plot = axes[1].scatter(
    SINGLEECHO_TE,
    t2sbased_singleecho_signal[:, 0],
    color="blue",
    s=150,
    alpha=1.0,
    label="$T_{2}^{*}$-Driven Single-Echo Signal",
    zorder=3,
)

# Set constant params for figure
axes[0].set_ylabel("Relative change\nin $S_0$ or $T_2^*$", fontsize=18)
axes[0].set_xlabel("Volume", fontsize=24)
axes[0].set_xlim(0, N_VOLS - 1)
axes[0].tick_params(axis="both", which="major", labelsize=14)
axes[1].legend(loc="upper right", fontsize=20)
axes[1].set_ylabel("Signal", fontsize=24)
axes[1].set_xlabel("Echo Time (ms)", fontsize=24)
axes[1].set_xticks([0, 30, 60, 90, 120])
axes[1].set_ylim(0, np.ceil(np.max(s0based_fullcurve_signal) / 1000) * 1000)
axes[1].set_xlim(0, np.max(FULLCURVE_TES))
axes[1].tick_params(axis="both", which="major", labelsize=14)
fig.tight_layout()


def AnimationFunction(frame):
    """Function takes frame as an input."""
    ax0_scatter_plot.set_offsets(
        np.column_stack(
            (
                frame,
                ts[frame],
            )
        )
    )

    ax1_s0_line_plot.set_data(
        (
            FULLCURVE_TES,
            s0based_fullcurve_signal[:, frame],
        )
    )
    ax1_t2s_line_plot.set_data(
        (
            FULLCURVE_TES,
            t2sbased_fullcurve_signal[:, frame],
        )
    )
    ax1_s0_scatter_plot.set_offsets(
        np.column_stack(
            (
                SINGLEECHO_TE,
                s0based_singleecho_signal[:, frame],
            )
        )
    )
    ax1_t2s_scatter_plot.set_offsets(
        np.column_stack(
            (
                SINGLEECHO_TE,
                t2sbased_singleecho_signal[:, frame],
            )
        )
    )


anim_created = FuncAnimation(fig, AnimationFunction, frames=N_VOLS, interval=100)
html = display.HTML(anim_created.to_jshtml())
glue("fig_signal_decay4", html, display=False)
```

```{glue:figure} fig_signal_decay4
:name: fig_signal_decay4
:align: center

The $S_{0}$-driven (red) and $T_{2}^{*}$-driven (blue) decay curves, measured at a single echo time (TE = 30 ms).
```

The same point can be made with time series.
In the figure below, the measured single-echo signal is plotted against the $T_{2}^{*}$ or $S_{0}$ fluctuations that produced it,
after scaling them to the same units.
When only $S_{0}$ fluctuates, the signal is exactly a scaled copy of $S_{0}$,
because $S$ is proportional to $S_{0}$ in {eq}`monoexponential_decay_undeveloped_single_echo`.
When only $T_{2}^{*}$ fluctuates, the signal is *almost* a scaled copy of $T_{2}^{*}$;
the small differences come from the exponential, which is not linear in $T_{2}^{*}$.
Since we never observe the underlying fluctuations in real data,
the single-echo time series alone does not tell us which of the two produced it.

```{code-cell} ipython3
:tags: [hide-cell]
fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), sharey=True)
volumes = np.arange(N_VOLS)
for ax, (label, param_ts, signal_ts, color) in zip(
    axes,
    [
        ("$T_2^*$", t2s_ts, t2sbased_fullcurve_signal[SINGLEECHO_TE[0], :], "blue"),
        ("$S_0$", s0_ts, s0based_fullcurve_signal[SINGLEECHO_TE[0], :], "red"),
    ],
):
    # Scale the parameter's fluctuations to the units of the signal
    centered = param_ts - param_ts.mean()
    scalar = np.linalg.lstsq(centered[:, None], signal_ts - signal_ts.mean(), rcond=None)[0]
    ax.plot(
        volumes,
        signal_ts,
        color="black",
        linewidth=4,
        alpha=0.4,
        label=f"Signal at TE = {SINGLEECHO_TE[0]} ms",
    )
    ax.plot(
        volumes,
        np.round(centered * scalar + signal_ts.mean(), decimals=5),
        color=color,
        linestyle="--",
        label=f"{label} (scaled to the signal)",
    )
    ax.set_title(f"Only {label} fluctuates", fontsize=16)
    ax.set_xlabel("Volume", fontsize=14)
    ax.set_xlim(0, N_VOLS - 1)
    ax.set_xticks(np.arange(0, N_VOLS, 5))
    ax.legend(fontsize=12)

axes[0].set_ylabel("Signal", fontsize=14)
fig.tight_layout()
glue_figure("fig_t2s_s0_bold_single-echo", fig, display=False)
```

```{figure} figures/fig_t2s_s0_bold_single-echo.png
:name: fig_t2s_s0_bold_single-echo
:align: center

Single-echo signal (gray) when only $T_{2}^{*}$ (left) or only $S_{0}$ (right) fluctuates,
compared with those fluctuations scaled to the units of the signal (dashed).
```

### Multi-echo data can tell the two apart

Finally, let us measure the same curves at the five echo times of our real data.
Now the two kinds of fluctuation produce clearly different patterns across echoes.
When $S_{0}$ changes, the red points all move by the same *fraction* of their value,
so the change is largest at the first echo.
When $T_{2}^{*}$ changes, the blue points barely move at the first echo and move much more at later echoes.
With several echoes, we could, in principle, fit the decay curve in each volume, as we did for the real voxel above,
and tell which parameter changed.

```{code-cell} ipython3
:tags: [hide-cell]
s0based_fullcurve_signal = predict_bold_signal(
    FULLCURVE_TES, s0_ts, np.full(N_VOLS, MEAN_T2S)
)
t2sbased_fullcurve_signal = predict_bold_signal(
    FULLCURVE_TES, np.full(N_VOLS, MEAN_S0), t2s_ts
)
s0based_multiecho_signal = s0based_fullcurve_signal[data['echo_times'], :]
t2sbased_multiecho_signal = t2sbased_fullcurve_signal[data['echo_times'], :]

fig, axes = plt.subplots(
    nrows=2, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 3]}
)

ax0_line_plot = axes[0].plot(
    ts,
    color="black",
    zorder=0,
)[0]
ax0_scatter_plot = axes[0].scatter(
    0,
    ts[0],
    color="purple",
    s=150,
    zorder=1,
)
ax1_s0_line_plot = axes[1].plot(
    FULLCURVE_TES,
    s0based_fullcurve_signal[:, 0],
    color="red",
    alpha=0.5,
    label="$S_0$-Driven Decay Curve",
    zorder=0,
)[0]
ax1_t2s_line_plot = axes[1].plot(
    FULLCURVE_TES,
    t2sbased_fullcurve_signal[:, 0],
    color="blue",
    alpha=0.5,
    label="$T_{2}^{*}$-Driven Decay Curve",
    zorder=1,
)[0]
ax1_s0_scatter_plot = axes[1].scatter(
    data['echo_times'],
    s0based_multiecho_signal[:, 0],
    color="red",
    s=150,
    alpha=1.0,
    label="$S_0$-Driven Multi-Echo Signal",
    zorder=2,
)
ax1_t2s_scatter_plot = axes[1].scatter(
    data['echo_times'],
    t2sbased_multiecho_signal[:, 0],
    color="blue",
    s=150,
    alpha=1.0,
    label="$T_{2}^{*}$-Driven Multi-Echo Signal",
    zorder=3,
)

axes[0].set_ylabel("Relative change\nin $S_0$ or $T_2^*$", fontsize=18)
axes[0].set_xlabel("Volume", fontsize=24)
axes[0].set_xlim(0, N_VOLS - 1)
axes[0].tick_params(axis="both", which="major", labelsize=14)
axes[1].legend(loc="upper right", fontsize=20)
axes[1].set_ylabel("Signal", fontsize=24)
axes[1].set_xlabel("Echo Time (ms)", fontsize=24)
axes[1].set_xticks(data['echo_times'])
axes[1].set_ylim(0, np.ceil(np.max(s0based_fullcurve_signal) / 1000) * 1000)
axes[1].set_xlim(0, np.max(FULLCURVE_TES))
axes[1].tick_params(axis="both", which="major", labelsize=14)
fig.tight_layout()


def AnimationFunction(frame):
    """Function takes frame as an input."""
    ax0_scatter_plot.set_offsets(
        np.column_stack(
            (
                frame,
                ts[frame],
            )
        )
    )

    ax1_s0_line_plot.set_data(
        (
            FULLCURVE_TES,
            s0based_fullcurve_signal[:, frame],
        )
    )
    ax1_t2s_line_plot.set_data(
        (
            FULLCURVE_TES,
            t2sbased_fullcurve_signal[:, frame],
        )
    )
    ax1_s0_scatter_plot.set_offsets(
        np.column_stack(
            (
                data['echo_times'],
                s0based_multiecho_signal[:, frame],
            )
        )
    )
    ax1_t2s_scatter_plot.set_offsets(
        np.column_stack(
            (
                data['echo_times'],
                t2sbased_multiecho_signal[:, frame],
            )
        )
    )


anim_created = FuncAnimation(fig, AnimationFunction, frames=N_VOLS, interval=100)
html = display.HTML(anim_created.to_jshtml())
glue("fig_signal_decay5", html, display=False)
```

```{glue:figure} fig_signal_decay5
:name: fig_signal_decay5
:align: center

The $S_{0}$-driven (red) and $T_{2}^{*}$-driven (blue) decay curves, measured at five echo times.
```


## Summary

- MR signal decays with echo time following, approximately, $S(TE) = S_{0} e^{-TE / T_{2}^{*}}$.
  With several echoes, we can estimate $S_{0}$ and $T_{2}^{*}$ by fitting a straight line to the log of the signal.
- The BOLD effect changes $T_{2}^{*}$, while many sources of noise change $S_{0}$.
- A single-echo scan measures one point on the decay curve in each volume,
  so it cannot tell a change in $S_{0}$ from a change in $T_{2}^{*}$.
- With multiple echoes, the two kinds of change leave different fingerprints:
  an $S_{0}$ change is the same fraction of the signal at every echo time,
  while a $T_{2}^{*}$ change grows, as a fraction of the signal, with echo time.

In practice, the signal in a single volume is too noisy to fit the decay curve reliably.
[](TE_Dependence.md) shows how `tedana` uses these fingerprints on components of the data instead,
to separate BOLD from non-BOLD signal.
Before that, [](T2star_S0_Variability.md) looks at why the *average* $T_{2}^{*}$ and $S_{0}$ vary so much across the brain.
