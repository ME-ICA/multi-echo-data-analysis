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

# BOLD, non-BOLD, and TE-dependence with tedana

```{important}
This chapter should differentiate itself from Signal_Decay by focusing on the application of {eq}`monoexponential_decay`
to decompositions, rather than raw signal compared between active and inactive states.
```

```{important}
The general flow of this chapter should be:

1.  Explain the monoexponential decay equation, but primarily reference back to Signal_Decay.
2.  Walk through optimal combination and adaptive masking somewhere around here.
3.  Describe why multi-echo denoising can't be done directly to the raw signal and why ICA is necessary.
    1.  This means talking about noise, really, and why the FIT method (volume-wise T2*/S0 estimation)
        is generally considered too noisy for practical application.
    2.  Walk through TEDPCA as well.
4.  The TE-(in)dependence models.
5.  Apply the models to a simulated component, as well as multiple real components.
    1.  Show model fit for different components.
6.  Compare optimally combined, denoised, and high-kappa data.
7.  Describe post-processing methods, like minimum image regression and tedana's version of global signal regression.
```

In [](Signal_Decay.md), we saw that changes in $S_{0}$ and changes in $T_{2}^{*}$ affect multi-echo data in different ways,
and that this is what lets multi-echo data tell them apart.
This chapter turns that observation into a test.
We first work out how each kind of change depends on echo time,
then use simulations to build two models that `tedana` fits to its components,
and finally follow the same ideas through `tedana`'s processing of a real multi-echo run.

```{code-cell} ipython3
:tags: [hide-cell]
import json
import os
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nb
import nitransforms as nit
import numpy as np
import pandas as pd
import seaborn as sns
from book_utils import (
    compute_te_dependence_statistics,
    glue_figure,
    load_pafin,
)
from myst_nb import glue
from nilearn import image, masking, plotting
from nilearn.glm import first_level
from scipy import signal, stats

sns.set_style("whitegrid")

data_path = os.path.abspath("../data")

out_dir = os.path.join(data_path, "te-dependence")
os.makedirs(out_dir, exist_ok=True)
```

## How signal changes depend on echo time

Recall the monoexponential decay model from [](Signal_Decay.md), now written for volume $t$:

```{math}
:label: monoexponential_decay_volume
S(t, TE) = S_{0}(t) \, e^{-TE \cdot R_{2}^{*}(t)}
```

Let $\bar{S}_{0}$ and $\bar{R}_{2}^{*}$ be the average values of $S_{0}$ and $R_{2}^{*}$ over the run,
and let ${\Delta}S_{0}(t)$ and ${\Delta}R_{2}^{*}(t)$ be how far they are from those averages in volume $t$.
The average signal at each echo time is then $\bar{S}(TE) = \bar{S}_{0} e^{-TE \cdot \bar{R}_{2}^{*}}$.

Real fluctuations in $S_{0}$ and $R_{2}^{*}$ are small, typically a few percent or less.
When they are small, a first-order Taylor expansion of {eq}`monoexponential_decay_volume` gives a simple, linear approximation:

```{math}
:label: monoexponential_decay
S(t, TE) \approx \bar{S}(TE) \left(1 + \frac{{\Delta}S_{0}(t)}{\bar{S}_{0}} - {\Delta}R_{2}^{*}(t) \cdot TE\right)
```

Subtracting $\bar{S}(TE)$ and dividing by it gives the fractional signal change at each echo time:

```{math}
:label: fractional_signal_change
\frac{{\Delta}S(t, TE)}{\bar{S}(TE)} \approx \frac{{\Delta}S_{0}(t)}{\bar{S}_{0}} - {\Delta}R_{2}^{*}(t) \cdot TE
```

This equation is the basis of multi-echo denoising.
A change in $S_{0}$ produces the *same* fractional change at every echo time.
A change in $R_{2}^{*}$, which is how the BOLD effect shows up, produces a fractional change that is zero at $TE = 0$
and grows in proportion to the echo time.

The figure below shows both kinds of change for a voxel with $\bar{S}_{0} = 16000$ and $\bar{T}_{2}^{*} = 45$ ms,
a typical value for gray matter at 3T.
In the left column, $S_{0}$ increases by 5%.
In the right column, $T_{2}^{*}$ increases by 5%, as it would when local blood oxygenation increases.

```{code-cell} ipython3
:tags: [hide-cell]
def decay(tes, s0, t2s):
    """Monoexponential decay, with echo times and T2* in milliseconds."""
    return s0 * np.exp(-tes / t2s)


# Echo times for smooth curves
dense_tes = np.arange(0, 120.5, 0.5)
# The five echo times of the dataset used in this book, rounded to 0.1 ms
sim_tes = np.array([14.2, 38.9, 63.7, 88.4, 113.1])

mean_s0 = 16000
mean_t2s = 45
frac = 0.05

mean_sig = decay(dense_tes, mean_s0, mean_t2s)
s0_sig = decay(dense_tes, mean_s0 * (1 + frac), mean_t2s)
t2s_sig = decay(dense_tes, mean_s0, mean_t2s * (1 + frac))
delta_r2s = 1 / (mean_t2s * (1 + frac)) - 1 / mean_t2s

cases = [
    {
        "title": "$S_0$ increases by 5%",
        "signal": s0_sig,
        "approx": np.full(dense_tes.shape, frac),
        "color": "red",
    },
    {
        "title": "$T_2^*$ increases by 5%",
        "signal": t2s_sig,
        "approx": -delta_r2s * dense_tes,
        "color": "blue",
    },
]

fig, axes = plt.subplots(3, 2, sharex=True, figsize=(14, 10))
for col, case in enumerate(cases):
    delta = case["signal"] - mean_sig

    # Row 1: the signal itself
    axes[0, col].plot(dense_tes, mean_sig, color="black", label="Average signal, $\\bar{S}(TE)$")
    axes[0, col].plot(
        dense_tes,
        case["signal"],
        color=case["color"],
        linestyle="--",
        label="Signal after the change",
    )
    axes[0, col].set_title(case["title"], fontsize=18)
    axes[0, col].set_ylim(0, None)
    axes[0, col].legend(fontsize=11)

    # Row 2: the absolute change
    axes[1, col].plot(dense_tes, delta, color=case["color"])
    axes[1, col].set_ylim(0, None)

    # Row 3: the fractional change
    axes[2, col].plot(
        dense_tes,
        100 * delta / mean_sig,
        color=case["color"],
        linewidth=5,
        alpha=0.4,
        label="Exact",
    )
    axes[2, col].plot(
        dense_tes,
        100 * case["approx"],
        color="black",
        linestyle="--",
        label="First-order approximation",
    )
    axes[2, col].set_ylim(0, 15)
    axes[2, col].set_xlabel("Echo Time (ms)", fontsize=14)
    axes[2, col].legend(fontsize=11, loc="upper left")

# Mark T2* in the absolute change panel for the T2* change
axes[1, 1].axvline(mean_t2s, color="gray", linestyle=":")
axes[1, 1].annotate(
    "TE = $T_2^*$",
    xy=(mean_t2s, 0.1),
    xycoords=("data", "axes fraction"),
    xytext=(5, 0),
    textcoords="offset points",
    fontsize=12,
)

axes[0, 0].set_ylabel("Signal", fontsize=14)
axes[1, 0].set_ylabel("Signal change,\n${\\Delta}S(TE)$", fontsize=14)
axes[2, 0].set_ylabel("Percent signal change,\n${\\Delta}S(TE) / \\bar{S}(TE)$", fontsize=14)
axes[2, 0].set_xlim(0, dense_tes.max())
fig.tight_layout()
glue_figure("fig_bold_nonbold_simulations", fig, display=False)
```

```{figure} figures/fig_bold_nonbold_simulations.png
:name: fig_bold_nonbold_simulations
:align: center

How a 5% increase in $S_{0}$ (left) or $T_{2}^{*}$ (right) changes the signal at each echo time.
Top: the average signal and the signal after the change.
Middle: the absolute signal change.
Bottom: the percent signal change, computed exactly (thick line) and with the first-order approximation in {eq}`fractional_signal_change` (dashed line).
```

There are a few things to notice in this figure.

- **Top row:** realistic changes are small, so the curves before and after the change nearly overlap.
- **Middle row:** the absolute changes differ in shape.
  The $S_{0}$ change is largest at $TE = 0$ and shrinks as the signal decays.
  The $T_{2}^{*}$ change is zero at $TE = 0$, peaks near $TE = T_{2}^{*}$, and then shrinks.
  This peak is why single-echo fMRI uses an echo time close to the typical $T_{2}^{*}$ of gray matter,
  and why the optimal combination weights echoes by $TE \cdot e^{-TE / T_{2}^{*}}$
  (see [](01_Optimal_Combination_with_t2smap.md)).
- **Bottom row:** the percent signal change makes the difference obvious.
  The $S_{0}$ change is 5% at every echo time, while the $T_{2}^{*}$ change grows in proportion to the echo time,
  following a straight line through the origin.
  The first-order approximation (dashed) is very close to the exact change, because the change is small.

## The TE-dependence and TE-independence models

In real data, we never observe ${\Delta}S_{0}$ and ${\Delta}R_{2}^{*}$ directly.
What we *can* measure is how large a given fluctuation is at each of our echo times, ${\Delta}S(TE_k)$,
where $TE_k$ is the $k$-th of our $n$ echo times.
Multiplying {eq}`fractional_signal_change` by $\bar{S}(TE_k)$ gives two candidate explanations for those measurements.

If the fluctuation comes from a change in $S_{0}$, then

```{math}
:label: te_independence_model
{\Delta}S(TE_k) = \bar{S}(TE_k) \cdot X_{S_0}, \qquad X_{S_0} = \frac{{\Delta}S_{0}}{\bar{S}_{0}}
```

If the fluctuation comes from a change in $R_{2}^{*}$, then

```{math}
:label: te_dependence_model
{\Delta}S(TE_k) = \bar{S}(TE_k) \cdot TE_k \cdot X_{R_2^*}, \qquad X_{R_2^*} = -{\Delta}R_{2}^{*}
```

The first is called the *TE-independence* (or $S_{0}$) model, because the fractional change does not depend on echo time,
and the second is called the *TE-dependence* (or $R_{2}^{*}$) model.
Each model is a regression with a single regressor and no intercept.
The regressor is known: it is built from the average signal at each echo, $\bar{S}(TE_k)$, and the echo times themselves.
The coefficient, $X_{S_0}$ or $X_{R_2^*}$, is unknown, and is estimated with least squares.

To measure how well each model fits, `tedana` computes an F-statistic,

```{math}
:label: te_model_fstat
F = \frac{\sum_k {\Delta}S(TE_k)^2 - SSE}{SSE / (n - 1)}
```

where $SSE$ is the sum of squared residuals of the model fit.
The better the model explains the measurements, the smaller $SSE$ is and the larger $F$ is.

Below, we fit both models to the two simulated changes from {numref}`fig_bold_nonbold_simulations`,
measured at the five echo times of our real data,
with a small amount of random noise added to each measurement.

```{code-cell} ipython3
:tags: [hide-cell]
def fit_te_models(delta_s, avg_signal, tes):
    """Fit the TE-independence (S0) and TE-dependence (R2*) models.

    Parameters
    ----------
    delta_s : numpy.ndarray of shape (n_echoes,)
        Signal change at each echo time.
    avg_signal : numpy.ndarray of shape (n_echoes,)
        Average signal at each echo time.
    tes : numpy.ndarray of shape (n_echoes,)
        Echo times, in milliseconds.

    Returns
    -------
    dict
        The coefficient and F-statistic of each model.
    """
    n = len(tes)
    total = np.sum(delta_s ** 2)
    results = {}
    for name, regressor in [("S0", avg_signal), ("R2", avg_signal * tes)]:
        coef = np.sum(delta_s * regressor) / np.sum(regressor ** 2)
        sse = np.sum((delta_s - coef * regressor) ** 2)
        results[name] = {"coef": coef, "F": (total - sse) * (n - 1) / sse}

    return results


rng = np.random.default_rng(42)
noise_sd = 20

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, case, name in zip(axes, cases, ["s0change", "t2schange"]):
    true_delta = case["signal"] - mean_sig
    sim_avg = decay(sim_tes, mean_s0, mean_t2s)
    measured = np.interp(sim_tes, dense_tes, true_delta) + rng.normal(0, noise_sd, sim_tes.size)
    fits = fit_te_models(measured, sim_avg, sim_tes)
    glue(f"F_S0_{name}", float(fits["S0"]["F"]), display=False)
    glue(f"F_R2_{name}", float(fits["R2"]["F"]), display=False)

    ax.plot(dense_tes, true_delta, color="gray", alpha=0.5, linewidth=5, label="True change")
    ax.scatter(sim_tes, measured, color="black", s=60, zorder=3, label="Measured change")
    ax.plot(
        dense_tes,
        fits["S0"]["coef"] * mean_sig,
        color="red",
        linestyle="--",
        label=f"$S_0$ model ($F$ = {fits['S0']['F']:.0f})",
    )
    ax.plot(
        dense_tes,
        fits["R2"]["coef"] * mean_sig * dense_tes,
        color="blue",
        linestyle="--",
        label=f"$R_2^*$ model ($F$ = {fits['R2']['F']:.0f})",
    )
    ax.set_title(case["title"], fontsize=16)
    ax.set_xlabel("Echo Time (ms)", fontsize=14)
    ax.set_xlim(0, dense_tes.max())
    ax.legend(fontsize=11)

axes[0].set_ylabel("Signal change, ${\\Delta}S(TE)$", fontsize=14)
fig.tight_layout()
glue_figure("fig_te_model_fits", fig, display=False)
```

```{figure} figures/fig_te_model_fits.png
:name: fig_te_model_fits
:align: center

The TE-independence ($S_{0}$, red) and TE-dependence ($R_{2}^{*}$, blue) models fit to a simulated change in $S_{0}$ (left)
and in $T_{2}^{*}$ (right), measured at five echo times with added noise.
The legend gives each model's F-statistic.
```

Each model fits the change it was built for and fails on the other one.
For the $S_{0}$ change, the $S_{0}$ model has $F$ ≈ {glue:text}`F_S0_s0change:.0f`
and the $R_{2}^{*}$ model has $F$ ≈ {glue:text}`F_R2_s0change:.0f`.
For the $T_{2}^{*}$ change, the $R_{2}^{*}$ model has $F$ ≈ {glue:text}`F_R2_t2schange:.0f`
and the $S_{0}$ model has $F$ ≈ {glue:text}`F_S0_t2schange:.0f`.

## Applying the models to components

In real data, we cannot apply the models to each volume separately,
because each volume contains many fluctuations at once, along with thermal noise.
Instead, `tedana` applies them to components.
Each component has a time series, and regressing that time series on the data from each echo
gives a parameter estimate, $PE(TE_k)$, in each voxel.
The parameter estimate is how much the signal at $TE_k$ changes for each unit of the component,
so it plays the role of ${\Delta}S(TE_k)$ in {eq}`te_independence_model` and {eq}`te_dependence_model`:

```{math}
:label: te_independence_model4
PE(TE_k) = \bar{S}(TE_k) \cdot X_{S_0}
```

```{math}
:label: te_dependence_model4
PE(TE_k) = \bar{S}(TE_k) \cdot TE_k \cdot X_{R_2^*}
```

Both models are fit in every voxel, which gives a map of F-statistics for each model.
`tedana` summarizes each map with a weighted average across voxels,
giving more weight to the voxels where the component is strongest.
The summary of the $R_{2}^{*}$ model is called $\kappa$ (kappa), and the summary of the $S_{0}$ model is called $\rho$ (rho).
Components with a high $\kappa$ and a low $\rho$ look like BOLD signal,
and components with a high $\rho$ and a low $\kappa$ look like non-BOLD noise.

To see how this works, we simulate 300 volumes of a single voxel at our five echo times.
The $T_{2}^{*}$ time series is a series of blocks of activity, convolved with a hemodynamic response function,
so it looks like a BOLD response to a task.
The $S_{0}$ time series is a slow drift with a few abrupt jumps, like the effects of head motion.
Both fluctuate by about 5%, and we add a small amount of thermal noise to the signal at each echo.

```{code-cell} ipython3
:tags: [hide-cell]
hrf = first_level.spm_hrf(1, oversampling=1)
n_trs = 300
rng = np.random.default_rng(42)

# T2*: blocks of activity with random amplitudes, convolved with the HRF
n_chunks = 10
scales = rng.uniform(size=n_chunks) * 3
t2s_ts = np.hstack(
    [np.hstack((np.zeros(10), np.ones(20), np.zeros(10))) * scale for scale in scales],
)
t2s_ts = signal.convolve(t2s_ts, hrf)[:n_trs]
t2s_ts *= mean_t2s * frac / np.std(t2s_ts)
t2s_ts += mean_t2s - np.mean(t2s_ts)

# S0: a slow drift, plus a few abrupt jumps like those caused by head motion
s0_ts = signal.detrend(np.cumsum(rng.normal(size=n_trs)))
jump_vols = rng.choice(n_trs, size=4, replace=False)
s0_ts[jump_vols] += rng.choice([-1, 1], size=4) * 4 * np.std(s0_ts)
s0_ts *= mean_s0 * frac / np.std(s0_ts)
s0_ts += mean_s0 - np.mean(s0_ts)

# Signal at each echo time, with thermal noise
multiecho_signal = decay(sim_tes[:, None], s0_ts[None, :], t2s_ts[None, :])
multiecho_signal += rng.normal(0, noise_sd, multiecho_signal.shape)

t2s_ts_z = stats.zscore(t2s_ts)
s0_ts_z = stats.zscore(s0_ts)

fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(t2s_ts_z, label="$T_2^*$ fluctuations", color="blue")
ax.plot(s0_ts_z, label="$S_0$ fluctuations", color="red")
ax.set_xlim(0, n_trs - 1)
ax.set_xlabel("Volume", fontsize=14)
ax.set_ylabel("Fluctuation (z)", fontsize=14)
ax.legend(fontsize=12, ncol=2, loc="upper right")
fig.tight_layout()
glue_figure("fig_component_curves", fig, display=False)
```

```{figure} figures/fig_component_curves.png
:name: fig_component_curves
:align: center

Simulated $T_{2}^{*}$ (blue) and $S_{0}$ (red) fluctuations, each converted to z-scores.
```

```{code-cell} ipython3
:tags: [hide-cell]
# Drop the last cubehelix color, which is too light to see on a white background
sim_pal = sns.color_palette("cubehelix", sim_tes.size + 1)[: sim_tes.size]
fig, axes = plt.subplots(sim_tes.size, sharex=True, figsize=(16, 6))
for i_echo, ax in enumerate(axes):
    ax.plot(multiecho_signal[i_echo, :], color=sim_pal[i_echo])
    ax.set_ylabel(f"{sim_tes[i_echo]:.1f} ms", rotation=0, va="center", ha="right", fontsize=12)
    ax.set_yticks([])
    ax.grid(False)

axes[-1].set_xlabel("Volume", fontsize=14)
axes[-1].set_xlim(0, n_trs - 1)
fig.tight_layout()
glue_figure("fig_component_curves_2", fig, display=False)
```

```{figure} figures/fig_component_curves_2.png
:name: fig_component_curves_2
:align: center

The simulated signal at each echo time.
Each time series is scaled separately.
The slow $S_{0}$ drift and abrupt $S_{0}$ jumps appear at every echo time,
while the block-shaped $T_{2}^{*}$ responses become more prominent at later echo times,
because the fractional $T_{2}^{*}$ effect grows with echo time.
```

Now we build three components from these time series:
one that follows the $T_{2}^{*}$ fluctuations only, one that follows the $S_{0}$ fluctuations only,
and one that is an equal mixture of the two.
For each component, we regress its time series on the signal at each echo to get $PE(TE_k)$,
and then fit the two models.

```{code-cell} ipython3
:tags: [hide-cell]
components = [
    ("Pure $T_2^*$ component", t2s_ts_z),
    ("Equal mixture", stats.zscore(t2s_ts_z + s0_ts_z)),
    ("Pure $S_0$ component", s0_ts_z),
]
comp_names = ["t2s", "mix", "s0"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, (title, comp), name in zip(axes, components, comp_names):
    # Regress the component (plus an intercept) on the signal at each echo
    comp_X = np.column_stack((comp, np.ones(n_trs)))
    pes = np.linalg.lstsq(comp_X, multiecho_signal.T, rcond=None)[0][0, :]

    F_S0, F_R2, pred_S0, pred_R2 = compute_te_dependence_statistics(
        multiecho_signal,
        pes,
        sim_tes,
    )
    glue(f"kappa_{name}", float(F_R2[0]), display=False)
    glue(f"rho_{name}", float(F_S0[0]), display=False)

    ax.plot(
        sim_tes,
        pes,
        color="black",
        alpha=0.4,
        linewidth=6,
        marker="o",
        markersize=10,
        label="Parameter estimates",
    )
    ax.plot(
        sim_tes,
        pred_R2,
        color="blue",
        marker="o",
        label=f"$R_2^*$ model ($\\kappa$ = {F_R2[0]:.0f})",
    )
    ax.plot(
        sim_tes,
        pred_S0,
        color="red",
        marker="o",
        label=f"$S_0$ model ($\\rho$ = {F_S0[0]:.0f})",
    )
    ax.set_title(title, fontsize=16)
    ax.set_xticks(sim_tes)
    ax.set_xlabel("Echo Time (ms)", fontsize=14)
    ax.legend(fontsize=11)

axes[0].set_ylabel("Parameter estimate", fontsize=14)
fig.tight_layout()
glue_figure("fig_component_model_fits", fig, display=False)
```

```{figure} figures/fig_component_model_fits.png
:name: fig_component_model_fits
:align: center

Parameter estimates (gray) and the fits of the $R_{2}^{*}$ (blue) and $S_{0}$ (red) models
for three simulated components.
```

For the pure $T_{2}^{*}$ component, the parameter estimates follow the $R_{2}^{*}$ model closely,
and $\kappa$ ({glue:text}`kappa_t2s:.0f`) is much larger than $\rho$ ({glue:text}`rho_t2s:.0f`).
For the pure $S_{0}$ component, the opposite is true:
$\rho$ is {glue:text}`rho_s0:.0f` and $\kappa$ is {glue:text}`kappa_s0:.0f`.
For the equal mixture, neither model fits well, and $\kappa$ ({glue:text}`kappa_mix:.0f`)
and $\rho$ ({glue:text}`rho_mix:.0f`) are much closer together.
Real components often fall somewhere between these extremes,
which is why `tedana` classifies components with a decision tree that uses several metrics, not only $\kappa$ and $\rho$.

The values of $\kappa$ and $\rho$ depend on the number of echoes, the noise level, and the number of voxels,
so they are only meaningful relative to other components from the same data,
not as absolute thresholds.

## TE-dependence in real data with `tedana`

So far, we have worked with simulated signals.
In this section, we follow the same ideas through `tedana`'s processing of the five-echo resting-state run used throughout this book,
from the raw echoes to the denoised data.
The `tedana` outputs come from [](03_Denoising_with_tedana.md).
To keep the figures readable, we focus on a single voxel:
the voxel with the highest weight on the most BOLD-like accepted component.

```{code-cell} ipython3
:tags: [hide-cell]
data = load_pafin(data_path)
prefix = "sub-24053_ses-1_task-rat_dir-PA_run-01"
ted_dir = os.path.join(data_path, "tedana")
func_dir = os.path.join(data_path, "ds006185/sub-24053/ses-1/func")
anat_dir = os.path.join(data_path, "ds006185/sub-24053/ses-1/anat")

# The echo times are stored in seconds. The figures below use milliseconds.
tes_ms = np.array(data["echo_times"]) * 1000
n_real_echoes = len(tes_ms)
# Drop the last cubehelix color, which is too light to see on a white background
echo_pal = sns.color_palette("cubehelix", n_real_echoes + 1)[:n_real_echoes]

# Background anatomical image, resampled onto the grid of the functional data.
# The transform moves images from the BOLD reference to the T1w image,
# so we apply its inverse to bring the T1w image into the space of the BOLD data.
xfm = nit.linear.load(
    os.path.join(func_dir, f"{prefix}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"),
    fmt="itk",
)
t1_file = os.path.join(anat_dir, f"{prefix.split('_task')[0]}_desc-preproc_T1w.nii.gz")
bg_img = nit.resampling.apply(~xfm, t1_file, reference=data["mask"])

# tedana's adaptive mask. tedana computes the TE-dependence metrics in voxels
# with at least three good echoes, so we use the same voxels here.
adaptive_mask_file = os.path.join(ted_dir, f"{prefix}_desc-adaptiveGoodSignal_mask.nii.gz")
mask = image.math_img("img >= 3", img=adaptive_mask_file)


def load_ted(desc):
    """Load a tedana output as a (time or component, voxel) array."""
    return masking.apply_mask(os.path.join(ted_dir, f"{prefix}_{desc}.nii.gz"), mask)


def load_ted_echoes(desc):
    """Load an echo-wise tedana output as a (voxel, echo, component) array."""
    files = [
        os.path.join(ted_dir, f"{prefix}_echo-{i_echo + 1}_{desc}.nii.gz")
        for i_echo in range(n_real_echoes)
    ]
    return np.stack([masking.apply_mask(f, mask).T for f in files], axis=1)


def zscore_time(arr):
    """Z-score each voxel's time series."""
    return (arr - arr.mean(axis=0)) / arr.std(axis=0)


# Echo-wise data, T2* map, and optimally combined data
echo_data = [masking.apply_mask(f, mask) for f in data["echo_files"]]
t2s_map = masking.apply_mask(os.path.join(ted_dir, f"{prefix}_T2starmap.nii.gz"), mask)
oc = load_ted("desc-optcom_bold")

# PCA and ICA results
pca_mixing = pd.read_table(os.path.join(ted_dir, f"{prefix}_desc-PCA_mixing.tsv")).values
ica_mixing = pd.read_table(os.path.join(ted_dir, f"{prefix}_desc-ICA_mixing.tsv")).values
ica_maps = load_ted("desc-ICA_components")
ica_pes = load_ted_echoes("desc-ICA_components")
r2_preds = load_ted_echoes("desc-ICAT2ModelPredictions_components")
s0_preds = load_ted_echoes("desc-ICAS0ModelPredictions_components")
comp_tbl = pd.read_table(
    os.path.join(ted_dir, f"{prefix}_desc-tedana_metrics.tsv"),
    index_col="Component",
)

# Denoised data, with and without minimum image regression
medn = load_ted("desc-denoised_bold")
mehk = load_ted("desc-optcomAccepted_bold")
medn_mir = load_ted("desc-optcomMIRDenoised_bold")
mehk_mir = load_ted("desc-optcomAcceptedMIRDenoised_bold")

# Example components: the accepted component with the highest kappa
# and the rejected component with the highest rho
accepted = comp_tbl.loc[comp_tbl["classification"] == "accepted"]
rejected = comp_tbl.loc[comp_tbl["classification"] == "rejected"]
bold_comp = accepted["kappa"].idxmax()
nonbold_comp = rejected["rho"].idxmax()
example_comps = [bold_comp, nonbold_comp]


def comp_index(comp):
    """Convert a component name (e.g., ICA_07) to its index."""
    return int(comp.split("_")[1])


# The voxel with the highest weight on the BOLD-like component
voxel_idx = np.argmax(ica_maps[comp_index(bold_comp), :])
voxel_ts = np.vstack([d[:, voxel_idx] for d in echo_data])
n_vols = voxel_ts.shape[1]
```

### The adaptive mask

Longer echo times are more susceptible to signal dropout,
which means that certain brain regions (e.g., orbitofrontal cortex, temporal poles) only have good signal for some echoes.
To avoid using bad signal from the affected echoes when it estimates $T_{2}^{*}$ and $S_{0}$,
`tedana` generates an adaptive mask, where the value for each voxel is the number of echoes with "good" signal.
Each voxel's $T_{2}^{*}$ and $S_{0}$ are then estimated from only its first $n$ echoes,
where $n$ is the value for that voxel in the adaptive mask,
using the log-linear fit described in [](Signal_Decay.md).

```{code-cell} ipython3
:tags: [hide-cell]
# One color per number of good echoes
mask_cmap = matplotlib.colors.ListedColormap(
    sns.color_palette("viridis", n_real_echoes),
)
mask_norm = matplotlib.colors.BoundaryNorm(
    np.arange(0.5, n_real_echoes + 1),
    mask_cmap.N,
)

fig = plt.figure(figsize=(16, 3.5), facecolor="black")
display = plotting.plot_stat_map(
    adaptive_mask_file,
    bg_img=bg_img,
    display_mode="z",
    cut_coords=[-30, -20, -10, 0, 10, 20, 30],
    threshold=0.5,
    vmin=0.5,
    vmax=n_real_echoes + 0.5,
    cmap=mask_cmap,
    colorbar=False,
    draw_cross=False,
    annotate=False,
    figure=fig,
    axes=fig.add_axes([0, 0, 0.93, 1]),
)
display.title("Number of echoes with good signal", size=16, bgcolor="white", color="black")
cax = fig.add_axes([0.945, 0.15, 0.01, 0.7])
cbar = fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=mask_norm, cmap=mask_cmap),
    cax=cax,
    ticks=np.arange(1, n_real_echoes + 1),
)
cbar.ax.tick_params(labelsize=12, colors="white")
glue_figure("fig_adaptive_mask", fig, display=False)
```

```{figure} figures/fig_adaptive_mask.png
:name: fig_adaptive_mask
:align: center

`tedana`'s adaptive mask, which counts the echoes with good signal in each voxel.
Voxels in ventral frontal and temporal cortex, near the sinuses and ear canals,
have good signal in fewer echoes than the rest of the brain.
```

### Optimal combination

Next, `tedana` combines the echoes into a single time series in each voxel,
using a weighted average in which each echo's weight is proportional to its expected BOLD sensitivity,
$TE_{n} e^{-TE_{n} / T_{2}^{*}}$.
The weights depend on the voxel's own $T_{2}^{*}$,
so echoes near the voxel's $T_{2}^{*}$ get the most weight.
[](01_Optimal_Combination_with_t2smap.md) describes the optimal combination in more detail.

```{code-cell} ipython3
:tags: [hide-cell]
# The T2* map is in seconds
voxel_t2s = t2s_map[voxel_idx] * 1000
weights = tes_ms * np.exp(-tes_ms / voxel_t2s)
weights /= weights.sum()
voxel_oc = np.average(voxel_ts, axis=0, weights=weights)

fig = plt.figure(figsize=(16, 6))
subfigs = fig.subfigures(1, 2, width_ratios=[1, 2.5])

# Left: the weights
ax = subfigs[0].subplots()
te_labels = [f"{te:.1f}" for te in tes_ms]
ax.bar(te_labels, weights, color=echo_pal)
ax.set_xlabel("Echo Time (ms)", fontsize=14)
ax.set_ylabel("Weight", fontsize=14)
ax.set_title(f"Weights for $T_2^*$ = {voxel_t2s:.1f} ms", fontsize=16)

# Right: the echo-wise and optimally combined time series
axes = subfigs[1].subplots(n_real_echoes + 1, sharex=True)
for i_echo in range(n_real_echoes):
    axes[i_echo].plot(voxel_ts[i_echo], color=echo_pal[i_echo])
    axes[i_echo].set_ylabel(
        f"{tes_ms[i_echo]:.1f} ms",
        rotation=0,
        va="center",
        ha="right",
        fontsize=12,
    )

axes[-1].plot(voxel_oc, color="red")
axes[-1].set_ylabel("Optimally\ncombined", rotation=0, va="center", ha="right", fontsize=12)
axes[-1].set_xlabel("Volume", fontsize=14)
axes[-1].set_xlim(0, n_vols - 1)
for ax in axes:
    ax.set_yticks([])
    ax.grid(False)

glue_figure("fig_optcom_voxel", fig, display=False)
```

```{figure} figures/fig_optcom_voxel.png
:name: fig_optcom_voxel
:align: center

Optimal combination in a single voxel.
Left: the weight of each echo, which peaks at the echo time closest to the voxel's $T_{2}^{*}$.
Right: the time series of each echo and the optimally combined time series (red).
Each time series is scaled separately, so the y-axes are not comparable.
```

### Why use a decomposition?

The optimally combined data still contain both BOLD and non-BOLD fluctuations.
In principle, we could separate them by fitting $S_{0}$ and $T_{2}^{*}$ to every volume
(see [](02_Volume-wise_T2star_estimation_with_t2smap.md)),
but with only a handful of echoes per volume, thermal noise makes the volume-wise estimates very noisy.
Instead, `tedana` decomposes the optimally combined data into components,
each with a time series and a spatial map.
The TE-dependence of a component is then estimated from all of the volumes at once,
and evaluated across all of the voxels where the component is present,
which makes it far more reliable than the TE-dependence of any single volume.

#### Principal component analysis

First, the optimally combined data are decomposed with principal component analysis (PCA),
and components that look like noise are discarded.
`tedana` offers several ways to choose the components to keep,
including model-selection criteria such as the Akaike information criterion,
a decision tree based on the TE-dependence metrics described below,
and a threshold on the variance explained.

```{code-cell} ipython3
:tags: [hide-cell]
fig, axes = plt.subplots(3, sharex=True, figsize=(16, 6))
for i_comp, ax in enumerate(axes):
    ax.plot(pca_mixing[:, i_comp], color="black")
    ax.set_title(f"PCA component {i_comp}", fontsize=14, loc="left")
    ax.set_yticks([])

axes[-1].set_xlim(0, pca_mixing.shape[0] - 1)
axes[-1].set_xlabel("Volume", fontsize=14)
fig.tight_layout()
glue_figure("fig_pca_timeseries", fig, display=False)
```

```{figure} figures/fig_pca_timeseries.png
:name: fig_pca_timeseries
:align: center

Time series of the first three PCA components.
```

The retained components are then recombined into a dimensionally reduced version of the optimally combined data,
which keeps most of the structured signal but less of the random noise.

#### Independent component analysis

The dimensionally reduced data are then decomposed with independent component analysis (ICA).
The number of ICA components is set to the number of components retained by the PCA,
in order to reflect the true dimensionality of the data.
ICA produces a mixing matrix, which contains a time series for each component.
Below are the time series of two components that `tedana` classified very differently:
the accepted component with the highest $\kappa$ and the rejected component with the highest $\rho$.

```{code-cell} ipython3
:tags: [hide-cell]
fig, axes = plt.subplots(2, sharex=True, figsize=(16, 5))
for ax, comp in zip(axes, example_comps):
    row = comp_tbl.loc[comp]
    ax.plot(ica_mixing[:, comp_index(comp)], color="black")
    ax.set_title(
        f"{comp}: $\\kappa$ = {row['kappa']:.1f}, $\\rho$ = {row['rho']:.1f} ({row['classification']})",
        fontsize=14,
        loc="left",
    )
    ax.set_yticks([])

axes[-1].set_xlim(0, ica_mixing.shape[0] - 1)
axes[-1].set_xlabel("Volume", fontsize=14)
fig.tight_layout()
glue_figure("fig_ica_timeseries", fig, display=False)
```

```{figure} figures/fig_ica_timeseries.png
:name: fig_ica_timeseries
:align: center

Time series of a BOLD-like (accepted) component and a non-BOLD-like (rejected) ICA component.
```

The accepted component fluctuates slowly throughout the run, as we would expect from BOLD signal.
Most of the rejected component's variance is in the first few volumes.
This pattern is typical of the start of a run, before the magnetization has reached a steady state
and the signal is brighter at every echo.
That kind of change affects $S_{0}$ rather than $T_{2}^{*}$, which fits this component's high $\rho$.

### Fitting the TE-(in)dependence models to components

As in the simulation above, `tedana` regresses each component's time series on the data from each echo
and fits {eq}`te_independence_model4` and {eq}`te_dependence_model4` to the resulting parameter estimates in every voxel.
The figure below shows the parameter estimates and model predictions in a single voxel for the two example components,
in the voxel where each component has its highest weight.
For the accepted component, the parameter estimates grow with echo time and closely follow the $R_{2}^{*}$ model.
For the rejected component, they do not grow with echo time,
so the $R_{2}^{*}$ model fits them poorly and the $S_{0}$ model fits them better.
Real components are rarely as clean as the pure simulated components above:
even the accepted component is not fit perfectly by either model,
which is why $\kappa$ and $\rho$ are summarized across many voxels.

```{code-cell} ipython3
:tags: [hide-cell]
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, comp in zip(axes, example_comps):
    idx = comp_index(comp)
    comp_voxel = np.argmax(ica_maps[idx, :])
    ax.plot(
        tes_ms,
        ica_pes[comp_voxel, :, idx],
        color="black",
        alpha=0.4,
        linewidth=6,
        marker="o",
        markersize=10,
        label="Parameter estimates",
    )
    ax.plot(
        tes_ms,
        r2_preds[comp_voxel, :, idx],
        color="blue",
        marker="o",
        label="$R_2^*$ model ($\\kappa$)",
    )
    ax.plot(
        tes_ms,
        s0_preds[comp_voxel, :, idx],
        color="red",
        marker="o",
        label="$S_0$ model ($\\rho$)",
    )
    ax.set_title(
        f"{comp} ({comp_tbl.loc[comp, 'classification']})",
        fontsize=16,
        loc="left",
    )
    ax.set_xticks(np.round(tes_ms, 1))
    ax.set_xlabel("Echo Time (ms)", fontsize=14)
    ax.set_ylabel("Parameter estimate", fontsize=14)
    ax.legend(fontsize=12)

fig.tight_layout()
glue_figure("fig_ica_weights", fig, display=False)
```

```{figure} figures/fig_ica_weights.png
:name: fig_ica_weights
:align: center

Echo-wise parameter estimates (gray) and the predictions of the $R_{2}^{*}$ (blue) and $S_{0}$ (red) models
for the accepted and rejected example components, each in the voxel where the component has its highest weight.
```

### Component selection and multi-echo denoising

A decision tree uses $\kappa$, $\rho$, and other metrics to classify each component as BOLD-like (accepted) or non-BOLD-like (rejected).
`tedana` then produces two main denoised datasets:

- The {term}`MEDN` data are the optimally combined data with the rejected components regressed out.
  They keep the accepted components, as well as any variance that was not explained by the ICA.
- The {term}`MEHK` data are built from the accepted components alone,
  so they also lose the variance that was not explained by the ICA.

```{code-cell} ipython3
:tags: [hide-cell]
fig, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(16, 6))
for ax, (label, arr) in zip(
    axes,
    [("Optimally combined", oc), ("MEDN", medn), ("MEHK", mehk)],
):
    ax.plot(zscore_time(arr)[:, voxel_idx], color="black")
    ax.set_title(label, fontsize=14, loc="left")

axes[1].set_ylabel("Signal (z)", fontsize=14)
axes[-1].set_xlim(0, n_vols - 1)
axes[-1].set_xlabel("Volume", fontsize=14)
fig.tight_layout()
glue_figure("fig_medn_timeseries", fig, display=False)
```

```{figure} figures/fig_medn_timeseries.png
:name: fig_medn_timeseries
:align: center

Time series of the optimally combined, denoised (MEDN), and high-kappa (MEHK) data in our voxel.
```

### Post-processing to remove spatially diffuse noise

Spatial ICA looks for components whose maps are spatially independent of one another.
That works well for noise that is localized, such as motion artifacts at the edges of the brain or pulsation near large vessels,
but noise that is spread out across the whole brain, such as fluctuations related to respiration,
overlaps with every other component's map and does not separate cleanly into components of its own.
One of several post-processing strategies may be applied to the MEDN or MEHK data to remove this spatially diffuse noise.
Methods that have been used in the past include global signal regression (GSR),
{prf:ref}`minimum image regression <minimum-image-regression>` (MIR), anatomical CompCor,
Go Decomposition (GODEC), and robust PCA.
`tedana` implements GSR and MIR.

```{code-cell} ipython3
:tags: [hide-cell]
fig, axes = plt.subplots(2, sharex=True, sharey=True, figsize=(16, 5))
for ax, (label, before, after) in zip(
    axes,
    [("MEDN", medn, medn_mir), ("MEHK", mehk, mehk_mir)],
):
    ax.plot(
        zscore_time(before)[:, voxel_idx],
        color="gray",
        alpha=0.6,
        linewidth=3,
        label=label,
    )
    ax.plot(zscore_time(after)[:, voxel_idx], color="black", label=f"{label} + MIR")
    ax.legend(frameon=True, fontsize=12, loc="upper right", framealpha=1)

axes[-1].set_xlim(0, n_vols - 1)
axes[-1].set_xlabel("Volume", fontsize=14)
fig.tight_layout()
glue_figure("fig_mir_timeseries", fig, display=False)
```

```{figure} figures/fig_mir_timeseries.png
:name: fig_mir_timeseries
:align: center

Time series of the MEDN and MEHK data in our voxel, before (gray) and after (black) {prf:ref}`minimum image regression <minimum-image-regression>`.
```

MIR estimates the global noise from the minimum, over time, of each voxel's BOLD-like signal.
This minimum image of the BOLD-like data tends to look like a $T_{1}$-weighted image,
so the global fluctuation that MIR removes is called the "$T_{1}$-like effect."
The steps of minimum image regression are as follows.

```{prf:algorithm} Minimum image regression
:label: minimum-image-regression
<!-- Written by Taylor Salo, Jessica Bartley, and Elizabeth DuPre -->

**Inputs**

- $\mathbf{O}$ is the matrix of optimally combined (OC) data, of shape $v \times t$,
where $v$ is the number of voxels in the brain mask and $t$ is the number of timepoints in the scan.
- $\mathbf{M}$ is the mixing matrix from the ICA decomposition, of shape $c \times t$, where $c$ is the number of components.
- $W$ is the set of indices of all components in $\mathbf{M}$: $W = \{1, 2, 3, ..., c\}$
- $N$ is the set of indices of all non-ignored components (i.e., all accepted or BOLD-like, and rejected or non-BOLD components) in
  $\mathbf{M}$: $N \in \mathbb{N}^k \text{ s.t. } 1 \leq k \leq c, N \subseteq W$
- $A$ is the set of indices of all accepted (i.e., BOLD-like) components in $\mathbf{M}$:
  $A \in \mathbb{N}^l \text{ s.t. } 1 \leq l \leq k, A \subseteq N$

**Outputs**

- Multi-echo denoised data without the T1-like effect, referred to as $\mathbf{D}$ or MEDN+MIR.
- Multi-echo BOLD-like data without the T1-like effect, referred to as $\mathbf{H}$ or MEHK+MIR.
- ICA mixing matrix with the T1-like effect removed from component time series ($\mathbf{K}$).
- Map of the T1-like effect ($\mathbf{m}$)

**Algorithm**

1.  The voxel-wise means ($\mathbf{\overline{O}} \in \mathbb{R}^{v}$) and standard deviations
    ($\mathbf{\sigma_{O}} \in \mathbb{R}^{v}$) of the optimally combined data are computed over time.
2.  The optimally combined data are z-normalized over time ($\mathbf{O_z} \in \mathbb{R}^{v \times t}$).
3.  The normalized optimally combined data matrix ($\mathbf{O_z}$) is regressed on the ICA mixing matrix
    ($\mathbf{M} \in \mathbb{R}^{c \times t}$) to construct component-wise parameter estimate maps
    ($\mathbf{B} \in \mathbb{R}^{v \times c}$).

    <!-- no intercept included, because unnecessary and data are normalized. -->
    $$
        \mathbf{O_{z}} = \mathbf{B} \mathbf{M} + \mathbf{\epsilon}, \enspace \mathbf{\epsilon} \in \mathbb{R}^{v \times t}
    $$

4.  $N$ is used to select rows from the mixing matrix $\mathbf{M}$ and columns from the parameter estimate matrix $\mathbf{B}$
    that correspond to non-ignored (i.e., accepted and rejected) components, forming reduced matrices $\mathbf{M}_N$ and $\mathbf{B}_N$.
    The normalized time series matrix for the combined ignored components and variance left unexplained by the ICA decomposition is then
    computed by subtracting the matrix product of the non-ignored beta weight and mixing matrices from the normalized OC data time series
    ($\mathbf{O_{z}}$).
    The result is referred to as the normalized residuals time series matrix ($\mathbf{R} \in \mathbb{R}^{v \times t}$).

    $$
        \mathbf{R} = \mathbf{O_{z}} - \mathbf{B}_N \mathbf{M}_N, \enspace \mathbf{B}_N \in \mathbb{R}^{v \times |N|},
        \enspace \mathbf{M}_N \in \mathbb{R}^{|N| \times t}
    $$

5.  We can likewise construct the normalized time series of BOLD-like components ($\mathbf{P} \in \mathbb{R}^{v \times t}$) by
    multiplying similarly reduced parameter estimate and mixing matrices composed of only the columns and rows, respectively,
    that are associated with the accepted components indexed in $A$.
    The resulting time series matrix is similar to the {term}`MEHK` data described above,
    except that the component time series have been normalized prior to reconstruction.

    $$
        \mathbf{P} = \mathbf{B}_A \mathbf{M}_A, \enspace \mathbf{B}_A \in \mathbb{R}^{v \times |A|},
        \enspace \mathbf{M}_A \in \mathbb{R}^{|A| \times t}
    $$

6.  The map of the T1-like effect ($\mathbf{m} \in \mathbb{R}^{v}$) is constructed by taking the minimum across timepoints
    from the normalized MEHK time series ($\mathbf{P}$) and then mean-centering across brain voxels.
    Let $J = \{1, ..., t\}$ denote the indices of the columns of matrix $\mathbf{P}$, and let $p_{ij}$ denote the value of the
    element $\mathbf{P}[i,j]$.

    <!-- adapted from https://math.stackexchange.com/a/871689 -->
    $$
        \mathbf{q_{i}} = \min_{j\in{J}}p_{ij} \quad \forall  i = 1,...,v
    $$

    $$
        \mathbf{m} = \mathbf{q} - \mathbf{\overline{q}}, \enspace \mathbf{q} \in \mathbb{R}^{v}
    $$

7.  The standardized optimally combined time series matrix ($\mathbf{O_z}$) is regressed on the T1-like effect map
    ($\mathbf{m}$) to estimate the volume-wise global signal time series ($\mathbf{g} \in \mathbb{R}^t$).

    $$
        \mathbf{O_{z}} = \mathbf{m} \otimes \mathbf{g} + \mathbf{\epsilon}, \enspace \mathbf{\epsilon} \in \mathbb{R}^{v \times t}
    $$

    Where $\otimes$ is the outer product.

8.  The normalized BOLD time series matrix ($\mathbf{P}$) is then regressed on this global signal time series ($\mathbf{g}$)
    in order to estimate a global signal map ($\mathbf{s} \in \mathbb{R}^v$) and the normalized BOLD time series matrix without
    the T1-like effect ($\mathbf{E} \in \mathbb{R}^{v \times t}$).

    $$
        \mathbf{P} = \mathbf{s} \otimes \mathbf{g} + \mathbf{E}
    $$

9.  The time series matrix of BOLD-like components without the T1-like effect (MEHK+MIR, $\mathbf{H} \in \mathbb{R}^{v \times t}$),
    scaled to match the original OC time series matrix, is constructed by multiplying each column of $\mathbf{E}$ by the vector
    $\mathbf{\sigma_{O}}$.

    $$
        \mathbf{H} = \mathbf{E} \circ
        \underbrace{
            \pmatrix{
                \mathbf{{\sigma_{O}}_1} & \cdots & \mathbf{{\sigma_{O}}_1}\\
                \vdots & \vdots & \vdots \\
                \mathbf{{\sigma_{O}}_v} & \cdots & \mathbf{{\sigma_{O}}_v}\\
            }
        }_{t}
    $$

    Where $\circ$ is the [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
    for element-wise multiplication of two matrices.

10. The ICA-denoised time series without the T1-like effect (MEDN+MIR, $\mathbf{D} \in \mathbb{R}^{v \times t}$)
    is constructed by adding the residuals time series ($\mathbf{R}$) to the normalized BOLD time series ($\mathbf{E}$),
    multiplying each column of the result by the vector $\sigma_{O}$, and adding back in the voxel-wise mean of the OC time series
    ($\mathbf{\overline{O}}$).

    $$
        \mathbf{D} = \mathbf{\overline{O}} + (\mathbf{E} + \mathbf{R}) \circ
        \underbrace{
            \pmatrix{
                \mathbf{{\sigma_{O}}_1} & \cdots & \mathbf{{\sigma_{O}}_1}\\
                \vdots & \vdots & \vdots \\
                \mathbf{{\sigma_{O}}_v} & \cdots & \mathbf{{\sigma_{O}}_v}\\
            }
        }_{t}
    $$

11. The ICA mixing matrix without the T1-like effect is then derived by regressing the global signal time series $\mathbf{g}$ from
    each component's time series. Let $\mathbf{r}$ be the vector of parameter estimates from these regressions
    ($\mathbf{r} \in \mathbb{R}^{c}$).

    $$
        \mathbf{M} = \mathbf{r} \otimes \mathbf{g} + \mathbf{K}
    $$

    $$
        \mathbf{K} = \mathbf{M} - \mathbf{r} \otimes \mathbf{g}
    $$
```
