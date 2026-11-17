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

# Why T₂* and S₀ Vary Across the Brain

The [previous chapter](Signal_Decay.md) described multi-echo data with the monoexponential decay model,

```{math}
:label: monoexponential_decay_variability
S(TE) = S_{0} e^{-TE / T_{2}^{*}} = S_{0} e^{-TE \cdot R_{2}^{*}}
```

and focused on how $T_{2}^{*}$ and $S_{0}$ fluctuate over time.
This chapter looks at their *average* values instead.
A map of mean $T_{2}^{*}$ or $S_{0}$ is far from uniform:
$T_{2}^{*}$ is short above the sinuses and in the basal ganglia and long in the ventricles,
and $S_{0}$ is brighter near the edges of the brain than in the middle.
Below, we describe the physical and physiological mechanisms behind these patterns,
starting with the different kinds of transverse relaxation.

```{code-cell} ipython3
:tags: [hide-cell]
import json
import os

import matplotlib.pyplot as plt
import nibabel as nb
import nitransforms as nit
import numpy as np
import pandas as pd
from myst_nb import glue
from nilearn import image, masking, plotting
from tedana import decay, utils

from book_utils import glue_figure, load_pafin

data_path = os.path.abspath("../data")
prefix = "sub-24053_ses-1_task-rat_dir-PA_run-01"
func_dir = os.path.join(data_path, "ds006185/sub-24053/ses-1/func")
anat_dir = os.path.join(data_path, "ds006185/sub-24053/ses-1/anat")
```

## $R_{2}^{*}$, $R_{2}$, and $R_{2}'$

It is often easier to think about relaxation *rates* than relaxation *times*.
A relaxation rate is the inverse of the corresponding time ($R_{2}^{*} = 1 / T_{2}^{*}$),
and is usually given in units of $s^{-1}$ (or Hz).
Rates are convenient because independent sources of relaxation add up:
if two mechanisms each dephase the signal,
the total decay rate is the sum of their individual rates.

After excitation, the transverse magnetization in a voxel decays for two broad reasons.

1.  **Irreversible dephasing ($R_{2} = 1 / T_{2}$).**
    Water molecules move and interact with their neighbors,
    so each spin experiences rapidly and randomly fluctuating magnetic fields.
    The phase errors that this produces are random, and they cannot be undone.
    This is "true" transverse relaxation, and it is what a spin-echo sequence measures.
2.  **Reversible dephasing ($R_{2}' = 1 / T_{2}'$).**
    If the magnetic field varies *within* a voxel but stays constant over time,
    spins in different parts of the voxel precess at slightly different frequencies.
    They gradually fall out of phase with one another, and the summed signal shrinks.
    Because each spin keeps the same frequency, a 180° refocusing pulse can reverse this dephasing,
    which is why a spin echo is insensitive to it.

A gradient-echo sequence, which is what almost all BOLD fMRI uses, has no refocusing pulse,
so it is sensitive to both:

```{math}
:label: r2star_sum
R_{2}^{*} = R_{2} + R_{2}'
\qquad \Longleftrightarrow \qquad
\frac{1}{T_{2}^{*}} = \frac{1}{T_{2}} + \frac{1}{T_{2}'}
```

Because $R_{2}'$ can never be negative, $T_{2}^{*}$ is always shorter than $T_{2}$.

The field variations that cause $R_{2}'$ are usually split by their spatial scale
{cite:p}`yablonskiy1994theory,ulrich2016separation`:

- **Macroscopic** variations happen on a scale similar to or larger than a voxel.
  The typical example is the field gradient near air-tissue boundaries,
  such as the sinuses and ear canals.
  These depend on the head, the shim, and the imaging parameters, not just on the tissue in the voxel.
- **Mesoscopic** variations happen on a scale much smaller than a voxel,
  but much larger than the distance a water molecule diffuses during the echo time.
  They are caused by small objects with a magnetic susceptibility that differs from that of the surrounding tissue,
  such as venules and veins containing deoxyhemoglobin, iron-rich cells, and myelinated axons.
  These are properties of the tissue.

The line between reversible and irreversible dephasing is not perfectly sharp.
Water that diffuses through the field around a very small object, such as a capillary,
experiences a field that changes over time from the spin's perspective.
That part of the dephasing cannot be refocused, so it contributes to $R_{2}$ rather than $R_{2}'$.
This is why spin-echo BOLD fMRI still has some sensitivity to deoxyhemoglobin,
mostly from the smallest vessels.

The figure below shows decay curves for a voxel with the same $R_{2}$, with and without additional $R_{2}'$.
The values are illustrative.

```{code-cell} ipython3
:tags: [hide-cell]
tes_full = np.linspace(0, 150, 301)
t2 = 80  # ms, illustrative
t2s = 50  # ms, illustrative
t2p = 1 / ((1 / t2s) - (1 / t2))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    tes_full,
    np.exp(-tes_full / t2),
    color="black",
    linewidth=2,
    label=f"Spin echo: $R_{{2}}$ only ($T_{{2}}$ = {t2} ms)",
)
ax.plot(
    tes_full,
    np.exp(-tes_full / t2s),
    color="tab:orange",
    linewidth=2,
    label=f"Gradient echo: $R_{{2}} + R_{{2}}'$ ($T_{{2}}^{{*}}$ = {t2s} ms)",
)
ax.fill_between(
    tes_full,
    np.exp(-tes_full / t2s),
    np.exp(-tes_full / t2),
    color="tab:orange",
    alpha=0.15,
    label=f"Signal lost to $R_{{2}}'$ ($T_{{2}}'$ = {t2p:.0f} ms)",
)
ax.set_xlim(0, 150)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Echo time (ms)", fontsize=14)
ax.set_ylabel("Signal (fraction of $S_{0}$)", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.legend(fontsize=12)
fig.tight_layout()
glue_figure("fig_relaxation_rates", fig, display=False)
```

```{figure} figures/fig_relaxation_rates.png
:name: fig_relaxation_rates
:align: center

Signal decay with only irreversible dephasing ($R_{2}$, as measured with a spin echo)
and with additional reversible dephasing ($R_{2}'$, as measured with a gradient echo).
The relaxation times are illustrative.
```

Everything that follows is a story about which of these terms changes from one region to another.
Multi-echo gradient-echo fMRI only measures their sum, $R_{2}^{*}$.
Separating $R_{2}$ from $R_{2}'$ requires additional measurements,
such as spin echoes or a combined gradient-echo/spin-echo sequence {cite:p}`he2007quantitative,ulrich2016separation`.

## Mean $T_{2}^{*}$ and $S_{0}$ in real data

To see what these maps look like, we fit the monoexponential model to the mean of each echo
from the five-echo run used throughout this book,
which was acquired at 3T.
We use `tedana`'s log-linear fit and adaptive mask,
which limits the fit in each voxel to the echoes with good signal.

This section uses a cortical and subcortical parcellation from fMRIPrep,
in addition to the files listed in [Download Data](00_Download_Data.md):

```bash
datalad get sub-24053/ses-1/anat/sub-24053_ses-1_desc-aparcaseg_dseg.nii.gz
```

```{code-cell} ipython3
:tags: [hide-cell]
data = load_pafin(data_path)
tes = np.array(data["echo_times"]) * 1000  # in ms
mask_img = nb.load(data["mask"])

# Fit T2* and S0 to the mean of each echo over time
mean_imgs = [image.mean_img(f, copy_header=True) for f in data["echo_files"]]
mean_data = np.stack([masking.apply_mask(img, mask_img) for img in mean_imgs], axis=1)
_, adaptive_mask = utils.make_adaptive_mask(mean_data[:, :, None], methods=["dropout"])
t2s_map, s0_map, *_ = decay.fit_decay(
    mean_data,
    tes,
    adaptive_mask,
    fittype="loglin",
    report=False,
)
good = adaptive_mask >= 2

# Bring the FreeSurfer parcellation into the space of the BOLD data
xfm = nit.linear.load(
    os.path.join(func_dir, f"{prefix}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"),
    fmt="itk",
)
aparc_img = nit.resampling.apply(
    ~xfm,
    os.path.join(anat_dir, "sub-24053_ses-1_desc-aparcaseg_dseg.nii.gz"),
    reference=mask_img,
    order=0,
)
labels = np.round(masking.apply_mask(aparc_img, mask_img)).astype(int)

# FreeSurfer labels for each region of interest (left and right hemispheres)
rois = {
    "Lateral orbitofrontal cortex": [1012, 2012],
    "Medial orbitofrontal cortex": [1014, 2014],
    "Inferior temporal cortex": [1009, 2009],
    "Lateral occipital cortex": [1011, 2011],
    "Precentral cortex": [1024, 2024],
    "Superior frontal cortex": [1028, 2028],
    "Cerebral white matter": [2, 41],
    "Globus pallidus": [13, 52],
    "Putamen": [12, 51],
    "Caudate": [11, 50],
    "Thalamus": [10, 49],
    "Cerebellar cortex": [8, 47],
    "Lateral ventricles": [4, 43],
}
rows = []
for roi_name, roi_labels in rois.items():
    roi_idx = np.isin(labels, roi_labels) & good
    rows.append(
        {
            "Region": roi_name,
            "Voxels": int(roi_idx.sum()),
            "T2* (ms)": np.median(t2s_map[roi_idx]),
            "S0 (a.u.)": np.median(s0_map[roi_idx]),
        }
    )

roi_df = pd.DataFrame(rows).set_index("Region")
cortex_idx = (labels >= 1000) & (labels < 3000) & good
cortex_t2s = np.median(t2s_map[cortex_idx])

glue("cortex_t2s", float(cortex_t2s), display=False)
glue("ofc_t2s", float(roi_df.loc["Lateral orbitofrontal cortex", "T2* (ms)"]), display=False)
glue("itc_t2s", float(roi_df.loc["Inferior temporal cortex", "T2* (ms)"]), display=False)
glue("gp_t2s", float(roi_df.loc["Globus pallidus", "T2* (ms)"]), display=False)
glue("putamen_t2s", float(roi_df.loc["Putamen", "T2* (ms)"]), display=False)
glue("caudate_t2s", float(roi_df.loc["Caudate", "T2* (ms)"]), display=False)
glue("wm_t2s", float(roi_df.loc["Cerebral white matter", "T2* (ms)"]), display=False)
glue("ventricle_t2s", float(roi_df.loc["Lateral ventricles", "T2* (ms)"]), display=False)
```

```{code-cell} ipython3
:tags: [hide-cell]
t2s_img = masking.unmask(np.where(good, t2s_map, 0), mask_img)
s0_img = masking.unmask(np.where(good, s0_map, 0), mask_img)
cut_coords = [-25, -12, 0, 12, 25, 40]

plt.style.use("dark_background")
fig, axes = plt.subplots(figsize=(18, 7), nrows=2)
plotting.plot_img(
    s0_img,
    display_mode="z",
    cut_coords=cut_coords,
    cmap="gray",
    vmin=0,
    vmax=np.percentile(s0_map[good], 99),
    colorbar=True,
    annotate=False,
    draw_cross=False,
    black_bg=True,
    axes=axes[0],
    figure=fig,
)
axes[0].set_title("$S_{0}$ (a.u.)", fontsize=18)
plotting.plot_img(
    t2s_img,
    display_mode="z",
    cut_coords=cut_coords,
    cmap="viridis",
    vmin=0,
    vmax=100,
    threshold=0.5,
    colorbar=True,
    annotate=False,
    draw_cross=False,
    black_bg=True,
    axes=axes[1],
    figure=fig,
)
axes[1].set_title("$T_{2}^{*}$ (ms)", fontsize=18)
glue_figure("fig_t2s_s0_maps", fig, display=False)
plt.style.use("default")
```

```{figure} figures/fig_t2s_s0_maps.png
:name: fig_t2s_s0_maps
:align: center

Mean $S_{0}$ (top) and $T_{2}^{*}$ (bottom) estimated from a five-echo resting-state run at 3T.
The $T_{2}^{*}$ color scale is capped at 100 ms, so the ventricles saturate.
```

```{code-cell} ipython3
:tags: [hide-cell]
fig, axes = plt.subplots(figsize=(14, 6), ncols=2, sharey=True)
y = np.arange(roi_df.shape[0])
axes[0].barh(y, roi_df["T2* (ms)"], color="tab:blue")
axes[0].axvline(cortex_t2s, color="black", linestyle="--", linewidth=1, label="All cortex")
axes[0].set_xlabel("Median $T_{2}^{*}$ (ms)", fontsize=14)
axes[0].legend(fontsize=12, loc="lower right")
axes[1].barh(y, roi_df["S0 (a.u.)"], color="tab:red")
axes[1].set_xlabel("Median $S_{0}$ (a.u.)", fontsize=14)
axes[0].set_yticks(y)
axes[0].set_yticklabels(roi_df.index, fontsize=12)
axes[0].invert_yaxis()
for ax in axes:
    ax.tick_params(axis="x", which="major", labelsize=12)

fig.tight_layout()
glue_figure("fig_t2s_s0_rois", fig, display=False)
```

```{figure} figures/fig_t2s_s0_rois.png
:name: fig_t2s_s0_rois
:align: center

Median $T_{2}^{*}$ and $S_{0}$ in selected regions from the FreeSurfer parcellation,
restricted to voxels with at least two good echoes.
```

Several patterns stand out, and each points to a different mechanism:

- The median $T_{2}^{*}$ across cortex is {glue:text}`cortex_t2s:.0f` ms,
  but it is only {glue:text}`ofc_t2s:.0f` ms in lateral orbitofrontal cortex
  and {glue:text}`itc_t2s:.0f` ms in inferior temporal cortex.
  These regions sit above the sinuses and next to the ear canals ({ref}`macroscopic-field-inhomogeneity`).
- In the basal ganglia, $T_{2}^{*}$ is shortest in the globus pallidus ({glue:text}`gp_t2s:.0f` ms),
  followed by the putamen ({glue:text}`putamen_t2s:.0f` ms) and the caudate ({glue:text}`caudate_t2s:.0f` ms).
  This is the same order as their iron content ({ref}`iron`).
- $T_{2}^{*}$ is shorter in white matter ({glue:text}`wm_t2s:.0f` ms) than in cortex ({ref}`myelin-and-fiber-orientation`).
- The ventricles have a very long $T_{2}^{*}$ ({glue:text}`ventricle_t2s:.0f` ms),
  but a *lower* $S_{0}$ than brain tissue, even though cerebrospinal fluid (CSF) contains more water ({ref}`t1-saturation`).
- $S_{0}$ also varies smoothly across the head, in patterns that do not follow tissue boundaries
  ({ref}`coil-sensitivity`).

The rest of this chapter describes these mechanisms.

## Sources of $T_{2}^{*}$ variability

(macroscopic-field-inhomogeneity)=
### Macroscopic field inhomogeneity and air-tissue boundaries

The magnetic susceptibility of air differs from that of tissue,
so the main magnetic field ($B_{0}$) is distorted wherever the two meet.
In the head, this happens mostly above the sinuses and around the ear canals,
which affects orbitofrontal cortex, ventromedial prefrontal cortex,
and the inferior and anterior temporal lobes, including the amygdala {cite:p}`ojemann1997anatomic`.
The field near these boundaries changes across a single voxel,
so spins at one edge of the voxel precess at a different frequency from spins at the other edge.
This is $R_{2}'$ dephasing on a macroscopic scale,
and it is responsible for the dark "dropout" regions in long-TE images.

Macroscopic dephasing is not a property of the tissue.
It depends on the voxel size (bigger voxels span a larger range of field values),
the slice orientation, the head's position in the scanner, and how well the field was shimmed
{cite:p}`reichenbach1997theory`.
The same orbitofrontal voxel will have a different apparent $T_{2}^{*}$ in different acquisitions.
Field gradients in the phase-encoding direction can also shift the echo in k-space,
which changes the *effective* echo time of the voxel {cite:p}`deichmann2002compensation`.

These gradients also change the *shape* of the decay.
For a field that changes linearly across a slice of thickness $\Delta z$,
the signal is multiplied by a sinc function {cite:p}`fernandezseara2000postprocessing`:

```{math}
:label: sinc_dephasing
S(TE) = S_{0} e^{-TE \cdot R_{2}^{*}} \left| \mathrm{sinc}\left(\Delta f \cdot TE\right) \right|
```

where $\Delta f$ is the difference in frequency between the top and the bottom of the slice
and $\mathrm{sinc}(x) = \sin(\pi x) / (\pi x)$.
This decay is not monoexponential.
It decays slowly at short echo times and quickly at longer echo times,
and the signal can even reappear briefly after the first zero of the sinc.
Fitting a monoexponential model to these data produces an apparent $R_{2}^{*}$ that is too high,
by up to 100% in brain at 1.5 and 4T {cite:p}`fernandezseara2000postprocessing`,
and an $S_{0}$ that is wrong as well.
The figure below simulates this for the echo times of our data.

```{code-cell} ipython3
:tags: [hide-cell]
true_s0 = 100
true_t2s = 50  # ms


def fit_loglinear(signal, echo_times):
    """Fit a monoexponential decay with a log-linear model."""
    design = np.column_stack([np.ones_like(echo_times), -echo_times])
    betas = np.linalg.lstsq(design, np.log(signal), rcond=None)[0]
    return np.exp(betas[0]), 1 / betas[1]


fig, axes = plt.subplots(figsize=(16, 6), ncols=2, sharey=True)
colors = ["black", "tab:blue", "tab:orange"]

# Through-slice field gradients
for delta_f, color in zip([0, 5, 10], colors):
    sinc_term = np.abs(np.sinc(delta_f * tes_full / 1000))
    full_curve = true_s0 * np.exp(-tes_full / true_t2s) * sinc_term
    echo_signal = true_s0 * np.exp(-tes / true_t2s) * np.abs(np.sinc(delta_f * tes / 1000))
    fit_s0, fit_t2s = fit_loglinear(echo_signal, tes)
    axes[0].plot(
        tes_full,
        full_curve,
        color=color,
        linewidth=2,
        label=(
            f"$\\Delta f$ = {delta_f} Hz: "
            f"fitted $T_{{2}}^{{*}}$ = {fit_t2s:.0f} ms, $S_{{0}}$ = {fit_s0:.0f}"
        ),
    )
    axes[0].plot(
        tes_full,
        fit_s0 * np.exp(-tes_full / fit_t2s),
        color=color,
        linestyle=":",
        linewidth=1.5,
    )
    axes[0].scatter(tes, echo_signal, color=color, s=60, zorder=3)

axes[0].set_title("A. Through-slice field gradient", fontsize=16)

# A fast-decaying component, such as myelin water
fast_fraction = 0.15
fast_t2s = 10  # ms
slow_t2s = 55  # ms
two_comp_full = true_s0 * (
    (1 - fast_fraction) * np.exp(-tes_full / slow_t2s)
    + fast_fraction * np.exp(-tes_full / fast_t2s)
)
two_comp_echoes = true_s0 * (
    (1 - fast_fraction) * np.exp(-tes / slow_t2s)
    + fast_fraction * np.exp(-tes / fast_t2s)
)
fit_s0, fit_t2s = fit_loglinear(two_comp_echoes, tes)
axes[1].plot(
    tes_full,
    two_comp_full,
    color="black",
    linewidth=2,
    label=(
        f"{1 - fast_fraction:.0%} at {slow_t2s} ms + {fast_fraction:.0%} at {fast_t2s} ms"
    ),
)
axes[1].plot(
    tes_full,
    fit_s0 * np.exp(-tes_full / fit_t2s),
    color="tab:red",
    linestyle=":",
    linewidth=2,
    label=f"Monoexponential fit: $T_{{2}}^{{*}}$ = {fit_t2s:.0f} ms, $S_{{0}}$ = {fit_s0:.0f}",
)
axes[1].scatter(tes, two_comp_echoes, color="black", s=60, zorder=3)
axes[1].set_title("B. Two compartments", fontsize=16)

for ax in axes:
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Echo time (ms)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=11, loc="upper right")

axes[0].set_ylabel("Signal", fontsize=14)
fig.tight_layout()
glue_figure("fig_nonexponential_decay", fig, display=False)
```

```{figure} figures/fig_nonexponential_decay.png
:name: fig_nonexponential_decay
:align: center

Monoexponential fits to decay curves that are not monoexponential.
Points mark the five echo times of our data, and dotted lines show the fitted monoexponential curves.
(A) A voxel with a true $S_{0}$ of 100 and a true $T_{2}^{*}$ of 50 ms,
with a linear field gradient through the slice that spans $\Delta f$ Hz.
(B) A voxel with two water compartments with different $T_{2}^{*}$ values,
such as myelin water and intra- and extra-axonal water in white matter.
The fractions and relaxation times are illustrative.
```

In practice, this means that the short $T_{2}^{*}$ values in dropout regions
reflect the acquisition as much as the tissue,
and that $T_{2}^{*}$ and $S_{0}$ estimates in those regions depend strongly on which echoes are used in the fit.

### Deoxygenated blood

Deoxyhemoglobin is paramagnetic, so blood vessels that contain it
distort the magnetic field in the tissue around them.
This is the source of BOLD contrast {cite:p}`ogawa1990brain`,
and it contributes to $R_{2}'$ (and, through diffusion around capillaries, to $R_{2}$) even at rest.
For a network of randomly oriented vessels,
the resulting $R_{2}'$ is proportional to the volume fraction of the vessels,
to the susceptibility difference between blood and tissue,
and to the field strength {cite:p}`yablonskiy1994theory`.
The susceptibility difference in turn depends on the amount of deoxyhemoglobin in the blood,
so the baseline contribution of blood to $R_{2}^{*}$ depends on
the deoxygenated blood volume, the oxygen extraction fraction, and the hematocrit.
Quantitative BOLD methods use exactly this relationship to map deoxygenated blood volume and oxygen extraction fraction
from the shape of the signal decay {cite:p}`he2007quantitative`.

Changing venous oxygenation without changing blood volume (by breathing oxygen-enriched air)
changes gray matter $R_{2}^{*}$ linearly with the deoxygenated fraction of venous blood {cite:p}`croal2017field`.
Blood volume differs across the brain:
it is higher in gray matter than in white matter,
and it is very high in and around large veins, such as the sagittal sinus.
Voxels that contain large veins can have very short $T_{2}^{*}$,
along with unusual signal fluctuations.

(iron)=
### Iron

Most of the non-heme iron in the brain is stored in ferritin, which is paramagnetic.
Iron increases $R_{2}$, but it increases $R_{2}^{*}$ even more,
because iron-rich cells and deposits create mesoscopic field inhomogeneities {cite:p}`haacke2005imaging`.
In a postmortem study that measured iron chemically in the same brains that were scanned at 3T,
$R_{2}^{*}$ was linearly related to iron concentration ($r^{2} = 0.90$),
more strongly than $R_{2}$ was ($r^{2} = 0.67$) {cite:p}`langkammer2010quantitative`.
Iron concentration was highest in the globus pallidus, followed by the putamen, the caudate, and the thalamus,
which matches the order of $T_{2}^{*}$ values in our data.

Iron accumulates in the basal ganglia with age,
and $R_{2}^{*}$ in these nuclei increases with age as well {cite:p}`aquino2009age`.
Iron also varies within the cortex.
At high field, iron follows the myeloarchitecture of each region,
varies across cortical layers,
and is the dominant source of contrast between layers and between regions in $T_{2}^{*}$-weighted images {cite:p}`fukunaga2010layer`.

(myelin-and-fiber-orientation)=
### Myelin and fiber orientation

Myelin is diamagnetic, and myelin water has a much shorter $T_{2}$ and $T_{2}^{*}$ than the water inside and between axons
{cite:p}`mackay1994in`.
Myelin therefore shortens $T_{2}^{*}$ in white matter, both by adding a fast-decaying water compartment
and by creating mesoscopic field inhomogeneities around the axons.
In demyelinated mice, white matter $T_{2}^{*}$ increased and the frequency contrast between gray and white matter was largely lost,
which shows that myelin is a primary source of both {cite:p}`lee2012contribution`.
Iron and myelin together account for much of the $T_{1}$ and $T_{2}^{*}$ contrast across the brain {cite:p}`stuber2014myelin`.

Because myelinated axons are long cylinders,
the field disturbance they create depends on the angle between the fibers and $B_{0}$
{cite:p}`wharton2012fiber,sati2013micro`.
As a result, white matter $T_{2}^{*}$ depends on the orientation of its fibers:
in one study, average white matter $T_{2}^{*}$ was 49 ms in fibers perpendicular to $B_{0}$
and 57 ms in fibers parallel to it {cite:p}`bender2010fibre`.
The same is true, to a lesser degree, for cortex.
At 7T, cortical $T_{2}^{*}$ is lower in heavily myelinated sensorimotor, visual, and auditory regions,
and it depends on the orientation of the cortex relative to $B_{0}$ {cite:p}`cohenadad2012t2`.

The compartments in white matter also mean that its decay is not truly monoexponential
{cite:p}`sati2013micro`.
Panel B of {numref}`fig_nonexponential_decay` shows what happens when a monoexponential model is fit to such data:
the fast compartment has mostly decayed by the first echo, so it is effectively missing from the fitted $S_{0}$.

### Water content and intrinsic $T_{2}$

The terms above mostly affect $R_{2}'$.
$R_{2}$ also varies across tissues, mainly with water content and with how much of that water is bound to macromolecules.
The largest difference is between brain tissue and CSF.
CSF is almost pure water, with a $T_{2}$ that is much longer than that of tissue,
and it has no iron, myelin, or deoxygenated blood to add $R_{2}'$.
This is why the ventricles have such long $T_{2}^{*}$ values in {numref}`fig_t2s_s0_rois`.
The last echo of our data is at {glue:text}`max_te:.0f` ms,
much shorter than the $T_{2}^{*}$ of CSF, so the decay in the ventricles is barely sampled
and the $T_{2}^{*}$ estimates there are imprecise.
Voxels that contain a mix of tissue and CSF also decay with more than one rate,
just like the white matter example in {numref}`fig_nonexponential_decay`.

```{code-cell} ipython3
:tags: [remove-cell]
glue("max_te", float(tes.max()), display=False)
```

### Field strength

Both macroscopic and mesoscopic $R_{2}'$ scale with the field strength {cite:p}`yablonskiy1994theory`,
and $R_{2}^{*}$ increases approximately linearly with field strength in cortex, white matter, and the basal ganglia
{cite:p}`peters2007t2`.
At 3T, {cite:t}`wansapura1999nmr` measured gray matter $T_{2}^{*}$ values of 41.6 ms in occipital cortex
and 51.8 ms in frontal cortex,
and at 7T, {cite:t}`cohenadad2012t2` measured an average cortical $T_{2}^{*}$ of 32 ms.
Absolute $T_{2}^{*}$ values, and the contrast between regions, are therefore specific to the field strength,
as well as to the other acquisition parameters described above.

### Does baseline neural activity affect $T_{2}^{*}$?

Yes, but only slightly.
Neural activity affects $T_{2}^{*}$ through blood oxygenation and blood volume,
so regions with different baseline activity could in principle have different baseline $T_{2}^{*}$ values.
In practice, the brain at rest matches blood flow to its metabolic demands,
so the fraction of oxygen extracted from the blood is remarkably uniform across the brain at rest {cite:p}`raichle2001default`.
Regional differences in baseline metabolism therefore produce much smaller differences in deoxyhemoglobin
than one might expect from the differences in metabolism alone.

We can also compare the size of the effects directly.
A BOLD signal change of $\Delta S / S$ at echo time $TE$ corresponds to a change in $R_{2}^{*}$ of roughly
$\Delta R_{2}^{*} \approx -(\Delta S / S) / TE$.
A large, 1% BOLD response at TE = 30 ms therefore corresponds to a change of about {glue:text}`bold_dr2s:.2f` $s^{-1}$.
In comparison, the difference in mean $R_{2}^{*}$ between typical cortex and lateral orbitofrontal cortex in our data is
about {glue:text}`ofc_cortex_dr2s:.0f` $s^{-1}$.
Changes in baseline activity, and in physiological factors that affect the whole brain,
such as arterial $CO_{2}$, hematocrit, or caffeine, can shift $T_{2}^{*}$ a little,
but the spatial pattern of mean $T_{2}^{*}$ is dominated by the static mechanisms described above.

```{code-cell} ipython3
:tags: [remove-cell]
bold_dr2s = 0.01 / 0.030  # 1% signal change at TE = 30 ms, in 1/s
ofc_cortex_dr2s = 1000 / roi_df.loc["Lateral orbitofrontal cortex", "T2* (ms)"] - 1000 / cortex_t2s
glue("bold_dr2s", float(bold_dr2s), display=False)
glue("ofc_cortex_dr2s", float(ofc_cortex_dr2s), display=False)
```

## Sources of $S_{0}$ variability

$S_{0}$ is often described as the "proton density", but it is really just the intercept of the decay model:
the signal we would measure if we could acquire an echo at TE = 0.
For a gradient-echo sequence in a steady state, this is approximately

```{math}
:label: s0_components
S_{0} \propto C(\mathbf{r}) \cdot \rho \cdot
\frac{\sin \alpha \left(1 - e^{-TR / T_{1}}\right)}{1 - \cos \alpha \, e^{-TR / T_{1}}}
```

where $C(\mathbf{r})$ is the sensitivity of the receive coil at location $\mathbf{r}$,
$\rho$ is the proton density,
$\alpha$ is the flip angle that the voxel actually experiences,
and the fraction describes how much the longitudinal magnetization has recovered between excitations.
Several of these terms vary across the brain.

### Proton density

The density of MR-visible water protons is higher in CSF than in gray matter,
and higher in gray matter than in white matter.
Quantitative water content mapping requires correcting for every other term in {eq}`s0_components`
{cite:p}`neeb2006new`,
so $S_{0}$ from a multi-echo fMRI scan is at best a heavily weighted version of proton density.

(t1-saturation)=
### $T_{1}$ saturation

fMRI uses short repetition times, which leave little time for the longitudinal magnetization to recover.
Tissues with a long $T_{1}$ recover less between excitations, so they have a lower $S_{0}$.
At 3T, $T_{1}$ is about 1.3 s in gray matter and 0.8 s in white matter {cite:p}`wansapura1999nmr`,
while the $T_{1}$ of ventricular CSF is about 4.3 s at any field strength {cite:p}`rooney2007magnetic`.
The figure below shows the recovery term of {eq}`s0_components` for these tissues.
At the TR of our data, CSF is saturated much more than tissue,
which offsets much or all of its higher proton density, depending on the flip angle.
This is the main reason why the ventricles are darker than the surrounding tissue in the $S_{0}$ map,
even though CSF contains more water.
The same effect makes $S_{0}$ sensitive to anything that changes the longitudinal magnetization,
such as the inflow of fresh blood, head motion (through spin history effects), and the flip angle that each voxel actually experiences,
which is not uniform across the head at 3T and above.

```{code-cell} ipython3
:tags: [hide-cell]
trs = np.linspace(0.1, 6, 300)  # in seconds
tissue_t1s = {
    "Gray matter ($T_{1}$ = 1.33 s)": 1.331,
    "White matter ($T_{1}$ = 0.83 s)": 0.832,
    "CSF ($T_{1}$ = 4.3 s)": 4.3,
}
tissue_colors = ["tab:gray", "tab:olive", "tab:blue"]
with open(data["echo_files"][0].replace(".nii.gz", ".json"), "r") as fo:
    data_tr = json.load(fo)["RepetitionTime"]


def steady_state(tr, t1, flip_angle):
    """Steady-state signal of a spoiled gradient echo, relative to full relaxation."""
    e1 = np.exp(-tr / t1)
    alpha = np.deg2rad(flip_angle)
    return np.sin(alpha) * (1 - e1) / (1 - np.cos(alpha) * e1)


fig, ax = plt.subplots(figsize=(10, 5))
for (label, t1), color in zip(tissue_t1s.items(), tissue_colors):
    ax.plot(trs, steady_state(trs, t1, 90), color=color, linewidth=2, label=label)
    ax.plot(trs, steady_state(trs, t1, 60), color=color, linewidth=2, linestyle="--")

ax.axvline(data_tr, color="black", linestyle=":", linewidth=1)
ax.text(data_tr + 0.05, 0.03, f"TR = {data_tr:.2f} s", fontsize=12)
ax.plot([], [], color="black", linestyle="-", label="Flip angle = 90°")
ax.plot([], [], color="black", linestyle="--", label="Flip angle = 60°")
ax.set_xlim(0, 6)
ax.set_ylim(0, 1)
ax.set_xlabel("Repetition time (s)", fontsize=14)
ax.set_ylabel("Steady-state signal (fraction of $M_{0}$)", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.legend(fontsize=11, loc="lower right")
fig.tight_layout()
glue_figure("fig_s0_t1_saturation", fig, display=False)
```

```{figure} figures/fig_s0_t1_saturation.png
:name: fig_s0_t1_saturation
:align: center

The steady-state signal of a spoiled gradient-echo sequence,
as a fraction of the equilibrium magnetization ($M_{0}$, the signal of a 90° pulse after full recovery),
for the $T_{1}$ values of gray matter and white matter at 3T and of CSF.
The dotted line marks the TR of the data in this book.
```

(coil-sensitivity)=
### Coil sensitivity

Modern head coils are arrays of many small receive coils placed around the head.
Each element is most sensitive to the tissue right next to it,
and combining them gives the sensitivity of a small surface coil over the whole field of view {cite:p}`roemer1990nmr`.
The gain in sensitivity is much larger near the surface of the brain than at its center.
For example, a 32-channel array improved SNR by up to 3.5 times in cortex,
but only by 1.4 times in the corpus callosum, compared with an 8-channel array {cite:p}`wiggins200632`.
The transmit field is not uniform either.
At 3T and above, the wavelength of the radiofrequency field in tissue is similar to the size of the head,
so the flip angle that each voxel actually experiences ($\alpha$ in {eq}`s0_components`) varies across the brain,
and is typically higher in the center of the head than at its edges.
Many scanners also apply an intensity normalization that partially removes the receive coil's profile.
The net pattern depends on the coil, the scanner, and the reconstruction,
but receive sensitivity, transmit field, and normalization all multiply the signal at every echo by the same amount,
so they appear entirely in $S_{0}$ and not at all in $T_{2}^{*}$.
This is why $S_{0}$ maps contain large, smooth variations that have nothing to do with the tissue,
and why their absolute values are not comparable across coils or scanners.

### Inflow

At short repetition times, blood that flows into the imaging slab has not experienced the previous excitations,
so it has more longitudinal magnetization than the stationary tissue around it.
This raises the signal near large arteries and veins,
particularly at the edges of the slab and with short TRs and large flip angles {cite:p}`gao2012inflow`.
Inflow mostly increases the longitudinal magnetization before any transverse decay happens,
so it appears mainly in $S_{0}$.
However, the inflowing blood also has its own $T_{2}^{*}$,
so a change in how much fresh blood a voxel contains changes the mix of compartments in the voxel,
and can bias the fitted $T_{2}^{*}$ as well.

### Model misspecification

Finally, $S_{0}$ is an extrapolation, so it inherits every error in the decay model.
When the decay is not monoexponential, as in {numref}`fig_nonexponential_decay`,
the intercept of the fitted line is not the true signal at TE = 0.
Fast-decaying compartments, such as myelin water, are mostly missing from $S_{0}$,
and through-slice field gradients can make $S_{0}$ either too high or too low.
These errors are largest in exactly the regions where $T_{2}^{*}$ is hardest to estimate.

## What about changes over time?

This chapter has focused on the *mean* values of $T_{2}^{*}$ and $S_{0}$,
but the same mechanisms help explain why they fluctuate.
Fluctuations in $S_{0}$ are often treated as noise,
since head motion, inflow, and changes in coil sensitivity as the head moves all appear in $S_{0}$.
However, not every $S_{0}$ fluctuation is an artifact.
Blood and tissue have different $S_{0}$ values (through their different $T_{1}$ values, proton densities, and inflow),
so a change in blood volume changes the composition of a voxel and, with it, the fitted $S_{0}$.
Vascular space occupancy (VASO) fMRI is based on the same principle,
since it measures changes in blood volume through the $T_{1}$ difference between blood and tissue {cite:p}`lu2003functional`.
Multi-echo studies have attributed negative BOLD signals near large veins to increases in local blood volume {cite:p}`bianciardi2011negative`,
and have found that the TE = 0 intercept of the BOLD response in visual cortex has its own time course,
which was best explained by a combination of deoxyhemoglobin and venous blood volume changes {cite:p}`havlicek2017echo`.
Deconvolution methods that estimate $T_{2}^{*}$ and $S_{0}$ fluctuations separately,
such as [multi-echo sparse paradigm free mapping](05_3dMEPFM.md) {cite:p}`caballerogaudes2019deconvolution`,
can be used to study these effects.

## Practical implications

- **$T_{2}^{*}$ maps are not tissue maps.**
  Mean $T_{2}^{*}$ mixes properties of the tissue (iron, myelin, blood, water)
  with properties of the acquisition (field strength, voxel size, slice orientation, head position, and shimming).
  Values are not directly comparable across scanners or protocols.
- **$S_{0}$ maps are mostly not interpretable.**
  They are dominated by coil sensitivity and $T_{1}$ saturation,
  and they absorb errors in the decay model.
  They are still useful for quality control, for example to spot coil problems or inflow artifacts.
- **Regional $T_{2}^{*}$ variability is why optimal combination helps.**
  Because $T_{2}^{*}$ varies so much across the brain, no single echo time is optimal everywhere.
  [Optimal combination](01_Optimal_Combination_with_t2smap.md) weights each echo by its expected BOLD sensitivity
  in each voxel, using the voxel's own $T_{2}^{*}$.
- **Estimates are least reliable where they matter most.**
  In dropout regions, only the first few echoes have usable signal,
  and the decay is not monoexponential,
  so $T_{2}^{*}$ and $S_{0}$ estimates depend strongly on which echoes are included in the fit.
  `tedana`'s adaptive mask exists to handle this.
- **Mean values do not classify components.**
  Denoising methods like `tedana` rely on how signal *changes* depend on echo time,
  not on the mean $T_{2}^{*}$ or $S_{0}$ of a voxel.
  The [next chapter](TE_Dependence.md) describes how these fluctuations are used.
