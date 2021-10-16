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

# Processing Multi-Echo Data

Most multi-echo denoising methods, including `tedana`,
must be called in the context of a larger ME-EPI preprocessing pipeline.
Two common pipelines which support ME-EPI processing include
[fMRIPrep](https://fmriprep.readthedocs.io) and
[afni_proc.py](https://afni.nimh.nih.gov/pub/dist/doc/program_help/afni_proc.py.html).

Users can also construct their own preprocessing pipeline for ME-EPI data from which to call the
multi-echo denoising method of their choice.
There are several general principles to keep in mind when constructing ME-EPI processing pipelines.

In general, we recommend the following:


## 1. Estimate motion correction parameters from one echo and apply those parameters to all echoes

When preparing ME-EPI data for multi-echo denoising with a tool like `tedana`,
it is important not to do anything that mean shifts the data or otherwise separately
scales the voxelwise values at each echo.

For example, head-motion correction parameters should *not* be calculated and applied at an
individual echo level (see above).
Instead, we recommend that researchers apply the same transforms to all echoes in an ME-EPI series.
That is, that they calculate head motion correction parameters from one echo
and apply the resulting transformation to all echoes.


## 2. Perform slice timing correction and motion correction **before** multi-echo denoising

Similarly to single-echo EPI data, slice time correction allows us to assume that voxels across
slices represent roughly simultaneous events.
If the TR is slow enough to necessitate slice-timing (i.e., TR >= 1 sec., as a rule of thumb), then
slice-timing correction should be done before `tedana`.
This is because slice timing differences may impact echo-dependent estimates.

The slice time is generally defined as the excitation pulse time for each slice.
For single-echo EPI data, that excitation time would be the same regardless of the echo time,
and the same is true when one is collecting multiple echoes after a single excitation pulse.
Therefore, we suggest using the same slice timing for all echoes in an ME-EPI series.


## 3. Perform distortion correction, spatial normalization, smoothing, and any rescaling or filtering **after** denoising

Any step that will alter the relationship of signal magnitudes between echoes should occur after denoising and combining
of the echoes. For example, if echo is separately scaled by its mean signal over time, then resulting intensity gradients
and the subsequent calculation of voxelwise T2* values will be distorted or incorrect. An aggressive temporal filter
(e.g., a 0.1Hz low pass filter) or spatial smoothing could similarly distort the relationship between the echoes at each
time point.

```{note}
We are assuming that spatial normalization and distortion correction, particularly non-linear normalization methods
with higher order interpolation functions, are likely to distort the relationship between echoes while rigid body
motion correction would linearly alter each echo in a similar manner. This assumption has not yet been empirically
tested and an affine normalzation with bilinear interpolation may not distort the relationship between echoes.
Additionally, there are benefits to applying only one spatial transform to data rather than applying one spatial
transform for motion correction and a later transform for normalization and distortion correction. Our advice
against doing normalization and distortion correction is a conservative choice and we encourage additional
research to better understand how these steps can be applied before denoising.
```
