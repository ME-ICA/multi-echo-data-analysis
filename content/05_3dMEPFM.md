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

# Model-free deconvolution with `pySPFM`

```{code-cell} ipython3
import json
import os
from glob import glob

import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker

data_path = os.path.abspath('../data')
```

```{code-cell} ipython3
func_dir = os.path.join(data_path, "ds006185/sub-24053/ses-1/func/")
data_files = sorted(
    glob(
        os.path.join(
            func_dir,
            "sub-24053_ses-1_task-rat_rec-nordic_dir-PA_run-01_echo-*_part-mag_desc-preproc_bold.nii.gz",
        ),
    ),
)
echo_times = []
for f in data_files:
    json_file = f.replace('.nii.gz', '.json')
    with open(json_file, 'r') as fo:
        metadata = json.load(fo)
    echo_times.append(metadata['EchoTime'] * 1000)

out_dir = os.path.join(data_path, "pySPFM")
```

```{code-cell} ipython3
:tags: [output_scroll]

from pySPFM import SparseDeconvolution

# Fetch the Schaefer 1000-parcel atlas (MNI152NLin2009cAsym, 2 mm).
# Running pySPFM parcel-wise rather than voxel-wise reduces the problem
# from ~218k voxels to 1000 ROIs, making it feasible on a laptop.
atlas = fetch_atlas_schaefer_2018(n_rois=1000)

# NiftiLabelsMasker averages all voxels within each parcel.
# resampling_target='data' resamples the atlas to the fMRI's native grid.
masker = NiftiLabelsMasker(
    labels_img=atlas.maps,
    resampling_target='data',
    standardize=False,
)
masker.fit(data_files[0])

# Load each echo and extract parcel-averaged timeseries, then concatenate
# along the time axis to form the multi-echo input matrix.
masked_data = []
for f in data_files:
    echo_data = masker.transform(f)  # Shape: (n_timepoints, n_parcels)
    masked_data.append(echo_data)

X = np.vstack(masked_data)  # Shape: (n_echoes * n_timepoints, n_parcels)

# Fit the sparse deconvolution model
model = SparseDeconvolution(
    tr=2.47,
    te=echo_times,
    criterion="bic",
)
model.fit(X)

# Get the deconvolved activity-inducing signals
# coef_ shape: (n_timepoints, n_parcels)
activity = model.coef_

# Save parcel-wise activity to avoid reconstructing a large 4D voxel-wise NIfTI
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "out_activity.npy"), activity)

# Compact summary: mean absolute activity per parcel → inverse_transform to a single-volume NIfTI
activity_mean = np.abs(activity).mean(axis=0, keepdims=True)
activity_mean_img = masker.inverse_transform(activity_mean)
activity_mean_img.to_filename(os.path.join(out_dir, "out_activity_mean.nii.gz"))

np.save(os.path.join(out_dir, "out_lambda.npy"), model.lambda_)

print(f"Activity shape: {activity.shape}")
print(f"Saved parcel-wise activity to: {os.path.join(out_dir, 'out_activity.npy')}")
print(f"Saved mean activity image to: {os.path.join(out_dir, 'out_activity_mean.nii.gz')}")
```

The `SparseDeconvolution` model provides several useful attributes and methods after fitting:

- `coef_`: The deconvolved activity-inducing signals (shape: n_timepoints × n_parcels)
- `lambda_`: The regularization parameter values (one per parcel)
- `hrf_matrix_`: The HRF convolution matrix used
- `get_fitted_signal()`: Returns the fitted (reconstructed) signal; takes no arguments
- `get_residuals(X)`: Returns the residuals between the original data and fitted signal; requires the input data `X` as an argument

```{code-cell} ipython3
fitted_signal = model.get_fitted_signal()
print(f"Fitted signal shape: {fitted_signal.shape}")
np.save(os.path.join(out_dir, "out_fitted.npy"), fitted_signal)

residuals = model.get_residuals(X)
print(f"Residuals shape: {residuals.shape}")
np.save(os.path.join(out_dir, "out_residuals.npy"), residuals)
```

The pySPFM workflow writes out a number of files.

```{code-cell} ipython3
out_files = sorted(glob(os.path.join(out_dir, "*")))
out_files = [os.path.basename(f) for f in out_files]
print("\n".join(out_files))
```
