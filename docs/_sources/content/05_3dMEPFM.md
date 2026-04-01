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
from nilearn.maskers import NiftiMasker

data_path = os.path.abspath('../data')
```

```{code-cell} ipython3
raise ValueError("SKIP")
func_dir = os.path.join(data_path, "ds006185/sub-24053/ses-1/func/")
data_files = sorted(
    glob(
        os.path.join(
            func_dir,
            "sub-24053_ses-1_task-rat_dir-PA_run-01_echo-*_part-mag_desc-preproc_bold.nii.gz",
        ),
    ),
)
assert len(data_files) == 5
echo_times = []
for f in data_files:
    json_file = f.replace('.nii.gz', '.json')
    with open(json_file, 'r') as fo:
        metadata = json.load(fo)
    echo_times.append(metadata['EchoTime'] * 1000)
mask_file = os.path.join(
    func_dir,
    "sub-24053_ses-1_task-rat_dir-PA_run-01_part-mag_desc-brain_mask.nii.gz"
)
confounds_file = os.path.join(
    func_dir,
    "sub-24053_ses-1_task-rat_dir-PA_run-01_part-mag_desc-confounds_timeseries.tsv",
)

out_dir = os.path.join(data_path, "pySPFM")
```

```{code-cell} ipython3
:tags: [output_scroll]

from pySPFM import SparseDeconvolution

# Create masker to convert 4D NIfTI data to 2D array
masker = NiftiMasker(mask_img=mask_file)

# Fit masker once on a representative image (first echo)
masker.fit(data_files[0])

# Load and mask each echo, then concatenate along the time axis
# For multi-echo data, each echo's data (shape: n_timepoints × n_voxels) are stacked sequentially
masked_data = []
for f in data_files:
    echo_data = masker.transform(f)  # Shape: (n_timepoints, n_voxels)
    masked_data.append(echo_data)

X = np.vstack(masked_data)  # Shape: (n_echoes * n_timepoints, n_voxels)

# Fit the sparse deconvolution model
model = SparseDeconvolution(
    tr=2.47,
    te=echo_times,
    criterion="bic",
    n_jobs=-1,  # Use all available CPU cores
)
model.fit(X)

# Get the deconvolved activity-inducing signals
# Note: coef_ has shape (n_timepoints, n_voxels) - the model recovers
# the underlying neural activity at the original temporal resolution
activity = model.coef_

# Transform back to NIfTI image and save
os.makedirs(out_dir, exist_ok=True)
activity_img = masker.inverse_transform(activity)
activity_img.to_filename(os.path.join(out_dir, "out_activity.nii.gz"))

# Also save the regularization parameter values
np.save(os.path.join(out_dir, "out_lambda.npy"), model.lambda_)

print(f"Activity shape: {activity.shape}")
print(f"Saved activity to: {os.path.join(out_dir, 'out_activity.nii.gz')}")
```

The `SparseDeconvolution` model provides several useful attributes and methods after fitting:

- `coef_`: The deconvolved activity-inducing signals (shape: n_timepoints × n_voxels)
- `lambda_`: The regularization parameter values
- `hrf_matrix_`: The HRF convolution matrix used
- `get_fitted_signal()`: Returns the fitted (reconstructed) signal; takes no arguments
- `get_residuals(X)`: Returns the residuals between the original data and fitted signal; requires the input data `X` as an argument

```{code-cell} ipython3
# Get the fitted signal and residuals
# Note: These have shape (n_echoes * n_timepoints, n_voxels) matching the input X
fitted_signal = model.get_fitted_signal()
residuals = model.get_residuals(X)

# Save the fitted signal and residuals as numpy arrays
# (shape doesn't match single-echo masker expectations for NIfTI output)
np.save(os.path.join(out_dir, "out_fitted.npy"), fitted_signal)
np.save(os.path.join(out_dir, "out_residuals.npy"), residuals)

print(f"Fitted signal shape: {fitted_signal.shape}")
print(f"Residuals shape: {residuals.shape}")
```

The pySPFM workflow writes out a number of files.

```{code-cell} ipython3
out_files = sorted(glob(os.path.join(out_dir, "*")))
out_files = [os.path.basename(f) for f in out_files]
print("\n".join(out_files))
```
