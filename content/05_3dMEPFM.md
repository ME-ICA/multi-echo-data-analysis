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

import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker

data_path = os.path.abspath('../DATA')
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
mask_file = os.path.join(
    func_dir,
    "sub-24053_ses-1_task-rat_rec-nordic_dir-PA_run-01_part-mag_desc-brain_mask.nii.gz"
)
confounds_file = os.path.join(
    func_dir,
    "sub-24053_ses-1_task-rat_rec-nordic_dir-PA_run-01_part-mag_desc-confounds_timeseries.tsv",
)

out_dir = os.path.join(data_path, "pySPFM")
```

```{code-cell} ipython3
:tags: [output_scroll]

from pySPFM import SparseDeconvolution

# Create masker to convert 4D NIfTI data to 2D array
masker = NiftiMasker(mask_img=mask_file)

# Load and mask each echo, then concatenate along time axis
# For multi-echo data, timepoints from different echoes are concatenated along the first axis
masked_data = []
for f in data_files:
    echo_data = masker.fit_transform(f)  # Shape: (n_timepoints, n_voxels)
    masked_data.append(echo_data)

X = np.vstack(masked_data)  # Shape: (n_timepoints * n_echoes, n_voxels)

# Fit the sparse deconvolution model
model = SparseDeconvolution(
    tr=2.47,
    te=echo_times,
    criterion="bic",
)
model.fit(X)

# Get the deconvolved activity-inducing signals
activity = model.coef_  # Shape: (n_timepoints, n_voxels)

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

- `coef_`: The deconvolved activity-inducing signals
- `lambda_`: The regularization parameter values
- `hrf_matrix_`: The HRF convolution matrix used
- `get_fitted_signal()`: Returns the fitted (reconstructed) signal
- `get_residuals(X)`: Returns the residuals between the original data and fitted signal

```{code-cell} ipython3
# Get the fitted signal and residuals
fitted_signal = model.get_fitted_signal()
residuals = model.get_residuals(X)

# Save additional outputs
fitted_img = masker.inverse_transform(fitted_signal)
fitted_img.to_filename(os.path.join(out_dir, "out_fitted.nii.gz"))

residuals_img = masker.inverse_transform(residuals)
residuals_img.to_filename(os.path.join(out_dir, "out_residuals.nii.gz"))

print(f"Fitted signal shape: {fitted_signal.shape}")
print(f"Residuals shape: {residuals.shape}")
```

The pySPFM workflow writes out a number of files.

```{code-cell} ipython3
out_files = sorted(glob(os.path.join(out_dir, "*")))
out_files = [os.path.basename(f) for f in out_files]
print("\n".join(out_files))
```
