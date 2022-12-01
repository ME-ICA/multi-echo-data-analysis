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

# Model-free deconvolution with `pySPFM`

```{code-cell} ipython3
import os
from glob import glob

from repo2data.repo2data import Repo2Data

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

out_dir = os.path.join(data_path, "pySPFM")
```

```{code-cell} ipython3
:tags: [output_scroll]

from pySPFM import pySPFM

pySPFM.pySPFM(
    data_fn=data_files,
    mask_fn=mask_file,
    output_filename=os.path.join(out_dir, "out"),
    tr=2.47,
    out_dir=out_dir,
    te=echo_times,
)
```

The pySPFM workflow writes out a number of files.

```{code-cell} ipython3
out_files = sorted(glob(os.path.join(out_dir, "*")))
out_files = [os.path.basename(f) for f in out_files]
print("\n".join(out_files))
```
