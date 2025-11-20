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

import nibabel as nb

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
