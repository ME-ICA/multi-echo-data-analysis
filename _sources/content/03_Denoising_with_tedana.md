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

# Multi-Echo Denoising with `tedana`

In this analysis tutorial, we will use `tedana` {cite:p}`DuPre2021` to perform multi-echo denoising.

Specifically, we will use {py:func}`tedana.workflows.tedana_workflow`.

```{code-cell} ipython3
import os
import matplotlib.pyplot as plt
from glob import glob

import numpy as np
import pandas as pd
from nilearn import image, plotting
from tedana import workflows
from IPython.display import display, HTML
import json
from pprint import pprint
```

```{code-cell} ipython3
data_dir = os.path.abspath("../data")
func_dir = os.path.join(data_dir, "sub-04570/func/")
data_files = [
    os.path.join(func_dir, "sub-04570_task-rest_echo-1_space-scanner_desc-partialPreproc_bold.nii.gz"),
    os.path.join(func_dir, "sub-04570_task-rest_echo-2_space-scanner_desc-partialPreproc_bold.nii.gz"),
    os.path.join(func_dir, "sub-04570_task-rest_echo-3_space-scanner_desc-partialPreproc_bold.nii.gz"),
    os.path.join(func_dir, "sub-04570_task-rest_echo-4_space-scanner_desc-partialPreproc_bold.nii.gz"),
]
echo_times = [12., 28., 44., 60.]
mask_file = os.path.join(func_dir, "sub-04570_task-rest_space-scanner_desc-brain_mask.nii.gz")
confounds_file = os.path.join(func_dir, "sub-04570_task-rest_desc-confounds_timeseries.tsv")

out_dir = os.path.join(data_dir, "tedana")
```


```{code-cell} ipython3
workflows.tedana_workflow(
    data_files,
    echo_times,
    out_dir=out_dir,
    mask=mask_file,
    prefix="sub-04570_task-rest_space-scanner",
    fittype="curvefit",
    tedpca="mdl",
)
```

```{code-cell} ipython3
out_files = sorted(glob(os.path.join(out_dir, "*")))
out_files = [os.path.basename(f) for f in out_files]
print("\n".join(out_files))
```

```{code-cell} ipython3
metrics = pd.read_table(os.path.join(out_dir, "sub-04570_task-rest_space-scanner_desc-tedana_metrics.tsv"))
metrics
```

```{code-cell} ipython3
with open(os.path.join(out_dir, "sub-04570_task-rest_space-scanner_desc-tedana_metrics.json"), "r") as fo:
    data = json.load(fo)

first_five_keys = list(data.keys())[:5]
reduced_data = {k: data[k] for k in first_five_keys}
pprint(reduced_data)
```

```{code-cell} ipython3
df = pd.DataFrame.from_dict(data, orient="index")
df = df.fillna("n/a")
display(HTML(df.to_html()))
```
