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

from myst_nb import glue
import numpy as np
import pandas as pd
from nilearn import image, plotting
from tedana import workflows
from IPython.display import display, HTML
import json
from pprint import pprint

from repo2data.repo2data import Repo2Data

# Install the data if running locally, or point to cached data if running on neurolibre
DATA_REQ_FILE = os.path.join("../binder/data_requirement.json")

# Download data
repo2data = Repo2Data(DATA_REQ_FILE)
data_path = repo2data.install()
data_path = os.path.abspath(os.path.join(data_path[0], "data"))
```

```{code-cell} ipython3
func_dir = os.path.join(data_path, "sub-04570/func/")
data_files = [
    os.path.join(func_dir, "sub-04570_task-rest_echo-1_space-scanner_desc-partialPreproc_bold.nii.gz"),
    os.path.join(func_dir, "sub-04570_task-rest_echo-2_space-scanner_desc-partialPreproc_bold.nii.gz"),
    os.path.join(func_dir, "sub-04570_task-rest_echo-3_space-scanner_desc-partialPreproc_bold.nii.gz"),
    os.path.join(func_dir, "sub-04570_task-rest_echo-4_space-scanner_desc-partialPreproc_bold.nii.gz"),
]
echo_times = [12., 28., 44., 60.]
mask_file = os.path.join(func_dir, "sub-04570_task-rest_space-scanner_desc-brain_mask.nii.gz")
confounds_file = os.path.join(func_dir, "sub-04570_task-rest_desc-confounds_timeseries.tsv")

out_dir = os.path.join(data_path, "tedana")
```

```{code-cell} ipython3
:tags: [hide-output]

workflows.tedana_workflow(
    data_files,
    echo_times,
    out_dir=out_dir,
    mask=mask_file,
    prefix="sub-04570_task-rest_space-scanner",
    fittype="curvefit",
    tedpca="mdl",
    verbose=True,
    gscontrol=["mir"],
)
```

The tedana workflow writes out a number of files.

```{code-cell} ipython3
out_files = sorted(glob(os.path.join(out_dir, "*")))
out_files = [os.path.basename(f) for f in out_files]
print("\n".join(out_files))
```

```{code-cell} ipython3
metrics = pd.read_table(os.path.join(out_dir, "sub-04570_task-rest_space-scanner_desc-tedana_metrics.tsv"))
```

```{code-cell} ipython3
:tags: [hide-input]

def color_rejected_red(series):
    """Color rejected components red."""
    return [f"color: red" if series["classification"] == "rejected" else "" for v in series]

metrics.style.apply(color_rejected_red, axis=1)
```

```{code-cell} ipython3
with open(os.path.join(out_dir, "sub-04570_task-rest_space-scanner_desc-tedana_metrics.json"), "r") as fo:
    data = json.load(fo)

first_five_keys = list(data.keys())[:5]
reduced_data = {k: data[k] for k in first_five_keys}
pprint(reduced_data)
```

```{code-cell} ipython3
:tags: [output_scroll]

df = pd.DataFrame.from_dict(data, orient="index")
df = df.fillna("n/a")
display(HTML(df.to_html()))
```

```{code-cell} ipython3
report = os.path.join(out_dir, "tedana_report.html")
with open(report, "r") as fo:
    report_data = fo.read()

figures_dir = os.path.relpath(os.path.join(out_dir, "figures"), os.getcwd())
report_data = report_data.replace("./figures", figures_dir)

display(HTML(report_data))
```
