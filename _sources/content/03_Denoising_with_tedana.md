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

# Multi-Echo Denoising with `tedana`

In this analysis tutorial, we will use `tedana` {cite:p}`DuPre2021` to perform multi-echo denoising.

Specifically, we will use {py:func}`tedana.workflows.tedana_workflow`.

```{code-cell} ipython3
import json
import os
from glob import glob
from pprint import pprint

import nibabel as nb
import pandas as pd
from IPython.display import HTML, display
from tedana import workflows

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

out_dir = os.path.join(data_path, "tedana")
```

```{code-cell} ipython3
:tags: [hide-output]

workflows.tedana_workflow(
    data_files,
    echo_times,
    out_dir=out_dir,
    mask=mask_file,
    prefix="sub-24053_ses-1_task-rat_rec-nordic_dir-PA_run-01",
    fittype="loglin",
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
metrics = pd.read_table(
    os.path.join(out_dir, "sub-24053_ses-1_task-rat_rec-nordic_dir-PA_run-01_desc-tedana_metrics.tsv")
)
```

```{code-cell} ipython3
:tags: [hide-input]
def color_rejected_red(series):
    """Color rejected components red."""
    return [
        "color: red" if series["classification"] == "rejected" else "" for v in series
    ]


metrics.style.apply(color_rejected_red, axis=1)
```

```{code-cell} ipython3
with open(
    os.path.join(out_dir, "sub-24053_ses-1_task-rat_rec-nordic_dir-PA_run-01_desc-tedana_metrics.json"),
    "r",
) as fo:
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
report = os.path.join(out_dir, "sub-24053_ses-1_task-rat_rec-nordic_dir-PA_run-01_tedana_report.html")
with open(report, "r") as fo:
    report_data = fo.read()

figures_dir = os.path.relpath(os.path.join(out_dir, "figures"), os.getcwd())
report_data = report_data.replace("./figures", figures_dir)

display(HTML(report_data))
```
