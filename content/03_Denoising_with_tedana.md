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
import shutil
from glob import glob
from pprint import pprint

import nibabel as nb
import pandas as pd
from IPython.display import HTML, display
from book_utils import export_html_report, load_pafin
from tedana import workflows

data_path = os.path.abspath('../data')
```

`tedana` keeps the logs and reports of earlier runs in its output folder,
so we start from an empty folder to list only the files from this run.

```{code-cell} ipython3
data = load_pafin(data_path)
out_dir = os.path.join(data_path, "tedana")
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
```

```{code-cell} ipython3
:tags: [hide-output]

workflows.tedana_workflow(
    data['echo_files'],
    data['echo_times'],
    out_dir=out_dir,
    mask=data['mask'],
    prefix="sub-24053_ses-1_task-rat_dir-PA_run-01",
    fittype="loglin",
    tedpca="mdl",
    verbose=True,
    gscontrol=["mir"],
    overwrite=True,
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
    os.path.join(out_dir, "sub-24053_ses-1_task-rat_dir-PA_run-01_desc-tedana_metrics.tsv")
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
    os.path.join(out_dir, "sub-24053_ses-1_task-rat_dir-PA_run-01_desc-tedana_metrics.json"),
    "r",
) as fo:
    metrics = json.load(fo)

first_five_keys = list(metrics.keys())[:5]
reduced_metrics = {k: metrics[k] for k in first_five_keys}
pprint(reduced_metrics)
```

```{code-cell} ipython3
:tags: [output_scroll]

df = pd.DataFrame.from_dict(metrics, orient="index")
df = df.fillna("n/a")
display(HTML(df.to_html()))
```

## The tedana report

The workflow also writes an interactive HTML report, `*_tedana_report.html`,
which loads its images from the `figures` folder next to it.
On your own computer you would simply open that file in a browser.
Here, we copy the report and its figures into the book so that it can be shown below.

```{code-cell} ipython3
report = os.path.join(out_dir, "sub-24053_ses-1_task-rat_dir-PA_run-01_tedana_report.html")
export_html_report(report, "tedana_report")
```

<p><a href="../tedana_report/index.html" target="_blank" rel="noopener">Open the report in its own tab</a> to see it at full size.</p>

````{div} full-width
<iframe src="../tedana_report/index.html" title="tedana report" style="width: 100%; height: 90vh; min-height: 600px; border: 1px solid #ccc; border-radius: 4px;"></iframe>
````
