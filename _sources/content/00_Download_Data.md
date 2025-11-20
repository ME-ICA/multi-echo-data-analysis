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

# Download Data

For the tutorials in this book,
we will use partially-preprocessed data from two open multi-echo datasets: Euskalibur and Cambridge.
For more information about these datasets, see {ref}`content:open-datasets`.

```python
import os

from datalad import api as dapi

DATA_DIR = os.path.abspath('../data')

# Download PAFIN fMRIPrep data
dset_dir = os.path.join(DATA_DIR, 'ds006185')
os.makedirs(dset_dir, exist_ok=True)
dapi.install(
    path=dset_dir,
    source='https://github.com/OpenNeuroDatasets/ds006185.git',
)
dapi.get(os.path.join(dset_dir, 'sub-24053', 'ses-1', 'func', 'sub-24053_ses-1_task-rat_rec-nordic_*'), recursive=True)
dapi.get(os.path.join(dset_dir, 'sub-24053', 'ses-1', 'anat', 'sub-24053_ses-1_*'), recursive=True)
```

For now, we will use the Datalab API to download some data we're storing on OpenNeuro.

```{code-cell} ipython3
:tags: [hide-output]

import os
from pathlib import Path

from datalad import api as dapi

DATA_DIR = os.path.abspath('../data')

# Download PAFIN fMRIPrep data
dset_dir = os.path.join(DATA_DIR, 'ds006185')
os.makedirs(dset_dir, exist_ok=True)
dapi.install(
    path=dset_dir,
    source='https://github.com/OpenNeuroDatasets/ds006185.git',
)
subj_dir = os.path.join(dset_dir, 'sub-24053', 'ses-1')
func_dir = Path(os.path.join(subj_dir, 'func'))
func_files = list(func_dir.glob('sub-24053_ses-1_task-rat_rec-nordic_*'))
for f in func_files:
    dapi.get(f)

anat_dir = Path(os.path.join(subj_dir, 'anat'))
anat_files = list(anat_dir.glob('*'))
for f in anat_files:
    dapi.get(f)
```
