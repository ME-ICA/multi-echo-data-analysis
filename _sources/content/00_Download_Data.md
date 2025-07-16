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

# Download Data

For the tutorials in this book,
we will use partially-preprocessed data from two open multi-echo datasets: Euskalibur and Cambridge.
For more information about these datasets, see {ref}`content:open-datasets`.

```python
import os
from pprint import pprint

from tedana import datasets

DATA_DIR = os.path.abspath("../data")

euskalibur_dataset = datasets.fetch_euskalibur(
    n_subjects=5,
    low_resolution=False,
    data_dir=DATA_DIR,
)
pprint(euskalibur_dataset)

cambridge_dataset = datasets.fetch_cambridge(
    n_subjects=5,
    low_resolution=False,
    data_dir=DATA_DIR,
)
pprint(cambridge_dataset)
```

For now, we will use repo2data to download data from OpenNeuro.

```{code-cell} ipython3
import os

from repo2data.repo2data import Repo2Data

# Install the data if running locally, or point to cached data if running on neurolibre
DATA_REQ_FILE = os.path.join("../binder/data_requirement.json")

# Download data
repo2data = Repo2Data(DATA_REQ_FILE)
data_path = repo2data.install()
data_path = os.path.abspath(data_path[0])
```
