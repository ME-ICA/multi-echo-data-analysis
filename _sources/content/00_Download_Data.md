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

```{code-cell} ipython3
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
