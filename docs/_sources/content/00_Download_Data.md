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
we will use partially-preprocessed data from the PAFIN fMRIPrep derivatives dataset.
For more information about these datasets, see {ref}`content:open-datasets`.

```bash
:tags: []
datalad install https://github.com/OpenNeuroDatasets/ds006185.git ../data/
cd ../data/ds006185
datalad get -J5 sub-24053/ses-1/func/sub-24053_ses-1_task-rat_dir-PA_run-01_echo-*
datalad get sub-24053/ses-1/func/sub-24053_ses-1_task-rat_dir-PA_run-01_part-mag_desc-brain_mask.nii.gz
datalad get sub-24053/ses-1/func/sub-24053_ses-1_task-rat_dir-PA_run-01_part-mag_desc-confounds_timeseries.tsv
```
