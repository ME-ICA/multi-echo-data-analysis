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

A few figures are drawn on top of the participant's anatomical image.
For those, we also need the preprocessed T1w image
and the transform that aligns the functional data with it.

```bash
datalad get sub-24053/ses-1/anat/sub-24053_ses-1_desc-preproc_T1w.nii.gz
datalad get sub-24053/ses-1/func/sub-24053_ses-1_task-rat_dir-PA_run-01_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt
```

## Data for the external regressor and task analysis pages

[External Regressors in `tedana` Decision Trees](09_External_Regressors.md) and
[Task Analysis of Multi-Echo Data](10_Task_Analysis.md) use one run of the localizer task
from the Dense Investigation of Variability of Affect (DIVA) dataset ([ds002278](https://openneuro.org/datasets/ds002278)),
because it has physiological recordings, strong task responses and four echoes.

The events, the physiological recording and the echo times come from the raw dataset:

```bash
datalad install https://github.com/OpenNeuroDatasets/ds002278.git ../data/ds002278
cd ../data/ds002278
datalad get sub-Blossom/ses-02/func/sub-Blossom_ses-02_task-localizerDetection_run-1_events.tsv
datalad get sub-Blossom/ses-02/func/sub-Blossom_ses-02_task-localizerDetection_run-1_physio.tsv.gz
```

The preprocessed data come from a development version of fMRIPrep 25.2, run with `--me-output-echos`,
which writes each preprocessed echo in the space of the run's BOLD reference.
These derivatives are not on OpenNeuro yet, so for now you need to run fMRIPrep on `sub-Blossom` yourself,
after getting the participant's raw images (`datalad get -J5 sub-Blossom` in `../data/ds002278`).
The pages expect these files in `../data/ds002278-fmriprep`:

```
sub-Blossom/ses-02/func/
    sub-Blossom_ses-02_task-localizerDetection_run-1_echo-{1,2,3,4}_part-mag_desc-preproc_bold.nii.gz
    sub-Blossom_ses-02_task-localizerDetection_run-1_part-mag_desc-preproc_bold.{nii.gz,json}
    sub-Blossom_ses-02_task-localizerDetection_run-1_part-mag_space-run_boldref.nii.gz
    sub-Blossom_ses-02_task-localizerDetection_run-1_part-mag_space-run_desc-brain_mask.nii.gz
    sub-Blossom_ses-02_task-localizerDetection_run-1_part-mag_desc-confounds_timeseries.tsv
    sub-Blossom_ses-02_task-localizerDetection_run-1_from-run_to-T1w_mode-image_desc-coreg_xfm.txt
sub-Blossom/ses-02/anat/
    sub-Blossom_ses-02_run-1_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
```
