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

!!! Abbreviations

  CVR = Cerebrovascular Reactivity
  ME  = Multi-Echo
  IC  = Independent Component
  ICA = Independent Component Analysis
  $P_{ET}CO_2$ = Pressure of End-Tidal Carbon Dioxide


!!! warning

  This tutorial assumes you run:
  - [Optimal combination with `t2smap`](./01_Optimal_Combination_with_t2smap.html)
  - [Multi-Echo Denoising with `tedana`](./03_Denoising_with_tedana.html)
  - [Manual Classification with `rica`](./07_Manual_Classification_with_rica.html)



# Cerebrovascular Reactivity Mapping with `phys2cvr`

Cerebrovascular Reactivity (CVR) Mapping can greatly benefit from multi-echo (ME) data acquisition and independent component analysis (ICA) denoising, especially when it is based on breath-holds, cued deep breaths, or resting-state.

While ME-ICA will greatly reduce motion-related noise (see {cite:t}`MOIA2021117914`), because CVR is an increase in blood flow triggered by vessels dilation, there is a component of signal that is proton-density related. For this reason, the default decision tree in tedana may categorise certain components that should (or at least could) be kept as noise, so it is _strongly_ suggested to perform a manual revision of the automatic categorisation and/or adapt the decision tree to accept a better suited set of components.

On top of that, in the context of breath-holds and cued deep breaths, attention should be made in the denoising stage, as an aggressive approach will remove all sources of signal that inform CVR and the associated haemodynamic lag. Instead, a conservative denoising approach, where noise IC timeseries are orthogonalised to the (temporally shifted) $P_{ET}CO_2$ and the non-noise IC timeseries is to be preferred ({cite:t}`MOIA2021117914`).

Assuming that you are following the other tutorials in this notebook, once you obtained your [optimally combined signal](./01_Optimal_Combination_with_t2smap.html) and your [ICA decomposition](./03_Denoising_with_tedana.html) that has been [manually inspected (and corrected if needed)](./07_Manual_Classification_with_rica.html), you will need to make sure you have:
  - a whole brain mask in functional space that will confine `phys2cvr` operations within it - the `desc-adaptiveGoodSignal_mask.nii.gz` output of `tedana` can work fine, but if you have a different mask that is also ok;
  - if you want, a ROI mask in functional space which average signal will be used as reference to align the regressor (e.g. the $CO_2$ trace) and which average haemodynamic lag will be set as 0-lag - grey matter or cerebellum are commonly used as reference ROI;
  - if you want to run ICA denoising, the timeseries of rejected and accepted components from tedana in two different files; the easiest way to obtain these are using the mixing matrix from `tedana` and the manual classification from `rica` with the help of `pandas`:
  
```{code-cell} ipython3
import pandas as pd

# Assuming RICA was used for manual classification, read the downloaded file (adjusting path as necessary)
man_class = pd.read_csv('../manual_classification.tsv', sep='\t', header=0)

# Read the ICA mixing matrix  (adjusting path as necessary)
ica_mix = pd.read_csv('../desc-ICA_mixing.tsv', sep='\t', header=0)

# Extract the list of rejected and accepted components
rej_comp =  man_class[man_class['classification'] == 'rejected']['Component'].tolist()
acc_comp =  man_class[man_class['classification'] == 'accepted']['Component'].tolist()

# Extract rejected vs accepted timeseries and save them
rej_ts = ica_mix[rej_comp]
acc_ts = ica_mix[acc_comp]

acc_ts.to_csv('../accecpted_ic_timeseries.csv', index=False, header=False)
rej_ts.to_csv('../rejected_ic_timeseries.csv', index=False, header=False)
```

need [TO POTENTIALLY EXTRACT NOISE AND NON-NOISE IC TIMESERIES AND] a brain mask and, if you want, 

After [installing `phys2cvr`](https://phys2cvr.readthedocs.io/en/latest/usage/installation.html#basic-installation), potentially with [extra dependencies](https://phys2cvr.readthedocs.io/en/latest/usage/installation.html#richer-installation), we can import the main workflow of `phys2cvr` and call its help to see all available parameters (many).

```{code-cell} ipython3
from phys2cvr.workflows import phys2cvr as p2c


help(p2c)
```

You may notice many and one potential parameters. `phys2cvr` is meant to be very versatile, but fear not! Only few of them are needed in different situations. For more info, check the [workflow API page](https://phys2cvr.readthedocs.io/en/latest/generated/phys2cvr.workflows.phys2cvr.html#phys2cvr.workflows.phys2cvr)

As default, `phys2cvr` takes anything given to it and creates lagged regressors of the interpolated end-tidal of a given trace (that is, the $P_{ET}CO_2hrf$) after aligning it with a reference signal and convolving it with a canonical HRF. However, you can run a lag regression internally.
Generally speaking, you may need these parameters - although only `fname_func` is mandatory:

``` python
  fname_func='../optcom.nii.gz',      # The optimally combined data
  fname_co2='../co2.phys',            # The CO2 trace with peaks
  fname_mask='../brain_mask.nii.gz',  # A mask to limit the analysis to
  fname_roi='../roi_mask.nii.gz',     # A ROI to use as signal reference
  outdir='../outdir',                 # Output folder
  run_regression=True,                # Run the analysis!
  lag_max=9,                          # Maximum lag to consider
  lag_step=0.3,                       # Lag step to consider
  l_degree=2,                         # Legendre polynomials degrees
```

You can also run a simple regression instead of a lagged one - in that case, remove `lag_max` and `lag_step`.

In terms of regressor for your CVR analysis, many options are supported:
  1. $CO_2$ traces (and their peak indices)
  2. pre-made $P_{ET}CO_2$ traces
  3. task designs (timeseries)
  4. BOLD signals averages from a ROI mask.


## $CO_2$ traces (and their peak indices)

You can get $CO_2$ traces (and their peak indices) with whatever program you want - simply run a very light lowpass filter (and maybe a downsampling), followed by peak detection.

Here's an example with [Physiopy's `peakdet`](https://peakdet.readthedocs.io/en/latest/).

```{code-cell} ipython3
import os

import numpy as np
import peakdet as pk


co2_path = os.path.abspath('../co2data')
```

```{code-cell} ipython3
raise ValueError("SKIP")
data = np.genfromtxt(co2_path)
```

```{code-cell} ipython3
# Set the desired "channel", i.e. column, of the file containing CO2 data
channel=0
# Set the sampling frequency of the data
samp_freq=10000

co2 = pk.Physio(data[:, channel], fs=samp_freq)

# Interpolation (downsampling), then filtering.
co2 = pk.operations.interpolate_physio(co2, 40.0, kind='linear')
co2 = pk.operations.filter_physio(co2, 2, method='lowpass', order=7)

# Automatic peak detection, then manual editing
# If you see that not enough/too many peaks get selected,
# you might tune the threshold and distance.
co2 = pk.operations.peakfind_physio(co2, thresh=0.5, dist=120)
co2 = pk.operations.edit_physio(co2)

outfile = '../co2.phys'
path = pk.save_physio(outfile, co2)
```

With this last output, we can run `phys2cvr`!

```{code-cell} ipython3
p2c(
  fname_func='../optcom.nii.gz',      # The optimally combined data
  fname_co2='../co2.phys',            # The CO2 trace with peaks
  fname_mask='../brain_mask.nii.gz',  # A mask to limit the analysis to
  fname_roi='../roi_mask.nii.gz',     # A ROI to use as signal reference
  outdir='../outdir',                 # Output folder
  run_regression=True,                # Run the analysis!
  lag_max=9,                          # Maximum lag to consider
  lag_step=0.3,                       # Lag step to consider
  l_degree=2,                         # Legendre polynomials degrees
  denoise_matrix_file=['../motion_params', '../other_noise'],  # All the noise you want to add
  orthogonalised_matrix_file=['../rejected_ic_timeseries.csv'],
  extra_matrix_file=['../accepted_ic_timeseries.csv'],
  scale_factor=None,                  # If CO2 trace is not in mmHg, you can scale the final map accordingly
  )
```

If you didn't use `peakdet` for your $CO_2$ processing and peak detection, add the parameter `fname_pidx='../co2_peak_indices` to the list.

`phys2cvr` will compute the $P_{ET}CO_2hrf$ trace by linearly interpolating the end-tidal peaks of your $CO_2$ trace and convolving the former with a canonical HRF. You can select another response function (as well as skipping the interpolation) using the parameter `response_function`.


## Pre-made $P_{ET}CO_2$ traces

If you have a pre-made $P_{ET}CO_2$ trace, simply skip the steps to make one.

```{code-cell} ipython3
p2c(
  fname_func='../optcom.nii.gz',      # The optimally combined data
  fname_co2='../petco2_trace',        # The pre-made PetCO2 trace
  fname_mask='../brain_mask.nii.gz',  # A mask to limit the analysis to
  fname_roi='../roi_mask.nii.gz',     # A ROI to use as signal reference
  outdir='../outdir',                 # Output folder
  run_regression=True,                # Run the analysis!
  lag_max=9,                          # Maximum lag to consider
  lag_step=0.3,                       # Lag step to consider
  l_degree=2,                         # Legendre polynomials degrees
  denoise_matrix_file=['../motion_params', '../other_noise'],  # All the noise you want to add
  orthogonalised_matrix_file=['../rejected_ic_timeseries.csv'],
  extra_matrix_file=['../accepted_ic_timeseries.csv'],
  scale_factor=None,                  # If CO2 trace is not in mmHg, you can scale the final map accordingly
  comp_endtidal=False,                # DO NOT compute PetCO2 internally
  response_function=None,             # DO NOT interpolate the PetCO2 trace.
  )
```


## Task designs (timeseries)

Simply use your task design as $CO_2$ trace. You can interpolate it with an HRF or a respiratory response function internally, or do this beforehand. 

```{code-cell} ipython3
p2c(
  fname_func='../optcom.nii.gz',      # The optimally combined data
  fname_co2='../task_design',        # The pre-made PetCO2 trace
  fname_mask='../brain_mask.nii.gz',  # A mask to limit the analysis to
  fname_roi='../roi_mask.nii.gz',     # A ROI to use as signal reference
  outdir='../outdir',                 # Output folder
  run_regression=True,                # Run the analysis!
  lag_max=9,                          # Maximum lag to consider
  lag_step=0.3,                       # Lag step to consider
  l_degree=2,                         # Legendre polynomials degrees
  denoise_matrix_file=['../motion_params', '../other_noise'],  # All the noise you want to add
  orthogonalised_matrix_file=['../rejected_ic_timeseries.csv'],
  extra_matrix_file=['../accepted_ic_timeseries.csv'],
  scale_factor=None,                  # If CO2 trace is not in mmHg, you can scale the final map accordingly
  comp_endtidal=False,                # DO NOT compute end-tidal trace
  response_function='rrf',            # Interpolate with RRF
  )
```


## BOLD signals averages from a ROI mask.

This is the simplest way to run `phys2cvr`, using the average BOLD signal from a desired ROI.

```{code-cell} ipython3
p2c(
  fname_func='../optcom.nii.gz',      # The optimally combined data
  fname_mask='../brain_mask.nii.gz',  # A mask to limit the analysis to
  fname_roi='../roi_mask.nii.gz',     # A ROI to use as signal reference
  outdir='../outdir',                 # Output folder
  run_regression=True,                # Run the analysis!
  lag_max=9,                          # Maximum lag to consider
  lag_step=0.3,                       # Lag step to consider
  l_degree=2,                         # Legendre polynomials degrees
  denoise_matrix_file=['../motion_params', '../other_noise'],  # All the noise you want to add
  orthogonalised_matrix_file=['../rejected_ic_timeseries.csv'],
  extra_matrix_file=['../accepted_ic_timeseries.csv'],
  comp_endtidal=False,                # DO NOT compute end-tidal trace
  response_function=None,             # DO NOT interpolate the trace
  apply_filter=True,                  # Filter the average BOLD signal (optional)
  highcut=0.04,                       # Lowpass frequency
  lowcut=0.02,                        # Highpass frequency
  butter_order=9,                     # Butterworth filter order
  )
```


# Final remarks

Remember to cite the right papers and resources depending on the desired pipeline!

For more info about `phys2cvr`, as well as knowing what to cite in which case, check out [its documentation](https://phys2cvr.readthedocs.io/en/latest/index.html).

Not a python person? Prefer to run bash CLI commands? Not a problem.
`phys2cvr` is in fact primarily [designed to be used via bash CLI](https://phys2cvr.readthedocs.io/en/latest/usage/cli.html)

```bash
phys2cvr --help
```

For more info about ME CVR, check out {cite:t}`MOIA2021117914`, {cite:t}`cohen2019improving`, and {cite:t}`Cohen2025RS`.

If you want to contribute to `phys2cvr`, have requests, comments, or questions, feel free to [open an issue](https://github.com/smoia/phys2cvr/issues)!