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

# Recommended Reading & History

## The first two decades of multi-echo fMRI

{cite:p}`POSSE2012665` includes an historical overview of multi-echo acquisition and research.
Multi-echo fMRI is sometimes discussed as a newer method, but it's as old as fMRI.
Much of this work set up the theory and empirical measurements that drive continued innovation in multi-echo methods.
Early work trying to understand BOLD responses to neural activity used multi-echo sequences {cite:p}`menon19934-e70`.
Multi-echo fMRI is effectively sampling the same frequency induced decay curve
as in NMR spectroscopy, and some of the early work in this area called itself functional spectroscopy {cite:p}`hennig1994detection-075,speck1998functional-366`.

{cite:p}`posse1999enhancement-176` built on much of this work to show how taking the weighted sum of multiple echoes can improve fMRI results over single-echo fMRI.
{cite:p}`poser2006bold-bda` added and validated more ways to calculate the weighted sum across echoes to increase sensitivity to BOLD contrast.
Poser's T2* weighting method is the default weighting used in {py:func}`tedana.combine.make_optcom` and other software and this weighted sum is sometimes called "Optimally Combined".
Despite this method being called "optimal," research continues in comparing and validating echo combination options {cite:p}`heunis2021effects`.


## ICA-based denoising

{cite:p}`KUNDU201759` is a review of multi-echo denoising with a focus on the Multi-Echo ICA (MEICA) algorithm.
One challenge of using multi-echo data is that we are trying to fit a line or decay curve to noisy data.
While the overall data quality might increase, there is noise added by fit errors.
One way to avoid fit errors is just to take the weighted sum across echoes, as highlighted in the previous section.
Averaging signals with independent measurement noise and similar noise distributions
should always be a net gain in data quality,
but it doesn't take full advantage of the relationship between echoes.

{cite:p}`buur2008extraction-629` evaluated and tested ICA and other data decomposition tools with the goal of identifying and removing non-T2* weighted signals with spatio-temporal structure.
If the decomposition method separates the data into sources with more and less T2* contribution,
then one can fit decay models within each source to get better estimates.
{cite:p}`kundu2012differentiating-263,Kundu2013-po` built on this idea of using ICA to develop and share an algorithm,
MEICA, that showed substantial improvements in noise removal.
The appendix of {cite:p}`OLAFSSON201543` includes a good explanation of the math underlying MEICA denoising.
The appendix of {cite:p}`dipasquale2017comparing` includes some recommendations for multi-echo acquisition.

## Broadening of multi-echo users and community

While multi-echo fMRI is old, through the 2000s it was mostly used by MRI physicists and methodology researchers.
There are several reasons for this.
MRI hardware and pulse sequences were too slow to collect a whole-brain fMRI volume in 2 seconds.
Much of the earlier work used slabs of the brain or even a just a few slices.
One could get most of cortex in 2 seconds using in-slice acceleration, but even then there was an additional challenge.
Even though collecting multi-echo data is a relatively minor tweak in standard fMRI pulse sequences,
most pulse sequences that could be used to collect multi-echo fMRI data were research sequences that were
passed researcher-to-researcher by request.
None of the mainstream vendors distributed a multi-echo fMRI sequence that users who weren't MRI experts could just run.
In the mid-2010s several things came together to address the above challenges.
Simultaneous multi-slice acceleration {cite:p}`setsompop2012blipped-controlled-1a0` and improvements in hardware meant that it was possible to collect whole-brain multi-echo fMRI data with minimal compromises.
Research sequences that included multi-echo acqusition started to get shared more widely.
Of particular note, the CMRR Multi-band sequence that was used for the Human Connectome Project on Siemens scanners added multi-echo acqusition in 2014.
While this was a research sequence, since it was widely accessible and used,
it greatly expanded the number of researchers who could easily acquire multi-echo data.

The growth in users was paralleled with a growth in community.
Researchers from multiple sites came together to build [tedana](https://tedana.readthedocs.io)
which is both a platform to run a range of methods that build on the MEICA approach,
and community resources that make it easier for researchers to learn about and use
multi-echo fMRI. The book is one product of this community work.



