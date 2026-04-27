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

# Advanced Applications

## Complex-Valued Multi-Echo fMRI

### T2* Estimation with Through-Slice Dropout Correction

Phase data can be used to correct for through-slice dropout in later echoes when estimating T2*.

### Dynamic Distortion Correction

The acquisition of multiple echoes and phase data makes dynamic distortion correction (i.e., estimating and applying a field map for each volume in the fMRI run) using algorithms like DOCMA and MEDIC.

### Functional Quantitative Susceptibility Mapping

Complex-valued multi-echo data makes it possible to estimate quantitative susceptibility at each TR using QSM approaches.
