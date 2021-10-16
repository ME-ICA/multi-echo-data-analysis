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

# Denoising Data with ICA

ICA classification methods like `tedana` will produce two important outputs: component time series and component classifications.
The component classifications will indicate whether each componet is "good" (accepted) or "bad" (rejected).
To remove noise from your data, you can regress the "bad" components out of it, though there are multiple methods to accomplish this.

```{tableofcontents}
```

## Aggressive Denoising

If you regress just nuisance regressors (i.e., rejected components) out of your data,
then retain the residuals for further analysis, you are doing aggressive denoising.

````{tab} Python
```python
from nilearn import glm
```
````
````{tab} FSL
```bash
3dcalc --input stuff
```
````
````{tab} AFNI
```bash
3dcalc --input stuff
```
````

## Non-Aggressive Denoising

If you include both nuisance regressors and regressors of interest in your regression,
you are doing nonaggressive denoising.

````{tab} Python
```python
from nilearn import glm
```
````
````{tab} AFNI
```bash
3dcalc --input stuff
```
````

## Component orthogonalization

Independent component analysis decomposes the data into _independent_ components, obviously.
Unlike principal components analysis, the components from ICA are not orthogonal, so they may explain shared variance.
If you want to ensure that variance shared between the accepted and rejected components does not contaminate the denoised data,
you may wish to orthogonalize the rejected components with respect to the accepted components.
This way, you can regress the rejected components out of the data in the form of, what we call, "pure evil" components.

````{tab} Python
```python
from nilearn import glm
```
````
````{tab} AFNI
```bash
3dcalc --input stuff
```
````

Once you have these "pure evil" components, you can perform aggressive denoising on the data.

````{tab} Python
```python
from nilearn import glm
```
````
````{tab} AFNI
```bash
3dcalc --input stuff
```
````