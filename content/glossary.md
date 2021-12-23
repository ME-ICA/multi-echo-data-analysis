```{glossary}
MEDN
  Stands for "multi-echo denoised".
  Refers to an output of the ``tedana`` workflow, constructed by regressing rejected components from the optimally combined data,
  thus retaining (1) accepted components, (2) low-variance ("ignored") components, and (3) unmodeled signal.

MEHK
  Stands of "multi-echo high-Kappa".
  Refers to an output of the ``tedana`` workflow, constructed by scaling and combining accepted components only.
  Thus, this output does not contain (1) low-variance ("ignore") components, (2) rejected components, or (3) unmodeled signal.
```
