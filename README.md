# multi-echo-data-analysis
A Jupyter Book shamelessly copying off of https://github.com/naturalistic-data-analysis/naturalistic_data_analysis.

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://me-ica.github.io/multi-echo-data-analysis)

## Building

This book uses **Jupyter Book 1** (Sphinx, `_config.yml` / `_toc.yml`). Build with the **hyphenated** command:

`jupyter-book build .`

Do **not** use `jupyter book …` (with a space). That invokes **Jupyter Book 2** / MyST, which expects a different project layout and will fail on this repository.

If you already installed a 2.x release, reinstall with `pip install -r requirements.txt` (which pins Jupyter Book 1.x).

## Publishing

The site is built locally; the HTML under `docs/` on `main` is what GitHub Pages serves.

1. Create a virtual environment, then install dependencies: `pip install -r requirements.txt`.
2. Build the book and copy `_build/html` into `docs/`: `make site-publish` (or run `make book` followed by `make sync-docs`).
3. Commit the updated `docs/` tree and push `main`.

In the GitHub repository settings, set **Pages** to **Deploy from a branch**, choose branch **`main`**, folder **`/docs`**. If the site was previously deployed with GitHub Actions, switch away from that source so Pages reads the committed `docs/` folder.

Deployment no longer uses GitHub Actions; repository secrets such as `CONFIG_EMAIL` and `CONFIG_NAME` are not required for publishing.
