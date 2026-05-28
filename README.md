# multi-echo-data-analysis
A Jupyter Book shamelessly copying off of https://github.com/naturalistic-data-analysis/naturalistic_data_analysis.

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://me-ica.github.io/multi-echo-data-analysis)

## Building

This book uses **Jupyter Book 1** (Sphinx, `_config.yml` / `_toc.yml`).

The Python environment is managed with [`uv`](https://docs.astral.sh/uv/) and
locked in `uv.lock`. The Makefile sets `UV_PROJECT_ENVIRONMENT=meda`, so the
project environment is created in `meda/` instead of the default `.venv/`.

To set up the environment:

```bash
make install
```

To check that the local environment still matches `uv.lock`:

```bash
make check-env
```

To build the Jupyter Book into `_build/html/`:

```bash
make book
```

Useful related targets:

- `make clean` removes generated site build files.
- `make runall` runs notebooks under `content/` in place and saves outputs.
- `make build` is an alias-style HTML build target that also writes to `_build/`.
- `make serve` and `make site` use the repository's Ruby/Jekyll tooling and
  require those dependencies to be installed separately.

## Publishing

The site is built locally; the HTML under `docs/` on `main` is what GitHub Pages serves.

1. Install or update the locked environment with `make install`.
2. Build the book and copy `_build/html/` into `docs/` with `make site-publish`
   (or run `make book` followed by `make sync-docs`).
3. Commit the updated `docs/` tree and push `main`.

In the GitHub repository settings, set **Pages** to **Deploy from a branch**, choose branch **`main`**, folder **`/docs`**. If the site was previously deployed with GitHub Actions, switch away from that source so Pages reads the committed `docs/` folder.

Deployment no longer uses GitHub Actions; repository secrets such as `CONFIG_EMAIL` and `CONFIG_NAME` are not required for publishing.
