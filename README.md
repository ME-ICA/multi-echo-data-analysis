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

To build the book into `_build/html/`:

```bash
make book
```

### How notebook execution works

The tutorials run `tedana`, `pySPFM` and friends on several GB of data, which is more
than a GitHub-hosted runner can handle. So execution and rendering are separated:

- **Execution happens locally.** The book is built with `execute_notebooks: cache`.
  Each notebook's outputs are stored in `_build/.jupyter_cache`, keyed by a hash of its
  code cells. `make book` only executes notebooks whose code has no match in the cache.
  Editing prose, the table of contents or the configuration never triggers execution.
- **Rendering happens in GitHub Actions.** The cache, together with the figures the
  notebooks write to `content/figures/`, is uploaded as the `execution-bundle.tar.gz`
  asset of the [`execution-cache` release](https://github.com/ME-ICA/multi-echo-data-analysis/releases/tag/execution-cache).
  The [`book` workflow](.github/workflows/book.yml) downloads it, renders the HTML and
  deploys it to GitHub Pages.

Neither the built HTML nor the execution results are committed to the repository.

### Changing text only

Edit the Markdown and open a pull request. No data and no local build are needed: CI
renders the book from the published execution cache, and the site is deployed when the
pull request is merged. The rendered site of a pull request can be downloaded from the
workflow run (the `github-pages` artifact).

To preview locally without the data, start from the published execution results:

```bash
make fetch-cache
make book
```

`make fetch-cache` refuses to replace an existing local cache, so that it cannot
discard results you have executed but not published yet.

### Changing code cells

Changing a code cell (its source or its tags) invalidates that notebook's cached
outputs, and it has to be executed again where the data are available
(see [Download Data](content/00_Download_Data.md)):

```bash
make fetch-cache     # optional: reuse the published results for unchanged notebooks
make book            # executes only the notebooks that changed
make check-cache     # lists notebooks that are still missing from the cache
make publish-cache   # uploads the execution results for CI (needs write access)
```

On pull requests, a notebook that is missing from the cache is reported as a warning.
On `main` it fails the workflow before anything is deployed, so the live site never
loses outputs; it keeps serving the previous build until the cache is published.
If the cache is published after the merge, run `make deploy` to rebuild the site.

Useful related targets:

- `make check-env` checks that the local environment still matches `uv.lock`.
- `make clean` removes build output but keeps the execution cache.
- `make clean-all` also removes the execution cache and `content/figures/`,
  so the next `make book` executes every notebook.

## Publishing

The site is deployed by the [`book` workflow](.github/workflows/book.yml) on every push
to `main`, and on demand with `make deploy`.

In the GitHub repository settings, **Pages** must be set to **Source: GitHub Actions**.
