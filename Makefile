# Name of the uv-managed project environment directory.
# `uv` honors UV_PROJECT_ENVIRONMENT to place the venv somewhere other than .venv.
export UV_PROJECT_ENVIRONMENT := meda

# Reproducible image outputs: matplotlib honors SOURCE_DATE_EPOCH and will stamp
# PNGs with a fixed timestamp instead of "now", so identical plots produce
# identical bytes (and identical content-hashed filenames in _images/).
# 1577836800 == 2020-01-01T00:00:00Z
export SOURCE_DATE_EPOCH := 1577836800

.PHONY: help check-env book clean serve sync-docs site-publish install runall build site

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  install     to sync the 'meda' environment from uv.lock"
	@echo "  check-env   to verify the 'meda' environment matches uv.lock"
	@echo "  book        to activate 'meda' and build the Jupyter Book into _build/"
	@echo "  clean       to clean out site build files"
	@echo "  runall      to run all notebooks in-place, capturing outputs with the notebook"
	@echo "  serve       to serve the repository locally with Jekyll"
	@echo "  build       to build the site HTML and store in _site/"
	@echo "  site        to build the site HTML, store in _site/, and serve with Jekyll"
	@echo "  sync-docs   to copy _build/html into docs/ for GitHub Pages"
	@echo "  site-publish to build the book and sync docs/ (book + sync-docs)"


install:
	uv sync --locked

# Verify that the local 'meda' environment is up-to-date with uv.lock.
# `uv sync --locked --check` exits non-zero if the environment would change.
check-env:
	@echo "Checking that '$(UV_PROJECT_ENVIRONMENT)' matches uv.lock..."
	uv sync --locked --check

# Activate the 'meda' environment (via `uv run`) and build the book.
book: check-env
	uv run --locked jupyter-book build ./

sync-docs:
	python scripts/sync_docs.py

site-publish: book sync-docs

runall:
	jupyter-book run ./content

clean:
	python scripts/clean.py

serve:
	bundle exec guard

build: check-env
	uv run --locked jupyter-book build ./

site: build
	bundle exec jekyll build
	touch _site/.nojekyll
