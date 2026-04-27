# Name of the uv-managed project environment directory.
# `uv` honors UV_PROJECT_ENVIRONMENT to place the venv somewhere other than .venv.
export UV_PROJECT_ENVIRONMENT := meda

# GitHub Pages serves this repository as a project site under this path.
# MyST/Jupyter Book 2 uses BASE_URL to prefix CSS, JS, images, JSON, and route links.
PAGES_BASE_URL ?= /multi-echo-data-analysis

# Reproducible image outputs: matplotlib honors SOURCE_DATE_EPOCH and will stamp
# PNGs with a fixed timestamp instead of "now", so identical plots produce
# identical bytes (and identical content-hashed filenames in _images/).
# 1577836800 == 2020-01-01T00:00:00Z
export SOURCE_DATE_EPOCH := 1577836800

.PHONY: help check-env book clean serve sync-docs site-publish install runall build preview-docs

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  install     to sync the 'meda' environment from uv.lock"
	@echo "  check-env   to verify the 'meda' environment matches uv.lock"
	@echo "  book        to activate 'meda' and build the Jupyter Book into _build/"
	@echo "  clean       to clean out site build files"
	@echo "  runall      to run all notebooks in-place, capturing outputs with the notebook"
	@echo "  serve       to build and serve the site locally at http://localhost:3000"
	@echo "  build       to build the site HTML into _build/html/"
	@echo "  preview-docs to serve committed docs/ files at the GitHub Pages base path"
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
	BASE_URL=$(PAGES_BASE_URL) uv run --locked jupyter-book build --html --execute

sync-docs:
	uv run --locked python scripts/sync_docs.py

site-publish: book sync-docs

runall:
	jupyter-book run ./content

clean:
	uv run --locked python scripts/clean.py

serve: check-env
	uv run --locked jupyter-book start

build: check-env
	BASE_URL=$(PAGES_BASE_URL) uv run --locked jupyter-book build --html --execute

preview-docs:
	uv run --locked python scripts/serve_docs.py
