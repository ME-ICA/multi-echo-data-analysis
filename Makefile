# Name of the uv-managed project environment directory.
# `uv` honors UV_PROJECT_ENVIRONMENT to place the venv somewhere other than .venv.
export UV_PROJECT_ENVIRONMENT := meda

# Reproducible image outputs: matplotlib honors SOURCE_DATE_EPOCH and will stamp
# PNGs with a fixed timestamp instead of "now", so identical plots produce
# identical bytes (and identical content-hashed filenames in _images/).
# 1577836800 == 2020-01-01T00:00:00Z
export SOURCE_DATE_EPOCH := 1577836800

# GitHub release whose `execution-bundle.tar.gz` asset holds the notebook
# execution results that CI renders the book from.
CACHE_RELEASE := execution-cache
BUNDLE := _build/execution-bundle.tar.gz

# `make check-cache STRICT=1` fails when a notebook is missing from the cache.
STRICT ?=

.PHONY: help install check-env book check-cache pack-cache unpack-cache fetch-cache publish-cache deploy clean clean-all

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  install        to sync the 'meda' environment from uv.lock"
	@echo "  check-env      to verify the 'meda' environment matches uv.lock"
	@echo "  book           to build the book into _build/html, executing only notebooks"
	@echo "                 whose code is not in the execution cache"
	@echo "  check-cache    to list notebooks whose code is newer than the execution cache"
	@echo "  fetch-cache    to download the published execution cache and unpack it"
	@echo "  publish-cache  to pack the local execution cache and upload it for CI"
	@echo "  deploy         to ask GitHub Actions to rebuild and deploy the site from main"
	@echo "  clean          to remove build output, keeping the execution cache"
	@echo "  clean-all      to also remove the execution cache and generated figures"

install:
	uv sync --locked

# Verify that the local 'meda' environment is up-to-date with uv.lock.
# `uv sync --locked --check` exits non-zero if the environment would change.
check-env:
	@echo "Checking that '$(UV_PROJECT_ENVIRONMENT)' matches uv.lock..."
	uv sync --locked --check

# Build the book in the 'meda' environment (via `uv run`).
# Execution mode is "cache" (_config.yml): notebooks with a match in
# _build/.jupyter_cache are rendered from it, the others are executed and cached.
book: check-env
	uv run --locked jupyter-book build ./

check-cache:
	uv run --locked python scripts/check_cache.py $(if $(STRICT),--strict,)

pack-cache:
	uv run --locked python scripts/execution_bundle.py pack

unpack-cache:
	uv run --locked python scripts/execution_bundle.py unpack

# Start from the published execution results instead of re-running every notebook.
fetch-cache:
	mkdir -p _build
	gh release download $(CACHE_RELEASE) --pattern execution-bundle.tar.gz --dir _build --clobber
	$(MAKE) unpack-cache

# Upload the local execution results so CI can render the book from them.
# Refuses to upload a cache that does not cover the current notebooks.
publish-cache:
	$(MAKE) check-cache STRICT=1
	$(MAKE) pack-cache
	gh release view $(CACHE_RELEASE) > /dev/null 2>&1 || gh release create $(CACHE_RELEASE) \
		--title "Notebook execution cache" \
		--notes "Executed notebook outputs and figures used by the book workflow. Not a versioned release of the book; the asset is replaced by 'make publish-cache'." \
		--latest=false
	gh release upload $(CACHE_RELEASE) $(BUNDLE) --clobber

deploy:
	gh workflow run book.yml --ref main

clean:
	uv run --locked python scripts/clean.py

clean-all:
	uv run --locked python scripts/clean.py --all
