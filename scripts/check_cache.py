"""Check that every notebook in the book has executed outputs in the jupyter-cache.

The book is built with ``execute_notebooks: cache``.  Notebook outputs are produced
locally (the tutorials need several GB of data and more memory than a hosted CI
runner offers), stored in ``_build/.jupyter_cache``, and shipped to CI as a release
asset (see ``scripts/execution_bundle.py``).  CI then only renders HTML.

A notebook is matched to its cached outputs by a hash of its code cells (source and
cell metadata such as tags) and its kernelspec, so editing prose never invalidates
the cache, while editing code does.  This script reads each notebook the same way
MyST-NB does and reports the ones without a match, i.e. the ones that would be
re-executed by the next build.

Exit status is 0 when every notebook is cached.  When some are not, the exit status
is 1 with ``--strict`` and 0 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path, PurePosixPath

import yaml
from jupyter_cache import get_cache
from myst_nb.core.config import NbParserConfig
from myst_nb.core.read import create_nb_reader
from myst_parser.config.main import MdParserConfig

ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = ROOT / "content"
CACHE_DIR = ROOT / "_build" / ".jupyter_cache"


def execution_exclude_patterns() -> list[str]:
    """Return the ``execute.exclude_patterns`` setting of the book."""
    config = yaml.safe_load((ROOT / "_config.yml").read_text(encoding="utf-8"))
    return list(config.get("execute", {}).get("exclude_patterns", []))


def find_notebooks() -> list[Path]:
    """Return the source files under content/ that the build looks up in the cache.

    These are the files MyST-NB treats as notebooks, minus the ones excluded from
    execution (matched the same way MyST-NB matches them).
    """
    patterns = execution_exclude_patterns()
    candidates = sorted([*CONTENT_DIR.glob("*.md"), *CONTENT_DIR.glob("*.ipynb")])
    notebooks = []
    for path in candidates:
        posix_path = PurePosixPath(path.as_posix())
        if any(posix_path.match(pattern) for pattern in patterns):
            print(f"  excluded from execution: {path.relative_to(ROOT).as_posix()}")
            continue

        if read_notebook(path) is not None:
            notebooks.append(path)

    return notebooks


def read_notebook(path: Path):
    """Read a source file as MyST-NB would, or return None if it is not a notebook."""
    text = path.read_text(encoding="utf-8")
    reader = create_nb_reader(str(path), MdParserConfig(), NbParserConfig(), text)
    if reader is None:
        return None

    return reader.read(text)


def find_stale(notebooks: list[Path]) -> list[Path]:
    """Return the notebooks whose current code has no record in the cache."""
    if not CACHE_DIR.is_dir():
        return notebooks

    cache = get_cache(str(CACHE_DIR))
    stale = []
    for path in notebooks:
        try:
            cache.match_cache_notebook(read_notebook(path))
        except KeyError:
            stale.append(path)

    return stale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any notebook is missing from the cache.",
    )
    args = parser.parse_args()

    notebooks = find_notebooks()
    stale = find_stale(notebooks)
    print(f"{len(notebooks) - len(stale)} of {len(notebooks)} notebooks are cached.")
    if not stale:
        return

    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    level = "error" if args.strict else "warning"
    for path in stale:
        rel_path = path.relative_to(ROOT).as_posix()
        message = (
            "Code cells changed since this notebook was last executed, "
            "so its outputs are not in the execution cache."
        )
        if in_actions:
            print(f"::{level} file={rel_path}::{message}")
        else:
            print(f"  stale: {rel_path}")

    print(
        "Re-execute the stale notebooks where the data are available "
        "(`make book`), then upload the cache (`make publish-cache`).",
        file=sys.stderr,
    )
    if args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
