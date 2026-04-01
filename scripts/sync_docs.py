"""Copy Jupyter Book HTML output from _build/html into docs/ for GitHub Pages.

Everything under ``_build/html`` is part of the static site except:

- ``reports/`` — notebook execution stderr logs (not linked by the HTML; not needed to view
  the book).
- ``.buildinfo`` — Sphinx incremental-build metadata (not used when serving static HTML).
- ``_sphinx_design_static/`` — duplicate of assets already present under ``_static/``; no
  page references this directory.

All other paths are required for normal viewing: HTML pages, ``_static/``, ``_images/``,
``_sources/`` (linked as “view/download source” from each page), ``searchindex.js``, etc.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "_build" / "html"
    if not src.is_dir():
        print(
            f"Missing {src}. Run `jupyter-book build .` or `make book` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    html_root = src.resolve()

    def ignore_unneeded(path: str, names: list[str]) -> set[str]:
        """Skip dirs/files that are not needed for static GitHub Pages viewing."""
        if Path(path).resolve() != html_root:
            return set()
        skip = {"reports", "_sphinx_design_static", ".buildinfo"}
        return {n for n in names if n in skip}

    dest = root / "docs"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=ignore_unneeded)

    nojekyll = dest / ".nojekyll"
    if not nojekyll.is_file():
        nojekyll.touch()

    print(f"Synced {src} -> {dest} (excluding reports/, .buildinfo, _sphinx_design_static/)")


if __name__ == "__main__":
    main()
