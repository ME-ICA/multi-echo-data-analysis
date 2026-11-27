"""Copy Jupyter Book HTML output from _build/html into docs/ for GitHub Pages.

Everything under ``_build/html`` is part of the static site except:

- ``reports/`` -- notebook execution stderr logs (not linked by the HTML; not needed to view
  the book).
- ``.buildinfo`` -- Sphinx incremental-build metadata (not used when serving static HTML).
- ``_sphinx_design_static/`` -- duplicate of assets already present under ``_static/``; no
  page references this directory.

All other paths are required for normal viewing: HTML pages, ``_static/``, ``_images/``,
``_sources/`` (linked as "view/download source" from each page), ``searchindex.js``, etc.

Sphinx incremental builds accumulate stale content-hashed files in ``_build/html/_images/``
(old images are never deleted when a figure changes).  After copying, this script prunes any
``docs/_images/`` file that is not referenced by any HTML page so that stale figures do not
build up in the repository over time.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


# Matches the basename of any file served from the _images/ directory, e.g.:
#   src="../_images/abc123.png"   href="_images/abc123.png"
_IMAGES_RE = re.compile(r"_images/([^\s\"'<>]+)")


def _prune_stale_images(dest: Path) -> list[str]:
    """Remove files from docs/_images/ that are not referenced by any HTML page.

    Returns a sorted list of the removed filenames.
    """
    images_dir = dest / "_images"
    if not images_dir.is_dir():
        return []

    referenced: set[str] = set()
    for html_file in dest.rglob("*.html"):
        text = html_file.read_text(encoding="utf-8", errors="replace")
        for m in _IMAGES_RE.finditer(text):
            referenced.add(m.group(1))

    removed = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.is_file() and img_path.name not in referenced:
            img_path.unlink()
            removed.append(img_path.name)

    return removed


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

    pruned = _prune_stale_images(dest)
    if pruned:
        print(f"Removed {len(pruned)} stale image(s) from docs/_images/:")
        for name in pruned:
            print(f"  {name}")

    print(f"Synced {src} -> {dest} (excluding reports/, .buildinfo, _sphinx_design_static/)")


if __name__ == "__main__":
    main()
