"""Copy Jupyter Book HTML output from _build/html into docs/ for GitHub Pages."""

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

    dest = root / "docs"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    nojekyll = dest / ".nojekyll"
    if not nojekyll.is_file():
        nojekyll.touch()

    print(f"Synced {src} -> {dest}")


if __name__ == "__main__":
    main()
