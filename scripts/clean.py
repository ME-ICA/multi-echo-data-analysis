"""Remove Jupyter Book build output, keeping the notebook execution cache.

Everything under ``_build/`` is removed except ``_build/.jupyter_cache``, which holds
the executed notebooks.  Rebuilding that cache means re-running every tutorial, so it
is only removed with ``--all``.  ``content/figures/`` is kept for the same reason: its
PNGs are written while notebooks execute and are not recreated on a cache hit.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

KEEP = {".jupyter_cache"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also remove the execution cache and the generated figures.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    build_dir = root / "_build"
    if args.all:
        targets = [build_dir, root / "content" / "figures"]
    elif build_dir.is_dir():
        targets = [path for path in sorted(build_dir.iterdir()) if path.name not in KEEP]
    else:
        targets = []

    if not targets:
        print("Nothing to remove.")

    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
            print(f"Removed {path}")
        elif path.exists():
            path.unlink()
            print(f"Removed {path}")
        else:
            print(f"Skip (not present): {path}")


if __name__ == "__main__":
    main()
