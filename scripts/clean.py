"""Remove _build/ and docs/ (Jupyter Book build output and synced GitHub Pages HTML)."""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    targets = [root / "_build", root / "docs"]

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
