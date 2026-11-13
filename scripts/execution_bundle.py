"""Pack or unpack the notebook execution results that CI needs to render the book.

The bundle holds everything a build produces by *running* code, and nothing that
Sphinx can regenerate from it:

- ``_build/.jupyter_cache/`` -- executed notebooks, keyed by a hash of their code.
- ``content/figures/`` -- PNGs written by ``book_utils.glue_figure`` and referenced
  from the pages with ``{figure}`` directives.

Usage::

    python scripts/execution_bundle.py pack     # after a local `make book`
    python scripts/execution_bundle.py unpack   # in CI, before `make book`

``unpack`` refuses to overwrite an existing local cache unless ``--force`` is given.

The bundle is written to (and read from) ``_build/execution-bundle.tar.gz``.  It is
published as an asset of the ``execution-cache`` GitHub release, not committed.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "_build" / "execution-bundle.tar.gz"
MEMBERS = ["_build/.jupyter_cache", "content/figures"]


def pack() -> None:
    missing = [member for member in MEMBERS if not (ROOT / member).is_dir()]
    if missing:
        print(
            f"Missing {', '.join(missing)}. Run `make book` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with tarfile.open(BUNDLE, "w:gz") as tar:
        for member in MEMBERS:
            tar.add(ROOT / member, arcname=member)

    size_mb = BUNDLE.stat().st_size / 1e6
    print(f"Wrote {BUNDLE} ({size_mb:.1f} MB)")


def unpack(force: bool = False) -> None:
    if not BUNDLE.is_file():
        print(f"Missing {BUNDLE}.", file=sys.stderr)
        sys.exit(1)

    cache_dir = ROOT / MEMBERS[0]
    if cache_dir.exists() and not force:
        print(
            f"{cache_dir} already exists. Unpacking would replace its database and "
            "drop any local execution results that are not in the bundle. "
            "Pass --force to overwrite it.",
            file=sys.stderr,
        )
        sys.exit(1)

    with tarfile.open(BUNDLE, "r:gz") as tar:
        names = tar.getnames()
        allowed = tuple(f"{member}/" for member in MEMBERS)
        unexpected = [
            name for name in names if name not in MEMBERS and not name.startswith(allowed)
        ]
        if unexpected:
            print(f"Unexpected paths in bundle: {unexpected[:5]}", file=sys.stderr)
            sys.exit(1)

        tar.extractall(ROOT, filter="data")

    print(f"Unpacked {len(names)} entries from {BUNDLE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["pack", "unpack"])
    parser.add_argument(
        "--force",
        action="store_true",
        help="With unpack, overwrite an existing local execution cache.",
    )
    args = parser.parse_args()
    if args.action == "pack":
        pack()
    else:
        unpack(force=args.force)


if __name__ == "__main__":
    main()
