"""Serve the built docs/ site at the same base path used by GitHub Pages."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


BASE_PATH = "/multi-echo-data-analysis"


class DocsHandler(SimpleHTTPRequestHandler):
    """Serve docs/ under BASE_PATH for local previewing."""

    def translate_path(self, path: str) -> str:
        path = unquote(path.split("?", 1)[0].split("#", 1)[0])
        if path == "/":
            path = f"{BASE_PATH}/"
        if path == BASE_PATH:
            path = f"{BASE_PATH}/"
        if path.startswith(f"{BASE_PATH}/"):
            path = path[len(BASE_PATH) :]
        return super().translate_path(path)

    def do_GET(self) -> None:
        if self.path == BASE_PATH:
            self.send_response(301)
            self.send_header("Location", f"{BASE_PATH}/")
            self.end_headers()
            return
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    docs = root / "docs"
    if not (docs / "index.html").is_file():
        raise SystemExit("Missing docs/index.html. Run `make build` first.")

    handler = partial(DocsHandler, directory=str(docs))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}{BASE_PATH}/"
    print(f"Serving {docs} at {url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
