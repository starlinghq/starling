"""Serve a published E5BrowserSearch the way a production host should, for a local demo.

`dotnet run` works, but its development server sends no cross-origin isolation headers, so onnxruntime-web gets one
thread. This server (Python standard library only) adds them and serves the publish output's precompressed .br files:

    dotnet publish -c Release -o out
    python3 serve.py out/wwwroot [--models <dir with model-int8-embed.onnx>] [--port 8080]

then open http://127.0.0.1:8080/. With --models, the "int8-embed, self-hosted" choice loads models/model-int8-embed.onnx
from that directory. The diagnostics panel shows "N of M logical cores" when isolation is on.
"""

from __future__ import annotations

import argparse
import contextlib
import mimetypes
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TYPES = {".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json"}


class Handler(SimpleHTTPRequestHandler):
    models: Path | None = None

    def end_headers(self) -> None:
        # Cross-origin isolation: SharedArrayBuffer, so onnxruntime-web can use several threads.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        super().end_headers()

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if self.models is not None and clean.startswith("/models/"):
            target = (self.models / clean[len("/models/") :]).resolve()
            if self.models.resolve() in target.parents:
                return str(target)
            return str(self.models / "__denied__")
        return super().translate_path(path)

    def guess_type(self, path: str) -> str:
        return TYPES.get(Path(path).suffix) or mimetypes.guess_type(path)[0] or "application/octet-stream"

    def send_head(self):
        # Serve the publish output's precompressed sibling when the browser accepts brotli, as a static host would.
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            path = path / "index.html"
        brotli = path.with_name(path.name + ".br")
        if "br" in self.headers.get("Accept-Encoding", "") and brotli.is_file():
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Encoding", "br")
            self.send_header("Content-Length", str(brotli.stat().st_size))
            self.end_headers()
            return brotli.open("rb")
        return super().send_head()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wwwroot", type=Path, help="the publish output's wwwroot directory")
    parser.add_argument("--models", type=Path, help="directory served under /models/ (for the self-hosted graph)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if not (args.wwwroot / "index.html").is_file():
        raise SystemExit(f"{args.wwwroot} has no index.html; pass the publish output's wwwroot")
    Handler.models = args.models
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(args.wwwroot)))
    print(f"E5BrowserSearch on http://127.0.0.1:{args.port}/ (cross-origin isolated)")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
