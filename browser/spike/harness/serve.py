"""Static server for the browser spike: the published app plus data mounted from outside it.

Routes (longest prefix wins):
    /ort/            onnxruntime-web dist (browser/spike/node_modules/onnxruntime-web/dist, npm-pinned)
    /data/tok/       pinned tokenizer.json assets ($STARLING_MODELS layout)
    /data/fixtures/  model-pipeline/protected/
    /data/models/    graph directory (fp32, fp16, int8 ONNX files)
    /                the published app's wwwroot

Headers: Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy: require-corp (cross-origin
isolation, needed for SharedArrayBuffer and so for onnxruntime-web threads) unless --no-isolation. A request that
accepts br (or gzip) gets the publish output's precompressed .br (.gz) sibling, as a production host would serve
it. /__stats returns the bytes sent per path since the last /__stats?reset=1.
"""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".wasm": "application/wasm",
    ".css": "text/css",
    ".dat": "application/octet-stream",
    ".onnx": "application/octet-stream",
    ".pdb": "application/octet-stream",
    ".map": "application/json",
}


class SpikeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, mounts: list[tuple[str, Path]], isolation: bool):
        super().__init__(address, SpikeHandler)
        self.mounts = sorted(mounts, key=lambda m: -len(m[0]))
        self.isolation = isolation
        self.sent: dict[str, dict] = {}
        self.lock = threading.Lock()


class SpikeHandler(BaseHTTPRequestHandler):
    server: SpikeServer

    def log_message(self, format, *args):  # noqa: A002 - signature from the base class
        pass

    def _headers(self, status: int, content_type: str, length: int, encoding: str | None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        if self.server.isolation:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.end_headers()

    def _resolve(self, path: str) -> Path | None:
        for prefix, root in self.server.mounts:
            if path.startswith(prefix):
                rel = path[len(prefix) :] or "index.html"
                target = (root / rel).resolve()
                if root.resolve() in target.parents or target == root.resolve():
                    return target
                return None
        return None

    def do_GET(self):  # noqa: N802 - http.server naming
        url = urlparse(self.path)
        path = unquote(url.path)
        if path == "/__stats":
            with self.server.lock:
                body = json.dumps(self.server.sent).encode()
                if parse_qs(url.query).get("reset"):
                    self.server.sent = {}
            self._headers(HTTPStatus.OK, "application/json", len(body), None)
            self.wfile.write(body)
            return
        target = self._resolve(path)
        if target is None or not target.is_file():
            self._headers(HTTPStatus.NOT_FOUND, "text/plain", 0, None)
            with self.server.lock:
                self.server.sent.setdefault(path, {"bytes": 0, "uncompressed": 0, "status": 404})
            return
        accept = self.headers.get("Accept-Encoding", "")
        encoding = None
        source = target
        if not path.startswith("/data/"):
            for enc, suffix in (("br", ".br"), ("gzip", ".gz")):
                candidate = target.with_name(target.name + suffix)
                if enc in accept and candidate.is_file():
                    encoding, source = enc, candidate
                    break
        content_type = TYPES.get(target.suffix, "application/octet-stream")
        size = source.stat().st_size
        self._headers(HTTPStatus.OK, content_type, size, encoding)
        with source.open("rb") as stream:
            while chunk := stream.read(1 << 20):
                self.wfile.write(chunk)
        with self.server.lock:
            entry = self.server.sent.setdefault(path, {"bytes": 0, "uncompressed": 0, "encoding": encoding})
            entry["bytes"] += size
            entry["uncompressed"] += target.stat().st_size


def start(port: int, wwwroot: Path, ort_dist: Path, tok: Path, fixtures: Path, models: Path, isolation: bool):
    mounts = [
        ("/ort/", ort_dist),
        ("/data/tok/", tok),
        ("/data/fixtures/", fixtures),
        ("/data/models/", models),
        ("/", wwwroot),
    ]
    server = SpikeServer(("127.0.0.1", port), mounts, isolation)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--wwwroot", type=Path, required=True)
    parser.add_argument("--ort-dist", type=Path, required=True)
    parser.add_argument("--tok", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--no-isolation", action="store_true")
    args = parser.parse_args()
    server = start(args.port, args.wwwroot, args.ort_dist, args.tok, args.fixtures, args.models, not args.no_isolation)
    print(f"serving on http://127.0.0.1:{args.port}/")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
