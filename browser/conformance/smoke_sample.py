"""Smoke test of the E5BrowserSearch sample (ai-dotnet/samples/E5BrowserSearch) through its UI in headless Chromium.

Serves a `dotnet publish` of the sample with COOP/COEP (../spike/harness/serve.py) and, for the self-hosted variant,
a graph directory under /models/. Picks the graph and backend, clicks Load, waits for the search box, runs each query
and records the top hits, the diagnostics panel and a screenshot. --variant fp32 loads the graph and tokenizer from the
Hugging Face Hub (network), exactly as a user of the sample would.

Usage:
    python smoke_sample.py --publish <publish dir> --graphs <dir with model-int8-embed.onnx> --variant int8embed \
        --out smoke.json --screenshot smoke.png [--chrome <path>] [--query "..."]...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spike" / "harness"))

import serve  # noqa: E402

DEFAULT_QUERIES = [
    "how long should I soak rice before cooking",
    "Wann kann ich Bücher ausleihen?",
    "earthquakes and tectonic plates",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--publish", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--variant", choices=["fp32", "int8embed"], default="int8embed")
    parser.add_argument("--provider", choices=["wasm", "webgpu"], default="wasm")
    parser.add_argument("--chrome", type=Path)
    parser.add_argument("--query", action="append")
    parser.add_argument("--no-isolation", action="store_true")
    parser.add_argument("--port", type=int, default=8785)
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    mounts = [("/models/", args.graphs), ("/", args.publish / "wwwroot")]
    server = serve.SpikeServer(("127.0.0.1", args.port), mounts, not args.no_isolation)
    import threading

    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{args.port}/"
    console: list[str] = []
    result: dict = {"variant": args.variant, "provider": args.provider, "queries": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=str(args.chrome) if args.chrome else None, headless=True)
        page = browser.new_page()
        page.set_default_timeout(args.timeout_s * 1000)
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"[:1000]))
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"[:1000]))
        try:
            drive(page, base, args, result)
        except Exception as e:  # noqa: BLE001 - record where the UI got stuck
            result["error"] = f"{type(e).__name__}: {e}"[:2000]
            result["page_text"] = page.inner_text("body")[:4000]
        if args.screenshot:
            page.set_viewport_size({"width": 900, "height": 1400})
            page.screenshot(path=str(args.screenshot), full_page=True)
        browser.close()
    server.shutdown()
    result["console"] = console[-50:]
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "console"}, indent=1, ensure_ascii=False))


def drive(page, base: str, args, result: dict) -> None:
    page.goto(base)
    page.wait_for_selector("button.primary")
    page.select_option("select >> nth=0", "Fp32" if args.variant == "fp32" else "Int8Embed")
    if args.variant == "int8embed":
        page.fill("input", "models/model-int8-embed.onnx")
    page.select_option("select >> nth=1", "WebGpu" if args.provider == "webgpu" else "Wasm")
    t0 = time.perf_counter()
    page.click("button.primary")
    page.wait_for_selector("form.query input, .card.error")
    result["load_s"] = time.perf_counter() - t0
    if page.query_selector(".card.error"):
        result["error"] = page.inner_text(".card.error")
    else:
        for query in args.query or DEFAULT_QUERIES:
            page.fill("form.query input", query)
            page.click("form.query button")
            page.wait_for_function(
                # The list names the query it answers (data-query), so earlier hits never pass for this query's.
                "q => document.querySelector('ol.hits')?.dataset.query === q",
                arg=query,
            )
            hits = page.eval_on_selector_all("ol.hits li", "els => els.map(e => e.innerText)")
            meta = page.inner_text("p.meta")
            result["queries"].append({"query": query, "meta": meta, "top": hits[:3]})
    result["diagnostics"] = page.inner_text(".diagnostics") if page.query_selector(".diagnostics") else None


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
