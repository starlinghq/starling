"""Smoke test of the JavaScript-host sample (ai-dotnet/samples/E5BrowserJs) through its React UI in headless Chromium.

Point it at the running page (npm run dev, or any server of dist/ with the isolation headers). It loads the selected
graph, samples the page's frame counter while loading (it advances only if the main thread is free, the reason for the
Web Worker), runs queries and records the top hits, the runtime panel and a screenshot.

Usage:
    python smoke_js_sample.py --url http://127.0.0.1:5173/ --out smoke-js.json \
        [--variant int8embed|fp32] [--chrome <path>]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

QUERIES = [
    "how long should I soak rice before cooking",
    "Wann kann ich Bücher ausleihen?",
    "earthquakes and tectonic plates",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--variant", choices=["int8embed", "fp32"], default="int8embed")
    parser.add_argument("--chrome", type=Path)
    parser.add_argument("--timeout-s", type=float, default=240)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    console: list[str] = []
    result: dict = {"url": args.url, "variant": args.variant, "queries": []}
    ticker = "() => document.querySelector('.ticker').textContent"
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=str(args.chrome) if args.chrome else None, headless=True)
        result["browser_version"] = browser.version
        page = browser.new_page()
        page.set_default_timeout(args.timeout_s * 1000)
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"[:500]))
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"[:500]))
        page.goto(args.url)
        page.wait_for_selector("button.primary")
        page.select_option("select >> nth=0", args.variant)
        started = time.perf_counter()
        page.click("button.primary")
        samples = []
        while not page.query_selector("form.query input, .card.error"):
            if time.perf_counter() - started > args.timeout_s:
                raise TimeoutError("the model did not load")
            samples.append(page.evaluate(ticker))
            page.wait_for_timeout(500)
        result["load_s"] = time.perf_counter() - started
        result["ticker_during_load"] = samples
        if page.query_selector(".card.error"):
            result["error"] = page.inner_text(".card.error")
        else:
            for query in QUERIES:
                page.fill("form.query input", query)
                page.click("form.query button")
                page.wait_for_function("q => document.querySelector('ol.hits')?.dataset.query === q", arg=query)
                hits = page.eval_on_selector_all("ol.hits li", "els => els.map(e => e.innerText)")
                result["queries"].append({"query": query, "meta": page.inner_text("p.meta"), "top": hits[:3]})
            result["diagnostics"] = page.inner_text(".diagnostics")
        page.set_viewport_size({"width": 900, "height": 1300})
        page.screenshot(path=str(args.out.with_suffix(".png")), full_page=True)
        browser.close()
    result["console"] = console[-30:]
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("console", "ticker_during_load")},
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
