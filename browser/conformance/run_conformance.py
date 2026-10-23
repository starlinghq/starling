"""Drive the e5 browser conformance app (E5Conformance) in headless Chromium and write one JSON result per run.

Serves the published app with ../spike/harness/serve.py (COOP/COEP on unless --no-isolation, precompressed .br/.gz,
per-path byte counts) and mounts:
    /data/tok/       $STARLING_MODELS layout holding intfloat/multilingual-e5-small/<revision>/tokenizer.json
    /data/fixtures/  model-pipeline/protected/
    /data/models/    graph directory with model-fp32.onnx and model-int8-embed.onnx
Then calls ConformanceEntryPoints.Run (or RunCacheCheck with --cache-check) through DotNet.invokeMethodAsync.

Usage:
    python run_conformance.py --publish <dotnet publish dir> --label aot --variant fp32 --provider wasm \
        --models-root ~/.cache/starlings/models --graphs <dir> --out result.json [--chrome <path>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "spike" / "harness"))

import serve  # noqa: E402
from run_spike import RssSampler, gpu_snapshot  # noqa: E402

REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
GRAPHS = {"fp32": "model-fp32.onnx", "int8embed": "model-int8-embed.onnx"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--publish", type=Path, required=True)
    parser.add_argument("--label", required=True, help="interp or aot (recorded only)")
    parser.add_argument("--variant", choices=sorted(GRAPHS), default="fp32")
    parser.add_argument("--provider", choices=["wasm", "webgpu"], default="wasm")
    parser.add_argument("--threads", type=int, default=0, help="0 = onnxruntime-web default")
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--cache-check", action="store_true", help="run the Cache Storage reload check instead")
    parser.add_argument("--models-root", type=Path, default=Path(os.environ.get("STARLING_MODELS", REPO / ".models")))
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--chrome", type=Path)
    parser.add_argument("--no-isolation", action="store_true")
    parser.add_argument("--vk-device-filter", default="0x2805")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    server = serve.start(
        args.port,
        args.publish / "wwwroot",
        args.publish / "wwwroot",  # /ort/ is unused: the packages serve onnxruntime-web under _content/.
        args.models_root,
        REPO / "model-pipeline" / "protected",
        args.graphs,
        not args.no_isolation,
    )
    base = f"http://127.0.0.1:{args.port}/"
    marker = f"/tmp/starlings-e5-conformance-{os.getpid()}"
    chrome_args: list[str] = []
    env = dict(os.environ)
    if args.provider == "webgpu":
        chrome_args += [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan,WebGPU,UseSkiaRenderer",
            "--use-angle=vulkan",
            "--use-vulkan=native",
            "--disable-vulkan-surface",
            "--ignore-gpu-blocklist",
        ]
        if args.vk_device_filter:
            env["VK_LOADER_DEVICE_ID_FILTER"] = args.vk_device_filter
    variant = "Fp32" if args.variant == "fp32" else "Int8Embed"
    model_url = f"data/models/{GRAPHS[args.variant]}"
    tokenizer_url = f"data/tok/intfloat/multilingual-e5-small/{REVISION}/tokenizer.json"
    console: list[str] = []
    result: dict = {
        "label": args.label,
        "variant": args.variant,
        "provider": args.provider,
        "isolation": not args.no_isolation,
        "gpu_before": gpu_snapshot() if args.provider == "webgpu" else None,
    }
    sampler = RssSampler(marker)
    sampler.start()
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            marker,
            executable_path=str(args.chrome) if args.chrome else None,
            headless=True,
            args=chrome_args,
            env=env,
        )
        result["browser_version"] = context.browser.version if context.browser else None
        page = context.new_page()
        page.set_default_timeout(args.timeout_s * 1000)
        # The console also streams to <out>.console.log as it happens, so a crash or hang shows the last stage reached.
        live = args.out.with_suffix(".console.log").open("w", encoding="utf-8")

        def log(line: str) -> None:
            console.append(line[:2000])
            live.write(f"{time.strftime('%H:%M:%S')} {line[:2000]}\n")
            live.flush()

        page.on("console", lambda m: log(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: log(f"pageerror: {e}"))
        page.on("crash", lambda _: log("renderer crashed"))
        urllib.request.urlopen(base + "__stats?reset=1").read()
        t0 = time.perf_counter()
        page.goto(base)
        page.wait_for_function("document.getElementById('status')?.textContent === 'ready'")
        result["startup_ms"] = (time.perf_counter() - t0) * 1000
        result["startup_download"] = json.loads(urllib.request.urlopen(base + "__stats?reset=1").read())
        result["cross_origin_isolated"] = page.evaluate("self.crossOriginIsolated")
        try:
            if args.cache_check:
                raw = page.evaluate(
                    "([v, m, t]) => DotNet.invokeMethodAsync('E5Conformance', 'RunCacheCheck', v, m, t)",
                    [variant, model_url, tokenizer_url],
                )
                result["cache_check"] = json.loads(raw)
            else:
                raw = page.evaluate(
                    "([v, m, t, b, n, x, i]) => DotNet.invokeMethodAsync('E5Conformance', 'Run', v, m, t, b, n, x, i)",
                    [variant, model_url, tokenizer_url, args.provider, args.threads, args.proxy, args.iterations],
                )
                result["run"] = json.loads(raw)
        except Exception as e:  # noqa: BLE001 - a renderer crash is a result to record, with the console up to it
            result["run"] = {"error": f"{type(e).__name__}: {e}"}
        result["run_download"] = json.loads(urllib.request.urlopen(base + "__stats?reset=1").read())
        context.close()
        live.close()
    sampler.stop.set()
    server.shutdown()
    result["peak_rss_tree_bytes"] = sampler.peak_total
    result["peak_rss_renderer_bytes"] = sampler.peak_renderer
    result["gpu_after"] = gpu_snapshot() if args.provider == "webgpu" else None
    result["console"] = console[-200:]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    error = result.get("run", {}).get("error")
    print(f"wrote {args.out}" + (f" (error: {error.splitlines()[0]})" if error else ""))


if __name__ == "__main__":
    main()
