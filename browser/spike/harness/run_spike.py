"""Drives the published browser spike headlessly with Playwright (docs/m0/browser-wasm-spike.md).

Starts serve.py in-process, launches Chromium (Playwright's pinned Chrome for Testing build, or any executable given
with --chrome) with a fresh temporary profile, and calls the app's [JSInvokable] entry points. Writes one JSON file
with the raw results, the startup download (bytes on the wire per path), console messages and the peak resident
memory of the browser's process tree.

    run_spike.py tokenizer --publish <publish dir> --label interp --out tok-interp.json
    run_spike.py e5 --publish <publish dir> --label interp --variant fp32 --backend wasm --threads 6 --out e5.json

WebGPU: --backend webgpu adds --enable-unsafe-webgpu --enable-features=Vulkan --use-vulkan=native
--disable-vulkan-surface (without the last two, headless Chromium gives WebGPU the SwiftShader CPU adapter) and runs
Chromium with VK_LOADER_DEVICE_ID_FILTER (default 0x2805, the RTX 4060 Ti) so that the Vulkan loader exposes only
that GPU. The suite `gpu` reports the adapter WebGPU gets.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import serve
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
REPO = SPIKE.parent.parent


def tree_rss(root_pid: int) -> tuple[int, int]:
    """Total RSS of the process tree under root_pid and the largest renderer's RSS, in bytes."""
    pids = [root_pid]
    i = 0
    while i < len(pids):
        try:
            for task in Path(f"/proc/{pids[i]}/task").iterdir():
                children = (task / "children").read_text().split()
                pids.extend(int(c) for c in children)
        except OSError:
            pass
        i += 1
    total = 0
    renderer = 0
    for pid in pids:
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            rss = next(int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith("VmRSS:"))
        except (OSError, StopIteration):
            continue
        total += rss
        try:
            if b"--type=renderer" in Path(f"/proc/{pid}/cmdline").read_bytes():
                renderer = max(renderer, rss)
        except OSError:
            pass
    return total, renderer


class RssSampler(threading.Thread):
    def __init__(self, marker: str):
        super().__init__(daemon=True)
        self.marker = marker.encode()
        self.peak_total = 0
        self.peak_renderer = 0
        self.stop = threading.Event()

    def find_root(self) -> int | None:
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                cmd = (proc / "cmdline").read_bytes()
            except OSError:
                continue
            if self.marker in cmd and b"--type=" not in cmd:
                return int(proc.name)
        return None

    def run(self):
        root = None
        while not self.stop.is_set():
            root = root or self.find_root()
            if root:
                total, renderer = tree_rss(root)
                self.peak_total = max(self.peak_total, total)
                self.peak_renderer = max(self.peak_renderer, renderer)
            time.sleep(0.25)


def gpu_snapshot() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "nvidia-smi unavailable"


def gpu_apps() -> str:
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite", choices=["tokenizer", "e5", "probe", "gpu"])
    parser.add_argument("--publish", type=Path, required=True, help="dotnet publish output directory")
    parser.add_argument("--label", required=True, help="interp or aot (recorded only)")
    parser.add_argument("--chrome", type=Path, help="Chromium executable (default: Playwright's)")
    parser.add_argument("--models-root", type=Path, default=Path(os.environ.get("STARLING_MODELS", REPO / ".models")))
    parser.add_argument("--graphs", type=Path, help="directory with model-<variant>.onnx")
    parser.add_argument("--variant", default="fp32")
    parser.add_argument("--probe-graph", help="probe: graph path under --graphs")
    parser.add_argument("--probe-data", help="probe: external data path under --graphs (optional)")
    parser.add_argument("--backend", default="wasm", choices=["wasm", "webgpu"])
    parser.add_argument("--threads", type=int, default=0, help="onnxruntime-web numThreads (0 = its default)")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--encode-seconds", type=float, default=3.0)
    parser.add_argument("--no-isolation", action="store_true")
    parser.add_argument("--vk-device-filter", default="0x2805")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--chrome-arg", action="append", default=[], help="extra Chromium switch (repeatable)")
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    server = serve.start(
        args.port,
        args.publish / "wwwroot",
        SPIKE / "node_modules" / "onnxruntime-web" / "dist",
        args.models_root,
        REPO / "model-pipeline" / "protected",
        args.graphs or args.models_root,
        not args.no_isolation,
    )
    base = f"http://127.0.0.1:{args.port}/"
    marker = f"/tmp/df-browser-spike-{os.getpid()}"
    chrome_args = [f"--user-data-dir={marker}"]
    env = dict(os.environ)
    if args.backend == "webgpu":
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
    chrome_args += args.chrome_arg
    console: list[str] = []
    result: dict = {
        "suite": args.suite,
        "label": args.label,
        "isolation": not args.no_isolation,
        "gpu_before": gpu_snapshot(),
    }
    sampler = RssSampler(marker)
    sampler.start()
    with sync_playwright() as p:
        # launch_persistent_context so the profile directory (and so the process tree) is identifiable; it is
        # a fresh temporary directory, never the user's profile.
        context = p.chromium.launch_persistent_context(
            marker,
            executable_path=str(args.chrome) if args.chrome else None,
            headless=True,
            args=chrome_args[1:],
            env=env,
        )
        result["browser_version"] = context.browser.version if context.browser else None
        page = context.new_page()
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"[:2000]))
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"[:2000]))
        urllib.request.urlopen(base + "__stats?reset=1").read()
        t0 = time.perf_counter()
        page.goto(base)
        page.wait_for_function(
            "document.getElementById('status')?.textContent === 'ready'", timeout=args.timeout_s * 1000
        )
        result["startup_ms"] = (time.perf_counter() - t0) * 1000
        result["startup_download"] = json.loads(urllib.request.urlopen(base + "__stats?reset=1").read())
        result["cross_origin_isolated"] = page.evaluate("self.crossOriginIsolated")
        if args.suite == "tokenizer":
            raw = page.evaluate(
                "([s, w]) => DotNet.invokeMethodAsync('BrowserSpike', 'RunTokenizerSuite', s, w)",
                [args.encode_seconds, 3],
            )
            result["tokenizer"] = json.loads(raw)
        elif args.suite == "gpu":
            result["gpu"] = page.evaluate(
                """async () => {
                    if (!navigator.gpu) return {gpu: false};
                    const t = new Promise(r => setTimeout(() => r('timeout'), 30000));
                    const req = navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
                    const a = await Promise.race([req, t]);
                    if (a === 'timeout' || !a) return {gpu: true, adapter: String(a)};
                    const i = a.info || {};
                    return {gpu: true, vendor: i.vendor, architecture: i.architecture, device: i.device,
                            description: i.description, isFallback: a.isFallbackAdapter ?? null,
                            features: [...a.features].sort()};
                }"""
            )
        elif args.suite == "probe":
            page.set_default_timeout(args.timeout_s * 1000)
            data_url = f"/data/models/{args.probe_data}" if args.probe_data else None
            data_name = Path(args.probe_data).name if args.probe_data else None
            raw = page.evaluate(
                "([g, d, n, b, t]) => import('/spike-ort.js').then(m => m.probe(g, d, n, b, t))",
                [f"/data/models/{args.probe_graph}", data_url, data_name, args.backend, args.threads],
            )
            result["probe"] = json.loads(raw)
        else:
            model_url = f"/data/models/model-{args.variant}.onnx"
            result["variant"] = args.variant
            result["backend"] = args.backend
            gpu_mid: list[str] = []
            if args.backend == "webgpu":

                def watch():
                    time.sleep(8)
                    gpu_mid.append(gpu_snapshot())
                    gpu_mid.append(gpu_apps())

                threading.Thread(target=watch, daemon=True).start()
            page.set_default_timeout(args.timeout_s * 1000)
            raw = page.evaluate(
                "([u, b, t, n]) => DotNet.invokeMethodAsync('BrowserSpike', 'RunE5', u, b, t, n)",
                [model_url, args.backend, args.threads, args.iterations],
            )
            result["e5"] = json.loads(raw)
            result["gpu_during"] = gpu_mid
            result["run_download"] = json.loads(urllib.request.urlopen(base + "__stats?reset=1").read())
        context.close()
    sampler.stop.set()
    server.shutdown()
    result["peak_rss_tree_bytes"] = sampler.peak_total
    result["peak_rss_renderer_bytes"] = sampler.peak_renderer
    result["gpu_after"] = gpu_snapshot()
    result["console"] = console[-200:]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
