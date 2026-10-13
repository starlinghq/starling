"""Copies the spike's raw results into docs/evidence/browser-wasm-spike/, without embeddings and user agents.

Usage: collect_evidence.py <run dir (run_all.sh out)> <probe/gpu dir> <variants.json> <evidence dir>
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    run, extra, variants, out = (Path(a) for a in sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(variants, out / "variants.json")
    shutil.copy(run / "load.txt", out / "load-average.txt")

    native = load(run / "native-ort-py.json")
    for v in native["variants"].values():
        v.pop("embeddings", None)
        for lat in v["latency"].values():
            lat["samples_ms"] = [round(x, 3) for x in lat["samples_ms"]]
    dump(native, out / "native-ort.json")

    drift = load(run / "browser-drift.json")
    for e in drift:
        e["load"].pop("user_agent", None)
    dump(drift, out / "browser-drift.json")

    runs = {}
    for e in drift:
        d = load(run / e["file"])
        runs[e["file"]] = {
            "latency_js": d["e5"].get("latency_js"),
            "latency_e2e": d["e5"].get("latency_e2e"),
            "run_download": {k: v for k, v in d.get("run_download", {}).items() if v["bytes"] > 100_000},
            "gpu_before": d.get("gpu_before"),
            "gpu_during": (d.get("gpu_during") or [None])[0],
            "gpu_after": d.get("gpu_after"),
            "console_warnings": [c[:300] for c in d["console"] if "VerifyEach" in c or "pageerror" in c][:3],
        }
    dump(runs, out / "browser-e5-runs.json")

    keys = ["label", "browser_version", "startup_ms", "cross_origin_isolated", "peak_rss_tree_bytes"]
    keys += ["peak_rss_renderer_bytes", "startup_download", "tokenizer"]
    tok = {"native": load(run / "tok-native.json")}
    for mode in ("interp", "aot"):
        d = load(run / f"tok-{mode}.json")
        tok[mode] = {k: d[k] for k in keys}
    dump(tok, out / "tokenizer.json")

    probes = {}
    for name in ("eg2-text-fp32", "gliner2-fp32", "qwen35-int4", "qwen35-fp32"):
        d = load(extra / f"probe-{name}.json")
        probes[name] = {k: d[k] for k in ("probe", "peak_rss_renderer_bytes", "peak_rss_tree_bytes")}
    dump(probes, out / "probes.json")

    gpu = {}
    for name, file in (
        ("vk_filter_0x2805", "gpu-probe-filter.json"),
        ("no_filter", "gpu-probe-nofilter.json"),
        ("vk_filter_with_allow_unsafe_apis", "gpu-probe-unsafe.json"),
    ):
        gpu[name] = load(extra / file)["gpu"]
    dump(gpu, out / "gpu-pinning.json")


if __name__ == "__main__":
    main()
