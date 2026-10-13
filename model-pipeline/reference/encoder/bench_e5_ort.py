"""Exploratory matched-backend timing for e5-small ONNX (toward P-008). Not a qualified benchmark.

Three subcommands:

- `workloads`: writes a workload file from the protected `embeddings.jsonl` ids (tokenization excluded).
  `short`: the single-text cases under 512 tokens in file order, cycled to the batch size and right-padded
  with id 1 to the longest row. `long512`: the 512-token `long-english` ids repeated. Batch sizes 1, 8, 32.
- `run`: times Python ONNX Runtime `InferenceSession.run` on that file (NumPy in, NumPy out, so the
  device-to-host copy and GPU synchronization are inside every timed call). Raw samples are written.
- `report`: prints a Markdown table (median, p5, p95, IQR, sample count) from several result files,
  including the C# probe's, which uses the same workload file and result layout.

Usage:
    python bench_e5_ort.py workloads --fixtures <protected onnx-fp32-cpu dir> --out workloads.json
    python bench_e5_ort.py run --model <onnx> --workloads workloads.json --provider cpu|cuda \
        [--threads N] [--tf32 0|1] --out result.json
    python bench_e5_ort.py report result1.json result2.json ...
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

PAD_ID = 1
BATCHES = [1, 8, 32]
SAMPLES = {"short": {1: 300, 8: 200, 32: 100}, "long512": {1: 50, 8: 20, 32: 10}}
WARMUP = 10


def make_workloads(fixtures_dir: Path, out: Path) -> None:
    lines = (fixtures_dir / "embeddings.jsonl").read_text(encoding="utf-8").split("\n")
    rows = [json.loads(line) for line in lines if line.strip()]
    short = [r for r in rows if len(r["input_ids"]) < 512]
    long_row = next(r for r in rows if r["id"] == "long-english")
    workloads = []
    for name, source in (("short", short), ("long512", [long_row])):
        for b in BATCHES:
            picked = list(itertools.islice(itertools.cycle(source), b))
            workloads.append(
                {
                    "name": f"{name}-b{b}",
                    "ids": [r["id"] for r in picked],
                    "rows": [r["input_ids"] for r in picked],
                    "samples": SAMPLES[name][b],
                }
            )
    spec = {
        "source": "protected onnx-fp32-cpu/embeddings.jsonl input_ids",
        "padding": "right, id 1, to the longest row",
        "warmup": WARMUP,
        "workloads": workloads,
    }
    out.write_text(json.dumps(spec) + "\n", encoding="utf-8")


def gpu_snapshot() -> list[str] | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,clocks.sm,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except OSError:
        return None


def run(args: argparse.Namespace) -> None:
    import onnxruntime as ort

    spec = json.loads(args.workloads.read_text())
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if args.threads:
        opts.intra_op_num_threads = args.threads
        opts.inter_op_num_threads = 1
    if args.provider == "cuda":
        ort.preload_dlls()  # CUDA 13 and cuDNN from the nvidia-* wheels, the same libraries the C# probe loads
        providers = [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": args.tf32}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.model), opts, providers=providers)
    if args.provider == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise SystemExit(f"CUDA EP not active: {session.get_providers()}")
    gpu_before = gpu_snapshot()
    results = []
    for w in spec["workloads"]:
        width = max(len(r) for r in w["rows"])
        ids = np.array([r + [PAD_ID] * (width - len(r)) for r in w["rows"]], dtype=np.int64)
        mask = np.array([[1] * len(r) + [0] * (width - len(r)) for r in w["rows"]], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        for _ in range(spec["warmup"]):
            session.run(["last_hidden_state"], feeds)
        times = []
        for _ in range(w["samples"]):
            start = time.perf_counter_ns()
            session.run(["last_hidden_state"], feeds)
            times.append((time.perf_counter_ns() - start) / 1e6)
        results.append(
            {
                "name": w["name"],
                "batch": len(w["rows"]),
                "width": width,
                "samples": w["samples"],
                "warmup": spec["warmup"],
                "ms": times,
            }
        )
        print(f"{w['name']}: median {np.median(times):.3f} ms (n={w['samples']})")
    out = {
        "harness": "Python bench_e5_ort.py: InferenceSession.run with NumPy feeds, NumPy output (device sync "
        "included); tokenization excluded",
        "environment": {
            "provider": args.provider,
            "intra_op_num_threads": args.threads,
            "use_tf32": args.tf32 if args.provider == "cuda" else None,
            "session_providers": session.get_providers(),
            "provider_options": session.get_provider_options(),
            "ort_version": ort.__version__,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_before": gpu_before,
            "gpu_after": gpu_snapshot(),
        },
        "results": results,
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def report(paths: list[Path]) -> None:
    print("| Harness | Provider | Threads | Workload | Batch × width | n | Median ms | p5–p95 ms | IQR ms |")
    print("| --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: |")
    for path in paths:
        data = json.loads(path.read_text())
        env = data["environment"]
        harness = "C#" if data["harness"].startswith("C#") else "Python"
        for r in data["results"]:
            ms = np.array(r["ms"])
            q = np.quantile(ms, [0.05, 0.25, 0.5, 0.75, 0.95])
            print(
                f"| {harness} | {env['provider']} | {env.get('intra_op_num_threads') or 'default'} | {r['name']} "
                f"| {r['batch']} × {r['width']} | {r['samples']} | {q[2]:.2f} | {q[0]:.2f}–{q[4]:.2f} "
                f"| {q[3] - q[1]:.2f} |"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("workloads")
    w.add_argument("--fixtures", type=Path, required=True)
    w.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--workloads", type=Path, required=True)
    r.add_argument("--provider", choices=["cpu", "cuda"], required=True)
    r.add_argument("--threads", type=int)
    r.add_argument("--tf32", type=int, default=0)
    r.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    if args.cmd == "workloads":
        make_workloads(args.fixtures, args.out)
    elif args.cmd == "run":
        run(args)
    else:
        report(args.paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
