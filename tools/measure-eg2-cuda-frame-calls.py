#!/usr/bin/env python3
"""W1: measure EmbeddingGemma 2 video on CUDA with 2, 3 and 4 frames per vision call (docs/plan/embeddinggemma2-follow-up.md).

For each frame-call size this runs the CUDA video parity test (`CudaVideoParityTests.RowsMatchTheReference` and
`SessionsUseCudaWithTf32Disabled` in ai-dotnet/tests/Starlings.Models.EmbeddingGemma2.Tests.Cuda) once, with
`STARLING_EG2_CUDA_FRAMES_PER_CALL` set, and samples the GPU with nvidia-smi while it runs. It records:

- embedding max abs and min cosine against the CPU PyTorch reference (the test's own output, bound from the protected
  video-fp32-cuda tolerances; a failing bound fails the test and is reported, not hidden),
- peak GPU memory of the test process (nvidia-smi per-process `used_memory`) and of the whole GPU,
- what the other workload leaves: GPU memory used before the run starts (baseline, other processes) and the headroom
  left at the run's peak (total minus the GPU's peak used memory),
- wall clock of the test run, and the test's per-row timing line.

Nothing is changed in the package: the default stays 2 frames per call until the maintainer decides (D7).

Requirements (GPU allowance: this runs on the shared GPU, so the maintainer runs it): the CUDA 13 runtime and cuDNN 9 on
the loader path, the EmbeddingGemma 2 image graphs (STARLING_MODELS or STARLING_EMBEDDINGGEMMA2_SNAPSHOT), the video set
(STARLING_EMBEDDINGGEMMA2_VIDEOS), the reference tensors (STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS, optional) and FFmpeg 8
(STARLING_FFMPEG_DIR). The GPU is chosen by PCI bus order; the default is GPU 1, as for every CUDA run here.

    tools/measure-eg2-cuda-frame-calls.py --gpu 1 --out docs/evidence/embeddinggemma2-video/frame-calls [--frames 2 3 4]
        [--dotnet <dotnet>] [--interval-ms 200] [--settle 5]

Writes <out>/frame-calls.json (every sample and parsed value) and <out>/frame-calls.md (the table for the evidence
note), plus each run's test log. Exits non-zero if any run failed; the table still records it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "ai-dotnet/tests/Starlings.Models.EmbeddingGemma2.Tests.Cuda"
EXE = PROJECT / "bin/Release/net10.0/Starlings.Models.EmbeddingGemma2.Tests.Cuda"
TESTS = [
    "Starlings.Models.EmbeddingGemma2.Tests.Cuda.CudaVideoParityTests.SessionsUseCudaWithTf32Disabled",
    "Starlings.Models.EmbeddingGemma2.Tests.Cuda.CudaVideoParityTests.RowsMatchTheReference",
]


def smi(gpu: int, query: str, apps: bool = False) -> list[list[str]]:
    kind = "--query-compute-apps" if apps else "--query-gpu"
    out = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), f"{kind}={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def gpu_memory(gpu: int) -> tuple[int, int]:
    used, total = smi(gpu, "memory.used,memory.total")[0]
    return int(used), int(total)


class Sampler(threading.Thread):
    """Samples the GPU's used memory and the test process's used memory until stopped."""

    def __init__(self, gpu: int, interval: float, pid_of):
        super().__init__(daemon=True)
        self.gpu, self.interval, self.pid_of = gpu, interval, pid_of
        self.samples: list[dict] = []
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            t = time.monotonic()
            try:
                used, total = gpu_memory(self.gpu)
                apps = {int(p): int(m) for p, m in smi(self.gpu, "pid,used_memory", apps=True) if m.isdigit()}
            except (subprocess.CalledProcessError, ValueError):
                apps, used, total = {}, -1, -1
            pid = self.pid_of()
            self.samples.append({"t": t, "gpu_used_mib": used, "gpu_total_mib": total, "process_mib": apps.get(pid)})
            self.stop.wait(self.interval)


def parse(log: str) -> dict:
    """The values the parity test prints (EmbeddingGemma2VideoParityChecks.RowsAgainstReference)."""
    found: dict = {}
    m = re.search(r"frames per vision call: (\d+)", log)
    found["frames_per_call_reported"] = int(m.group(1)) if m else None
    m = re.search(r"rows \(default contract;.*?\) vs reference: (.*)", log)
    found["all_rows"] = m.group(1).strip() if m else None
    m = re.search(r"video and interleaved rows: (.*)", log)
    found["video_rows"] = m.group(1).strip() if m else None
    if found["all_rows"]:
        a = re.search(r"max abs ([0-9.eE+-]+)", found["all_rows"])
        c = re.search(r"min cosine ([0-9.eE+-]+)", found["all_rows"])
        found["embedding_max_abs"] = float(a.group(1)) if a else None
        found["min_cosine"] = float(c.group(1)) if c else None
    m = re.search(r"time per row: (.*)", log)
    found["time_per_row"] = m.group(1).strip() if m else None
    m = re.search(r"Total: (\d+), Errors: (\d+), Failed: (\d+), Skipped: (\d+)", log)
    found["tests"] = (
        dict(zip(("total", "errors", "failed", "skipped"), map(int, m.groups()), strict=True)) if m else None
    )
    return found


def run_one(frames: int, args, out: Path) -> dict:
    env = dict(os.environ)
    env.update(
        STARLING_CUDA_TESTS="1",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        STARLING_EG2_CUDA_FRAMES_PER_CALL=str(frames),
    )
    time.sleep(args.settle)
    baseline_used, total = gpu_memory(args.gpu)
    baseline_apps = smi(args.gpu, "pid,process_name,used_memory", apps=True)
    cmd = [str(EXE), "-noColor", "-showLiveOutput"]
    for t in TESTS:
        cmd += ["-method", t]
    log_path = out / f"frame-calls-{frames}.log"
    holder: dict = {}
    sampler = Sampler(args.gpu, args.interval_ms / 1000, lambda: holder.get("pid"))
    started = time.monotonic()
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        holder["pid"] = proc.pid
        sampler.start()
        code = proc.wait()
    wall = time.monotonic() - started
    sampler.stop.set()
    sampler.join()
    text = log_path.read_text()
    process_peak = max((s["process_mib"] or 0) for s in sampler.samples) if sampler.samples else None
    gpu_peak = max((s["gpu_used_mib"] for s in sampler.samples), default=None)
    result = {
        "frames_per_call": frames,
        "exit_code": code,
        "wall_clock_s": round(wall, 1),
        "gpu_total_mib": total,
        "baseline_used_mib_other_workload": baseline_used,
        "baseline_processes": baseline_apps,
        "process_peak_mib": process_peak,
        "gpu_peak_used_mib": gpu_peak,
        "headroom_at_peak_mib": None if gpu_peak is None else total - gpu_peak,
        "samples": sampler.samples,
        "log": log_path.name,
        **parse(text),
    }
    print(json.dumps({k: v for k, v in result.items() if k not in ("samples", "baseline_processes")}), flush=True)
    return result


def table(results: list[dict]) -> str:
    rows = [
        "| Frames per call | Embedding max abs | Min cosine | Process peak GPU memory (MiB) | GPU used before run, other workload (MiB) | Headroom at peak (MiB) | Wall clock (s) | Tests |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        tests = r.get("tests") or {}
        status = "passed" if r["exit_code"] == 0 else f"exit {r['exit_code']}"
        rows.append(
            f"| {r['frames_per_call']} | {r.get('embedding_max_abs')} | {r.get('min_cosine')} | {r['process_peak_mib']} | "
            f"{r['baseline_used_mib_other_workload']} | {r['headroom_at_peak_mib']} | {r['wall_clock_s']} | "
            f"{status} ({tests.get('total', '?')} run, {tests.get('failed', '?')} failed, {tests.get('skipped', '?')} skipped) |"
        )
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--gpu", type=int, default=1, help="GPU index in PCI bus order (default 1)")
    parser.add_argument("--frames", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dotnet", default="dotnet")
    parser.add_argument("--interval-ms", type=int, default=200)
    parser.add_argument("--settle", type=float, default=5.0, help="seconds to wait before each baseline sample")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_build:
        subprocess.run([args.dotnet, "build", str(PROJECT), "-c", "Release"], check=True, cwd=ROOT)
    results = [run_one(n, args, args.out) for n in args.frames]
    record = {
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "gpu": args.gpu,
        "gpu_name": smi(args.gpu, "name")[0][0],
        "driver": smi(args.gpu, "driver_version")[0][0],
        "repository_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "results": results,
    }
    (args.out / "frame-calls.json").write_text(json.dumps(record, indent=1) + "\n")
    (args.out / "frame-calls.md").write_text(table(results))
    print(table(results))
    return 0 if all(r["exit_code"] == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
