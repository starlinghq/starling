"""Matched-backend benchmark runner for e5-small on Python ONNX Runtime (P-008).

The Python side of the C#-versus-Python comparison in plan section 11: same ONNX Runtime version, same model
file, same thread settings, same workload file and fixture inputs as the C# harness
(`ai-dotnet/benchmarks/Starlings.Benchmarks.E5Harness`). Both write the raw result format
`starlings-bench-raw/1` (`model-pipeline/schemas/benchmark-raw.schema.json`), one JSON object per line; summarize
with `model-pipeline bench summarize`.

Subcommands:

- `warm`: one session per process; for every workload, path (`ids`, `text`) and concurrency level: warmup
  calls, then `n` timed calls. Each timed call records its total and its stages: `tokenize` (text path only,
  one `Tokenizer.encode` per text, sequential like the C# tokenizer), `prep` (NumPy padding to the longest
  row), `run` (`InferenceSession.run` with NumPy feeds and a NumPy output, so the device-to-host copy and GPU
  synchronization are inside it) and `pool` (NumPy float32 mean over the mask, L2 normalization).
- `cold`: spawns fresh processes (`cold-child`), each loading the tokenizer and model and embedding one text.
  The parent records spawn-to-result and spawn-to-exit wall times; the child reports its stages.

Every run also measures quality on the same run: the 46 protected fixture cases embedded from ids and from
text, and every workload's outputs, against the protected reference embeddings (max abs, min cosine).

Usage (see `ai-dotnet/benchmarks/README.md` for the full protocol):
    python bench_e5_matched.py warm --snapshot <dir> --provider cpu --threads 6 --out raw.jsonl
    python bench_e5_matched.py cold --snapshot <dir> --provider cuda --threads 6 --processes 20 --out raw.jsonl
"""

from __future__ import annotations

import time

_T_MAIN_NS = time.monotonic_ns()  # before the heavy imports, for cold-start stages

import argparse  # noqa: E402
import contextlib  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

SCHEMA = "starlings-bench-raw/1"
HARNESS = "bench_e5_matched.py"
HARNESS_VERSION = "1"
REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "model-pipeline" / "protected" / "multilingual-e5-small" / "onnx-fp32-cpu"
DEFAULT_WORKLOADS = REPO / "ai-dotnet" / "benchmarks" / "workloads" / "e5-v1.json"
CPU_FLAGS = [
    "sse4_2",
    "avx",
    "avx2",
    "fma",
    "f16c",
    "avx512f",
    "avx512bw",
    "avx512vl",
    "avx512_vnni",
    "avx512_bf16",
    "avx_vnni",
    "amx_tile",
    "amx_bf16",
    "amx_int8",
]
SPAWN_ENV = "STARLING_BENCH_SPAWN_MONOTONIC_NS"


# ----------------------------------------------------------------------------------------------- inputs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_fixtures() -> dict[str, dict]:
    """Case id -> {"ids", "text", "embedding"} from the protected CPU fixtures (read only)."""
    cases: dict[str, dict] = {}
    for line in (FIXTURES / "embeddings.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            cases[row["id"]] = {"ids": row["input_ids"], "embedding": row["embedding"]}
    for line in (FIXTURES / "tokenizer.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            if row["id"] in cases:
                cases[row["id"]]["text"] = row["text"]
    return cases


def read_tolerance(fixture_dir: Path) -> float:
    inside = False
    for raw in (fixture_dir / "tolerances.yaml").read_text(encoding="utf-8").split("\n"):
        line = raw.split("#")[0].rstrip()
        if not line:
            continue
        if not line[0].isspace():
            inside = line == "float:"
            continue
        key, _, value = line.strip().partition(":")
        if inside and key == "embedding_max_abs":
            return float(value)
    raise SystemExit(f"float.embedding_max_abs missing in {fixture_dir / 'tolerances.yaml'}")


# ----------------------------------------------------------------------------------------- environment


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def gpu_load() -> list[dict] | None:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,pci.bus_id,utilization.gpu,memory.used,memory.total,clocks.sm,pstate,"
            "power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if out is None:
        return None
    keys = ["index", "name", "pci_bus_id", "util_pct", "mem_used_mib", "mem_total_mib", "sm_clock_mhz", "pstate"]
    keys += ["power_w", "temp_c"]
    return [dict(zip(keys, [v.strip() for v in line.split(",")], strict=False)) for line in out.splitlines()]


def load_snapshot() -> dict:
    loadavg = (_read("/proc/loadavg") or "").split()
    return {"loadavg": [float(x) for x in loadavg[:3]] if loadavg else None, "gpus": gpu_load()}


def cpu_info() -> dict:
    text = _read("/proc/cpuinfo") or ""
    model, flags, cores = None, set(), set()
    phys = core = None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "model name" and model is None:
            model = value
        elif key == "flags" and not flags:
            flags = set(value.split())
        elif key == "physical id":
            phys = value
        elif key == "core id":
            core = value
            cores.add((phys, core))
    mem = _read("/proc/meminfo") or ""
    ram = next((int(line.split()[1]) * 1024 for line in mem.splitlines() if line.startswith("MemTotal:")), None)
    return {
        "cpu_model": model or platform.processor(),
        "cpu_flags": [f for f in CPU_FLAGS if f in flags],
        "physical_cores": len(cores) or None,
        "logical_cpus": os.cpu_count(),
        "ram_bytes": ram,
    }


def os_name() -> str:
    release = _read("/etc/os-release") or ""
    for line in release.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip('"')
    return platform.platform()


def loaded_native_libraries() -> list[str]:
    maps = _read("/proc/self/maps") or ""
    names = ("libonnxruntime", "libcudart", "libcublas", "libcudnn", "libcufft", "libcurand")
    paths = {line.split()[-1] for line in maps.splitlines() if any(n in line for n in names) and "/" in line}
    return sorted(paths)


def gpu_identity() -> dict | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip().isdigit():
        return None
    out = _run(
        [
            "nvidia-smi",
            "-i",
            visible.strip(),
            "--query-gpu=index,name,pci.bus_id,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return None
    keys = ["index", "name", "pci_bus_id", "driver_version", "memory_total_mib", "compute_capability"]
    gpu = dict(zip(keys, [v.strip() for v in out.splitlines()[0].split(",")], strict=False))
    header = _run(["nvidia-smi"]) or ""
    match = re.search(r"CUDA (?:UMD )?Version:\s*([0-9.]+)", header)  # "UMD" in newer drivers
    if match:
        gpu["cuda_driver_api_version"] = match.group(1)
    return gpu


def environment(args: argparse.Namespace, ort, session, load_before: dict) -> dict:
    env = cpu_info()
    env.update(
        {
            "os": os_name(),
            "kernel": platform.release(),
            "runtime": f"{platform.python_implementation()} {platform.python_version()}",
            "ort_version": ort.__version__,
            "ort_available_providers": ort.get_available_providers(),
            "session_providers": session.get_providers() if session is not None else None,
            "gpu": gpu_identity() if args.provider == "cuda" else None,
            "native_libraries": loaded_native_libraries(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "energy_performance_preference": _read(
                "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"
            ),
            "no_turbo": _read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
            "load_before": load_before,
            "load_after": load_snapshot(),
            "machine_note": args.note,
        }
    )
    return env


def package_versions() -> dict:
    result = {}
    for dist in ("onnxruntime", "onnxruntime-gpu", "numpy", "tokenizers"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            result[dist] = importlib.metadata.version(dist)
    return result


def git_state() -> tuple[str | None, bool | None]:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=REPO)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO)
    return commit, (bool(status) if status is not None else None)


def artifacts(args: argparse.Namespace) -> dict:
    model = args.snapshot / "onnx" / "model.onnx"
    tokenizer = args.snapshot / "tokenizer.json"
    commit, dirty = git_state()
    return {
        "model_path": str(model),
        "model_sha256": sha256_file(model),
        "tokenizer_sha256": sha256_file(tokenizer),
        "workload_file": str(args.workloads.relative_to(REPO)) if args.workloads.is_relative_to(REPO) else None,
        "workload_sha256": sha256_file(args.workloads),
        "fixture_sha256": {name: sha256_file(FIXTURES / name) for name in ("embeddings.jsonl", "tokenizer.jsonl")},
        "git_commit": commit,
        "git_dirty": dirty,
        "packages": package_versions(),
    }


# --------------------------------------------------------------------------------------------- pipeline


class Pipeline:
    """Tokenizer + session + NumPy pre/post-processing, with per-stage timing."""

    def __init__(self, args: argparse.Namespace, ort, np, tokenizers):
        self.np = np
        t0 = time.perf_counter_ns()
        self.tokenizer = tokenizers.Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        self.tokenizer.enable_truncation(512)  # right truncation, like E5Tokenizer
        self.tokenizer.no_padding()
        t1 = time.perf_counter_ns()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if args.threads:
            opts.intra_op_num_threads = args.threads
        if args.inter_threads:
            opts.inter_op_num_threads = args.inter_threads
        if args.provider == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(args.snapshot / "onnx" / "model.onnx"), opts, providers=providers)
        if args.provider == "cuda" and self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise SystemExit(f"CUDA EP not active: {self.session.get_providers()}")
        t2 = time.perf_counter_ns()
        self.tokenizer_load_ms = (t1 - t0) / 1e6
        self.session_create_ms = (t2 - t1) / 1e6
        self.settings = {
            "provider": args.provider,
            "intra_op_num_threads": args.threads,
            "inter_op_num_threads": args.inter_threads,
            "execution_mode": "sequential",
            "graph_optimization_level": "all",
            "provider_options": {k: dict(v) for k, v in self.session.get_provider_options().items()},
            "pad_id": 1,
            "output_location": "host",
            "max_batch_size": 32,
        }

    def embed(self, request: list[dict], path: str) -> tuple[object, dict[str, float]]:
        np = self.np
        stages: dict[str, float] = {}
        t = time.perf_counter_ns()
        if path == "text":
            rows = [self.tokenizer.encode(c["text"]).ids for c in request]
            t1 = time.perf_counter_ns()
            stages["tokenize"] = (t1 - t) / 1e6
            t = t1
        else:
            rows = [c["ids"] for c in request]
        width = max(len(r) for r in rows)
        ids = np.full((len(rows), width), 1, dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = r
            mask[i, : len(r)] = 1
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        t1 = time.perf_counter_ns()
        stages["prep"] = (t1 - t) / 1e6
        (hidden,) = self.session.run(["last_hidden_state"], feeds)
        t2 = time.perf_counter_ns()
        stages["run"] = (t2 - t1) / 1e6
        m = mask[:, :, None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
        out = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        stages["pool"] = (time.perf_counter_ns() - t2) / 1e6
        return out, stages


def compare(np, actual, expected_rows: list[list[float]]) -> dict:
    exp = np.asarray(expected_rows, dtype=np.float64)
    act = np.asarray(actual, dtype=np.float64)
    cos = (act * exp).sum(axis=1) / (np.linalg.norm(act, axis=1) * np.linalg.norm(exp, axis=1))
    return {
        "max_abs": float(np.abs(act - exp).max()),
        "min_cosine": float(cos.min()),
        "finite": bool(np.isfinite(act).all()),
    }


def merge_quality(np, parts: list[dict]) -> dict:
    return {
        "max_abs": max(p["max_abs"] for p in parts),
        "min_cosine": min(p["min_cosine"] for p in parts),
        "finite": all(p["finite"] for p in parts),
    }


def run_quality(pipe: Pipeline, cases: dict[str, dict], tolerance: float, fixture_dir: Path) -> dict:
    np = pipe.np
    ordered = list(cases.values())
    result: dict = {
        "reference": str((FIXTURES / "embeddings.jsonl").relative_to(REPO)),
        "tolerance_file": str((fixture_dir / "tolerances.yaml").relative_to(REPO)),
        "embedding_max_abs_tolerance": tolerance,
        "cases": len(ordered),
    }
    for path in ("ids", "text"):
        parts = [compare(np, pipe.embed([c], path)[0], [c["embedding"]]) for c in ordered]
        result[f"{path}_single"] = merge_quality(np, parts)
    batch_parts = []
    for start in range(0, len(ordered), 16):
        chunk = ordered[start : start + 16]
        batch_parts.append(compare(np, pipe.embed(chunk, "ids")[0], [c["embedding"] for c in chunk]))
    result["ids_batch16"] = merge_quality(np, batch_parts)
    result["pass"] = all(
        result[k]["max_abs"] <= tolerance and result[k]["finite"] for k in ("ids_single", "text_single", "ids_batch16")
    )
    return result


# ----------------------------------------------------------------------------------------- measurement


def resolve(spec: dict, cases: dict[str, dict]) -> list[dict]:
    workloads = []
    for w in spec["workloads"]:
        requests = [[cases[cid] for cid in req] for req in w["requests"]]
        workloads.append({**w, "resolved": requests})
    return workloads


def measure_series(pipe: Pipeline, w: dict, path: str, concurrency: int, n: int, warmup: int) -> dict:
    requests = w["resolved"]
    for i in range(max(warmup, len(requests))):
        pipe.embed(requests[i % len(requests)], path)
    samples: list[tuple[float, dict[str, float]] | None] = [None] * n
    counter = iter(range(n))
    lock = threading.Lock()

    def worker(offset: int) -> None:
        while True:
            with lock:
                i = next(counter, None)
            if i is None:
                return
            req = requests[(i + offset) % len(requests)]
            t = time.perf_counter_ns()
            _, stages = pipe.embed(req, path)
            samples[i] = ((time.perf_counter_ns() - t) / 1e6, stages)

    start = time.perf_counter_ns()
    if concurrency == 1:
        worker(0)
    else:
        threads = [threading.Thread(target=worker, args=(k,)) for k in range(concurrency)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    wall_ms = (time.perf_counter_ns() - start) / 1e6
    stage_names = list(samples[0][1].keys())
    parts = [compare(pipe.np, pipe.embed(req, path)[0], [c["embedding"] for c in req]) for req in requests]
    return {
        "path": path,
        "workload": w["name"],
        "input": w["input"],
        "batch": w["batch"],
        "concurrency": concurrency,
        "warmup_calls": max(warmup, len(requests)),
        "n": n,
        "wall_ms": wall_ms,
        "ms": [s[0] for s in samples],
        "stages_ms": {name: [s[1][name] for s in samples] for name in stage_names},
        "quality": merge_quality(pipe.np, parts),
    }


def header(args: argparse.Namespace, mode: str, run_id: str, started: str) -> dict:
    return {
        "schema": SCHEMA,
        "kind": "run",
        "run_id": run_id,
        "started_utc": started,
        "mode": mode,
        "model": "multilingual-e5-small",
        "harness": {"name": HARNESS, "language": "python", "impl": "python-ort", "version": HARNESS_VERSION},
    }


def fixture_dir_for(provider: str) -> Path:
    return FIXTURES.parent / ("onnx-fp32-cuda" if provider == "cuda" else "onnx-fp32-cpu")


def import_runtime(provider: str):
    import numpy as np
    import onnxruntime as ort
    import tokenizers

    if provider == "cuda":
        ort.preload_dlls()  # CUDA 13 and cuDNN from the nvidia-* wheels; the C# harness loads the same files
    return ort, np, tokenizers


def cmd_warm(args: argparse.Namespace) -> int:
    load_before = load_snapshot()
    started = datetime.datetime.now(datetime.UTC).isoformat()
    ort, np, tokenizers = import_runtime(args.provider)
    cases = load_fixtures()
    spec = json.loads(args.workloads.read_text(encoding="utf-8"))
    pipe = Pipeline(args, ort, np, tokenizers)
    tolerance = read_tolerance(fixture_dir_for(args.provider))
    quality = run_quality(pipe, cases, tolerance, fixture_dir_for(args.provider))
    run_id = str(uuid.uuid4())
    series = []
    paths = args.paths.split(",")
    only = set(args.only.split(",")) if args.only else None
    for w in resolve(spec, cases):
        if only and w["name"] not in only:
            continue
        for path in paths:
            for c in w["concurrency"]:
                if c not in args.concurrency:
                    continue
                n = max(1, round(w["samples"][args.provider] * args.samples_scale))
                s = measure_series(pipe, w, path, c, n, spec["warmup_calls"])
                s.update({"schema": SCHEMA, "kind": "series", "run_id": run_id, "mode": "warm"})
                series.append(s)
                med = sorted(s["ms"])[len(s["ms"]) // 2]
                print(f"{w['name']} {path} c={c}: median {med:.3f} ms (n={n})", file=sys.stderr)
    run = header(args, "warm", run_id, started)
    run.update(
        {
            "artifacts": artifacts(args),
            "environment": environment(args, ort, pipe.session, load_before),
            "settings": pipe.settings,
            "quality": quality,
        }
    )
    passed = apply_quality_gate(quality, series, tolerance)
    write(args.out, [run, *series])
    report_failures(quality)
    return 0 if passed else 1


def cmd_cold_child(args: argparse.Namespace) -> int:
    spawn_ns = int(os.environ[SPAWN_ENV])
    t_imports = time.monotonic_ns()
    ort, np, tokenizers = import_runtime(args.provider)
    t_ready = time.monotonic_ns()
    cases = load_fixtures()
    spec = json.loads(args.workloads.read_text(encoding="utf-8"))
    request = [cases[cid] for cid in spec["cold"]["request"]]
    t_fixtures = time.monotonic_ns()
    pipe = Pipeline(args, ort, np, tokenizers)
    out, stages = pipe.embed(request, "text")
    t_done = time.monotonic_ns()
    q = compare(np, out, [c["embedding"] for c in request])
    result = {
        "spawn_to_main": (_T_MAIN_NS - spawn_ns) / 1e6,
        "runtime_init": (t_ready - t_imports) / 1e6,
        "tokenizer_load": pipe.tokenizer_load_ms,
        "session_create": pipe.session_create_ms,
        **stages,
        "spawn_to_result": (t_done - spawn_ns) / 1e6,
        "fixture_read": (t_fixtures - t_ready) / 1e6,
        "quality": q,
        "settings": pipe.settings,
    }
    print(json.dumps(result))
    return 0


def cmd_cold(args: argparse.Namespace) -> int:
    load_before = load_snapshot()
    started = datetime.datetime.now(datetime.UTC).isoformat()
    spec = json.loads(args.workloads.read_text(encoding="utf-8"))
    children = []
    walls = []
    for _ in range(args.processes):
        argv = [sys.executable, __file__, "cold-child", "--snapshot", str(args.snapshot), "--provider", args.provider]
        argv += ["--workloads", str(args.workloads)]
        if args.threads:
            argv += ["--threads", str(args.threads)]
        if args.inter_threads:
            argv += ["--inter-threads", str(args.inter_threads)]
        env = dict(os.environ)
        t0 = time.monotonic_ns()
        env[SPAWN_ENV] = str(t0)
        out = subprocess.run(argv, capture_output=True, text=True, env=env)
        walls.append((time.monotonic_ns() - t0) / 1e6)
        if out.returncode != 0:
            raise SystemExit(f"cold child failed: {out.stderr}")
        children.append(json.loads(out.stdout.strip().splitlines()[-1]))
    ort, np, _ = import_runtime(args.provider)
    run_id = str(uuid.uuid4())
    stage_names = [k for k in children[0] if k not in ("quality", "settings", "spawn_to_result")]
    series = {
        "schema": SCHEMA,
        "kind": "series",
        "run_id": run_id,
        "mode": "cold",
        "path": "text",
        "workload": "cold-start",
        "input": "short",
        "batch": len(spec["cold"]["request"]),
        "concurrency": 1,
        "warmup_calls": 0,
        "n": len(children),
        "wall_ms": sum(walls),
        "ms": [c["spawn_to_result"] for c in children],
        "stages_ms": {**{k: [c[k] for c in children] for k in stage_names}, "process_wall": walls},
        "quality": merge_quality(np, [c["quality"] for c in children]),
    }
    run = header(args, "cold", run_id, started)
    tolerance = read_tolerance(fixture_dir_for(args.provider))
    run.update(
        {
            "artifacts": artifacts(args),
            "environment": environment(args, ort, None, load_before),
            "settings": children[0]["settings"],
            "quality": {
                "reference": str((FIXTURES / "embeddings.jsonl").relative_to(REPO)),
                "tolerance_file": str((fixture_dir_for(args.provider) / "tolerances.yaml").relative_to(REPO)),
                "embedding_max_abs_tolerance": tolerance,
                "cases": len(spec["cold"]["request"]),
                "cold_first_request": series["quality"],
                "pass": True,
            },
        }
    )
    passed = apply_quality_gate(run["quality"], [series], tolerance)
    write(args.out, [run, series])
    report_failures(run["quality"])
    med = sorted(series["ms"])[len(series["ms"]) // 2]
    print(f"cold-start: median spawn-to-result {med:.1f} ms (n={len(children)})", file=sys.stderr)
    return 0 if passed else 1


def apply_quality_gate(run_quality: dict, series: list[dict], tolerance: float) -> bool:
    """The run passes only if the preliminary sweep and every timed series are within tolerance and finite.

    Marks each series' ``quality.pass`` and records ``preliminary_pass``, ``series_pass`` and ``failed_series`` in the
    run's quality block, so a failure specific to one workload (a large batch, the text path) fails the whole run and
    its timings are never presented as valid. Same rule as the C# harness (``QualityGate``).
    """
    preliminary = bool(run_quality.get("pass", False))
    failed = []
    for s in series:
        q = s["quality"]
        q["pass"] = bool(q["max_abs"] <= tolerance and q["finite"])
        if not q["pass"]:
            failed.append(f"{s['mode']}/{s['path']}/{s['workload']}/c{s['concurrency']}")
    run_quality["preliminary_pass"] = preliminary
    run_quality["series_pass"] = not failed
    run_quality["failed_series"] = failed
    run_quality["pass"] = preliminary and not failed
    return run_quality["pass"]


def report_failures(quality: dict) -> None:
    if not quality["pass"]:
        print(
            f"QUALITY FAILED: preliminary sweep pass={quality['preliminary_pass']}, failed series: "
            f"{quality['failed_series']}. Timings of this run are not valid results.",
            file=sys.stderr,
        )


def write(out: Path, records: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("warm", "cold", "cold-child"):
        p = sub.add_parser(name)
        p.add_argument("--snapshot", type=Path, default=os.environ.get("STARLING_E5_SNAPSHOT"))
        p.add_argument("--provider", choices=["cpu", "cuda"], required=True)
        p.add_argument("--threads", type=int, help="intra_op_num_threads (default: ORT default)")
        p.add_argument("--inter-threads", type=int, default=1, help="inter_op_num_threads (default 1)")
        p.add_argument("--workloads", type=Path, default=DEFAULT_WORKLOADS)
        if name != "cold-child":
            p.add_argument("--out", type=Path, required=True, help="raw JSON lines file (appended)")
            p.add_argument("--note", default=None, help="free-text machine note recorded in the environment")
        if name == "warm":
            p.add_argument("--paths", default="ids,text")
            p.add_argument("--concurrency", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4])
            p.add_argument("--only", default=None, help="comma-separated workload names")
            p.add_argument("--samples-scale", type=float, default=1.0)
        if name == "cold":
            p.add_argument("--processes", type=int, default=20)
    args = parser.parse_args()
    if args.snapshot is None:
        parser.error("--snapshot or STARLING_E5_SNAPSHOT is required")
    args.snapshot = Path(args.snapshot)
    args.workloads = args.workloads.resolve()
    return {"warm": cmd_warm, "cold": cmd_cold, "cold-child": cmd_cold_child}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
