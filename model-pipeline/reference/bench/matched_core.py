"""Shared core of the Python matched-backend benchmark runners (G5, plan section 11).

Every Python runner is the Python side of a C#-versus-Python comparison: same ONNX Runtime (or ONNX Runtime GenAI)
version, same graph files, same thread settings, same workload file and fixture inputs, same warm/cold protocol and
the same raw result format (`starlings-bench-raw/1`, `model-pipeline/schemas/benchmark-raw.schema.json`) as the C#
harness it is compared with (`ai-dotnet/benchmarks/Core/HarnessHost.cs`). The runners:

- `model-pipeline/reference/encoder/bench_e5_matched.py` (multilingual-e5-small),
- `bench_eg2_matched.py` (EmbeddingGemma 2 text and image paths),
- `bench_gliner2_matched.py` (GLiNER2, the inference stage on the tensors the C# harness prepared),
- `bench_qwen35_matched.py` (Qwen3.5-0.8B through Python `onnxruntime-genai`).

This module holds what they share: the environment manifest, artifact digests, the timed loop, the warm, cold and
cold-child commands, the quality gate and the raw writer. A runner supplies a `Driver` (model id, artifacts, options)
whose `load` returns a `LoadedRun` (load stages, settings, preliminary quality sweep, warm series and the cold-start
series). Standard library only at import time: NumPy and the runtimes are imported by the drivers when they load.

Protocol (mirrors HarnessHost):

- `warm`: one process loads the model once, runs the preliminary quality sweep, then per series `warmup_calls`
  untimed calls and `n` timed calls cycling through the workload's requests (sample `i` on worker `k` runs request
  `(i + k) % requests`), then every request once more, judged against the reference. Writes one `run` record and one
  `series` record per series.
- `cold`: spawns `cold-child` processes, each started with its spawn time on CLOCK_MONOTONIC (`STARLING_BENCH_SPAWN_
  MONOTONIC_NS`); each child loads, runs the cold-start request once and prints its stages; the parent adds
  `process_wall` and writes one `run` and one `series` record.
- The run passes only if the preliminary sweep and every series pass; otherwise it exits 1 and its timings are not
  valid results (the summarizer excludes them).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "starlings-bench-raw/1"
REPO = Path(__file__).resolve().parents[3]
WORKLOADS = REPO / "ai-dotnet" / "benchmarks" / "workloads"
PROTECTED = REPO / "model-pipeline" / "protected"
SPAWN_ENV = "STARLING_BENCH_SPAWN_MONOTONIC_NS"
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
NATIVE_LIBRARIES = ("libonnxruntime", "libcudart", "libcublas", "libcudnn", "libcufft", "libcurand")


# ------------------------------------------------------------------------------------------------- files


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def repo_relative(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def file_digests(*paths: Path) -> dict[str, str]:
    """SHA-256 of each file, keyed by its repository-relative path (as the C# harnesses record them)."""
    return {repo_relative(p): sha256_file(p) for p in paths}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def read_tolerance(path: Path, section: str, key: str) -> float:
    """``section.key`` of a protected ``tolerances.yaml`` (flat two-level YAML; read without PyYAML)."""
    inside = False
    for raw in Path(path).read_text(encoding="utf-8").split("\n"):
        line = raw.split("#")[0].rstrip()
        if not line:
            continue
        if not line[0].isspace():
            inside = line == f"{section}:"
            continue
        name, _, value = line.strip().partition(":")
        if inside and name == key:
            return float(value)
    raise SystemExit(f"{section}.{key} missing in {path}")


# ------------------------------------------------------------------------------------------ environment


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
    phys = None
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
            cores.add((phys, value))
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


def loaded_native_libraries(names: Iterable[str] = NATIVE_LIBRARIES) -> list[str]:
    names = tuple(names)
    maps = _read("/proc/self/maps") or ""
    paths = {line.split()[-1] for line in maps.splitlines() if any(n in line for n in names) and "/" in line}
    return sorted(paths)


ORT_LIBRARY = re.compile(r"^(lib)?onnxruntime[\w.-]*\.so(\.\d+)*$")


def onnxruntime_library_paths(maps: str) -> list[str]:
    """Distinct, sorted absolute paths of the ONNX Runtime and GenAI shared objects in a ``/proc/<pid>/maps`` listing
    (``libonnxruntime.so[.1.30.0]``, ``libonnxruntime-genai.so``, ``libonnxruntime_providers_*.so``,
    ``onnxruntime_pybind11_state.*.so``; not ``onnxruntime-genai.db-shm``)."""
    paths = set()
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = fields[5].strip()
            if ORT_LIBRARY.match(Path(path).name):
                paths.add(path)
    return sorted(paths)


def file_sha256(path: str) -> str | None:
    try:
        return sha256_file(Path(path))
    except OSError:
        return None


def native_library_digests() -> list[dict] | str:
    """``environment.native_library_digests``: every ONNX Runtime and GenAI native library this process mapped, with
    its SHA-256, read from ``/proc/self/maps`` (Linux); ``"unavailable"`` elsewhere. Two runs can report the same
    ONNX Runtime version and load different native builds (pip against NuGet); the digests make that checkable."""
    if sys.platform != "linux":
        return "unavailable"
    maps = _read("/proc/self/maps")
    if maps is None:
        return "unavailable"
    return [{"path": p, "sha256": file_sha256(p)} for p in onnxruntime_library_paths(maps)]


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


def ort_identity() -> tuple[str | None, list[str]]:
    """Version and available providers of the Python ONNX Runtime package, if it is installed."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None, []
    return ort.__version__, ort.get_available_providers()


def environment(
    provider: str, note: str | None, session_providers: list[str] | None, load_before: dict, extra: dict | None = None
) -> dict:
    env = cpu_info()
    ort_version, ort_providers = ort_identity()
    env.update(
        {
            "os": os_name(),
            "kernel": platform.release(),
            "runtime": f"{platform.python_implementation()} {platform.python_version()}",
            "ort_version": ort_version,
            "ort_available_providers": ort_providers,
            "session_providers": session_providers,
            "gpu": gpu_identity() if provider == "cuda" else None,
            "native_libraries": loaded_native_libraries(),
            "native_library_digests": native_library_digests(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "energy_performance_preference": _read(
                "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"
            ),
            "no_turbo": _read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
            "load_before": load_before,
            "load_after": load_snapshot(),
            "machine_note": note,
        }
    )
    env.update(extra or {})
    return env


def package_versions(dists: Iterable[str]) -> dict[str, str]:
    result = {}
    for dist in dists:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            result[dist] = importlib.metadata.version(dist)
    return result


def git_state() -> tuple[str | None, bool | None]:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=REPO)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO)
    return commit, (bool(status) if status is not None else None)


def artifacts(
    model: Path, tokenizer: Path, workloads: Path, fixtures: dict[str, str], packages: dict, extra: dict | None = None
) -> dict:
    """The schema's artifact block (same keys as BenchEnvironment.Artifacts in C#) plus ``extra``."""
    commit, dirty = git_state()
    result = {
        "model_path": str(model),
        "model_sha256": sha256_file(model),
        "tokenizer_sha256": sha256_file(tokenizer),
        "workload_file": repo_relative(workloads),
        "workload_sha256": sha256_file(workloads),
        "fixture_sha256": fixtures,
        "git_commit": commit,
        "git_dirty": dirty,
        "packages": packages,
    }
    result.update(extra or {})
    return result


def session_settings(args: argparse.Namespace, provider_options: dict | None = None) -> dict:
    """The settings block every ONNX Runtime run records (sequential execution, all graph optimizations)."""
    return {
        "provider": args.provider,
        "intra_op_num_threads": args.threads,
        "inter_op_num_threads": args.inter_threads,
        "execution_mode": "sequential",
        "graph_optimization_level": "all",
        "provider_options": provider_options or {},
    }


def ort_session(ort, path: Path, args: argparse.Namespace):
    """An InferenceSession with the settings of HarnessContext.SessionOptions (C#): sequential, all optimizations,
    the given intra- and inter-op threads; CUDA on device 0 (of CUDA_VISIBLE_DEVICES) with TF32 off."""
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
    session = ort.InferenceSession(str(path), opts, providers=providers)
    if args.provider == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise SystemExit(f"CUDA EP not active: {session.get_providers()}")
    return session


def import_ort(provider: str):
    import onnxruntime as ort

    if provider == "cuda":
        ort.preload_dlls()  # CUDA 13 and cuDNN from the nvidia-* wheels; the C# harness loads the same files
    return ort


# --------------------------------------------------------------------------------------------- series


def ms_since(t0_ns: int) -> float:
    return (time.perf_counter_ns() - t0_ns) / 1e6


def median(values: list[float]) -> float:
    """The upper median, as the C# harness prints it (summaries use model_pipeline.benchmarks)."""
    return sorted(values)[len(values) // 2]


@dataclass
class Series:
    """One timed series. ``call(r)`` runs request ``r`` and returns ``(output, stages_ms, metrics)``; ``judge`` rates
    ``[(r, output), ...]`` and returns the series quality (with ``pass`` unless the runner's gate computes it)."""

    workload: str
    path: str
    input: str
    batch: int
    samples: int
    requests: int
    stages: list[str]
    call: Callable[[int], tuple[object, dict[str, float], dict[str, float]]]
    judge: Callable[[list[tuple[int, object]]], dict]
    concurrency: int = 1
    warmup_calls: int = 1
    metrics: list[str] = field(default_factory=list)


def measure(spec: Series, n: int) -> dict:
    """Warmup calls, ``n`` timed calls on ``spec.concurrency`` threads, then every request once more, judged."""
    for i in range(spec.warmup_calls):
        spec.call(i % spec.requests)
    samples: list[tuple[float, dict, dict] | None] = [None] * n
    counter = iter(range(n))
    lock = threading.Lock()

    def worker(offset: int) -> None:
        while True:
            with lock:
                i = next(counter, None)
            if i is None:
                return
            t = time.perf_counter_ns()
            _, stages, metrics = spec.call((i + offset) % spec.requests)
            samples[i] = (ms_since(t), stages, metrics)

    start = time.perf_counter_ns()
    if spec.concurrency == 1:
        worker(0)
    else:
        threads = [threading.Thread(target=worker, args=(k,)) for k in range(spec.concurrency)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    wall_ms = ms_since(start)
    outputs = [(r, spec.call(r)[0]) for r in range(spec.requests)]
    series = {
        "path": spec.path,
        "workload": spec.workload,
        "input": spec.input,
        "batch": spec.batch,
        "concurrency": spec.concurrency,
        "warmup_calls": spec.warmup_calls,
        "n": n,
        "wall_ms": wall_ms,
        "ms": [s[0] for s in samples],
        "stages_ms": {name: [s[1][name] for s in samples] for name in spec.stages},
    }
    if spec.metrics:
        series["metrics"] = {name: [s[2][name] for s in samples] for name in spec.metrics}
    series["quality"] = spec.judge(outputs)
    return series


def recorded_pass(quality: dict) -> bool:
    return bool(quality.get("pass", False))


def apply_quality_gate(run_quality: dict, series: list[dict], series_passes: Callable[[dict], bool]) -> bool:
    """The run passes only if the preliminary sweep and every timed series pass (QualityGate.Apply in C#).

    Marks each series' ``quality.pass`` and records ``preliminary_pass``, ``series_pass`` and ``failed_series`` in the
    run's quality block, so a failure specific to one workload fails the whole run and its timings are never presented
    as valid.
    """
    preliminary = bool(run_quality.get("pass", False))
    failed = []
    for s in series:
        q = s["quality"]
        q["pass"] = bool(series_passes(q))
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


# --------------------------------------------------------------------------------------------- drivers


class LoadedRun:
    """A model loaded in one process. Subclasses set ``load_stages`` (ms, in order), ``settings`` and
    ``session_providers`` and implement the sweep and the series."""

    load_stages: dict[str, float]
    settings: dict
    session_providers: list[str] | None = None

    def preliminary_quality(self) -> dict:
        """Must contain ``reference``, ``cases`` and ``pass``."""
        raise NotImplementedError

    def warm_series(self) -> Iterable[Series]:
        raise NotImplementedError

    def cold_series(self) -> Series:
        raise NotImplementedError


class Driver:
    model: str
    harness: str
    impl: str = "python-ort"
    version: str = "1"
    default_workloads: Path
    snapshot_env: str | None = None
    cold_skip: tuple[str, ...] = ()

    def add_arguments(self, parser: argparse.ArgumentParser, command: str) -> None:
        """Model-specific options; those of ``cold`` are forwarded to ``cold-child``."""

    def forwarded(self, args: argparse.Namespace) -> list[str]:
        """Model-specific options forwarded from ``cold`` to ``cold-child``."""
        return []

    def profile(self, args: argparse.Namespace) -> str:
        raise NotImplementedError

    def load(self, args: argparse.Namespace) -> LoadedRun:
        raise NotImplementedError

    def artifacts(self, args: argparse.Namespace) -> dict:
        raise NotImplementedError

    def reference(self, args: argparse.Namespace) -> str:
        raise NotImplementedError

    def series_passes(self, args: argparse.Namespace) -> Callable[[dict], bool]:
        """How a series' quality is rated; default: its own verdict in ``pass``."""
        return recorded_pass

    def run_quality_extra(self, args: argparse.Namespace) -> dict:
        """Fields added to every run record's quality block (e5: ``embedding_max_abs_tolerance``)."""
        return {}

    def environment_extra(self, args: argparse.Namespace) -> dict:
        return {}

    def session_providers(self, run: LoadedRun | None) -> list[str] | None:
        return run.session_providers if run is not None else None


def header(driver: Driver, args: argparse.Namespace, mode: str, run_id: str, started: str) -> dict:
    return {
        "schema": SCHEMA,
        "kind": "run",
        "run_id": run_id,
        "started_utc": started,
        "mode": mode,
        "model": driver.model,
        "profile": driver.profile(args),
        "harness": {"name": driver.harness, "language": "python", "impl": driver.impl, "version": driver.version},
    }


def cmd_warm(driver: Driver, args: argparse.Namespace) -> int:
    load_before = load_snapshot()
    started = datetime.datetime.now(datetime.UTC).isoformat()
    run = driver.load(args)
    quality = run.preliminary_quality()
    print(f"preliminary quality: pass={quality['pass']}", file=sys.stderr)
    run_id = str(uuid.uuid4())
    only = set(args.only.split(",")) if args.only else None
    series = []
    for spec in run.warm_series():
        if only and spec.workload not in only:
            continue
        n = max(1, round(spec.samples * args.samples_scale))
        s = measure(spec, n)
        s.update({"schema": SCHEMA, "kind": "series", "run_id": run_id, "mode": "warm"})
        series.append(s)
        print(
            f"{spec.workload} {spec.path} c={spec.concurrency}: median {median(s['ms']):.3f} ms (n={n}) "
            f"quality pass={s['quality'].get('pass')}",
            file=sys.stderr,
        )
    record = header(driver, args, "warm", run_id, started)
    record.update(
        {
            "artifacts": driver.artifacts(args),
            "environment": environment(
                args.provider, args.note, driver.session_providers(run), load_before, driver.environment_extra(args)
            ),
            "settings": run.settings,
            "load_ms": dict(run.load_stages),
            "quality": quality,
        }
    )
    quality.update(driver.run_quality_extra(args))
    passed = apply_quality_gate(quality, series, driver.series_passes(args))
    write(args.out, [record, *series])
    report_failures(quality)
    return 0 if passed else 1


def common_forwarded(args: argparse.Namespace) -> list[str]:
    argv = ["--provider", args.provider, "--workloads", str(args.workloads)]
    if args.snapshot is not None:
        argv += ["--snapshot", str(args.snapshot)]
    if args.threads:
        argv += ["--threads", str(args.threads)]
    if args.inter_threads:
        argv += ["--inter-threads", str(args.inter_threads)]
    return argv


def cmd_cold(driver: Driver, args: argparse.Namespace, script: Path) -> int:
    load_before = load_snapshot()
    started = datetime.datetime.now(datetime.UTC).isoformat()
    children, walls = [], []
    for _ in range(args.processes):
        argv = [sys.executable, str(script), "cold-child", *common_forwarded(args), *driver.forwarded(args)]
        env = dict(os.environ)
        t0 = time.monotonic_ns()
        env[SPAWN_ENV] = str(t0)
        out = subprocess.run(argv, capture_output=True, text=True, env=env)
        walls.append((time.monotonic_ns() - t0) / 1e6)
        if out.returncode != 0:
            raise SystemExit(f"cold child failed ({out.returncode}): {out.stderr}")
        children.append(json.loads(out.stdout.strip().splitlines()[-1]))
    first = children[0]
    skip = {"quality", "settings", "series", "spawn_to_result", *driver.cold_skip}
    stage_names = [k for k in first if k not in skip]
    qualities = [c["quality"] for c in children]
    run_id = str(uuid.uuid4())
    spec = first["series"]
    series = {
        "schema": SCHEMA,
        "kind": "series",
        "run_id": run_id,
        "mode": "cold",
        "path": spec["path"],
        "workload": "cold-start",
        "input": spec["input"],
        "batch": spec["batch"],
        "concurrency": 1,
        "warmup_calls": 0,
        "n": len(children),
        "wall_ms": sum(walls),
        "ms": [c["spawn_to_result"] for c in children],
        "stages_ms": {**{k: [c[k] for c in children] for k in stage_names}, "process_wall": walls},
        "quality": cold_quality(qualities),
    }
    record = header(driver, args, "cold", run_id, started)
    quality = {
        "reference": driver.reference(args),
        "cases": spec["requests"],
        "cold_first_request": qualities[0],
        "pass": True,
        **driver.run_quality_extra(args),
    }
    record.update(
        {
            "artifacts": driver.artifacts(args),
            "environment": environment(args.provider, args.note, None, load_before, driver.environment_extra(args)),
            "settings": first["settings"],
            "quality": quality,
        }
    )
    passed = apply_quality_gate(quality, [series], driver.series_passes(args))
    write(args.out, [record, series])
    report_failures(quality)
    print(f"cold-start: median spawn-to-result {median(series['ms']):.1f} ms (n={len(children)})", file=sys.stderr)
    return 0 if passed else 1


def cold_quality(qualities: list[dict]) -> dict:
    """Every child's verdict; embedding fields (max_abs, min_cosine, finite) merged when the children report them."""
    failed = sum(1 for q in qualities if not q.get("pass", False))
    merged: dict = {"processes": len(qualities), "failed_processes": failed, "first_process": qualities[0]}
    if all("max_abs" in q for q in qualities):
        merged["max_abs"] = max(q["max_abs"] for q in qualities)
        merged["min_cosine"] = min(q.get("min_cosine", 1.0) for q in qualities)
        merged["finite"] = all(q.get("finite", True) for q in qualities)
    merged["pass"] = failed == 0
    return merged


def cmd_cold_child(driver: Driver, args: argparse.Namespace, t_main_ns: int) -> int:
    spawn_ns = int(os.environ[SPAWN_ENV])
    t_load = time.monotonic_ns()
    run = driver.load(args)
    spec = run.cold_series()
    output, stages, _ = spec.call(0)
    t_done = time.monotonic_ns()
    result = {"spawn_to_main": (t_main_ns - spawn_ns) / 1e6, "main_to_load": (t_load - t_main_ns) / 1e6}
    result.update(run.load_stages)
    result.update({name: stages[name] for name in spec.stages})
    result["spawn_to_result"] = (t_done - spawn_ns) / 1e6
    result["main_to_result"] = (t_done - t_main_ns) / 1e6
    quality = spec.judge([(0, output)])
    quality["pass"] = bool(driver.series_passes(args)(quality))
    result["quality"] = quality
    result["settings"] = run.settings
    result["series"] = {"path": spec.path, "input": spec.input, "batch": spec.batch, "requests": 1}
    print(json.dumps(result))
    return 0


def parser_for(driver: Driver, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("warm", "cold", "cold-child"):
        p = sub.add_parser(name)
        p.add_argument(
            "--snapshot",
            type=Path,
            default=os.environ.get(driver.snapshot_env) if driver.snapshot_env else None,
            help=f"model directory (default ${driver.snapshot_env})" if driver.snapshot_env else "model directory",
        )
        p.add_argument("--provider", choices=["cpu", "cuda"], required=True)
        p.add_argument("--threads", type=int, help="intra_op_num_threads (default: runtime default)")
        p.add_argument("--inter-threads", type=int, default=1, help="inter_op_num_threads (default 1)")
        p.add_argument("--workloads", type=Path, default=driver.default_workloads)
        if name != "cold-child":
            p.add_argument("--out", type=Path, required=True, help="raw JSON lines file (appended)")
            p.add_argument("--note", default=None, help="free-text machine note recorded in the environment")
        if name == "warm":
            p.add_argument("--only", default=None, help="comma-separated workload names")
            p.add_argument("--samples-scale", type=float, default=1.0)
        if name == "cold":
            p.add_argument("--processes", type=int, default=10)
        driver.add_arguments(p, name)
    return parser


def main(driver: Driver, script: Path, t_main_ns: int, description: str, argv: list[str] | None = None) -> int:
    """Entry point of a runner script; ``t_main_ns`` is CLOCK_MONOTONIC taken before the script's heavy imports."""
    args = parser_for(driver, description).parse_args(argv)
    args.workloads = Path(args.workloads).resolve()
    if args.snapshot is not None:
        args.snapshot = Path(args.snapshot)
    if args.cmd == "warm":
        return cmd_warm(driver, args)
    if args.cmd == "cold":
        return cmd_cold(driver, args, script)
    return cmd_cold_child(driver, args, t_main_ns)
