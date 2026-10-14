"""Matched-backend benchmark runner for e5-small on Python ONNX Runtime (P-008).

The Python side of the C#-versus-Python comparison in plan section 11: same ONNX Runtime version, same model
file, same thread settings, same workload file and fixture inputs as the C# harness
(`ai-dotnet/benchmarks/Starlings.Benchmarks.E5Harness`). Both write the raw result format
`starlings-bench-raw/1` (`model-pipeline/schemas/benchmark-raw.schema.json`), one JSON object per line; summarize
with `model-pipeline bench summarize`, check against the G5 budgets with `model-pipeline bench g5`. The warm/cold
protocol, environment manifest and writer are the shared core (`model-pipeline/reference/bench/matched_core.py`).

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
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import matched_core as core  # noqa: E402

HARNESS = "bench_e5_matched.py"
REPO = core.REPO
FIXTURES = REPO / "model-pipeline" / "protected" / "multilingual-e5-small" / "onnx-fp32-cpu"
DEFAULT_WORKLOADS = core.WORKLOADS / "e5-v1.json"


# ----------------------------------------------------------------------------------------------- inputs


def load_fixtures() -> dict[str, dict]:
    """Case id -> {"ids", "text", "embedding"} from the protected CPU fixtures (read only)."""
    cases: dict[str, dict] = {}
    for row in core.read_jsonl(FIXTURES / "embeddings.jsonl"):
        cases[row["id"]] = {"ids": row["input_ids"], "embedding": row["embedding"]}
    for row in core.read_jsonl(FIXTURES / "tokenizer.jsonl"):
        if row["id"] in cases:
            cases[row["id"]]["text"] = row["text"]
    return cases


def fixture_dir_for(provider: str) -> Path:
    return FIXTURES.parent / ("onnx-fp32-cuda" if provider == "cuda" else "onnx-fp32-cpu")


def read_tolerance(fixture_dir: Path) -> float:
    return core.read_tolerance(fixture_dir / "tolerances.yaml", "float", "embedding_max_abs")


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
        self.session = core.ort_session(ort, args.snapshot / "onnx" / "model.onnx", args)
        t2 = time.perf_counter_ns()
        self.tokenizer_load_ms = (t1 - t0) / 1e6
        self.session_create_ms = (t2 - t1) / 1e6
        self.settings = core.session_settings(
            args, {k: dict(v) for k, v in self.session.get_provider_options().items()}
        )
        self.settings.update({"pad_id": 1, "output_location": "host", "max_batch_size": 32})

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


def apply_quality_gate(run_quality: dict, series: list[dict], tolerance: float) -> bool:
    """The run passes only if the preliminary sweep and every timed series are within tolerance and finite.

    Marks each series' ``quality.pass`` and records ``preliminary_pass``, ``series_pass`` and ``failed_series`` in the
    run's quality block, so a failure specific to one workload (a large batch, the text path) fails the whole run and
    its timings are never presented as valid. Same rule as the C# harness (``QualityGate``).
    """
    return core.apply_quality_gate(run_quality, series, lambda q: q["max_abs"] <= tolerance and q["finite"])


# ----------------------------------------------------------------------------------------------- driver


class E5Run(core.LoadedRun):
    def __init__(self, args: argparse.Namespace):
        t0 = time.perf_counter_ns()
        ort = core.import_ort(args.provider)
        import numpy as np
        import tokenizers

        t1 = time.perf_counter_ns()
        self.args = args
        self.np = np
        self.cases = load_fixtures()
        self.spec = json.loads(args.workloads.read_text(encoding="utf-8"))
        t2 = time.perf_counter_ns()
        self.pipe = Pipeline(args, ort, np, tokenizers)
        self.tolerance = read_tolerance(fixture_dir_for(args.provider))
        self.load_stages = {
            "runtime_init": (t1 - t0) / 1e6,
            "fixture_read": (t2 - t1) / 1e6,
            "tokenizer_load": self.pipe.tokenizer_load_ms,
            "session_create": self.pipe.session_create_ms,
        }
        self.settings = self.pipe.settings
        self.session_providers = self.pipe.session.get_providers()

    def preliminary_quality(self) -> dict:
        return run_quality(self.pipe, self.cases, self.tolerance, fixture_dir_for(self.args.provider))

    def _series(self, name, path, input_, batch, concurrency, samples, warmup, requests) -> core.Series:
        np = self.np

        def call(r: int):
            out, stages = self.pipe.embed(requests[r], path)
            return out, stages, {}

        def judge(outputs):
            parts = [compare(np, out, [c["embedding"] for c in requests[r]]) for r, out in outputs]
            return merge_quality(np, parts)

        return core.Series(
            workload=name,
            path=path,
            input=input_,
            batch=batch,
            concurrency=concurrency,
            samples=samples,
            warmup_calls=max(warmup, len(requests)),
            requests=len(requests),
            stages=(["tokenize"] if path == "text" else []) + ["prep", "run", "pool"],
            call=call,
            judge=judge,
        )

    def warm_series(self):
        for w in self.spec["workloads"]:
            requests = [[self.cases[cid] for cid in req] for req in w["requests"]]
            for path in self.args.paths.split(","):
                for c in w["concurrency"]:
                    if c not in self.args.concurrency:
                        continue
                    samples = w["samples"][self.args.provider]
                    yield self._series(
                        w["name"], path, w["input"], w["batch"], c, samples, self.spec["warmup_calls"], requests
                    )

    def cold_series(self) -> core.Series:
        request = [self.cases[cid] for cid in self.spec["cold"]["request"]]
        return self._series("cold-start", "text", "short", len(request), 1, 1, 0, [request])


class E5Driver(core.Driver):
    model = "multilingual-e5-small"
    harness = HARNESS
    default_workloads = DEFAULT_WORKLOADS
    snapshot_env = "STARLING_E5_SNAPSHOT"

    def add_arguments(self, parser: argparse.ArgumentParser, command: str) -> None:
        if command == "warm":
            parser.add_argument("--paths", default="ids,text")
            parser.add_argument("--concurrency", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4])

    def profile(self, args: argparse.Namespace) -> str:
        return f"onnx-fp32-{args.provider}"

    def load(self, args: argparse.Namespace) -> core.LoadedRun:
        if args.snapshot is None:
            raise SystemExit("--snapshot or STARLING_E5_SNAPSHOT is required")
        return E5Run(args)

    def reference(self, args: argparse.Namespace) -> str:
        return str((FIXTURES / "embeddings.jsonl").relative_to(REPO))

    def tolerance(self, args: argparse.Namespace) -> float:
        return read_tolerance(fixture_dir_for(args.provider))

    def series_passes(self, args: argparse.Namespace):
        tolerance = self.tolerance(args)
        return lambda q: q["max_abs"] <= tolerance and q["finite"]

    def run_quality_extra(self, args: argparse.Namespace) -> dict:
        fixture_dir = fixture_dir_for(args.provider)
        return {
            "tolerance_file": str((fixture_dir / "tolerances.yaml").relative_to(REPO)),
            "embedding_max_abs_tolerance": self.tolerance(args),
        }

    def artifacts(self, args: argparse.Namespace) -> dict:
        return core.artifacts(
            args.snapshot / "onnx" / "model.onnx",
            args.snapshot / "tokenizer.json",
            args.workloads,
            {name: core.sha256_file(FIXTURES / name) for name in ("embeddings.jsonl", "tokenizer.jsonl")},
            core.package_versions(("onnxruntime", "onnxruntime-gpu", "numpy", "tokenizers")),
        )


def main() -> int:
    return core.main(E5Driver(), Path(__file__).resolve(), _T_MAIN_NS, __doc__.split("\n", 1)[0])


if __name__ == "__main__":
    raise SystemExit(main())
