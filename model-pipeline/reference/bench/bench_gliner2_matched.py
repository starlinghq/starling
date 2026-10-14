"""Matched-backend benchmark runner for GLiNER2 on Python ONNX Runtime: the inference stage (G5, plan section 11).

The Python side of `ai-dotnet/benchmarks/Starlings.Benchmarks.Gliner2Harness`, for the budget
`gliner2-base-v1/onnx-fp32-{cpu,cuda}.inference_ratio_max` in `model-pipeline/protected/performance-budgets.yaml`.
GLiNER2's own preprocessing is the upstream `gliner2` package on PyTorch, so this runner does not prepare inputs: it
runs the graph on the exact tensors the C# harness built, written by

    Starlings.Benchmarks.Gliner2Harness dump-inputs --out gliner2-inputs.jsonl

(one line per workload request: every item's `input_ids`, `word_index`, `schema_index`, labels and threshold, as
`Gliner2EntityExtractor.Prepare` returns them; a header line carries the C# artifacts). The runner checks that the
dump was made from the same workload file and graph, and that every fixture item's tensors equal the protected
reference tensors (`extraction.jsonl`).

Stage `inference`, as the C# `RunModel`: chunks of at most 8 items (`Gliner2ExtractorOptions.MaxBatchSize`), right
padding (ids and mask with 0, word positions with 0, one `schema_index` row per item), `InferenceSession.run` to
`count_logits` and `span_logits`, and each item's valid outputs copied out. Its part `run` (the graph calls alone)
is also recorded. Prepare and decode are C#-only stages (budget `prepare_plus_decode_share_of_inference_max`).

Quality on every series (no decode in Python, so entities are not compared): fixture items against the protected
logits (`float.count_logits_max_abs`, `float.span_logits_relative` with relative error `|d| / max(1, |ref|)` over
the valid spans), synthetic items against the same input run alone at load
(`float.padded_batch_vs_single_span_logits_relative`), every output finite.

Usage:
    python bench_gliner2_matched.py warm --snapshot <dir> --inputs gliner2-inputs.jsonl --provider cpu --threads 6 \
        --out raw.jsonl
    python bench_gliner2_matched.py cold --snapshot <dir> --inputs gliner2-inputs.jsonl --provider cpu --threads 6 \
        --processes 10 --out raw.jsonl
"""

from __future__ import annotations

import time

_T_MAIN_NS = time.monotonic_ns()  # before the heavy imports, for cold-start stages

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_core as core  # noqa: E402

HARNESS = "bench_gliner2_matched.py"
PROTECTED = core.PROTECTED / "gliner2-base-v1"
FIXTURE = PROTECTED / "onnx-fp32-cpu" / "extraction.jsonl"
GRAPH = "gliner2-entities-fp32.onnx"
MAX_BATCH_SIZE = 8


def read_inputs(path: Path) -> tuple[dict, dict[str, list[list[dict]]]]:
    """The dump's header and its requests by workload (request index order)."""
    header, requests = None, {}
    for row in core.read_jsonl(path):
        if row["kind"] == "header":
            header = row
            continue
        workload = requests.setdefault(row["workload"], [])
        if row["request"] != len(workload):
            raise SystemExit(f"{path}: {row['workload']} request {row['request']} out of order")
        workload.append(row["items"])
    if header is None or header.get("schema") != "starlings-gliner2-inputs/1":
        raise SystemExit(f"{path}: not a dump-inputs file (header missing)")
    return header, requests


class Gliner2Run(core.LoadedRun):
    def __init__(self, args: argparse.Namespace):
        t0 = time.perf_counter_ns()
        ort = core.import_ort(args.provider)
        import numpy as np

        self.np = np
        self.args = args
        stages = {"runtime_init": core.ms_since(t0)}
        t = time.perf_counter_ns()
        self.spec = json.loads(args.workloads.read_text(encoding="utf-8"))
        self.header, self.requests = read_inputs(args.inputs)
        if self.header["artifacts"]["workload_sha256"] != core.sha256_file(args.workloads):
            raise SystemExit(f"{args.inputs} was dumped from another workload file than {args.workloads}")
        self.cases = {c["id"]: c for c in core.read_jsonl(FIXTURE)}
        tolerances = PROTECTED / f"onnx-fp32-{args.provider}" / "tolerances.yaml"
        self.count_tol = core.read_tolerance(tolerances, "float", "count_logits_max_abs")
        self.span_tol = core.read_tolerance(tolerances, "float", "span_logits_relative")
        self.self_tol = core.read_tolerance(tolerances, "float", "padded_batch_vs_single_span_logits_relative")
        self.tolerance_file = core.repo_relative(tolerances)
        stages["fixture_read"] = core.ms_since(t)
        t = time.perf_counter_ns()
        graph = args.snapshot / "onnx" / GRAPH
        if self.header["artifacts"]["model_sha256"] != core.sha256_file(graph):
            raise SystemExit(f"{args.inputs} was dumped with another graph than {graph}")
        stages["model_hash"] = core.ms_since(t)
        t = time.perf_counter_ns()
        self.session = core.ort_session(ort, graph, args)
        stages["session_create"] = core.ms_since(t)
        self.load_stages = stages
        self.session_providers = self.session.get_providers()
        self.settings = core.session_settings(
            args, {k: dict(v) for k, v in self.session.get_provider_options().items()}
        )
        self.settings.update(
            {
                "max_sequence_length": self.header["max_sequence_length"],
                "max_batch_size": MAX_BATCH_SIZE,
                "inputs": "C# Prepare tensors (dump-inputs)",
                "inputs_sha256": core.sha256_file(args.inputs),
            }
        )
        self.self_reference: dict[str, tuple] = {}

    def run_model(self, items: list[dict]):
        """RunModel: right-padded chunks of MAX_BATCH_SIZE; returns each item's (count_logits, span_logits[:, :n])."""
        np = self.np
        t_run = 0.0
        outputs = []
        for start in range(0, len(items), MAX_BATCH_SIZE):
            chunk = items[start : start + MAX_BATCH_SIZE]
            seq = max(len(x["input_ids"]) for x in chunk)
            words = max(len(x["word_index"]) for x in chunk)
            ids = np.zeros((len(chunk), seq), dtype=np.int64)
            mask = np.zeros((len(chunk), seq), dtype=np.int64)
            word_index = np.zeros((len(chunk), words), dtype=np.int64)
            schema = np.array([x["schema_index"] for x in chunk], dtype=np.int64)
            for r, x in enumerate(chunk):
                ids[r, : len(x["input_ids"])] = x["input_ids"]
                mask[r, : len(x["input_ids"])] = 1
                word_index[r, : len(x["word_index"])] = x["word_index"]
            t = time.perf_counter_ns()
            count, span = self.session.run(
                ["count_logits", "span_logits"],
                {"input_ids": ids, "attention_mask": mask, "word_index": word_index, "schema_index": schema},
            )
            t_run += core.ms_since(t)
            for r, x in enumerate(chunk):
                outputs.append((count[r].copy(), span[r, :, : x["word_count"], :].copy()))
        return outputs, t_run

    def call(self, items: list[dict]):
        t = time.perf_counter_ns()
        outputs, t_run = self.run_model(items)
        return outputs, {"inference": core.ms_since(t), "run": t_run}, {}

    def rel(self, actual, expected) -> float:
        np = self.np
        a, e = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        return float((np.abs(a - e) / np.maximum(1.0, np.abs(e))).max()) if a.size else 0.0

    def judge(self, requests: list[list[dict]]):
        np = self.np

        def judge(outputs):
            count_abs = span_rel = self_rel = 0.0
            finite, failures, fixture, synthetic = True, [], 0, 0
            for r, out in outputs:
                for x, (count, span) in zip(requests[r], out, strict=True):
                    finite &= bool(np.isfinite(count).all() and np.isfinite(span).all())
                    n, k = span.shape[1], span.shape[2]
                    valid = (np.arange(n)[:, None] + np.arange(k)[None, :]) < n
                    flat = np.stack([span[m][valid] for m in range(span.shape[0])])
                    if x["fixture"]:
                        fixture += 1
                        case = self.cases[x["id"]]
                        c = float(np.abs(count.astype(np.float64) - np.asarray(case["count_logits"])).max())
                        s = self.rel(flat.ravel(), case["span_logits_valid"])
                        count_abs, span_rel = max(count_abs, c), max(span_rel, s)
                        if c > self.count_tol or s > self.span_tol:
                            failures.append(f"{x['id']}: count max abs {c:.2e}, span relative {s:.2e}")
                    else:
                        synthetic += 1
                        ref = self.self_reference[x["id"]]
                        s = self.rel(flat.ravel(), ref)
                        self_rel = max(self_rel, s)
                        if s > self.self_tol:
                            failures.append(f"{x['id']}: span relative {s:.2e} against the single run at load")
            return {
                "items": fixture + synthetic,
                "synthetic_items": synthetic,
                "count_logits_max_abs": count_abs,
                "count_logits_tolerance": self.count_tol,
                "span_logits_relative": span_rel,
                "span_logits_tolerance": self.span_tol,
                "synthetic_vs_single_relative": self_rel,
                "synthetic_tolerance": self.self_tol,
                "finite": finite,
                "failures": failures,
                "pass": bool(finite and not failures),
            }

        return judge

    def prepare_self_reference(self, items: list[dict]) -> None:
        np = self.np
        for x in items:
            if not x["fixture"] and x["id"] not in self.self_reference:
                _, span = self.run_model([x])[0][0]
                n, k = span.shape[1], span.shape[2]
                valid = (np.arange(n)[:, None] + np.arange(k)[None, :]) < n
                self.self_reference[x["id"]] = np.stack([span[m][valid] for m in range(span.shape[0])]).ravel()

    def series(self, name, input_, batch, concurrency, samples, warmup, requests) -> core.Series:
        for req in requests:
            self.prepare_self_reference(req)
        return core.Series(
            workload=name,
            path="text",
            input=input_,
            batch=batch,
            concurrency=concurrency,
            samples=samples,
            warmup_calls=max(warmup, len(requests)),
            requests=len(requests),
            stages=["inference", "run"],
            call=lambda r: self.call(requests[r]),
            judge=self.judge(requests),
        )

    def dumped(self, name: str) -> list[list[dict]]:
        if name not in self.requests:
            raise SystemExit(f"{self.args.inputs} has no requests for workload '{name}'; dump the inputs again")
        return self.requests[name]

    def warm_series(self):
        for w in self.spec["workloads"]:
            requests = self.dumped(w["name"])
            for c in w.get("concurrency", [1]):
                warmup = w.get("warmup_calls", self.spec["warmup_calls"])
                yield self.series(w["name"], w["input"], w.get("batch", 1), c, w["samples"], warmup, requests)

    def cold_series(self) -> core.Series:
        requests = self.dumped("cold-start")
        return self.series("cold-start", "short", len(requests[0]), 1, 1, 0, requests)

    def preliminary_quality(self) -> dict:
        """Every distinct fixture item of the dump alone: tensors equal to the reference, logits within tolerance."""
        items = {}
        for requests in self.requests.values():
            for req in requests:
                for x in req:
                    if x["fixture"]:
                        items.setdefault(x["id"], x)
        tensors_equal = [
            i
            for i, x in items.items()
            if all(x[k] == self.cases[i][k] for k in ("input_ids", "word_index", "schema_index"))
        ]
        singles = [[x] for x in items.values()]
        quality = self.judge(singles)([(i, self.run_model(req)[0]) for i, req in enumerate(singles)])
        quality.update(
            {
                "reference": core.repo_relative(FIXTURE),
                "tolerance_file": self.tolerance_file,
                "cases": len(items),
                "tensors_equal_reference": len(tensors_equal),
            }
        )
        quality["pass"] = bool(quality["pass"] and len(tensors_equal) == len(items))
        return quality


class Gliner2Driver(core.Driver):
    model = "gliner2-base-v1"
    harness = HARNESS
    default_workloads = core.WORKLOADS / "gliner2-v1.json"
    snapshot_env = "STARLING_GLINER2_SNAPSHOT"

    def add_arguments(self, parser: argparse.ArgumentParser, command: str) -> None:
        parser.add_argument("--inputs", type=Path, required=True, help="the C# harness's dump-inputs file")

    def forwarded(self, args: argparse.Namespace) -> list[str]:
        return ["--inputs", str(args.inputs)]

    def profile(self, args: argparse.Namespace) -> str:
        return f"onnx-fp32-{args.provider}"

    def load(self, args: argparse.Namespace) -> core.LoadedRun:
        if args.snapshot is None:
            raise SystemExit("--snapshot or STARLING_GLINER2_SNAPSHOT is required")
        return Gliner2Run(args)

    def reference(self, args: argparse.Namespace) -> str:
        return core.repo_relative(FIXTURE)

    def artifacts(self, args: argparse.Namespace) -> dict:
        return core.artifacts(
            args.snapshot / "onnx" / GRAPH,
            args.snapshot / "tokenizer.json",
            args.workloads,
            core.file_digests(FIXTURE, PROTECTED / "onnx-fp32-cpu" / "tolerances.yaml"),
            core.package_versions(("onnxruntime", "onnxruntime-gpu", "numpy")),
            {"inputs_sha256": core.sha256_file(args.inputs), "inputs_header": read_inputs(args.inputs)[0]["artifacts"]},
        )


def main() -> int:
    return core.main(Gliner2Driver(), Path(__file__).resolve(), _T_MAIN_NS, __doc__.split("\n", 1)[0])


if __name__ == "__main__":
    raise SystemExit(main())
